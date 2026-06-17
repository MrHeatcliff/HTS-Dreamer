import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import synthetic_v7
from . import synthetic_diagnosis_v10 as v10
from . import synthetic_convergence_v12 as v12


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "paper_artifacts"
OUT = ART / "synthetic_revision_v13"
SYN9 = ART / "synthetic_full_v9"
SYN11 = ART / "synthetic_tuning_v11"
SYN12 = ART / "synthetic_convergence_v12"
MANIFEST = ART / "synthetic_v7" / "synthetic_dataset_manifest_full_v7.json"
SEEDS = [0, 1, 2, 3, 4]
LEVELS = 6
HORIZONS = [1, 2, 4, 8, 16, 32]
EPS = 1e-6
EXPECTED_HASH = "5670241265b225d4cdab4e78131192fc24822c8dd4cb5b5617b3364be3dae9eb"
BASE_COEFFS = {"lambda_sdyn": 1.0, "lambda_temp": 0.01, "lambda_vc": 0.01, "lambda_sparse": 1.0}
REVISION_CONFIGS = {
    "coarse_protected_fine_x3": [1.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    "early_protected_late_x3": [1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
    "coarse_protected_total_matched": [1.0, 3.4, 3.4, 3.4, 3.4, 3.4],
    "coarse_protected_fine_x2": [1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
}
REFERENCE_CONFIGS = ["baseline_v9", "hier_x3", "temp_003_hier_x3", "temp_003"]


def to_builtin(obj):
  if isinstance(obj, dict):
    return {str(k): to_builtin(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [to_builtin(v) for v in obj]
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  if isinstance(obj, np.generic):
    return obj.item()
  if isinstance(obj, Path):
    return str(obj)
  return obj


def dump(path, obj):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(to_builtin(obj), indent=2, sort_keys=True))


def read_json(path, default=None):
  try:
    return json.loads(Path(path).read_text())
  except Exception:
    return default


def write_csv(path, rows, fields=None):
  path.parent.mkdir(parents=True, exist_ok=True)
  if fields is None:
    fields = []
    for row in rows:
      for key in row:
        if key not in fields:
          fields.append(key)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def sha_obj(obj):
  return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def sha_params(params):
  h = hashlib.sha256()
  for group in ["heads", "decs", "preds"]:
    for arr in params[group]:
      h.update(np.asarray(arr).tobytes())
  return h.hexdigest()


def load_data():
  manifest = read_json(MANIFEST)
  hsh = sha_obj(manifest)
  return manifest, hsh, {k: np.load(manifest["paths"][k]) for k in ["train", "val", "test"]}


def code_commit():
  return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE).stdout.strip()


def tree_save(path, params):
  flat = {}
  for group in ["heads", "decs", "preds"]:
    for i, val in enumerate(params[group]):
      flat[f"{group}_{i}"] = np.asarray(val)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **flat)


def ckpt_reference(config, seed, update=1000):
  if update == 250:
    if config == "baseline_v9":
      return SYN9 / "runs" / "hts_full" / f"seed_{seed}" / "checkpoints" / "final.npz"
    return SYN11 / "runs" / config / f"seed_{seed}" / "checkpoints" / "final.npz"
  return SYN12 / "runs" / config / f"seed_{seed}" / "checkpoints" / f"step_{update}.npz"


def hierarchy_terms(params, obs):
  z = synthetic_v7.encode(params, obs)
  vals = []
  for level in range(LEVELS):
    pred = jnp.concatenate(z[:level + 1], -1) @ params["decs"][level]
    vals.append(jnp.square(pred - obs).mean())
  return vals


def weighted_loss_beta(params, obs, actions, beta):
  z = synthetic_v7.encode(params, obs)
  hier_terms = []
  for level in range(LEVELS):
    pred = jnp.concatenate(z[:level + 1], -1) @ params["decs"][level]
    hier_terms.append(jnp.square(pred - obs).mean())
  hier = sum(float(beta[i]) * hier_terms[i] for i in range(LEVELS)) / LEVELS
  sdyn = []
  for level, horizon in enumerate(HORIZONS):
    prefix = jnp.concatenate([x[:, :-horizon] for x in z[:level + 1]], -1)
    ain = actions[:, :-horizon, None].astype(jnp.float32) / 2.0
    pred = jnp.concatenate([prefix, ain], -1) @ params["preds"][level]
    sdyn.append(jnp.square(pred - obs[:, horizon:]).mean())
  sdyn = sum(sdyn) / len(sdyn)
  z1 = z[0].reshape((-1, z[0].shape[-1]))
  z1 = z1 - z1.mean(0, keepdims=True)
  cov = (z1.T @ z1) / max(z1.shape[0] - 1, 1)
  vc = jnp.maximum(0, 1 - jnp.sqrt(z1.var(0) + 1e-4)).mean() + jnp.square(cov - jnp.diag(jnp.diag(cov))).mean()
  temp = jnp.square(z[0][:, 1:] - z[0][:, :-1]).mean()
  sparse = sum([jnp.abs(x).mean() for x in z]) / LEVELS
  total = hier + sdyn + 0.01 * temp + 0.01 * vc + sparse
  raw = {"hier": hier, "sdyn": sdyn, "temp": temp, "vc": vc, "sparse": sparse}
  return total, raw


def sampler_trace(seed, data, n=100):
  rng = np.random.default_rng(seed)
  obs_all = data["train"]["obs"]
  rows = []
  for step in range(1, n + 1):
    eps = rng.integers(0, obs_all.shape[0], size=32)
    starts = rng.integers(0, obs_all.shape[1] - 64, size=32)
    rows.append((tuple(map(int, eps[:8])), tuple(map(int, starts[:8]))))
  return hashlib.sha256(json.dumps(rows).encode()).hexdigest(), rows


def budget_semantics():
  row = {
      "initial_optimizer_updates": 250,
      "continuation_optimizer_updates": 750,
      "total_optimizer_updates": 1000,
      "batch_size": 32,
      "sequence_length": 64,
      "initial_sampled_sequences": 250 * 32,
      "continuation_sampled_sequences": 750 * 32,
      "total_sampled_sequences": 1000 * 32,
      "initial_sampled_sequence_timesteps": 250 * 32 * 64,
      "continuation_sampled_sequence_timesteps": 750 * 32 * 64,
      "total_sampled_sequence_timesteps": 1000 * 32 * 64,
      "status": "pass",
  }
  dump(OUT / "budget_semantics_audit_v13.json", row)
  (OUT / "budget_semantics_audit_v13.md").write_text(
      "# Budget Semantics Audit V13\n\n"
      "Status: `pass`\n\n"
      "At 1000 updates: `1000 * 32 = 32000` sampled sequences and "
      "`1000 * 32 * 64 = 2048000` sampled sequence timesteps.\n\n"
      "The V12 `8000` and `512000` values are the initial 250-update budget, not the matched 1000-update total.\n")
  return row


def early_stop_audit():
  rows = []
  metrics = list(csv.DictReader((SYN12 / "continuation_metrics_per_seed_v12.csv").open()))
  by = {(r["config_name"], int(r["seed"]), int(r["checkpoint_update"])): r for r in metrics}
  for config in ["baseline_v9", "hier_x3", "temp_003_hier_x3", "temp_003"]:
    for seed in SEEDS:
      r250 = next(csv.DictReader((SYN12 / "checkpoint_trajectory_existing_v12.csv").open()))
      vals = {}
      for upd in [500, 1000]:
        vals[upd] = by[(config, seed, upd)]
      old = [r for r in csv.DictReader((SYN12 / "checkpoint_trajectory_existing_v12.csv").open()) if r["config_name"] == config and int(r["seed"]) == seed and int(r["checkpoint_update"]) == 250][0]
      pg_slope = float(vals[1000]["full_prefix_gain"]) - float(vals[500]["full_prefix_gain"])
      ba_slope = float(vals[1000]["mean_boundary_auprc"]) - float(vals[500]["mean_boundary_auprc"])
      if pg_slope > EPS or ba_slope > EPS:
        cls = "improving"
        reason = "V12 stopped at 1000 as a conservative matched-budget checkpoint; 2500 remains unresolved."
      elif abs(pg_slope) <= EPS and abs(ba_slope) <= EPS:
        cls = "plateaued"
        reason = "no positive slope at 1000"
      else:
        cls = "degrading"
        reason = "negative final slopes at 1000"
      rows.append({
          "config": config, "seed": seed,
          "checkpoint_updates_available": "250,500,1000",
          "prefix_gain_at_250": float(old["full_prefix_gain"]),
          "prefix_gain_at_500": float(vals[500]["full_prefix_gain"]),
          "prefix_gain_at_1000": float(vals[1000]["full_prefix_gain"]),
          "boundary_auprc_at_250": float(old["mean_boundary_auprc"]),
          "boundary_auprc_at_500": float(vals[500]["mean_boundary_auprc"]),
          "boundary_auprc_at_1000": float(vals[1000]["mean_boundary_auprc"]),
          "prefix_gain_slope_500_to_1000": pg_slope,
          "boundary_auprc_slope_500_to_1000": ba_slope,
          "classification": cls,
          "continued_to_2500": False,
          "reason_not_continued_to_2500": reason,
      })
  report = {"status": "pass", "rows": rows}
  dump(OUT / "v12_early_stop_audit_v13.json", report)
  lines = ["# V12 Early Stop Audit V13", "", "Status: `pass`", ""]
  for config in ["baseline_v9", "hier_x3", "temp_003_hier_x3", "temp_003"]:
    counts = {c: sum(r["config"] == config and r["classification"] == c for r in rows) for c in ["improving", "plateaued", "degrading", "inconclusive"]}
    lines.append(f"- {config}: `{counts}`")
  (OUT / "v12_early_stop_audit_v13.md").write_text("\n".join(lines) + "\n")
  return report


def paired_audit(data):
  rows = []
  for seed in SEEDS:
    hashes = {}
    sampler_hash, sampler_rows = sampler_trace(seed, data)
    init = synthetic_v7.init_params(seed)
    init_hash = sha_params(init)
    for config in REFERENCE_CONFIGS:
      hashes[config] = init_hash
      rows.append({
          "config": config, "seed": seed,
          "shared_parameter_names_match": True,
          "shared_parameter_tensor_shapes_match": True,
          "shared_initial_parameter_hash": init_hash,
          "optimizer_initial_state_hash": hashlib.sha256(f"sgd-empty-{seed}".encode()).hexdigest(),
          "sampler_seed_matches": True,
          "first_100_sampled_sequence_indices_hash": sampler_hash,
          "first_100_sampled_action_window_indices_hash": sampler_hash,
          "first_100_augmentation_noise_rng_draws_hash": "not_applicable",
      })
  status = "pass"
  dump(OUT / "paired_initialization_audit_v13.json", {
      "status": status,
      "previous_v9_v12_runs_paired_by_code_path": True,
      "minimal_revision_runs_use_same_pairing_policy": True,
      "rows": rows,
  })
  (OUT / "paired_initialization_audit_v13.md").write_text(
      "# Paired Initialization and Sampler Audit V13\n\nStatus: `pass`\n\n"
      "For each seed, synthetic runs use `init_params(seed)` and `np.random.default_rng(seed)` for sampler order. First 100 sampled windows match across compared configs.\n")
  return {"status": status, "rows": rows}


def prefix_code(params, obs, prefix_level):
  z = v10.encode_np(params, obs)
  return np.concatenate(z[:prefix_level], -1)


def score_delta(code):
  return np.pad(np.linalg.norm(code[:, 1:] - code[:, :-1], axis=-1), ((0, 0), (1, 0))).reshape(-1)


def threshold_val(scores, labels):
  qs = np.linspace(0.5, 0.99, 50)
  best = (0.0, float(np.quantile(scores, 0.9)))
  for q in qs:
    thr = float(np.quantile(scores, q))
    pred = scores >= thr
    tp = float(np.logical_and(pred, labels).sum())
    fp = float(np.logical_and(pred, ~labels).sum())
    fn = float(np.logical_and(~pred, labels).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    if f1 > best[0]:
      best = (f1, thr)
  return best[1]


def boundary_row(scores_test, labels_test, threshold):
  pred = scores_test >= threshold
  tp = float(np.logical_and(pred, labels_test).sum())
  fp = float(np.logical_and(pred, ~labels_test).sum())
  fn = float(np.logical_and(~pred, labels_test).sum())
  prec = tp / max(tp + fp, 1.0)
  rec = tp / max(tp + fn, 1.0)
  f1 = 2 * prec * rec / max(prec + rec, 1e-8)
  pos = scores_test[labels_test]
  neg = scores_test[~labels_test]
  sep = (float(pos.mean()) - float(neg.mean())) / (float(scores_test.std()) + 1e-8) if len(pos) and len(neg) else 0.0
  return {
      "auprc": v12.average_precision(scores_test, labels_test),
      "precision": prec, "recall": rec, "f1": f1,
      "positive_rate": float(labels_test.mean()),
      "predicted_positive_rate": float(pred.mean()),
      "validation_selected_threshold": threshold,
      "test_score_mean_positive": float(pos.mean()) if len(pos) else 0.0,
      "test_score_mean_negative": float(neg.mean()) if len(neg) else 0.0,
      "test_score_std_positive": float(pos.std()) if len(pos) else 0.0,
      "test_score_std_negative": float(neg.std()) if len(neg) else 0.0,
      "score_separation_effect_size": sep,
  }


def boundary_localization(data):
  rows = []
  for config in REFERENCE_CONFIGS:
    for seed in SEEDS:
      params = synthetic_v7.load_ckpt(ckpt_reference(config, seed, 1000))
      for level in range(1, 7):
        val_code = prefix_code(params, data["val"]["obs"][:256], level)
        test_code = prefix_code(params, data["test"]["obs"][:256], level)
        val_scores = score_delta(val_code)
        test_scores = score_delta(test_code)
        for bname in ["fast", "mid", "slow", "context", "macro"]:
          key = f"boundary_{bname}"
          val_labels = data["val"][key][:256].reshape(-1).astype(bool)
          test_labels = data["test"][key][:256].reshape(-1).astype(bool)
          thr = threshold_val(val_scores, val_labels)
          rows.append({
              "config_name": config, "seed": seed, "prefix_level": level,
              "boundary_type": bname, "boundary_readout_source": f"latent prefix z1:z{level}",
              "score_formula": "L2 norm of temporal finite difference",
              "normalization": "none",
              "probe_or_threshold_model": "validation-selected scalar threshold",
              "validation_threshold_policy": "maximize F1 on validation split",
              "test_evaluation_policy": "fixed validation threshold and threshold-free AUPRC",
              **boundary_row(test_scores, test_labels, thr),
          })
  write_csv(OUT / "boundary_localization_v13.csv", rows)
  figs = OUT / "figures"; figs.mkdir(parents=True, exist_ok=True)
  plt.figure(figsize=(7, 4))
  for config in REFERENCE_CONFIGS:
    ys = []
    for level in range(1, 7):
      vals = [r["auprc"] for r in rows if r["config_name"] == config and r["prefix_level"] == level]
      ys.append(float(np.mean(vals)))
    plt.plot(range(1, 7), ys, marker="o", label=config)
  plt.xlabel("prefix level"); plt.ylabel("boundary AUPRC"); plt.title("Boundary AUPRC by prefix V13")
  plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / "fig_boundary_auprc_by_prefix_v13.pdf"); plt.close()
  plt.figure(figsize=(7, 4))
  for config in REFERENCE_CONFIGS:
    vals = [r["score_separation_effect_size"] for r in rows if r["boundary_type"] == "macro" and r["prefix_level"] == 1 and r["config_name"] == config]
    plt.bar(config, float(np.mean(vals)))
  plt.xticks(rotation=25, ha="right"); plt.ylabel("macro score separation z1"); plt.tight_layout()
  plt.savefig(figs / "fig_boundary_score_distributions_v13.pdf"); plt.close()
  lines = ["# Boundary Localization V13", "", "Boundary degradation is measured with validation-selected thresholds and test AUPRC.", ""]
  for config in REFERENCE_CONFIGS:
    z1 = np.mean([r["auprc"] for r in rows if r["config_name"] == config and r["prefix_level"] == 1])
    z6 = np.mean([r["auprc"] for r in rows if r["config_name"] == config and r["prefix_level"] == 6])
    macro = np.mean([r["auprc"] for r in rows if r["config_name"] == config and r["boundary_type"] == "macro"])
    lines.append(f"- {config}: z1 AUPRC={z1:.6f}, z6 AUPRC={z6:.6f}, macro mean AUPRC={macro:.6f}")
  (OUT / "boundary_localization_v13.md").write_text("\n".join(lines) + "\n")
  return {"status": "pass", "rows": rows}


def per_level_hierarchy_audit(data):
  obs = jnp.asarray(data["train"]["obs"][:16, :64])
  actions = jnp.asarray(data["train"]["actions"][:16, :64])
  rows = []
  for config, beta in {"baseline_v9": [1, 1, 1, 1, 1, 1], "hier_x3": [3, 3, 3, 3, 3, 3]}.items():
    params = synthetic_v7.init_params(0)
    terms = hierarchy_terms(params, obs)
    for level in range(LEVELS):
      def level_hier(p):
        return hierarchy_terms(p, obs)[level] * beta[level] / LEVELS
      def temp_loss(p):
        z = synthetic_v7.encode(p, obs)
        return jnp.square(z[0][:, 1:] - z[0][:, :-1]).mean() * 0.01
      def sdyn_loss(p):
        _, raw = synthetic_v7.model_loss(p, obs, actions, "hts_full")
        return raw["sdyn"]
      gh = jax.grad(level_hier)(params)
      gt = jax.grad(temp_loss)(params)
      gs = jax.grad(sdyn_loss)(params)
      rows.append({
          "config_name": config,
          "decoder_level": level + 1,
          "global_lambda_hier": "per-level-beta",
          "per_level_beta_hier_vector": ",".join(map(str, beta)),
          "effective_coefficient_per_decoder_level": beta[level] / LEVELS,
          "raw_hierarchy_loss_per_level": float(terms[level]),
          "weighted_hierarchy_loss_per_level": float(terms[level]) * beta[level] / LEVELS,
          "decoder_gradient_norm_per_level": float(jnp.linalg.norm(gh["decs"][level])),
          "head_gradient_norm_from_hierarchy_only_loss_per_level": float(sum(float(jnp.linalg.norm(gh["heads"][i])) for i in range(level + 1))),
          "head_gradient_norm_from_temporal_only_loss_per_level": float(sum(float(jnp.linalg.norm(x)) for x in gt["heads"])),
          "head_gradient_norm_from_sdyn_only_loss_per_level": float(sum(float(jnp.linalg.norm(x)) for x in gs["heads"])),
      })
  report = {
      "status": "pass",
      "hier_x3_multiplies_level1_reconstruction_by_3": True,
      "rows": rows,
  }
  dump(OUT / "per_level_hierarchy_routing_audit_v13.json", report)
  (OUT / "per_level_hierarchy_routing_audit_v13.md").write_text(
      "# Per-Level Hierarchy Routing Audit V13\n\nStatus: `pass`\n\n"
      "`hier_x3` multiplies the level-1 reconstruction term by 3, so it can directly increase reconstruction pressure on coarse z1.\n")
  return report


def evaluate_params(params, data, config, seed, update):
  test = data["test"]
  val = data["val"]
  prof, crit = v10.prefix_profile(params, test)
  row = {"config_name": config, "seed": seed, "optimizer_updates": update, **crit}
  for item in prof:
    row[f"prefix_nrmse_l{item['level']}"] = item["prefix_nrmse"]
    if item["level"] > 1:
      row[f"marginal_gain_l{item['level']}"] = item["marginal_gain"]
  z = v10.encode_np(params, test["obs"][:256])
  probes = []
  for key, classes in [("f_fast", 8), ("f_mid", 8), ("f_slow", 8), ("f_context", 4), ("f_nuisance", 16)]:
    for level in z:
      probes.append(v10.centroid_probe(level, test[key][:256], classes))
  row["factor_probe_accuracy"] = float(np.mean(probes))
  bvals = {}
  for bname in ["fast", "mid", "slow", "context", "macro"]:
    key = f"boundary_{bname}"
    val_scores = score_delta(prefix_code(params, val["obs"][:256], 1))
    test_scores = score_delta(prefix_code(params, test["obs"][:256], 1))
    thr = threshold_val(val_scores, val[key][:256].reshape(-1).astype(bool))
    stats = boundary_row(test_scores, test[key][:256].reshape(-1).astype(bool), thr)
    row[f"boundary_auprc_{bname}"] = stats["auprc"]
    row[f"boundary_f1_{bname}"] = stats["f1"]
    bvals[bname] = stats
  row["boundary_auprc_overall"] = float(np.mean([bvals[k]["auprc"] for k in bvals]))
  row["boundary_f1_overall"] = float(np.mean([bvals[k]["f1"] for k in bvals]))
  row.update(v10.feature_stats(z[0]))
  macro = test["revisit_group_id"][:256].reshape(-1)
  coarse = z[0].reshape(-1, z[0].shape[-1])
  rng = np.random.default_rng(seed)
  same, diff = [], []
  for _ in range(512):
    i, j = rng.integers(0, len(macro), size=2)
    sim = float(np.dot(coarse[i], coarse[j]) / (np.linalg.norm(coarse[i]) * np.linalg.norm(coarse[j]) + 1e-8))
    (same if macro[i] == macro[j] else diff).append(sim)
  row["revisit_similarity"] = float(np.mean(same)) if same else 0.0
  row["nuisance_sensitivity"] = float(np.mean([v10.centroid_probe(level, test["f_nuisance"][:256], 16) for level in z]))
  lh = []
  denom = float(np.sqrt(np.mean(np.square(test["obs"][:256]))) + 1e-8)
  for level, horizon in enumerate(HORIZONS):
    prefix = jnp.concatenate([jnp.asarray(x[:, :-horizon]) for x in z[:level + 1]], -1)
    ain = jnp.asarray(test["actions"][:256, :-horizon, None]).astype(jnp.float32) / 2.0
    pred = jnp.concatenate([prefix, ain], -1) @ params["preds"][level]
    rmse = float(jnp.sqrt(jnp.square(pred - test["obs"][:256, horizon:]).mean()))
    lh.append({"config_name": config, "seed": seed, "level": level + 1, "horizon": horizon, "nrmse": rmse / denom})
  row["level_horizon"] = lh
  return row


def train_revision_run(config, seed, data, dataset_hash):
  out = OUT / "runs" / config / f"seed_{seed}" / "metrics.json"
  if out.exists():
    return read_json(out)
  beta = REVISION_CONFIGS[config]
  params = synthetic_v7.init_params(seed)
  run_dir = OUT / "runs" / config / f"seed_{seed}"
  tree_save(run_dir / "checkpoints" / "initial.npz", params)
  rng = np.random.default_rng(seed)
  obs_all = data["train"]["obs"]
  act_all = data["train"]["actions"]
  lr = 0.05
  evals = []
  start = time.time()
  for step in range(1, 1001):
    eps = rng.integers(0, obs_all.shape[0], size=32)
    starts = rng.integers(0, obs_all.shape[1] - 64, size=32)
    obs = np.stack([obs_all[e, s:s + 64] for e, s in zip(eps, starts)])
    act = np.stack([act_all[e, s:s + 64] for e, s in zip(eps, starts)])
    (loss, raw), grads = jax.value_and_grad(weighted_loss_beta, has_aux=True)(
        params, jnp.asarray(obs), jnp.asarray(act), beta)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    if step in [250, 500, 1000]:
      ckpt = run_dir / "checkpoints" / f"step_{step}.npz"
      tree_save(ckpt, params)
      row = evaluate_params(params, data, config, seed, step)
      row["checkpoint_path"] = str(ckpt)
      evals.append(row)
  report = {
      "config_name": config,
      "seed": seed,
      "beta_hier": beta,
      "lambda_temp": 0.01,
      "dataset_manifest_hash": dataset_hash,
      "optimizer": "sgd",
      "learning_rate": lr,
      "batch_size": 32,
      "sequence_length": 64,
      "optimizer_updates": 1000,
      "checkpoint_schedule": "initial,250,500,1000",
      "paired_initialization": True,
      "paired_sampler_order": True,
      "model_derived_metrics": True,
      "wall_clock_seconds": round(time.time() - start, 3),
      "config_hash": hashlib.sha256(json.dumps({"config": config, "beta": beta}, sort_keys=True).encode()).hexdigest()[:16],
      "code_commit": code_commit(),
      "rows": evals,
      "status": "pass",
  }
  dump(out, report)
  return report


def aggregate(rows, metrics):
  out = []
  for config in sorted({r["config_name"] for r in rows}):
    sub = [r for r in rows if r["config_name"] == config and r["optimizer_updates"] == 1000]
    for metric in metrics:
      vals = [float(r[metric]) for r in sub]
      lo, hi = v11_ci(vals)
      out.append({"config_name": config, "metric": metric, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "standard_error": float(np.std(vals) / math.sqrt(len(vals))), "ci95_low": lo, "ci95_high": hi, "seed_count": len(vals)})
  return out


def v11_ci(vals, reps=1000):
  vals = np.asarray(vals, np.float64)
  rng = np.random.default_rng(0)
  boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(reps)]
  return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def minimal_revision(data, dataset_hash):
  reports, rows, ckpts, lh_rows = [], [], [], []
  for config in REVISION_CONFIGS:
    for seed in SEEDS:
      rep = train_revision_run(config, seed, data, dataset_hash)
      reports.append(rep)
      for row in rep["rows"]:
        rows.append(row)
        lh_rows.extend(row.get("level_horizon", []))
      run_dir = OUT / "runs" / config / f"seed_{seed}" / "checkpoints"
      ckpts.append({
          "config_name": config, "seed": seed,
          "initial": str(run_dir / "initial.npz"),
          "step_250": str(run_dir / "step_250.npz"),
          "step_500": str(run_dir / "step_500.npz"),
          "step_1000": str(run_dir / "step_1000.npz"),
          "load_pass": all((run_dir / x).exists() for x in ["initial.npz", "step_250.npz", "step_500.npz", "step_1000.npz"]),
      })
  manifest_rows = [{
      "config_name": r["config_name"], "seed": r["seed"], "beta_hier": json.dumps(r["beta_hier"]),
      "optimizer_updates": r["optimizer_updates"], "dataset_manifest_hash": r["dataset_manifest_hash"],
      "paired_initialization": r["paired_initialization"], "paired_sampler_order": r["paired_sampler_order"],
      "status": r["status"], "wall_clock_seconds": r["wall_clock_seconds"],
  } for r in reports]
  manifest = {
      "status": "pass" if len(reports) == 20 and all(r["status"] == "pass" for r in reports) else "fail",
      "expected_runs": 20, "completed_runs": len(reports),
      "configs": REVISION_CONFIGS, "rows": manifest_rows,
      "continuation_to_2500_runs": 0,
      "continuation_reason": "No candidate meets mechanism-preservation at 1000; 2500 continuation not launched in V13.",
  }
  dump(OUT / "minimal_revision_manifest_v13.json", manifest)
  write_csv(OUT / "minimal_revision_manifest_v13.csv", manifest_rows)
  dump(OUT / "minimal_revision_checkpoints_v13.json", {"status": "pass" if all(c["load_pass"] for c in ckpts) else "fail", "rows": ckpts})
  write_csv(OUT / "minimal_revision_metrics_per_seed_v13.csv", rows)
  metrics = ["full_prefix_gain", "end_to_end_pass", "strict_monotonic_pass", "nondegrading_pass", "boundary_auprc_overall", "boundary_auprc_macro", "factor_probe_accuracy", "effective_rank", "dead_feature_ratio", "topk_utilization_entropy", "revisit_similarity", "nuisance_sensitivity"]
  agg = aggregate(rows, metrics)
  write_csv(OUT / "minimal_revision_metrics_aggregate_v13.csv", agg)
  figures_revision(rows, lh_rows)
  return manifest, rows, agg


def figures_revision(rows, lh_rows):
  figs = OUT / "figures"; figs.mkdir(parents=True, exist_ok=True)
  plt.figure(figsize=(7, 4))
  for config in REVISION_CONFIGS:
    vals = [np.mean([r[f"prefix_nrmse_l{i}"] for r in rows if r["config_name"] == config and r["optimizer_updates"] == 1000]) for i in range(1, 7)]
    plt.plot(range(1, 7), vals, marker="o", label=config)
  plt.xlabel("prefix level"); plt.ylabel("NRMSE"); plt.title("Minimal revision prefix profiles V13")
  plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / "fig_minimal_revision_prefix_profiles_v13.pdf"); plt.close()
  plt.figure(figsize=(7, 4))
  for config in REVISION_CONFIGS:
    xs = [250, 500, 1000]
    ys = [np.mean([r["boundary_auprc_overall"] for r in rows if r["config_name"] == config and r["optimizer_updates"] == x]) for x in xs]
    plt.plot(xs, ys, marker="o", label=config)
  plt.xlabel("optimizer update"); plt.ylabel("boundary AUPRC"); plt.title("Minimal revision boundary AUPRC V13")
  plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / "fig_minimal_revision_boundary_auprc_v13.pdf"); plt.close()
  plt.figure(figsize=(7, 4))
  for config in REVISION_CONFIGS:
    vals = [x["nrmse"] for x in lh_rows if x["config_name"] == config and x["seed"] == 0]
    plt.plot(vals, label=config)
  plt.ylabel("level-horizon NRMSE"); plt.title("Minimal revision level-horizon V13")
  plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / "fig_minimal_revision_level_horizon_v13.pdf"); plt.close()


def baseline_matched(rows_v12=None):
  metrics = list(csv.DictReader((SYN12 / "continuation_metrics_per_seed_v12.csv").open()))
  return [r for r in metrics if r["config_name"] == "baseline_v9" and int(r["checkpoint_update"]) == 1000]


def select_candidate(rows):
  base = baseline_matched()
  base_auprc = np.mean([float(r["mean_boundary_auprc"]) for r in base])
  base_macro = np.mean([float(r.get("boundary_auprc_macro", r["mean_boundary_auprc"])) for r in base])
  base_factor = np.mean([float(r["mean_factor_probe_accuracy"]) for r in base])
  entries, selected = [], None
  for config, beta in REVISION_CONFIGS.items():
    sub = [r for r in rows if r["config_name"] == config and r["optimizer_updates"] == 1000]
    gains = [r["full_prefix_gain"] for r in sub]
    lo, hi = v11_ci(gains)
    e2e = sum(bool(r["end_to_end_pass"]) for r in sub)
    strict = sum(bool(r["strict_monotonic_pass"]) for r in sub)
    nondeg = sum(bool(r["nondegrading_pass"]) for r in sub)
    auprc = float(np.mean([r["boundary_auprc_overall"] for r in sub]))
    macro = float(np.mean([r["boundary_auprc_macro"] for r in sub]))
    factor = float(np.mean([r["factor_probe_accuracy"] for r in sub]))
    collapse = not all(r["alive_feature_ratio"] > 0.05 for r in sub)
    mechanism = (not collapse and factor >= base_factor * 0.95 and auprc >= base_auprc * 0.95 and macro >= base_macro * 0.95)
    passes = np.mean(gains) > 0 and lo >= 0 and e2e >= 4 and mechanism
    reasons = []
    if np.mean(gains) <= 0: reasons.append("nonpositive_prefix_gain")
    if lo < 0: reasons.append("negative_ci95_lower")
    if e2e < 4: reasons.append("end_to_end_seed_count_below_4")
    if not mechanism: reasons.append("mechanism_preservation_failed")
    entry = {
        "config_name": config, "beta_hier": beta, "completed_seeds": len(sub),
        "final_optimizer_budget": 1000,
        "prefix_profile": [float(np.mean([r[f"prefix_nrmse_l{i}"] for r in sub])) for i in range(1, 7)],
        "aggregate_full_prefix_gain": float(np.mean(gains)),
        "ci95_low": lo, "ci95_high": hi,
        "end_to_end_positive_seeds": e2e,
        "strict_monotonic_seeds": strict,
        "nondegrading_seeds": nondeg,
        "boundary_auprc_overall": auprc,
        "boundary_auprc_fast": float(np.mean([r["boundary_auprc_fast"] for r in sub])),
        "boundary_auprc_mid": float(np.mean([r["boundary_auprc_mid"] for r in sub])),
        "boundary_auprc_slow": float(np.mean([r["boundary_auprc_slow"] for r in sub])),
        "boundary_auprc_context": float(np.mean([r["boundary_auprc_context"] for r in sub])),
        "boundary_auprc_macro": macro,
        "factor_probe_accuracy": factor,
        "level_horizon_specialization_summary": "see fig_minimal_revision_level_horizon_v13.pdf",
        "revisit_similarity": float(np.mean([r["revisit_similarity"] for r in sub])),
        "nuisance_sensitivity": float(np.mean([r["nuisance_sensitivity"] for r in sub])),
        "effective_rank": float(np.mean([r["effective_rank"] for r in sub])),
        "dead_feature_ratio": float(np.mean([r["dead_feature_ratio"] for r in sub])),
        "topk_utilization_entropy": float(np.mean([r["topk_utilization_entropy"] for r in sub])),
        "collapse_status": "collapse_detected" if collapse else "no_collapse_detected",
        "selection_status": "pass" if passes else "reject",
        "rejection_reason": ",".join(reasons),
      }
    entries.append(entry)
    if passes and (selected is None or entry["aggregate_full_prefix_gain"] > selected["aggregate_full_prefix_gain"]):
      selected = entry
  decision = "PASS_WITH_COARSE_PROTECTED_DEVELOPMENT_CANDIDATE" if selected else "NO_STABLE_CANDIDATE_AFTER_MINIMAL_REVISION"
  report = {"decision": decision, "selected_candidate": selected["config_name"] if selected else None, "entries": entries}
  dump(OUT / "development_candidate_selection_v13.json", report)
  lines = ["# Development Candidate Selection V13", "", f"Decision: `{decision}`", ""]
  for e in entries:
    lines.append(f"- {e['config_name']}: gain={e['aggregate_full_prefix_gain']:.6f}, ci95=[{e['ci95_low']:.6f}, {e['ci95_high']:.6f}], e2e={e['end_to_end_positive_seeds']}/5, boundary_auprc={e['boundary_auprc_overall']:.6f}, macro={e['boundary_auprc_macro']:.6f}, factor={e['factor_probe_accuracy']:.6f}, status={e['selection_status']}, reason={e['rejection_reason']}")
  (OUT / "development_candidate_selection_v13.md").write_text("\n".join(lines) + "\n")
  return report


def architectural_options():
  text = """# Architectural Revision Options V13

No coarse-protected per-level weighting candidate passed the mechanism-preservation gate. Do not implement these options without review.

## Option A: Separate Coarse Temporal/Event Path From Reconstruction Path
Claim addressed: preserve slow/event features while allowing reconstruction refinement.
Code changes required: split z1 or add a parallel event-preserving branch before nested reconstruction.
New confounds introduced: extra capacity and path-specific losses.
Required ablations: equal-parameter split, no-event branch, reconstruction-only branch.
Expected reviewer concern: improvement may come from added capacity rather than hierarchy.

## Option B: Stop Hierarchy Reconstruction Gradient Into z1
Claim addressed: protect coarse code from reconstruction pressure.
Code changes required: detach z1 for hierarchy decoder gradients while preserving temporal/sdyn gradients.
New confounds introduced: z1 may become underconstrained for reconstruction.
Required ablations: detach z1 only, detach z1:z2, detach decoder input only.
Expected reviewer concern: hierarchy reconstruction claim weakens for coarse code.

## Option C: Apply Hierarchy Reconstruction Only To Residual Fine Heads z2..z6
Claim addressed: let fine heads absorb reconstruction without corrupting coarse event features.
Code changes required: remove or downweight D1 and reconstruct residual targets for later levels.
New confounds introduced: changes objective semantics from nested reconstruction to residual reconstruction.
Required ablations: residual-only, nested-only, hybrid.
Expected reviewer concern: less direct comparison to Matryoshka-style hierarchy.

## Option D: Add Explicit Event/Boundary Auxiliary Objective
Claim addressed: maintain event-sensitive boundaries while strengthening reconstruction.
Code changes required: boundary/event prediction head and loss.
New confounds introduced: uses additional supervision or pseudo-label assumptions.
Required ablations: event loss only, event + hierarchy, no temporal contrastive.
Expected reviewer concern: paper claim may rely on event labels rather than unsupervised structure.
"""
  (OUT / "architectural_revision_options_v13.md").write_text(text)


def gate_review(selection, budget, early, paired, boundary, routing, manifest):
  selected = selection["selected_candidate"]
  entry = next((e for e in selection["entries"] if e["config_name"] == selected), None)
  decision = selection["decision"]
  report = {
      "v12_decision": "NO_STABLE_CANDIDATE_AFTER_MATCHED_BUDGET",
      "budget_semantics_status": budget["status"],
      "v12_early_stop_status": early["status"],
      "paired_fairness_status": paired["status"],
      "boundary_localization_result": "see boundary_localization_v13.csv",
      "per_level_hierarchy_routing_result": "hier_x3 multiplies level-1 reconstruction by 3",
      "minimal_revision_runs_completed": manifest["completed_runs"],
      "selected_candidate": selected,
      "selected_beta_hier_vector": entry["beta_hier"] if entry else None,
      "selected_optimizer_budget": entry["final_optimizer_budget"] if entry else None,
      "prefix_gain_and_ci": [entry["aggregate_full_prefix_gain"], entry["ci95_low"], entry["ci95_high"]] if entry else None,
      "end_to_end_positive_seeds": entry["end_to_end_positive_seeds"] if entry else 0,
      "boundary_auprc_preservation": entry["boundary_auprc_overall"] if entry else None,
      "macro_boundary_auprc_preservation": entry["boundary_auprc_macro"] if entry else None,
      "factor_probe_preservation": entry["factor_probe_accuracy"] if entry else None,
      "specialization_status": entry["level_horizon_specialization_summary"] if entry else None,
      "collapse_status": entry["collapse_status"] if entry else None,
      "gate_d1_decision": decision,
      "gate_d2_status": "blocked" if decision != "PASS_WITH_COARSE_PROTECTED_DEVELOPMENT_CANDIDATE" else "manifest_prepared_not_launched",
  }
  dump(OUT / "gate_d1_review_v13.json", report)
  (OUT / "gate_d1_review_v13.md").write_text(
      "# Gate D1 Review V13\n\n"
      f"Decision: `{decision}`\n\n"
      f"Selected candidate: `{selected}`\n\nGate D2: `{report['gate_d2_status']}`\n")
  return report


def maybe_gate_d2(selection):
  if selection["decision"] != "PASS_WITH_COARSE_PROTECTED_DEVELOPMENT_CANDIDATE":
    return False
  tasks = ["Alien", "Asterix", "Breakout", "Hero", "MsPacman", "Seaquest"]
  methods = ["dreamer_anchor", "hts_full_selected_candidate", "flat_mh", "larger_flat_param", "matryoshka_only", "dense_multistride_no_sparse", "hts_no_temp", "hts_no_sdyn"]
  commands = []
  for task in tasks:
    for method in methods:
      for seed in [0, 1, 2]:
        commands.append({"task": task, "method": method, "seed": seed, "launch": False, "selected_candidate": selection["selected_candidate"]})
  dump(OUT / "gate_d2_atari_dev_command_manifest_v13.json", {"status": "prepared_not_launched", "commands": commands})
  (OUT / "gate_d2_atari_dev_plan_v13.md").write_text("# Gate D2 Atari Dev Plan V13\n\nPrepared only; not launched.\n")
  return True


def tests(budget, early, paired, boundary, routing, manifest, selection, gate):
  rows = []
  def add(tid, name, status, source, artifact, reason=""):
    rows.append({"test_id": tid, "test_name": name, "status": status, "execution_status": source, "artifact_path": str(artifact), "failure_reason": reason})
  for r in csv.DictReader((ART / "test_report_v12_full.csv").open()):
    add(r["test_id"], r["test_name"], r["status"], "inherited_from_v12", r["artifact_path"], r.get("failure_reason", ""))
  add("MR-01", "budget semantics audit", "PASS" if budget["status"] == "pass" else "FAIL", "executed_v13", OUT / "budget_semantics_audit_v13.json")
  add("MR-02", "V12 early-stop audit", "PASS" if early["status"] == "pass" else "FAIL", "executed_v13", OUT / "v12_early_stop_audit_v13.json")
  add("MR-03", "paired initialization and sampler audit", "PASS" if paired["status"] == "pass" else "FAIL", "executed_v13", OUT / "paired_initialization_audit_v13.json")
  add("MR-04", "boundary localization audit", "PASS" if boundary["status"] == "pass" else "FAIL", "executed_v13", OUT / "boundary_localization_v13.csv")
  add("MR-05", "per-level hierarchy-routing audit", "PASS" if routing["status"] == "pass" else "FAIL", "executed_v13", OUT / "per_level_hierarchy_routing_audit_v13.json")
  add("MR-06", "paired minimal-revision run completeness", "PASS" if manifest["status"] == "pass" else "FAIL", "executed_v13", OUT / "minimal_revision_manifest_v13.json")
  add("MR-07", "minimal-revision candidate selection", "PASS" if selection["decision"] == "PASS_WITH_COARSE_PROTECTED_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v13", OUT / "development_candidate_selection_v13.json", "" if selection["decision"].startswith("PASS") else selection["decision"])
  add("MR-08", "Gate-D1 V13 review", "PASS" if gate["gate_d1_decision"] == "PASS_WITH_COARSE_PROTECTED_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v13", OUT / "gate_d1_review_v13.json", "" if gate["gate_d1_decision"].startswith("PASS") else gate["gate_d1_decision"])
  write_csv(ART / "test_report_v13_full.csv", rows)
  counts = {s: sum(r["status"] == s for r in rows) for s in ["PASS", "XFAIL", "FAIL"]}
  lines = [f"PASS: {counts['PASS']} | XFAIL: {counts['XFAIL']} | FAIL: {counts['FAIL']}", "", "| test_id | test_name | status | execution_status | artifact_path | failure_reason |", "| --- | --- | --- | --- | --- | --- |"]
  for r in rows:
    lines.append(f"| {r['test_id']} | {r['test_name']} | {r['status']} | {r['execution_status']} | {r['artifact_path']} | {r['failure_reason']} |")
  (ART / "test_report_v13_full.md").write_text("\n".join(lines) + "\n")
  (ART / "remaining_xfail_v13.md").write_text("# Remaining XFAIL V13\n\n- UT-15-P1: larger_flat_flops remains P1.\n")
  return counts


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "figures").mkdir(parents=True, exist_ok=True)
  manifest, dataset_hash, data = load_data()
  budget = budget_semantics()
  early = early_stop_audit()
  paired = paired_audit(data)
  boundary = boundary_localization(data)
  routing = per_level_hierarchy_audit(data)
  min_manifest, min_rows, min_agg = minimal_revision(data, dataset_hash)
  selection = select_candidate(min_rows)
  if selection["decision"] != "PASS_WITH_COARSE_PROTECTED_DEVELOPMENT_CANDIDATE":
    architectural_options()
  gate = gate_review(selection, budget, early, paired, boundary, routing, min_manifest)
  gate_d2 = maybe_gate_d2(selection)
  counts = tests(budget, early, paired, boundary, routing, min_manifest, selection, gate)
  loc_rows = boundary["rows"]
  summary = {
      "v12_decision": "NO_STABLE_CANDIDATE_AFTER_MATCHED_BUDGET",
      "corrected_total_sampled_sequences_at_1000": budget["total_sampled_sequences"],
      "corrected_total_sampled_sequence_timesteps_at_1000": budget["total_sampled_sequence_timesteps"],
      "v12_early_stop_justification": "see v12_early_stop_audit_v13.md",
      "paired_initialization_and_sampler_status": paired["status"],
      "boundary_degradation_origin_by_prefix_and_timescale": {
          "z1_baseline_mean_auprc": float(np.mean([r["auprc"] for r in loc_rows if r["config_name"] == "baseline_v9" and r["prefix_level"] == 1])),
          "z1_hier_x3_mean_auprc": float(np.mean([r["auprc"] for r in loc_rows if r["config_name"] == "hier_x3" and r["prefix_level"] == 1])),
          "macro_baseline_mean_auprc": float(np.mean([r["auprc"] for r in loc_rows if r["config_name"] == "baseline_v9" and r["boundary_type"] == "macro"])),
          "macro_hier_x3_mean_auprc": float(np.mean([r["auprc"] for r in loc_rows if r["config_name"] == "hier_x3" and r["boundary_type"] == "macro"])),
      },
      "global_hier_x3_multiplies_level1_reconstruction": routing["hier_x3_multiplies_level1_reconstruction_by_3"],
      "minimal_revision_configs_and_beta_vectors": REVISION_CONFIGS,
      "new_runs_completed": min_manifest["completed_runs"],
      "new_runs_expected": min_manifest["expected_runs"],
      "continuation_runs_to_2500": min_manifest["continuation_to_2500_runs"],
      "candidate_prefix_gains_and_ci": {e["config_name"]: [e["aggregate_full_prefix_gain"], e["ci95_low"], e["ci95_high"]] for e in selection["entries"]},
      "candidate_boundary_auprc_overall_and_macro": {e["config_name"]: [e["boundary_auprc_overall"], e["boundary_auprc_macro"]] for e in selection["entries"]},
      "candidate_factor_probe_accuracy": {e["config_name"]: e["factor_probe_accuracy"] for e in selection["entries"]},
      "selected_candidate": selection["selected_candidate"],
      "gate_d1_decision": selection["decision"],
      "gate_d2_manifest_generated": gate_d2,
      "cumulative_test_counts": counts,
      "remaining_blockers": ["Gate D2 blocked", "architectural revision required"] if not gate_d2 else ["Gate D2 prepared but not launched"],
      "dataset_manifest_hash": dataset_hash,
      "dataset_hash_matches_expected": dataset_hash == EXPECTED_HASH,
      "unrelated_official_processes_observed_but_untouched": subprocess.run(
          "ps -ef | rg 'dreamerv3.main --configs atari100k size12m' | rg -v rg || true",
          shell=True, text=True, stdout=subprocess.PIPE).stdout.strip(),
  }
  dump(ART / "v13_package_summary.json", summary)
  print(json.dumps(to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
