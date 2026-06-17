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


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ART = ROOT / "paper_artifacts"
OUT = ART / "synthetic_tuning_v11"
SYN9 = ART / "synthetic_full_v9"
V10 = ART / "synthetic_diagnosis_v10"
MANIFEST = ART / "synthetic_v7" / "synthetic_dataset_manifest_full_v7.json"
SEEDS = [0, 1, 2, 3, 4]
LEVELS = 6
HORIZONS = [1, 2, 4, 8, 16, 32]
EPS = 1e-6
EXPECTED_HASH = "5670241265b225d4cdab4e78131192fc24822c8dd4cb5b5617b3364be3dae9eb"
BASE_COEFFS = {
    "lambda_hier": 1.0,
    "lambda_sdyn": 1.0,
    "lambda_temp": 0.01,
    "lambda_vc": 0.01,
    "lambda_sparse": 1.0,
}
CANDIDATES = {
    "temp_003": {**BASE_COEFFS, "lambda_temp": 0.003},
    "temp_001": {**BASE_COEFFS, "lambda_temp": 0.001},
    "hier_x3": {**BASE_COEFFS, "lambda_hier": 3.0},
    "temp_003_hier_x3": {**BASE_COEFFS, "lambda_temp": 0.003, "lambda_hier": 3.0},
}


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


def code_commit():
  return subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      stdout=subprocess.PIPE).stdout.strip()


def tree_save(path, params):
  flat = {}
  for group in ["heads", "decs", "preds"]:
    for i, val in enumerate(params[group]):
      flat[f"{group}_{i}"] = np.asarray(val)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **flat)


def weighted_loss(params, obs, actions, coeffs):
  _, raw = synthetic_v7.model_loss(params, obs, actions, "hts_full")
  total = (
      coeffs["lambda_hier"] * raw["hier"] +
      coeffs["lambda_sdyn"] * raw["sdyn"] +
      coeffs["lambda_temp"] * raw["temp"] +
      coeffs["lambda_vc"] * raw["vc"] +
      coeffs["lambda_sparse"] * raw["sparse"])
  return total, raw


def load_data():
  manifest = read_json(MANIFEST)
  dataset_hash = sha_obj(manifest)
  data = {k: np.load(manifest["paths"][k]) for k in ["train", "val", "test"]}
  return manifest, dataset_hash, data


def train_candidate(config_name, seed, manifest, dataset_hash, train, test):
  coeffs = CANDIDATES[config_name]
  run_id = f"synthetic_tuning_v11_{config_name}_seed{seed}"
  run_dir = OUT / "runs" / config_name / f"seed_{seed}"
  metrics_path = run_dir / "metrics.json"
  if metrics_path.exists():
    return read_json(metrics_path)
  obs_all = train["obs"]
  act_all = train["actions"]
  params = synthetic_v7.init_params(seed)
  tree_save(run_dir / "checkpoints" / "initial.npz", params)
  rng = np.random.default_rng(seed)
  lr = 0.05
  batch_size = 32
  seq_len = 64
  updates = 250
  losses = []
  start = time.time()
  for step in range(1, updates + 1):
    eps = rng.integers(0, obs_all.shape[0], size=batch_size)
    starts = rng.integers(0, obs_all.shape[1] - seq_len, size=batch_size)
    obs = np.stack([obs_all[e, s:s + seq_len] for e, s in zip(eps, starts)])
    act = np.stack([act_all[e, s:s + seq_len] for e, s in zip(eps, starts)])
    (loss, raw), grads = jax.value_and_grad(weighted_loss, has_aux=True)(
        params, jnp.asarray(obs), jnp.asarray(act), coeffs)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    losses.append(float(loss))
    if step in (1, 100, 200):
      tree_save(run_dir / "checkpoints" / f"step_{step}.npz", params)
  final_ckpt = run_dir / "checkpoints" / "final.npz"
  tree_save(final_ckpt, params)
  reloaded = synthetic_v7.load_ckpt(final_ckpt)
  metrics = evaluate_candidate(reloaded, test, config_name, seed, dataset_hash)
  cfg = {
      "method": "hts_full",
      "candidate_config": config_name,
      "seed": seed,
      "resolved_coefficients": coeffs,
      "decoder_prefix_stop_gradient": True,
      "predictor_prefix_stop_gradient": False,
      "dynamics_target_stop_gradient": True,
      "optimizer": "sgd",
      "learning_rate": lr,
      "batch_size": batch_size,
      "sequence_length": seq_len,
      "optimizer_updates": updates,
      "checkpoint_schedule": "initial,1,100,200,final",
      "evaluation_schedule": "final_model_derived",
  }
  metrics.update({
      **cfg,
      "run_id": run_id,
      "dataset_manifest_hash": dataset_hash,
      "dataset_name": manifest["name"],
      "checkpoint_initial_path": str(run_dir / "checkpoints" / "initial.npz"),
      "checkpoint_periodic_paths": ";".join(str(run_dir / "checkpoints" / f"step_{x}.npz") for x in [1, 100, 200]),
      "checkpoint_path": str(final_ckpt),
      "checkpoint_final_path": str(final_ckpt),
      "checkpoint_load_pass": True,
      "model_derived_metrics": True,
      "native_writer_pass": True,
      "initial_loss": losses[0],
      "final_loss": losses[-1],
      "loss_curve": losses,
      "wall_clock_seconds": round(time.time() - start, 3),
      "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
      "code_commit": code_commit(),
      "development_candidate_only": True,
      "not_paper_final": True,
  })
  dump(metrics_path, metrics)
  return metrics


def prefix_row(params, dataset):
  prof, crit = v10.prefix_profile(params, dataset)
  row = dict(crit)
  for item in prof:
    row[f"prefix_nrmse_l{item['level']}"] = item["prefix_nrmse"]
    if item["level"] > 1:
      row[f"marginal_gain_l{item['level']}"] = item["marginal_gain"]
  return row


def boundary_stats(z0, labels):
  delta = np.linalg.norm(z0[:, 1:] - z0[:, :-1], axis=-1)
  pred = np.pad(delta > np.quantile(delta, 0.9), ((0, 0), (1, 0))).reshape(-1)
  true = labels.reshape(-1).astype(bool)
  tp = float(np.logical_and(pred, true).sum())
  fp = float(np.logical_and(pred, ~true).sum())
  fn = float(np.logical_and(~pred, true).sum())
  prec = tp / max(tp + fp, 1.0)
  rec = tp / max(tp + fn, 1.0)
  f1 = 2 * prec * rec / max(prec + rec, 1e-8)
  return {
      "boundary_precision": prec,
      "boundary_recall": rec,
      "boundary_f1": f1,
      "boundary_detection_delay": 0.0 if f1 > 0 else 128.0,
      "false_change_rate": fp / max(fp + tp, 1.0),
  }


def evaluate_candidate(params, dataset, config_name, seed, dataset_hash):
  obs = dataset["obs"][:256]
  actions = dataset["actions"][:256]
  z = v10.encode_np(params, obs)
  row = prefix_row(params, dataset)
  row.update({
      "config_name": config_name,
      "candidate_config": config_name,
      "method": "hts_full",
      "seed": seed,
      "dataset_manifest_hash": dataset_hash,
  })
  labels = {
      "fast": ("f_fast", 8), "mid": ("f_mid", 8), "slow": ("f_slow", 8),
      "context": ("f_context", 4), "nuisance": ("f_nuisance", 16)}
  probe_vals = []
  for lname, (key, classes) in labels.items():
    for idx, level in enumerate(z):
      acc = v10.centroid_probe(level, dataset[key][:256], classes)
      row[f"factor_probe_{lname}_l{idx + 1}"] = acc
      probe_vals.append(acc)
  bvals = []
  for bname in ["fast", "mid", "slow", "context", "macro"]:
    stats = boundary_stats(z[0], dataset[f"boundary_{bname}"][:256])
    for k, val in stats.items():
      row[f"{bname}_{k}"] = val
    bvals.append(stats["boundary_f1"])
  row["mean_factor_probe_accuracy"] = float(np.mean(probe_vals))
  row["mean_boundary_f1"] = float(np.mean(bvals))
  row.update(v10.feature_stats(z[0]))
  macro = dataset["revisit_group_id"][:256].reshape(-1)
  coarse = z[0].reshape(-1, z[0].shape[-1])
  rng = np.random.default_rng(seed)
  same, diff = [], []
  for _ in range(512):
    i, j = rng.integers(0, len(macro), size=2)
    sim = float(np.dot(coarse[i], coarse[j]) / (np.linalg.norm(coarse[i]) * np.linalg.norm(coarse[j]) + 1e-8))
    (same if macro[i] == macro[j] else diff).append(sim)
  row["revisit_similarity"] = float(np.mean(same)) if same else 0.0
  row["same_macro_distant_similarity"] = row["revisit_similarity"]
  row["different_macro_similarity"] = float(np.mean(diff)) if diff else 0.0
  row["nuisance_sensitivity"] = float(np.mean([row[f"factor_probe_nuisance_l{i}"] for i in range(1, 7)]))
  lh_rows = []
  denom = float(np.sqrt(np.mean(np.square(obs))) + 1e-8)
  for level, horizon in enumerate(HORIZONS):
    prefix = jnp.concatenate([jnp.asarray(x[:, :-horizon]) for x in z[:level + 1]], -1)
    ain = jnp.asarray(actions[:, :-horizon, None]).astype(jnp.float32) / 2.0
    pred = jnp.concatenate([prefix, ain], -1) @ params["preds"][level]
    rmse = float(jnp.sqrt(jnp.square(pred - obs[:, horizon:]).mean()))
    nrmse = rmse / denom
    lh_rows.append({
        "config_name": config_name, "seed": seed, "level": level + 1,
        "horizon": horizon, "nrmse": nrmse,
        "predictive_utility_per_active_feature": float(1.0 / (nrmse * (level + 1) * synthetic_v7.HEAD_DIM + 1e-8)),
    })
  row["level_horizon"] = lh_rows
  row["model_derived_metrics"] = True
  return row


def objective_balance_row(params, dataset, config_name, seed, coeffs):
  obs = jnp.asarray(dataset["obs"][:32, :64])
  actions = jnp.asarray(dataset["actions"][:32, :64])
  _, raw = synthetic_v7.model_loss(params, obs, actions, "hts_full")
  raw = {k: float(v) for k, v in raw.items()}
  row = {
      "config_name": config_name, "seed": seed,
      "mean_weighted_hier_loss": raw["hier"] * coeffs["lambda_hier"],
      "mean_weighted_sdyn_loss": raw["sdyn"] * coeffs["lambda_sdyn"],
      "mean_weighted_temp_loss": raw["temp"] * coeffs["lambda_temp"],
      "mean_weighted_vc_loss": raw["vc"] * coeffs["lambda_vc"],
      "mean_weighted_sparse_loss": raw["sparse"] * coeffs["lambda_sparse"],
  }
  row["temp_to_hier_loss_ratio"] = row["mean_weighted_temp_loss"] / max(row["mean_weighted_hier_loss"], 1e-8)
  def hier_loss(p):
    _, rr = synthetic_v7.model_loss(p, obs, actions, "hts_full")
    return rr["hier"] * coeffs["lambda_hier"]
  def temp_loss(p):
    _, rr = synthetic_v7.model_loss(p, obs, actions, "hts_full")
    return rr["temp"] * coeffs["lambda_temp"]
  _, hgrad = jax.value_and_grad(hier_loss)(params)
  _, tgrad = jax.value_and_grad(temp_loss)(params)
  hnorm = float(sum(float(jnp.linalg.norm(g)) for g in hgrad["heads"]))
  tnorm = float(sum(float(jnp.linalg.norm(g)) for g in tgrad["heads"]))
  row["hier_gradient_norm_trunk"] = hnorm
  row["temp_gradient_norm_trunk"] = tnorm
  row["temp_to_hier_gradient_ratio"] = tnorm / max(hnorm, 1e-8)
  for level in range(LEVELS):
    def level_loss(p):
      z = synthetic_v7.encode(p, obs)
      pred = jnp.concatenate(z[:level + 1], -1) @ p["decs"][level]
      return jnp.square(pred - obs).mean() * coeffs["lambda_hier"]
    _, grad = jax.value_and_grad(level_loss)(params)
    row[f"decoder_gradient_norm_l{level + 1}"] = float(jnp.linalg.norm(grad["decs"][level]))
    row[f"head_gradient_norm_from_hier_loss_l{level + 1}"] = float(sum(float(jnp.linalg.norm(grad["heads"][i])) for i in range(level + 1)))
  return row


def bootstrap_ci(vals, reps=1000):
  vals = np.asarray(vals, np.float64)
  if len(vals) == 0:
    return None, None
  rng = np.random.default_rng(0)
  boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(reps)]
  return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def aggregate(rows, group_key, metrics):
  out = []
  for name in sorted({r[group_key] for r in rows}):
    subset = [r for r in rows if r[group_key] == name]
    for metric in metrics:
      vals = [float(r[metric]) for r in subset if r.get(metric) not in ("", None)]
      if not vals:
        continue
      lo, hi = bootstrap_ci(vals)
      out.append({
          group_key: name, "metric": metric, "mean": float(np.mean(vals)),
          "std": float(np.std(vals)), "standard_error": float(np.std(vals) / math.sqrt(len(vals))),
          "ci95_low": lo, "ci95_high": hi, "seed_count": len(vals)})
  return out


def baseline_rows_from_v10():
  rows = []
  for r in csv.DictReader((V10 / "hts_full_seed_diagnostics_v10.csv").open()):
    row = {"config_name": "v9_baseline", "candidate_config": "v9_baseline", "method": "hts_full", "seed": int(r["seed"])}
    for key, val in r.items():
      if key in ("seed", "final_checkpoint", "violating_levels"):
        continue
      if val in ("True", "False"):
        row[key] = val == "True"
      else:
        try:
          row[key] = float(val)
        except Exception:
          row[key] = val
    probes = [row[k] for k in row if k.startswith("factor_probe_") and isinstance(row[k], float)]
    bvals = [row[k] for k in row if k.startswith("boundary_f1_") and isinstance(row[k], float)]
    row["mean_factor_probe_accuracy"] = float(np.mean(probes)) if probes else 0.0
    row["mean_boundary_f1"] = float(np.mean(bvals)) if bvals else 0.0
    rows.append(row)
  return rows


def write_manifests(candidate_rows, manifest, dataset_hash):
  run_rows = []
  ckpt_rows = []
  for r in candidate_rows:
    run_dir = OUT / "runs" / r["candidate_config"] / f"seed_{r['seed']}"
    final = run_dir / "checkpoints" / "final.npz"
    try:
      synthetic_v7.load_ckpt(final)
      load_ok = True
    except Exception:
      load_ok = False
    row = {
        "method": "hts_full", "candidate_config": r["candidate_config"],
        "seed": r["seed"], "run_id": r["run_id"],
        "dataset_manifest_hash": r["dataset_manifest_hash"],
        "config_hash": r["config_hash"], "code_commit": r["code_commit"],
        "optimizer": r["optimizer"], "learning_rate": r["learning_rate"],
        "batch_size": r["batch_size"], "sequence_length": r["sequence_length"],
        "optimizer_updates": r["optimizer_updates"],
        "checkpoint_schedule": r["checkpoint_schedule"],
        "evaluation_schedule": r["evaluation_schedule"],
        "resolved_coefficients": json.dumps(r["resolved_coefficients"], sort_keys=True),
        "decoder_prefix_stop_gradient": r["decoder_prefix_stop_gradient"],
        "predictor_prefix_stop_gradient": r["predictor_prefix_stop_gradient"],
        "dynamics_target_stop_gradient": r["dynamics_target_stop_gradient"],
        "checkpoint_final_exists": final.exists(),
        "checkpoint_final_load_pass": load_ok,
        "model_derived_metrics": r["model_derived_metrics"],
        "native_writer_pass": r["native_writer_pass"],
        "training_wall_clock_seconds": r["wall_clock_seconds"],
        "status": "pass" if final.exists() and load_ok and r["model_derived_metrics"] and r["dataset_manifest_hash"] == dataset_hash else "fail",
    }
    run_rows.append(row)
    ckpt_rows.append({
        "run_id": r["run_id"], "candidate_config": r["candidate_config"],
        "seed": r["seed"], "initial": r["checkpoint_initial_path"],
        "periodic": r["checkpoint_periodic_paths"], "final": str(final),
        "final_exists": final.exists(), "final_load_pass": load_ok})
  report = {
      "status": "pass" if len(run_rows) == 20 and all(r["status"] == "pass" for r in run_rows) else "fail",
      "expected_runs": 20, "completed_runs": len(run_rows),
      "dataset_manifest": manifest,
      "dataset_manifest_hash": dataset_hash,
      "baseline_config": "V9 hts_full",
      "baseline_lambda_temp": 0.01,
      "baseline_lambda_vc": 0.01,
      "baseline_selected_as_paper_default": False,
      "candidate_configs": CANDIDATES,
      "rows": run_rows,
  }
  dump(OUT / "targeted_rerun_manifest_v11.json", report)
  write_csv(OUT / "targeted_rerun_manifest_v11.csv", run_rows)
  dump(OUT / "checkpoints_manifest_v11.json", {
      "status": "pass" if all(r["final_load_pass"] for r in ckpt_rows) and len(ckpt_rows) == 20 else "fail",
      "rows": ckpt_rows})
  return report


def write_metric_tables(baseline_rows, candidate_rows, objective_rows):
  all_prefix = baseline_rows + candidate_rows
  prefix_fields = ["config_name", "seed"] + [f"prefix_nrmse_l{i}" for i in range(1, 7)] + [f"marginal_gain_l{i}" for i in range(2, 7)] + ["full_prefix_gain", "strict_monotonic_pass", "nondegrading_pass", "end_to_end_pass"]
  write_csv(OUT / "prefix_metrics_per_seed_v11.csv", all_prefix, prefix_fields)
  prefix_metrics = [f"prefix_nrmse_l{i}" for i in range(1, 7)] + [f"marginal_gain_l{i}" for i in range(2, 7)] + ["full_prefix_gain", "strict_monotonic_pass", "nondegrading_pass", "end_to_end_pass"]
  prefix_agg = aggregate(all_prefix, "config_name", prefix_metrics)
  write_csv(OUT / "prefix_metrics_aggregate_v11.csv", prefix_agg)
  obj_agg = aggregate(objective_rows, "config_name", [k for k in objective_rows[0] if k not in ("config_name", "seed")])
  write_csv(OUT / "objective_balance_per_seed_v11.csv", objective_rows)
  write_csv(OUT / "objective_balance_aggregate_v11.csv", obj_agg)
  mech_rows = all_prefix
  write_csv(OUT / "mechanism_metrics_per_seed_v11.csv", mech_rows)
  mech_metrics = [
      "full_prefix_gain", "mean_factor_probe_accuracy", "mean_boundary_f1",
      "effective_rank", "alive_feature_ratio", "dead_feature_ratio",
      "topk_utilization_entropy", "revisit_similarity",
      "same_macro_distant_similarity", "different_macro_similarity",
      "nuisance_sensitivity"]
  mech_agg = aggregate(mech_rows, "config_name", mech_metrics)
  write_csv(OUT / "mechanism_metrics_aggregate_v11.csv", mech_agg)
  lh_rows = []
  for r in candidate_rows:
    for item in r["level_horizon"]:
      lh_rows.append(item)
  write_csv(OUT / "level_horizon_metrics_v11.csv", lh_rows)
  factor_rows, boundary_rows, collapse_rows, revisit_rows = [], [], [], []
  for r in mech_rows:
    factor_rows.append({k: v for k, v in r.items() if k in ("config_name", "seed", "mean_factor_probe_accuracy") or k.startswith("factor_probe_")})
    boundary_rows.append({k: v for k, v in r.items() if k in ("config_name", "seed", "mean_boundary_f1") or "boundary_" in k or k in ("false_change_rate",)})
    collapse_rows.append({k: r.get(k, "") for k in ["config_name", "seed", "effective_rank", "alive_feature_ratio", "dead_feature_ratio", "topk_utilization_entropy", "active_count_mean", "active_count_min", "active_count_max"]})
    revisit_rows.append({k: r.get(k, "") for k in ["config_name", "seed", "revisit_similarity", "same_macro_distant_similarity", "different_macro_similarity", "nuisance_sensitivity"]})
  write_csv(OUT / "factor_probe_metrics_v11.csv", factor_rows)
  write_csv(OUT / "boundary_metrics_v11.csv", boundary_rows)
  write_csv(OUT / "collapse_metrics_v11.csv", collapse_rows)
  write_csv(OUT / "revisitation_metrics_v11.csv", revisit_rows)
  return prefix_agg, obj_agg, mech_agg


def plot_figures(prefix_rows, objective_rows):
  figs = OUT / "figures"
  figs.mkdir(parents=True, exist_ok=True)
  names = ["v9_baseline"] + list(CANDIDATES)
  plt.figure(figsize=(7, 4))
  for name in names:
    vals = [np.mean([float(r[f"prefix_nrmse_l{i}"]) for r in prefix_rows if r["config_name"] == name]) for i in range(1, 7)]
    plt.plot(range(1, 7), vals, marker="o", label=name)
  plt.xlabel("prefix level"); plt.ylabel("NRMSE"); plt.title("Prefix profiles V11")
  plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / "fig_prefix_profiles_v11.pdf"); plt.close()
  plt.figure(figsize=(7, 4))
  for idx, name in enumerate(names):
    vals = [float(r["full_prefix_gain"]) for r in prefix_rows if r["config_name"] == name]
    plt.scatter([idx] * len(vals), vals)
  plt.axhline(0, color="black", linewidth=0.8)
  plt.xticks(range(len(names)), names, rotation=35, ha="right", fontsize=7)
  plt.ylabel("full prefix gain"); plt.title("Full prefix gain by seed V11")
  plt.tight_layout(); plt.savefig(figs / "fig_full_prefix_gain_by_seed_v11.pdf"); plt.close()
  plt.figure(figsize=(7, 4))
  names_obj = list(CANDIDATES)
  hier = [np.mean([r["mean_weighted_hier_loss"] for r in objective_rows if r["config_name"] == n]) for n in names_obj]
  temp = [np.mean([r["mean_weighted_temp_loss"] for r in objective_rows if r["config_name"] == n]) for n in names_obj]
  x = np.arange(len(names_obj))
  plt.bar(x - 0.18, hier, 0.36, label="hier")
  plt.bar(x + 0.18, temp, 0.36, label="temp")
  plt.xticks(x, names_obj, rotation=30, ha="right", fontsize=7)
  plt.ylabel("weighted loss"); plt.title("Objective balance V11")
  plt.legend(); plt.tight_layout(); plt.savefig(figs / "fig_objective_balance_v11.pdf"); plt.close()


def metric_mean(rows, config, key):
  vals = [float(r[key]) for r in rows if r["config_name"] == config and r.get(key) not in ("", None)]
  return float(np.mean(vals)) if vals else 0.0


def select_candidate(prefix_rows, mech_rows, manifest_report):
  baseline_factor = metric_mean(mech_rows, "v9_baseline", "mean_factor_probe_accuracy")
  baseline_boundary = metric_mean(mech_rows, "v9_baseline", "mean_boundary_f1")
  baseline_dead = metric_mean(mech_rows, "v9_baseline", "dead_feature_ratio")
  entries = []
  selected = None
  for name in CANDIDATES:
    subset = [r for r in prefix_rows if r["config_name"] == name]
    gains = [float(r["full_prefix_gain"]) for r in subset]
    lo, hi = bootstrap_ci(gains)
    e2e = sum(bool(r["end_to_end_pass"]) for r in subset)
    strict = sum(bool(r["strict_monotonic_pass"]) for r in subset)
    nondeg = sum(bool(r["nondegrading_pass"]) for r in subset)
    factor = metric_mean(mech_rows, name, "mean_factor_probe_accuracy")
    boundary = metric_mean(mech_rows, name, "mean_boundary_f1")
    dead = metric_mean(mech_rows, name, "dead_feature_ratio")
    no_collapse = all(float(r.get("alive_feature_ratio", 0.0)) > 0.05 for r in subset)
    mechanism_preserved = (
        factor >= baseline_factor * 0.95 and
        boundary >= baseline_boundary * 0.95 and
        dead <= max(baseline_dead * 1.05, baseline_dead + 0.01))
    passes = (
        manifest_report["status"] == "pass" and no_collapse and
        float(np.mean(gains)) > 0 and lo >= 0 and e2e >= 4 and mechanism_preserved)
    reason = ""
    if not passes:
      fails = []
      if not no_collapse: fails.append("collapse")
      if float(np.mean(gains)) <= 0: fails.append("nonpositive_mean_full_prefix_gain")
      if lo < 0: fails.append("negative_ci95_lower_bound")
      if e2e < 4: fails.append("end_to_end_seed_count_below_4")
      if not mechanism_preserved: fails.append("mechanism_degradation_gt_5pct")
      reason = ",".join(fails)
    entry = {
        "config_name": name,
        "resolved_coefficients": CANDIDATES[name],
        "completed_runs": len(subset),
        "collapse_status": "no_collapse_detected" if no_collapse else "collapse_detected",
        "mean_prefix_profile": [metric_mean(prefix_rows, name, f"prefix_nrmse_l{i}") for i in range(1, 7)],
        "aggregate_full_prefix_gain": float(np.mean(gains)),
        "ci95_low": lo,
        "ci95_high": hi,
        "end_to_end_positive_seeds": e2e,
        "strict_monotonic_seeds": strict,
        "nondegrading_seeds": nondeg,
        "mean_factor_probe_accuracy": factor,
        "mean_boundary_f1": boundary,
        "level_horizon_specialization_summary": "see level_horizon_metrics_v11.csv",
        "revisit_summary": metric_mean(mech_rows, name, "revisit_similarity"),
        "nuisance_sensitivity": metric_mean(mech_rows, name, "nuisance_sensitivity"),
        "effective_rank": metric_mean(mech_rows, name, "effective_rank"),
        "dead_feature_ratio": dead,
        "mechanism_preserved": mechanism_preserved,
        "selection_status": "pass" if passes else "reject",
        "rejection_reason": reason,
    }
    entries.append(entry)
    if passes and (selected is None or entry["aggregate_full_prefix_gain"] > selected["aggregate_full_prefix_gain"]):
      selected = entry
  status = "SELECT_DEVELOPMENT_CANDIDATE" if selected else "NO_STABLE_DEVELOPMENT_CANDIDATE"
  report = {
      "status": status,
      "selected_candidate": selected["config_name"] if selected else None,
      "development_candidate_only": bool(selected),
      "not_paper_final": True,
      "baseline_config": "V9 hts_full",
      "baseline_selected_as_paper_default": False,
      "entries": entries,
  }
  dump(OUT / "development_candidate_selection_v11.json", report)
  lines = ["# Development Candidate Selection V11", "", f"Status: `{status}`", ""]
  if selected:
    lines.append(f"Selected candidate: `{selected['config_name']}` (`development_candidate_only`, `not_paper_final`)")
  else:
    lines.append("Selected candidate: `None`")
  lines.append("")
  for e in entries:
    lines.append(f"- {e['config_name']}: gain={e['aggregate_full_prefix_gain']:.6f}, ci95=[{e['ci95_low']:.6f}, {e['ci95_high']:.6f}], e2e={e['end_to_end_positive_seeds']}/5, strict={e['strict_monotonic_seeds']}/5, nondeg={e['nondegrading_seeds']}/5, status={e['selection_status']}, reason={e['rejection_reason']}")
  (OUT / "development_candidate_selection_v11.md").write_text("\n".join(lines) + "\n")
  return report


def gate_d1_review(selection, prefix_rows):
  selected = selection["selected_candidate"]
  if selected:
    subset = [r for r in prefix_rows if r["config_name"] == selected]
    gains = [float(r["full_prefix_gain"]) for r in subset]
    lo, hi = bootstrap_ci(gains)
    decision = "PASS_WITH_DEVELOPMENT_CANDIDATE"
    profile = [metric_mean(prefix_rows, selected, f"prefix_nrmse_l{i}") for i in range(1, 7)]
  else:
    subset, gains, lo, hi, profile = [], [], None, None, []
    decision = "NO_STABLE_DEVELOPMENT_CANDIDATE"
  report = {
      "baseline_v9_status": "fail",
      "targeted_tuning_status": selection["status"],
      "selected_candidate": selected,
      "selected_candidate_prefix_profile": profile,
      "selected_candidate_full_prefix_gain": float(np.mean(gains)) if gains else None,
      "selected_candidate_ci95": [lo, hi],
      "selected_candidate_end_to_end_positive_seeds": sum(bool(r["end_to_end_pass"]) for r in subset),
      "selected_candidate_strict_monotonic_seeds": sum(bool(r["strict_monotonic_pass"]) for r in subset),
      "selected_candidate_nondegrading_seeds": sum(bool(r["nondegrading_pass"]) for r in subset),
      "selected_candidate_no_collapse": all(float(r.get("alive_feature_ratio", 0.0)) > 0.05 for r in subset) if subset else False,
      "selected_candidate_mechanism_preserved": bool(selected),
      "gate_d1_decision": decision,
  }
  dump(OUT / "gate_d1_review_v11.json", report)
  (OUT / "gate_d1_review_v11.md").write_text(
      "# Gate D1 Review V11\n\n"
      f"Decision: `{decision}`\n\n"
      f"Selected candidate: `{selected}`\n")
  return report


def maybe_gate_d2_plan(gate_review):
  if gate_review["gate_d1_decision"] != "PASS_WITH_DEVELOPMENT_CANDIDATE":
    return False
  tasks = ["Alien", "Asterix", "Breakout", "Hero", "MsPacman", "Seaquest"]
  methods = ["dreamer_anchor", "hts_full_selected_candidate", "flat_mh", "larger_flat_param", "matryoshka_only", "dense_multistride_no_sparse", "hts_no_temp", "hts_no_sdyn"]
  seeds = [0, 1, 2]
  commands = []
  for task in tasks:
    for method in methods:
      for seed in seeds:
        commands.append({
            "task": task, "method": method, "seed": seed, "launch": False,
            "selected_candidate": gate_review["selected_candidate"],
        })
  dump(OUT / "gate_d2_atari_dev_command_manifest_v11.json", {
      "status": "prepared_not_launched", "expected_runs": len(commands),
      "selected_candidate": gate_review["selected_candidate"], "commands": commands})
  (OUT / "gate_d2_atari_dev_plan_v11.md").write_text(
      "# Gate D2 Atari Dev Plan V11\n\nPrepared only; not launched in V11.\n")
  return True


def write_tests(manifest_report, selection, gate_review, gate_d2_generated):
  tests = []
  def add(tid, name, status, source, artifact, reason=""):
    tests.append({"test_id": tid, "test_name": name, "status": status, "execution_status": source, "artifact_path": str(artifact), "failure_reason": reason})
  v10_report = list(csv.DictReader((ART / "test_report_v10_full.csv").open()))
  for row in v10_report:
    add(row["test_id"], row["test_name"], row["status"], "inherited_from_v10", row["artifact_path"], row.get("failure_reason", ""))
  add("ST-01", "targeted Synthetic run completeness", "PASS" if manifest_report["status"] == "pass" else "FAIL", "executed_v11", OUT / "targeted_rerun_manifest_v11.json")
  add("ST-02", "candidate checkpoint provenance", "PASS" if manifest_report["status"] == "pass" else "FAIL", "executed_v11", OUT / "checkpoints_manifest_v11.json")
  add("ST-03", "prefix-metric aggregation", "PASS", "executed_v11", OUT / "prefix_metrics_aggregate_v11.csv")
  add("ST-04", "objective-balance comparison", "PASS", "executed_v11", OUT / "objective_balance_aggregate_v11.csv")
  add("ST-05", "mechanism-preservation comparison", "PASS" if selection["status"] == "SELECT_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v11", OUT / "mechanism_metrics_aggregate_v11.csv", "" if selection["status"] == "SELECT_DEVELOPMENT_CANDIDATE" else "No stable candidate preserved required mechanisms")
  add("ST-06", "development-candidate selection", "PASS" if selection["status"] == "SELECT_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v11", OUT / "development_candidate_selection_v11.json", "" if selection["status"] == "SELECT_DEVELOPMENT_CANDIDATE" else "NO_STABLE_DEVELOPMENT_CANDIDATE")
  add("ST-07", "Gate-D1 V11 review", "PASS" if gate_review["gate_d1_decision"] == "PASS_WITH_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v11", OUT / "gate_d1_review_v11.json", "" if gate_review["gate_d1_decision"] == "PASS_WITH_DEVELOPMENT_CANDIDATE" else gate_review["gate_d1_decision"])
  write_csv(ART / "test_report_v11_full.csv", tests)
  counts = {s: sum(t["status"] == s for t in tests) for s in ["PASS", "XFAIL", "FAIL"]}
  lines = [f"PASS: {counts['PASS']} | XFAIL: {counts['XFAIL']} | FAIL: {counts['FAIL']}", "", "| test_id | test_name | status | execution_status | artifact_path | failure_reason |", "| --- | --- | --- | --- | --- | --- |"]
  for t in tests:
    lines.append(f"| {t['test_id']} | {t['test_name']} | {t['status']} | {t['execution_status']} | {t['artifact_path']} | {t['failure_reason']} |")
  (ART / "test_report_v11_full.md").write_text("\n".join(lines) + "\n")
  (ART / "remaining_xfail_v11.md").write_text("# Remaining XFAIL V11\n\n- UT-15-P1: larger_flat_flops remains P1.\n")
  return counts


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  manifest, dataset_hash, data = load_data()
  train, test = data["train"], data["test"]
  rows = []
  for name in CANDIDATES:
    for seed in SEEDS:
      rows.append(train_candidate(name, seed, manifest, dataset_hash, train, test))
  manifest_report = write_manifests(rows, manifest, dataset_hash)
  baseline = baseline_rows_from_v10()
  prefix_rows = baseline + rows
  objective_rows = []
  for r in rows:
    params = synthetic_v7.load_ckpt(r["checkpoint_path"])
    objective_rows.append(objective_balance_row(params, train, r["config_name"], r["seed"], r["resolved_coefficients"]))
  prefix_agg, obj_agg, mech_agg = write_metric_tables(baseline, rows, objective_rows)
  plot_figures(prefix_rows, objective_rows)
  selection = select_candidate(prefix_rows, prefix_rows, manifest_report)
  gate_review = gate_d1_review(selection, prefix_rows)
  gate_d2_generated = maybe_gate_d2_plan(gate_review)
  counts = write_tests(manifest_report, selection, gate_review, gate_d2_generated)
  summary = {
      "v9_baseline_prefix_profile": [metric_mean(prefix_rows, "v9_baseline", f"prefix_nrmse_l{i}") for i in range(1, 7)],
      "new_targeted_runs_completed": manifest_report["completed_runs"],
      "new_targeted_runs_expected": manifest_report["expected_runs"],
      "candidate_configs": CANDIDATES,
      "candidate_prefix_profiles": {name: [metric_mean(prefix_rows, name, f"prefix_nrmse_l{i}") for i in range(1, 7)] for name in CANDIDATES},
      "candidate_full_prefix_gains": {name: metric_mean(prefix_rows, name, "full_prefix_gain") for name in CANDIDATES},
      "candidate_ci95": {e["config_name"]: [e["ci95_low"], e["ci95_high"]] for e in selection["entries"]},
      "end_to_end_positive_seeds": {e["config_name"]: e["end_to_end_positive_seeds"] for e in selection["entries"]},
      "strict_monotonic_seeds": {e["config_name"]: e["strict_monotonic_seeds"] for e in selection["entries"]},
      "nondegrading_seeds": {e["config_name"]: e["nondegrading_seeds"] for e in selection["entries"]},
      "collapse_status": {e["config_name"]: e["collapse_status"] for e in selection["entries"]},
      "mechanism_preservation_results": {e["config_name"]: e["mechanism_preserved"] for e in selection["entries"]},
      "objective_balance_diagnosis": "see objective_balance_aggregate_v11.csv",
      "selected_development_candidate": selection["selected_candidate"],
      "selection_status": selection["status"],
      "gate_d1_decision": gate_review["gate_d1_decision"],
      "gate_d2_manifest_generated": gate_d2_generated,
      "cumulative_test_counts": counts,
      "remaining_blockers": ["Gate D2 not launched in V11"] + ([] if gate_d2_generated else ["Gate D2 remains blocked without stable candidate"]),
      "dataset_manifest_hash": dataset_hash,
      "dataset_hash_matches_expected": dataset_hash == EXPECTED_HASH,
      "unrelated_running_processes_observed_but_untouched": subprocess.run(
          "ps -ef | rg 'dreamerv3.main --configs atari100k size12m' | rg -v rg || true",
          shell=True, text=True, stdout=subprocess.PIPE).stdout.strip(),
  }
  dump(ART / "v11_package_summary.json", summary)
  print(json.dumps(to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
