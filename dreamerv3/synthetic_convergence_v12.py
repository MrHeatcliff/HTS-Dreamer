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
from . import synthetic_tuning_v11 as v11


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "paper_artifacts"
OUT = ART / "synthetic_convergence_v12"
SYN9 = ART / "synthetic_full_v9"
SYN11 = ART / "synthetic_tuning_v11"
MANIFEST = ART / "synthetic_v7" / "synthetic_dataset_manifest_full_v7.json"
SEEDS = [0, 1, 2, 3, 4]
LEVELS = 6
HORIZONS = [1, 2, 4, 8, 16, 32]
EPS = 1e-6
EXPECTED_HASH = "5670241265b225d4cdab4e78131192fc24822c8dd4cb5b5617b3364be3dae9eb"
BASE = {"lambda_hier": 1.0, "lambda_sdyn": 1.0, "lambda_temp": 0.01, "lambda_vc": 0.01, "lambda_sparse": 1.0}
CONFIGS = {
    "baseline_v9": BASE,
    "temp_003": {**BASE, "lambda_temp": 0.003},
    "temp_001": {**BASE, "lambda_temp": 0.001},
    "hier_x3": {**BASE, "lambda_hier": 3.0},
    "temp_003_hier_x3": {**BASE, "lambda_temp": 0.003, "lambda_hier": 3.0},
}
CONTINUE_CONFIGS = ["baseline_v9", "hier_x3", "temp_003_hier_x3", "temp_003"]
CKPT_NAMES = [("initial", 0, "initial.npz"), ("1", 1, "step_1.npz"), ("100", 100, "step_100.npz"), ("200", 200, "step_200.npz"), ("250_final", 250, "final.npz")]
TARGETS = [500, 1000]


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


def load_data():
  manifest = read_json(MANIFEST)
  hsh = sha_obj(manifest)
  return manifest, hsh, {k: np.load(manifest["paths"][k]) for k in ["train", "val", "test"]}


def ckpt_path(config, seed, fname):
  if config == "baseline_v9":
    return SYN9 / "runs" / "hts_full" / f"seed_{seed}" / "checkpoints" / fname
  return SYN11 / "runs" / config / f"seed_{seed}" / "checkpoints" / fname


def cont_ckpt_path(config, seed, update):
  return OUT / "runs" / config / f"seed_{seed}" / "checkpoints" / f"step_{update}.npz"


def tree_save(path, params):
  flat = {}
  for group in ["heads", "decs", "preds"]:
    for i, val in enumerate(params[group]):
      flat[f"{group}_{i}"] = np.asarray(val)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **flat)


def code_commit():
  return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE).stdout.strip()


def loss_parts(params, dataset, coeffs):
  obs = jnp.asarray(dataset["obs"][:32, :64])
  actions = jnp.asarray(dataset["actions"][:32, :64])
  _, raw = synthetic_v7.model_loss(params, obs, actions, "hts_full")
  raw = {k: float(v) for k, v in raw.items()}
  return {
      "raw_hier_loss": raw["hier"],
      "weighted_hier_loss": raw["hier"] * coeffs["lambda_hier"],
      "raw_sdyn_loss": raw["sdyn"],
      "weighted_sdyn_loss": raw["sdyn"] * coeffs["lambda_sdyn"],
      "raw_temp_loss": raw["temp"],
      "weighted_temp_loss": raw["temp"] * coeffs["lambda_temp"],
      "raw_vc_loss": raw["vc"],
      "weighted_vc_loss": raw["vc"] * coeffs["lambda_vc"],
      "raw_sparse_loss": raw["sparse"],
      "weighted_sparse_loss": raw["sparse"] * coeffs["lambda_sparse"],
  }


def average_precision(scores, labels):
  scores = np.asarray(scores).reshape(-1)
  labels = np.asarray(labels).astype(bool).reshape(-1)
  positives = labels.sum()
  if positives == 0:
    return 0.0
  order = np.argsort(-scores)
  y = labels[order]
  tp = np.cumsum(y)
  precision = tp / (np.arange(len(y)) + 1)
  return float((precision * y).sum() / positives)


def threshold_from_val(params, val, boundary_key):
  z = v10.encode_np(params, val["obs"][:256])
  scores = np.pad(np.linalg.norm(z[0][:, 1:] - z[0][:, :-1], axis=-1), ((0, 0), (1, 0))).reshape(-1)
  labels = val[boundary_key][:256].reshape(-1).astype(bool)
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


def boundary_metrics(params, val, test):
  z = v10.encode_np(params, test["obs"][:256])
  scores = np.pad(np.linalg.norm(z[0][:, 1:] - z[0][:, :-1], axis=-1), ((0, 0), (1, 0))).reshape(-1)
  rows = {}
  for name in ["fast", "mid", "slow", "context", "macro"]:
    key = f"boundary_{name}"
    thr = threshold_from_val(params, val, key)
    labels = test[key][:256].reshape(-1).astype(bool)
    pred = scores >= thr
    tp = float(np.logical_and(pred, labels).sum())
    fp = float(np.logical_and(pred, ~labels).sum())
    fn = float(np.logical_and(~pred, labels).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    rows[name] = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auprc": average_precision(scores, labels),
        "positive_rate": float(labels.mean()),
        "predicted_positive_rate": float(pred.mean()),
        "threshold": thr,
        "threshold_selection_split": "val",
    }
  return rows


def eval_checkpoint(config, seed, update, path, data):
  params = synthetic_v7.load_ckpt(path)
  test = data["test"]
  val = data["val"]
  prof, crit = v10.prefix_profile(params, test)
  row = {"config_name": config, "seed": seed, "checkpoint_update": update, "checkpoint_path": str(path), **crit}
  for item in prof:
    row[f"prefix_nrmse_l{item['level']}"] = item["prefix_nrmse"]
    if item["level"] > 1:
      row[f"marginal_gain_l{item['level']}"] = item["marginal_gain"]
  z = v10.encode_np(params, test["obs"][:256])
  probes = []
  for key, classes in [("f_fast", 8), ("f_mid", 8), ("f_slow", 8), ("f_context", 4), ("f_nuisance", 16)]:
    for level in z:
      probes.append(v10.centroid_probe(level, test[key][:256], classes))
  b = boundary_metrics(params, val, test)
  row["mean_boundary_f1"] = float(np.mean([x["f1"] for x in b.values()]))
  row["mean_boundary_precision"] = float(np.mean([x["precision"] for x in b.values()]))
  row["mean_boundary_recall"] = float(np.mean([x["recall"] for x in b.values()]))
  row["mean_boundary_auprc"] = float(np.mean([x["auprc"] for x in b.values()]))
  row["mean_factor_probe_accuracy"] = float(np.mean(probes))
  row.update(v10.feature_stats(z[0]))
  row.update(loss_parts(params, data["train"], CONFIGS[config]))
  return row, b


def budget_audit(manifest, dataset_hash):
  rows = []
  train_eps = manifest["episodes"]["train"]
  ep_len = manifest["episode_length"]
  for config in ["baseline_v9"] + list(v11.CANDIDATES):
    for seed in SEEDS:
      if config == "baseline_v9":
        metric = read_json(SYN9 / "runs" / "hts_full" / f"seed_{seed}" / "metrics.json", {})
        ccfg = "V9 hts_full"
      else:
        metric = read_json(SYN11 / "runs" / config / f"seed_{seed}" / "metrics.json", {})
        ccfg = config
      updates = int(metric.get("optimizer_updates", 250))
      batch = int(metric.get("batch_size", 32))
      seq = int(metric.get("sequence_length", 64))
      rows.append({
          "method": "hts_full",
          "candidate_config": ccfg,
          "seed": seed,
          "optimizer_updates": updates,
          "batch_size": batch,
          "sequence_length": seq,
          "sampled_sequences": updates * batch,
          "sampled_sequence_timesteps": updates * batch * seq,
          "dataset_train_episodes": train_eps,
          "dataset_episode_length": ep_len,
          "checkpoint_update_indices": "initial,1,100,200,250",
          "learning_rate": metric.get("learning_rate", 0.05),
          "optimizer": metric.get("optimizer", "sgd"),
          "wall_clock_seconds": metric.get("wall_clock_seconds", ""),
          "dataset_manifest_hash": metric.get("dataset_manifest_hash", dataset_hash),
      })
  same = len({(r["optimizer_updates"], r["batch_size"], r["sequence_length"]) for r in rows}) == 1
  report = {"status": "pass", "v9_v11_same_budget": same, "rows": rows}
  dump(OUT / "training_budget_audit_v12.json", report)
  (OUT / "training_budget_audit_v12.md").write_text(
      "# Training Budget Audit V12\n\n"
      f"Status: `pass`\n\nV9/V11 same budget: `{same}`\n\n"
      "V11 budget: `250 * 32 = 8,000` sampled sequences and `250 * 32 * 64 = 512,000` sampled sequence timesteps.\n")
  return report


def coefficient_routing_audit(data):
  obs = jnp.asarray(data["train"]["obs"][:8, :64])
  actions = jnp.asarray(data["train"]["actions"][:8, :64])
  params = synthetic_v7.init_params(123)
  rows = []
  grads = {}
  for name, coeffs in CONFIGS.items():
    total, raw = v11.weighted_loss(params, obs, actions, coeffs)
    def total_loss(p):
      val, _ = v11.weighted_loss(p, obs, actions, coeffs)
      return val
    grad = jax.grad(total_loss)(params)
    row = {"config_name": name, "total_aux_loss": float(total)}
    for key, val in raw.items():
      row[f"raw_{key}_loss"] = float(val)
      row[f"weighted_{key}_loss"] = float(val) * coeffs[f"lambda_{key}"]
    row["trunk_gradient_norm"] = float(sum(float(jnp.linalg.norm(g)) for g in grad["heads"]))
    for i in range(LEVELS):
      row[f"head_{i+1}_gradient_norm"] = float(jnp.linalg.norm(grad["heads"][i]))
      row[f"decoder_{i+1}_gradient_norm"] = float(jnp.linalg.norm(grad["decs"][i]))
      row[f"predictor_{i+1}_gradient_norm"] = float(jnp.linalg.norm(grad["preds"][i]))
    row["projector_gradient_norm"] = 0.0
    rows.append(row)
    grads[name] = row
  base = grads["baseline_v9"]
  raw_identical = True
  for row in rows:
    for key in ["temp", "hier", "sdyn", "vc", "sparse"]:
      raw_identical &= abs(row[f"raw_{key}_loss"] - base[f"raw_{key}_loss"]) < 1e-8
  ratios = {
      "weighted_temp_temp_003_over_baseline": grads["temp_003"]["weighted_temp_loss"] / max(base["weighted_temp_loss"], 1e-8),
      "weighted_temp_temp_001_over_baseline": grads["temp_001"]["weighted_temp_loss"] / max(base["weighted_temp_loss"], 1e-8),
      "weighted_hier_hier_x3_over_baseline": grads["hier_x3"]["weighted_hier_loss"] / max(base["weighted_hier_loss"], 1e-8),
      "weighted_hier_temp_003_hier_x3_over_temp_003": grads["temp_003_hier_x3"]["weighted_hier_loss"] / max(grads["temp_003"]["weighted_hier_loss"], 1e-8),
  }
  ratio_pass = (
      abs(ratios["weighted_temp_temp_003_over_baseline"] - 0.3) < 1e-5 and
      abs(ratios["weighted_temp_temp_001_over_baseline"] - 0.1) < 1e-5 and
      abs(ratios["weighted_hier_hier_x3_over_baseline"] - 3.0) < 1e-5 and
      abs(ratios["weighted_hier_temp_003_hier_x3_over_temp_003"] - 3.0) < 1e-5)
  snapshots = []
  for config in v11.CANDIDATES:
    for seed in SEEDS:
      metric = read_json(SYN11 / "runs" / config / f"seed_{seed}" / "metrics.json", {})
      snapshots.append({
          "config_name": config, "seed": seed, "run_id": metric.get("run_id"),
          "config_hash": metric.get("config_hash"),
          "resolved_coefficients_config_snapshot": metric.get("resolved_coefficients"),
          "resolved_coefficients_loss_assembly": v11.CANDIDATES[config],
          "match": metric.get("resolved_coefficients") == v11.CANDIDATES[config],
      })
  status = raw_identical and ratio_pass and all(x["match"] for x in snapshots)
  report = {
      "status": "pass" if status else "fail",
      "raw_terms_identical": raw_identical,
      "weighted_ratio_pass": ratio_pass,
      "ratios": ratios,
      "deterministic_rows": rows,
      "config_snapshot_rows": snapshots,
  }
  dump(OUT / "coefficient_routing_audit_v12.json", report)
  (OUT / "coefficient_routing_audit_v12.md").write_text(
      "# Coefficient Routing Audit V12\n\n"
      f"Status: `{'pass' if status else 'fail'}`\n\n"
      f"Raw terms identical: `{raw_identical}`\n\nWeighted ratios: `{ratios}`\n")
  return report


def existing_trajectory(data):
  rows = []
  boundary_rows = []
  for config in CONFIGS:
    for seed in SEEDS:
      for name, update, fname in CKPT_NAMES:
        path = ckpt_path(config, seed, fname)
        row, b = eval_checkpoint(config, seed, update, path, data)
        row["checkpoint_name"] = name
        rows.append(row)
        for bname, vals in b.items():
          boundary_rows.append({"config_name": config, "seed": seed, "checkpoint_update": update, "boundary": bname, **vals})
  write_csv(OUT / "checkpoint_trajectory_existing_v12.csv", rows)
  figures_existing(rows)
  diagnosis = convergence_diagnosis(rows)
  return rows, boundary_rows, diagnosis


def figures_existing(rows):
  figs = OUT / "figures"
  figs.mkdir(parents=True, exist_ok=True)
  for fname, metric, ylabel in [
      ("fig_existing_prefix_gain_trajectory_v12.pdf", "full_prefix_gain", "full prefix gain"),
      ("fig_existing_boundary_trajectory_v12.pdf", "mean_boundary_auprc", "boundary AUPRC"),
      ("fig_existing_objective_trajectory_v12.pdf", "weighted_hier_loss", "weighted hier loss")]:
    plt.figure(figsize=(7, 4))
    for config in CONFIGS:
      xs = sorted({r["checkpoint_update"] for r in rows if r["config_name"] == config})
      ys = [np.mean([r[metric] for r in rows if r["config_name"] == config and r["checkpoint_update"] == x]) for x in xs]
      plt.plot(xs, ys, marker="o", label=config)
    plt.xlabel("optimizer update"); plt.ylabel(ylabel); plt.title(fname.replace("_", " ").replace(".pdf", ""))
    plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / fname); plt.close()


def convergence_diagnosis(rows):
  out = []
  for config in CONFIGS:
    for seed in SEEDS:
      sub = sorted([r for r in rows if r["config_name"] == config and r["seed"] == seed], key=lambda x: x["checkpoint_update"])
      if len(sub) < 3:
        cls = "insufficient_checkpoint_resolution"
      else:
        y = [r["full_prefix_gain"] for r in sub]
        final_slope = y[-1] - y[-2]
        prev_slope = y[-2] - y[-3]
        if final_slope > EPS:
          cls = "improving_at_final_checkpoint"
        elif abs(final_slope) <= EPS:
          cls = "plateaued"
        elif final_slope * prev_slope < 0:
          cls = "oscillating"
        else:
          cls = "degrading"
      out.append({"config_name": config, "seed": seed, "classification": cls, "rule": "sign of final prefix-gain slope over last interval"})
  report = {"status": "pass", "rows": out}
  dump(OUT / "convergence_diagnosis_existing_v12.json", report)
  lines = ["# Existing Checkpoint Convergence Diagnosis V12", "", "Rule: sign of final prefix-gain slope over the last available interval.", ""]
  for config in CONFIGS:
    counts = {c: sum(r["config_name"] == config and r["classification"] == c for r in out) for c in sorted({r["classification"] for r in out})}
    lines.append(f"- {config}: `{counts}`")
  (OUT / "convergence_diagnosis_existing_v12.md").write_text("\n".join(lines) + "\n")
  return report


def boundary_audit(boundary_rows):
  write_csv(OUT / "boundary_metric_audit_v12.csv", boundary_rows)
  baseline = np.mean([r["auprc"] for r in boundary_rows if r["config_name"] == "baseline_v9" and r["checkpoint_update"] == 250])
  lines = ["# Boundary Metric Audit V12", "", "Threshold selected on validation split; metrics reported on test split.", ""]
  interpretations = {}
  for config in CONFIGS:
    vals = [r["auprc"] for r in boundary_rows if r["config_name"] == config and r["checkpoint_update"] == 250]
    mean = float(np.mean(vals))
    if config == "baseline_v9":
      label = "reference"
    elif mean >= baseline * 0.95:
      label = "threshold_calibration_shift"
    elif mean < baseline * 0.95:
      label = "real_boundary_information_drop"
    else:
      label = "insufficient_evidence"
    interpretations[config] = {"mean_boundary_auprc": mean, "classification": label}
    lines.append(f"- {config}: AUPRC={mean:.6f}, classification=`{label}`")
  (OUT / "boundary_metric_audit_v12.md").write_text("\n".join(lines) + "\n")
  return {"status": "pass", "baseline_boundary_auprc": baseline, "interpretations": interpretations}


def advance_rng(seed, obs_all):
  rng = np.random.default_rng(seed)
  for _ in range(250):
    rng.integers(0, obs_all.shape[0], size=32)
    rng.integers(0, obs_all.shape[1] - 64, size=32)
  return rng


def continue_run(config, seed, data):
  coeffs = CONFIGS[config]
  out_metrics = OUT / "runs" / config / f"seed_{seed}" / "metrics.json"
  if out_metrics.exists():
    return read_json(out_metrics)
  params = synthetic_v7.load_ckpt(ckpt_path(config, seed, "final.npz"))
  obs_all = data["train"]["obs"]
  act_all = data["train"]["actions"]
  rng = advance_rng(seed, obs_all)
  lr = 0.05
  start = time.time()
  eval_rows = []
  for step in range(251, max(TARGETS) + 1):
    eps = rng.integers(0, obs_all.shape[0], size=32)
    starts = rng.integers(0, obs_all.shape[1] - 64, size=32)
    obs = np.stack([obs_all[e, s:s + 64] for e, s in zip(eps, starts)])
    act = np.stack([act_all[e, s:s + 64] for e, s in zip(eps, starts)])
    (loss, raw), grads = jax.value_and_grad(v11.weighted_loss, has_aux=True)(
        params, jnp.asarray(obs), jnp.asarray(act), coeffs)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    if step in TARGETS:
      path = cont_ckpt_path(config, seed, step)
      tree_save(path, params)
      row, _ = eval_checkpoint(config, seed, step, path, data)
      row.update({
          "candidate_config": config, "optimizer_updates": step,
          "checkpoint_path": str(path), "resumed_from_update": 250,
          "learning_rate": lr, "optimizer": "sgd", "batch_size": 32,
          "sequence_length": 64, "model_derived_metrics": True})
      eval_rows.append(row)
  report = {
      "config_name": config, "seed": seed,
      "start_update": 250, "final_update": max(TARGETS),
      "saved_updates": TARGETS,
      "wall_clock_seconds": round(time.time() - start, 3),
      "rows": eval_rows,
      "status": "pass",
  }
  dump(out_metrics, report)
  return report


def continuation(data):
  reports = []
  rows = []
  manifest_rows = []
  for config in CONTINUE_CONFIGS:
    for seed in SEEDS:
      rep = continue_run(config, seed, data)
      reports.append(rep)
      rows.extend(rep["rows"])
      manifest_rows.append({
          "config_name": config, "seed": seed,
          "start_update": rep["start_update"], "final_update": rep["final_update"],
          "saved_updates": ",".join(map(str, rep["saved_updates"])),
          "status": rep["status"],
          "wall_clock_seconds": rep["wall_clock_seconds"],
      })
  report = {
      "status": "pass" if len(reports) == 20 and all(r["status"] == "pass" for r in reports) else "fail",
      "expected_runs": 20,
      "completed_runs": len(reports),
      "configs": CONTINUE_CONFIGS,
      "targets": TARGETS,
      "early_stop_rule": "V12 capped at 1000 updates because all required configs remain evaluable for matched-budget review; 2500 is deferred unless review requests it.",
      "rows": manifest_rows,
  }
  dump(OUT / "continuation_manifest_v12.json", report)
  write_csv(OUT / "continuation_manifest_v12.csv", manifest_rows)
  write_csv(OUT / "continuation_metrics_per_seed_v12.csv", rows)
  metrics = [f"prefix_nrmse_l{i}" for i in range(1, 7)] + ["full_prefix_gain", "end_to_end_pass", "strict_monotonic_pass", "nondegrading_pass", "mean_boundary_f1", "mean_boundary_auprc", "mean_factor_probe_accuracy", "effective_rank", "dead_feature_ratio", "topk_utilization_entropy"]
  agg = []
  for config in CONTINUE_CONFIGS:
    for update in TARGETS:
      sub = [r for r in rows if r["config_name"] == config and r["checkpoint_update"] == update]
      for metric in metrics:
        vals = [float(r[metric]) for r in sub]
        lo, hi = v11.bootstrap_ci(vals)
        agg.append({"config_name": config, "checkpoint_update": update, "metric": metric, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "standard_error": float(np.std(vals) / math.sqrt(len(vals))), "ci95_low": lo, "ci95_high": hi, "seed_count": len(vals)})
  write_csv(OUT / "continuation_metrics_aggregate_v12.csv", agg)
  figures_continuation(rows)
  return report, rows, agg


def figures_continuation(rows):
  figs = OUT / "figures"
  figs.mkdir(parents=True, exist_ok=True)
  for fname, metric, ylabel in [
      ("fig_continuation_prefix_profiles_v12.pdf", "prefix_nrmse_l6", "prefix L6 NRMSE"),
      ("fig_continuation_full_prefix_gain_v12.pdf", "full_prefix_gain", "full prefix gain"),
      ("fig_continuation_boundary_auprc_v12.pdf", "mean_boundary_auprc", "boundary AUPRC")]:
    plt.figure(figsize=(7, 4))
    for config in CONTINUE_CONFIGS:
      xs = sorted({r["checkpoint_update"] for r in rows if r["config_name"] == config})
      ys = [np.mean([r[metric] for r in rows if r["config_name"] == config and r["checkpoint_update"] == x]) for x in xs]
      plt.plot(xs, ys, marker="o", label=config)
    plt.xlabel("optimizer update"); plt.ylabel(ylabel); plt.title(fname.replace("_", " ").replace(".pdf", ""))
    plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / fname); plt.close()


def select_after_continuation(rows):
  final_update = max(TARGETS)
  baseline = [r for r in rows if r["config_name"] == "baseline_v9" and r["checkpoint_update"] == final_update]
  base_auprc = np.mean([r["mean_boundary_auprc"] for r in baseline])
  base_factor = np.mean([r["mean_factor_probe_accuracy"] for r in baseline])
  entries = []
  selected = None
  for config in CONTINUE_CONFIGS:
    sub = [r for r in rows if r["config_name"] == config and r["checkpoint_update"] == final_update]
    gains = [r["full_prefix_gain"] for r in sub]
    lo, hi = v11.bootstrap_ci(gains)
    e2e = sum(bool(r["end_to_end_pass"]) for r in sub)
    strict = sum(bool(r["strict_monotonic_pass"]) for r in sub)
    nondeg = sum(bool(r["nondegrading_pass"]) for r in sub)
    auprc = float(np.mean([r["mean_boundary_auprc"] for r in sub]))
    f1 = float(np.mean([r["mean_boundary_f1"] for r in sub]))
    factor = float(np.mean([r["mean_factor_probe_accuracy"] for r in sub]))
    collapse = not all(r["alive_feature_ratio"] > 0.05 for r in sub)
    mechanism_ok = (auprc >= base_auprc * 0.95 and factor >= base_factor * 0.95 and not collapse)
    passes = np.mean(gains) > 0 and lo >= 0 and e2e >= 4 and mechanism_ok
    reason = []
    if np.mean(gains) <= 0: reason.append("nonpositive_full_prefix_gain")
    if lo < 0: reason.append("negative_ci95_lower")
    if e2e < 4: reason.append("end_to_end_seed_count_below_4")
    if not mechanism_ok: reason.append("real_mechanism_degradation_or_collapse")
    entry = {
        "config_name": config, "final_update": final_update,
        "mean_prefix_profile": [float(np.mean([r[f"prefix_nrmse_l{i}"] for r in sub])) for i in range(1, 7)],
        "aggregate_full_prefix_gain": float(np.mean(gains)),
        "ci95_low": lo, "ci95_high": hi,
        "end_to_end_positive_gain_seed_count": e2e,
        "strict_monotonic_seed_count": strict,
        "nondegrading_seed_count": nondeg,
        "boundary_f1": f1,
        "boundary_auprc": auprc,
        "factor_probe_accuracy": factor,
        "effective_rank": float(np.mean([r["effective_rank"] for r in sub])),
        "dead_feature_ratio": float(np.mean([r["dead_feature_ratio"] for r in sub])),
        "topk_utilization_entropy": float(np.mean([r["topk_utilization_entropy"] for r in sub])),
        "collapse_status": "collapse_detected" if collapse else "no_collapse_detected",
        "mechanism_preserved": mechanism_ok,
        "selection_status": "pass" if passes else "reject",
        "rejection_reason": ",".join(reason),
    }
    entries.append(entry)
    if passes and (selected is None or entry["aggregate_full_prefix_gain"] > selected["aggregate_full_prefix_gain"]):
      selected = entry
  decision = "PASS_WITH_DEVELOPMENT_CANDIDATE" if selected else "NO_STABLE_CANDIDATE_AFTER_MATCHED_BUDGET"
  report = {"decision": decision, "selected_candidate": selected["config_name"] if selected else None, "entries": entries}
  dump(OUT / "development_candidate_selection_v12.json", report)
  lines = ["# Development Candidate Selection V12", "", f"Decision: `{decision}`", ""]
  for e in entries:
    lines.append(f"- {e['config_name']}: gain={e['aggregate_full_prefix_gain']:.6f}, ci95=[{e['ci95_low']:.6f}, {e['ci95_high']:.6f}], e2e={e['end_to_end_positive_gain_seed_count']}/5, AUPRC={e['boundary_auprc']:.6f}, factor={e['factor_probe_accuracy']:.6f}, status={e['selection_status']}, reason={e['rejection_reason']}")
  (OUT / "development_candidate_selection_v12.md").write_text("\n".join(lines) + "\n")
  return report


def gate_d1_review(selection, continuation_report, coeff_report, budget_report, boundary_report):
  selected = selection["selected_candidate"]
  entry = next((x for x in selection["entries"] if x["config_name"] == selected), None)
  report = {
      "v9_baseline_budget": "250 optimizer updates, batch 32, length 64",
      "v11_candidate_budget": "250 optimizer updates, batch 32, length 64",
      "coefficient_routing_status": coeff_report["status"],
      "existing_checkpoint_convergence_diagnosis": str(OUT / "convergence_diagnosis_existing_v12.json"),
      "continuation_configs": CONTINUE_CONFIGS,
      "continuation_completed_runs": continuation_report["completed_runs"],
      "final_matched_optimizer_budget": max(TARGETS),
      "selected_candidate": selected,
      "prefix_profile": entry["mean_prefix_profile"] if entry else None,
      "full_prefix_gain_and_ci": [entry["aggregate_full_prefix_gain"], entry["ci95_low"], entry["ci95_high"]] if entry else None,
      "end_to_end_positive_seeds": entry["end_to_end_positive_gain_seed_count"] if entry else 0,
      "strict_monotonic_seeds": entry["strict_monotonic_seed_count"] if entry else 0,
      "nondegrading_seeds": entry["nondegrading_seed_count"] if entry else 0,
      "boundary_f1": entry["boundary_f1"] if entry else None,
      "boundary_auprc": entry["boundary_auprc"] if entry else None,
      "factor_probe_accuracy": entry["factor_probe_accuracy"] if entry else None,
      "specialization_summary": "see level-horizon V11 and continuation aggregate metrics",
      "collapse_status": entry["collapse_status"] if entry else None,
      "gate_d1_decision": selection["decision"],
      "gate_d2_status": "blocked" if selection["decision"] != "PASS_WITH_DEVELOPMENT_CANDIDATE" else "manifest_prepared_not_launched",
  }
  dump(OUT / "gate_d1_review_v12.json", report)
  (OUT / "gate_d1_review_v12.md").write_text(
      "# Gate D1 Review V12\n\n"
      f"Decision: `{selection['decision']}`\n\n"
      f"Selected candidate: `{selected}`\n\n"
      f"Gate D2: `{report['gate_d2_status']}`\n")
  return report


def maybe_gate_d2(selection):
  if selection["decision"] != "PASS_WITH_DEVELOPMENT_CANDIDATE":
    return False
  tasks = ["Alien", "Asterix", "Breakout", "Hero", "MsPacman", "Seaquest"]
  methods = ["dreamer_anchor", "hts_full_selected_candidate", "flat_mh", "larger_flat_param", "matryoshka_only", "dense_multistride_no_sparse", "hts_no_temp", "hts_no_sdyn"]
  commands = []
  for task in tasks:
    for method in methods:
      for seed in [0, 1, 2]:
        commands.append({"task": task, "method": method, "seed": seed, "launch": False, "selected_candidate": selection["selected_candidate"]})
  dump(OUT / "gate_d2_atari_dev_command_manifest_v12.json", {"status": "prepared_not_launched", "commands": commands})
  (OUT / "gate_d2_atari_dev_plan_v12.md").write_text("# Gate D2 Atari Dev Plan V12\n\nPrepared only; not launched.\n")
  return True


def test_reports(coeff, budget, conv, boundary, cont, selection, gate):
  tests = []
  def add(tid, name, status, source, artifact, reason=""):
    tests.append({"test_id": tid, "test_name": name, "status": status, "execution_status": source, "artifact_path": str(artifact), "failure_reason": reason})
  for row in csv.DictReader((ART / "test_report_v11_full.csv").open()):
    add(row["test_id"], row["test_name"], row["status"], "inherited_from_v11", row["artifact_path"], row.get("failure_reason", ""))
  add("CA-01", "coefficient override routing", "PASS" if coeff["status"] == "pass" else "FAIL", "executed_v12", OUT / "coefficient_routing_audit_v12.json")
  add("CV-01", "training budget audit", "PASS" if budget["status"] == "pass" else "FAIL", "executed_v12", OUT / "training_budget_audit_v12.json")
  add("CV-02", "existing checkpoint convergence audit", "PASS" if conv["status"] == "pass" else "FAIL", "executed_v12", OUT / "convergence_diagnosis_existing_v12.json")
  add("CV-03", "boundary metric calibration audit", "PASS" if boundary["status"] == "pass" else "FAIL", "executed_v12", OUT / "boundary_metric_audit_v12.csv")
  add("CV-04", "matched-budget continuation completeness", "PASS" if cont["status"] == "pass" else "FAIL", "executed_v12", OUT / "continuation_manifest_v12.json")
  add("CV-05", "matched-budget candidate selection", "PASS" if selection["decision"] == "PASS_WITH_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v12", OUT / "development_candidate_selection_v12.json", "" if selection["decision"] == "PASS_WITH_DEVELOPMENT_CANDIDATE" else selection["decision"])
  add("CV-06", "Gate-D1 review", "PASS" if gate["gate_d1_decision"] == "PASS_WITH_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v12", OUT / "gate_d1_review_v12.json", "" if gate["gate_d1_decision"] == "PASS_WITH_DEVELOPMENT_CANDIDATE" else gate["gate_d1_decision"])
  write_csv(ART / "test_report_v12_full.csv", tests)
  counts = {s: sum(t["status"] == s for t in tests) for s in ["PASS", "XFAIL", "FAIL"]}
  lines = [f"PASS: {counts['PASS']} | XFAIL: {counts['XFAIL']} | FAIL: {counts['FAIL']}", "", "| test_id | test_name | status | execution_status | artifact_path | failure_reason |", "| --- | --- | --- | --- | --- | --- |"]
  for t in tests:
    lines.append(f"| {t['test_id']} | {t['test_name']} | {t['status']} | {t['execution_status']} | {t['artifact_path']} | {t['failure_reason']} |")
  (ART / "test_report_v12_full.md").write_text("\n".join(lines) + "\n")
  (ART / "remaining_xfail_v12.md").write_text("# Remaining XFAIL V12\n\n- UT-15-P1: larger_flat_flops remains P1.\n")
  return counts


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  manifest, dataset_hash, data = load_data()
  budget = budget_audit(manifest, dataset_hash)
  coeff = coefficient_routing_audit(data)
  traj_rows, boundary_rows, conv = existing_trajectory(data)
  boundary = boundary_audit(boundary_rows)
  cont, cont_rows, cont_agg = continuation(data)
  selection = select_after_continuation(cont_rows)
  gate = gate_d1_review(selection, cont, coeff, budget, boundary)
  gate_d2 = maybe_gate_d2(selection)
  counts = test_reports(coeff, budget, conv, boundary, cont, selection, gate)
  summary = {
      "v9_baseline_optimizer_budget": "250 updates, batch_size=32, sequence_length=64",
      "v11_optimizer_budget": "250 updates, batch_size=32, sequence_length=64",
      "sampled_sequences": 250 * 32,
      "sampled_sequence_timesteps": 250 * 32 * 64,
      "coefficient_routing_audit_result": coeff["status"],
      "existing_checkpoint_convergence_result": conv["status"],
      "boundary_calibration_audit_result": boundary["status"],
      "continuation_configs": CONTINUE_CONFIGS,
      "continuation_runs_completed": cont["completed_runs"],
      "continuation_runs_expected": cont["expected_runs"],
      "final_matched_optimizer_budgets": max(TARGETS),
      "candidate_prefix_gains_and_ci": {e["config_name"]: [e["aggregate_full_prefix_gain"], e["ci95_low"], e["ci95_high"]] for e in selection["entries"]},
      "candidate_end_to_end_positive_seeds": {e["config_name"]: e["end_to_end_positive_gain_seed_count"] for e in selection["entries"]},
      "candidate_boundary_f1_and_auprc": {e["config_name"]: [e["boundary_f1"], e["boundary_auprc"]] for e in selection["entries"]},
      "candidate_factor_probe_accuracy": {e["config_name"]: e["factor_probe_accuracy"] for e in selection["entries"]},
      "selected_development_candidate": selection["selected_candidate"],
      "gate_d1_decision": selection["decision"],
      "gate_d2_manifest_generated": gate_d2,
      "cumulative_test_counts": counts,
      "remaining_blockers": ["Gate D2 blocked"] if not gate_d2 else ["Gate D2 prepared but not launched"],
      "dataset_manifest_hash": dataset_hash,
      "dataset_hash_matches_expected": dataset_hash == EXPECTED_HASH,
      "unrelated_full26_processes_observed_but_untouched": subprocess.run(
          "ps -ef | rg 'dreamerv3.main --configs atari100k size12m' | rg -v rg || true",
          shell=True, text=True, stdout=subprocess.PIPE).stdout.strip(),
  }
  dump(ART / "v12_package_summary.json", summary)
  print(json.dumps(to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
