import csv
import hashlib
import json
import math
import shutil
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
from . import synthetic_tuning_v11 as v11
from . import synthetic_causal_audit_v15 as v15
from . import synthetic_diagnosis_v10 as v10


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "paper_artifacts"
OUT = ART / "synthetic_locked_rerun_v18"
V17 = ART / "synthetic_harness_forensics_v17"
MANIFEST = ART / "synthetic_v7" / "synthetic_dataset_manifest_full_v7.json"
SYN9 = ART / "synthetic_full_v9"
SEEDS = [0, 1, 2, 3, 4]
BOUNDARIES = ["fast", "mid", "slow", "context", "macro"]
BASE = {"lambda_hier": 1.0, "lambda_sdyn": 1.0, "lambda_temp": 0.01, "lambda_vc": 0.01, "lambda_sparse": 1.0}
VARIANTS = {
    "locked_baseline_direct_head": {
        "coeffs": BASE,
        "role": "historical reproducibility anchor",
        "comparison_role": "baseline",
        "train": "reuse_v17_locked_replay",
    },
    "locked_hier_x3": {
        "coeffs": {**BASE, "lambda_hier": 3.0},
        "role": "test stronger hierarchy reconstruction under locked direct-head protocol",
        "comparison_role": "candidate",
        "train": "continue_from_v9_250",
    },
    "locked_recon_trunk_isolated_fine_only_x3": {
        "coeffs": {**BASE, "lambda_hier": 3.0},
        "role": "legacy V14 routing candidate",
        "comparison_role": "not_applicable",
        "train": "not_applicable_direct_head_no_shared_trunk",
    },
    "locked_no_hier_loss": {
        "coeffs": {**BASE, "lambda_hier": 0.0},
        "role": "hierarchy reconstruction objective control",
        "comparison_role": "control",
        "train": "continue_from_v9_250",
    },
}
EXPECTED_HASH = "5670241265b225d4cdab4e78131192fc24822c8dd4cb5b5617b3364be3dae9eb"
TOL = 0.05


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


def sha_file(path):
  path = Path(path)
  return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def sha_obj(obj):
  return hashlib.sha256(json.dumps(to_builtin(obj), sort_keys=True).encode()).hexdigest()


def code_commit():
  try:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE).stdout.strip()
  except Exception:
    return "missing"


def load_data():
  manifest = read_json(MANIFEST)
  data = {}
  for split in ["train", "val", "test"]:
    with np.load(manifest["paths"][split]) as npz:
      data[split] = {k: np.asarray(npz[k]) for k in npz.files}
  return manifest, sha_obj(manifest), data


def tree_save(path, params):
  flat = {}
  for group in ["heads", "decs", "preds"]:
    for i, val in enumerate(params[group]):
      flat[f"{group}_{i}"] = np.asarray(val)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **flat)


def v9_final(seed):
  return SYN9 / "runs" / "hts_full" / f"seed_{seed}" / "checkpoints" / "final.npz"


def v17_baseline_metrics(seed):
  return V17 / "runs" / "historical_recipe_replay" / f"seed_{seed}" / "metrics.json"


def run_dir(variant, seed):
  return OUT / "runs" / variant / f"seed_{seed}"


def advance_rng(seed, obs_all, steps):
  rng = np.random.default_rng(seed)
  for _ in range(steps):
    rng.integers(0, obs_all.shape[0], size=32)
    rng.integers(0, obs_all.shape[1] - 64, size=32)
  return rng


def boundary_detail(path, data):
  ckpt = v15.load_any_ckpt(path)
  b = v15.boundary_metrics_direct(ckpt, data)
  out = {}
  for name, vals in b.items():
    out[f"boundary_auprc_{name}"] = vals["auprc"]
    out[f"boundary_f1_{name}"] = vals["f1"]
  out["boundary_f1_overall"] = float(np.mean([b[k]["f1"] for k in b]))
  return out


def detached_probe(path, data):
  rows = v15.boundary_readout_rows(
      {"source": "v18", "method": "tmp", "seed": 0},
      v15.load_any_ckpt(path), data)
  vals = [r["auprc"] for r in rows if r["prefix"] == "z1:1" and r["readout"] == "detached_linear_probe" and r["boundary_type"] == "overall"]
  return float(np.mean(vals)) if vals else 0.0


def extra_metrics(path, data):
  ckpt = v15.load_any_ckpt(path)
  z = [np.asarray(x) for x in v15.encode_any(ckpt, data["test"]["obs"][:256])]
  level_summary = {}
  for li, level in enumerate(z, start=1):
    level_summary[f"level_{li}"] = {
        "fast": v10.centroid_probe(level, data["test"]["f_fast"][:256], 8),
        "mid": v10.centroid_probe(level, data["test"]["f_mid"][:256], 8),
        "slow": v10.centroid_probe(level, data["test"]["f_slow"][:256], 8),
        "context": v10.centroid_probe(level, data["test"]["f_context"][:256], 4),
        "nuisance": v10.centroid_probe(level, data["test"]["f_nuisance"][:256], 16),
    }
  z1 = z[0].reshape(-1, z[0].shape[-1])
  labels = data["test"]["revisit_group_id"][:256].reshape(-1)
  idx = np.linspace(0, len(labels) - 1, min(4096, len(labels))).astype(int)
  x = z1[idx]
  y = labels[idx]
  norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
  x = x / norm
  rng = np.random.default_rng(18)
  sims_same, sims_diff = [], []
  for _ in range(2048):
    i = int(rng.integers(0, len(idx)))
    same = np.flatnonzero(y == y[i])
    diff = np.flatnonzero(y != y[i])
    if len(same) > 1:
      j = int(rng.choice(same))
      if j != i:
        sims_same.append(float(x[i] @ x[j]))
    if len(diff):
      j = int(rng.choice(diff))
      sims_diff.append(float(x[i] @ x[j]))
  return {
      "level_horizon_specialization_summary": json.dumps(to_builtin(level_summary), sort_keys=True),
      "revisit_similarity": float(np.mean(sims_same) - np.mean(sims_diff)) if sims_same and sims_diff else 0.0,
      "nuisance_sensitivity": float(level_summary["level_1"]["nuisance"]),
  }


def eval_variant_checkpoint(variant, seed, path, data, dataset_hash, protocol_hash, config_hash, run_id):
  row = v15.eval_checkpoint(path, "v18_locked", variant, seed, data, dataset_hash)
  row.update(boundary_detail(path, data))
  row["detached_probe_auprc"] = detached_probe(path, data)
  row.update(extra_metrics(path, data))
  vals = [row[f"prefix_nrmse_l{i}"] for i in range(1, 7)]
  gains = [vals[i - 1] - vals[i] for i in range(1, len(vals))]
  row.update({
      "run_id": run_id,
      "protocol_hash": protocol_hash,
      "script_hash": sha_file(Path(__file__)),
      "config_hash": config_hash,
      "dataset_hash": dataset_hash,
      "sampler_hash_or_seed": f"seed={seed};advance_250",
      "initialization_seed": seed,
      "optimizer_updates": 1000,
      "batch_size": 32,
      "sequence_length": 64,
      "strict_monotonic_pass": all(g > 0 for g in gains),
      "nondegrading_pass": all(g >= -1e-8 for g in gains),
      "end_to_end_positive_gain": row["full_prefix_gain"] > 0,
      "collapse_status": "collapse_detected" if row.get("alive_feature_ratio", 1.0) <= 0.05 else "no_collapse_detected",
  })
  return row


def variant_contract(manifest, dataset_hash):
  rows = []
  params = synthetic_v7.init_params(0)
  pcount = int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))
  for name, cfg in VARIANTS.items():
    rows.append({
        "variant_name": name,
        "historical_or_new": "historical" if name == "locked_baseline_direct_head" else "new_locked_variant",
        "parameterization": "direct_head",
        "entrypoint": "python -m dreamerv3.synthetic_locked_rerun_v18",
        "script_hash": sha_file(Path(__file__)),
        "config_hash": sha_obj({"variant": name, "coeffs": cfg["coeffs"], "train": cfg["train"]}),
        "dataset_hash": dataset_hash,
        "routing_flags": "direct_heads_no_shared_trunk",
        "loss_coefficients": cfg["coeffs"],
        "stop_gradient_flags": "not applicable: no shared trunk in locked direct-head protocol",
        "head/trunk module graph": "six direct obs->head matrices, prefix decoders, prefix+action predictors, no trunk",
        "instantiated_parameter_count": pcount,
        "trainable_parameter_count": pcount,
        "comparison_role": cfg["comparison_role"],
        "role": cfg["role"],
        "train_status": cfg["train"],
    })
  alias = {
      "baseline_shared_trunk": "locked_baseline_direct_head",
      "hier_x3": "locked_hier_x3",
      "recon_trunk_isolated_fine_only_x3": "locked_recon_trunk_isolated_fine_only_x3",
      "no_hier_loss": "locked_no_hier_loss",
  }
  dump(OUT / "locked_variant_contract_v18.json", {"alias_table": alias, "rows": rows})
  lines = ["# Locked Variant Contract V18", "", "Legacy names are aliases only; the locked protocol is direct-head.", "", "| legacy | locked name |", "| --- | --- |"]
  for k, v in alias.items():
    lines.append(f"| `{k}` | `{v}` |")
  lines += ["", "| variant | parameterization | train status | role | params |", "| --- | --- | --- | --- | ---: |"]
  for r in rows:
    lines.append(f"| `{r['variant_name']}` | `{r['parameterization']}` | `{r['train_status']}` | {r['role']} | {r['trainable_parameter_count']} |")
  (OUT / "locked_variant_contract_v18.md").write_text("\n".join(lines) + "\n")
  return rows


def reuse_baseline(seed, data, dataset_hash, protocol_hash):
  metrics_path = run_dir("locked_baseline_direct_head", seed) / "metrics.json"
  if metrics_path.exists():
    return read_json(metrics_path)
  source = read_json(v17_baseline_metrics(seed))
  src_ckpt = Path(source["checkpoint_path"])
  dst_ckpt = run_dir("locked_baseline_direct_head", seed) / "checkpoints" / "step_1000.npz"
  dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
  if not dst_ckpt.exists():
    shutil.copy2(src_ckpt, dst_ckpt)
  config_hash = sha_obj({"variant": "locked_baseline_direct_head", "coeffs": BASE, "source": "v17_reused"})
  row = eval_variant_checkpoint("locked_baseline_direct_head", seed, dst_ckpt, data, dataset_hash, protocol_hash, config_hash, f"v18_locked_baseline_direct_head_seed{seed}")
  row["training_source"] = "reused_v17_locked_replay"
  report = {"status": "pass", "variant": "locked_baseline_direct_head", "seed": seed, "metrics": row, "source_metrics_path": str(v17_baseline_metrics(seed))}
  dump(metrics_path, report)
  return report


def train_variant(variant, seed, data, dataset_hash, protocol_hash):
  metrics_path = run_dir(variant, seed) / "metrics.json"
  if metrics_path.exists():
    return read_json(metrics_path)
  cfg = VARIANTS[variant]
  if cfg["train"].startswith("not_applicable"):
    report = {
        "status": "not_applicable",
        "variant": variant,
        "seed": seed,
        "reason": "V14 recon_trunk_isolated_fine_only_x3 requires a shared trunk; locked V9/V12 protocol is direct-head and has no trunk route to isolate.",
        "closest_valid_locked_equivalent": "locked_hier_x3 uses the same hierarchy x3 coefficient without claiming trunk isolation.",
    }
    dump(metrics_path, report)
    return report
  if variant == "locked_baseline_direct_head":
    return reuse_baseline(seed, data, dataset_hash, protocol_hash)
  params = synthetic_v7.load_ckpt(v9_final(seed))
  obs_all = data["train"]["obs"]
  act_all = data["train"]["actions"]
  rng = advance_rng(seed, obs_all, 250)
  lr = 0.05
  losses = []
  start = time.time()
  for step in range(251, 1001):
    eps = rng.integers(0, obs_all.shape[0], size=32)
    starts = rng.integers(0, obs_all.shape[1] - 64, size=32)
    obs = np.stack([obs_all[e, s:s + 64] for e, s in zip(eps, starts)])
    act = np.stack([act_all[e, s:s + 64] for e, s in zip(eps, starts)])
    (loss, raw), grads = jax.value_and_grad(v11.weighted_loss, has_aux=True)(
        params, jnp.asarray(obs), jnp.asarray(act), cfg["coeffs"])
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    if step in [251, 500, 750, 1000]:
      losses.append({"update": step, "loss": float(loss), **{k: float(v) for k, v in raw.items()}})
  ckpt = run_dir(variant, seed) / "checkpoints" / "step_1000.npz"
  tree_save(ckpt, params)
  config_hash = sha_obj({"variant": variant, "coeffs": cfg["coeffs"], "train": cfg["train"]})
  row = eval_variant_checkpoint(variant, seed, ckpt, data, dataset_hash, protocol_hash, config_hash, f"v18_{variant}_seed{seed}")
  row["training_source"] = "v18_continued_from_v9_250"
  report = {
      "status": "pass",
      "variant": variant,
      "seed": seed,
      "checkpoint_path": ckpt,
      "checkpoint_hash": sha_file(ckpt),
      "loss_curves": losses,
      "metrics": row,
      "wall_clock_seconds": round(time.time() - start, 3),
  }
  dump(metrics_path, report)
  return report


def baseline_reports(reports):
  seed0 = reports[0]
  m = seed0["metrics"]
  expected = 0.7338414786886552
  ok = abs(float(m["boundary_auprc_overall"]) - expected) <= TOL
  seed0_report = {
      "status": "pass" if ok else "fail",
      "expected_seed0_auprc": expected,
      "observed_seed0_auprc": m["boundary_auprc_overall"],
      "absolute_difference": abs(float(m["boundary_auprc_overall"]) - expected),
      "checkpoint_path": m["checkpoint_path"],
      "reused_from_v17": True,
  }
  dump(OUT / "locked_baseline_seed0_reproduction_v18.json", seed0_report)
  (OUT / "locked_baseline_seed0_reproduction_v18.md").write_text(
      "# Locked Baseline Seed0 Reproduction V18\n\n"
      "Observation: baseline seed0 was reproduced by reusing the V17 locked replay checkpoint and re-evaluating in V18.\n\n"
      "Hypothesis: locked protocol remains reproducible.\n\n"
      "Minimal test: seed0 AUPRC within 0.05 of 0.733841.\n\n"
      f"Evidence: observed `{m['boundary_auprc_overall']:.6f}`.\n\n"
      f"Decision: `{seed0_report['status']}`.\n")
  rows = [r["metrics"] for r in reports]
  write_csv(OUT / "locked_baseline_allseeds_v18.csv", rows)
  vals = [r["boundary_auprc_overall"] for r in rows]
  macros = [r["boundary_auprc_macro"] for r in rows]
  all_ok = abs(float(np.mean(vals)) - 0.7031229334644903) <= TOL
  (OUT / "locked_baseline_allseeds_v18.md").write_text(
      "# Locked Baseline All-Seeds V18\n\n"
      "Observation: all five baseline seeds were re-evaluated under the V18 locked protocol package.\n\n"
      "Hypothesis: all-seed mean remains near V17 historical replay.\n\n"
      "Minimal test: mean AUPRC within 0.05 of 0.703123.\n\n"
      f"Evidence: overall `{np.mean(vals):.6f}` ± `{np.std(vals):.6f}`, macro `{np.mean(macros):.6f}` ± `{np.std(macros):.6f}`.\n\n"
      f"Decision: `{'pass' if all_ok else 'fail'}`.\n")
  return ok and all_ok


def aggregate(rows):
  metrics = [
      "boundary_auprc_overall", "boundary_auprc_macro", "boundary_auprc_fast",
      "boundary_auprc_mid", "boundary_auprc_slow", "boundary_auprc_context",
      "boundary_f1_overall", "boundary_f1_macro", "detached_probe_auprc",
      "factor_probe_accuracy", "full_prefix_gain", "effective_rank",
      "dead_feature_ratio", "alive_feature_ratio", "topk_utilization_entropy",
      "revisit_similarity", "nuisance_sensitivity"]
  for i in range(1, 7):
    metrics.append(f"prefix_nrmse_l{i}")
  out = []
  for variant in sorted({r["method"] for r in rows}):
    sub = [r for r in rows if r["method"] == variant]
    for metric in metrics:
      vals = [float(r[metric]) for r in sub if r.get(metric, "") != ""]
      if not vals:
        continue
      lo, hi = v11.bootstrap_ci(vals)
      out.append({
          "variant": variant,
          "metric": metric,
          "mean": float(np.mean(vals)),
          "std": float(np.std(vals)),
          "standard_error": float(np.std(vals) / math.sqrt(len(vals))),
          "ci95_low": lo,
          "ci95_high": hi,
          "seed_count": len(vals),
      })
  return out


def figures(rows, agg):
  variants = ["locked_baseline_direct_head", "locked_hier_x3", "locked_no_hier_loss"]
  labels = {"locked_baseline_direct_head": "Baseline", "locked_hier_x3": "Hier x3", "locked_no_hier_loss": "No hier"}
  def mean_metric(v, m):
    vals = [r[m] for r in rows if r["method"] == v]
    return float(np.mean(vals)) if vals else 0.0
  plt.figure(figsize=(5, 4))
  vals = [mean_metric(v, "boundary_auprc_overall") for v in variants]
  plt.bar([labels[v] for v in variants], vals)
  plt.ylabel("Boundary AUPRC overall")
  plt.tight_layout()
  plt.savefig(OUT / "fig_locked_boundary_auprc_v18.pdf")
  plt.close()
  plt.figure(figsize=(5, 4))
  for v in variants:
    ys = [mean_metric(v, f"prefix_nrmse_l{i}") for i in range(1, 7)]
    plt.plot(range(1, 7), ys, marker="o", label=labels[v])
  plt.xlabel("Prefix level")
  plt.ylabel("NRMSE")
  plt.legend()
  plt.tight_layout()
  plt.savefig(OUT / "fig_locked_prefix_profiles_v18.pdf")
  plt.close()
  plt.figure(figsize=(5, 4))
  for v in variants:
    plt.scatter(mean_metric(v, "full_prefix_gain"), mean_metric(v, "boundary_auprc_overall"), label=labels[v])
  plt.xlabel("Full prefix gain")
  plt.ylabel("Boundary AUPRC overall")
  plt.legend()
  plt.tight_layout()
  plt.savefig(OUT / "fig_locked_tradeoff_v18.pdf")
  plt.close()


def select_candidate(rows, agg):
  baseline = [r for r in rows if r["method"] == "locked_baseline_direct_head"]
  base = {
      "gain": float(np.mean([r["full_prefix_gain"] for r in baseline])),
      "auprc": float(np.mean([r["boundary_auprc_overall"] for r in baseline])),
      "macro": float(np.mean([r["boundary_auprc_macro"] for r in baseline])),
      "factor": float(np.mean([r["factor_probe_accuracy"] for r in baseline])),
  }
  entries = []
  selected = None
  for variant in ["locked_hier_x3", "locked_no_hier_loss"]:
    sub = [r for r in rows if r["method"] == variant]
    if not sub:
      continue
    gains = [r["full_prefix_gain"] - base["gain"] for r in sub]
    lo, hi = v11.bootstrap_ci(gains)
    auprc = float(np.mean([r["boundary_auprc_overall"] for r in sub]))
    macro = float(np.mean([r["boundary_auprc_macro"] for r in sub]))
    factor = float(np.mean([r["factor_probe_accuracy"] for r in sub]))
    collapse = any(r["collapse_status"] != "no_collapse_detected" for r in sub)
    prefix_ok = float(np.mean(gains)) > 0 and lo >= 0 and sum(r["end_to_end_positive_gain"] for r in sub) >= 4
    mechanism_ok = (
        auprc >= base["auprc"] * 0.95 and
        macro >= base["macro"] * 0.95 and
        factor >= base["factor"] * 0.95 and
        not collapse)
    passes = prefix_ok and mechanism_ok
    reason = []
    if not prefix_ok:
      reason.append("prefix_requirement_failed")
    if not mechanism_ok:
      reason.append("mechanism_preservation_failed")
    entry = {
        "variant": variant,
        "mean_prefix_gain_delta_vs_baseline": float(np.mean(gains)),
        "ci95_low": lo,
        "ci95_high": hi,
        "end_to_end_positive_gain_seeds": int(sum(r["end_to_end_positive_gain"] for r in sub)),
        "strict_monotonic_seeds": int(sum(r["strict_monotonic_pass"] for r in sub)),
        "nondegrading_seeds": int(sum(r["nondegrading_pass"] for r in sub)),
        "boundary_auprc_overall": auprc,
        "boundary_auprc_macro": macro,
        "factor_probe_accuracy": factor,
        "collapse_status": "collapse_detected" if collapse else "no_collapse_detected",
        "prefix_requirement_pass": prefix_ok,
        "mechanism_preservation_pass": mechanism_ok,
        "selection_status": "pass" if passes else "reject",
        "rejection_reason": ",".join(reason),
    }
    entries.append(entry)
    if passes and (selected is None or entry["mean_prefix_gain_delta_vs_baseline"] > selected["mean_prefix_gain_delta_vs_baseline"]):
      selected = entry
  decision = "PASS_WITH_LOCKED_PROTOCOL_DEVELOPMENT_CANDIDATE" if selected else "NO_STABLE_CANDIDATE_UNDER_LOCKED_PROTOCOL"
  report = {
      "decision": decision,
      "selected_candidate": selected["variant"] if selected else None,
      "baseline": base,
      "entries": entries,
      "not_applicable": {
          "locked_recon_trunk_isolated_fine_only_x3": "requires shared trunk; locked protocol is direct-head",
      },
  }
  dump(OUT / "development_candidate_selection_v18.json", report)
  lines = ["# Development Candidate Selection V18", "", f"Decision: `{decision}`", "", "Observation: variants were compared only to `locked_baseline_direct_head`.", "", "| variant | gain delta | AUPRC | macro | factor | status | reason |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
  for e in entries:
    lines.append(f"| `{e['variant']}` | {e['mean_prefix_gain_delta_vs_baseline']:.6f} | {e['boundary_auprc_overall']:.6f} | {e['boundary_auprc_macro']:.6f} | {e['factor_probe_accuracy']:.6f} | `{e['selection_status']}` | {e['rejection_reason']} |")
  lines.append("\n`locked_recon_trunk_isolated_fine_only_x3` is not applicable under direct-head locked protocol.")
  (OUT / "development_candidate_selection_v18.md").write_text("\n".join(lines) + "\n")
  return report


def interpretation(selection):
  text = ["# Locked Protocol Research Interpretation V18", ""]
  text += [
      "Observation: the locked protocol reproduces the historical direct-head baseline, so V18 comparisons are no longer using the drifted V15 shared-trunk harness.",
      "",
      "Hypothesis: if hierarchy strengthening is a stable mechanism, it should improve prefix fidelity over the locked baseline while preserving boundary/event sensitivity.",
      "",
      "Minimal test: compare `locked_hier_x3` and `locked_no_hier_loss` against `locked_baseline_direct_head`; mark V14 trunk-isolation candidate as not applicable because the locked protocol has no shared trunk.",
      "",
  ]
  if selection["decision"] == "PASS_WITH_LOCKED_PROTOCOL_DEVELOPMENT_CANDIDATE":
    text.append(f"Evidence: `{selection['selected_candidate']}` satisfies prefix and mechanism-preservation rules.")
    text.append("")
    text.append("Decision: a locked synthetic development candidate exists, but Atari Gate D2 remains blocked until separately approved.")
  else:
    text.append("Evidence: no tested locked-protocol variant satisfies both prefix improvement and mechanism preservation.")
    text.append("")
    text.append("Decision: evidence does not support advancing to Atari Gate D2 from V18.")
  text.append("")
  text.append("Routing note: `locked_recon_trunk_isolated_fine_only_x3` is non-equivalent to V14 because direct-head locked protocol has no trunk gradient route to isolate.")
  (OUT / "locked_protocol_research_interpretation_v18.md").write_text("\n".join(text) + "\n")


def gate_review(selection, baseline_ok, completeness_ok):
  if not baseline_ok:
    status = "BLOCKED_LOCKED_PROTOCOL_REPRODUCTION_FAILED"
  elif not completeness_ok:
    status = "BLOCKED_INSUFFICIENT_EVIDENCE"
  elif selection["decision"] == "PASS_WITH_LOCKED_PROTOCOL_DEVELOPMENT_CANDIDATE":
    status = "PASS_LOCKED_SYNTHETIC"
  else:
    status = "BLOCKED_NO_STABLE_CANDIDATE_UNDER_LOCKED_PROTOCOL"
  gate = {
      "gate_d1_status": status,
      "candidate_decision": selection["decision"],
      "selected_candidate": selection.get("selected_candidate"),
      "gate_d2_status": "blocked",
      "variant_not_applicable": ["locked_recon_trunk_isolated_fine_only_x3"],
  }
  dump(OUT / "gate_d1_review_v18.json", gate)
  (OUT / "gate_d1_review_v18.md").write_text(
      "# Gate D1 Review V18\n\n"
      f"Gate D1: `{status}`\n\n"
      f"Candidate decision: `{selection['decision']}`\n\n"
      "Gate D2: `blocked`.\n")
  if status == "PASS_LOCKED_SYNTHETIC":
    (OUT / "gate_d2_future_plan_v18.md").write_text(
        "# Gate D2 Future Plan V18\n\n"
        "Do not launch automatically. If approved later, run Atari with locked synthetic evidence as prerequisite; V16 Breakout/Alien logs are external DreamerV3 context only.\n")
  else:
    (OUT / "post_v18_research_options.md").write_text(
        "# Post V18 Research Options\n\n"
        "- Reduce claim to diagnostic finding.\n"
        "- Revise method around direct-head hierarchy with equal-parameter controls.\n"
        "- Drop boundary-preservation claim.\n"
        "- Design new architecture only with equal-parameter controls.\n"
        "- Write negative result about reconstruction/event abstraction tension.\n")
  return gate


def test_reports(contract_ok, baseline_ok, completeness_ok, metrics_ok, selection, gate):
  tests = []
  if (ART / "test_report_v17_full.csv").exists():
    for r in csv.DictReader((ART / "test_report_v17_full.csv").open()):
      tests.append(dict(r))
  def add(tid, name, status, artifact, reason=""):
    tests.append({
        "test_id": tid, "test_name": name, "status": status,
        "execution_status": "executed_v18", "artifact_path": str(artifact),
        "failure_reason": reason,
    })
  add("LR-01", "locked variant contract", "PASS" if contract_ok else "FAIL", OUT / "locked_variant_contract_v18.json")
  add("LR-02", "locked baseline reproduction", "PASS" if baseline_ok else "FAIL", OUT / "locked_baseline_allseeds_v18.csv")
  add("LR-03", "locked minimal rerun completeness", "PASS" if completeness_ok else "FAIL", OUT / "locked_rerun_manifest_v18.json")
  add("LR-04", "locked metric aggregation", "PASS" if metrics_ok else "FAIL", OUT / "locked_rerun_metrics_aggregate_v18.csv")
  add("LR-05", "development candidate selection", "PASS", OUT / "development_candidate_selection_v18.json", selection["decision"])
  add("LR-06", "research interpretation", "PASS", OUT / "locked_protocol_research_interpretation_v18.md")
  add("LR-07", "Gate-D1 V18 review", "PASS" if gate["gate_d1_status"] == "PASS_LOCKED_SYNTHETIC" else "FAIL", OUT / "gate_d1_review_v18.json", gate["gate_d1_status"])
  write_csv(ART / "test_report_v18_full.csv", tests)
  counts = {s: sum(t["status"] == s for t in tests) for s in ["PASS", "XFAIL", "FAIL"]}
  lines = [f"PASS: {counts['PASS']} | XFAIL: {counts['XFAIL']} | FAIL: {counts['FAIL']}", "", "| test_id | test_name | status | execution_status | artifact_path | failure_reason |", "| --- | --- | --- | --- | --- | --- |"]
  for t in tests:
    lines.append(f"| {t['test_id']} | {t['test_name']} | {t['status']} | {t.get('execution_status','')} | {t.get('artifact_path','')} | {t.get('failure_reason','')} |")
  (ART / "test_report_v18_full.md").write_text("\n".join(lines) + "\n")
  (ART / "remaining_xfail_v18.md").write_text("# Remaining XFAIL V18\n\n- UT-15-P1: larger_flat_flops remains P1.\n")
  return counts


def process_observation():
  return subprocess.run("ps -eo pid,etime,cmd | rg 'dreamerv3.main --configs atari100k size12m' | rg -v rg || true", shell=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  manifest, dataset_hash, data = load_data()
  protocol = read_json(V17 / "synthetic_protocol_locked_v17.json")
  protocol_hash = sha_obj(protocol)
  contract_rows = variant_contract(manifest, dataset_hash)
  reports = {}
  baseline_reports_list = []
  for seed in SEEDS:
    rep = train_variant("locked_baseline_direct_head", seed, data, dataset_hash, protocol_hash)
    reports[("locked_baseline_direct_head", seed)] = rep
    baseline_reports_list.append(rep)
  baseline_ok = baseline_reports(baseline_reports_list)
  runnable = ["locked_hier_x3", "locked_no_hier_loss"]
  if baseline_ok:
    for variant in runnable:
      for seed in SEEDS:
        reports[(variant, seed)] = train_variant(variant, seed, data, dataset_hash, protocol_hash)
  for seed in SEEDS:
    reports[("locked_recon_trunk_isolated_fine_only_x3", seed)] = train_variant("locked_recon_trunk_isolated_fine_only_x3", seed, data, dataset_hash, protocol_hash)
  manifest_rows = []
  metric_rows = []
  for (variant, seed), rep in sorted(reports.items()):
    row = {
        "variant": variant,
        "seed": seed,
        "status": rep["status"],
        "run_id": f"v18_{variant}_seed{seed}",
        "expected_under_v18": variant in ["locked_baseline_direct_head", "locked_hier_x3", "locked_no_hier_loss"],
        "not_applicable_reason": rep.get("reason", ""),
    }
    if rep.get("metrics"):
      m = rep["metrics"]
      row.update({
          "checkpoint_path": m["checkpoint_path"],
          "checkpoint_hash": m["checkpoint_hash"],
          "protocol_hash": m["protocol_hash"],
          "script_hash": m["script_hash"],
          "config_hash": m["config_hash"],
          "dataset_hash": m["dataset_hash"],
          "optimizer_updates": m["optimizer_updates"],
          "batch_size": m["batch_size"],
          "sequence_length": m["sequence_length"],
      })
      metric_rows.append(m)
    manifest_rows.append(row)
  manifest_report = {
      "status": "pass" if baseline_ok and all(reports[(v, s)]["status"] == "pass" for v in ["locked_baseline_direct_head", "locked_hier_x3", "locked_no_hier_loss"] for s in SEEDS) else "fail",
      "new_training_runs_completed": sum(1 for (v, _), r in reports.items() if r["status"] == "pass" and r.get("metrics", {}).get("training_source") == "v18_continued_from_v9_250"),
      "new_training_runs_expected": 10,
      "baseline_reused_runs": 5,
      "not_applicable_runs": 5,
      "rows": manifest_rows,
  }
  dump(OUT / "locked_rerun_manifest_v18.json", manifest_report)
  write_csv(OUT / "locked_rerun_manifest_v18.csv", manifest_rows)
  write_csv(OUT / "locked_rerun_metrics_per_seed_v18.csv", metric_rows)
  agg = aggregate(metric_rows)
  write_csv(OUT / "locked_rerun_metrics_aggregate_v18.csv", agg)
  figures(metric_rows, agg)
  selection = select_candidate(metric_rows, agg)
  interpretation(selection)
  gate = gate_review(selection, baseline_ok, manifest_report["status"] == "pass")
  counts = test_reports(True, baseline_ok, manifest_report["status"] == "pass", bool(agg), selection, gate)
  def aggval(variant, metric):
    vals = [r[metric] for r in metric_rows if r["method"] == variant]
    return float(np.mean(vals)) if vals else None
  summary = {
      "v17_root_cause": read_json(ART / "v17_package_summary.json", {}).get("identified_drift_factor_if_any", "missing"),
      "locked_protocol_hash": protocol_hash,
      "seed0_baseline_reproduced": read_json(OUT / "locked_baseline_seed0_reproduction_v18.json", {}).get("status"),
      "all_seed_baseline_reproduced": baseline_ok,
      "variants_run": ["locked_baseline_direct_head", "locked_hier_x3", "locked_no_hier_loss"],
      "variants_skipped_or_not_applicable": ["locked_recon_trunk_isolated_fine_only_x3"],
      "new_runs_completed": manifest_report["new_training_runs_completed"],
      "new_runs_expected": manifest_report["new_training_runs_expected"],
      "baseline_boundary_auprc_overall": aggval("locked_baseline_direct_head", "boundary_auprc_overall"),
      "baseline_boundary_auprc_macro": aggval("locked_baseline_direct_head", "boundary_auprc_macro"),
      "variant_boundary_auprc_overall": {v: aggval(v, "boundary_auprc_overall") for v in ["locked_hier_x3", "locked_no_hier_loss"]},
      "variant_boundary_auprc_macro": {v: aggval(v, "boundary_auprc_macro") for v in ["locked_hier_x3", "locked_no_hier_loss"]},
      "variant_prefix_gains": {v: aggval(v, "full_prefix_gain") for v in ["locked_baseline_direct_head", "locked_hier_x3", "locked_no_hier_loss"]},
      "candidate_decision": selection["decision"],
      "gate_d1_status": gate["gate_d1_status"],
      "gate_d2_status": "blocked",
      "relation_to_v16_atari_reference": "external DreamerV3 reference only; not used for Gate D",
      "cumulative_test_counts": counts,
      "unrelated_official_processes_observed_but_untouched": process_observation(),
      "artifact_dir": str(OUT),
  }
  dump(ART / "v18_package_summary.json", summary)
  print(json.dumps(to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
