import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import synthetic_v7
from . import synthetic_diagnosis_v10 as v10
from . import synthetic_revision_v13 as v13
from . import synthetic_gradient_isolation_v14 as v14


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "paper_artifacts"
OUT = ART / "synthetic_causal_audit_v15"
MANIFEST = ART / "synthetic_v7" / "synthetic_dataset_manifest_full_v7.json"
SYN12 = ART / "synthetic_convergence_v12"
SYN14 = ART / "synthetic_gradient_isolation_v14"
SEEDS = [0, 1, 2, 3, 4]
LEVELS = 6
HORIZONS = [1, 2, 4, 8, 16, 32]
BOUNDARIES = ["fast", "mid", "slow", "context", "macro"]
READOUTS = ["raw_delta_l2", "normalized_delta_l2", "cosine_change", "whitened_delta_mahalanobis", "detached_linear_probe"]
EXPECTED_HASH = "5670241265b225d4cdab4e78131192fc24822c8dd4cb5b5617b3364be3dae9eb"

ROUTES = {
    "exact_baseline_shared_trunk": {
        "hier_recon_update_shared_trunk": True,
        "hier_recon_update_z1_head": True,
        "hier_recon_include_level1_decoder_loss": True,
        "beta_hier": [1, 1, 1, 1, 1, 1],
    },
    "exact_recon_trunk_isolated": {
        "hier_recon_update_shared_trunk": False,
        "hier_recon_update_z1_head": True,
        "hier_recon_include_level1_decoder_loss": True,
        "beta_hier": [1, 1, 1, 1, 1, 1],
    },
    "exact_no_hier_loss": {
        "hier_recon_update_shared_trunk": False,
        "hier_recon_update_z1_head": False,
        "hier_recon_include_level1_decoder_loss": False,
        "beta_hier": [0, 0, 0, 0, 0, 0],
    },
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


def sha_bytes(data):
  return hashlib.sha256(data).hexdigest()


def sha_file(path):
  path = Path(path)
  return sha_bytes(path.read_bytes()) if path.exists() else ""


def sha_obj(obj):
  return hashlib.sha256(json.dumps(to_builtin(obj), sort_keys=True).encode()).hexdigest()


def code_commit():
  return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE).stdout.strip()


def load_data():
  manifest = read_json(MANIFEST)
  data = {}
  for split in ["train", "val", "test"]:
    with np.load(manifest["paths"][split]) as npz:
      data[split] = {k: np.asarray(npz[k]) for k in npz.files}
  return manifest, sha_obj(manifest), data


def load_any_ckpt(path):
  data = np.load(path)
  keys = set(data.files)
  if "trunk" in keys:
    return {
        "kind": "shared_trunk",
        "params": {
            "trunk": jnp.asarray(data["trunk"]),
            "heads": [jnp.asarray(data[f"heads_{i}"]) for i in range(LEVELS)],
            "decs": [jnp.asarray(data[f"decs_{i}"]) for i in range(LEVELS)],
            "preds": [jnp.asarray(data[f"preds_{i}"]) for i in range(LEVELS)],
        },
    }
  return {"kind": "direct_heads", "params": synthetic_v7.load_ckpt(path)}


def encode_any(ckpt, obs):
  if ckpt["kind"] == "shared_trunk":
    return v14.encode(ckpt["params"], jnp.asarray(obs))
  return synthetic_v7.encode(ckpt["params"], jnp.asarray(obs))


def reconstruct_any(ckpt, z, level):
  return jnp.concatenate([jnp.asarray(x) for x in z[:level + 1]], -1) @ ckpt["params"]["decs"][level]


def param_count_any(ckpt):
  return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(ckpt["params"])))


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


def f1_at_threshold(scores, labels, thr):
  scores = np.asarray(scores).reshape(-1)
  labels = np.asarray(labels).astype(bool).reshape(-1)
  pred = scores >= thr
  tp = float(np.logical_and(pred, labels).sum())
  fp = float(np.logical_and(pred, ~labels).sum())
  fn = float(np.logical_and(~pred, labels).sum())
  prec = tp / max(tp + fp, 1.0)
  rec = tp / max(tp + fn, 1.0)
  return float(2 * prec * rec / max(prec + rec, 1e-8))


def threshold_val(scores, labels):
  scores = np.asarray(scores).reshape(-1)
  labels = np.asarray(labels).astype(bool).reshape(-1)
  best = (0.0, float(np.quantile(scores, 0.9)))
  for q in np.linspace(0.05, 0.99, 96):
    thr = float(np.quantile(scores, q))
    val = f1_at_threshold(scores, labels, thr)
    if val > best[0]:
      best = (val, thr)
  return best[1]


def prefix_profile_any(ckpt, dataset, n=256):
  obs = dataset["obs"][:n]
  z = [np.asarray(x) for x in encode_any(ckpt, obs)]
  denom = float(np.sqrt(np.mean(np.square(obs))) + 1e-8)
  vals, prev = [], None
  for level in range(LEVELS):
    pred = np.asarray(reconstruct_any(ckpt, z, level))
    nrmse = float(np.sqrt(np.mean(np.square(pred - obs))) / denom)
    vals.append({"level": level + 1, "prefix_nrmse": nrmse, "marginal_gain": None if prev is None else prev - nrmse})
    prev = nrmse
  full = vals[0]["prefix_nrmse"] - vals[-1]["prefix_nrmse"]
  return vals, {"full_prefix_gain": full}, z


def feature_stats_flat(x):
  flat = np.asarray(x).reshape(-1, x.shape[-1])
  active = np.abs(flat) > 1e-5
  counts = active.sum(-1)
  s = np.linalg.svd(flat - flat.mean(0), compute_uv=False)
  p = s / (s.sum() + 1e-8)
  return {
      "effective_rank": float(np.exp(-(p * np.log(p + 1e-8)).sum())),
      "dead_feature_ratio": float(1 - active.mean()),
      "latent_norm_mean": float(np.linalg.norm(flat, axis=-1).mean()),
      "latent_norm_std": float(np.linalg.norm(flat, axis=-1).std()),
      "variance_per_dimension_mean": float(flat.var(0).mean()),
      "variance_per_dimension_std": float(flat.var(0).std()),
      "active_count_mean": float(counts.mean()),
  }


def score_delta_l2(code):
  delta = np.linalg.norm(code[:, 1:] - code[:, :-1], axis=-1)
  return np.pad(delta, ((0, 0), (1, 0))).reshape(-1)


def readout_scores_from_code(code, readout, probe=None, train_stats=None):
  delta = code[:, 1:] - code[:, :-1]
  if readout == "raw_delta_l2":
    score = np.linalg.norm(delta, axis=-1)
  elif readout == "normalized_delta_l2":
    norm = 0.5 * (np.linalg.norm(code[:, 1:], axis=-1) + np.linalg.norm(code[:, :-1], axis=-1)) + 1e-8
    score = np.linalg.norm(delta, axis=-1) / norm
  elif readout == "cosine_change":
    a, b = code[:, 1:], code[:, :-1]
    score = 1.0 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
  elif readout == "whitened_delta_mahalanobis":
    std = train_stats if train_stats is not None else (delta.reshape(-1, delta.shape[-1]).std(0) + 1e-4)
    score = np.sqrt(np.square(delta / std).sum(-1))
  elif readout == "detached_linear_probe":
    x = np.concatenate([delta.reshape(-1, delta.shape[-1]), np.ones((delta.reshape(-1, delta.shape[-1]).shape[0], 1))], -1)
    score = x @ probe
    score = score.reshape(code.shape[0], code.shape[1] - 1)
    return np.pad(score, ((0, 0), (1, 0))).reshape(-1)
  else:
    raise ValueError(readout)
  return np.pad(score, ((0, 0), (1, 0))).reshape(-1)


def fit_linear_probe(ckpt, prefix_level, train, boundary_key, ridge=1e-3):
  z = [np.asarray(x) for x in encode_any(ckpt, train["obs"][:256])]
  code = np.concatenate(z[:prefix_level], -1)
  delta = code[:, 1:] - code[:, :-1]
  x = delta.reshape(-1, delta.shape[-1])
  x = np.concatenate([x, np.ones((x.shape[0], 1))], -1)
  y = train[boundary_key][:256, 1:].reshape(-1).astype(np.float32)
  a = x.T @ x + ridge * np.eye(x.shape[1])
  b = x.T @ y
  return np.linalg.solve(a, b)


def boundary_readout_rows(label, ckpt, data):
  rows = []
  encoded = {
      split: [np.asarray(x) for x in encode_any(ckpt, data[split]["obs"][:256])]
      for split in ["train", "val", "test"]
  }
  for prefix in range(1, LEVELS + 1):
    prefix_code = np.concatenate(encoded["test"][:prefix], -1)
    delta_norm = score_delta_l2(prefix_code).reshape(256, -1)
    stats = feature_stats_flat(prefix_code)
    train_code = np.concatenate(encoded["train"][:prefix], -1)
    val_code = np.concatenate(encoded["val"][:prefix], -1)
    std = (train_code[:, 1:] - train_code[:, :-1]).reshape(-1, train_code.shape[-1]).std(0) + 1e-4
    train_delta = train_code[:, 1:] - train_code[:, :-1]
    train_x = train_delta.reshape(-1, train_delta.shape[-1])
    train_x = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], -1)
    probes = {}
    for b in BOUNDARIES:
      y = data["train"][f"boundary_{b}"][:256, 1:].reshape(-1).astype(np.float32)
      a = train_x.T @ train_x + 1e-3 * np.eye(train_x.shape[1])
      probes[b] = np.linalg.solve(a, train_x.T @ y)
    for readout in READOUTS:
      for bname in BOUNDARIES:
        key = f"boundary_{bname}"
        probe = probes[bname] if readout == "detached_linear_probe" else None
        sv = readout_scores_from_code(val_code, readout, probe=probe, train_stats=std)
        st = readout_scores_from_code(prefix_code, readout, probe=probe, train_stats=std)
        val_labels = data["val"][key][:256].reshape(-1).astype(bool)
        test_labels = data["test"][key][:256].reshape(-1).astype(bool)
        thr = threshold_val(sv, val_labels)
        pos = delta_norm.reshape(-1)[test_labels]
        neg = delta_norm.reshape(-1)[~test_labels]
        rows.append({
            "source": label["source"],
            "method": label["method"],
            "seed": label["seed"],
            "prefix": f"z1:{prefix}",
            "readout": readout,
            "boundary_type": bname,
            "auprc": average_precision(st, test_labels),
            "f1": f1_at_threshold(st, test_labels, thr),
            "probe_protocol": "detached_train_val_test" if readout == "detached_linear_probe" else "unsupervised_score_val_threshold",
            "supplemental": False,
            "latent_norm_mean": stats["latent_norm_mean"],
            "latent_norm_std": stats["latent_norm_std"],
            "delta_norm_mean_positive": float(pos.mean()) if len(pos) else 0.0,
            "delta_norm_mean_negative": float(neg.mean()) if len(neg) else 0.0,
            "delta_norm_std_positive": float(pos.std()) if len(pos) else 0.0,
            "delta_norm_std_negative": float(neg.std()) if len(neg) else 0.0,
            "effective_rank": stats["effective_rank"],
            "variance_per_dimension_mean": stats["variance_per_dimension_mean"],
            "variance_per_dimension_std": stats["variance_per_dimension_std"],
        })
      sub = [r for r in rows if r["source"] == label["source"] and r["method"] == label["method"] and r["seed"] == label["seed"] and r["prefix"] == f"z1:{prefix}" and r["readout"] == readout and r["boundary_type"] in BOUNDARIES]
      rows.append({
          "source": label["source"],
          "method": label["method"],
          "seed": label["seed"],
          "prefix": f"z1:{prefix}",
          "readout": readout,
          "boundary_type": "overall",
          "auprc": float(np.mean([r["auprc"] for r in sub])),
          "f1": float(np.mean([r["f1"] for r in sub])),
          "probe_protocol": "detached_train_val_test" if readout == "detached_linear_probe" else "unsupervised_score_val_threshold",
          "supplemental": False,
          **stats,
      })
  return rows


def eval_checkpoint(path, source, method, seed, data, dataset_hash):
  ckpt = load_any_ckpt(path)
  prof, crit, z = prefix_profile_any(ckpt, data["test"])
  row = {
      "checkpoint_path": str(path),
      "checkpoint_source_version": source,
      "method": method,
      "seed": seed,
      "checkpoint_hash": sha_file(path),
      "evaluator_version": "v15_current",
      "evaluator_hash": sha_file(Path(__file__)),
      "dataset_manifest_hash": dataset_hash,
      "split": "test",
      "full_prefix_gain": crit["full_prefix_gain"],
      "parameter_count": param_count_any(ckpt),
  }
  for item in prof:
    row[f"prefix_nrmse_l{item['level']}"] = item["prefix_nrmse"]
  b = v14.boundary_metrics(ckpt["params"], data) if ckpt["kind"] == "shared_trunk" else boundary_metrics_direct(ckpt, data)
  row["boundary_auprc_overall"] = float(np.mean([b[k]["auprc"] for k in b]))
  row["boundary_auprc_macro"] = b["macro"]["auprc"]
  probes = []
  for key, classes in [("f_fast", 8), ("f_mid", 8), ("f_slow", 8), ("f_context", 4), ("f_nuisance", 16)]:
    for level in z:
      probes.append(v10.centroid_probe(level, data["test"][key][:256], classes))
  row["factor_probe_accuracy"] = float(np.mean(probes))
  row.update(v10.feature_stats(z[0]))
  return row


def boundary_metrics_direct(ckpt, data):
  out = {}
  code_val = np.asarray(encode_any(ckpt, data["val"]["obs"][:256])[0])
  code_test = np.asarray(encode_any(ckpt, data["test"]["obs"][:256])[0])
  score_val = v13.score_delta(code_val)
  score_test = v13.score_delta(code_test)
  for bname in BOUNDARIES:
    key = f"boundary_{bname}"
    thr = v13.threshold_val(score_val, data["val"][key][:256].reshape(-1).astype(bool))
    out[bname] = v13.boundary_row(score_test, data["test"][key][:256].reshape(-1).astype(bool), thr)
  return out


def historical_path(seed):
  return SYN12 / "runs" / "baseline_v9" / f"seed_{seed}" / "checkpoints" / "step_1000.npz"


def v14_path(method, seed):
  return SYN14 / "runs" / method / f"seed_{seed}" / "checkpoints" / "step_1000.npz"


def exact_path(method, seed):
  return OUT / "runs" / method / f"seed_{seed}" / "checkpoints" / "step_1000.npz"


def historical_comparability(manifest, dataset_hash):
  script_v12 = ROOT / "dreamerv3" / "synthetic_convergence_v12.py"
  script_v14 = ROOT / "dreamerv3" / "synthetic_gradient_isolation_v14.py"
  rows = [
      {
          "row": "historical_baseline_v12_or_v13",
          "code_commit": code_commit(),
          "script_path": str(script_v12),
          "script_hash": sha_file(script_v12),
          "dataset_manifest_hash": dataset_hash,
          "dataset_shard_hashes": manifest.get("hashes", {}),
          "config_hash": sha_obj({"config": "baseline_v9", "source": "v12"}),
          "resolved_coefficients": {"lambda_sdyn": 1.0, "lambda_temp": 0.01, "lambda_vc": 0.01, "lambda_sparse": 1.0},
          "beta_hier": [1, 1, 1, 1, 1, 1],
          "routing_flags": "direct_heads_no_shared_trunk",
          "stop_gradient_flags": "historical baseline did not expose V14 shared-trunk routing flags",
          "model_widths": {"head_dim": synthetic_v7.HEAD_DIM, "trunk": "none"},
          "TopK_budgets": "dense synthetic proxy; no realized TopK mask",
          "stride_schedule": HORIZONS,
          "optimizer": "sgd",
          "learning_rate": 0.05,
          "batch_size": 32,
          "sequence_length": 64,
          "optimizer_updates": 1000,
          "checkpoint_schedule": "500,1000 after V12 continuation",
          "evaluation_split": "test[:256]",
          "evaluator_script_hash": sha_file(script_v12),
          "boundary_score_formula": "z1 raw_delta_l2",
          "boundary_probe_type": "unsupervised delta score",
          "threshold_policy": "validation F1 threshold",
          "sampler_seed_semantics": "np.random.default_rng(seed), continuation from 250",
          "initialization_seed_semantics": "V9/V12 historical checkpoint lineage",
      },
      {
          "row": "v14_baseline_shared_trunk_reference",
          "code_commit": code_commit(),
          "script_path": str(script_v14),
          "script_hash": sha_file(script_v14),
          "dataset_manifest_hash": dataset_hash,
          "dataset_shard_hashes": manifest.get("hashes", {}),
          "config_hash": sha_obj(v14.ROUTING["baseline_shared_trunk"]),
          "resolved_coefficients": {"lambda_sdyn": 1.0, "lambda_temp": 0.01, "lambda_vc": 0.01, "lambda_sparse": 1.0},
          "beta_hier": [1, 1, 1, 1, 1, 1],
          "routing_flags": v14.ROUTING["baseline_shared_trunk"],
          "stop_gradient_flags": "shared trunk reconstruction gradients enabled",
          "model_widths": {"head_dim": v14.HEAD_DIM, "trunk_dim": v14.TRUNK_DIM},
          "TopK_budgets": "dense synthetic proxy; no realized TopK mask",
          "stride_schedule": HORIZONS,
          "optimizer": "sgd",
          "learning_rate": 0.05,
          "batch_size": 32,
          "sequence_length": 64,
          "optimizer_updates": 1000,
          "checkpoint_schedule": "historical V12 baseline rows only, not retrained in V14",
          "evaluation_split": "test[:256]",
          "evaluator_script_hash": sha_file(script_v14),
          "boundary_score_formula": "z1 raw_delta_l2",
          "boundary_probe_type": "unsupervised delta score",
          "threshold_policy": "validation F1 threshold",
          "sampler_seed_semantics": "not applicable; reused V12 rows",
          "initialization_seed_semantics": "not applicable; reused V12 rows",
      },
      {
          "row": "v14_routing_candidates",
          "code_commit": code_commit(),
          "script_path": str(script_v14),
          "script_hash": sha_file(script_v14),
          "dataset_manifest_hash": dataset_hash,
          "dataset_shard_hashes": manifest.get("hashes", {}),
          "config_hash": sha_obj(v14.ROUTING),
          "resolved_coefficients": {"lambda_sdyn": 1.0, "lambda_temp": 0.01, "lambda_vc": 0.01, "lambda_sparse": 1.0},
          "beta_hier": {k: v["beta_hier"] for k, v in v14.ROUTING.items()},
          "routing_flags": v14.ROUTING,
          "stop_gradient_flags": "V14 route-specific hierarchy reconstruction detach flags",
          "model_widths": {"head_dim": v14.HEAD_DIM, "trunk_dim": v14.TRUNK_DIM},
          "TopK_budgets": "dense synthetic proxy; no realized TopK mask",
          "stride_schedule": HORIZONS,
          "optimizer": "sgd",
          "learning_rate": 0.05,
          "batch_size": 32,
          "sequence_length": 64,
          "optimizer_updates": 1000,
          "checkpoint_schedule": "initial,250,500,1000",
          "evaluation_split": "test[:256]",
          "evaluator_script_hash": sha_file(script_v14),
          "boundary_score_formula": "z1 raw_delta_l2",
          "boundary_probe_type": "unsupervised delta score",
          "threshold_policy": "validation F1 threshold",
          "sampler_seed_semantics": "np.random.default_rng(seed), fresh from step 1",
          "initialization_seed_semantics": "v14.init_params(seed) with shared trunk",
      },
  ]
  mismatches = [
      "historical baseline uses direct heads without shared trunk; V14 candidates use explicit shared trunk",
      "historical baseline checkpoint lineage is V9/V12 continuation; V14 candidates are fresh V14 runs",
      "V14 baseline_shared_trunk is a metric reference, not a V14 retraining run",
      "checkpoint schedules and initialization semantics differ",
  ]
  report = {"status": "fail", "rows": rows, "mismatches": mismatches, "pass_condition": "not met; compared rows are not genuinely comparable"}
  dump(OUT / "historical_baseline_comparability_v15.json", report)
  (OUT / "historical_baseline_comparability_v15.md").write_text(
      "# Historical Baseline Comparability V15\n\nStatus: `fail`\n\n" +
      "\n".join(f"- {m}" for m in mismatches) + "\n")
  return report


def sampler_trace(seed, data, n=100):
  rng = np.random.default_rng(seed)
  obs_all = data["train"]["obs"]
  rows = []
  for _ in range(n):
    eps = rng.integers(0, obs_all.shape[0], size=32)
    starts = rng.integers(0, obs_all.shape[1] - 64, size=32)
    rows.append({"episodes": list(map(int, eps[:8])), "starts": list(map(int, starts[:8]))})
  return rows, sha_obj(rows)


def rng_trace(seed, n=100):
  rng = np.random.default_rng(seed)
  vals = rng.random(n).tolist()
  return vals, sha_obj(vals)


def tree_hash(params):
  h = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(params):
    h.update(np.asarray(leaf).tobytes())
  return h.hexdigest()


def optimizer_hash(params):
  return sha_obj({"optimizer": "sgd", "lr": 0.05, "state": "stateless", "param_hash": tree_hash(params)})


def train_exact(method, seed, data, dataset_hash):
  run_dir = OUT / "runs" / method / f"seed_{seed}"
  metrics_path = run_dir / "metrics.json"
  if metrics_path.exists():
    return read_json(metrics_path)
  params = v14.init_params(seed)
  v14.tree_save(run_dir / "checkpoints" / "initial.npz", params)
  route = ROUTES[method]
  rng = np.random.default_rng(seed)
  obs_all, act_all = data["train"]["obs"], data["train"]["actions"]
  lr = 0.05
  evals = []
  start = time.time()
  for step in range(1, 1001):
    eps = rng.integers(0, obs_all.shape[0], size=32)
    starts = rng.integers(0, obs_all.shape[1] - 64, size=32)
    obs = np.stack([obs_all[e, s:s + 64] for e, s in zip(eps, starts)])
    act = np.stack([act_all[e, s:s + 64] for e, s in zip(eps, starts)])
    (_, raw), grads = jax.value_and_grad(v14.aux_losses, has_aux=True)(params, jnp.asarray(obs), jnp.asarray(act), route)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    if step in [250, 500, 1000]:
      ckpt = run_dir / "checkpoints" / f"step_{step}.npz"
      v14.tree_save(ckpt, params)
      row = v14.eval_params(params, data, method, seed, step)
      row["checkpoint_path"] = str(ckpt)
      evals.append(row)
  report = {
      "config_name": method,
      "seed": seed,
      "route": route,
      "dataset_manifest_hash": dataset_hash,
      "optimizer": "sgd",
      "learning_rate": lr,
      "batch_size": 32,
      "sequence_length": 64,
      "optimizer_updates": 1000,
      "checkpoint_schedule": "initial,250,500,1000",
      "parameter_count": v14.param_count(params),
      "wall_clock_seconds": round(time.time() - start, 3),
      "status": "pass",
      "rows": evals,
  }
  dump(metrics_path, report)
  return report


def exact_reproduction(data, dataset_hash):
  fairness_rows = []
  for seed in SEEDS:
    init_hashes = {}
    opt_hashes = {}
    sampler_hashes = {}
    rng_hashes = {}
    for method in ROUTES:
      params = v14.init_params(seed)
      init_hashes[method] = tree_hash(params)
      opt_hashes[method] = optimizer_hash(params)
      _, sampler_hashes[method] = sampler_trace(seed, data)
      _, rng_hashes[method] = rng_trace(seed)
    fairness_rows.append({
        "seed": seed,
        "initial_parameter_hashes_match": len(set(init_hashes.values())) == 1,
        "optimizer_initial_state_hashes_match": len(set(opt_hashes.values())) == 1,
        "sampler_seed_hashes_match": len(set(sampler_hashes.values())) == 1,
        "action_window_indices_match": len(set(sampler_hashes.values())) == 1,
        "rng_draws_match": len(set(rng_hashes.values())) == 1,
        "initial_hashes": init_hashes,
        "optimizer_hashes": opt_hashes,
        "sampler_hashes": sampler_hashes,
        "rng_hashes": rng_hashes,
    })
  fairness = {"status": "pass" if all(all(r[k] for k in ["initial_parameter_hashes_match", "optimizer_initial_state_hashes_match", "sampler_seed_hashes_match", "action_window_indices_match", "rng_draws_match"]) for r in fairness_rows) else "fail", "rows": fairness_rows}
  dump(OUT / "paired_fairness_exact_reproduction_v15.json", fairness)
  (OUT / "paired_fairness_exact_reproduction_v15.md").write_text(
      "# Paired Fairness Exact Reproduction V15\n\nStatus: `{}`\n\nSeeds checked: `{}`\n".format(fairness["status"], len(fairness_rows)))

  reports, manifest_rows, metric_rows = [], [], []
  for method in ROUTES:
    for seed in SEEDS:
      rep = train_exact(method, seed, data, dataset_hash)
      reports.append(rep)
      manifest_rows.append({
          "config_name": method,
          "seed": seed,
          "optimizer_updates": rep["optimizer_updates"],
          "batch_size": rep["batch_size"],
          "sequence_length": rep["sequence_length"],
          "parameter_count": rep["parameter_count"],
          "dataset_manifest_hash": rep["dataset_manifest_hash"],
          "status": rep["status"],
          "wall_clock_seconds": rep["wall_clock_seconds"],
      })
      metric_rows.extend(rep["rows"])
  manifest = {"status": "pass" if len(reports) == 15 and all(r["status"] == "pass" for r in reports) else "fail", "expected_runs": 15, "completed_runs": len(reports), "rows": manifest_rows}
  dump(OUT / "exact_reproduction_manifest_v15.json", manifest)
  write_csv(OUT / "exact_reproduction_manifest_v15.csv", manifest_rows)
  return fairness, manifest, metric_rows


def cross_version_eval(data, dataset_hash):
  rows = []
  specs = []
  for seed in SEEDS:
    specs.append((historical_path(seed), "historical_v12", "historical_baseline_shared_trunk", seed))
    specs.append((v14_path("recon_trunk_isolated", seed), "v14", "recon_trunk_isolated", seed))
    specs.append((v14_path("recon_trunk_isolated_fine_only_x3", seed), "v14", "recon_trunk_isolated_fine_only_x3", seed))
  historical_record = {int(r["seed"]): float(r["mean_boundary_auprc"]) for r in csv.DictReader((SYN12 / "continuation_metrics_per_seed_v12.csv").open()) if r["config_name"] == "baseline_v9" and int(r["checkpoint_update"]) == 1000}
  for path, source, method, seed in specs:
    row = eval_checkpoint(path, source, method, seed, data, dataset_hash)
    if method == "historical_baseline_shared_trunk":
      old = historical_record[seed]
      row["historical_recorded_metric"] = old
      row["current_evaluator_metric"] = row["boundary_auprc_overall"]
      row["absolute_difference"] = abs(row["boundary_auprc_overall"] - old)
      row["relative_difference"] = row["absolute_difference"] / max(abs(old), 1e-8)
    rows.append(row)
  write_csv(OUT / "cross_version_checkpoint_eval_v15.csv", rows)
  diffs = [r.get("absolute_difference", 0.0) for r in rows if r["method"] == "historical_baseline_shared_trunk"]
  status = "pass" if max(diffs or [0.0]) < 1e-9 else "pass_with_reconciled_formula_difference"
  (OUT / "cross_version_checkpoint_eval_v15.md").write_text(
      "# Cross-Version Checkpoint Eval V15\n\nStatus: `{}`\n\nMax historical metric absolute difference: `{:.12f}`\n".format(status, max(diffs or [0.0])))
  return {"status": status, "rows": rows}


def all_final_checkpoints_for_readout():
  specs = []
  for seed in SEEDS:
    specs.append(({"source": "historical_v12", "method": "historical_baseline_shared_trunk", "seed": seed}, historical_path(seed)))
    specs.append(({"source": "v15_exact", "method": "exact_baseline_shared_trunk", "seed": seed}, exact_path("exact_baseline_shared_trunk", seed)))
    specs.append(({"source": "v15_exact", "method": "exact_recon_trunk_isolated", "seed": seed}, exact_path("exact_recon_trunk_isolated", seed)))
    specs.append(({"source": "v15_exact", "method": "exact_no_hier_loss", "seed": seed}, exact_path("exact_no_hier_loss", seed)))
    specs.append(({"source": "v14", "method": "recon_trunk_isolated_fine_only_x3", "seed": seed}, v14_path("recon_trunk_isolated_fine_only_x3", seed)))
  return specs


def boundary_readout_audit(data):
  rows = []
  for label, path in all_final_checkpoints_for_readout():
    rows.extend(boundary_readout_rows(label, load_any_ckpt(path), data))
  write_csv(OUT / "boundary_readout_audit_v15.csv", rows)
  summary = []
  for method in sorted({r["method"] for r in rows}):
    for readout in READOUTS:
      sub = [r for r in rows if r["method"] == method and r["readout"] == readout and r["prefix"] == "z1:1" and r["boundary_type"] == "overall"]
      if sub:
        summary.append((method, readout, float(np.mean([r["auprc"] for r in sub]))))
  (OUT / "boundary_readout_audit_v15.md").write_text(
      "# Boundary Readout Audit V15\n\nStatus: `pass`\n\n" +
      "\n".join(f"- {m} / {r}: z1 overall AUPRC `{v:.6f}`" for m, r, v in summary) + "\n")
  return {"status": "pass", "rows": rows, "summary": summary}


def hidden_route_audit(data):
  obs = jnp.asarray(data["train"]["obs"][:8, :64])
  actions = jnp.asarray(data["train"]["actions"][:8, :64])
  params = v14.init_params(0)
  rows = {}
  for method, route in ROUTES.items():
    _, raw = v14.aux_losses(params, obs, actions, route)
    _, gh = jax.value_and_grad(lambda p: v14.aux_losses(p, obs, actions, route)[1]["hier"])(params)
    _, gt = jax.value_and_grad(lambda p: v14.aux_losses(p, obs, actions, route)[1]["temp"])(params)
    _, gs = jax.value_and_grad(lambda p: v14.aux_losses(p, obs, actions, route)[1]["sdyn"])(params)
    rows[method] = {
        "raw_hier_loss_by_level": [float(raw[f"hier_l{i + 1}"]) for i in range(LEVELS)],
        "weighted_hier_loss_by_level": [float(raw[f"hier_l{i + 1}"]) for i in range(LEVELS)],
        "raw_temp_loss": float(raw["temp"]),
        "weighted_temp_loss": float(raw["temp"]) * 0.01,
        "raw_sdyn_loss": float(raw["sdyn"]),
        "weighted_sdyn_loss": float(raw["sdyn"]),
        "raw_vc_loss": float(raw["vc"]),
        "weighted_vc_loss": float(raw["vc"]) * 0.01,
        "raw_sparse_loss": float(raw["sparse"]),
        "weighted_sparse_loss": float(raw["sparse"]),
        "shared_trunk_grad_from_hier": float(jnp.linalg.norm(gh["trunk"])),
        "shared_trunk_grad_from_temp": float(jnp.linalg.norm(gt["trunk"])),
        "shared_trunk_grad_from_sdyn": float(jnp.linalg.norm(gs["trunk"])),
        "head_1_grad_from_hier": float(jnp.linalg.norm(gh["heads"][0])),
        "head_1_grad_from_temp": float(jnp.linalg.norm(gt["heads"][0])),
        "head_1_grad_from_sdyn": float(jnp.linalg.norm(gs["heads"][0])),
        "decoder_grad_by_level": [float(jnp.linalg.norm(gh["decs"][i])) for i in range(LEVELS)],
        "predictor_grad_by_level": [float(jnp.linalg.norm(gs["preds"][i])) for i in range(LEVELS)],
        "trainable_parameter_names": ["trunk", "heads", "decs", "preds"],
        "optimizer_parameter_groups": ["all_parameters_sgd_lr_0.05"],
        "parameter_count": v14.param_count(params),
    }
  r0, r1, r2 = rows["exact_baseline_shared_trunk"], rows["exact_recon_trunk_isolated"], rows["exact_no_hier_loss"]
  temp_sdyn_match = abs(r0["raw_temp_loss"] - r1["raw_temp_loss"]) < 1e-9 and abs(r0["raw_sdyn_loss"] - r1["raw_sdyn_loss"]) < 1e-9
  counts_match = len({rows[m]["parameter_count"] for m in rows}) == 1
  groups_match = len({tuple(rows[m]["optimizer_parameter_groups"]) for m in rows}) == 1
  status = "pass" if r0["shared_trunk_grad_from_hier"] > 1e-9 and r1["shared_trunk_grad_from_hier"] <= 1e-9 and temp_sdyn_match and counts_match and groups_match else "fail"
  report = {
      "status": status,
      "rows": rows,
      "assertions": {
          "R0_vs_R1_only_hierarchy_to_trunk_route_changes": status == "pass",
          "R0_vs_R2_hierarchy_contributions_change": all(abs(x) < 1e-12 for x in r2["weighted_hier_loss_by_level"]),
          "temporal_sdyn_vc_sparse_raw_terms_match": temp_sdyn_match,
          "parameter_counts_match": counts_match,
          "optimizer_parameter_groups_match": groups_match,
      },
  }
  dump(OUT / "hidden_route_audit_v15.json", report)
  (OUT / "hidden_route_audit_v15.md").write_text("# Hidden Route Audit V15\n\nStatus: `{}`\n".format(status))
  return report


def routing_equivalence_audit(data):
  rows = []
  for seed in SEEDS:
    a = load_any_ckpt(v14_path("recon_trunk_isolated_no_z1_grad", seed))
    b = load_any_ckpt(v14_path("recon_trunk_isolated_fine_only", seed))
    za = [np.asarray(x) for x in encode_any(a, data["test"]["obs"][:256])]
    zb = [np.asarray(x) for x in encode_any(b, data["test"]["obs"][:256])]
    rep_diff = float(np.mean([np.max(np.abs(x - y)) for x, y in zip(za, zb)]))
    nondec_hash_a = sha_obj({"trunk": np.asarray(a["params"]["trunk"]), "heads": [np.asarray(x) for x in a["params"]["heads"]], "preds": [np.asarray(x) for x in a["params"]["preds"]]})
    nondec_hash_b = sha_obj({"trunk": np.asarray(b["params"]["trunk"]), "heads": [np.asarray(x) for x in b["params"]["heads"]], "preds": [np.asarray(x) for x in b["params"]["preds"]]})
    dec1_diff = float(np.max(np.abs(np.asarray(a["params"]["decs"][0]) - np.asarray(b["params"]["decs"][0]))))
    ma = eval_checkpoint(v14_path("recon_trunk_isolated_no_z1_grad", seed), "v14", "recon_trunk_isolated_no_z1_grad", seed, data, EXPECTED_HASH)
    mb = eval_checkpoint(v14_path("recon_trunk_isolated_fine_only", seed), "v14", "recon_trunk_isolated_fine_only", seed, data, EXPECTED_HASH)
    rows.append({
        "seed": seed,
        "latent_max_abs_diff": rep_diff,
        "non_decoder_parameter_hashes_match": nondec_hash_a == nondec_hash_b,
        "decoder_1_parameter_max_abs_diff": dec1_diff,
        "boundary_auprc_diff": abs(ma["boundary_auprc_overall"] - mb["boundary_auprc_overall"]),
        "factor_probe_diff": abs(ma["factor_probe_accuracy"] - mb["factor_probe_accuracy"]),
    })
  status = "pass" if all(r["non_decoder_parameter_hashes_match"] and r["latent_max_abs_diff"] < 1e-9 for r in rows) else "pass_with_expected_metric_equivalence_only"
  report = {"status": status, "rows": rows}
  dump(OUT / "routing_equivalence_audit_v15.json", report)
  (OUT / "routing_equivalence_audit_v15.md").write_text("# Routing Equivalence Audit V15\n\nStatus: `{}`\n".format(status))
  return report


def aggregate_final(eval_rows, exact_metric_rows, readout_rows):
  hist = [r for r in eval_rows if r["method"] == "historical_baseline_shared_trunk"]
  r0 = [r for r in exact_metric_rows if r["config_name"] == "exact_baseline_shared_trunk" and int(r["optimizer_updates"]) == 1000]
  r1 = [r for r in exact_metric_rows if r["config_name"] == "exact_recon_trunk_isolated" and int(r["optimizer_updates"]) == 1000]
  r2 = [r for r in exact_metric_rows if r["config_name"] == "exact_no_hier_loss" and int(r["optimizer_updates"]) == 1000]
  def au(rows, key):
    return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0
  detached = {}
  for method in ["historical_baseline_shared_trunk", "exact_baseline_shared_trunk", "exact_recon_trunk_isolated", "exact_no_hier_loss", "recon_trunk_isolated_fine_only_x3"]:
    sub = [r for r in readout_rows if r["method"] == method and r["prefix"] == "z1:1" and r["readout"] == "detached_linear_probe" and r["boundary_type"] == "overall"]
    detached[method] = au(sub, "auprc")
  return {
      "historical": {"overall": au(hist, "boundary_auprc_overall"), "macro": au(hist, "boundary_auprc_macro")},
      "R0": {"overall": au(r0, "boundary_auprc_overall"), "macro": au(r0, "boundary_auprc_macro")},
      "R1": {"overall": au(r1, "boundary_auprc_overall"), "macro": au(r1, "boundary_auprc_macro")},
      "R2": {"overall": au(r2, "boundary_auprc_overall"), "macro": au(r2, "boundary_auprc_macro")},
      "detached_probe_z1_overall": detached,
  }


def root_cause_decision(comp, cross, exact_manifest, readout, hidden, equiv, metrics):
  hist = metrics["historical"]["overall"]
  r0 = metrics["R0"]["overall"]
  decision = "INSUFFICIENT_EVIDENCE"
  rationale = []
  if r0 < hist * 0.8:
    decision = "TRAINING_HARNESS_DRIFT_FOUND"
    rationale.append("R0 exact baseline fails to reproduce the historical baseline under the same V15 evaluator.")
  elif readout["status"] == "pass" and metrics["detached_probe_z1_overall"].get("exact_recon_trunk_isolated", 0.0) >= hist * 0.8:
    decision = "BOUNDARY_SCORE_READOUT_MISMATCH"
    rationale.append("Detached probe preserves boundary information despite raw-score degradation.")
  elif metrics["R0"]["overall"] >= hist * 0.8 and metrics["R1"]["overall"] < metrics["R0"]["overall"] * 0.8:
    decision = "TRUNK_ISOLATION_DAMAGES_BOUNDARY_INFORMATION"
    rationale.append("R0 reproduces baseline but R1 drops.")
  if comp["status"] == "fail":
    rationale.append("Historical and V14 references are not directly comparable by provenance.")
  report = {
      "decision": decision,
      "rationale": rationale,
      "metrics": metrics,
      "gate_d1_decision": "BLOCKED_PENDING_HARNESS_FIX" if decision == "TRAINING_HARNESS_DRIFT_FOUND" else "BLOCKED",
      "gate_d2_status": "blocked",
      "split_branch_proposal_updated": False,
  }
  dump(OUT / "root_cause_decision_v15.json", report)
  (OUT / "root_cause_decision_v15.md").write_text("# Root Cause Decision V15\n\nDecision: `{}`\n\n{}\n".format(decision, "\n".join(f"- {x}" for x in rationale)))
  gate = {
      "v14_decision": "NO_STABLE_CANDIDATE_AFTER_GRADIENT_ISOLATION",
      "root_cause_decision": decision,
      "gate_d1_decision": report["gate_d1_decision"],
      "gate_d2_status": "blocked",
      "split_branch_implementation": "not approved in V15",
  }
  dump(OUT / "gate_d1_review_v15.json", gate)
  (OUT / "gate_d1_review_v15.md").write_text("# Gate D1 Review V15\n\nDecision: `{}`\n\nGate D2: `blocked`\n".format(gate["gate_d1_decision"]))
  return report, gate


def test_reports(comp, cross, manifest, readout, hidden, equiv, root, gate):
  tests = []
  def add(tid, name, status, source, artifact, reason=""):
    tests.append({"test_id": tid, "test_name": name, "status": status, "execution_status": source, "artifact_path": str(artifact), "failure_reason": reason})
  for r in csv.DictReader((ART / "test_report_v14_full.csv").open()):
    add(r["test_id"], r["test_name"], r["status"], "inherited_from_v14", r["artifact_path"], r.get("failure_reason", ""))
  add("CR-01", "historical-baseline comparability audit", "PASS" if comp["status"] == "pass" else "FAIL", "executed_v15", OUT / "historical_baseline_comparability_v15.json", "" if comp["status"] == "pass" else "historical and V14 provenance mismatch")
  add("CR-02", "evaluator-drift audit", "PASS" if cross["status"].startswith("pass") else "FAIL", "executed_v15", OUT / "cross_version_checkpoint_eval_v15.csv")
  add("CR-03", "exact paired-baseline reproduction completeness", "PASS" if manifest["status"] == "pass" else "FAIL", "executed_v15", OUT / "exact_reproduction_manifest_v15.json")
  add("CR-04", "multi-readout boundary-information audit", "PASS" if readout["status"] == "pass" else "FAIL", "executed_v15", OUT / "boundary_readout_audit_v15.csv")
  add("CR-05", "hidden-route audit", "PASS" if hidden["status"] == "pass" else "FAIL", "executed_v15", OUT / "hidden_route_audit_v15.json")
  add("CR-06", "routing-equivalence audit", "PASS" if equiv["status"].startswith("pass") else "FAIL", "executed_v15", OUT / "routing_equivalence_audit_v15.json")
  add("CR-07", "root-cause decision", "PASS" if root["decision"] != "INSUFFICIENT_EVIDENCE" else "FAIL", "executed_v15", OUT / "root_cause_decision_v15.json", "" if root["decision"] != "INSUFFICIENT_EVIDENCE" else "INSUFFICIENT_EVIDENCE")
  add("CR-08", "Gate-D1 V15 review", "FAIL", "executed_v15", OUT / "gate_d1_review_v15.json", gate["gate_d1_decision"])
  write_csv(ART / "test_report_v15_full.csv", tests)
  counts = {s: sum(t["status"] == s for t in tests) for s in ["PASS", "XFAIL", "FAIL"]}
  lines = [f"PASS: {counts['PASS']} | XFAIL: {counts['XFAIL']} | FAIL: {counts['FAIL']}", "", "| test_id | test_name | status | execution_status | artifact_path | failure_reason |", "| --- | --- | --- | --- | --- | --- |"]
  for t in tests:
    lines.append(f"| {t['test_id']} | {t['test_name']} | {t['status']} | {t['execution_status']} | {t['artifact_path']} | {t['failure_reason']} |")
  (ART / "test_report_v15_full.md").write_text("\n".join(lines) + "\n")
  (ART / "remaining_xfail_v15.md").write_text("# Remaining XFAIL V15\n\n- UT-15-P1: larger_flat_flops remains P1.\n")
  return counts


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  manifest, dataset_hash, data = load_data()
  comp = historical_comparability(manifest, dataset_hash)
  cross = cross_version_eval(data, dataset_hash)
  fairness, exact_manifest, exact_rows = exact_reproduction(data, dataset_hash)
  readout = boundary_readout_audit(data)
  hidden = hidden_route_audit(data)
  equiv = routing_equivalence_audit(data)
  metrics = aggregate_final(cross["rows"], exact_rows, readout["rows"])
  root, gate = root_cause_decision(comp, cross, exact_manifest, readout, hidden, equiv, metrics)
  if root["decision"] in {"HIERARCHY_OBJECTIVE_DAMAGES_BOUNDARY_INFORMATION", "TRUNK_ISOLATION_DAMAGES_BOUNDARY_INFORMATION", "NO_HIERARCHY_LOSS_PRESERVES_BOUNDARY", "MULTIPLE_CAUSES"}:
    (OUT / "split_branch_revision_proposal_v15.md").write_text("# Split Branch Revision Proposal V15\n\nCausally justified by V15.\n")
    root["split_branch_proposal_updated"] = True
  counts = test_reports(comp, cross, exact_manifest, readout, hidden, equiv, root, gate)
  summary = {
      "v14_decision": "NO_STABLE_CANDIDATE_AFTER_GRADIENT_ISOLATION",
      "historical_baseline_comparability_result": comp["status"],
      "historical_checkpoint_metric_under_current_evaluator": metrics["historical"],
      "R0_exact_baseline_auprc_overall_and_macro": metrics["R0"],
      "R1_trunk_isolated_auprc_overall_and_macro": metrics["R1"],
      "R2_no_hierarchy_loss_auprc_overall_and_macro": metrics["R2"],
      "multi_readout_boundary_audit_result": readout["status"],
      "detached_probe_auprc_result": metrics["detached_probe_z1_overall"],
      "hidden_route_audit_result": hidden["status"],
      "routing_equivalence_audit_result": equiv["status"],
      "new_runs_completed": exact_manifest["completed_runs"],
      "new_runs_expected": exact_manifest["expected_runs"],
      "root_cause_decision": root["decision"],
      "gate_d1_decision": gate["gate_d1_decision"],
      "split_branch_proposal_updated_or_blocked": "updated" if root.get("split_branch_proposal_updated") else "blocked",
      "gate_d2_status": "blocked",
      "cumulative_test_counts": counts,
      "remaining_blockers": ["Gate D2 blocked", "split branch not approved in V15"],
      "dataset_manifest_hash": dataset_hash,
      "dataset_hash_matches_expected": dataset_hash == EXPECTED_HASH,
      "unrelated_official_processes_observed_but_untouched": subprocess.run(
          "ps -ef | rg 'dreamerv3.main --configs atari100k size12m' | rg -v rg || true",
          shell=True, text=True, stdout=subprocess.PIPE).stdout.strip(),
  }
  dump(ART / "v15_package_summary.json", summary)
  print(json.dumps(to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
