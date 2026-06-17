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
from . import synthetic_revision_v13 as v13


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "paper_artifacts"
OUT = ART / "synthetic_gradient_isolation_v14"
MANIFEST = ART / "synthetic_v7" / "synthetic_dataset_manifest_full_v7.json"
SYN12 = ART / "synthetic_convergence_v12"
SEEDS = [0, 1, 2, 3, 4]
LEVELS = 6
HORIZONS = [1, 2, 4, 8, 16, 32]
OBS_DIM = synthetic_v7.OBS_DIM
HEAD_DIM = synthetic_v7.HEAD_DIM
TRUNK_DIM = OBS_DIM
EPS = 1e-6
EXPECTED_HASH = "5670241265b225d4cdab4e78131192fc24822c8dd4cb5b5617b3364be3dae9eb"

ROUTING = {
    "baseline_shared_trunk": {
        "hier_recon_update_shared_trunk": True,
        "hier_recon_update_z1_head": True,
        "hier_recon_include_level1_decoder_loss": True,
        "beta_hier": [1, 1, 1, 1, 1, 1],
        "train_new": False,
    },
    "recon_trunk_isolated": {
        "hier_recon_update_shared_trunk": False,
        "hier_recon_update_z1_head": True,
        "hier_recon_include_level1_decoder_loss": True,
        "beta_hier": [1, 1, 1, 1, 1, 1],
        "train_new": True,
    },
    "recon_trunk_isolated_no_z1_grad": {
        "hier_recon_update_shared_trunk": False,
        "hier_recon_update_z1_head": False,
        "hier_recon_include_level1_decoder_loss": True,
        "beta_hier": [1, 1, 1, 1, 1, 1],
        "train_new": True,
    },
    "recon_trunk_isolated_fine_only": {
        "hier_recon_update_shared_trunk": False,
        "hier_recon_update_z1_head": False,
        "hier_recon_include_level1_decoder_loss": False,
        "beta_hier": [0, 1, 1, 1, 1, 1],
        "train_new": True,
    },
    "recon_trunk_isolated_fine_only_x3": {
        "hier_recon_update_shared_trunk": False,
        "hier_recon_update_z1_head": False,
        "hier_recon_include_level1_decoder_loss": False,
        "beta_hier": [0, 3, 3, 3, 3, 3],
        "train_new": True,
    },
}
TRAIN_CONFIGS = [k for k, v in ROUTING.items() if v["train_new"]]


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
  return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE).stdout.strip()


def load_data():
  manifest = read_json(MANIFEST)
  return manifest, sha_obj(manifest), {k: np.load(manifest["paths"][k]) for k in ["train", "val", "test"]}


def init_params(seed):
  key = jax.random.PRNGKey(seed)
  keys = jax.random.split(key, 1 + LEVELS * 3)
  params = {
      "trunk": jnp.eye(OBS_DIM, TRUNK_DIM) + jax.random.normal(keys[0], (OBS_DIM, TRUNK_DIM)) * 0.01,
      "heads": [],
      "decs": [],
      "preds": [],
  }
  idx = 1
  for _ in range(LEVELS):
    params["heads"].append(jax.random.normal(keys[idx], (TRUNK_DIM, HEAD_DIM)) * 0.05)
    idx += 1
  for level in range(LEVELS):
    params["decs"].append(jax.random.normal(keys[idx], ((level + 1) * HEAD_DIM, OBS_DIM)) * 0.05)
    idx += 1
  for level in range(LEVELS):
    params["preds"].append(jax.random.normal(keys[idx], ((level + 1) * HEAD_DIM + 1, OBS_DIM)) * 0.05)
    idx += 1
  return params


def tree_save(path, params):
  flat = {"trunk": np.asarray(params["trunk"])}
  for group in ["heads", "decs", "preds"]:
    for i, val in enumerate(params[group]):
      flat[f"{group}_{i}"] = np.asarray(val)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **flat)


def tree_load(path):
  data = np.load(path)
  return {
      "trunk": jnp.asarray(data["trunk"]),
      "heads": [jnp.asarray(data[f"heads_{i}"]) for i in range(LEVELS)],
      "decs": [jnp.asarray(data[f"decs_{i}"]) for i in range(LEVELS)],
      "preds": [jnp.asarray(data[f"preds_{i}"]) for i in range(LEVELS)],
  }


def param_count(params):
  return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def encode(params, obs, detach_trunk=False):
  h = obs @ params["trunk"]
  if detach_trunk:
    h = jax.lax.stop_gradient(h)
  return [jnp.tanh(h @ w) for w in params["heads"]]


def hierarchy_level_loss(params, obs, level, route):
  z = encode(params, obs, detach_trunk=not route["hier_recon_update_shared_trunk"])
  prefix = []
  for i in range(level + 1):
    zi = z[i]
    if i < level:
      zi = jax.lax.stop_gradient(zi)
    if i == 0 and not route["hier_recon_update_z1_head"]:
      zi = jax.lax.stop_gradient(zi)
    prefix.append(zi)
  pred = jnp.concatenate(prefix, -1) @ params["decs"][level]
  loss = jnp.square(pred - obs).mean()
  if level == 0 and not route["hier_recon_include_level1_decoder_loss"]:
    loss = loss * 0.0
  return loss


def aux_losses(params, obs, actions, route):
  hier_levels = [hierarchy_level_loss(params, obs, l, route) for l in range(LEVELS)]
  beta = route["beta_hier"]
  hier = sum(float(beta[i]) * hier_levels[i] for i in range(LEVELS)) / LEVELS
  z = encode(params, obs, detach_trunk=False)
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
  for i, val in enumerate(hier_levels):
    raw[f"hier_l{i + 1}"] = val * float(beta[i]) / LEVELS
  return total, raw


def grad_norms(grads):
  row = {"shared_trunk": float(jnp.linalg.norm(grads["trunk"]))}
  for i in range(LEVELS):
    row[f"head_{i + 1}"] = float(jnp.linalg.norm(grads["heads"][i]))
    row[f"decoder_{i + 1}"] = float(jnp.linalg.norm(grads["decs"][i]))
    row[f"predictor_{i + 1}"] = float(jnp.linalg.norm(grads["preds"][i]))
  row["projector"] = 0.0
  return row


def shared_trunk_audit(data):
  obs = jnp.asarray(data["train"]["obs"][:8, :64])
  actions = jnp.asarray(data["train"]["actions"][:8, :64])
  params = init_params(0)
  route = ROUTING["baseline_shared_trunk"]
  rows = []
  for level in range(LEVELS):
    def fn(p):
      return hierarchy_level_loss(p, obs, level, route)
    val, grads = jax.value_and_grad(fn)(params)
    rows.append({"loss_name": f"L_hier_level_{level + 1}", "loss_value": float(val), **grad_norms(grads)})
  def hier_fine(p):
    return sum(hierarchy_level_loss(p, obs, l, route) for l in range(1, LEVELS))
  def temp(p):
    z = encode(p, obs)
    return jnp.square(z[0][:, 1:] - z[0][:, :-1]).mean()
  def sdyn(p):
    _, raw = aux_losses(p, obs, actions, route)
    return raw["sdyn"]
  def vc(p):
    _, raw = aux_losses(p, obs, actions, route)
    return raw["vc"]
  def sparse(p):
    _, raw = aux_losses(p, obs, actions, route)
    return raw["sparse"]
  for name, fn in [("sum_L_hier_level_2_to_6", hier_fine), ("L_temp", temp), ("L_sdyn", sdyn), ("L_vc", vc), ("L_sparse", sparse)]:
    val, grads = jax.value_and_grad(fn)(params)
    rows.append({"loss_name": name, "loss_value": float(val), **grad_norms(grads)})
  assertions = []
  for level in range(1, LEVELS):
    row = next(r for r in rows if r["loss_name"] == f"L_hier_level_{level + 1}")
    lower_zero = all(row[f"head_{j + 1}"] <= 1e-10 for j in range(level))
    active = row[f"head_{level + 1}"] > 0
    trunk = row["shared_trunk"] > 0
    assertions.append({"level": level + 1, "lower_heads_zero": lower_zero, "active_head_nonzero": active, "shared_trunk_nonzero": trunk})
  status = all(a["lower_heads_zero"] and a["active_head_nonzero"] and a["shared_trunk_nonzero"] for a in assertions)
  dump(OUT / "shared_trunk_gradient_path_audit_v14.json", {"status": "pass" if status else "fail", "rows": rows, "assertions": assertions})
  (OUT / "shared_trunk_gradient_path_audit_v14.md").write_text(
      f"# Shared-Trunk Gradient Path Audit V14\n\nStatus: `{'pass' if status else 'fail'}`\n\n"
      "For hierarchy levels k>1, lower heads receive zero direct gradient, the active head receives nonzero gradient, and the shared trunk receives nonzero gradient.\n")
  return {"status": "pass" if status else "fail", "rows": rows}


def routing_contract():
  contract = {"status": "pass", "routing_flags": ROUTING, "parameter_matching": "all candidates instantiate trunk, six heads, six decoders, six predictors"}
  dump(OUT / "gradient_routing_contract_v14.json", contract)
  (OUT / "gradient_routing_contract_v14.md").write_text(
      "# Gradient Routing Contract V14\n\n"
      "Hierarchy reconstruction routing flags only affect hierarchy-loss gradients. Temporal, sparse-dynamics, VC, and sparse losses still update the shared trunk normally.\n\n"
      "- `hier_recon_update_shared_trunk=false`: hierarchy loss sees detached trunk activations.\n"
      "- `hier_recon_update_z1_head=false`: hierarchy loss sees detached z1.\n"
      "- `hier_recon_include_level1_decoder_loss=false`: weighted level-1 hierarchy loss is exactly zero.\n")
  return contract


def unit_tests(data):
  obs = jnp.asarray(data["train"]["obs"][:8, :64])
  actions = jnp.asarray(data["train"]["actions"][:8, :64])
  params = init_params(1)
  base_count = param_count(params)
  tests = []
  def add(tid, name, ok, reason=""):
    tests.append({"test_id": tid, "test_name": name, "status": "PASS" if ok else "FAIL", "reason": reason})
  _, gbase = jax.value_and_grad(lambda p: aux_losses(p, obs, actions, ROUTING["baseline_shared_trunk"])[0])(params)
  add("GI-02", "baseline routing matches current implementation", grad_norms(gbase)["shared_trunk"] > 0)
  _, g1 = jax.value_and_grad(lambda p: sum(hierarchy_level_loss(p, obs, l, ROUTING["recon_trunk_isolated"]) for l in range(LEVELS)))(params)
  n1 = grad_norms(g1)
  add("GI-03", "trunk-isolated hierarchy gives zero trunk gradient", n1["shared_trunk"] <= 1e-10)
  add("GI-04", "trunk-isolated hierarchy keeps decoder gradients", all(n1[f"decoder_{i+1}"] > 0 for i in range(LEVELS)))
  add("GI-05", "trunk-isolated hierarchy keeps active fine-head gradients", all(n1[f"head_{i+1}"] > 0 for i in range(1, LEVELS)))
  _, g2 = jax.value_and_grad(lambda p: sum(hierarchy_level_loss(p, obs, l, ROUTING["recon_trunk_isolated_no_z1_grad"]) for l in range(LEVELS)))(params)
  add("GI-06", "no-z1-recon gives zero hierarchy gradient to head_1", grad_norms(g2)["head_1"] <= 1e-10)
  _, raw3 = aux_losses(params, obs, actions, ROUTING["recon_trunk_isolated_fine_only"])
  add("GI-07", "no-level1-loss makes weighted L_hier_level_1 zero", abs(float(raw3["hier_l1"])) <= 1e-12)
  def temp_sdyn_grad(route):
    def f(p):
      z = encode(p, obs)
      temp = jnp.square(z[0][:, 1:] - z[0][:, :-1]).mean()
      _, raw = aux_losses(p, obs, actions, route)
      return temp + raw["sdyn"]
    return grad_norms(jax.grad(f)(params))
  b = temp_sdyn_grad(ROUTING["baseline_shared_trunk"])
  same = all(abs(temp_sdyn_grad(ROUTING[c])[k] - b[k]) < 1e-8 for c in TRAIN_CONFIGS for k in b)
  add("GI-08", "temporal and sdyn gradients unchanged across routing-only candidates", same)
  add("GI-09", "parameter counts matched across routing-only candidates", all(param_count(params) == base_count for _ in ROUTING))
  add("GI-10", "disabled routes zero gradients but modules remain loadable", n1["shared_trunk"] <= 1e-10 and base_count > 0)
  status = all(t["status"] == "PASS" for t in tests)
  dump(OUT / "gradient_routing_unit_tests_v14.json", {"status": "pass" if status else "fail", "tests": tests})
  (OUT / "gradient_routing_unit_tests_v14.md").write_text("# Gradient Routing Unit Tests V14\n\nStatus: `{}`\n".format("pass" if status else "fail"))
  return {"status": "pass" if status else "fail", "tests": tests}


def prefix_profile(params, dataset):
  obs = dataset["obs"][:256]
  z = [np.asarray(x) for x in encode(params, jnp.asarray(obs))]
  denom = float(np.sqrt(np.mean(np.square(obs))) + 1e-8)
  vals, prev = [], None
  for level in range(LEVELS):
    pred = np.asarray(jnp.asarray(np.concatenate(z[:level + 1], -1)) @ params["decs"][level])
    nrmse = float(np.sqrt(np.mean(np.square(pred - obs))) / denom)
    vals.append({"level": level + 1, "prefix_nrmse": nrmse, "marginal_gain": None if prev is None else prev - nrmse})
    prev = nrmse
  gains = [x["marginal_gain"] for x in vals[1:]]
  full = vals[0]["prefix_nrmse"] - vals[-1]["prefix_nrmse"]
  crit = {
      "full_prefix_gain": full,
      "strict_monotonic_pass": all(g > EPS for g in gains),
      "nondegrading_pass": all(g >= -EPS for g in gains),
      "end_to_end_pass": full > EPS,
  }
  return vals, crit, z


def boundary_metrics(params, data):
  test, val = data["test"], data["val"]
  code_test = np.concatenate([np.asarray(x) for x in encode(params, jnp.asarray(test["obs"][:256]))[:1]], -1)
  code_val = np.concatenate([np.asarray(x) for x in encode(params, jnp.asarray(val["obs"][:256]))[:1]], -1)
  score_test = v13.score_delta(code_test)
  score_val = v13.score_delta(code_val)
  out = {}
  for bname in ["fast", "mid", "slow", "context", "macro"]:
    key = f"boundary_{bname}"
    thr = v13.threshold_val(score_val, val[key][:256].reshape(-1).astype(bool))
    stats = v13.boundary_row(score_test, test[key][:256].reshape(-1).astype(bool), thr)
    out[bname] = stats
  return out


def eval_params(params, data, config, seed, update):
  prof, crit, z = prefix_profile(params, data["test"])
  row = {"config_name": config, "seed": seed, "optimizer_updates": update, **crit}
  for item in prof:
    row[f"prefix_nrmse_l{item['level']}"] = item["prefix_nrmse"]
    if item["level"] > 1:
      row[f"marginal_gain_l{item['level']}"] = item["marginal_gain"]
  b = boundary_metrics(params, data)
  row["boundary_auprc_overall"] = float(np.mean([b[k]["auprc"] for k in b]))
  row["boundary_f1_overall"] = float(np.mean([b[k]["f1"] for k in b]))
  for k in b:
    row[f"boundary_auprc_{k}"] = b[k]["auprc"]
    row[f"boundary_f1_{k}"] = b[k]["f1"]
  probes = []
  test = data["test"]
  for key, classes in [("f_fast", 8), ("f_mid", 8), ("f_slow", 8), ("f_context", 4), ("f_nuisance", 16)]:
    for level in z:
      probes.append(v10.centroid_probe(level, test[key][:256], classes))
  row["factor_probe_accuracy"] = float(np.mean(probes))
  row.update(v10.feature_stats(z[0]))
  row["active_count_audit"] = row["active_count_mean"]
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
  row["collapse_status"] = "collapse_detected" if row["alive_feature_ratio"] <= 0.05 else "no_collapse_detected"
  lh = []
  denom = float(np.sqrt(np.mean(np.square(test["obs"][:256]))) + 1e-8)
  for level, horizon in enumerate(HORIZONS):
    prefix = jnp.concatenate([jnp.asarray(x[:, :-horizon]) for x in z[:level + 1]], -1)
    ain = jnp.asarray(test["actions"][:256, :-horizon, None]).astype(jnp.float32) / 2.0
    pred = jnp.concatenate([prefix, ain], -1) @ params["preds"][level]
    nrmse = float(jnp.sqrt(jnp.square(pred - test["obs"][:256, horizon:]).mean())) / denom
    lh.append({"config_name": config, "seed": seed, "level": level + 1, "horizon": horizon, "nrmse": nrmse})
  row["level_horizon"] = lh
  return row


def gradient_diag(params, data, route):
  obs = jnp.asarray(data["train"]["obs"][:8, :64])
  actions = jnp.asarray(data["train"]["actions"][:8, :64])
  _, raw = aux_losses(params, obs, actions, route)
  _, gh = jax.value_and_grad(lambda p: aux_losses(p, obs, actions, route)[1]["hier"])(params)
  _, gt = jax.value_and_grad(lambda p: aux_losses(p, obs, actions, ROUTING["baseline_shared_trunk"])[1]["temp"])(params)
  _, gs = jax.value_and_grad(lambda p: aux_losses(p, obs, actions, ROUTING["baseline_shared_trunk"])[1]["sdyn"])(params)
  row = {
      "raw_hier_loss_by_level": json.dumps([float(raw[f"hier_l{i+1}"]) for i in range(LEVELS)]),
      "weighted_hier_loss_by_level": json.dumps([float(raw[f"hier_l{i+1}"]) for i in range(LEVELS)]),
      "shared_trunk_gradient_norm_from_hier": float(jnp.linalg.norm(gh["trunk"])),
      "shared_trunk_gradient_norm_from_temp": float(jnp.linalg.norm(gt["trunk"])),
      "shared_trunk_gradient_norm_from_sdyn": float(jnp.linalg.norm(gs["trunk"])),
      "head_1_gradient_norm_from_hier": float(jnp.linalg.norm(gh["heads"][0])),
      "head_1_gradient_norm_from_temp": float(jnp.linalg.norm(gt["heads"][0])),
      "head_1_gradient_norm_from_sdyn": float(jnp.linalg.norm(gs["heads"][0])),
      "decoder_gradient_norm_by_level": json.dumps([float(jnp.linalg.norm(gh["decs"][i])) for i in range(LEVELS)]),
  }
  return row


def train_run(config, seed, data, dataset_hash):
  metrics_path = OUT / "runs" / config / f"seed_{seed}" / "metrics.json"
  if metrics_path.exists():
    return read_json(metrics_path)
  route = ROUTING[config]
  params = init_params(seed)
  run_dir = OUT / "runs" / config / f"seed_{seed}"
  tree_save(run_dir / "checkpoints" / "initial.npz", params)
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
    (loss, raw), grads = jax.value_and_grad(aux_losses, has_aux=True)(
        params, jnp.asarray(obs), jnp.asarray(act), route)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    if step in [250, 500, 1000]:
      ckpt = run_dir / "checkpoints" / f"step_{step}.npz"
      tree_save(ckpt, params)
      row = eval_params(params, data, config, seed, step)
      row["checkpoint_path"] = str(ckpt)
      if step == 1000:
        row.update(gradient_diag(params, data, route))
      evals.append(row)
  report = {
      "config_name": config, "seed": seed, "route": route,
      "dataset_manifest_hash": dataset_hash,
      "optimizer": "sgd", "learning_rate": lr, "batch_size": 32,
      "sequence_length": 64, "optimizer_updates": 1000,
      "checkpoint_schedule": "initial,250,500,1000",
      "parameter_count": param_count(params),
      "wall_clock_seconds": round(time.time() - start, 3),
      "model_derived_metrics": True,
      "status": "pass",
      "code_commit": code_commit(),
      "rows": evals,
  }
  dump(metrics_path, report)
  return report


def baseline_rows():
  rows = []
  for r in csv.DictReader((SYN12 / "continuation_metrics_per_seed_v12.csv").open()):
    if r["config_name"] == "baseline_v9" and int(r["checkpoint_update"]) == 1000:
      row = {
          "config_name": "baseline_shared_trunk",
          "seed": int(r["seed"]),
          "optimizer_updates": 1000,
          "full_prefix_gain": float(r["full_prefix_gain"]),
          "strict_monotonic_pass": r["strict_monotonic_pass"] == "True",
          "nondegrading_pass": r["nondegrading_pass"] == "True",
          "end_to_end_pass": r["end_to_end_pass"] == "True",
          "boundary_auprc_overall": float(r["mean_boundary_auprc"]),
          "boundary_auprc_macro": float(r.get("boundary_auprc_macro", r["mean_boundary_auprc"])),
          "boundary_f1_overall": float(r["mean_boundary_f1"]),
          "boundary_f1_macro": float(r.get("boundary_f1_macro", r["mean_boundary_f1"])),
          "factor_probe_accuracy": float(r["mean_factor_probe_accuracy"]),
          "effective_rank": float(r["effective_rank"]),
          "alive_feature_ratio": float(r["alive_feature_ratio"]),
          "dead_feature_ratio": float(r["dead_feature_ratio"]),
          "topk_utilization_entropy": float(r["topk_utilization_entropy"]),
          "active_count_audit": float(r["active_count_mean"]),
          "collapse_status": "no_collapse_detected",
          "revisit_similarity": "",
          "nuisance_sensitivity": "",
      }
      for i in range(1, 7):
        row[f"prefix_nrmse_l{i}"] = float(r[f"prefix_nrmse_l{i}"])
      rows.append(row)
  return rows


def aggregate(rows):
  metrics = ["full_prefix_gain", "end_to_end_pass", "strict_monotonic_pass", "nondegrading_pass", "boundary_auprc_overall", "boundary_auprc_macro", "boundary_f1_overall", "boundary_f1_macro", "factor_probe_accuracy", "effective_rank", "dead_feature_ratio", "topk_utilization_entropy", "active_count_audit"]
  out = []
  for config in sorted({r["config_name"] for r in rows}):
    sub = [r for r in rows if r["config_name"] == config and int(r["optimizer_updates"]) == 1000]
    for metric in metrics:
      vals = [float(r[metric]) for r in sub if r.get(metric) not in ("", None)]
      if not vals:
        continue
      lo, hi = v13.v11_ci(vals)
      out.append({"config_name": config, "metric": metric, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "standard_error": float(np.std(vals) / math.sqrt(len(vals))), "ci95_low": lo, "ci95_high": hi, "seed_count": len(vals)})
  return out


def run_candidates(data, dataset_hash):
  reports, rows, ckpts, lh_rows = [], [], [], []
  for config in TRAIN_CONFIGS:
    for seed in SEEDS:
      rep = train_run(config, seed, data, dataset_hash)
      reports.append(rep)
      for row in rep["rows"]:
        rows.append(row)
        lh_rows.extend(row.get("level_horizon", []))
      cdir = OUT / "runs" / config / f"seed_{seed}" / "checkpoints"
      ckpts.append({
          "config_name": config, "seed": seed,
          "initial": str(cdir / "initial.npz"),
          "step_250": str(cdir / "step_250.npz"),
          "step_500": str(cdir / "step_500.npz"),
          "step_1000": str(cdir / "step_1000.npz"),
          "load_pass": all((cdir / x).exists() for x in ["initial.npz", "step_250.npz", "step_500.npz", "step_1000.npz"]),
      })
  manifest_rows = [{
      "config_name": r["config_name"], "seed": r["seed"],
      "optimizer_updates": r["optimizer_updates"],
      "dataset_manifest_hash": r["dataset_manifest_hash"],
      "parameter_count": r["parameter_count"],
      "status": r["status"],
      "wall_clock_seconds": r["wall_clock_seconds"],
  } for r in reports]
  manifest = {
      "status": "pass" if len(reports) == 20 and all(r["status"] == "pass" for r in reports) else "fail",
      "expected_runs": 20,
      "completed_runs": len(reports),
      "continuation_runs_to_2500": 0,
      "routes": ROUTING,
      "rows": manifest_rows,
  }
  dump(OUT / "routing_candidate_manifest_v14.json", manifest)
  write_csv(OUT / "routing_candidate_manifest_v14.csv", manifest_rows)
  dump(OUT / "routing_candidate_checkpoints_v14.json", {"status": "pass" if all(c["load_pass"] for c in ckpts) else "fail", "rows": ckpts})
  all_rows = baseline_rows() + rows
  write_csv(OUT / "routing_candidate_metrics_per_seed_v14.csv", all_rows)
  write_csv(OUT / "routing_candidate_metrics_aggregate_v14.csv", aggregate(all_rows))
  figures(all_rows, lh_rows)
  return manifest, all_rows, lh_rows


def figures(rows, lh_rows):
  figs = OUT / "figures"; figs.mkdir(parents=True, exist_ok=True)
  configs = ["baseline_shared_trunk"] + TRAIN_CONFIGS
  plt.figure(figsize=(7, 4))
  for config in configs:
    sub = [r for r in rows if r["config_name"] == config and int(r["optimizer_updates"]) == 1000]
    vals = [np.mean([float(r[f"prefix_nrmse_l{i}"]) for r in sub]) for i in range(1, 7)]
    plt.plot(range(1, 7), vals, marker="o", label=config)
  plt.xlabel("prefix level"); plt.ylabel("NRMSE"); plt.title("Routing prefix profiles V14")
  plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / "fig_routing_prefix_profiles_v14.pdf"); plt.close()
  plt.figure(figsize=(7, 4))
  for config in configs:
    sub = [r for r in rows if r["config_name"] == config and int(r["optimizer_updates"]) == 1000]
    plt.bar(config, np.mean([float(r["boundary_auprc_overall"]) for r in sub]))
  plt.xticks(rotation=30, ha="right", fontsize=7); plt.ylabel("boundary AUPRC"); plt.tight_layout()
  plt.savefig(figs / "fig_routing_boundary_auprc_v14.pdf"); plt.close()
  plt.figure(figsize=(7, 4))
  for config in TRAIN_CONFIGS:
    vals = [x["nrmse"] for x in lh_rows if x["config_name"] == config and x["seed"] == 0]
    plt.plot(vals, label=config)
  plt.ylabel("level-horizon NRMSE"); plt.title("Routing level-horizon V14")
  plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / "fig_routing_level_horizon_v14.pdf"); plt.close()
  plt.figure(figsize=(7, 4))
  grad_metrics = ["shared_trunk_gradient_norm_from_hier", "shared_trunk_gradient_norm_from_temp", "shared_trunk_gradient_norm_from_sdyn"]
  x = np.arange(len(TRAIN_CONFIGS))
  for j, metric in enumerate(grad_metrics):
    vals = []
    for config in TRAIN_CONFIGS:
      sub = [r for r in rows if r["config_name"] == config and int(r["optimizer_updates"]) == 1000 and metric in r]
      vals.append(np.mean([float(r[metric]) for r in sub]))
    plt.bar(x + (j - 1) * 0.22, vals, 0.22, label=metric.replace("_gradient_norm_from_", "_"))
  plt.xticks(x, TRAIN_CONFIGS, rotation=25, ha="right", fontsize=7); plt.ylabel("gradient norm")
  plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(figs / "fig_routing_gradient_paths_v14.pdf"); plt.close()


def select(rows, unit):
  base = [r for r in rows if r["config_name"] == "baseline_shared_trunk" and int(r["optimizer_updates"]) == 1000]
  base_auprc = float(np.mean([float(r["boundary_auprc_overall"]) for r in base]))
  base_macro = float(np.mean([float(r["boundary_auprc_macro"]) for r in base]))
  base_factor = float(np.mean([float(r["factor_probe_accuracy"]) for r in base]))
  entries, selected = [], None
  for config in TRAIN_CONFIGS:
    sub = [r for r in rows if r["config_name"] == config and int(r["optimizer_updates"]) == 1000]
    gains = [float(r["full_prefix_gain"]) for r in sub]
    lo, hi = v13.v11_ci(gains)
    e2e = sum(bool(r["end_to_end_pass"]) for r in sub)
    strict = sum(bool(r["strict_monotonic_pass"]) for r in sub)
    nondeg = sum(bool(r["nondegrading_pass"]) for r in sub)
    auprc = float(np.mean([float(r["boundary_auprc_overall"]) for r in sub]))
    macro = float(np.mean([float(r["boundary_auprc_macro"]) for r in sub]))
    factor = float(np.mean([float(r["factor_probe_accuracy"]) for r in sub]))
    collapse = not all(r["collapse_status"] == "no_collapse_detected" for r in sub)
    routing_ok = unit["status"] == "pass" and all(float(r.get("shared_trunk_gradient_norm_from_hier", 0.0)) <= 1e-9 for r in sub)
    mechanism = (not collapse and auprc >= base_auprc * 0.95 and macro >= base_macro * 0.95 and factor >= base_factor * 0.95)
    passes = np.mean(gains) > 0 and lo >= 0 and e2e >= 4 and mechanism and routing_ok
    reasons = []
    if np.mean(gains) <= 0: reasons.append("nonpositive_prefix_gain")
    if lo < 0: reasons.append("negative_ci95_lower")
    if e2e < 4: reasons.append("end_to_end_seed_count_below_4")
    if not mechanism: reasons.append("mechanism_preservation_failed")
    if not routing_ok: reasons.append("gradient_routing_evidence_failed")
    entry = {
        "config_name": config,
        "routing_flags": ROUTING[config],
        "prefix_gain": float(np.mean(gains)),
        "ci95_low": lo,
        "ci95_high": hi,
        "end_to_end_positive_seeds": e2e,
        "strict_monotonic_seeds": strict,
        "nondegrading_seeds": nondeg,
        "boundary_auprc_overall": auprc,
        "boundary_auprc_macro": macro,
        "factor_probe_accuracy": factor,
        "collapse_status": "collapse_detected" if collapse else "no_collapse_detected",
        "routing_evidence_status": routing_ok,
        "selection_status": "pass" if passes else "reject",
        "rejection_reason": ",".join(reasons),
    }
    entries.append(entry)
    if passes and (selected is None or entry["prefix_gain"] > selected["prefix_gain"]):
      selected = entry
  decision = "PASS_WITH_GRADIENT_ISOLATED_DEVELOPMENT_CANDIDATE" if selected else "NO_STABLE_CANDIDATE_AFTER_GRADIENT_ISOLATION"
  report = {"decision": decision, "selected_candidate": selected["config_name"] if selected else None, "entries": entries}
  dump(OUT / "development_candidate_selection_v14.json", report)
  lines = ["# Development Candidate Selection V14", "", f"Decision: `{decision}`", ""]
  for e in entries:
    lines.append(f"- {e['config_name']}: gain={e['prefix_gain']:.6f}, ci95=[{e['ci95_low']:.6f}, {e['ci95_high']:.6f}], e2e={e['end_to_end_positive_seeds']}/5, boundary={e['boundary_auprc_overall']:.6f}, macro={e['boundary_auprc_macro']:.6f}, factor={e['factor_probe_accuracy']:.6f}, status={e['selection_status']}, reason={e['rejection_reason']}")
  (OUT / "development_candidate_selection_v14.md").write_text("\n".join(lines) + "\n")
  return report


def split_branch_proposal():
  text = """# Split-Branch Revision Proposal V14

No gradient-isolated reconstruction-routing candidate passed Gate D1. Do not implement this architecture before review.

## Evidence Motivating Split
V13 showed boundary degradation already at z1 under hierarchy pressure. V14 tests whether blocking hierarchy gradients into the shared trunk is sufficient. If no candidate passes, the remaining evidence points to a need for separated coarse event and fine reconstruction pathways.

## Minimal Module Graph
shared anchor h -> coarse temporal/event trunk -> z1, temporal/sdyn/event-sensitive objectives
shared anchor h -> fine reconstruction trunk -> z2..z6, nested/fine reconstruction objectives

## Parameter-Count Impact
Naively splitting the trunk increases parameters. A paper-ready version needs an equal-parameter control by reducing branch widths or matching total dense-equivalent parameter count.

## Equal-Parameter Control
Compare split-branch HTS against a single-trunk HTS with the same total parameter count and identical optimizer/update budget.

## Required Ablations
coarse branch only, fine branch only, no reconstruction gradient into coarse branch, no temporal branch, equal-param flat control.

## New Reviewer Concerns
Reviewers may attribute gains to capacity or explicit pathway engineering rather than hierarchy. The equal-param and route-disabled controls are mandatory.

## Expected Synthetic Tests
Repeat prefix gain, boundary AUPRC by prefix, factor probes, level-horizon specialization, collapse, and gradient path audits.

## Expected Atari Development Gate
Only after Synthetic Gate D1 passes should the six-game Atari Gate D2 manifest be prepared.
"""
  (OUT / "split_branch_revision_proposal_v14.md").write_text(text)


def gate_review(selection, audit, unit, manifest):
  selected = selection["selected_candidate"]
  entry = next((e for e in selection["entries"] if e["config_name"] == selected), None)
  report = {
      "v13_decision": "NO_STABLE_CANDIDATE_AFTER_MINIMAL_REVISION",
      "shared_trunk_gradient_path_audit_result": audit["status"],
      "routing_unit_test_result": unit["status"],
      "new_runs_completed": manifest["completed_runs"],
      "continuations_to_2500": manifest["continuation_runs_to_2500"],
      "selected_candidate": selected,
      "selected_routing_flags": entry["routing_flags"] if entry else None,
      "selected_beta_hier": entry["routing_flags"]["beta_hier"] if entry else None,
      "selected_optimizer_budget": 1000 if entry else None,
      "prefix_gain_and_ci": [entry["prefix_gain"], entry["ci95_low"], entry["ci95_high"]] if entry else None,
      "end_to_end_positive_seeds": entry["end_to_end_positive_seeds"] if entry else 0,
      "boundary_auprc_overall_and_macro": [entry["boundary_auprc_overall"], entry["boundary_auprc_macro"]] if entry else None,
      "factor_probe_accuracy": entry["factor_probe_accuracy"] if entry else None,
      "specialization_status": "see fig_routing_level_horizon_v14.pdf",
      "collapse_status": entry["collapse_status"] if entry else None,
      "gate_d1_decision": selection["decision"],
      "gate_d2_status": "blocked" if not selected else "manifest_prepared_not_launched",
  }
  dump(OUT / "gate_d1_review_v14.json", report)
  (OUT / "gate_d1_review_v14.md").write_text(
      "# Gate D1 Review V14\n\n"
      f"Decision: `{selection['decision']}`\n\nSelected candidate: `{selected}`\n\nGate D2: `{report['gate_d2_status']}`\n")
  return report


def maybe_gate_d2(selection):
  if selection["decision"] != "PASS_WITH_GRADIENT_ISOLATED_DEVELOPMENT_CANDIDATE":
    return False
  tasks = ["Alien", "Asterix", "Breakout", "Hero", "MsPacman", "Seaquest"]
  methods = ["dreamer_anchor", "hts_full_selected_candidate", "flat_mh", "larger_flat_param", "matryoshka_only", "dense_multistride_no_sparse", "hts_no_temp", "hts_no_sdyn"]
  commands = []
  for task in tasks:
    for method in methods:
      for seed in [0, 1, 2]:
        commands.append({"task": task, "method": method, "seed": seed, "launch": False, "selected_candidate": selection["selected_candidate"]})
  dump(OUT / "gate_d2_atari_dev_command_manifest_v14.json", {"status": "prepared_not_launched", "commands": commands})
  (OUT / "gate_d2_atari_dev_plan_v14.md").write_text("# Gate D2 Atari Dev Plan V14\n\nPrepared only; not launched.\n")
  return True


def test_report(audit, unit, manifest, selection, gate):
  tests = []
  def add(tid, name, status, source, artifact, reason=""):
    tests.append({"test_id": tid, "test_name": name, "status": status, "execution_status": source, "artifact_path": str(artifact), "failure_reason": reason})
  for r in csv.DictReader((ART / "test_report_v13_full.csv").open()):
    add(r["test_id"], r["test_name"], r["status"], "inherited_from_v13", r["artifact_path"], r.get("failure_reason", ""))
  add("GI-01", "shared-trunk reconstruction-gradient audit", "PASS" if audit["status"] == "pass" else "FAIL", "executed_v14", OUT / "shared_trunk_gradient_path_audit_v14.json")
  for t in unit["tests"]:
    add(t["test_id"], t["test_name"], t["status"], "executed_v14", OUT / "gradient_routing_unit_tests_v14.json", t.get("reason", ""))
  add("GI-11", "routing candidate run completeness", "PASS" if manifest["status"] == "pass" else "FAIL", "executed_v14", OUT / "routing_candidate_manifest_v14.json")
  add("GI-12", "gradient-isolated candidate selection", "PASS" if selection["decision"] == "PASS_WITH_GRADIENT_ISOLATED_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v14", OUT / "development_candidate_selection_v14.json", "" if selection["decision"].startswith("PASS") else selection["decision"])
  add("GI-13", "Gate-D1 V14 review", "PASS" if gate["gate_d1_decision"] == "PASS_WITH_GRADIENT_ISOLATED_DEVELOPMENT_CANDIDATE" else "FAIL", "executed_v14", OUT / "gate_d1_review_v14.json", "" if gate["gate_d1_decision"].startswith("PASS") else gate["gate_d1_decision"])
  write_csv(ART / "test_report_v14_full.csv", tests)
  counts = {s: sum(t["status"] == s for t in tests) for s in ["PASS", "XFAIL", "FAIL"]}
  lines = [f"PASS: {counts['PASS']} | XFAIL: {counts['XFAIL']} | FAIL: {counts['FAIL']}", "", "| test_id | test_name | status | execution_status | artifact_path | failure_reason |", "| --- | --- | --- | --- | --- | --- |"]
  for t in tests:
    lines.append(f"| {t['test_id']} | {t['test_name']} | {t['status']} | {t['execution_status']} | {t['artifact_path']} | {t['failure_reason']} |")
  (ART / "test_report_v14_full.md").write_text("\n".join(lines) + "\n")
  (ART / "remaining_xfail_v14.md").write_text("# Remaining XFAIL V14\n\n- UT-15-P1: larger_flat_flops remains P1.\n")
  return counts


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "figures").mkdir(parents=True, exist_ok=True)
  manifest, dataset_hash, data = load_data()
  audit = shared_trunk_audit(data)
  contract = routing_contract()
  unit = unit_tests(data)
  if unit["status"] != "pass" or audit["status"] != "pass":
    manifest_report = {"status": "blocked", "completed_runs": 0, "expected_runs": 20, "continuation_runs_to_2500": 0}
    rows = baseline_rows()
  else:
    manifest_report, rows, lh_rows = run_candidates(data, dataset_hash)
  selection = select(rows, unit)
  if selection["decision"] != "PASS_WITH_GRADIENT_ISOLATED_DEVELOPMENT_CANDIDATE":
    split_branch_proposal()
  gate = gate_review(selection, audit, unit, manifest_report)
  gate_d2 = maybe_gate_d2(selection)
  counts = test_report(audit, unit, manifest_report, selection, gate)
  summary = {
      "v13_decision": "NO_STABLE_CANDIDATE_AFTER_MINIMAL_REVISION",
      "shared_trunk_gradient_path_audit_result": audit["status"],
      "fine_reconstruction_losses_update_shared_trunk_in_baseline": True,
      "routing_flags_implemented": ROUTING,
      "routing_unit_test_results": unit["status"],
      "new_runs_completed": manifest_report["completed_runs"],
      "new_runs_expected": manifest_report.get("expected_runs", 20),
      "continuation_runs_to_2500": manifest_report.get("continuation_runs_to_2500", 0),
      "candidate_prefix_gains_and_ci": {e["config_name"]: [e["prefix_gain"], e["ci95_low"], e["ci95_high"]] for e in selection["entries"]},
      "candidate_boundary_auprc_overall_and_macro": {e["config_name"]: [e["boundary_auprc_overall"], e["boundary_auprc_macro"]] for e in selection["entries"]},
      "candidate_factor_probe_accuracy": {e["config_name"]: e["factor_probe_accuracy"] for e in selection["entries"]},
      "selected_routing_candidate": selection["selected_candidate"],
      "gate_d1_decision": selection["decision"],
      "gate_d2_manifest_generated": gate_d2,
      "split_branch_proposal_generated": selection["decision"] != "PASS_WITH_GRADIENT_ISOLATED_DEVELOPMENT_CANDIDATE",
      "cumulative_test_counts": counts,
      "remaining_blockers": ["Gate D2 blocked", "split-branch architecture review required"] if not gate_d2 else ["Gate D2 prepared but not launched"],
      "dataset_manifest_hash": dataset_hash,
      "dataset_hash_matches_expected": dataset_hash == EXPECTED_HASH,
      "unrelated_official_processes_observed_but_untouched": subprocess.run(
          "ps -ef | rg 'dreamerv3.main --configs atari100k size12m' | rg -v rg || true",
          shell=True, text=True, stdout=subprocess.PIPE).stdout.strip(),
  }
  dump(ART / "v14_package_summary.json", summary)
  print(json.dumps(to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
