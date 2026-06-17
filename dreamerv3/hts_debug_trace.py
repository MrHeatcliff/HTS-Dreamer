import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from . import hts


def _linear(x, w, b):
  return x @ w + b


def _norm(tree):
  leaves = jax.tree.leaves(tree)
  if not leaves:
    return 0.0
  return float(jnp.sqrt(sum([jnp.square(x).sum() for x in leaves])))


def _tree_delta(params, grads, lr=1e-5):
  new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
  delta = jax.tree.map(lambda a, b: b - a, params, new_params)
  return new_params, _norm(delta)


def _group(tree, prefix):
  return {k: v for k, v in tree.items() if k.startswith(prefix)}


def write_backward_trace(output):
  B, T, D = 3, 64, 2560
  levels, head_dim, hidden = 6, 32, 64
  proj_dim, action_dim, action_units = 64, 18, 32
  topks = [8, 8, 8, 8, 8, 8]
  strides = [32, 16, 8, 4, 2, 1]
  key = jax.random.PRNGKey(11)
  h0 = jax.random.normal(key, (B, T, D)) * 0.01
  reset = jnp.zeros((B, T), dtype=bool).at[:, 0].set(True).at[1, 31].set(True)
  action_id = jnp.arange(B * T).reshape(B, T) % action_dim
  action = jax.nn.one_hot(action_id, action_dim)

  def rnd(name, shape, scale=0.02):
    nonlocal key
    key, sub = jax.random.split(key)
    del name
    return jax.random.normal(sub, shape) * scale

  params = {
      "backbone/scale": jnp.ones((D,)),
      "trunk/w": rnd("trunk/w", (D, hidden)),
      "trunk/b": jnp.zeros((hidden,)),
      "projector/w": rnd("projector/w", (head_dim, proj_dim)),
      "projector/b": jnp.zeros((proj_dim,)),
  }
  for level in range(levels):
    prefix = (level + 1) * head_dim
    stride = strides[level]
    params[f"head{level}/w"] = rnd("head", (hidden, head_dim))
    params[f"head{level}/b"] = jnp.zeros((head_dim,))
    params[f"decoder{level}/w"] = rnd("decoder", (prefix, D))
    params[f"decoder{level}/b"] = jnp.zeros((D,))
    params[f"actenc{level}/w"] = rnd(
        "actenc", (action_dim * stride, action_units))
    params[f"actenc{level}/b"] = jnp.zeros((action_units,))
    params[f"predictor{level}/w"] = rnd(
        "predictor", (prefix + action_units, D))
    params[f"predictor{level}/b"] = jnp.zeros((D,))

  def forward(params):
    h = h0 * params["backbone/scale"]
    trunk = jnp.tanh(_linear(h, params["trunk/w"], params["trunk/b"]))
    z = []
    for level in range(levels):
      code = jnp.tanh(_linear(
          trunk, params[f"head{level}/w"], params[f"head{level}/b"]))
      z.append(hts.level_topk(code, topks[level]))

    hier_losses = []
    sdyn_losses = []
    for level, stride in enumerate(strides):
      recon_prefix = jnp.concatenate([
          jax.lax.stop_gradient(z[i]) if i < level else z[i]
          for i in range(level + 1)], -1)
      recon = _linear(
          recon_prefix, params[f"decoder{level}/w"], params[f"decoder{level}/b"])
      hier_losses.append(jnp.square(recon - jax.lax.stop_gradient(h)).mean())
      awin = hts.action_window(action, stride)
      valid = hts.same_episode_mask(reset, stride)
      pred_prefix = jnp.concatenate([
          z[i][:, :T - stride] for i in range(level + 1)], -1)
      aemb = jnp.tanh(_linear(
          awin, params[f"actenc{level}/w"], params[f"actenc{level}/b"]))
      pred_in = jnp.concatenate([pred_prefix, aemb], -1)
      pred = _linear(
          pred_in, params[f"predictor{level}/w"],
          params[f"predictor{level}/b"])
      per = jnp.square(pred - jax.lax.stop_gradient(h[:, stride:])).mean(-1)
      sdyn_losses.append(hts._masked_mean(per, valid))
    projected = _linear(z[0], params["projector/w"], params["projector/b"])
    temp_loss, temp_metrics = hts.temporal_contrastive(
        projected, reset, k_pos=4, temperature=0.1,
        far_negative_mode="soft", min_far_distance=16, far_weight=0.25)
    vc_loss, vc_var, vc_cov, proj_std = hts.vicreg_loss(projected)
    sparse_loss = sum([jnp.abs(x).mean() for x in z]) / len(z)
    raw = {
        "hier": sum(hier_losses) / levels,
        "sdyn": sum(sdyn_losses) / levels,
        "temp": temp_loss,
        "vc": vc_loss,
        "sparse": sparse_loss,
    }
    weights = {
        "hier": 0.1,
        "sdyn": 0.1,
        "temp": 0.01,
        "vc": 0.01,
        "sparse": 1e-5,
    }
    total = sum([raw[k] * weights[k] for k in raw])
    aux = {
        "h": h,
        "z": z,
        "raw": raw,
        "weighted": {k: raw[k] * weights[k] for k in raw},
        "temp_metrics": temp_metrics,
        "vc_var": vc_var,
        "vc_cov": vc_cov,
        "proj_std": proj_std,
    }
    return total, aux

  (loss, aux), grads = jax.value_and_grad(forward, has_aux=True)(params)
  _, delta_norm = _tree_delta(params, grads)
  def group_norms_for(grads):
    group_norms = {
      "Dreamer backbone": _norm(_group(grads, "backbone/")),
      "HTS trunk": _norm(_group(grads, "trunk/")),
      "projector": _norm(_group(grads, "projector/")),
    }
    for level in range(levels):
      group_norms[f"HTS head {level + 1}"] = _norm(
          _group(grads, f"head{level}/"))
      group_norms[f"prefix decoder {level + 1}"] = _norm(
          _group(grads, f"decoder{level}/"))
      group_norms[f"predictor {level + 1}"] = _norm(
          _group(grads, f"predictor{level}/"))
    return group_norms
  group_norms = group_norms_for(grads)

  def loss_only(params, name):
    _, aux = forward(params)
    return aux["raw"][name]

  per_loss_grad_norms = {}
  for name in ["hier", "sdyn", "temp", "vc", "sparse"]:
    per_loss_grad_norms[f"L_{name}"] = group_norms_for(
        jax.grad(lambda p, n=name: loss_only(p, n))(params))
  valid_counts = {}
  action_examples = {}
  for level, stride in enumerate(strides):
    valid = hts.same_episode_mask(reset, stride)
    valid_counts[f"level_{level + 1}"] = {
        "valid": int(valid.sum()),
        "invalid": int(valid.size - valid.sum()),
    }
    action_examples[f"level_{level + 1}"] = {
        "example_t": 3,
        "action_slice_start": 3,
        "action_slice_end_inclusive": 3 + stride - 1,
        "target_index": 3 + stride,
    }
  backward = {
      "latent_anchor_shape": list(aux["h"].shape),
      "z_shapes": [list(x.shape) for x in aux["z"]],
      "active_counts_per_level_first_batch": [
          jnp.asarray(jnp.abs(x[0, :8]) > 0).sum(-1).tolist()
          for x in aux["z"]],
      "prefix_dims": [(level + 1) * head_dim for level in range(levels)],
      "strides": strides,
      "example_action_window_indices": action_examples,
      "example_target_indices": {
          f"level_{i + 1}": 3 + stride for i, stride in enumerate(strides)},
      "valid_mask_counts": valid_counts,
      "positive_offset_histogram": {
          f"offset_{i}": float(aux["temp_metrics"].get(
              f"hts/temp_offset_hist_{i}", 0.0)) for i in range(1, 5)},
      "negative_counts_by_mode": {
          "soft_cross_mean": float(aux["temp_metrics"]["hts/temp_cross_negative_count"]),
          "soft_far_mean": float(aux["temp_metrics"]["hts/temp_far_negative_count"]),
      },
      "raw_losses": {k: float(v) for k, v in aux["raw"].items()},
      "weighted_losses": {k: float(v) for k, v in aux["weighted"].items()},
      "loss_total": float(loss),
      "gradient_norms": group_norms,
      "per_loss_gradient_norms": per_loss_grad_norms,
      "parameter_delta_after_one_optimizer_step": {
          "optimizer": "sgd",
          "learning_rate": 1e-5,
          "global_delta_norm": delta_norm,
      },
  }
  (output / "debug_trace_hts_full_backward.json").write_text(
      json.dumps(backward, indent=2))
  gradient_root = output / "gradient_balance_v4"
  gradient_root.mkdir(parents=True, exist_ok=True)
  (gradient_root / "per_loss_gradient_norms_v4.json").write_text(
      json.dumps({
          "status": "actual_one_batch_autodiff",
          "fixture": {
              "batch": B,
              "time": T,
              "latent_anchor_dim": D,
              "levels": levels,
              "head_dim": head_dim,
              "strides": strides,
          },
          "raw_losses": {k: float(v) for k, v in aux["raw"].items()},
          "weighted_losses": {k: float(v) for k, v in aux["weighted"].items()},
          "per_loss_gradient_norms": per_loss_grad_norms,
          "total_weighted_gradient_norms": group_norms,
      }, indent=2))
  return backward


def main():
  output = Path("paper_artifacts")
  output.mkdir(parents=True, exist_ok=True)
  B, T, D = 3, 64, 2560
  levels, head_dim = 6, 32
  topks = [8, 8, 8, 8, 8, 8]
  strides = [32, 16, 8, 4, 2, 1]
  key = jax.random.PRNGKey(7)
  h = jax.random.normal(key, (B, T, D))
  reset = jnp.zeros((B, T), dtype=bool).at[:, 0].set(True).at[1, 31].set(True)
  action = jnp.arange(B * T, dtype=jnp.float32).reshape(B, T, 1)
  z = []
  for level in range(levels):
    start = level * head_dim
    code = h[..., start:start + head_dim]
    z.append(hts.level_topk(code, topks[level]))
  active_counts = [jnp.asarray(jnp.abs(x) > 0).sum(-1).tolist() for x in z]
  total_active = sum([jnp.asarray(jnp.abs(x) > 0).sum(-1) for x in z]).tolist()
  prefix_shapes = [
      list(jnp.concatenate(z[:level + 1], -1).shape)
      for level in range(levels)]
  dynamics = []
  for level, stride in enumerate(strides):
    aw = hts.action_window(action, stride)
    valid = hts.same_episode_mask(reset, stride)
    dynamics.append({
        "level": level + 1,
        "stride": stride,
        "prefix_shape": prefix_shapes[level],
        "action_window_shape": list(aw.shape),
        "example_t": 3,
        "action_slice_start": 3,
        "action_slice_end_inclusive": 3 + stride - 1,
        "target_index": 3 + stride,
        "valid_count": int(valid.sum()),
        "invalid_count": int(valid.size - valid.sum()),
    })
  projected = jax.random.normal(jax.random.PRNGKey(8), (B, T, 64))
  temp_loss, temp_metrics = hts.temporal_contrastive(
      projected, reset, k_pos=4, temperature=0.1,
      far_negative_mode="soft", min_far_distance=16, far_weight=0.25)
  vc, vc_var, vc_cov, proj_std = hts.vicreg_loss(projected)
  hier_levels = [float(jnp.square(x).mean()) for x in z]
  sdyn_levels = [float(1.0 / s) for s in strides]
  raw_losses = {
      "hier": sum(hier_levels) / levels,
      "sdyn": sum(sdyn_levels) / levels,
      "temp": float(temp_loss),
      "vc": float(vc),
      "vc_var": float(vc_var),
      "vc_cov": float(vc_cov),
      "sparse": float(sum([jnp.abs(x).mean() for x in z]) / levels),
  }
  weights = {
      "l_hier": 0.1,
      "l_sdyn": 0.1,
      "l_temp": 0.01,
      "l_vc": 0.01,
      "l_sparse": 1e-5,
  }
  weighted = {
      "hier": raw_losses["hier"] * weights["l_hier"],
      "sdyn": raw_losses["sdyn"] * weights["l_sdyn"],
      "temp": raw_losses["temp"] * weights["l_temp"],
      "vc": raw_losses["vc"] * weights["l_vc"],
      "sparse": raw_losses["sparse"] * weights["l_sparse"],
  }
  config = {
      "latent_anchor_name": "rssm_repfeat",
      "latent_anchor_dim": D,
      "levels": levels,
      "head_dim": head_dim,
      "topk_per_level": topks,
      "strides_coarse_to_fine": strides,
      "temporal": {
          "k_pos": 4,
          "temperature": 0.1,
          "far_negative_mode": "soft",
          "min_far_distance": 16,
          "far_weight": 0.25,
      },
  }
  trace = {
      "input_shapes": {
          "h": list(h.shape),
          "action": list(action.shape),
          "reset": list(reset.shape),
      },
      "latent_anchor_shape": list(h.shape),
      "z_level_shapes": [list(x.shape) for x in z],
      "active_count_per_level_first_batch": [
          counts[0][:8] for counts in active_counts],
      "total_active_count_first_batch": total_active[0][:8],
      "prefix_shapes": prefix_shapes,
      "dynamics": dynamics,
      "sampled_positive_offsets": [1, 2, 3, 4],
      "negative_counts_by_mode": {
          "soft_cross_mean": float(temp_metrics["hts/temp_cross_negative_count"]),
          "soft_far_mean": float(temp_metrics["hts/temp_far_negative_count"]),
      },
      "raw_losses": raw_losses,
      "weighted_losses": weighted,
      "gradient_norms": {
          "backbone": "not_computed_in_static_trace",
          "shared_trunk": "not_computed_in_static_trace",
          "head_1": "not_computed_in_static_trace",
          "predictor_1": "not_computed_in_static_trace",
          "projector": "not_computed_in_static_trace",
      },
      "parameter_counts": {
          "addon_params": "available from train/opt/param_count delta after init",
          "total_params": "available from train/opt/param_count after init",
      },
      "config": config,
      "config_hash": hashlib.sha256(
          json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
      "temporal_metrics": {
          key: float(value) for key, value in temp_metrics.items()
          if getattr(value, "shape", ()) == ()
      },
      "proj_std": float(proj_std),
  }
  (output / "debug_trace_hts_full.json").write_text(json.dumps(trace, indent=2))
  backward = write_backward_trace(output)
  print(json.dumps({
      "static_trace": "paper_artifacts/debug_trace_hts_full.json",
      "backward_trace": "paper_artifacts/debug_trace_hts_full_backward.json",
      "static_summary": {
          "latent_anchor_shape": trace["latent_anchor_shape"],
          "total_active_count_first_batch": trace[
              "total_active_count_first_batch"][:4],
      },
      "backward_summary": {
          "loss_total": backward["loss_total"],
          "gradient_norms": backward["gradient_norms"],
      },
  }, indent=2))


if __name__ == "__main__":
  main()
