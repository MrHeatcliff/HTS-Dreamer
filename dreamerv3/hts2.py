import jax
import jax.numpy as jnp
import ninjax as nj

import embodied.jax.nets as nn


f32 = jnp.float32
sg = jax.lax.stop_gradient
EPS = 1e-8


def _masked_mean(value, mask):
  value = f32(value)
  mask = f32(mask)
  while mask.ndim < value.ndim:
    mask = mask[..., None]
  return (value * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def _offdiag(x):
  n = x.shape[0]
  return x.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()


def _as_tuple(value, length, default):
  if value is None:
    value = default
  if isinstance(value, (int, float)):
    return tuple([value] * length)
  value = tuple(value)
  if len(value) != length:
    raise ValueError((len(value), length, value))
  return value


def level_topk(x, k):
  if not k or k <= 0 or k >= x.shape[-1]:
    return x
  _, idx = jax.lax.top_k(jnp.abs(x), int(k))
  mask = jax.nn.one_hot(idx, x.shape[-1], dtype=x.dtype).sum(-2)
  return x * sg(mask)


def action_window(action, stride):
  T = action.shape[1]
  pieces = [action[:, offset:T - stride + offset] for offset in range(stride)]
  return jnp.concatenate(pieces, -1)


def episode_ids(reset):
  return jnp.cumsum(reset.astype(jnp.int32), axis=1)


def same_episode_mask(reset, stride):
  if stride <= 0:
    return jnp.ones_like(reset, dtype=bool)
  valid = jnp.ones_like(reset[:, :reset.shape[1] - stride], dtype=bool)
  for offset in range(1, stride + 1):
    valid = valid & (~reset[:, offset:reset.shape[1] - stride + offset])
  return valid


def same_episode_pair_mask(reset):
  ids = episode_ids(reset)
  return ids[:, :, None] == ids[:, None, :]


def far_negative_weight(mode, far_weight):
  if mode == 'none':
    return 0.0
  if mode == 'hard':
    return 1.0
  if mode == 'soft':
    return float(far_weight)
  raise ValueError(f'Unknown far_negative_mode: {mode}')


def temporal_contrastive(
    v, reset, k_pos=4, temperature=0.1, far_negative_mode='none',
    min_far_distance=16, far_weight=0.25):
  """Masked InfoNCE over projected coarse states.

  Args:
    v: Projected coarse representation with shape [B, T, D].
    reset: Boolean is_first/reset mask with shape [B, T].

  Returns:
    loss, metrics dictionary.
  """
  B, T, D = v.shape
  del D
  k_pos = int(k_pos)
  temperature = f32(temperature)
  q = v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), EPS)
  t = jnp.arange(T)
  # Deterministic uniform coverage of offsets. This is auditable and avoids
  # introducing an RNG dependency into the Dreamer loss API.
  offsets = 1 + (t % max(k_pos, 1))
  pos_t = jnp.minimum(t + offsets, T - 1)
  offset_valid = (t + offsets) < T
  ids = episode_ids(reset)
  pos_valid = offset_valid[None, :] & (ids == jnp.take(ids, pos_t, axis=1))

  q_pos = jnp.take(q, pos_t, axis=1)
  pos_logits = (q * sg(q_pos)).sum(-1) / temperature

  q_flat = sg(q.reshape((B * T, q.shape[-1])))
  flat_b = jnp.repeat(jnp.arange(B), T)
  flat_t = jnp.tile(jnp.arange(T), B)
  flat_ids = ids.reshape((B * T,))
  logits = jnp.einsum('btd,nd->btn', q, q_flat) / temperature

  cross_mask = flat_b[None, None, :] != jnp.arange(B)[:, None, None]
  cross_mask = jnp.broadcast_to(cross_mask, logits.shape)
  same_batch = flat_b[None, None, :] == jnp.arange(B)[:, None, None]
  same_batch = jnp.broadcast_to(same_batch, logits.shape)
  same_ep = flat_ids[None, None, :] == ids[:, :, None]
  far_dist = jnp.abs(flat_t[None, None, :] - t[None, :, None])
  not_self = far_dist > 0
  far_mask = (
      same_batch & same_ep & not_self &
      (far_dist >= int(min_far_distance)))
  cross_mask = cross_mask & jnp.broadcast_to(
      pos_valid[:, :, None], cross_mask.shape)
  fweight = far_negative_weight(far_negative_mode, far_weight)
  far_mask = far_mask & jnp.broadcast_to(
      pos_valid[:, :, None], far_mask.shape)
  far_mask = far_mask & (f32(fweight) > 0)

  exp_pos = jnp.exp(pos_logits)
  exp_logits = jnp.exp(logits)
  cross_denom = (exp_logits * f32(cross_mask)).sum(-1)
  far_denom = (exp_logits * f32(far_mask)).sum(-1) * f32(fweight)
  denom = exp_pos + cross_denom + far_denom + EPS
  nce = -jnp.log(exp_pos / denom)
  loss = _masked_mean(nce, pos_valid)

  pos_sim = (q * sg(q_pos)).sum(-1)
  cross_count = f32(cross_mask).sum(-1)
  far_count = f32(far_mask).sum(-1)
  cross_sim = (logits * temperature * f32(cross_mask)).sum(-1) / jnp.maximum(cross_count, 1.0)
  far_sim = (logits * temperature * f32(far_mask)).sum(-1) / jnp.maximum(far_count, 1.0)
  hist = jnp.stack([
      (f32(pos_valid) * f32(offsets[None, :] == (i + 1))).sum()
      for i in range(k_pos)])
  return loss, {
      'hts/temp_positive_valid_frac': f32(pos_valid).mean(),
      'hts/temp_terminal_boundary_invalid_frac': 1.0 - f32(pos_valid).mean(),
      'hts/temp_cross_negative_count': _masked_mean(cross_count, pos_valid),
      'hts/temp_far_negative_count': _masked_mean(far_count, pos_valid),
      'hts/temp_mean_positive_similarity': _masked_mean(pos_sim, pos_valid),
      'hts/temp_mean_cross_negative_similarity': _masked_mean(cross_sim, pos_valid),
      'hts/temp_mean_far_negative_similarity': _masked_mean(far_sim, pos_valid),
      'hts/temp_sampled_positive_offset_mean': _masked_mean(
          f32(offsets)[None, :], pos_valid),
      **{f'hts/temp_offset_hist_{i + 1}': hist[i] for i in range(k_pos)},
  }


def vicreg_loss(v, gamma=1.0, cov_scale=1.0):
  v = v.reshape((-1, v.shape[-1]))
  v = v - v.mean(0, keepdims=True)
  std = jnp.sqrt(v.var(0) + 1e-4)
  var_loss = jnp.maximum(0.0, f32(gamma) - std).mean()
  cov = (v.T @ v) / jnp.maximum(v.shape[0] - 1, 1)
  cov_loss = jnp.square(_offdiag(cov)).mean()
  return var_loss + f32(cov_scale) * cov_loss, var_loss, cov_loss, std.mean()


class HTS2Aux(nj.Module):

  levels: int = 6
  latent_anchor_name: str = 'rssm_repfeat'
  latent_anchor_source_module: str = 'dreamerv3.rssm.RSSM.loss'
  head_dim: int = 32
  topk: int = 8
  topk_per_level: tuple = (8, 8, 8, 8, 8, 8)
  strides: tuple = (32, 16, 8, 4, 2, 1)
  strides_coarse_to_fine: tuple = (32, 16, 8, 4, 2, 1)
  hidden: int = 256
  layers: int = 2
  action_units: int = 128
  proj_dim: int = 64
  beta_hier: tuple = (1/6, 1/6, 1/6, 1/6, 1/6, 1/6)
  alpha_sdyn: tuple = (1/6, 1/6, 1/6, 1/6, 1/6, 1/6)
  act: str = 'silu'
  norm: str = 'rms'
  l_hier: float = 1.0
  l_sdyn: float = 1.0
  l_ctrl: float = 0.0
  l_temp: float = 0.1
  l_vc: float = 0.1
  l_sparse: float = 1e-4
  use_lhier: bool = True
  use_lsdyn: bool = True
  use_lctrl: bool = False
  use_ltemp: bool = True
  use_lvc: bool = True
  use_lsparse: bool = True
  use_lred: bool = False
  impl: str = 'hts2'
  method: str = 'auxiliary_hierarchy'
  sparsity_mode: str = 'levelwise_topk'
  actor_critic_input: str = 'rssm_feat'
  actor_critic_stopgrad_z: bool = False
  hier_beta0: float = 1.0
  hier_rho: float = 0.75
  ctrl_reward_scale: float = 1.0
  ctrl_continue_scale: float = 1.0
  ctrl_value_scale: float = 1.0
  discount: float = 0.997
  temp_margin: float = 1.0
  far_gap: int = 16
  temporal_k_pos: int = 4
  temporal_temperature: float = 0.1
  temporal_far_negative_mode: str = 'none'
  temporal_min_far_distance: int = 16
  temporal_far_weight: float = 0.25
  decoder_prefix_stop_gradient: bool = True
  predictor_prefix_stop_gradient: bool = False
  dynamics_target_stop_gradient: bool = True
  training_regime: str = 'joint'
  phase1_steps: int = 0
  phase2_steps: int = 0
  backbone_lr_scale: float = 1.0
  hts_lr_scale: float = 1.0
  hierarchy_lr_scale: float = 1.0
  warmup_steps: int = 0
  aux_warmup_raw_frames: int = 0
  aux_warmup_agent_actions: int = 0
  aux_warmup_optimizer_updates: int = 0
  aux_warmup_mode: str = 'linear'
  aux_warmup_action_repeat: int = 4
  aux_warmup_train_ratio: int = 256
  vicreg_gamma: float = 1.0
  vicreg_cov_scale: float = 1.0
  variant: str = 'hts_full'
  flat_width: int = 192
  flat_topk: int = 48
  sgf_action_units: int = 18

  def __init__(self, act_space, feat_dim, **kw):
    self.act_space = act_space
    self.feat_dim = int(feat_dim)
    self.kw = kw

  def __call__(
      self, h, prevact, reset, training, reward=None, cont=None,
      value_bootstrap=None):
    del training
    h = nn.cast(h)
    B, T, D = h.shape
    assert D == self.feat_dim, (D, self.feat_dim)
    action = nn.DictConcat(self.act_space, 1)(prevact)
    action = nn.cast(action)
    variant = getattr(self, 'variant', 'hts_full')
    if getattr(self, 'method', '') == 'paper_core_control_prefix':
      return self._call_paper_core(
          h, action, reset, reward, cont, value_bootstrap)
    if variant == 'paper_core_control_prefix':
      return self._call_paper_core(
          h, action, reset, reward, cont, value_bootstrap)
    if variant == 'flat_sae':
      return self._call_flat_sae(h)
    if variant == 'flat_mh':
      return self._call_flat_mh(h, action, reset)
    if variant == 'larger_flat_param':
      return self._call_flat_mh(h, action, reset)
    if variant == 'sgf_style_flat_same_code':
      return self._call_sgf_style(h, action, reset)
    if variant == 'flat_partition_dim_matched':
      return self._call_flat_partition_dim_matched(h)
    if variant == 'recon_only_hierarchy':
      return self._call_hierarchy_variant(
          h, action, reset, sparse=False, recon=True, sdyn=False,
          temp=False, vc=False, sparse_penalty=False)
    if variant == 'matryoshka_only':
      return self._call_hierarchy_variant(
          h, action, reset, sparse=True, recon=True, sdyn=False,
          temp=False, vc=False, sparse_penalty=True)
    if variant == 'dense_multistride_no_sparse':
      return self._call_hierarchy_variant(
          h, action, reset, sparse=False, recon=True, sdyn=True,
          temp=True, vc=True, sparse_penalty=False)
    z = self._encode(h)
    losses = {}
    metrics = {}

    hier_loss, hier_metrics = self._nested_recon(h, z)
    sdyn_loss, sdyn_metrics = self._sparse_dynamics(h, z, action, reset)
    temp_loss, temp_metrics = self._temporal(z[0], reset)
    vc_loss, vc_metrics = self._vicreg(z[0])
    sparse_loss = sum([jnp.abs(x).mean() for x in z]) / len(z)

    losses['hts_hier'] = hier_loss
    losses['hts_sdyn'] = sdyn_loss
    losses['hts_ctrl'] = jnp.zeros((), f32)
    losses['hts_temp'] = temp_loss
    losses['hts_vc'] = vc_loss
    losses['hts_sparse'] = sparse_loss
    metrics.update(hier_metrics)
    metrics.update(sdyn_metrics)
    metrics.update(temp_metrics)
    metrics.update(vc_metrics)
    metrics['hts/sparse_l1'] = sparse_loss
    metrics['hts/active_ratio'] = jnp.mean(jnp.stack([
        (jnp.abs(x) > 0).mean() for x in z]))
    metrics['hts/mean_abs'] = jnp.mean(jnp.stack([
        jnp.abs(x).mean() for x in z]))
    metrics['hts/latent_anchor_dim'] = f32(self.feat_dim)
    metrics['hts/total_dictionary_width'] = f32(self.levels * self.head_dim)
    metrics['hts/total_active_budget'] = f32(sum(_as_tuple(
        self.topk_per_level, self.levels, self.topk)))
    metrics['loss/raw/hier'] = hier_loss
    metrics['loss/raw/sdyn'] = sdyn_loss
    metrics['loss/raw/ctrl'] = losses['hts_ctrl']
    metrics['loss/raw/temp'] = temp_loss
    metrics['loss/raw/vc'] = vc_loss
    metrics['loss/raw/sparse'] = sparse_loss
    metrics['loss/weighted/hier'] = hier_loss * f32(self.l_hier)
    metrics['loss/weighted/sdyn'] = sdyn_loss * f32(self.l_sdyn)
    metrics['loss/weighted/ctrl'] = losses['hts_ctrl'] * f32(self.l_ctrl)
    metrics['loss/weighted/temp'] = temp_loss * f32(self.l_temp)
    metrics['loss/weighted/vc'] = vc_loss * f32(self.l_vc)
    metrics['loss/weighted/sparse'] = sparse_loss * f32(self.l_sparse)
    return losses, metrics

  def _zero_losses(self):
    zero = jnp.array(0.0, f32)
    return {
        'hts_hier': zero,
        'hts_sdyn': zero,
        'hts_ctrl': zero,
        'hts_temp': zero,
        'hts_vc': zero,
        'hts_sparse': zero,
    }

  def _base_variant_metrics(self, variant):
    return {
        'hts/variant_flat_sae': f32(variant == 'flat_sae'),
        'hts/variant_flat_mh': f32(variant == 'flat_mh'),
        'hts/variant_sgf_style_flat_same_code': f32(
            variant == 'sgf_style_flat_same_code'),
        'hts/variant_flat_partition_dim_matched': f32(
            variant == 'flat_partition_dim_matched'),
        'hts/variant_recon_only_hierarchy': f32(
            variant == 'recon_only_hierarchy'),
        'hts/variant_matryoshka_only': f32(variant == 'matryoshka_only'),
        'hts/variant_dense_multistride_no_sparse': f32(
            variant == 'dense_multistride_no_sparse'),
    }

  def _call_flat_sae(self, h):
    code = self._flat_code(h, sparse=True)
    losses = self._zero_losses()
    recon = self._flat_recon(h, code)
    sparse = jnp.abs(code).mean()
    losses['hts_hier'] = recon
    losses['hts_sparse'] = sparse
    metrics = self._base_variant_metrics('flat_sae')
    metrics.update({
        'hts/flat_recon': recon,
        'hts/sparse_l1': sparse,
        'hts/active_ratio': (jnp.abs(code) > 0).mean(),
        'hts/mean_abs': jnp.abs(code).mean(),
        'hts/latent_anchor_dim': f32(self.feat_dim),
        'hts/total_dictionary_width': f32(self.flat_width),
        'hts/total_active_budget': f32(self.flat_topk),
        'loss/raw/hier': recon,
        'loss/raw/sdyn': losses['hts_sdyn'],
        'loss/raw/ctrl': losses['hts_ctrl'],
        'loss/raw/temp': losses['hts_temp'],
        'loss/raw/vc': losses['hts_vc'],
        'loss/raw/sparse': sparse,
        'loss/weighted/hier': recon * f32(self.l_hier),
        'loss/weighted/sdyn': losses['hts_sdyn'],
        'loss/weighted/ctrl': losses['hts_ctrl'],
        'loss/weighted/temp': losses['hts_temp'],
        'loss/weighted/vc': losses['hts_vc'],
        'loss/weighted/sparse': sparse * f32(self.l_sparse),
    })
    return losses, metrics

  def _call_flat_partition_dim_matched(self, h):
    # Dense equal-width partition control with a single full-code decoder.
    code = self._flat_project(h, self.levels * self.head_dim, 'flat_partition')
    parts = jnp.split(code, self.levels, axis=-1)
    losses = self._zero_losses()
    x = self._mlp('flat_partition_recon', code)
    pred = self.sub(
        'flat_partition_recon_out', nn.Linear, self.feat_dim, **self.kw)(x)
    recon = jnp.square(pred - sg(h)).mean()
    losses['hts_hier'] = recon
    metrics = self._base_variant_metrics('flat_partition_dim_matched')
    metrics.update({
        'hts/flat_partition_recon': recon,
        'hts/active_ratio': jnp.ones((), f32),
        'hts/mean_abs': jnp.mean(jnp.stack([jnp.abs(x).mean() for x in parts])),
        'hts/latent_anchor_dim': f32(self.feat_dim),
        'hts/total_dictionary_width': f32(self.levels * self.head_dim),
        'hts/total_active_budget': f32(self.levels * self.head_dim),
        'loss/raw/hier': recon,
        'loss/raw/sdyn': losses['hts_sdyn'],
        'loss/raw/ctrl': losses['hts_ctrl'],
        'loss/raw/temp': losses['hts_temp'],
        'loss/raw/vc': losses['hts_vc'],
        'loss/raw/sparse': losses['hts_sparse'],
        'loss/weighted/hier': recon * f32(self.l_hier),
        'loss/weighted/sdyn': losses['hts_sdyn'],
        'loss/weighted/ctrl': losses['hts_ctrl'],
        'loss/weighted/temp': losses['hts_temp'],
        'loss/weighted/vc': losses['hts_vc'],
        'loss/weighted/sparse': losses['hts_sparse'],
    })
    return losses, metrics

  def _call_flat_mh(self, h, action, reset):
    code = self._flat_code(h, sparse=False)
    losses = self._zero_losses()
    sdyn, sdyn_metrics = self._flat_multihorizon(h, code, action, reset)
    losses['hts_sdyn'] = sdyn
    metrics = self._base_variant_metrics('flat_mh')
    metrics.update(sdyn_metrics)
    metrics.update({
        'hts/active_ratio': jnp.ones((), f32),
        'hts/mean_abs': jnp.abs(code).mean(),
        'hts/latent_anchor_dim': f32(self.feat_dim),
        'hts/total_dictionary_width': f32(self.flat_width),
        'hts/total_active_budget': f32(self.flat_width),
        'loss/raw/hier': losses['hts_hier'],
        'loss/raw/sdyn': sdyn,
        'loss/raw/ctrl': losses['hts_ctrl'],
        'loss/raw/temp': losses['hts_temp'],
        'loss/raw/vc': losses['hts_vc'],
        'loss/raw/sparse': losses['hts_sparse'],
        'loss/weighted/hier': losses['hts_hier'],
        'loss/weighted/sdyn': sdyn * f32(self.l_sdyn),
        'loss/weighted/ctrl': losses['hts_ctrl'],
        'loss/weighted/temp': losses['hts_temp'],
        'loss/weighted/vc': losses['hts_vc'],
        'loss/weighted/sparse': losses['hts_sparse'],
    })
    return losses, metrics

  def _call_sgf_style(self, h, action, reset):
    del reset
    code = self._flat_project(h, self.proj_dim, 'sgf')
    losses = self._zero_losses()
    pred_in = jnp.concatenate([code[:, :-1], action[:, :-1]], -1)
    x = self._mlp('sgf_pred', pred_in)
    pred = self.sub('sgf_pred_out', nn.Linear, self.feat_dim, **self.kw)(x)
    sdyn = jnp.square(pred - sg(h[:, 1:])).mean()
    vc, vc_var, vc_cov, proj_std = vicreg_loss(code, self.vicreg_gamma)
    losses['hts_sdyn'] = sdyn
    losses['hts_vc'] = vc
    metrics = self._base_variant_metrics('sgf_style_flat_same_code')
    metrics.update({
        'hts/sgf_one_step': sdyn,
        'hts/vicreg_var': vc_var,
        'hts/vicreg_cov': vc_cov,
        'hts/proj_std': proj_std,
        'hts/active_ratio': jnp.ones((), f32),
        'hts/mean_abs': jnp.abs(code).mean(),
        'hts/latent_anchor_dim': f32(self.feat_dim),
        'hts/total_dictionary_width': f32(self.proj_dim),
        'hts/total_active_budget': f32(self.proj_dim),
        'loss/raw/hier': losses['hts_hier'],
        'loss/raw/sdyn': sdyn,
        'loss/raw/ctrl': losses['hts_ctrl'],
        'loss/raw/temp': losses['hts_temp'],
        'loss/raw/vc': vc,
        'loss/raw/sparse': losses['hts_sparse'],
        'loss/weighted/hier': losses['hts_hier'],
        'loss/weighted/sdyn': sdyn * f32(self.l_sdyn),
        'loss/weighted/ctrl': losses['hts_ctrl'],
        'loss/weighted/temp': losses['hts_temp'],
        'loss/weighted/vc': vc * f32(self.l_vc),
        'loss/weighted/sparse': losses['hts_sparse'],
    })
    return losses, metrics

  def _call_hierarchy_variant(
      self, h, action, reset, sparse, recon, sdyn, temp, vc, sparse_penalty):
    z = self._encode(h, sparse=sparse)
    losses = self._zero_losses()
    metrics = self._base_variant_metrics(getattr(self, 'variant', 'hts_full'))
    if recon:
      hier_loss, hier_metrics = self._nested_recon(h, z)
      losses['hts_hier'] = hier_loss
      metrics.update(hier_metrics)
    if sdyn:
      sdyn_loss, sdyn_metrics = self._sparse_dynamics(h, z, action, reset)
      losses['hts_sdyn'] = sdyn_loss
      metrics.update(sdyn_metrics)
    if temp:
      temp_loss, temp_metrics = self._temporal(z[0], reset)
      losses['hts_temp'] = temp_loss
      metrics.update(temp_metrics)
    if vc:
      vc_loss, vc_metrics = self._vicreg(z[0])
      losses['hts_vc'] = vc_loss
      metrics.update(vc_metrics)
    if sparse_penalty:
      losses['hts_sparse'] = sum([jnp.abs(x).mean() for x in z]) / len(z)
    metrics['hts/sparse_l1'] = losses['hts_sparse']
    metrics['hts/active_ratio'] = jnp.mean(jnp.stack([
        (jnp.abs(x) > 0).mean() for x in z]))
    metrics['hts/mean_abs'] = jnp.mean(jnp.stack([
        jnp.abs(x).mean() for x in z]))
    metrics['hts/latent_anchor_dim'] = f32(self.feat_dim)
    metrics['hts/total_dictionary_width'] = f32(self.levels * self.head_dim)
    metrics['hts/total_active_budget'] = f32(
        sum(_as_tuple(self.topk_per_level, self.levels, self.topk))
        if sparse else self.levels * self.head_dim)
    for name, key in [
        ('hier', 'hts_hier'), ('sdyn', 'hts_sdyn'), ('ctrl', 'hts_ctrl'),
        ('temp', 'hts_temp'),
        ('vc', 'hts_vc'), ('sparse', 'hts_sparse')]:
      scale = getattr(self, f'l_{name}')
      metrics[f'loss/raw/{name}'] = losses[key]
      metrics[f'loss/weighted/{name}'] = losses[key] * f32(scale)
    return losses, metrics

  def z_levels(self, h, sparse=True, stopgrad=False):
    squeeze = h.ndim == 2
    h3 = h[:, None] if squeeze else h
    z = self._encode(h3, sparse=sparse)
    if stopgrad:
      z = [sg(x) for x in z]
    if squeeze:
      z = [x[:, 0] for x in z]
    return z

  def zfull(self, h, sparse=True, stopgrad=False):
    return jnp.concatenate(self.z_levels(h, sparse, stopgrad), -1)

  def actor_critic_features(self, h):
    mode = getattr(self, 'actor_critic_input', 'rssm_feat')
    z = self.zfull(
        h, sparse=True, stopgrad=getattr(self, 'actor_critic_stopgrad_z', False))
    if mode == 'rssm_feat':
      return h
    if mode == 'z_full':
      return z
    if mode == 'h_z_hybrid':
      return jnp.concatenate([h, z], -1)
    raise ValueError(f'Unknown actor_critic_input: {mode}')

  def actor_critic_metrics(self, h):
    z = self.z_levels(h, sparse=True, stopgrad=True)
    zfull = jnp.concatenate(z, -1)
    metrics = {
        'hts/actor_input_zfull': f32(
            getattr(self, 'actor_critic_input', 'rssm_feat') == 'z_full'),
        'hts/actor_input_h_z_hybrid': f32(
            getattr(self, 'actor_critic_input', 'rssm_feat') == 'h_z_hybrid'),
        'hts/actor_critic_stopgrad_z': f32(
            getattr(self, 'actor_critic_stopgrad_z', False)),
        'hts/z_full_norm': jnp.linalg.norm(zfull, axis=-1).mean(),
        'hts/z_full_variance': zfull.var(),
        'z_full_norm': jnp.linalg.norm(zfull, axis=-1).mean(),
        'z_full_variance': zfull.var(),
    }
    for level, item in enumerate(z):
      active = (jnp.abs(item) > 0).sum(-1)
      metrics[f'hts/z_active_count_level_{level + 1}'] = f32(active).mean()
      metrics[f'hts/z_norm_level_{level + 1}'] = (
          jnp.linalg.norm(item, axis=-1).mean())
      metrics[f'hts/z_variance_level_{level + 1}'] = item.var()
    return metrics

  def _call_paper_core(self, h, action, reset, reward, cont, value_bootstrap):
    z = self._encode(h, sparse=True)
    losses = self._zero_losses()
    metrics = self._base_variant_metrics('paper_core_control_prefix')
    hier = jnp.array(0.0, f32)
    sdyn = jnp.array(0.0, f32)
    ctrl = jnp.array(0.0, f32)
    if getattr(self, 'use_lhier', True):
      hier, hier_metrics = self._nested_recon(h, z, beta_schedule='front')
      losses['hts_hier'] = hier
      metrics.update(hier_metrics)
    if getattr(self, 'use_lsdyn', True):
      sdyn, sdyn_metrics = self._prefix_dynamics(z, action, reset)
      losses['hts_sdyn'] = sdyn
      metrics.update(sdyn_metrics)
    if getattr(self, 'use_lctrl', True):
      ctrl, ctrl_metrics = self._control_prefix(
          z, action, reset, reward, cont, value_bootstrap)
      losses['hts_ctrl'] = ctrl
      metrics.update(ctrl_metrics)
    if getattr(self, 'use_ltemp', False):
      temp_loss, temp_metrics = self._temporal(z[0], reset)
      losses['hts_temp'] = temp_loss
      metrics.update(temp_metrics)
    if getattr(self, 'use_lvc', False):
      vc_loss, vc_metrics = self._vicreg(z[0])
      losses['hts_vc'] = vc_loss
      metrics.update(vc_metrics)
    if getattr(self, 'use_lsparse', False):
      losses['hts_sparse'] = sum([jnp.abs(x).mean() for x in z]) / len(z)

    metrics.update(self.actor_critic_metrics(h))
    metrics.update({
        'hts/method_paper_core_control_prefix': jnp.array(1.0, f32),
        'hts/sparse_l1': losses['hts_sparse'],
        'hts/active_ratio': jnp.mean(jnp.stack([
            (jnp.abs(x) > 0).mean() for x in z])),
        'hts/mean_abs': jnp.mean(jnp.stack([
            jnp.abs(x).mean() for x in z])),
        'hts/latent_anchor_dim': f32(self.feat_dim),
        'hts/total_dictionary_width': f32(self.levels * self.head_dim),
        'hts/total_active_budget': f32(sum(_as_tuple(
            self.topk_per_level, self.levels, self.topk))),
        'hts/vicreg_cov_scale': f32(self.vicreg_cov_scale),
    })
    for name, key in [
        ('hier', 'hts_hier'), ('sdyn', 'hts_sdyn'), ('ctrl', 'hts_ctrl'),
        ('temp', 'hts_temp'), ('vc', 'hts_vc'), ('sparse', 'hts_sparse')]:
      scale = getattr(self, f'l_{name}')
      metrics[f'loss/raw/{name}'] = losses[key]
      metrics[f'loss/weighted/{name}'] = losses[key] * f32(scale)
    return losses, metrics

  def _mlp(self, name, x, units=None, layers=None):
    units = self.hidden if units is None else units
    layers = self.layers if layers is None else layers
    for i in range(layers):
      x = self.sub(f'{name}_lin{i}', nn.Linear, units, **self.kw)(x)
      x = self.sub(f'{name}_norm{i}', nn.Norm, self.norm)(x)
      x = nn.act(self.act)(x)
    return x

  def _flat_project(self, h, width, name):
    trunk = self._mlp(f'{name}_trunk', h)
    x = self.sub(f'{name}_code', nn.Linear, int(width), **self.kw)(trunk)
    return nn.act(self.act)(x)

  def _flat_code(self, h, sparse):
    code = self._flat_project(h, self.flat_width, 'flat')
    return self._topk(code, self.flat_topk) if sparse else code

  def _flat_recon(self, h, code):
    x = self._mlp('flat_recon', code)
    pred = self.sub('flat_recon_out', nn.Linear, self.feat_dim, **self.kw)(x)
    return jnp.square(pred - sg(h)).mean()

  def _flat_multihorizon(self, h, code, action, reset):
    horizons = (1, 2, 4, 8, 16, 32)
    losses = []
    metrics = {}
    for idx, horizon in enumerate(horizons):
      if h.shape[1] <= horizon:
        loss = jnp.array(0.0, h.dtype)
        losses.append(loss)
        metrics[f'hts/flat_mh_h{horizon}'] = loss
        metrics[f'hts/flat_mh_valid_h{horizon}'] = loss
        continue
      awin = self._action_window(action, horizon)
      aemb = self._mlp(f'flat_mh_actenc{idx}', awin, self.action_units, 1)
      inp = jnp.concatenate([code[:, :h.shape[1] - horizon], aemb], -1)
      x = self._mlp(f'flat_mh_pred{idx}', inp)
      pred = self.sub(
          f'flat_mh_pred{idx}_out', nn.Linear, self.feat_dim, **self.kw)(x)
      per = jnp.square(pred - sg(h[:, horizon:])).mean(-1)
      valid = self._same_episode(reset, horizon)
      loss = _masked_mean(per, valid)
      losses.append(loss / len(horizons))
      metrics[f'hts/flat_mh_h{horizon}'] = loss
      metrics[f'hts/flat_mh_valid_h{horizon}'] = f32(valid).mean()
    return sum(losses), metrics

  def _encode(self, h, sparse=True):
    trunk = self._mlp('trunk', h)
    heads = []
    topks = _as_tuple(self.topk_per_level, self.levels, self.topk)
    for level in range(self.levels):
      x = self.sub(f'head{level}', nn.Linear, self.head_dim, **self.kw)(trunk)
      x = nn.act(self.act)(x)
      if sparse:
        x = self._topk(x, topks[level])
      heads.append(x)
    return heads

  def _topk(self, x, k=None):
    return level_topk(x, self.topk if k is None else int(k))

  def _nested_recon(self, h, z, beta_schedule='tuple'):
    losses = []
    metrics = {}
    if beta_schedule == 'front':
      raw = jnp.array([
          f32(self.hier_beta0) * (f32(self.hier_rho) ** i)
          for i in range(self.levels)], f32)
      weights = raw / jnp.maximum(raw.sum(), 1e-8)
    else:
      weights = _as_tuple(self.beta_hier, self.levels, 1.0 / self.levels)
    for level in range(self.levels):
      prefix = [
          sg(z[i]) if self.decoder_prefix_stop_gradient and i < level else z[i]
          for i in range(level + 1)]
      inp = jnp.concatenate(prefix, -1)
      x = self._mlp(f'recon{level}', inp)
      pred = self.sub(f'recon{level}_out', nn.Linear, self.feat_dim, **self.kw)(x)
      loss = jnp.square(pred - sg(h)).mean(-1)
      losses.append(loss.mean() * f32(weights[level]))
      metrics[f'hts/hier_l{level + 1}'] = loss.mean()
      metrics[f'hts/loss_hier_level_{level + 1}'] = loss.mean()
      metrics[f'hts/prefix_recon_error_level_{level + 1}'] = loss.mean()
      metrics[f'loss/raw/hier_level_{level + 1}'] = loss.mean()
      metrics[f'loss/weighted/hier_level_{level + 1}'] = (
          loss.mean() * f32(weights[level]) * f32(self.l_hier))
    metrics['hts/loss_hier_total'] = sum(losses)
    return sum(losses), metrics

  def _action_window(self, action, stride):
    return action_window(action, stride)

  def _same_episode(self, reset, stride):
    return same_episode_mask(reset, stride)

  def _sparse_dynamics(self, h, z, action, reset):
    T = h.shape[1]
    deltas = tuple(int(x) for x in self.strides_coarse_to_fine or self.strides)
    losses = []
    metrics = {}
    weights = _as_tuple(self.alpha_sdyn, self.levels, 1.0 / self.levels)
    for level, stride in enumerate(deltas):
      if T <= stride:
        masked = jnp.array(0.0, h.dtype)
        losses.append(masked)
        metrics[f'hts/sdyn_l{level + 1}'] = masked
        metrics[f'hts/sdyn_valid_l{level + 1}'] = masked
        metrics[f'loss/raw/sdyn_level_{level + 1}'] = masked
        metrics[f'loss/weighted/sdyn_level_{level + 1}'] = masked
        continue
      prefix = [
          sg(z[i][:, :T - stride])
          if self.predictor_prefix_stop_gradient and i < level
          else z[i][:, :T - stride]
          for i in range(level + 1)]
      zinp = jnp.concatenate(prefix, -1)
      awin = self._action_window(action, stride)
      aemb = self._mlp(f'actenc{level}', awin, self.action_units, 1)
      inp = jnp.concatenate([zinp, aemb], -1)
      x = self._mlp(f'pred{level}', inp)
      pred = self.sub(f'pred{level}_out', nn.Linear, self.feat_dim, **self.kw)(x)
      target = sg(h[:, stride:]) if self.dynamics_target_stop_gradient else h[:, stride:]
      loss = jnp.square(pred - target).mean(-1)
      valid = self._same_episode(reset, stride)
      masked = _masked_mean(loss, valid)
      losses.append(masked * f32(weights[level]))
      metrics[f'hts/sdyn_l{level + 1}'] = masked
      metrics[f'hts/sdyn_valid_l{level + 1}'] = f32(valid).mean()
      metrics[f'loss/raw/sdyn_level_{level + 1}'] = masked
      metrics[f'loss/weighted/sdyn_level_{level + 1}'] = (
          masked * f32(weights[level]) * f32(self.l_sdyn))
    return sum(losses), metrics

  def _prefix_dynamics(self, z, action, reset):
    T = z[0].shape[1]
    deltas = tuple(int(x) for x in self.strides_coarse_to_fine or self.strides)
    losses = []
    metrics = {}
    weights = _as_tuple(self.alpha_sdyn, self.levels, 1.0 / self.levels)
    for level, stride in enumerate(deltas):
      prefix_width = self.head_dim * (level + 1)
      if T <= stride:
        masked = jnp.array(0.0, f32)
        losses.append(masked)
        metrics[f'hts/sdyn_l{level + 1}'] = masked
        metrics[f'hts/sdyn_valid_l{level + 1}'] = masked
        metrics[f'hts/loss_sdyn_level_{level + 1}'] = masked
        metrics[f'hts/prefix_dyn_error_level_{level + 1}'] = masked
        metrics[f'loss/raw/sdyn_level_{level + 1}'] = masked
        metrics[f'loss/weighted/sdyn_level_{level + 1}'] = masked
        continue
      zinp = jnp.concatenate([item[:, :T - stride] for item in z[:level + 1]], -1)
      target = sg(jnp.concatenate(
          [item[:, stride:] for item in z[:level + 1]], -1))
      awin = self._action_window(action, stride)
      aemb = self._mlp(f'pcore_actenc{level}', awin, self.action_units, 1)
      inp = jnp.concatenate([zinp, aemb], -1)
      x = self._mlp(f'pcore_pred{level}', inp)
      pred = self.sub(
          f'pcore_pred{level}_out', nn.Linear, prefix_width, **self.kw)(x)
      per = jnp.square(pred - target).mean(-1)
      valid = self._same_episode(reset, stride)
      masked = _masked_mean(per, valid)
      losses.append(masked * f32(weights[level]))
      metrics[f'hts/sdyn_l{level + 1}'] = masked
      metrics[f'hts/sdyn_valid_l{level + 1}'] = f32(valid).mean()
      metrics[f'hts/loss_sdyn_level_{level + 1}'] = masked
      metrics[f'hts/prefix_dyn_error_level_{level + 1}'] = masked
      metrics[f'hts/prefix_dyn_error_stride_{stride}'] = masked
      metrics[f'loss/raw/sdyn_level_{level + 1}'] = masked
      metrics[f'loss/weighted/sdyn_level_{level + 1}'] = (
          masked * f32(weights[level]) * f32(self.l_sdyn))
    total = sum(losses)
    metrics['hts/loss_sdyn_total'] = total
    return total, metrics

  def _discounted_prefix_return(self, reward, cont, bootstrap, stride):
    B, T = reward.shape
    n = T - stride
    gamma = f32(self.discount)
    ret = jnp.zeros((B, n), f32)
    alive = jnp.ones((B, n), f32)
    for offset in range(stride):
      r = reward[:, offset:n + offset]
      c = cont[:, offset:n + offset]
      ret = ret + alive * (gamma ** offset) * sg(r)
      alive = alive * sg(c)
    ret = ret + alive * (gamma ** stride) * sg(bootstrap[:, stride:])
    return ret

  def _control_prefix(self, z, action, reset, reward, cont, value_bootstrap):
    if reward is None or cont is None:
      zero = jnp.array(0.0, f32)
      return zero, {'hts/loss_ctrl_missing_targets': jnp.array(1.0, f32)}
    T = z[0].shape[1]
    reward = f32(reward)
    cont = f32(cont)
    if value_bootstrap is None:
      value_bootstrap = jnp.zeros_like(reward)
    value_bootstrap = f32(value_bootstrap)
    deltas = tuple(int(x) for x in self.strides_coarse_to_fine or self.strides)
    weights = _as_tuple(self.alpha_sdyn, self.levels, 1.0 / self.levels)
    reward_losses = []
    continue_losses = []
    value_losses = []
    metrics = {}
    for level, stride in enumerate(deltas):
      if T <= stride:
        rloss = closs = vloss = jnp.array(0.0, f32)
        valid = jnp.zeros((z[0].shape[0], 1), bool)
      else:
        prefix = jnp.concatenate([item[:, :T - stride] for item in z[:level + 1]], -1)
        awin = self._action_window(action, stride)
        aemb = self._mlp(f'ctrl_actenc{level}', awin, self.action_units, 1)
        inp = jnp.concatenate([prefix, aemb], -1)
        x = self._mlp(f'ctrl_seq{level}', inp)
        rew_pred = self.sub(
            f'ctrl_rew{level}_out', nn.Linear, stride, **self.kw)(x)
        con_logit = self.sub(
            f'ctrl_con{level}_out', nn.Linear, stride, **self.kw)(x)
        v = self._mlp(f'ctrl_val{level}', prefix)
        val_pred = self.sub(
            f'ctrl_val{level}_out', nn.Linear, 1, **self.kw)(v)[..., 0]
        rtarg = jnp.stack(
            [reward[:, offset:T - stride + offset] for offset in range(stride)],
            -1)
        ctarg = jnp.stack(
            [cont[:, offset:T - stride + offset] for offset in range(stride)],
            -1)
        valid = self._same_episode(reset, stride)
        rper = jnp.square(rew_pred - sg(rtarg)).mean(-1)
        bce = (
            jax.nn.softplus(con_logit) -
            con_logit * sg(ctarg))
        cper = bce.mean(-1)
        gtarg = self._discounted_prefix_return(
            reward, cont, value_bootstrap, stride)
        vper = jnp.square(val_pred - sg(gtarg))
        rloss = _masked_mean(rper, valid)
        closs = _masked_mean(cper, valid)
        vloss = _masked_mean(vper, valid)
      reward_losses.append(rloss * f32(weights[level]))
      continue_losses.append(closs * f32(weights[level]))
      value_losses.append(vloss * f32(weights[level]))
      metrics[f'hts/loss_ctrl_reward_level_{level + 1}'] = rloss
      metrics[f'hts/loss_ctrl_continue_level_{level + 1}'] = closs
      metrics[f'hts/loss_ctrl_value_level_{level + 1}'] = vloss
      metrics[f'hts/prefix_value_pred_level_{level + 1}'] = (
          val_pred.mean() if T > stride else jnp.array(0.0, f32))
      metrics[f'hts/ctrl_valid_level_{level + 1}'] = f32(valid).mean()
    rtotal = sum(reward_losses) * f32(self.ctrl_reward_scale)
    ctotal = sum(continue_losses) * f32(self.ctrl_continue_scale)
    vtotal = sum(value_losses) * f32(self.ctrl_value_scale)
    total = rtotal + ctotal + vtotal
    metrics.update({
        'hts/loss_ctrl_total': total,
        'hts/loss_ctrl_reward_total': rtotal,
        'hts/loss_ctrl_continue_total': ctotal,
        'hts/loss_ctrl_value_total': vtotal,
        'loss/raw/ctrl_reward': rtotal,
        'loss/raw/ctrl_continue': ctotal,
        'loss/raw/ctrl_value': vtotal,
    })
    return total, metrics

  def _project(self, z):
    x = self._mlp('proj', z, self.proj_dim, 1)
    return self.sub('proj_out', nn.Linear, self.proj_dim, **self.kw)(x)

  def _temporal(self, z1, reset):
    v = self._project(z1)
    return temporal_contrastive(
        v, reset,
        k_pos=self.temporal_k_pos,
        temperature=self.temporal_temperature,
        far_negative_mode=self.temporal_far_negative_mode,
        min_far_distance=self.temporal_min_far_distance,
        far_weight=self.temporal_far_weight)

  def _vicreg(self, z1):
    v = self._project(z1)
    loss, var_loss, cov_loss, proj_std = vicreg_loss(
        v, self.vicreg_gamma, self.vicreg_cov_scale)
    return loss, {
        'hts/vicreg_var': var_loss,
        'hts/vicreg_cov': cov_loss,
        'hts/vicreg_cov_scale': f32(self.vicreg_cov_scale),
        'hts/proj_std': proj_std,
        'loss/raw/vc_var': var_loss,
        'loss/raw/vc_cov': cov_loss,
    }


# Backward-compatible module-local alias. Prefer HTS2Aux explicitly.
HTSAux = HTS2Aux
