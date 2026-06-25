import jax
import jax.numpy as jnp

from .agent import Agent as DreamerAgent
from .agent import f32, sg, sample, prefix, concat, isimage, imag_loss, repl_loss
from . import hts


def aux_warmup_alpha(
    optimizer_step, batch_size, batch_length, train_ratio=256,
    action_repeat=4, warmup_raw_frames=0, warmup_agent_actions=0,
    warmup_optimizer_updates=0, mode='linear'):
  optimizer_step = f32(optimizer_step)
  minibatch_steps = f32(batch_size * batch_length)
  train_ratio = f32(train_ratio)
  action_repeat = f32(action_repeat)
  updates_per_action = train_ratio / jnp.maximum(minibatch_steps, 1.0)
  agent_actions = optimizer_step / jnp.maximum(updates_per_action, 1e-8)
  raw_frames = agent_actions * action_repeat
  raw_horizon = f32(warmup_raw_frames)
  default_agent_horizon = raw_horizon / jnp.maximum(action_repeat, 1.0)
  configured_agent_horizon = f32(warmup_agent_actions)
  agent_horizon = jnp.where(
      configured_agent_horizon > 0,
      configured_agent_horizon,
      default_agent_horizon)
  opt_horizon = f32(warmup_optimizer_updates)
  use_opt_horizon = opt_horizon > 0
  denom = jnp.where(use_opt_horizon, opt_horizon, agent_horizon)
  numer = jnp.where(use_opt_horizon, optimizer_step, agent_actions)
  enabled = (raw_horizon > 0) | (agent_horizon > 0) | (opt_horizon > 0)
  linear = jnp.minimum(1.0, numer / jnp.maximum(denom, 1.0))
  hard = f32(numer >= denom)
  alpha = jnp.where(mode == 'hard', hard, linear)
  alpha = jnp.where(enabled, alpha, 1.0)
  return jnp.clip(alpha, 0.0, 1.0), agent_actions, raw_frames, agent_horizon


class Agent(DreamerAgent):

  banner = DreamerAgent.banner + [
      r"--- HTS-WM auxiliary latent hierarchy enabled ---",
  ]

  def __init__(self, obs_space, act_space, config):
    super().__init__(obs_space, act_space, config)
    feat_dim = (
        int(config.dyn.rssm.deter) +
        int(config.dyn.rssm.stoch) * int(config.dyn.rssm.classes))
    self.hts = hts.HTSAux(
        act_space, feat_dim, **config.hts, name='hts')
    self.modules.append(self.hts)
    self.opt = self.opt.__class__(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')
    self.scales.update({
        'hts_hier': config.hts.l_hier,
        'hts_sdyn': config.hts.l_sdyn,
        'hts_ctrl': getattr(config.hts, 'l_ctrl', 0.0),
        'hts_temp': config.hts.l_temp,
        'hts_vc': config.hts.l_vc,
        'hts_sparse': config.hts.l_sparse,
    })
    self.latent_anchor_name = getattr(config.hts, 'latent_anchor_name', 'rssm_repfeat')
    self.latent_anchor_source_module = getattr(
        config.hts, 'latent_anchor_source_module', 'dreamerv3.rssm.RSSM.loss')
    self.latent_anchor_dim = feat_dim

  @property
  def policy_keys(self):
    if self._uses_hts_actor_critic():
      return '^(enc|dyn|dec|pol|hts)/'
    return super().policy_keys

  def _uses_hts_actor_critic(self):
    return getattr(self.config.hts, 'actor_critic_input', 'rssm_feat') != 'rssm_feat'

  def _ac_input(self, feat):
    h = self.feat2tensor(feat)
    if not self._uses_hts_actor_critic():
      return h
    return self.hts.actor_critic_features(h)

  def _ac_metrics(self, feat):
    if not self._uses_hts_actor_critic():
      return {}
    h = self.feat2tensor(feat)
    z = self.hts.zfull(h, sparse=True, stopgrad=True)
    inp = self.hts.actor_critic_features(h)
    mode = getattr(self.config.hts, 'actor_critic_input', 'rssm_feat')
    return {
        'hts/actor_input_dim': f32(inp.shape[-1]),
        'hts/critic_input_dim': f32(inp.shape[-1]),
        'hts/actor_input_mode_z_full': f32(mode == 'z_full'),
        'hts/critic_input_mode_z_full': f32(mode == 'z_full'),
        'hts/actor_input_mode_h_z_hybrid': f32(mode == 'h_z_hybrid'),
        'hts/critic_input_mode_h_z_hybrid': f32(mode == 'h_z_hybrid'),
        'hts/z_full_norm_policy_path': jnp.linalg.norm(z, axis=-1).mean(),
        'hts/z_full_variance_policy_path': z.var(),
    }

  def policy(self, carry, obs, mode='train'):
    if not self._uses_hts_actor_critic():
      return super().policy(carry, obs, mode)
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self._ac_input(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = self._policy_finite(obs, carry, tokens, feat, act)
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      import elements
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def _policy_finite(self, obs, carry, tokens, feat, act):
    import elements
    return elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))

  def _hts_phase(self):
    regime = getattr(self.config.hts, 'training_regime', 'joint')
    step = self.opt.step.read()
    phase1_steps = int(getattr(self.config.hts, 'phase1_steps', 0))
    detach = False
    phase = 2
    if regime in ('detach_hts_anchor', 'frozen'):
      detach = True
      phase = 1
    elif regime == 'two_phase':
      detach = step < phase1_steps
      phase = jnp.where(step < phase1_steps, 1, 2)
    elif regime == 'posthoc_frozen_backbone':
      detach = True
      phase = 1
    return regime, detach, phase

  def _hts_aux_warmup(self):
    cfg = self.config.hts
    step = f32(self.opt.step.read())
    raw_horizon = f32(getattr(cfg, 'aux_warmup_raw_frames', 0))
    action_repeat = f32(getattr(cfg, 'aux_warmup_action_repeat', 4))
    train_ratio = f32(getattr(cfg, 'aux_warmup_train_ratio', 256))
    opt_horizon = f32(getattr(cfg, 'aux_warmup_optimizer_updates', 0))
    agent_horizon = f32(getattr(
        cfg, 'aux_warmup_agent_actions', 0))
    mode = getattr(cfg, 'aux_warmup_mode', 'linear')
    alpha, agent_actions, raw_frames, agent_horizon = aux_warmup_alpha(
        step, self.config.batch_size, self.config.batch_length,
        train_ratio=train_ratio, action_repeat=action_repeat,
        warmup_raw_frames=raw_horizon,
        warmup_agent_actions=agent_horizon,
        warmup_optimizer_updates=opt_horizon,
        mode=mode)
    updates_per_action = train_ratio / jnp.maximum(
        f32(self.config.batch_size * self.config.batch_length), 1.0)
    return alpha, {
        'hts/aux_warmup_alpha': alpha,
        'hts/aux_warmup_raw_frames': raw_frames,
        'hts/aux_warmup_agent_actions': agent_actions,
        'hts/aux_warmup_horizon_raw_frames': raw_horizon,
        'hts/aux_warmup_horizon_agent_actions': agent_horizon,
        'hts/aux_warmup_horizon_optimizer_updates': opt_horizon,
        'hts/aux_warmup_updates_per_agent_action': updates_per_action,
        'hts/aux_warmup_mode_hard': f32(mode == 'hard'),
        'hts_aux_warmup_alpha': alpha,
        'hts_aux_warmup_raw_frames': raw_frames,
        'hts_aux_warmup_agent_actions': agent_actions,
        'hts_aux_warmup_horizon_raw_frames': raw_horizon,
        'hts_aux_warmup_horizon_agent_actions': agent_horizon,
    }

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    hts_anchor = self.feat2tensor(repfeat)
    regime, detach_hts_anchor, active_phase = self._hts_phase()
    hts_anchor = jnp.where(detach_hts_anchor, sg(hts_anchor), hts_anchor)
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    value_bootstrap = None
    if self._uses_hts_actor_critic():
      value_bootstrap = self.val(sg(self._ac_input(repfeat)), 2).pred()
    hts_losses, hts_metrics = self.hts(
        hts_anchor, prevact, reset, training,
        reward=obs['reward'], cont=con, value_bootstrap=value_bootstrap)
    hts_alpha, warmup_metrics = self._hts_aux_warmup()
    hts_raw_losses = hts_losses
    hts_losses = {key: value * hts_alpha for key, value in hts_losses.items()}
    losses.update(hts_losses)
    metrics.update(hts_metrics)
    metrics.update(warmup_metrics)
    metrics.update(self._ac_metrics(repfeat))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for k, x in shapes.items()
               if not k.startswith('hts_')), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self._ac_input(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    ac_inp = self._ac_input(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
        self.pol(ac_inp, 2),
        self.val(ac_inp, 2),
        self.slowval(ac_inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self._ac_input(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    weighted = {k: v.mean() * self.scales[k] for k, v in losses.items()}
    loss = sum(weighted.values())
    hts_keys = {
        'hts_hier', 'hts_sdyn', 'hts_ctrl', 'hts_temp', 'hts_vc',
        'hts_sparse'}
    wm_keys = [k for k in losses if k not in hts_keys]
    metrics['loss/raw/wm'] = sum([losses[k].mean() for k in wm_keys])
    metrics['loss/weighted/wm'] = sum([weighted[k] for k in wm_keys])
    raw_name = {
        'hts_hier': 'hier',
        'hts_sdyn': 'sdyn',
        'hts_ctrl': 'ctrl',
        'hts_temp': 'temp',
        'hts_vc': 'vc',
        'hts_sparse': 'sparse',
    }
    for key, name in raw_name.items():
      raw = hts_raw_losses[key].mean()
      effective_coef = f32(self.scales[key]) * hts_alpha
      metrics[f'loss/{name}_raw'] = raw
      metrics[f'loss/{name}_weighted'] = raw * effective_coef
      metrics[f'loss/raw/{name}'] = raw
      metrics[f'loss/weighted/{name}'] = raw * effective_coef
      metrics[f'hts/coef_{name}_effective'] = effective_coef
      metrics[f'coef_{name}_effective'] = effective_coef
    metrics['loss/total'] = loss
    metrics['hts/latent_anchor_dim'] = f32(self.latent_anchor_dim)
    metrics['hts/active_phase'] = f32(active_phase)
    metrics['hts/detach_hts_anchor_active'] = f32(detach_hts_anchor)
    metrics['hts/training_regime_joint'] = f32(regime == 'joint')
    metrics['hts/training_regime_joint_online_initial'] = f32(
        regime == 'joint_online_initial')
    metrics['hts/training_regime_detach_hts_anchor'] = f32(
        regime in ('detach_hts_anchor', 'frozen'))
    metrics['hts/training_regime_posthoc_frozen_backbone'] = f32(
        regime == 'posthoc_frozen_backbone')
    metrics['hts/training_regime_two_phase'] = f32(regime == 'two_phase')
    metrics['hts/backbone_lr_scale'] = f32(getattr(
        self.config.hts, 'backbone_lr_scale', 1.0))
    metrics['hts/hts_lr_scale'] = f32(getattr(
        self.config.hts, 'hts_lr_scale',
        getattr(self.config.hts, 'hierarchy_lr_scale', 1.0)))
    metrics['hts/phase1_steps'] = f32(getattr(self.config.hts, 'phase1_steps', 0))
    metrics['hts/phase2_steps'] = f32(getattr(self.config.hts, 'phase2_steps', 0))

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)
