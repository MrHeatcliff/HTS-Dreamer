import jax
import jax.numpy as jnp

from .agent import Agent as DreamerAgent
from .agent import f32, sg, sample, prefix, concat, isimage, imag_loss, repl_loss
from . import hts1


class HTS1Agent(DreamerAgent):

  banner = DreamerAgent.banner + [
      r"--- HTS-WM auxiliary latent hierarchy enabled ---",
  ]

  def __init__(self, obs_space, act_space, config):
    super().__init__(obs_space, act_space, config)
    feat_dim = (
        int(config.dyn.rssm.deter) +
        int(config.dyn.rssm.stoch) * int(config.dyn.rssm.classes))
    self.hts = hts1.HTS1Aux(
        act_space, feat_dim, **config.hts, name='hts')
    self.modules.append(self.hts)
    self.opt = self.opt.__class__(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')
    self.scales.update({
        'hts_hier': config.hts.l_hier,
        'hts_sdyn': config.hts.l_sdyn,
        'hts_temp': config.hts.l_temp,
        'hts_vc': config.hts.l_vc,
        'hts_sparse': config.hts.l_sparse,
    })
    self.latent_anchor_name = getattr(config.hts, 'latent_anchor_name', 'rssm_repfeat')
    self.latent_anchor_source_module = getattr(
        config.hts, 'latent_anchor_source_module', 'dreamerv3.rssm.RSSM.loss')
    self.latent_anchor_dim = feat_dim

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
    hts_losses, hts_metrics = self.hts(hts_anchor, prevact, reset, training)
    losses.update(hts_losses)
    metrics.update(hts_metrics)

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for k, x in shapes.items()
               if not k.startswith('hts_')), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
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
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
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
      inp = self.feat2tensor(feat)
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
    hts_keys = {'hts_hier', 'hts_sdyn', 'hts_temp', 'hts_vc', 'hts_sparse'}
    wm_keys = [k for k in losses if k not in hts_keys]
    metrics['loss/raw/wm'] = sum([losses[k].mean() for k in wm_keys])
    metrics['loss/weighted/wm'] = sum([weighted[k] for k in wm_keys])
    metrics['loss/total'] = loss
    metrics['hts/latent_anchor_dim'] = f32(self.latent_anchor_dim)
    metrics['hts/active_phase'] = f32(active_phase)
    metrics['hts/detach_hts_anchor_active'] = f32(detach_hts_anchor)
    metrics['hts/training_regime_joint'] = f32(regime == 'joint')
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


# Backward-compatible module-local alias. Prefer HTS1Agent explicitly.
Agent = HTS1Agent
