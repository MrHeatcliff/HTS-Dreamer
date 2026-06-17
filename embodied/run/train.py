import collections
from functools import partial as bind

import elements
import embodied
import numpy as np

from .paper_artifacts import PaperArtifactWriter


class TraceableRatio:

  def __init__(self, ratio):
    self.ratio = ratio
    self.prev = None

  def __call__(self, step):
    step = int(step)
    before = self.prev
    if self.ratio == 0:
      repeats = 0
    elif self.ratio < 0:
      repeats = 1
    elif self.prev is None:
      self.prev = step
      repeats = 1
    else:
      repeats = int((step - self.prev) * self.ratio)
      self.prev += repeats / self.ratio
    return repeats, before, self.prev


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()
  paper = PaperArtifactWriter(logdir, args)

  batch_steps = args.batch_size * args.batch_length
  should_train = TraceableRatio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3 and args.log_policy_video:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      score = result.pop('score')
      length = result.pop('length')
      logger.add({'score': score, 'length': length}, prefix='episode')
      paper.write_episode(step, score, length)
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      paper.write_update_event(
          step, requested_updates=0, executed_updates=0,
          optimizer_updates_cumulative=int(getattr(agent, 'n_updates', 0)),
          is_prefill=True, is_compile_only=False,
          scheduler_accumulator_before=should_train.prev,
          scheduler_accumulator_after=should_train.prev)
      return
    requested, sched_before, sched_after = should_train(step)
    executed = 0
    for _ in range(requested):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      executed += 1
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
    paper.write_update_event(
        step, requested_updates=requested, executed_updates=executed,
        optimizer_updates_cumulative=int(getattr(agent, 'n_updates', 0)),
        is_prefill=False, is_compile_only=False,
        scheduler_accumulator_before=sched_before,
        scheduler_accumulator_after=sched_after)
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      train_stats = train_agg.result()
      ep_stats = epstats.result()
      replay_stats = replay.stats()
      usage_stats = usage.stats()
      fps_stats = {
          'fps/policy': policy_fps.result(),
          'fps/train': train_fps.result(),
      }
      timer_stats = {'timer': elements.timer.stats()['summary']}
      logger.add(train_stats)
      logger.add(ep_stats, prefix='epstats')
      logger.add(replay_stats, prefix='replay')
      logger.add(usage_stats, prefix='usage')
      logger.add(fps_stats)
      logger.add(timer_stats)
      paper_stats = {}
      paper_stats.update(train_stats)
      paper_stats.update({f'epstats/{k}': v for k, v in ep_stats.items()})
      paper_stats.update({f'replay/{k}': v for k, v in replay_stats.items()})
      paper_stats.update({f'usage/{k}': v for k, v in usage_stats.items()})
      paper_stats.update(fps_stats)
      paper_stats.update(timer_stats)
      paper.write_train_metrics(step, paper_stats)
      logger.write()

    if should_save(step):
      cp.save()

  cp.save()
  paper.finalize(step, checkpoint_path=logdir / 'ckpt')
  logger.close()
