from . import main as base


def make_agent(config):
  from .hts1_agent import HTS1Agent
  env = base.make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return base.embodied.RandomAgent(obs_space, act_space)
  return HTS1Agent(obs_space, act_space, base.elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def main(argv=None):
  original = base.make_agent
  base.make_agent = make_agent
  try:
    return base.main(argv)
  finally:
    base.make_agent = original


if __name__ == '__main__':
  main()
