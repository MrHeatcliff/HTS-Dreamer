# Full26 Reference Interpretation V16

## What the logs can support

- External DreamerV3 reference for Breakout and Alien.
- Rough learning-curve context from existing official log files.
- Sanity check that official Atari logging is producing episode scores.
- Future comparison target for HTS Atari Gate D2 after Gate D is unblocked.

## What the logs cannot support

- Synthetic mechanism claims.
- HTS architecture selection.
- Gate D1 pass/fail.
- Gate D2 pass/fail.
- HTS hyperparameter tuning.
- Paper-final Atari benchmark unless run completion and protocol matching are verified.

## Main caveats

- `episode/score` is parsed as single episode score, not an aggregated evaluation mean.
- The x-axis is recoverable as logged `step`; from config, `action_repeat=4`, so this report records `step` as frames and computes `agent_actions=step/4`.
- Completed status is inferred by comparing latest logged step with `run.steps * action_repeat`; partial runs remain labeled.
- Eval episode count for `episode/score` rows is recorded as `1`; aggregate eval episodes are not recovered from these logs.
- These logs are not used to pass Gate D2 or tune HTS.
- Plot generation: {'breakout': True, 'alien': True}.
