# Official DreamerV3 Replay-Ratio Semantics

This note documents the semantics used by the official DreamerV3 port.

## Config Fields

For Atari100K:

```text
run.train_ratio = 256
batch_size = 16
batch_length = 64
env.atari100k.repeat = 4
```

The training loop computes:

```python
batch_steps = batch_size * batch_length
should_train = Ratio(train_ratio / batch_steps)
```

Therefore, `train_ratio` is measured as replayed timesteps per collected agent
action. It is not directly "gradient updates per environment step".

With `batch_size * batch_length = 1024` and `train_ratio = 256`, the expected
minibatch update rate is:

```text
256 / 1024 = 0.25 optimizer updates per agent action
```

Because Atari100K uses action repeat `4`:

```text
0.25 / 4 = 0.0625 optimizer updates per raw frame
```

Equivalently:

```text
1 update per 4 agent actions
1 update per 16 raw Atari frames
```

## Required Runtime Fields

Every paper run should export unambiguous fields:

```text
train_ratio_replayed_steps_per_agent_action
batch_size
batch_length
minibatch_steps
expected_updates_per_agent_action
realized_optimizer_updates
realized_agent_actions
expected_updates_per_raw_frame
realized_frames
action_repeat
```

Current paper artifact writer stores these fields in:

```text
paper_artifacts/episode_scores.jsonl
paper_artifacts/train_metrics.jsonl
paper_artifacts/latest_train_summary.json
paper_artifacts/replay_ratio_consistency.json
```

The artifact writer intentionally does not store `256` and `0.25` under the
same ambiguous key. The configured official value is
`train_ratio_replayed_steps_per_agent_action = 256`; the derived minibatch
update rate is `expected_updates_per_agent_action = 0.25`.

The consistency check excludes initial replay prefill and compilation. It
compares realized optimizer updates against agent actions after the first
observed optimizer update:

```text
abs(realized_updates_per_agent_action_excluding_prefill
    - expected_updates_per_agent_action) < tolerance
```

## Sweep Policy

Replay-ratio sweeps must be separate experiments. Do not mix replay-ratio
settings inside main comparison rows.
