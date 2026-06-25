# V22 Determinism Forensics Log

Date: 2026-06-24

Scope: HTS-WM V22 official DreamerV3 port, Atari100K Breakout determinism checks.

## Why This Exists

We observed that two runs with the same seed, seeded Atari env, deterministic JAX flag, no sticky actions, and no no-op randomness could still diverge after training started.

The goal of this debug pass was not to tune HTS. It was to identify whether the mismatch came from:

- environment randomness,
- model/action RNG,
- learning-rate schedule,
- optimizer update schedule,
- replay batch sampling,
- GPU numerical drift,
- or asynchronous execution.

## Main Finding

The first true source of mismatch was the training batch stream, not LR scheduling and not late GPU drift.

With normal stream prefetch enabled:

- action trace matched before the first updates,
- update schedule matched,
- first optimizer update happened at the same agent step,
- but the first training batch already differed at optimizer update 0.

Observed in:

```text
paper_artifacts/atari_v22_paper_core_control_prefix/v22_determinism_topk_detail_lambda1_breakout_seed0_steps14000_noops0_tr256_onlineTrue_trace1_uuid1_cuda_bfloat16_jitTrue.json
```

Important mismatch:

```text
batch_trace first mismatch:
  step: 1088
  optimizer_updates_before: 0
  batch_hash: different
  stepid_hash: different
  action_hash: different
```

This means the two runs were training on different batches before any model update could create numerical drift.

## Root Cause

DreamerV3's JAX agent wraps the replay stream in a background prefetch thread:

```python
return embodied.streams.Prefetch(st, fn)
```

The prefetch thread can sample from replay as soon as replay becomes available. Because this happens asynchronously with the driver/replay insertion loop, two separate processes can sample their first batch at slightly different replay states, even when:

- env seed is fixed,
- model seed is fixed,
- action sequence initially matches,
- update scheduler matches.

Replay also has:

```yaml
replay.online: True
```

so train sampling can consume from the online replay queue. That makes the first sampled batch especially timing-sensitive.

## What Was Added

The following debug-only switches were added. They are inactive unless explicitly set.

### `PAPER_DETERMINISM_TRACE=1`

Writes:

```text
paper_artifacts/determinism/action_trace.jsonl
paper_artifacts/determinism/batch_trace.jsonl
```

Action trace logs per environment step:

- step,
- worker,
- optimizer update count,
- transition hash,
- action hash,
- reward,
- first/last/terminal flags,
- action value.

Batch trace logs per optimizer update:

- step,
- update index in step,
- optimizer updates before/after,
- full batch hash,
- stepid hash,
- reward/action/is_first/is_last/is_terminal hashes,
- selected loss/optimizer metrics after update.

### `PAPER_DETERMINISTIC_UUID=1`

Calls:

```python
elements.UUID.reset(debug=True)
```

This makes replay chunk UUIDs deterministic for forensic runs. It should be used only for debug/reproducibility checks, not as a claim about official paper protocol.

### `PAPER_DISABLE_STREAM_PREFETCH=1`

Switches the agent training stream from asynchronous prefetch to synchronous map:

```python
embodied.streams.Map(st, fn)
```

This disables the replay-sampling race during determinism checks.

## Validation

With prefetch disabled:

```text
PAPER_DISABLE_STREAM_PREFETCH=1
PAPER_DETERMINISM_TRACE=1
PAPER_DETERMINISTIC_UUID=1
```

The 3K-step Breakout repeat test produced:

```text
action_trace: exact match
batch_trace: exact match
episode-score: exact match
```

Report:

```text
paper_artifacts/atari_v22_paper_core_control_prefix/v22_determinism_topk_detail_lambda1_breakout_seed0_steps3000_noops0_tr256_onlineTrue_trace1_uuid1_prefetch1_cuda_bfloat16_jitTrue.json
```

Note: scalar train logs can still appear at different wall/logging boundaries in some settings. For determinism, action and batch traces are the authoritative evidence.

## Learning Rate Check

The active optimizer config was:

```yaml
lr: 4e-5
schedule: const
warmup: 1000
anneal: 0
```

`schedule: const` still includes a 1000-update warmup because the code wraps the constant schedule with:

```python
optax.linear_schedule(0.0, lr, warmup)
optax.join_schedules([ramp, sched], [warmup])
```

However, LR scheduling was not the mismatch source because:

- optimizer update schedule matched across repeats,
- first update step matched,
- batch mismatch appeared before any update-induced model drift.

## Files Modified

```text
embodied/run/paper_artifacts.py
embodied/run/train.py
embodied/jax/agent.py
dreamerv3/main.py
paper_artifacts/atari_v22_paper_core_control_prefix/launch_v22_determinism_check.sh
paper_artifacts/atari_v22_paper_core_control_prefix/analyze_v22_determinism.py
```

## How To Run Determinism Forensics

From:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
```

Run with normal prefetch to reproduce the timing-sensitive mismatch:

```bash
GPU=0 \
GAME=breakout \
SEED=0 \
REPEATS='0 1' \
STEPS=14000 \
MODE=topk_detail_lambda1 \
WANDB_MODE=disabled \
NOOPS=0 \
CLEAN=1 \
TRACE=1 \
DETERMINISTIC_UUID=1 \
TRAIN_RATIO=256 \
REPLAY_ONLINE=True \
JAX_PLATFORM=cuda \
JAX_DTYPE=bfloat16 \
JAX_JIT=True \
paper_artifacts/atari_v22_paper_core_control_prefix/launch_v22_determinism_check.sh
```

Run with prefetch disabled to verify deterministic batch/action traces:

```bash
GPU=0 \
GAME=breakout \
SEED=0 \
REPEATS='0 1' \
STEPS=3000 \
MODE=topk_detail_lambda1 \
WANDB_MODE=disabled \
NOOPS=0 \
CLEAN=1 \
TRACE=1 \
DETERMINISTIC_UUID=1 \
DISABLE_PREFETCH=1 \
TRAIN_RATIO=256 \
REPLAY_ONLINE=True \
JAX_PLATFORM=cuda \
JAX_DTYPE=bfloat16 \
JAX_JIT=True \
paper_artifacts/atari_v22_paper_core_control_prefix/launch_v22_determinism_check.sh
```

## Guidance For Future Experiments

For normal stochastic/paper-style experiments:

- do not enable `PAPER_DETERMINISM_TRACE=1`; it is heavy,
- do not enable `PAPER_DETERMINISTIC_UUID=1` unless doing forensic replay checks,
- keep official behavior unless a deterministic ablation explicitly requires otherwise,
- do not interpret same-seed repeats with prefetch enabled as bitwise-reproducible.

For exact reproducibility debugging:

- use `NOOPS=0`,
- use `--env.atari100k.use_seed True`,
- use `PAPER_DETERMINISTIC_UUID=1`,
- use `PAPER_DETERMINISM_TRACE=1`,
- use `PAPER_DISABLE_STREAM_PREFETCH=1`,
- compare `action_trace.jsonl` and `batch_trace.jsonl`, not only scalar logs.

## Current Interpretation

Seed fixes control RNG, but they do not control asynchronous replay stream timing. In this codebase, asynchronous stream prefetch can make same-seed training runs diverge from the first optimizer batch. This is expected for official-style stochastic training, but it must be disabled or traced when testing bitwise determinism.
