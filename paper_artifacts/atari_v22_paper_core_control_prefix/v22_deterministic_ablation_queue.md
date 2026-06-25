# V22 Deterministic Ablation Queue

This queue runs the current V22 HTS paper-core config sequentially:

1. `joint`: joint HTS training baseline.
2. `joint_no_hier_loss`: same as joint, but disables hierarchical reconstruction loss.
3. `two_phase_phase1_50000raw`.
4. `two_phase_phase1_100000raw`.
5. `two_phase_phase1_200000raw`.

Default schedule uses raw Atari environment steps for the phase labels and total
budget:

```text
ACTION_REPEAT=4
TOTAL_RAW_STEPS=400000
run.steps=100000 agent actions
raw Atari env steps = 400000
TRAIN_RATIO=256
batch_size x batch_length = 16 x 64
```

The two phases always sum to `TOTAL_RAW_STEPS`. For the default list, the
phase splits are:

```text
50K raw + 350K raw
100K raw + 300K raw
200K raw + 200K raw
```

The script converts phase-1 raw env-step horizons to agent actions and optimizer
updates before launching DreamerV3.

## Run

From the outer repo:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

GPU=0 \
GAME=breakout \
SEEDS="0" \
WANDB_MODE=online \
WANDB_PROJECT=hts-wm-atari-dev \
PHASE_UNIT=raw \
TOTAL_RAW_STEPS=400000 \
PHASE1_STEPS_LIST="50000 100000 200000" \
external_baselines/dreamerv3-official/paper_artifacts/atari_v22_paper_core_control_prefix/launch_v22_deterministic_ablation_queue.sh
```

Use `DRY_RUN=1` to print all commands without launching training.

```bash
DRY_RUN=1 \
WANDB_MODE=disabled \
PHASE1_STEPS_LIST="50000" \
external_baselines/dreamerv3-official/paper_artifacts/atari_v22_paper_core_control_prefix/launch_v22_deterministic_ablation_queue.sh
```

## Determinism Flags

The queue enables:

```text
--run.debug True
--env.atari100k.use_seed True
--env.atari100k.noops 0
--jax.deterministic True
PAPER_DETERMINISTIC_UUID=1
PAPER_DISABLE_STREAM_PREFETCH=1
```

`PAPER_DISABLE_STREAM_PREFETCH=1` is important. The V22 determinism audit found
that replay stream prefetch can change the first sampled optimizer batch even
when model and environment seeds are fixed.

`TRACE=0` by default. Set `TRACE=1` only for short debug runs because it writes
large action and batch trace artifacts.

## Agent-Action Phase Variant

If the phase horizons should mean post-action-repeat agent actions instead:

```bash
PHASE_UNIT=agent \
TOTAL_AGENT_STEPS=100000 \
PHASE1_STEPS_LIST="12500 25000 50000" \
external_baselines/dreamerv3-official/paper_artifacts/atari_v22_paper_core_control_prefix/launch_v22_deterministic_ablation_queue.sh
```
