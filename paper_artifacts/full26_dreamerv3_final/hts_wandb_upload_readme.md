# Upload HTS Runs to the Same W&B Projects as DreamerV3

Target project naming:

```text
dreamv3-<game>
```

So DreamerV3 and HTS for Alien both appear in:

```text
dreamv3-alien
```

The uploader replays local JSONL logs and 20-bin curve metrics. It does not
train anything.

## Matched Metrics

HTS upload uses the same metric names as DreamerV3 upload:

```text
episode/score
episode/length
episode_score
raw_frames
agent_actions_est
curve_20bin/score
curve_20bin/bin_index
curve_20bin/bin_count
summary/final_20pct_mean
summary/final_bin_mean
summary/auc_20bin_mean
summary/latest_episode_score
```

W&B step is raw Atari env frames for both DreamerV3 and HTS.

## Families

Available HTS families:

```text
v20_hts       # main locked_hier_x3 Alien/Breakout, 5 seeds each
no_vc         # no-VC Alien/Breakout, 5 seeds each
cov_sweep     # Breakout seed0 covariance sweep
levels_sweep  # Breakout seed0 levels sweep
v21_warmup    # Breakout auxiliary warmup variants
```

## Dry-Run All

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

DRY_RUN=1 \
PROJECT_PREFIX=dreamv3 \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_all_hts_families_to_wandb.sh
```

## Upload All

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

DRY_RUN=0 \
PROJECT_PREFIX=dreamv3 \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_all_hts_families_to_wandb.sh
```

## Upload Main HTS Only

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

DRY_RUN=0 \
FAMILY=v20_hts \
PROJECT_PREFIX=dreamv3 \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_hts_runs_to_wandb.sh
```

## Upload A Subset

Alien only:

```bash
DRY_RUN=0 \
FAMILY=v20_hts \
GAMES=alien \
PROJECT_PREFIX=dreamv3 \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_hts_runs_to_wandb.sh
```

Seed 0 only:

```bash
DRY_RUN=0 \
FAMILY=v20_hts \
SEEDS=0 \
PROJECT_PREFIX=dreamv3 \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_hts_runs_to_wandb.sh
```

