# Upload DreamerV3 Full26 Local Logs to W&B

Local source:

```text
/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m
```

Selected manifest:

```text
full26_selected_seed_metrics.json
```

The uploader replays local `metrics.jsonl`/`scores.jsonl` into W&B. It does not
train anything.

## Recommended: Project Per Game/Data

This creates 26 projects:

```text
dreamv3-alien
dreamv3-breakout
...
```

Dry-run first:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

DRY_RUN=1 \
PROJECT_MODE=per_game \
PROJECT_PREFIX=dreamv3 \
DATASET=atari100k \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_full26_dreamerv3_to_wandb.sh
```

Upload:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

DRY_RUN=0 \
PROJECT_MODE=per_game \
PROJECT_PREFIX=dreamv3 \
DATASET=atari100k \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_full26_dreamerv3_to_wandb.sh
```

## Alternative: Existing Single Project

This uploads all games into the old existing project, grouped by `atari100k/<game>`:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

DRY_RUN=0 \
PROJECT_MODE=single \
PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
DATASET=atari100k \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_full26_dreamerv3_to_wandb.sh
```

## Subset Upload

Only Alien and Breakout:

```bash
DRY_RUN=0 \
GAMES=alien,breakout \
PROJECT_MODE=per_game \
PROJECT_PREFIX=dreamv3 \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_full26_dreamerv3_to_wandb.sh
```

Only seed 0:

```bash
DRY_RUN=0 \
SEEDS=0 \
PROJECT_MODE=per_game \
PROJECT_PREFIX=dreamv3 \
external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_full26_dreamerv3_to_wandb.sh
```

## Uploaded Metrics

Each seed becomes one W&B run with:

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

W&B step is raw Atari environment frames.
