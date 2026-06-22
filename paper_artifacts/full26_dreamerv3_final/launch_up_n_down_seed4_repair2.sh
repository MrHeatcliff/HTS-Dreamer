#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-HTS-WM-HarmonyDream-Alien-Curve}"
export WANDB_GROUP="${WANDB_GROUP:-dreamerv3-official-full26-size12m-repair2}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-full26_repair_up_n_down_seed4_finalbin}"
export WANDB_TAGS="${WANDB_TAGS:-dreamerv3,official,atari100k,size12m,full26,repair2,no_video,up_n_down}"
export WANDB_RUN_NAME="DreamerV3-official-size12m-atari100k-up_n_down-seed4-repair2-no-video"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${TMPDIR:-/mnt/disk1/backup_user/dat.tt2/xuance/tmp}"
mkdir -p "$TMPDIR"

LOGDIR="/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m/up_n_down/seed_4_repair2_no_video"
LOGFILE="/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/run_logs/DreamerV3-official-size12m-atari100k-up_n_down-seed4-repair2-no-video.log"

rm -rf "$LOGDIR"
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main \
  --configs atari100k size12m \
  --task atari100k_up_n_down \
  --seed 4 \
  --logdir "$LOGDIR" \
  --run.steps 125000 \
  --run.envs 1 \
  --run.train_ratio 256 \
  --run.log_every 250 \
  --run.report_every 999999 \
  --run.log_policy_video False \
  --run.save_every 10000 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.report False \
  --logger.outputs jsonl,scope,wandb \
  --jax.prealloc False \
  --jax.jit True \
  2>&1 | tee "$LOGFILE"
