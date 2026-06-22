#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hts-wm-atari-dev}"
export WANDB_GROUP="${WANDB_GROUP:-v21_aux_warmup_breakout}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-v21_stage_a_repair_finalbin}"
export WANDB_TAGS="${WANDB_TAGS:-v21,aux_warmup,warmup50k,locked_hier_x3,atari100k,breakout,no_video,repair_finalbin}"
export WANDB_RUN_NAME="v21__hts_hier_x3_warmup50k__breakout__seed1_repair_finalbin"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${TMPDIR:-/mnt/disk1/backup_user/dat.tt2/xuance/tmp}"
mkdir -p "$TMPDIR" paper_artifacts/atari_v21_aux_warmup/run_logs

LOGDIR="/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v21_aux_warmup/hts_locked_hier_x3_warmup_50k_raw/breakout/seed_1_repair_finalbin"
LOGFILE="/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/atari_v21_aux_warmup/run_logs/v21__hts_hier_x3_warmup50k__breakout__seed1_repair_finalbin.log"
rm -rf "$LOGDIR"

/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts \
  --configs hts_atari100k size12m hts_warmup_50k_raw \
  --task atari100k_breakout \
  --seed 1 \
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
  --agent.hts.l_hier 0.3 \
  --agent.report False \
  --logger.outputs jsonl,scope,wandb \
  --jax.prealloc False \
  --jax.jit True \
  2>&1 | tee "$LOGFILE"
