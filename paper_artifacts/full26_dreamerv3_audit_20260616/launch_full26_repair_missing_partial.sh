#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-HTS-WM-HarmonyDream-Alien-Curve}"
export WANDB_GROUP="${WANDB_GROUP:-dreamerv3-official-full26-size12m-repair}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-full26_repair_missing_partial}"
export WANDB_TAGS="${WANDB_TAGS:-dreamerv3,official,atari100k,size12m,full26,repair,no_video}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${TMPDIR:-/mnt/disk1/backup_user/dat.tt2/xuance/tmp}"

mkdir -p "$TMPDIR"
mkdir -p /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_audit_20260616/run_logs

run_one() {
  local game="$1"
  local seed="$2"
  local task="atari100k_${game}"
  local run_name="DreamerV3-official-size12m-atari100k-${game}-seed${seed}-repair-no-video"
  local logdir="/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m/${game}/seed_${seed}_repair_no_video"
  local logfile="/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_audit_20260616/run_logs/${run_name}.log"

  echo "===== START ${run_name} ====="
  export WANDB_RUN_NAME="$run_name"
  /mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main \
    --configs atari100k size12m \
    --task "$task" \
    --seed "$seed" \
    --logdir "$logdir" \
    --run.steps 110000 \
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
    2>&1 | tee "$logfile"
  echo "===== DONE ${run_name} ====="
}

# Partial or missing seeds from the 2026-06-16 audit.
run_one breakout 3
run_one road_runner 4
run_one seaquest 0
run_one seaquest 1
run_one seaquest 2
run_one seaquest 3
run_one seaquest 4
run_one up_n_down 0
run_one up_n_down 1
run_one up_n_down 2
run_one up_n_down 3
run_one up_n_down 4
