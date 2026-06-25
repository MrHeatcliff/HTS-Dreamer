#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
PY=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python
GPU="${GPU:-0}"
STEPS="${STEPS:-110000}"
SEEDS="${SEEDS:-0}"
MODE="${MODE:-default}"
OUT="$ROOT/paper_artifacts/atari_v22_paper_core_control_prefix/run_logs"
mkdir -p "$OUT"

case "$MODE" in
  default)
    CONFIG="hts_atari100k size12m hts_paper_core_zfull hts_paper_core_zfull_lambda1"
    LOG_BRANCH="lambda1_stage_a"
    WANDB_GROUP="v22_lambda1_breakout_stage_a"
    TAGS="v22,paper_core,zfull,lambda1,stage_a,no_video"
    ;;
  topk_detail)
    CONFIG="hts_atari100k size12m hts_paper_core_zfull_topk_detail hts_paper_core_zfull_topk_detail_lambda1"
    LOG_BRANCH="topk_detail_lambda1_stage_a"
    WANDB_GROUP="v22_topk_detail_lambda1_breakout_stage_a"
    TAGS="v22,paper_core,zfull,topk_detail,lambda1,stage_a,no_video"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use MODE=default or MODE=topk_detail." >&2
    exit 2
    ;;
esac

cd "$ROOT"

for SEED in $SEEDS; do
  RUN_NAME="v22_${MODE}_lambda1__breakout__seed${SEED}"
  LOGDIR="/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v22_paper_core/${LOG_BRANCH}/breakout/seed_${SEED}"
  echo "===== RUN ${RUN_NAME} ====="
  echo "CONFIG=${CONFIG}"
  echo "LOGDIR=${LOGDIR}"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export WANDB_MODE=online
  export WANDB_PROJECT=hts-wm-atari-dev
  export WANDB_GROUP="$WANDB_GROUP"
  export WANDB_JOB_TYPE=v22_lambda1_stage_a
  export WANDB_TAGS="$TAGS"
  export WANDB_RUN_NAME="$RUN_NAME"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp
  "$PY" -m dreamerv3.main_hts \
    --configs $CONFIG \
    --task atari100k_breakout \
    --seed "$SEED" \
    --logdir "$LOGDIR" \
    --run.steps "$STEPS" \
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
    --jax.platform cuda \
    --jax.prealloc False \
    --jax.jit True \
    2>&1 | tee "$OUT/${RUN_NAME}.log"
done
