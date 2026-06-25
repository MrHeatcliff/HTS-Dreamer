#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
PY=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0}"
PHASE1_RAW="${PHASE1_RAW:-200000}"
PHASE2_RAW="${PHASE2_RAW:-400000}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
TRAIN_RATIO="${TRAIN_RATIO:-256}"
MODE="${MODE:-topk_detail_lambda1}"
OUT="$ROOT/paper_artifacts/atari_v22_paper_core_control_prefix/run_logs"
mkdir -p "$OUT"

if [[ "$PHASE1_RAW" != "200000" && "$PHASE1_RAW" != "400000" ]]; then
  echo "PHASE1_RAW must be 200000 or 400000 for this controlled comparison." >&2
  exit 2
fi

TOTAL_RAW=$((PHASE1_RAW + PHASE2_RAW))
TOTAL_AGENT_STEPS=$((TOTAL_RAW / ACTION_REPEAT))
PHASE1_AGENT_STEPS=$((PHASE1_RAW / ACTION_REPEAT))
PHASE2_AGENT_STEPS=$((PHASE2_RAW / ACTION_REPEAT))
MINIBATCH_STEPS=$((BATCH_SIZE * BATCH_LENGTH))
PHASE1_OPT_UPDATES=$((PHASE1_AGENT_STEPS * TRAIN_RATIO / MINIBATCH_STEPS))
PHASE2_OPT_UPDATES=$((PHASE2_AGENT_STEPS * TRAIN_RATIO / MINIBATCH_STEPS))

case "$MODE" in
  default_lambda1)
    BASE_CONFIG="hts_atari100k size12m hts_paper_core_zfull hts_paper_core_zfull_lambda1"
    MODE_TAG="default_lambda1"
    ;;
  topk_detail_lambda1)
    BASE_CONFIG="hts_atari100k size12m hts_paper_core_zfull_topk_detail hts_paper_core_zfull_topk_detail_lambda1"
    MODE_TAG="topk_detail_lambda1"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use MODE=default_lambda1 or MODE=topk_detail_lambda1." >&2
    exit 2
    ;;
esac

if [[ "$PHASE1_RAW" == "200000" ]]; then
  PHASE_CONFIG="hts_hard_phase1_200k_raw"
else
  PHASE_CONFIG="hts_hard_phase1_400k_raw"
fi

CONFIG="$BASE_CONFIG $PHASE_CONFIG"
BRANCH="two_phase_${MODE_TAG}_phase1_${PHASE1_RAW}_phase2_${PHASE2_RAW}"

cd "$ROOT"

for SEED in $SEEDS; do
  RUN_NAME="v22_${MODE_TAG}_hardphase1_${PHASE1_RAW}_phase2_${PHASE2_RAW}__breakout__seed${SEED}"
  LOGDIR="/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v22_paper_core/${BRANCH}/breakout/seed_${SEED}"
  echo "===== RUN ${RUN_NAME} ====="
  echo "CONFIG=${CONFIG}"
  echo "PHASE1_RAW=${PHASE1_RAW}"
  echo "PHASE2_RAW=${PHASE2_RAW}"
  echo "TOTAL_RAW=${TOTAL_RAW}"
  echo "PHASE1_AGENT_STEPS=${PHASE1_AGENT_STEPS}"
  echo "PHASE2_AGENT_STEPS=${PHASE2_AGENT_STEPS}"
  echo "TOTAL_AGENT_STEPS=${TOTAL_AGENT_STEPS}"
  echo "PHASE1_OPT_UPDATES=${PHASE1_OPT_UPDATES}"
  echo "PHASE2_OPT_UPDATES=${PHASE2_OPT_UPDATES}"
  echo "LOGDIR=${LOGDIR}"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export WANDB_MODE=online
  export WANDB_PROJECT=hts-wm-atari-dev
  export WANDB_GROUP="v22_two_phase_${MODE_TAG}_breakout"
  export WANDB_JOB_TYPE=v22_two_phase_breakout
  export WANDB_TAGS="v22,paper_core,zfull,two_phase,hard_phase1,phase1_${PHASE1_RAW},phase2_${PHASE2_RAW},${MODE_TAG},no_video"
  export WANDB_RUN_NAME="$RUN_NAME"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp
  "$PY" -m dreamerv3.main_hts \
    --configs $CONFIG \
    --task atari100k_breakout \
    --seed "$SEED" \
    --logdir "$LOGDIR" \
    --run.steps "$TOTAL_AGENT_STEPS" \
    --run.envs 1 \
    --run.train_ratio "$TRAIN_RATIO" \
    --run.log_every 250 \
    --run.report_every 999999 \
    --run.log_policy_video False \
    --run.save_every 10000 \
    --batch_size "$BATCH_SIZE" \
    --batch_length "$BATCH_LENGTH" \
    --agent.hts.phase1_steps "$PHASE1_OPT_UPDATES" \
    --agent.hts.phase2_steps "$PHASE2_OPT_UPDATES" \
    --agent.report False \
    --logger.outputs jsonl,scope,wandb \
    --jax.platform cuda \
    --jax.prealloc False \
    --jax.jit True \
    2>&1 | tee "$OUT/${RUN_NAME}.log"
done
