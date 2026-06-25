#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
PY=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python
GPU="${GPU:-0}"
GAME="${GAME:-breakout}"
SEED="${SEED:-0}"
REPEATS="${REPEATS:-0 1}"
STEPS="${STEPS:-20000}"
MODE="${MODE:-topk_detail_lambda1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
NOOPS="${NOOPS:-0}"
CLEAN="${CLEAN:-0}"
TRAIN_RATIO="${TRAIN_RATIO:-256}"
REPLAY_ONLINE="${REPLAY_ONLINE:-True}"
TRACE="${TRACE:-0}"
DETERMINISTIC_UUID="${DETERMINISTIC_UUID:-0}"
DISABLE_PREFETCH="${DISABLE_PREFETCH:-0}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda}"
JAX_JIT="${JAX_JIT:-True}"
JAX_DTYPE="${JAX_DTYPE:-bfloat16}"
OUT="$ROOT/paper_artifacts/atari_v22_paper_core_control_prefix/run_logs"
mkdir -p "$OUT"

case "$MODE" in
  default_lambda1)
    CONFIG="hts_atari100k size12m hts_paper_core_zfull hts_paper_core_zfull_lambda1"
    ;;
  topk_detail_lambda1)
    CONFIG="hts_atari100k size12m hts_paper_core_zfull_topk_detail hts_paper_core_zfull_topk_detail_lambda1"
    ;;
  baseline_dreamerv3)
    CONFIG="atari100k size12m"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use MODE=default_lambda1, topk_detail_lambda1, or baseline_dreamerv3." >&2
    exit 2
    ;;
esac

TASK="atari100k_${GAME}"
BASE_LOGDIR="/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v22_paper_core/determinism/${MODE}/${GAME}/seed_${SEED}/steps_${STEPS}_noops_${NOOPS}_tr${TRAIN_RATIO}_online${REPLAY_ONLINE}_trace${TRACE}_uuid${DETERMINISTIC_UUID}_prefetch${DISABLE_PREFETCH}_${JAX_PLATFORM}_${JAX_DTYPE}_jit${JAX_JIT}"

cd "$ROOT"

if [[ "$CLEAN" == "1" ]]; then
  echo "Removing old determinism logdir: $BASE_LOGDIR"
  rm -rf "$BASE_LOGDIR"
fi

for REP in $REPEATS; do
  RUN_NAME="v22_determinism__${MODE}__${GAME}__seed${SEED}__repeat${REP}"
  LOGDIR="$BASE_LOGDIR/repeat_${REP}"
  echo "===== RUN ${RUN_NAME} ====="
  echo "CONFIG=${CONFIG}"
  echo "TASK=${TASK}"
  echo "LOGDIR=${LOGDIR}"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export WANDB_MODE="$WANDB_MODE"
  export WANDB_PROJECT=hts-wm-atari-dev
  export WANDB_GROUP="v22_determinism_${MODE}_${GAME}"
  export WANDB_JOB_TYPE=v22_determinism
  export WANDB_TAGS="v22,determinism,${MODE},${GAME},seed_${SEED},noops_${NOOPS},no_video"
  export WANDB_RUN_NAME="$RUN_NAME"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PAPER_DETERMINISM_TRACE="$TRACE"
  export PAPER_DETERMINISTIC_UUID="$DETERMINISTIC_UUID"
  export PAPER_DISABLE_STREAM_PREFETCH="$DISABLE_PREFETCH"
  export TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp
  "$PY" -m dreamerv3.main_hts \
    --configs $CONFIG \
    --task "$TASK" \
    --seed "$SEED" \
    --logdir "$LOGDIR" \
    --run.steps "$STEPS" \
    --run.envs 1 \
    --run.train_ratio "$TRAIN_RATIO" \
    --run.log_every 100 \
    --run.report_every 999999 \
    --run.log_policy_video False \
    --run.save_every 999999 \
    --run.debug True \
    --env.atari100k.noops "$NOOPS" \
    --env.atari100k.use_seed True \
    --replay.online "$REPLAY_ONLINE" \
    --batch_size 16 \
    --batch_length 64 \
    --agent.report False \
    --logger.outputs jsonl,scope,wandb \
    --jax.platform "$JAX_PLATFORM" \
    --jax.compute_dtype "$JAX_DTYPE" \
    --jax.prealloc False \
    --jax.jit "$JAX_JIT" \
    --jax.deterministic True \
    2>&1 | tee "$OUT/${RUN_NAME}.log"
done

"$PY" "$ROOT/paper_artifacts/atari_v22_paper_core_control_prefix/analyze_v22_determinism.py" \
  --root "$BASE_LOGDIR" \
  --repeats "$REPEATS" \
  --output "$ROOT/paper_artifacts/atari_v22_paper_core_control_prefix/v22_determinism_${MODE}_${GAME}_seed${SEED}_steps${STEPS}_noops${NOOPS}_tr${TRAIN_RATIO}_online${REPLAY_ONLINE}_trace${TRACE}_uuid${DETERMINISTIC_UUID}_prefetch${DISABLE_PREFETCH}_${JAX_PLATFORM}_${JAX_DTYPE}_jit${JAX_JIT}.json"
