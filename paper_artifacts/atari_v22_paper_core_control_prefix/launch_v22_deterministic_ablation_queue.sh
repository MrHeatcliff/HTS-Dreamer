#!/usr/bin/env bash
set -euo pipefail

# Sequential deterministic V22 HTS ablation queue.
#
# Default phase units are raw Atari environment steps. The requested controlled
# comparison uses a fixed total of 400K env steps, equivalent to 100K agent
# actions under action repeat 4. Set PHASE_UNIT=agent only if you explicitly
# want post-action-repeat agent-action horizons.

ROOT="${ROOT:-/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official}"
PY="${PY:-/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python}"
GPU="${GPU:-0}"
GAME="${GAME:-breakout}"
SEEDS="${SEEDS:-0}"

PHASE_UNIT="${PHASE_UNIT:-raw}"  # raw or agent
TOTAL_RAW_STEPS="${TOTAL_RAW_STEPS:-400000}"
TOTAL_AGENT_STEPS="${TOTAL_AGENT_STEPS:-100000}"
PHASE1_STEPS_LIST="${PHASE1_STEPS_LIST:-50000 100000 200000}"

ACTION_REPEAT="${ACTION_REPEAT:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
TRAIN_RATIO="${TRAIN_RATIO:-256}"
REPLAY_ONLINE="${REPLAY_ONLINE:-True}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-hts-wm-atari-dev}"
LOG_EVERY="${LOG_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
CLEAN="${CLEAN:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Determinism knobs. TRACE=1 writes large action/batch trace files; keep it for
# short forensic runs only.
TRACE="${TRACE:-0}"
DETERMINISTIC_UUID="${DETERMINISTIC_UUID:-1}"
DISABLE_PREFETCH="${DISABLE_PREFETCH:-1}"
NOOPS="${NOOPS:-0}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda}"
JAX_JIT="${JAX_JIT:-True}"
JAX_DTYPE="${JAX_DTYPE:-bfloat16}"

CONFIG="hts_atari100k size12m hts_paper_core_zfull_topk_detail hts_paper_core_zfull_topk_detail_lambda1"
OUT="$ROOT/paper_artifacts/atari_v22_paper_core_control_prefix/run_logs"
LOG_ROOT="${LOG_ROOT:-/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts_v22_paper_core/deterministic_ablation}"
mkdir -p "$OUT"

if [[ "$PHASE_UNIT" != "raw" && "$PHASE_UNIT" != "agent" ]]; then
  echo "PHASE_UNIT must be raw or agent, got: $PHASE_UNIT" >&2
  exit 2
fi

if (( ACTION_REPEAT <= 0 )); then
  echo "ACTION_REPEAT must be positive." >&2
  exit 2
fi

MINIBATCH_STEPS=$((BATCH_SIZE * BATCH_LENGTH))
if (( MINIBATCH_STEPS <= 0 )); then
  echo "BATCH_SIZE * BATCH_LENGTH must be positive." >&2
  exit 2
fi

if [[ "$PHASE_UNIT" == "raw" ]]; then
  if (( TOTAL_RAW_STEPS % ACTION_REPEAT != 0 )); then
    echo "TOTAL_RAW_STEPS must be divisible by ACTION_REPEAT." >&2
    exit 2
  fi
  RUN_AGENT_STEPS=$((TOTAL_RAW_STEPS / ACTION_REPEAT))
  TOTAL_LABEL="${TOTAL_RAW_STEPS}raw"
else
  RUN_AGENT_STEPS="$TOTAL_AGENT_STEPS"
  TOTAL_RAW_STEPS=$((TOTAL_AGENT_STEPS * ACTION_REPEAT))
  TOTAL_LABEL="${TOTAL_AGENT_STEPS}agent"
fi

TASK="atari100k_${GAME}"
cd "$ROOT"

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_one() {
  local variant="$1"
  local seed="$2"
  shift 2
  local -a extra_args=("$@")
  local run_name="v22_detabl_${variant}__${GAME}__seed${seed}__total${TOTAL_LABEL}"
  local logdir="${LOG_ROOT}/${GAME}/${variant}/seed_${seed}/total_${TOTAL_LABEL}_tr${TRAIN_RATIO}_online${REPLAY_ONLINE}_prefetch${DISABLE_PREFETCH}"

  echo "===== RUN ${run_name} ====="
  echo "CONFIG=${CONFIG}"
  echo "TASK=${TASK}"
  echo "VARIANT=${variant}"
  echo "SEED=${seed}"
  echo "PHASE_UNIT=${PHASE_UNIT}"
  echo "TOTAL_RAW_STEPS=${TOTAL_RAW_STEPS}"
  echo "RUN_AGENT_STEPS=${RUN_AGENT_STEPS}"
  echo "TRAIN_RATIO=${TRAIN_RATIO}"
  echo "REPLAY_ONLINE=${REPLAY_ONLINE}"
  echo "NOOPS=${NOOPS}"
  echo "DETERMINISTIC_UUID=${DETERMINISTIC_UUID}"
  echo "DISABLE_PREFETCH=${DISABLE_PREFETCH}"
  echo "TRACE=${TRACE}"
  echo "LOGDIR=${logdir}"

  if [[ "$CLEAN" == "1" ]]; then
    echo "Removing old logdir: ${logdir}"
    rm -rf "$logdir"
  fi

  export CUDA_VISIBLE_DEVICES="$GPU"
  export WANDB_MODE="$WANDB_MODE"
  export WANDB_PROJECT="$WANDB_PROJECT"
  export WANDB_GROUP="v22_deterministic_ablation_${GAME}"
  export WANDB_JOB_TYPE="v22_deterministic_ablation"
  export WANDB_TAGS="v22,deterministic_ablation,${GAME},${variant},seed_${seed},noops_${NOOPS},no_video,no_prefetch_${DISABLE_PREFETCH},uuid_${DETERMINISTIC_UUID}"
  export WANDB_RUN_NAME="$run_name"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PAPER_DETERMINISM_TRACE="$TRACE"
  export PAPER_DETERMINISTIC_UUID="$DETERMINISTIC_UUID"
  export PAPER_DISABLE_STREAM_PREFETCH="$DISABLE_PREFETCH"
  export TMPDIR=/mnt/disk1/backup_user/dat.tt2/xuance/tmp

  local -a configs
  read -r -a configs <<< "$CONFIG"

  local -a cmd=(
    "$PY" -m dreamerv3.main_hts
    --configs "${configs[@]}"
    --task "$TASK"
    --seed "$seed"
    --logdir "$logdir"
    --run.steps "$RUN_AGENT_STEPS"
    --run.envs 1
    --run.train_ratio "$TRAIN_RATIO"
    --run.log_every "$LOG_EVERY"
    --run.report_every 999999
    --run.log_policy_video False
    --run.save_every "$SAVE_EVERY"
    --run.debug True
    --env.atari100k.noops "$NOOPS"
    --env.atari100k.use_seed True
    --replay.online "$REPLAY_ONLINE"
    --batch_size "$BATCH_SIZE"
    --batch_length "$BATCH_LENGTH"
    --agent.report False
    --logger.outputs jsonl,scope,wandb
    --jax.platform "$JAX_PLATFORM"
    --jax.compute_dtype "$JAX_DTYPE"
    --jax.prealloc False
    --jax.jit "$JAX_JIT"
    --jax.deterministic True
    "${extra_args[@]}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN command:"
    print_cmd "${cmd[@]}"
  else
    "${cmd[@]}" 2>&1 | tee "$OUT/${run_name}.log"
  fi
}

phase_to_agent_steps() {
  local value="$1"
  if [[ "$PHASE_UNIT" == "raw" ]]; then
    if (( value % ACTION_REPEAT != 0 )); then
      echo "PHASE1 raw step value must be divisible by ACTION_REPEAT: ${value}" >&2
      exit 2
    fi
    echo $((value / ACTION_REPEAT))
  else
    echo "$value"
  fi
}

phase_label() {
  local value="$1"
  if [[ "$PHASE_UNIT" == "raw" ]]; then
    echo "${value}raw"
  else
    echo "${value}agent"
  fi
}

for SEED in $SEEDS; do
  run_one "joint" "$SEED" \
    --agent.hts.training_regime joint_online_initial \
    --agent.hts.aux_warmup_mode hard \
    --agent.hts.aux_warmup_raw_frames 0 \
    --agent.hts.aux_warmup_agent_actions 0 \
    --agent.hts.aux_warmup_optimizer_updates 0 \
    --agent.hts.phase1_steps 0 \
    --agent.hts.phase2_steps 0

  run_one "joint_no_hier_loss" "$SEED" \
    --agent.hts.training_regime joint_online_initial \
    --agent.hts.use_lhier False \
    --agent.hts.l_hier 0.0 \
    --agent.hts.aux_warmup_mode hard \
    --agent.hts.aux_warmup_raw_frames 0 \
    --agent.hts.aux_warmup_agent_actions 0 \
    --agent.hts.aux_warmup_optimizer_updates 0 \
    --agent.hts.phase1_steps 0 \
    --agent.hts.phase2_steps 0

  for PHASE1_STEPS in $PHASE1_STEPS_LIST; do
    PHASE1_AGENT_STEPS="$(phase_to_agent_steps "$PHASE1_STEPS")"
    if (( PHASE1_AGENT_STEPS >= RUN_AGENT_STEPS )); then
      echo "PHASE1 must be smaller than total run steps: phase1_agent=${PHASE1_AGENT_STEPS}, total_agent=${RUN_AGENT_STEPS}" >&2
      exit 2
    fi
    PHASE2_AGENT_STEPS=$((RUN_AGENT_STEPS - PHASE1_AGENT_STEPS))
    PHASE1_OPT_UPDATES=$((PHASE1_AGENT_STEPS * TRAIN_RATIO / MINIBATCH_STEPS))
    PHASE2_OPT_UPDATES=$((PHASE2_AGENT_STEPS * TRAIN_RATIO / MINIBATCH_STEPS))
    PHASE1_LABEL="$(phase_label "$PHASE1_STEPS")"

    echo "Computed two-phase schedule:"
    echo "  phase1=${PHASE1_LABEL}"
    echo "  phase1_agent_steps=${PHASE1_AGENT_STEPS}"
    echo "  phase2_agent_steps=${PHASE2_AGENT_STEPS}"
    echo "  phase1_opt_updates=${PHASE1_OPT_UPDATES}"
    echo "  phase2_opt_updates=${PHASE2_OPT_UPDATES}"

    run_one "two_phase_phase1_${PHASE1_LABEL}" "$SEED" \
      --agent.hts.training_regime two_phase \
      --agent.hts.aux_warmup_mode hard \
      --agent.hts.aux_warmup_raw_frames 0 \
      --agent.hts.aux_warmup_agent_actions "$PHASE1_AGENT_STEPS" \
      --agent.hts.aux_warmup_optimizer_updates 0 \
      --agent.hts.aux_warmup_action_repeat "$ACTION_REPEAT" \
      --agent.hts.aux_warmup_train_ratio "$TRAIN_RATIO" \
      --agent.hts.phase1_steps "$PHASE1_OPT_UPDATES" \
      --agent.hts.phase2_steps "$PHASE2_OPT_UPDATES"
  done
done
