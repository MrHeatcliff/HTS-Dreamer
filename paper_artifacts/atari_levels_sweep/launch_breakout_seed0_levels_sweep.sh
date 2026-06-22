#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk1/backup_user/dat.tt2/xuance"
REPO="$ROOT/external_baselines/dreamerv3-official"
PY="$ROOT/.venv/bin/python"
LOG_ROOT="$ROOT/logs/external_baselines/dreamerv3_official_hts_levels_sweep/breakout_seed0_cov0p001"
ART="$REPO/paper_artifacts/atari_levels_sweep"

mkdir -p "$ART/run_logs" "$ROOT/tmp"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hts-wm-atari-dev}"
export WANDB_GROUP="${WANDB_GROUP:-levels_sweep_breakout_seed0_cov0p001}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-levels_sweep}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${TMPDIR:-$ROOT/tmp}"

GAME="breakout"
SEED="${SEED:-0}"
COV_SCALE="${COV_SCALE:-0.001}"
LEVELS=(2 4 6 8)

echo "HTS levels sweep started at $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "Game=$GAME Seed=$SEED CovScale=$COV_SCALE"
echo "Levels=${LEVELS[*]}"

for LEVEL in "${LEVELS[@]}"; do
  RUN="levels_sweep__hts_full__${GAME}__seed${SEED}__L${LEVEL}__cov0p001"
  LOGDIR="$LOG_ROOT/levels_${LEVEL}/seed_${SEED}"
  LOGFILE="$ART/run_logs/${RUN}.log"

  if [[ -f "$LOGDIR/scores.jsonl" ]] && "$PY" - "$LOGDIR/scores.jsonl" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
ok = rows and max(float(r.get("step", 0)) for r in rows) >= 418000
raise SystemExit(0 if ok else 1)
PY
  then
    echo "===== SKIP complete $RUN ====="
    continue
  fi

  echo "===== START $RUN at $(date -Is) ====="
  export WANDB_TAGS="levels_sweep,levels_${LEVEL},cov0p001,hts_full,atari100k,breakout,seed${SEED},no_video"
  export WANDB_RUN_NAME="$RUN"

  "$PY" -m dreamerv3.main_hts \
    --configs hts_atari100k size12m "hts_levels_${LEVEL}" \
    --task atari100k_breakout \
    --seed "$SEED" \
    --logdir "$LOGDIR" \
    --run.steps 110000 \
    --run.envs 1 \
    --run.train_ratio 256 \
    --run.log_every 250 \
    --run.report_every 999999 \
    --run.log_policy_video False \
    --run.save_every 10000 \
    --batch_size 16 \
    --batch_length 64 \
    --agent.hts.l_hier 0.3 \
    --agent.hts.vicreg_cov_scale "$COV_SCALE" \
    --agent.report False \
    --logger.outputs jsonl,scope,wandb \
    --jax.prealloc False \
    --jax.jit True \
    2>&1 | tee "$LOGFILE"

  echo "===== DONE $RUN at $(date -Is) ====="
done

echo "HTS levels sweep completed at $(date -Is)"
