#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk1/backup_user/dat.tt2/xuance"
REPO="$ROOT/external_baselines/dreamerv3-official"
PY="$ROOT/.venv/bin/python"
LOG_ROOT="$ROOT/logs/external_baselines/dreamerv3_official_hts_vc_cov_sweep/breakout_seed0"
ART="$REPO/paper_artifacts/atari_vc_cov_sweep"

mkdir -p "$ART/run_logs" "$ROOT/tmp"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hts-wm-atari-dev}"
export WANDB_GROUP="${WANDB_GROUP:-vc_cov_sweep_breakout_seed0}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-vc_cov_sweep}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${TMPDIR:-$ROOT/tmp}"

GAME="breakout"
SEED="${SEED:-0}"
SCALES=(0.001 0.003 0.01 0.03 0.1 0.3 1.0)

echo "VC covariance sweep started at $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "Game=$GAME Seed=$SEED"
echo "Scales=${SCALES[*]}"

for SCALE in "${SCALES[@]}"; do
  SAFE_SCALE="${SCALE//./p}"
  RUN="vc_cov_sweep__hts_full__${GAME}__seed${SEED}__cov${SAFE_SCALE}"
  LOGDIR="$LOG_ROOT/cov_${SAFE_SCALE}/seed_${SEED}"
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
  export WANDB_TAGS="vc_cov_sweep,cov_${SAFE_SCALE},hts_full,atari100k,breakout,seed${SEED},no_video"
  export WANDB_RUN_NAME="$RUN"

  "$PY" -m dreamerv3.main_hts \
    --configs hts_atari100k size12m \
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
    --agent.hts.vicreg_cov_scale "$SCALE" \
    --agent.report False \
    --logger.outputs jsonl,scope,wandb \
    --jax.prealloc False \
    --jax.jit True \
    2>&1 | tee "$LOGFILE"

  echo "===== DONE $RUN at $(date -Is) ====="
done

echo "VC covariance sweep completed at $(date -Is)"
