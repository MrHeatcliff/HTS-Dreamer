#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk1/backup_user/dat.tt2/xuance"
REPO="$ROOT/external_baselines/dreamerv3-official"
PY="$ROOT/.venv/bin/python"
LOG_ROOT="$ROOT/logs/external_baselines/dreamerv3_official_hts_no_vc/hts_locked_hier_x3_no_vc"
ART="$REPO/paper_artifacts/atari_no_vc_ablation"

mkdir -p "$ART/run_logs" "$ROOT/tmp"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hts-wm-atari-dev}"
export WANDB_GROUP="${WANDB_GROUP:-no_vc_alien_breakout_probe}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-no_vc_5seed_probe}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${TMPDIR:-$ROOT/tmp}"

echo "No-VC Alien/Breakout probe started at $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "Log root: $LOG_ROOT"

for GAME in alien breakout; do
  for SEED in 0 1 2 3 4; do
    RUN="no_vc__hts_locked_hier_x3__${GAME}__seed${SEED}"
    LOGDIR="$LOG_ROOT/$GAME/seed_${SEED}"
    LOGFILE="$ART/run_logs/${RUN}.log"
    mkdir -p "$(dirname "$LOGDIR")"

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
    export WANDB_TAGS="no_vc,hts_locked_hier_x3,atari100k,${GAME},seed${SEED},no_video"
    export WANDB_RUN_NAME="$RUN"

    "$PY" -m dreamerv3.main_hts \
      --configs hts_atari100k size12m hts_no_vc \
      --task "atari100k_${GAME}" \
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
      --agent.report False \
      --logger.outputs jsonl,scope,wandb \
      --jax.prealloc False \
      --jax.jit True \
      2>&1 | tee "$LOGFILE"

    echo "===== DONE $RUN at $(date -Is) ====="
  done
done

echo "No-VC Alien/Breakout probe completed at $(date -Is)"
