#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk1/backup_user/dat.tt2/xuance
PY="${PY:-$ROOT/.venv/bin/python}"
SCRIPT="$ROOT/external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_full26_dreamerv3_to_wandb.py"

GAMES="${GAMES:-}"
SEEDS="${SEEDS:-}"
DATASET="${DATASET:-atari100k}"
PROJECT_MODE="${PROJECT_MODE:-per_game}"
PROJECT_PREFIX="${PROJECT_PREFIX:-dreamv3}"
PROJECT="${PROJECT:-dreamv3-atari100k-full26}"
UPLOAD_TAG="${UPLOAD_TAG:-local-replay-v1}"
DRY_RUN="${DRY_RUN:-1}"
SAVE_SOURCE_FILES="${SAVE_SOURCE_FILES:-0}"
LOG_GAME_CURVE_IMAGE="${LOG_GAME_CURVE_IMAGE:-1}"

ARGS=(
  --dataset "$DATASET"
  --project-mode "$PROJECT_MODE"
  --project-prefix "$PROJECT_PREFIX"
  --project "$PROJECT"
  --upload-tag "$UPLOAD_TAG"
)

if [[ -n "$GAMES" ]]; then
  ARGS+=(--games "$GAMES")
fi

if [[ -n "$SEEDS" ]]; then
  ARGS+=(--seeds "$SEEDS")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi

if [[ "$SAVE_SOURCE_FILES" == "1" ]]; then
  ARGS+=(--save-source-files)
fi

if [[ "$LOG_GAME_CURVE_IMAGE" == "1" ]]; then
  ARGS+=(--log-game-curve-image)
fi

echo "PY=$PY"
echo "PROJECT_MODE=$PROJECT_MODE"
echo "PROJECT_PREFIX=$PROJECT_PREFIX"
echo "PROJECT=$PROJECT"
echo "DATASET=$DATASET"
echo "GAMES=${GAMES:-ALL}"
echo "SEEDS=${SEEDS:-ALL}"
echo "DRY_RUN=$DRY_RUN"

"$PY" "$SCRIPT" "${ARGS[@]}"
