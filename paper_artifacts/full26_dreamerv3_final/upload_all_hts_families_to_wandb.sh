#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk1/backup_user/dat.tt2/xuance
UPLOADER="$ROOT/external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_hts_runs_to_wandb.sh"

FAMILIES="${FAMILIES:-v20_hts no_vc cov_sweep levels_sweep v21_warmup}"
DRY_RUN="${DRY_RUN:-1}"
PROJECT_PREFIX="${PROJECT_PREFIX:-dreamv3}"
PROJECT_MODE="${PROJECT_MODE:-per_game}"
GAMES="${GAMES:-}"
SEEDS="${SEEDS:-}"
UPLOAD_TAG="${UPLOAD_TAG:-local-replay-v1}"

for FAMILY in $FAMILIES; do
  echo "===== UPLOAD FAMILY: $FAMILY ====="
  FAMILY="$FAMILY" \
  DRY_RUN="$DRY_RUN" \
  PROJECT_PREFIX="$PROJECT_PREFIX" \
  PROJECT_MODE="$PROJECT_MODE" \
  GAMES="$GAMES" \
  SEEDS="$SEEDS" \
  UPLOAD_TAG="$UPLOAD_TAG" \
  "$UPLOADER"
done
