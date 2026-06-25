#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk1/backup_user/dat.tt2/xuance
PY="${PY:-$ROOT/.venv/bin/python}"
SCRIPT="$ROOT/external_baselines/dreamerv3-official/paper_artifacts/full26_dreamerv3_final/upload_atari_manifest_to_wandb.py"
ART="$ROOT/external_baselines/dreamerv3-official/paper_artifacts"

FAMILY="${FAMILY:-v20_hts}"
GAMES="${GAMES:-}"
SEEDS="${SEEDS:-}"
PROJECT_PREFIX="${PROJECT_PREFIX:-dreamv3}"
PROJECT_MODE="${PROJECT_MODE:-per_game}"
PROJECT="${PROJECT:-dreamv3-atari100k-full26}"
DRY_RUN="${DRY_RUN:-1}"
SAVE_SOURCE_FILES="${SAVE_SOURCE_FILES:-0}"
UPLOAD_TAG="${UPLOAD_TAG:-local-replay-v1}"

METHODS=""
CONDITIONS=""
METHOD_SLUG="hts"
METHOD_INDEX="1"
MANIFEST=""
PRIMARY_ONLY=0
FORCE_CONDITION=0

case "$FAMILY" in
  v20_hts)
    MANIFEST="$ART/atari_fair_compare_v20/hts_candidate_reextract_v20.json"
    METHOD_SLUG="hts-v20-locked-hier-x3"
    METHODS="HTS"
    CONDITIONS=""
    CONDITION_OVERRIDE="v20_locked_hier_x3"
    FORCE_CONDITION=1
    METHOD_INDEX="1"
    ;;
  no_vc)
    MANIFEST="$ART/atari_no_vc_ablation/no_vc_raw_metrics.json"
    METHOD_SLUG="hts-no-vc"
    METHODS="HTS"
    CONDITIONS="no_vc_locked_hier_x3"
    METHOD_INDEX="2"
    ;;
  no_vc_with_reference)
    MANIFEST="$ART/atari_no_vc_ablation/no_vc_raw_metrics.json"
    METHOD_SLUG="hts-no-vc-mixed-manifest"
    METHOD_INDEX="2"
    ;;
  cov_sweep)
    MANIFEST="$ART/atari_vc_cov_sweep/cov_sweep_raw_metrics.json"
    METHOD_SLUG="hts-cov-sweep"
    METHOD_INDEX="3"
    ;;
  levels_sweep)
    MANIFEST="$ART/atari_levels_sweep/levels_sweep_raw_metrics.json"
    METHOD_SLUG="hts-levels-sweep"
    METHOD_INDEX="4"
    ;;
  v21_warmup)
    MANIFEST="$ART/atari_v21_aux_warmup/v21_stage_a_raw_metrics.json"
    METHOD_SLUG="hts-v21-warmup"
    METHODS="HTS"
    METHOD_INDEX="5"
    ;;
  v22_new)
    "$PY" "$ART/atari_v22_paper_core_control_prefix/extract_v22_upload_manifest.py"
    MANIFEST="$ART/atari_v22_paper_core_control_prefix/v22_upload_manifest_metrics.json"
    METHOD_SLUG="hts-new-v22"
    METHODS="HTS-new"
    METHOD_INDEX="6"
    ;;
  custom)
    if [[ -z "${MANIFEST:-}" ]]; then
      echo "FAMILY=custom requires MANIFEST=/path/to/manifest.json" >&2
      exit 2
    fi
    METHOD_SLUG="${METHOD_SLUG:-hts-custom}"
    METHOD_INDEX="${METHOD_INDEX:-9}"
    ;;
  *)
    echo "Unknown FAMILY=$FAMILY" >&2
    echo "Use one of: v20_hts, no_vc, no_vc_with_reference, cov_sweep, levels_sweep, v21_warmup, v22_new, custom" >&2
    exit 2
    ;;
esac

ARGS=(
  --manifest "$MANIFEST"
  --method-slug "$METHOD_SLUG"
  --condition "${CONDITION_OVERRIDE:-$METHOD_SLUG}"
  --method-index "$METHOD_INDEX"
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
if [[ -n "$METHODS" ]]; then
  ARGS+=(--methods "$METHODS")
fi
if [[ -n "$CONDITIONS" ]]; then
  ARGS+=(--conditions "$CONDITIONS")
fi
if [[ "$PRIMARY_ONLY" == "1" ]]; then
  ARGS+=(--primary-only)
fi
if [[ "$FORCE_CONDITION" == "1" ]]; then
  ARGS+=(--force-condition)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "$SAVE_SOURCE_FILES" == "1" ]]; then
  ARGS+=(--save-source-files)
fi

echo "PY=$PY"
echo "FAMILY=$FAMILY"
echo "MANIFEST=$MANIFEST"
echo "METHOD_SLUG=$METHOD_SLUG"
echo "METHODS=${METHODS:-ANY}"
echo "CONDITIONS=${CONDITIONS:-ANY}"
echo "PROJECT_MODE=$PROJECT_MODE"
echo "PROJECT_PREFIX=$PROJECT_PREFIX"
echo "GAMES=${GAMES:-ALL}"
echo "SEEDS=${SEEDS:-ALL}"
echo "DRY_RUN=$DRY_RUN"

"$PY" "$SCRIPT" "${ARGS[@]}"
