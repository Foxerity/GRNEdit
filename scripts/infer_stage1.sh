#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/launch_common.sh"

require_path "CHECKPOINT_PATH" "${CHECKPOINT_PATH:-}"
require_path "WEIGHTS_DIR" "${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
require_path "EDIT_OFFICIAL_META_ROOT" "${EDIT_OFFICIAL_META_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/stage1}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/scripts/official_edit/infer_stage15_layerwise_source_conditioned.py" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --weights_dir "${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}" \
  --output_dir "${OUTPUT_DIR}" \
  --official_meta_root "${EDIT_OFFICIAL_META_ROOT}" \
  --official_meta_subdirs "${EDIT_OFFICIAL_META_SUBDIRS:-}" \
  --num_samples_per_dataset "${NUM_SAMPLES_PER_DATASET:-4}" \
  --shuffle "${SHUFFLE:-1}" \
  --sample_seed "${SAMPLE_SEED:-42}" \
  --seed "${SEED:-1234}" \
  --GPUS "${GPUS:-1}" \
  --workers "${WORKERS:-2}" \
  --model GRN2bOfficialEditStage15 \
  --block_chunks 7 \
  --pn 0.41M \
  --video_fps "${VIDEO_FPS:-20}" \
  --video_frames "${VIDEO_FRAMES:-61}" \
  --duration_resolution "${DURATION_RESOLUTION:-0.25}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --max_infer_steps "${MAX_INFER_STEPS:-50}" \
  --complexity_aware_Tmin "${COMPLEXITY_AWARE_TMIN:-10}" \
  --complexity_aware_Tmax "${COMPLEXITY_AWARE_TMAX:-0}" \
  --snr_shift "${SNR_SHIFT:-1.0}" \
  --use_slow_attn "${USE_SLOW_ATTN:-0}" \
  --use_reprompt_text "${USE_REPROMPT:-0}" \
  --t5_max_tokens "${T5_MAX_TOKENS:-512}" \
  --use_ema "${USE_EMA:-1}" \
  --stage15_source_residual_modulation text_pt \
  --stage15_source_injection_mask '[1,0,0,0,0,0,1]' \
  "$@"
