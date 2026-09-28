#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/launch_common.sh"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full checkpoint or compact adapter directory}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/stage_lora}"
configure_runtime "${OUTPUT_DIR}" Stage-LoRA "${PROJECT_ROOT}"

EXTRA_ARGS=()
if [[ -n "${BASE_CHECKPOINT:-}" ]]; then EXTRA_ARGS+=(--base_checkpoint "${BASE_CHECKPOINT}"); fi
if [[ -n "${VAE_PATH:-}" ]]; then EXTRA_ARGS+=(--vae_path "${VAE_PATH}"); fi
if [[ -n "${T5_PATH:-}" ]]; then EXTRA_ARGS+=(--t5_path "${T5_PATH}"); fi
if [[ -n "${RUNTIME_PATH:-}" ]]; then EXTRA_ARGS+=(--runtime_path "${RUNTIME_PATH}"); fi

# Architecture, branch positions/count, and LoRA settings come from the checkpoint.
python "${PROJECT_ROOT}/scripts/official_edit/infer_stage_lora.py" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --weights_dir "${WEIGHTS_DIR}" \
  --official_meta_root "${EDIT_OFFICIAL_META_ROOT:-${PROJECT_ROOT}/data/metadata}" \
  --official_meta_subdirs "${EDIT_OFFICIAL_META_SUBDIRS:-}" \
  --output_dir "${OUTPUT_DIR}" \
  --GPUS "${GPUS:-1}" \
  --num_samples_per_dataset "${NUM_SAMPLES_PER_DATASET:-4}" \
  --shuffle "${SHUFFLE:-1}" --sample_seed "${SAMPLE_SEED:-42}" \
  --seed "${SEED:-1234}" --skip_existing "${SKIP_EXISTING:-1}" \
  --guidance_mode "${GUIDANCE_MODE:-none}" --guidance_scale "${GUIDANCE_SCALE:-3.0}" \
  --negative_prompt "${NEGATIVE_PROMPT:-}" --cfg_interval "${CFG_INTERVAL:-0.0}" \
  --max_infer_steps "${MAX_INFER_STEPS:-50}" --temperature "${TEMPERATURE:-1.0}" \
  --snr_shift "${SNR_SHIFT:-1.0}" --use_slow_attn "${USE_SLOW_ATTN:-0}" \
  "${EXTRA_ARGS[@]}" "$@"
