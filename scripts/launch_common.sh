#!/usr/bin/env bash

require_path() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" || ! -e "${value}" ]]; then
    echo "${name} does not exist: ${value}" >&2
    exit 2
  fi
}

configure_metadata() {
  local root="$1"
  local subdirs="$2"
  require_path "EDIT_OFFICIAL_META_ROOT" "${root}"

  local folders="["
  local repeats="["
  local identifiers="["
  local sep=""
  local specs=()
  if [[ -n "${subdirs}" ]]; then
    IFS=',' read -r -a specs <<< "${subdirs}"
  else
    specs=(".")
  fi

  local spec subdir repeat meta_dir
  for spec in "${specs[@]}"; do
    spec="${spec#"${spec%%[![:space:]]*}"}"
    spec="${spec%"${spec##*[![:space:]]}"}"
    repeat="1"
    subdir="${spec}"
    if [[ "${spec}" == *@* ]]; then
      subdir="${spec%@*}"
      repeat="${spec##*@}"
    fi
    if [[ "${subdir}" == "." ]]; then
      meta_dir="${root}"
    else
      meta_dir="${root%/}/${subdir}"
    fi
    if ! compgen -G "${meta_dir}/*.jsonl" >/dev/null &&
       ! compgen -G "${meta_dir}/*/*.jsonl" >/dev/null; then
      echo "No JSONL metadata found under ${meta_dir}" >&2
      exit 2
    fi
    folders+="${sep}\"${meta_dir}\""
    repeats+="${sep}${repeat}"
    identifiers+="${sep}\"\""
    sep=","
  done
  export OFFICIAL_T2IV_META_FOLDERS="${folders}]"
  export OFFICIAL_T2IV_META_FOLDER_REPEATS="${repeats}]"
  export OFFICIAL_T2IV_META_FOLDER_IDENTIFIERS="${identifiers}]"
}

configure_runtime() {
  local output_dir="$1"
  local run_name="$2"
  local project_root="$3"
  local runtime_dir="${RUNTIME_CACHE_DIR:-${output_dir}/runtime_cache}"
  mkdir -p \
    "${output_dir}" \
    "${runtime_dir}/token_cache" \
    "${runtime_dir}/tmp" \
    "${runtime_dir}/torchinductor" \
    "${runtime_dir}/triton" \
    "${runtime_dir}/xdg"
  export TOKEN_CACHE_DIR="${runtime_dir}/token_cache"
  export TMPDIR="${runtime_dir}/tmp"
  export TORCHINDUCTOR_CACHE_DIR="${runtime_dir}/torchinductor"
  export TRITON_CACHE_DIR="${runtime_dir}/triton"
  export XDG_CACHE_HOME="${runtime_dir}/xdg"
  export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TOKENIZERS_PARALLELISM=false
  export WANDB_CONSOLE=off
  if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
    export GRN_DISABLE_WANDB=0
    export WANDB_MODE="${WANDB_MODE:-online}"
    unset WANDB_DISABLED
  else
    export GRN_DISABLE_WANDB=1
    export WANDB_DISABLED=true
    export WANDB_MODE=disabled
  fi
}

configure_torchrun() {
  local gpus="$1"
  local nnodes="${NNODES:-1}"
  TORCHRUN_ARGS=(--nproc_per_node="${gpus}")
  if [[ "${nnodes}" == "1" ]]; then
    TORCHRUN_ARGS+=(--standalone)
    return
  fi
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "MASTER_ADDR is required for multi-node training." >&2
    exit 2
  fi
  TORCHRUN_ARGS+=(
    --nnodes="${nnodes}"
    --node_rank="${NODE_RANK:-0}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT:-29500}"
  )
}
