#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASETS="20newsgroups,yahoo_subset"
ALGORITHMS="d-federaser"
SEED="42"
KS="1,2,3,4,5,6,7,8,9"
PHYSICAL_GPUS="0,1,2,3"
OUT_DIR="artifacts/tdb_k_sweep_20260525"
MAX_EVAL_SAMPLES="100"
BATCH_SIZE="1"
GRAD_ACCUM_STEPS="1"
LR="1e-4"
CALIBRATION_STEPS="1"
CALIBRATION_INTERVAL="10"
TDB_SKETCH_DIM="16"
TDB_MAX_INTERVALS="2"
TDB_TIME_LIMIT="20"

usage() {
  cat <<EOF
Usage: bash scripts/run_tdb_k_sweep_parallel.sh [options]

Options:
  --datasets CSV              Default: ${DATASETS}
  --algorithms CSV            Default: ${ALGORITHMS}
  --seed INT                  Default: ${SEED}
  --ks CSV                    Default: ${KS}
  --physical_gpus CSV         Default: ${PHYSICAL_GPUS}
  --out_dir PATH              Default: ${OUT_DIR}
  --max_eval_samples INT      Default: ${MAX_EVAL_SAMPLES}
  --tdb_sketch_dim INT        Default: ${TDB_SKETCH_DIM}
  --tdb_max_intervals INT     Default: ${TDB_MAX_INTERVALS}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --algorithms) ALGORITHMS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --ks) KS="$2"; shift 2 ;;
    --physical_gpus) PHYSICAL_GPUS="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --max_eval_samples) MAX_EVAL_SAMPLES="$2"; shift 2 ;;
    --tdb_sketch_dim) TDB_SKETCH_DIM="$2"; shift 2 ;;
    --tdb_max_intervals) TDB_MAX_INTERVALS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

snapshot_for_dataset() {
  local dataset="$1"
  local seed="$2"
  case "${dataset}:${seed}" in
    20newsgroups:42) echo "checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624" ;;
    20newsgroups:43) echo "checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed43_20260209_075703" ;;
    20newsgroups:44) echo "checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed44_20260209_082328" ;;
    yahoo_subset:42) echo "checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20251223_081958" ;;
    yahoo_subset:43) echo "checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed43_20260209_081351" ;;
    yahoo_subset:44) echo "checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed44_20260209_083957" ;;
    *) echo "Unsupported dataset/seed: ${dataset}/${seed}" >&2; return 1 ;;
  esac
}

IFS=',' read -r -a DATASET_ARR <<< "${DATASETS}"
IFS=',' read -r -a ALGO_ARR <<< "${ALGORITHMS}"
IFS=',' read -r -a K_ARR <<< "${KS}"
IFS=',' read -r -a GPU_ARR <<< "${PHYSICAL_GPUS}"

mkdir -p "${OUT_DIR}/logs"
MAX_PARALLEL="${#GPU_ARR[@]}"
if [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "No GPUs provided" >&2
  exit 1
fi

pids=()
gpu_slots=()

wait_one() {
  local i
  while true; do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}"
        unset 'pids[i]'
        unset 'gpu_slots[i]'
        pids=("${pids[@]}")
        gpu_slots=("${gpu_slots[@]}")
        return
      fi
    done
    sleep 2
  done
}

next_gpu() {
  local used gpu ok
  for gpu in "${GPU_ARR[@]}"; do
    ok=1
    for used in "${gpu_slots[@]}"; do
      if [[ "${used}" == "${gpu}" ]]; then
        ok=0
        break
      fi
    done
    if [[ "${ok}" -eq 1 ]]; then
      echo "${gpu}"
      return 0
    fi
  done
  return 1
}

for dataset in "${DATASET_ARR[@]}"; do
  snapshot="$(snapshot_for_dataset "${dataset}" "${SEED}")"
  for algo in "${ALGO_ARR[@]}"; do
    for k in "${K_ARR[@]}"; do
      while [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; do
        wait_one
      done
      gpu="$(next_gpu)"
      log="${OUT_DIR}/logs/${dataset}_${algo}_seed${SEED}_k${k}_gpu${gpu}.log"
      echo "[launch] dataset=${dataset} algo=${algo} seed=${SEED} k=${k} gpu=${gpu} log=${log}"
      (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export LLMDFL_ALLOWED_PHYSICAL_GPUS="${gpu}"
        export LLMDFL_LOCAL_FILES_ONLY=1
        export TOKENIZERS_PARALLELISM=false
        export PYTHONUNBUFFERED=1
        python -u scripts/run_dfu.py \
          --dfl_snapshot "${snapshot}" \
          --dfu_algorithm "${algo}" \
          --output_dir "${OUT_DIR}" \
          --target_agent 0 \
          --seed "${SEED}" \
          --batch_size "${BATCH_SIZE}" \
          --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
          --lr "${LR}" \
          --calibration_steps "${CALIBRATION_STEPS}" \
          --calibration_interval "${CALIBRATION_INTERVAL}" \
          --eval_every 0 \
          --max_eval_samples "${MAX_EVAL_SAMPLES}" \
          --gpu 0 \
          --no_save_lora_states \
          --selection_strategy tdb \
          --selection_count "${k}" \
          --tdb_sketch_dim "${TDB_SKETCH_DIM}" \
          --tdb_max_intervals "${TDB_MAX_INTERVALS}" \
          --tdb_time_limit "${TDB_TIME_LIMIT}"
      ) >"${log}" 2>&1 &
      pids+=("$!")
      gpu_slots+=("${gpu}")
    done
  done
done

while [[ "${#pids[@]}" -gt 0 ]]; do
  wait_one
done

echo "TDB k sweep completed: ${OUT_DIR}"
