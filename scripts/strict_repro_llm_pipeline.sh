#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PHYSICAL_GPU=""
LOGICAL_GPU="0"
SEEDS="42,43,44"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --physical_gpu)
      PHYSICAL_GPU="$2"; shift 2 ;;
    --gpu)
      LOGICAL_GPU="$2"; shift 2 ;;
    --seeds)
      SEEDS="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PHYSICAL_GPU}" ]]; then
  echo "Missing --physical_gpu" >&2
  exit 2
fi

source /home/xzq/miniconda3/etc/profile.d/conda.sh
conda activate uld

export LLMDFL_LOCAL_FILES_ONLY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LLMDFL_ALLOWED_PHYSICAL_GPUS="${LLMDFL_ALLOWED_PHYSICAL_GPUS:-${PHYSICAL_GPU}}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
RESULTS_ROOT="${LLMDFL_EXPERIMENT_DIR:-实验结果/运行产物}"

mkdir -p logs reports "${RESULTS_ROOT}"

echo "[strict-repro] gpu=${PHYSICAL_GPU} logical=${LOGICAL_GPU} seeds=${SEEDS}"

has_snapshot() {
  local dataset="$1"
  local seed="$2"
  local base="checkpoints/${dataset}/K10/G10_L5/alpha0.5"
  local snap
  snap="$(ls -d "${base}/seed${seed}_"* 2>/dev/null | sort | tail -n 1 || true)"
  [[ -n "${snap}" && -d "${snap}/round_10" && -f "${snap}/config.json" ]]
}

train_if_missing() {
  local dataset="$1"
  local seed="$2"

  if has_snapshot "${dataset}" "${seed}"; then
    echo "[strict-repro] Reuse snapshot dataset=${dataset} seed=${seed}"
    return
  fi

  echo "[strict-repro] Train snapshot dataset=${dataset} seed=${seed}"
  python -u scripts/train_dfl.py \
    --dataset "${dataset}" \
    --output_dir checkpoints \
    --num_agents 10 \
    --alpha 0.5 \
    --seed "${seed}" \
    --global_rounds 10 \
    --local_steps 5 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 1e-3 \
    --eval_every 0 \
    --max_eval_samples 2000 \
    --gpu "${LOGICAL_GPU}" \
    2>&1 | tee "logs/strict_repro_train_${dataset}_seed${seed}_gpu${PHYSICAL_GPU}.log"
}

IFS=',' read -r -a seed_arr <<< "${SEEDS}"
for s in "${seed_arr[@]}"; do
  s="${s//[[:space:]]/}"
  [[ -z "${s}" ]] && continue
  train_if_missing 20newsgroups "${s}"
  train_if_missing yahoo_subset "${s}"
done

# Backfill strict seed-aligned DFU runs (DFU seed == DFL snapshot seed).
python -u scripts/backfill_llm_seed_aligned.py \
  --seeds "${SEEDS}" \
  --physical_gpu "${PHYSICAL_GPU}" \
  --gpu "${LOGICAL_GPU}" \
  --max_eval_samples 100 \
  --out_root "${RESULTS_ROOT}/dfu_seed_aligned_llm_strict_${SEEDS//,/}" \
  --summary_csv "${RESULTS_ROOT}/artifacts/seed_alignment_llm.csv" \
  2>&1 | tee "logs/strict_repro_backfill_seed_aligned_gpu${PHYSICAL_GPU}.log"

python -u scripts/reproduce_paper_llm.py --seeds "${SEEDS}" --out_dir reports --require_seed_aligned_snapshot true \
  2>&1 | tee "logs/strict_repro_reproduce_llm_gpu${PHYSICAL_GPU}.log"

python -u scripts/recompute_mia_privacy.py --seeds "${SEEDS}" --gpu "${LOGICAL_GPU}" \
  2>&1 | tee "logs/strict_repro_recompute_mia_gpu${PHYSICAL_GPU}.log"

echo "[strict-repro] DONE"
