#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET=""
PHYSICAL_GPU=""
LOGICAL_GPU="0"
SEED="44"
TRIGGER="cf_trigger_xzq"
TARGET_LABEL="0"
TARGET_AGENT="0"
EVAL_AGENT_ID="1"
EVAL_SCOPE="${EVAL_SCOPE:-all}"
DFU_STATE_MODE="${DFU_STATE_MODE:-participant}"
DFL_EVAL_SCOPE="${DFL_EVAL_SCOPE:-same_as_dfu}"
BACKDOOR_AUDIT_MODELS="${BACKDOOR_AUDIT_MODELS:-dfl,dfu,retrain}"
BACKDOOR_RATE="${BACKDOOR_RATE:-0.2}"
BACKDOOR_POSITION="${BACKDOOR_POSITION:-prefix}"
BACKDOOR_SAMPLE_SOURCE="${BACKDOOR_SAMPLE_SOURCE:-public_test}"

MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-400}"
AUDIT_MAX_SAMPLES="${AUDIT_MAX_SAMPLES:-400}"
DFU_BATCH_SIZE="${DFU_BATCH_SIZE:-4}"
DFU_GRAD_ACCUM_STEPS="${DFU_GRAD_ACCUM_STEPS:-2}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-16}"
MIA_BATCH_SIZE="${MIA_BATCH_SIZE:-8}"

RESULTS_ROOT="${LLMDFL_EXPERIMENT_DIR:-实验结果/运行产物}"
ARTIFACT_ROOT="${RESULTS_ROOT}/artifacts"
DFL_OUTPUT_DIR="${RESULTS_ROOT}/checkpoints_backdoor_audit"
RETRAIN_OUTPUT_DIR="${RESULTS_ROOT}/retrain_checkpoints_backdoor_audit"
DFU_OUTPUT_DIR="${RESULTS_ROOT}/dfu_checkpoints_backdoor_audit"
LEGACY_DFL_OUTPUT_DIR="checkpoints_backdoor_audit"
LEGACY_RETRAIN_OUTPUT_DIR="retrain_checkpoints_backdoor_audit"
LEGACY_DFU_OUTPUT_DIR="dfu_checkpoints_backdoor_audit"

FORCE_DFL_TRAIN="${FORCE_DFL_TRAIN:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_AUDIT="${FORCE_AUDIT:-0}"
FORCE_DFU="${FORCE_DFU:-0}"
ALGORITHMS="d-federaser,d-fedosd,d-fedrecovery,d-oblivionis"
STRATEGIES="full_all,full_ours,ours_all,ours_ours"
TAG_SUFFIX=""
SKIP_MIA="${SKIP_MIA:-0}"
BACKDOOR_CLEANUP_DFU_STATES="${BACKDOOR_CLEANUP_DFU_STATES:-1}"

META_JSON="reports/tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_best.csv"
DSU_META_JSON="${DSU_META_JSON:-reports/tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_best.csv}"
OVERRIDE_K=""
OVERRIDE_R=""

DFL_LR="${DFL_LR:-1e-3}"
DFU_LR="${DFU_LR:-1e-3}"
FEDERASER_CALIBRATION_STEPS="${FEDERASER_CALIBRATION_STEPS:-5}"
FEDERASER_CALIBRATION_INTERVAL="${FEDERASER_CALIBRATION_INTERVAL:-2}"
FEDOSD_UNLEARN_ROUNDS="${FEDOSD_UNLEARN_ROUNDS:-3}"
FEDOSD_UNLEARN_LR="${FEDOSD_UNLEARN_LR:-1e-3}"
FEDOSD_RECOVERY_ROUNDS="${FEDOSD_RECOVERY_ROUNDS:-2}"
FEDOSD_RECOVERY_LOCAL_STEPS="${FEDOSD_RECOVERY_LOCAL_STEPS:-5}"
FEDOSD_RECOVERY_LR="${FEDOSD_RECOVERY_LR:-1e-3}"
FEDOSD_RETAIN_GRAD_SAMPLES="${FEDOSD_RETAIN_GRAD_SAMPLES:-50}"
FEDOSD_ORTHOGONAL_UPDATE_NORM="${FEDOSD_ORTHOGONAL_UPDATE_NORM:-}"
FEDRECOVERY_CORRECTION_WEIGHT="${FEDRECOVERY_CORRECTION_WEIGHT:-5.0}"
FEDRECOVERY_NOISE_STD="${FEDRECOVERY_NOISE_STD:-0.0}"
FEDRECOVERY_RECOVERY_ROUNDS="${FEDRECOVERY_RECOVERY_ROUNDS:-3}"
FEDRECOVERY_RECOVERY_LOCAL_STEPS="${FEDRECOVERY_RECOVERY_LOCAL_STEPS:-5}"
FEDRECOVERY_RECOVERY_LR="${FEDRECOVERY_RECOVERY_LR:-1e-3}"
OBLIVIONIS_UNLEARN_ROUNDS="${OBLIVIONIS_UNLEARN_ROUNDS:-1}"
OBLIVIONIS_UNLEARN_LR="${OBLIVIONIS_UNLEARN_LR:-5e-4}"
OBLIVIONIS_PROPAGATION_ROUNDS="${OBLIVIONIS_PROPAGATION_ROUNDS:-3}"
OBLIVIONIS_PROPAGATION_LOCAL_STEPS="${OBLIVIONIS_PROPAGATION_LOCAL_STEPS:-}"
OBLIVIONIS_PROPAGATION_LR="${OBLIVIONIS_PROPAGATION_LR:-1e-3}"
TDB_SKETCH_DIM="${TDB_SKETCH_DIM:-16}"
TDB_MAX_INTERVALS="${TDB_MAX_INTERVALS:-2}"
TDB_TIME_LIMIT="${TDB_TIME_LIMIT:-20}"
TDB_ALPHA_U="${TDB_ALPHA_U:-1.0}"
TDB_ALPHA_P="${TDB_ALPHA_P:-1.0}"
TDB_ALPHA_Q="${TDB_ALPHA_Q:-0.1}"
TDB_TAU_Q="${TDB_TAU_Q:-0.0}"
TDB_AGGREGATION_SCOPE="${TDB_AGGREGATION_SCOPE:-local}"

ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --physical_gpu)
      PHYSICAL_GPU="$2"; shift 2 ;;
    --gpu)
      LOGICAL_GPU="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --trigger)
      TRIGGER="$2"; shift 2 ;;
    --backdoor_rate)
      BACKDOOR_RATE="$2"; shift 2 ;;
    --backdoor_position)
      BACKDOOR_POSITION="$2"; shift 2 ;;
    --sample_source|--backdoor_sample_source)
      BACKDOOR_SAMPLE_SOURCE="$2"; shift 2 ;;
    --target_label)
      TARGET_LABEL="$2"; shift 2 ;;
    --target_agent)
      TARGET_AGENT="$2"; shift 2 ;;
    --eval_agent_id)
      EVAL_AGENT_ID="$2"; shift 2 ;;
    --eval_scope)
      EVAL_SCOPE="$2"; shift 2 ;;
    --dfu_state_mode)
      DFU_STATE_MODE="$2"; shift 2 ;;
    --dfl_eval_scope)
      DFL_EVAL_SCOPE="$2"; shift 2 ;;
    --audit_models)
      BACKDOOR_AUDIT_MODELS="$2"; shift 2 ;;
    --max_eval_samples)
      MAX_EVAL_SAMPLES="$2"; shift 2 ;;
    --audit_max_samples)
      AUDIT_MAX_SAMPLES="$2"; shift 2 ;;
    --dfl_output_dir)
      DFL_OUTPUT_DIR="$2"; shift 2 ;;
    --retrain_output_dir)
      RETRAIN_OUTPUT_DIR="$2"; shift 2 ;;
    --dfu_output_dir)
      DFU_OUTPUT_DIR="$2"; shift 2 ;;
    --force_dfl_train)
      FORCE_DFL_TRAIN="$2"; shift 2 ;;
    --force_retrain)
      FORCE_RETRAIN="$2"; shift 2 ;;
    --force_audit)
      FORCE_AUDIT="$2"; shift 2 ;;
    --algorithms)
      ALGORITHMS="$2"; shift 2 ;;
    --strategies)
      STRATEGIES="$2"; shift 2 ;;
    --tag_suffix)
      TAG_SUFFIX="$2"; shift 2 ;;
    --skip_mia)
      SKIP_MIA="$2"; shift 2 ;;
    --meta_json)
      META_JSON="$2"; shift 2 ;;
    --dsu_meta_json)
      DSU_META_JSON="$2"; shift 2 ;;
    --override_k)
      OVERRIDE_K="$2"; shift 2 ;;
    --override_r|--override_ratio)
      OVERRIDE_R="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PHYSICAL_GPU}" ]]; then
  echo "Missing required arg: --physical_gpu" >&2
  exit 2
fi

if [[ -z "${DATASET}" || "${DATASET}" == "all" ]]; then
  echo "[grid] No --dataset provided; running both 20newsgroups and yahoo_subset sequentially."
  for ds in 20newsgroups yahoo_subset; do
    bash "$0" "${ORIGINAL_ARGS[@]}" --dataset "${ds}"
  done
  exit 0
fi

if [[ "${DATASET}" != "20newsgroups" && "${DATASET}" != "yahoo_subset" ]]; then
  echo "Unsupported --dataset=${DATASET}. Use 20newsgroups, yahoo_subset, or all." >&2
  exit 2
fi

source /home/xzq/miniconda3/etc/profile.d/conda.sh
conda activate uld

export LLMDFL_LOCAL_FILES_ONLY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LLMDFL_ALLOWED_PHYSICAL_GPUS="${LLMDFL_ALLOWED_PHYSICAL_GPUS:-${PHYSICAL_GPU}}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export LLMDFL_ARTIFACT_ROOT="${ARTIFACT_ROOT}"

mkdir -p logs "${ARTIFACT_ROOT}/unlearning_audit/backdoor" "${ARTIFACT_ROOT}/unlearning_audit/mia" "${ARTIFACT_ROOT}/unlearning_audit/sens_cache"

echo "[grid] dataset=${DATASET} seed=${SEED} physical_gpu=${PHYSICAL_GPU} logical_gpu=${LOGICAL_GPU}"
echo "[grid] backdoor trigger=${TRIGGER} rate=${BACKDOOR_RATE} position=${BACKDOOR_POSITION} target_label=${TARGET_LABEL} target_agent=${TARGET_AGENT}"
echo "[grid] audit eval_scope=${EVAL_SCOPE} dfu_state_mode=${DFU_STATE_MODE} dfl_eval_scope=${DFL_EVAL_SCOPE} sample_source=${BACKDOOR_SAMPLE_SOURCE}"
echo "[grid] audit models=${BACKDOOR_AUDIT_MODELS}"
echo "[grid] hparams dfu_lr=${DFU_LR} federaser_steps=${FEDERASER_CALIBRATION_STEPS} fedosd_u=${FEDOSD_UNLEARN_ROUNDS},r=${FEDOSD_RECOVERY_ROUNDS} fedrecovery_r=${FEDRECOVERY_RECOVERY_ROUNDS} oblivionis_u=${OBLIVIONIS_UNLEARN_ROUNDS},p=${OBLIVIONIS_PROPAGATION_ROUNDS},pls=${OBLIVIONIS_PROPAGATION_LOCAL_STEPS:-default}"
echo "[grid] tdb sketch_dim=${TDB_SKETCH_DIM} max_intervals=${TDB_MAX_INTERVALS} alphas=${TDB_ALPHA_U}/${TDB_ALPHA_P}/${TDB_ALPHA_Q} tau_q=${TDB_TAU_Q} agg=${TDB_AGGREGATION_SCOPE}"

rate_tag() {
  local x="$1"
  R_ENV="${x}" python - <<'PY'
import os
x = float(os.environ["R_ENV"])
s = f"{x:.6f}".rstrip("0").rstrip(".")
print((s if s else "0").replace(".", "p"))
PY
}

if [[ -z "${TAG_SUFFIX}" ]]; then
  bd_rate_tag="$(rate_tag "${BACKDOOR_RATE}")"
  if [[ "${bd_rate_tag}" != "1" || "${BACKDOOR_POSITION}" != "prefix" ]]; then
    TAG_SUFFIX="_bd${bd_rate_tag}_${BACKDOOR_POSITION}"
  fi
fi
echo "[grid] tag_suffix=${TAG_SUFFIX:-<none>}"

read_best_cfg() {
  local dataset="$1"
  local algo="$2"
  local meta_path="$3"
  local strat="$4"
  local dsu_meta_path="$5"
  DATASET_ENV="${dataset}" ALGO_ENV="${algo}" META_ENV="${meta_path}" STRAT_ENV="${strat}" DSU_META_ENV="${dsu_meta_path}" python - <<'PY'
import json
import os
import csv
from pathlib import Path

dataset = os.environ["DATASET_ENV"]
algo = os.environ["ALGO_ENV"]
meta_path = Path(os.environ["META_ENV"])
dsu_meta_path = Path(os.environ["DSU_META_ENV"]) if os.environ.get("DSU_META_ENV") else None
strategy = os.environ["STRAT_ENV"]

default_k = 5
default_r = 0.5

as_k = None
ls_r = None
dsu_k = None
dsu_r = None

def _int_or_none(x):
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None

def _float_or_none(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def read_csv_meta(path: Path):
    global as_k, ls_r, dsu_k, dsu_r
    if not path or not path.exists() or path.suffix.lower() != ".csv":
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row_dataset = row.get("dataset")
            row_algo = row.get("method") or row.get("algorithm")
            if row_dataset != dataset or row_algo != algo:
                continue
            setting = str(row.get("setting") or "").upper()
            if setting:
                if setting == "AS":
                    as_k = _int_or_none(row.get("k")) or as_k
                elif setting == "LS":
                    ls_r = _float_or_none(row.get("r")) or ls_r
                elif setting == "DSU":
                    dsu_k = _int_or_none(row.get("k")) or dsu_k
                    dsu_r = _float_or_none(row.get("r")) or dsu_r
                continue
            as_k = _int_or_none(row.get("as_best_k")) or as_k
            ls_r = _float_or_none(row.get("ls_best_r")) or ls_r
            dsu_k = _int_or_none(row.get("dsu_best_k")) or dsu_k
            dsu_r = _float_or_none(row.get("dsu_best_r")) or dsu_r

def read_json_meta(path: Path):
    global as_k, ls_r
    if not path or not path.exists() or path.suffix.lower() == ".csv":
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    key = f"{dataset}::{algo}"
    raw_k = (meta.get("best_k") or {}).get(key)
    if isinstance(raw_k, dict):
        raw_k = raw_k.get("k")
    raw_r = (meta.get("best_r") or meta.get("best_ratio") or {}).get(key)
    if isinstance(raw_r, dict):
        raw_r = raw_r.get("ratio")
    as_k = _int_or_none(raw_k) or as_k
    ls_r = _float_or_none(raw_r) or ls_r

read_csv_meta(meta_path)
read_json_meta(meta_path)
if dsu_meta_path:
    read_csv_meta(dsu_meta_path)

k = as_k or default_k
r = ls_r or default_r
if strategy == "ours_ours":
    k = dsu_k or k
    r = dsu_r or r
print(f"{k},{r}")
PY
}

format_ratio() {
  local x="$1"
  R_ENV="${x}" python - <<'PY'
import os
x = float(os.environ["R_ENV"])
if x.is_integer():
    print(f"{x:.1f}")
    raise SystemExit(0)
s = f"{x:.6f}".rstrip("0").rstrip(".")
print(s if s else "0")
PY
}

latest_seed_snapshot() {
  local base_dir="$1"
  local dataset="$2"
  local seed="$3"
  ls -d "${base_dir}/${dataset}/K10/G10_L5/alpha0.5/seed${seed}_"* 2>/dev/null | sort | tail -n 1 || true
}

is_dfl_snapshot_complete() {
  local snap="$1"
  SNAP_ENV="${snap}" EXPECT_TRIGGER="${TRIGGER}" EXPECT_RATE="${BACKDOOR_RATE}" EXPECT_LABEL="${TARGET_LABEL}" EXPECT_AGENT="${TARGET_AGENT}" EXPECT_POSITION="${BACKDOOR_POSITION}" python - <<'PY'
import json
import os
from pathlib import Path

snap = Path(os.environ["SNAP_ENV"])
cfg = snap / "config.json"
hist = snap / "history.json"
if not cfg.exists() or not hist.exists():
    raise SystemExit(1)

try:
    data = json.loads(hist.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

rounds = data.get("global_rounds")
if not isinstance(rounds, list) or not rounds:
    raise SystemExit(1)

last_round = int(rounds[-1])
if last_round < 10:
    raise SystemExit(1)

config = json.loads(cfg.read_text(encoding="utf-8"))
expect_trigger = os.environ.get("EXPECT_TRIGGER")
expect_rate = os.environ.get("EXPECT_RATE")
expect_label = os.environ.get("EXPECT_LABEL")
expect_agent = os.environ.get("EXPECT_AGENT")
expect_position = os.environ.get("EXPECT_POSITION")

def _same_float(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

if expect_trigger and str(config.get("backdoor_trigger")) != str(expect_trigger):
    raise SystemExit(1)
if expect_rate and not _same_float(config.get("backdoor_rate"), expect_rate):
    raise SystemExit(1)
if expect_label and str(config.get("backdoor_target_label")) != str(expect_label):
    raise SystemExit(1)
if expect_agent and str(config.get("backdoor_target_agent")) != str(expect_agent):
    raise SystemExit(1)
if expect_position and str(config.get("backdoor_position") or "prefix") != str(expect_position):
    raise SystemExit(1)

print("ok")
PY
}

latest_valid_seed_snapshot() {
  local base_dir="$1"
  local dataset="$2"
  local seed="$3"
  local cand
  while IFS= read -r cand; do
    if [[ -n "${cand}" ]] && is_dfl_snapshot_complete "${cand}" >/dev/null 2>&1; then
      echo "${cand}"
      return 0
    fi
  done < <(ls -d "${base_dir}/${dataset}/K10/G10_L5/alpha0.5/seed${seed}_"* 2>/dev/null | sort -r || true)
  return 1
}

run_if_needed_dfl() {
  local dfl_snapshot=""
  if [[ "${FORCE_DFL_TRAIN}" == "1" ]]; then
    echo "[grid] Training poisoned DFL from scratch..." >&2
    python -u scripts/train_dfl.py \
      --dataset "${DATASET}" \
      --output_dir "${DFL_OUTPUT_DIR}" \
      --num_agents 10 \
      --alpha 0.5 \
      --seed "${SEED}" \
      --global_rounds 10 \
      --local_steps 5 \
      --batch_size "${DFU_BATCH_SIZE}" \
      --grad_accum_steps "${DFU_GRAD_ACCUM_STEPS}" \
      --lr "${DFL_LR}" \
      --eval_every 0 \
      --max_eval_samples "${MAX_EVAL_SAMPLES}" \
      --backdoor_trigger "${TRIGGER}" \
      --backdoor_rate "${BACKDOOR_RATE}" \
      --backdoor_target_label "${TARGET_LABEL}" \
      --backdoor_target_agent "${TARGET_AGENT}" \
      --backdoor_position "${BACKDOOR_POSITION}" \
      --gpu "${LOGICAL_GPU}" \
      2>&1 | tee "logs/backdoor_grid_dfl_${DATASET}_seed${SEED}_gpu${PHYSICAL_GPU}.log" >&2
  fi

  dfl_snapshot="$(latest_valid_seed_snapshot "${DFL_OUTPUT_DIR}" "${DATASET}" "${SEED}" || true)"
  if [[ -z "${dfl_snapshot}" ]]; then
    dfl_snapshot="$(latest_valid_seed_snapshot "${LEGACY_DFL_OUTPUT_DIR}" "${DATASET}" "${SEED}" || true)"
  fi
  if [[ -z "${dfl_snapshot}" ]]; then
    echo "[grid][ERROR] matching DFL snapshot missing/incomplete for ${DATASET} seed=${SEED} trigger=${TRIGGER} rate=${BACKDOOR_RATE}" >&2
    exit 1
  fi
  echo "${dfl_snapshot}"
}

run_if_needed_retrain() {
  local dfl_snapshot="$1"
  local snap_name
  snap_name="$(basename "${dfl_snapshot}")"

  local retrain_root="${RETRAIN_OUTPUT_DIR}/${DATASET}/K10/G10_L5/alpha0.5/strategy_retrain/${snap_name}"
  local legacy_retrain_root="${LEGACY_RETRAIN_OUTPUT_DIR}/${DATASET}/K10/G10_L5/alpha0.5/strategy_retrain/${snap_name}"
  local retrain_dir=""
  retrain_dir="$(ls -d "${retrain_root}/retrain_"* 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${retrain_dir}" || ! -d "${retrain_dir}/round_10" ]]; then
    retrain_dir="$(ls -d "${legacy_retrain_root}/retrain_"* 2>/dev/null | sort | tail -n 1 || true)"
  fi

  if [[ "${FORCE_RETRAIN}" == "1" || -z "${retrain_dir}" || ! -d "${retrain_dir}/round_10" ]]; then
    echo "[grid] Running retrain baseline..." >&2
    python -u scripts/run_retrain.py \
      --dfl_checkpoint "${dfl_snapshot}" \
      --output_dir "${RETRAIN_OUTPUT_DIR}" \
      --target_agent "${TARGET_AGENT}" \
      --eval_every 0 \
      --max_eval_samples "${MAX_EVAL_SAMPLES}" \
      --batch_size 2 \
      --grad_accum_steps 4 \
      --gpu "${LOGICAL_GPU}" \
      2>&1 | tee "logs/backdoor_grid_retrain_${DATASET}_seed${SEED}_gpu${PHYSICAL_GPU}.log" >&2

    retrain_dir="$(ls -d "${retrain_root}/retrain_"* 2>/dev/null | sort | tail -n 1 || true)"
  fi

  if [[ -z "${retrain_dir}" || ! -d "${retrain_dir}/round_10" ]]; then
    echo "[grid][ERROR] retrain checkpoint missing/incomplete: ${retrain_root}" >&2
    exit 1
  fi
  echo "${retrain_dir}"
}

ensure_sens_cache() {
  local dfl_snapshot="$1"
  local out_path="$2"
  if [[ -f "${out_path}" ]]; then
    return
  fi
  SNAP_PATH="${dfl_snapshot}" SENS_PATH_ENV="${out_path}" TARGET_AGENT_ENV="${TARGET_AGENT}" python - <<'PY'
import json
import os
from pathlib import Path
from src.dfu.snapshot_loader import SnapshotLoader
from src.dfu.lora_param_selection import compute_module_sensitivities

snap = Path(os.environ["SNAP_PATH"])
target_agent = int(os.environ["TARGET_AGENT_ENV"])
loader = SnapshotLoader(str(snap))
sens = compute_module_sensitivities(loader, target_agent=target_agent, verbose=True)
out = Path(os.environ["SENS_PATH_ENV"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(sens, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved", out, "modules", len(sens))
PY
}

has_valid_run() {
  local run_base="$1"
  local expected_seed="$2"
  local expected_max_eval="$3"
  local expected_algo="$4"
  local expected_strat="$5"
  local expected_k="$6"
  local expected_ratio="$7"
  RUN_BASE="${run_base}" EXPECT_SEED="${expected_seed}" EXPECT_MAX_EVAL="${expected_max_eval}" EXPECT_ALGO="${expected_algo}" EXPECT_STRAT="${expected_strat}" EXPECT_K="${expected_k}" EXPECT_RATIO="${expected_ratio}" EXPECT_REQUIRES_STATE="${FORCE_AUDIT}" \
    EXPECT_DFU_LR="${DFU_LR}" EXPECT_FEDERASER_STEPS="${FEDERASER_CALIBRATION_STEPS}" EXPECT_FEDERASER_INTERVAL="${FEDERASER_CALIBRATION_INTERVAL}" \
    EXPECT_FEDOSD_UNLEARN_ROUNDS="${FEDOSD_UNLEARN_ROUNDS}" EXPECT_FEDOSD_UNLEARN_LR="${FEDOSD_UNLEARN_LR}" EXPECT_FEDOSD_RECOVERY_ROUNDS="${FEDOSD_RECOVERY_ROUNDS}" EXPECT_FEDOSD_RECOVERY_LOCAL_STEPS="${FEDOSD_RECOVERY_LOCAL_STEPS}" EXPECT_FEDOSD_RECOVERY_LR="${FEDOSD_RECOVERY_LR}" EXPECT_FEDOSD_RETAIN_GRAD_SAMPLES="${FEDOSD_RETAIN_GRAD_SAMPLES}" \
    EXPECT_FEDRECOVERY_CORRECTION_WEIGHT="${FEDRECOVERY_CORRECTION_WEIGHT}" EXPECT_FEDRECOVERY_NOISE_STD="${FEDRECOVERY_NOISE_STD}" EXPECT_FEDRECOVERY_RECOVERY_ROUNDS="${FEDRECOVERY_RECOVERY_ROUNDS}" EXPECT_FEDRECOVERY_RECOVERY_LOCAL_STEPS="${FEDRECOVERY_RECOVERY_LOCAL_STEPS}" EXPECT_FEDRECOVERY_RECOVERY_LR="${FEDRECOVERY_RECOVERY_LR}" \
    EXPECT_OBLIVIONIS_UNLEARN_ROUNDS="${OBLIVIONIS_UNLEARN_ROUNDS}" EXPECT_OBLIVIONIS_UNLEARN_LR="${OBLIVIONIS_UNLEARN_LR}" EXPECT_OBLIVIONIS_PROPAGATION_ROUNDS="${OBLIVIONIS_PROPAGATION_ROUNDS}" EXPECT_OBLIVIONIS_PROPAGATION_LR="${OBLIVIONIS_PROPAGATION_LR}" EXPECT_OBLIVIONIS_PROPAGATION_LOCAL_STEPS="${OBLIVIONIS_PROPAGATION_LOCAL_STEPS}" \
    EXPECT_TDB_SKETCH_DIM="${TDB_SKETCH_DIM}" EXPECT_TDB_MAX_INTERVALS="${TDB_MAX_INTERVALS}" EXPECT_TDB_TIME_LIMIT="${TDB_TIME_LIMIT}" EXPECT_TDB_ALPHA_U="${TDB_ALPHA_U}" EXPECT_TDB_ALPHA_P="${TDB_ALPHA_P}" EXPECT_TDB_ALPHA_Q="${TDB_ALPHA_Q}" EXPECT_TDB_TAU_Q="${TDB_TAU_Q}" EXPECT_TDB_AGGREGATION_SCOPE="${TDB_AGGREGATION_SCOPE}" \
    EXPECT_BACKDOOR_TRIGGER="${TRIGGER}" EXPECT_BACKDOOR_RATE="${BACKDOOR_RATE}" EXPECT_BACKDOOR_TARGET_AGENT="${TARGET_AGENT}" EXPECT_BACKDOOR_TARGET_LABEL="${TARGET_LABEL}" EXPECT_BACKDOOR_POSITION="${BACKDOOR_POSITION}" python - <<'PY'
import glob
import json
import os
import sys

run_base = os.environ["RUN_BASE"]
expect_seed = int(os.environ["EXPECT_SEED"])
expect_max_eval = int(os.environ["EXPECT_MAX_EVAL"])
expect_algo = os.environ["EXPECT_ALGO"]
expect_strat = os.environ["EXPECT_STRAT"]
expect_k = int(float(os.environ["EXPECT_K"]))
expect_ratio = float(os.environ["EXPECT_RATIO"])
expect_requires_state = str(os.environ.get("EXPECT_REQUIRES_STATE", "0")) == "1"
expect_backdoor_trigger = str(os.environ.get("EXPECT_BACKDOOR_TRIGGER") or "")
expect_backdoor_rate = os.environ.get("EXPECT_BACKDOOR_RATE") or "0"
expect_backdoor_target_agent = os.environ.get("EXPECT_BACKDOOR_TARGET_AGENT") or ""
expect_backdoor_target_label = os.environ.get("EXPECT_BACKDOOR_TARGET_LABEL") or ""
expect_backdoor_position = os.environ.get("EXPECT_BACKDOOR_POSITION") or "prefix"
try:
    expect_is_backdoor = bool(expect_backdoor_trigger) and float(expect_backdoor_rate) > 0.0
except Exception:
    expect_is_backdoor = False


def same_float(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def same_int(a, b):
    try:
        return int(float(a)) == int(float(b))
    except Exception:
        return False


def check_common(cfg):
    if cfg.get("dfu_algorithm") != expect_algo:
        return False
    if not same_float(cfg.get("lr"), os.environ["EXPECT_DFU_LR"]):
        return False

    wants_tdb = expect_strat in {"ours_all", "ours_ours"}
    if wants_tdb:
        if cfg.get("selection_strategy") != "tdb":
            return False
        if not same_int(cfg.get("selection_count"), expect_k):
            return False
        for key, env_key, cmp in [
            ("tdb_sketch_dim", "EXPECT_TDB_SKETCH_DIM", same_int),
            ("tdb_max_intervals", "EXPECT_TDB_MAX_INTERVALS", same_int),
            ("tdb_time_limit", "EXPECT_TDB_TIME_LIMIT", same_float),
            ("tdb_alpha_u", "EXPECT_TDB_ALPHA_U", same_float),
            ("tdb_alpha_p", "EXPECT_TDB_ALPHA_P", same_float),
            ("tdb_alpha_q", "EXPECT_TDB_ALPHA_Q", same_float),
            ("tdb_tau_q", "EXPECT_TDB_TAU_Q", same_float),
        ]:
            if not cmp(cfg.get(key), os.environ[env_key]):
                return False
        if str(cfg.get("tdb_aggregation_scope") or "") != os.environ["EXPECT_TDB_AGGREGATION_SCOPE"]:
            return False
    else:
        if cfg.get("selection_strategy") != "full":
            return False

    wants_lora = expect_strat in {"full_ours", "ours_ours"}
    if bool(cfg.get("enable_param_selection")) != wants_lora:
        return False
    if wants_lora and not same_float(cfg.get("param_selection_ratio"), expect_ratio):
        return False
    return True


def check_method(cfg):
    if expect_algo == "d-federaser":
        return same_int(cfg.get("calibration_steps"), os.environ["EXPECT_FEDERASER_STEPS"]) and same_int(cfg.get("calibration_interval"), os.environ["EXPECT_FEDERASER_INTERVAL"])
    if expect_algo == "d-fedosd":
        return (
            same_int(cfg.get("unlearn_rounds"), os.environ["EXPECT_FEDOSD_UNLEARN_ROUNDS"])
            and same_float(cfg.get("unlearn_lr"), os.environ["EXPECT_FEDOSD_UNLEARN_LR"])
            and same_int(cfg.get("recovery_rounds"), os.environ["EXPECT_FEDOSD_RECOVERY_ROUNDS"])
            and same_int(cfg.get("recovery_local_steps"), os.environ["EXPECT_FEDOSD_RECOVERY_LOCAL_STEPS"])
            and same_float(cfg.get("recovery_lr"), os.environ["EXPECT_FEDOSD_RECOVERY_LR"])
            and same_int(cfg.get("retain_grad_samples"), os.environ["EXPECT_FEDOSD_RETAIN_GRAD_SAMPLES"])
        )
    if expect_algo == "d-fedrecovery":
        return (
            same_float(cfg.get("correction_weight"), os.environ["EXPECT_FEDRECOVERY_CORRECTION_WEIGHT"])
            and same_float(cfg.get("noise_std"), os.environ["EXPECT_FEDRECOVERY_NOISE_STD"])
            and same_int(cfg.get("recovery_rounds"), os.environ["EXPECT_FEDRECOVERY_RECOVERY_ROUNDS"])
            and same_int(cfg.get("recovery_local_steps"), os.environ["EXPECT_FEDRECOVERY_RECOVERY_LOCAL_STEPS"])
            and same_float(cfg.get("recovery_lr"), os.environ["EXPECT_FEDRECOVERY_RECOVERY_LR"])
        )
    if expect_algo == "d-oblivionis":
        ok = (
            same_int(cfg.get("unlearn_rounds"), os.environ["EXPECT_OBLIVIONIS_UNLEARN_ROUNDS"])
            and same_float(cfg.get("unlearn_lr"), os.environ["EXPECT_OBLIVIONIS_UNLEARN_LR"])
            and same_int(cfg.get("propagation_rounds"), os.environ["EXPECT_OBLIVIONIS_PROPAGATION_ROUNDS"])
            and same_float(cfg.get("propagation_lr"), os.environ["EXPECT_OBLIVIONIS_PROPAGATION_LR"])
        )
        pls = os.environ.get("EXPECT_OBLIVIONIS_PROPAGATION_LOCAL_STEPS") or ""
        if pls:
            ok = ok and same_int(cfg.get("propagation_local_steps"), pls)
        return ok
    return False


def check_backdoor_replay(cfg):
    if not expect_is_backdoor:
        return True
    replay = cfg.get("backdoor_forget_replay") or {}
    if not bool(replay.get("enabled")):
        return False
    if str(replay.get("trigger")) != expect_backdoor_trigger:
        return False
    if not same_float(replay.get("poison_rate"), expect_backdoor_rate):
        return False
    if expect_backdoor_target_agent and not same_int(replay.get("poison_agent"), expect_backdoor_target_agent):
        return False
    if expect_backdoor_target_label and not same_int(replay.get("target_label"), expect_backdoor_target_label):
        return False
    if str(replay.get("position") or "prefix") != str(expect_backdoor_position or "prefix"):
        return False
    try:
        if int(replay.get("poisoned", 0)) <= 0:
            return False
    except Exception:
        return False
    return True


def has_any_state(run_dir):
    for pattern in [
        "agent_*/lora_state.pt",
        "final/agent_*/lora_state.pt",
        "round_*/agent_*/lora_state.pt",
    ]:
        if glob.glob(os.path.join(run_dir, pattern)):
            return True
    return False

for cfg_path in sorted(glob.glob(os.path.join(run_base, "dfu_*", "dfu_config.json"))):
    history_path = os.path.join(os.path.dirname(cfg_path), "history.json")
    if not os.path.isfile(history_path):
        continue
    try:
        cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
    except Exception:
        continue
    if int(cfg.get("seed", -1)) != expect_seed:
        continue

    cfg_max_eval = cfg.get("max_eval_samples", None)
    if cfg_max_eval is not None:
        try:
            cfg_max_eval = int(cfg_max_eval)
        except Exception:
            cfg_max_eval = None
    if cfg_max_eval is not None and cfg_max_eval != expect_max_eval:
        continue

    if not check_common(cfg):
        continue
    if not check_method(cfg):
        continue
    if not check_backdoor_replay(cfg):
        continue

    run_dir = os.path.dirname(cfg_path)
    if expect_requires_state and not has_any_state(run_dir):
        continue

    print(run_dir)
    raise SystemExit(0)

raise SystemExit(1)
PY
}

run_one_dfu_and_audit() {
  local dfl_snapshot="$1"
  local retrain_dir="$2"
  local sens_cache="$3"
  local algo="$4"
  local strat="$5"
  local k="$6"
  local ratio="$7"
  local snap_name
  snap_name="$(basename "${dfl_snapshot}")"

  local strategy_dir=""
  local -a sel=(--selection_strategy full)
  local -a lora=()

  local ratio_fmt
  ratio_fmt="$(format_ratio "${ratio}")"

  case "${strat}" in
    full_all)
      strategy_dir="strategy_full"
      ;;
    ours_all)
      strategy_dir="strategy_tdb_count${k}"
      sel=(--selection_strategy tdb --selection_count "${k}" --tdb_sketch_dim "${TDB_SKETCH_DIM}" --tdb_max_intervals "${TDB_MAX_INTERVALS}" --tdb_time_limit "${TDB_TIME_LIMIT}" --tdb_alpha_u "${TDB_ALPHA_U}" --tdb_alpha_p "${TDB_ALPHA_P}" --tdb_alpha_q "${TDB_ALPHA_Q}" --tdb_tau_q "${TDB_TAU_Q}" --tdb_aggregation_scope "${TDB_AGGREGATION_SCOPE}")
      ;;
    full_ours)
      strategy_dir="strategy_full_lora${ratio_fmt}_topratio_ours"
      lora=(--enable_param_selection --param_selection_mode top_ratio --param_selection_ratio "${ratio_fmt}" --param_sensitivity_cache "${sens_cache}")
      ;;
    ours_ours)
      strategy_dir="strategy_tdb_count${k}_lora${ratio_fmt}_topratio_ours"
      sel=(--selection_strategy tdb --selection_count "${k}" --tdb_sketch_dim "${TDB_SKETCH_DIM}" --tdb_max_intervals "${TDB_MAX_INTERVALS}" --tdb_time_limit "${TDB_TIME_LIMIT}" --tdb_alpha_u "${TDB_ALPHA_U}" --tdb_alpha_p "${TDB_ALPHA_P}" --tdb_alpha_q "${TDB_ALPHA_Q}" --tdb_tau_q "${TDB_TAU_Q}" --tdb_aggregation_scope "${TDB_AGGREGATION_SCOPE}")
      lora=(--enable_param_selection --param_selection_mode top_ratio --param_selection_ratio "${ratio_fmt}" --param_sensitivity_cache "${sens_cache}")
      ;;
    *)
      echo "[grid][ERROR] unknown strategy=${strat}" >&2
      exit 1
      ;;
  esac

  local -a algo_extra=()
  case "${algo}" in
    d-federaser)
      algo_extra=(--calibration_steps "${FEDERASER_CALIBRATION_STEPS}" --calibration_interval "${FEDERASER_CALIBRATION_INTERVAL}")
      ;;
    d-fedosd)
      algo_extra=(--unlearn_rounds "${FEDOSD_UNLEARN_ROUNDS}" --unlearn_lr "${FEDOSD_UNLEARN_LR}" --recovery_rounds "${FEDOSD_RECOVERY_ROUNDS}" --recovery_local_steps "${FEDOSD_RECOVERY_LOCAL_STEPS}" --recovery_lr "${FEDOSD_RECOVERY_LR}" --retain_grad_samples "${FEDOSD_RETAIN_GRAD_SAMPLES}")
      if [[ -n "${FEDOSD_ORTHOGONAL_UPDATE_NORM}" ]]; then
        algo_extra+=(--fedosd_orthogonal_update_norm "${FEDOSD_ORTHOGONAL_UPDATE_NORM}")
      fi
      ;;
    d-fedrecovery)
      algo_extra=(--correction_weight "${FEDRECOVERY_CORRECTION_WEIGHT}" --noise_std "${FEDRECOVERY_NOISE_STD}" --recovery_rounds "${FEDRECOVERY_RECOVERY_ROUNDS}" --recovery_local_steps "${FEDRECOVERY_RECOVERY_LOCAL_STEPS}" --recovery_lr "${FEDRECOVERY_RECOVERY_LR}")
      ;;
    d-oblivionis)
      algo_extra=(--unlearn_rounds "${OBLIVIONIS_UNLEARN_ROUNDS}" --unlearn_lr "${OBLIVIONIS_UNLEARN_LR}" --propagation_rounds "${OBLIVIONIS_PROPAGATION_ROUNDS}" --propagation_lr "${OBLIVIONIS_PROPAGATION_LR}")
      if [[ -n "${OBLIVIONIS_PROPAGATION_LOCAL_STEPS}" ]]; then
        algo_extra+=(--propagation_local_steps "${OBLIVIONIS_PROPAGATION_LOCAL_STEPS}")
      fi
      ;;
    *)
      echo "[grid][ERROR] unknown algorithm=${algo}" >&2
      exit 1
      ;;
  esac

  local run_base="${DFU_OUTPUT_DIR}/${DATASET}/${algo}/${strategy_dir}/K10/G10_L5/alpha0.5/${snap_name}"
  local legacy_run_base="${LEGACY_DFU_OUTPUT_DIR}/${DATASET}/${algo}/${strategy_dir}/K10/G10_L5/alpha0.5/${snap_name}"
  local run_dir=""

  if [[ "${FORCE_DFU}" != "1" ]] && run_dir="$(has_valid_run "${run_base}" "${SEED}" "${MAX_EVAL_SAMPLES}" "${algo}" "${strat}" "${k}" "${ratio_fmt}" 2>/dev/null)"; then
    echo "[grid] Reuse DFU ${algo}/${strat}: ${run_dir}"
  elif [[ "${FORCE_DFU}" != "1" ]] && run_dir="$(has_valid_run "${legacy_run_base}" "${SEED}" "${MAX_EVAL_SAMPLES}" "${algo}" "${strat}" "${k}" "${ratio_fmt}" 2>/dev/null)"; then
    echo "[grid] Reuse LEGACY DFU ${algo}/${strat}: ${run_dir}"
  else
    echo "[grid] Run DFU ${algo}/${strat} (k=${k}, r=${ratio_fmt})"
    python -u scripts/run_dfu.py \
      --dfl_snapshot "${dfl_snapshot}" \
      --dfu_algorithm "${algo}" \
      --output_dir "${DFU_OUTPUT_DIR}" \
      --target_agent "${TARGET_AGENT}" \
      --seed "${SEED}" \
      --batch_size "${DFU_BATCH_SIZE}" \
      --grad_accum_steps "${DFU_GRAD_ACCUM_STEPS}" \
      --lr "${DFU_LR}" \
      --eval_every 0 \
      --max_eval_samples "${MAX_EVAL_SAMPLES}" \
      --gpu "${LOGICAL_GPU}" \
      "${algo_extra[@]}" \
      "${sel[@]}" \
      "${lora[@]}" \
      2>&1 | tee "logs/backdoor_grid_dfu_${DATASET}_seed${SEED}_${algo}_${strat}_gpu${PHYSICAL_GPU}.log"

    run_dir="$(ls -d "${run_base}/dfu_"* 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "${run_dir}" || ! -f "${run_dir}/history.json" ]]; then
      echo "[grid][ERROR] DFU output missing: ${run_base}" >&2
      exit 1
    fi
  fi

  local tag_base="${DATASET}_seed${SEED}_${algo}_${strat}"
  local bd_tag="bd_grid_${tag_base}${TAG_SUFFIX}"
  local mia_tag="mia_grid_${tag_base}${TAG_SUFFIX}_nonmemberVAL"
  local audit_eval_agent_id="${EVAL_AGENT_ID}"

  if [[ "${EVAL_SCOPE}" == "single" ]]; then
    audit_eval_agent_id="$(
      RUN_DIR_ENV="${run_dir}" DEFAULT_AID_ENV="${EVAL_AGENT_ID}" TARGET_AGENT_ENV="${TARGET_AGENT}" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR_ENV"])
default_aid = int(os.environ["DEFAULT_AID_ENV"])
target_agent = int(os.environ["TARGET_AGENT_ENV"])

def has_state(aid: int) -> bool:
    candidates = [
        run_dir / "final" / f"agent_{aid}" / "lora_state.pt",
        run_dir / f"agent_{aid}" / "lora_state.pt",
    ]
    round_dirs = []
    if run_dir.exists():
        round_dirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("round_")],
            key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else -1,
        )
    for rd in reversed(round_dirs):
        candidates.append(rd / f"agent_{aid}" / "lora_state.pt")
    return any(p.exists() for p in candidates)

if has_state(default_aid):
    print(default_aid)
    raise SystemExit(0)

cfg = {}
try:
    cfg = json.loads((run_dir / "dfu_config.json").read_text(encoding="utf-8"))
except Exception:
    cfg = {}

selected = []
for raw in cfg.get("selected_agents") or []:
    try:
        aid = int(raw)
    except Exception:
        continue
    if aid != target_agent and has_state(aid):
        selected.append(aid)

if selected:
    print(selected[0])
    raise SystemExit(0)

available = set()
for parent in [run_dir / "final", run_dir]:
    if not parent.exists():
        continue
    for p in parent.glob("agent_*/lora_state.pt"):
        try:
            aid = int(p.parent.name.split("_", 1)[1])
        except Exception:
            continue
        if aid != target_agent:
            available.add(aid)

print(min(available) if available else default_aid)
PY
    )"
    if [[ "${audit_eval_agent_id}" != "${EVAL_AGENT_ID}" ]]; then
      echo "[grid] single audit eval_agent_id ${EVAL_AGENT_ID} has no DFU state; using selected/available agent ${audit_eval_agent_id}"
    fi
  fi

  local bd_json="${ARTIFACT_ROOT}/unlearning_audit/backdoor/${bd_tag}/backdoor_audit.json"
  local legacy_bd_json="artifacts/unlearning_audit/backdoor/${bd_tag}/backdoor_audit.json"
  local mia_json="${ARTIFACT_ROOT}/unlearning_audit/mia/${mia_tag}/mia_audit.json"
  local legacy_mia_json="artifacts/unlearning_audit/mia/${mia_tag}/mia_audit.json"

  if [[ "${FORCE_AUDIT}" == "1" ]]; then
    python -u scripts/eval_backdoor_audit.py \
      --dfl_snapshot "${dfl_snapshot}" \
      --dfu_dir "${run_dir}" \
      --retrain_dir "${retrain_dir}" \
      --target_agent "${TARGET_AGENT}" \
      --eval_agent_id "${audit_eval_agent_id}" \
        --eval_scope "${EVAL_SCOPE}" \
        --dfu_state_mode "${DFU_STATE_MODE}" \
        --dfl_eval_scope "${DFL_EVAL_SCOPE}" \
        --audit_models "${BACKDOOR_AUDIT_MODELS}" \
        --trigger "${TRIGGER}" \
      --trigger_position "${BACKDOOR_POSITION}" \
      --sample_source "${BACKDOOR_SAMPLE_SOURCE}" \
      --target_label "${TARGET_LABEL}" \
      --max_samples "${AUDIT_MAX_SAMPLES}" \
      --batch_size "${AUDIT_BATCH_SIZE}" \
      --gpu "${LOGICAL_GPU}" \
      --tag "${bd_tag}" \
      2>&1 | tee "logs/backdoor_grid_eval_${DATASET}_seed${SEED}_${algo}_${strat}_gpu${PHYSICAL_GPU}.log"
  elif [[ -f "${bd_json}" ]]; then
    echo "[grid] Reuse backdoor audit: ${bd_json}"
  elif [[ -f "${legacy_bd_json}" ]]; then
    echo "[grid] Reuse LEGACY backdoor audit: ${legacy_bd_json}"
  else
    python -u scripts/eval_backdoor_audit.py \
      --dfl_snapshot "${dfl_snapshot}" \
      --dfu_dir "${run_dir}" \
      --retrain_dir "${retrain_dir}" \
      --target_agent "${TARGET_AGENT}" \
      --eval_agent_id "${audit_eval_agent_id}" \
      --eval_scope "${EVAL_SCOPE}" \
      --dfu_state_mode "${DFU_STATE_MODE}" \
      --dfl_eval_scope "${DFL_EVAL_SCOPE}" \
      --audit_models "${BACKDOOR_AUDIT_MODELS}" \
      --trigger "${TRIGGER}" \
      --trigger_position "${BACKDOOR_POSITION}" \
      --sample_source "${BACKDOOR_SAMPLE_SOURCE}" \
      --target_label "${TARGET_LABEL}" \
      --max_samples "${AUDIT_MAX_SAMPLES}" \
      --batch_size "${AUDIT_BATCH_SIZE}" \
      --gpu "${LOGICAL_GPU}" \
      --tag "${bd_tag}" \
      2>&1 | tee "logs/backdoor_grid_eval_${DATASET}_seed${SEED}_${algo}_${strat}_gpu${PHYSICAL_GPU}.log"
  fi

  if [[ "${SKIP_MIA}" == "1" ]]; then
    echo "[grid] Skip MIA audit: ${mia_tag}"
  elif [[ "${FORCE_AUDIT}" == "1" ]]; then
    python -u scripts/eval_unlearning_detectors.py \
      --dfl_snapshot "${dfl_snapshot}" \
      --dfu_dir "${run_dir}" \
      --retrain_dir "${retrain_dir}" \
      --target_agent "${TARGET_AGENT}" \
      --eval_agent_id "${audit_eval_agent_id}" \
      --max_samples "${AUDIT_MAX_SAMPLES}" \
      --batch_size "${MIA_BATCH_SIZE}" \
      --nonmember_source val \
      --gpu "${LOGICAL_GPU}" \
      --tag "${mia_tag}" \
      2>&1 | tee "logs/backdoor_grid_mia_${DATASET}_seed${SEED}_${algo}_${strat}_gpu${PHYSICAL_GPU}.log"
  elif [[ -f "${mia_json}" ]]; then
    echo "[grid] Reuse MIA audit: ${mia_json}"
  elif [[ -f "${legacy_mia_json}" ]]; then
    echo "[grid] Reuse LEGACY MIA audit: ${legacy_mia_json}"
  else
    python -u scripts/eval_unlearning_detectors.py \
      --dfl_snapshot "${dfl_snapshot}" \
      --dfu_dir "${run_dir}" \
      --retrain_dir "${retrain_dir}" \
      --target_agent "${TARGET_AGENT}" \
      --eval_agent_id "${audit_eval_agent_id}" \
      --max_samples "${AUDIT_MAX_SAMPLES}" \
      --batch_size "${MIA_BATCH_SIZE}" \
      --nonmember_source val \
      --gpu "${LOGICAL_GPU}" \
      --tag "${mia_tag}" \
      2>&1 | tee "logs/backdoor_grid_mia_${DATASET}_seed${SEED}_${algo}_${strat}_gpu${PHYSICAL_GPU}.log"
  fi

  if [[ "${BACKDOOR_CLEANUP_DFU_STATES}" == "1" && -d "${run_dir}" ]]; then
    find "${run_dir}" -type f -name 'lora_state.pt' -delete
  fi
}

DFL_SNAP="$(run_if_needed_dfl)"
echo "[grid] DFL_SNAP=${DFL_SNAP}"

RETRAIN_DIR="$(run_if_needed_retrain "${DFL_SNAP}")"
echo "[grid] RETRAIN_DIR=${RETRAIN_DIR}"

SENS_PATH="${ARTIFACT_ROOT}/unlearning_audit/sens_cache/${DATASET}_$(basename "${DFL_SNAP}")_agent${TARGET_AGENT}.json"
ensure_sens_cache "${DFL_SNAP}" "${SENS_PATH}"
echo "[grid] SENS_PATH=${SENS_PATH}"

IFS=',' read -r -a algo_arr <<< "${ALGORITHMS}"
IFS=',' read -r -a strat_arr <<< "${STRATEGIES}"

for algo in "${algo_arr[@]}"; do
  algo="${algo//[[:space:]]/}"
  [[ -z "${algo}" ]] && continue
  for strat in "${strat_arr[@]}"; do
    strat="${strat//[[:space:]]/}"
    [[ -z "${strat}" ]] && continue
    cfg="$(read_best_cfg "${DATASET}" "${algo}" "${META_JSON}" "${strat}" "${DSU_META_JSON}")"
    k="${cfg%%,*}"
    r="${cfg##*,}"
    if [[ -n "${OVERRIDE_K}" ]]; then
      k="${OVERRIDE_K}"
    fi
    if [[ -n "${OVERRIDE_R}" ]]; then
      r="${OVERRIDE_R}"
    fi
    run_one_dfu_and_audit "${DFL_SNAP}" "${RETRAIN_DIR}" "${SENS_PATH}" "${algo}" "${strat}" "${k}" "${r}"
  done
done

echo "[grid] DONE dataset=${DATASET} seed=${SEED}"
echo "[grid] DFL=${DFL_SNAP}"
echo "[grid] RETRAIN=${RETRAIN_DIR}"
