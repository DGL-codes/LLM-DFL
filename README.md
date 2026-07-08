# LLM DFL/DFU Reproducible Release

This repository contains the code and compact result layer for reproducing the LLM decentralized federated learning and decentralized federated unlearning experiments.

The project studies:

- decentralized federated learning with LoRA fine-tuning;
- decentralized federated unlearning after one or more agents withdraw;
- four unlearning methods: D-FedEraser, D-FedRecovery, D-FedOSD, and D-Oblivionis;
- four comparison settings: Base, AS, LS, and DSU;
- two main datasets: 20Newsgroups and Yahoo subset;
- clean public-test F1, MIA AUC, backdoor ASR as a direct forgetting audit, and additional validation experiments.

The AS implementation used by the final tables is a trajectory-aware distribution-balanced mixed-integer agent selection method. In the command-line interface it is selected by `--selection_strategy tdb`. It does not use F1, MIA, or backdoor ASR as a selection signal.

## What Is Included

- `src/`: core implementation.
- `scripts/`: fixed training, unlearning, auditing, sweeping, and reporting entrypoints.
- `data/yahoo_subset/`: small Yahoo subset data included with this release.
- `reports/`: final compact result files, raw CSV sweep tables, and plots.
- `requirements_dfu.txt` and `environment_dfu.yml`: environment specifications.

## What Is Not Included

The release intentionally excludes large local artifacts:

- base model weights;
- DFL/DFU/retrain checkpoints;
- raw logs;
- raw run artifacts;
- local caches;
- local CodeGraph index files.

To reproduce full experiments from scratch, prepare the required base model locally. In our local runs, the TinyLlama base model was usually available at:

```bash
models/TinyLlama-1.1B-Chat-v1.0
```

## Environment

The working conda environment used in the local experiments was `uld`.

```bash
conda activate uld
export LLMDFL_LOCAL_FILES_ONLY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
```

Install dependencies from either:

```bash
conda env create -f environment_dfu.yml
```

or:

```bash
pip install -r requirements_dfu.txt
```

The exact GPU mapping convention is:

- `--physical_gpu`: real physical GPU id used by shell entrypoints through `CUDA_VISIBLE_DEVICES`;
- `--gpu`: logical id inside the visible device set, usually `0`.

## Fixed Reproduction Entrypoints

### 1. DFL Training

```bash
python -u scripts/train_dfl.py \
  --dataset 20newsgroups \
  --output_dir checkpoints \
  --num_agents 10 \
  --alpha 0.5 \
  --seed 42 \
  --global_rounds 10 \
  --local_steps 5 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --lr 1e-3 \
  --eval_every 0 \
  --max_eval_samples 2000 \
  --gpu 0
```

### 2. Single DFU Run

Base:

```bash
python -u scripts/run_dfu.py \
  --dfl_snapshot checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_RUN_ID \
  --dfu_algorithm d-federaser \
  --output_dir dfu_checkpoints \
  --target_agent 0 \
  --selection_strategy full \
  --seed 42 \
  --gpu 0
```

DSU with TDB-AS and LoRA module selection:

```bash
python -u scripts/run_dfu.py \
  --dfl_snapshot checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_RUN_ID \
  --dfu_algorithm d-federaser \
  --output_dir dfu_checkpoints \
  --target_agent 0 \
  --selection_strategy tdb \
  --selection_count 4 \
  --tdb_sketch_dim 16 \
  --tdb_max_intervals 2 \
  --tdb_time_limit 20 \
  --tdb_alpha_u 1.0 \
  --tdb_alpha_p 1.0 \
  --tdb_alpha_q 0.1 \
  --tdb_tau_q 0.0 \
  --tdb_aggregation_scope local \
  --enable_param_selection \
  --param_selection_mode top_ratio \
  --param_selection_ratio 0.8 \
  --seed 42 \
  --gpu 0
```

The final paper-facing DSU results use local ring aggregation, not global aggregation.

### 3. Strict LLM Pipeline

```bash
bash scripts/strict_repro_llm_pipeline.sh --physical_gpu 0 --gpu 0 --seeds 42,43,44
```

### 4. Clean AS/LS/DSU Sweep

```bash
python scripts/run_tdb_fair_grid.py \
  --datasets 20newsgroups,yahoo_subset \
  --algorithms d-federaser,d-fedosd,d-fedrecovery,d-oblivionis \
  --seeds 42,43,44 \
  --settings base,as,ls,dsu \
  --ks 1,2,3,4,5,6,7,8,9 \
  --rs 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --profile paper \
  --max_eval_samples 100 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --lr 1e-3 \
  --physical_gpus 0,1,2,3 \
  --out_dir artifacts/tdb_fair_grid
```

### 5. Backdoor and MIA Audit

```bash
bash scripts/backdoor_audit_grid_pipeline.sh \
  --dataset 20newsgroups \
  --physical_gpu 0 \
  --gpu 0 \
  --seed 42 \
  --target_agent 0 \
  --target_label 0 \
  --backdoor_rate 0.5 \
  --trigger cf_trigger_xzq
```

### 6. Final Table Aggregation

```bash
python scripts/report_final_tables_20260603.py
python scripts/report_tdb_clean_final_local_20260602.py
python scripts/report_sequential_cumulative_20260603.py
python scripts/plot_tdb_final_local_20260603.py
```

### 7. Release Health Check

```bash
python scripts/verify_github_release.py
```

## Main Results

The most important result files are:

- clean public-test F1/MIA: `reports/tdb_clean_final_local_f1_mia_20260602.md`;
- AS/LS sweeps: `reports/tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_all_rows.csv`;
- DSU k-by-r sweeps: `reports/tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_all_rows.csv`;
- backdoor forgetting audit: `reports/backdoor_forgetting_final_seed42_20260603.md`;
- sequential cumulative unlearning audit: `reports/sequential_cumulative_tdb_dsu_20260603.md`;
- unlearning detector validation summary: `reports/unlearning_detector_validation.md`.

## CodeGraph

The source tree was inspected with [colbymchenry/codegraph](https://github.com/colbymchenry/codegraph) during repository preparation. The generated index is not included because it is a derived local artifact. The uploaded code is the source under `src/` and the fixed-entry scripts under `scripts/`.
