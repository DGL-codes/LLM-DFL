# Reproduction Commands

Exact commands used to reproduce the main results. The fixed entry point is
`scripts/run_tdb_fair_grid.py`. All main results use the same evaluation caliber:
`--max_eval_samples 100 --batch_size 4 --grad_accum_steps 2 --lr 1e-3`, and the
per-method unlearning hyperparameters are selected by `--profile paper`.

## Environment

```bash
conda env create -f environment_dfu.yml
conda activate uld
export LLMDFL_LOCAL_FILES_ONLY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
```

## Main table and AS / LS / DSU sweeps (Base, AS, LS, DSU)

The main table reports 3-seed means. The AS budget sweep (per selection count `k`),
the LS ratio sweep (per module ratio `r`), and the DSU joint `k x r` grid are all
produced by the same fixed entry.

```bash
python scripts/run_tdb_fair_grid.py \
  --datasets 20newsgroups,yahoo_subset \
  --algorithms d-federaser,d-fedosd,d-fedrecovery,d-oblivionis \
  --settings base,as,ls,dsu \
  --ks 1,2,3,4,5,6,7,8,9 \
  --rs 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --seeds 42,43,44 \
  --profile paper \
  --max_eval_samples 100 --batch_size 4 --grad_accum_steps 2 --lr 1e-3 \
  --physical_gpus 0,1,2,3 \
  --out_dir artifacts/tdb_fair_grid
```

For a single-seed trend check, use `--seeds 42`. The AS sweep uses
`--settings as`, the LS sweep uses `--settings ls`, and the DSU heatmap uses
`--settings dsu`.

## Sequential cumulative withdrawal (Table VI)

FedEraser with a cumulative removed set that grows `{0}` -> `{0,1}` -> `{0,1,2}`.
20Newsgroups uses `k=4, r=0.5`; Yahoo uses `k=5, r=0.3`. Each step calls
`scripts/run_dfu.py` with the DFL snapshot, `--target_agent` for the current step,
and `--removed_agents` for the previously removed agents, e.g. step 3:

```bash
python scripts/run_dfu.py \
  --dfl_snapshot checkpoints/20newsgroups/K10/G10_L5/alpha0.5/<seed42-snapshot> \
  --dfu_algorithm d-federaser --seed 42 \
  --selection_strategy tdb --selection_count 4 \
  --tdb_sketch_dim 64 --tdb_max_intervals 3 --tdb_aggregation_scope local \
  --enable_param_selection --param_selection_mode top_ratio --param_selection_ratio 0.5 \
  --calibration_steps 5 --calibration_interval 2 \
  --max_eval_samples 100 --batch_size 4 --grad_accum_steps 2 --lr 1e-3 \
  --removed_agents 0,1 --target_agent 2 --gpu 0 \
  --output_dir artifacts/sequential_cumulative_tdb_dsu/20news_step3
```

## Backdoor forgetting audit (Table IV)

Per-dataset pipeline (poisoned DFL training, unlearning, retraining, and ASR audit)
covering all four methods:

```bash
bash scripts/backdoor_audit_grid_pipeline.sh \
  --dataset 20newsgroups --physical_gpu 0 --gpu 0 --seed 42 \
  --target_agent 0 --target_label 0 --backdoor_rate 0.5 --trigger cf_trigger_xzq
```

## Aggregation and figures

```bash
python scripts/report_tdb_clean_final_local_20260602.py
python scripts/report_sequential_cumulative_20260603.py
python scripts/report_backdoor_direct_forgetting_table_20260601.py \
  --tag_contains bd_grid --asr_family asr_non_target \
  --out_csv reports/backdoor_forgetting.csv --out_md reports/backdoor_forgetting.md
python scripts/plot_tdb_final_local_20260603.py
```
