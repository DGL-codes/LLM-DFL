# 后门攻击复现实验固定路径

更新时间：2026-05-22

## 固定结论

当前后门攻击闭环已经有完整结果，不是空白实验。

- 聚合结果：`reports/unlearning_detector_validation.md`
- 聚合 CSV：`reports/unlearning_detector_validation.csv`
- 原始审计 JSON：`artifacts/unlearning_audit/backdoor/*/backdoor_audit.json`
- MIA 辅助审计 JSON：`artifacts/unlearning_audit/mia/*/mia_audit.json`
- 固定训练/审计入口：`scripts/backdoor_audit_grid_pipeline.sh`
- 固定聚合入口：`scripts/report_unlearning_detector_grid.py`

当前聚合口径是：

- 数据集：`20newsgroups`、`yahoo_subset`
- 算法：`d-federaser`、`d-fedosd`、`d-fedrecovery`、`d-oblivionis`
- 策略：`full_all`、`full_ours`、`ours_all`、`ours_ours`
- seeds：`42,43,44`
- 后门触发器：`cf_trigger_xzq`
- target label：`0`
- MIA nonmember：`val`

## 当前结果

- 方向性通过：`32/32`
  - 所有组合都满足 `DFU ASR < DFL ASR`。
  - 说明 DFU 确实降低了后门攻击成功率。
- 接近重训通过：`19/32`
  - 以 `|DFU ASR - Retrain ASR| <= 0.05` 为阈值。
  - 说明不是所有策略都能忘到 retrain 水平。
- 最稳算法：`d-federaser`
  - `20newsgroups` 和 `yahoo_subset` 上 `8/8` 都接近 retrain。
  - 代价是 clean F1 损失较明显。
- 最不稳区域：
  - `20newsgroups / d-fedosd / full_ours`
  - `20newsgroups / d-oblivionis / full_ours`
  - `20newsgroups / d-fedosd / ours_ours`

## 最差异常格

| 数据集 | 算法 | 策略 | DFL ASR | DFU ASR | Retrain ASR | ASR gap | DFU clean F1 | Retrain clean F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 20newsgroups | d-fedosd | full_ours | 0.6034 | 0.4104 | 0.0279 | 0.3825 | 0.4960 | 0.6069 |
| 20newsgroups | d-oblivionis | full_ours | 0.6034 | 0.3409 | 0.0279 | 0.3130 | 0.4856 | 0.6069 |
| 20newsgroups | d-fedosd | ours_ours | 0.6034 | 0.3276 | 0.0279 | 0.2997 | 0.4804 | 0.6069 |
| 20newsgroups | d-oblivionis | ours_ours | 0.6034 | 0.2168 | 0.0279 | 0.1889 | 0.4669 | 0.6069 |
| 20newsgroups | d-fedrecovery | full_ours | 0.6034 | 0.1413 | 0.0279 | 0.1134 | 0.4711 | 0.6069 |

## 固定复现命令

先固定环境变量，允许四卡复现：

```bash
source /home/xzq/miniconda3/etc/profile.d/conda.sh
conda activate uld
export LLMDFL_LOCAL_FILES_ONLY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LLMDFL_ALLOWED_PHYSICAL_GPUS=0,1,2,3
export LLMDFL_EXPERIMENT_DIR="实验结果/运行产物"
```

单数据集单 seed 固定入口：

```bash
bash scripts/backdoor_audit_grid_pipeline.sh \
  --dataset 20newsgroups \
  --seed 42 \
  --physical_gpu 0 \
  --gpu 0 \
  --force_dfl_train 0 \
  --force_retrain 0 \
  --force_audit 0
```

四卡并行推荐分配：

```bash
bash scripts/backdoor_audit_grid_pipeline.sh --dataset 20newsgroups --seed 42 --physical_gpu 0 --gpu 0 --force_dfl_train 0 --force_retrain 0 --force_audit 0
bash scripts/backdoor_audit_grid_pipeline.sh --dataset 20newsgroups --seed 43 --physical_gpu 1 --gpu 0 --force_dfl_train 0 --force_retrain 0 --force_audit 0
bash scripts/backdoor_audit_grid_pipeline.sh --dataset 20newsgroups --seed 44 --physical_gpu 2 --gpu 0 --force_dfl_train 0 --force_retrain 0 --force_audit 0
bash scripts/backdoor_audit_grid_pipeline.sh --dataset yahoo_subset --seed 42 --physical_gpu 3 --gpu 0 --force_dfl_train 0 --force_retrain 0 --force_audit 0
```

剩余 yahoo seeds 可在任意空闲 GPU 上继续：

```bash
bash scripts/backdoor_audit_grid_pipeline.sh --dataset yahoo_subset --seed 43 --physical_gpu 0 --gpu 0 --force_dfl_train 0 --force_retrain 0 --force_audit 0
bash scripts/backdoor_audit_grid_pipeline.sh --dataset yahoo_subset --seed 44 --physical_gpu 1 --gpu 0 --force_dfl_train 0 --force_retrain 0 --force_audit 0
```

聚合报告固定命令：

```bash
python scripts/report_unlearning_detector_grid.py \
  --out_md reports/unlearning_detector_validation.md \
  --out_csv reports/unlearning_detector_validation.csv
```

## 口径说明

- 后门主指标不是 MIA，而是 ASR。
- 通过条件分两层：
  - 方向性：`DFU ASR < DFL ASR`
  - 接近重训：`|DFU ASR - Retrain ASR| <= 0.05`
- 当前可以写成：
  - “后门攻击遗忘方向性成立，所有组合 ASR 均下降。”
  - “部分组合不能达到 retrain 水平，尤其 20newsgroups 上 d-fedosd/d-oblivionis 的 LS 相关策略。”
- 当前不能写成：
  - “所有后门攻击实验都已经达到 retrain 水平。”
  - “所有组件组合都稳定提升后门遗忘效果。”
