# LLM 严格复现重跑记录

更新时间：2026-05-22

## 结论

本轮按 clean DFL snapshot 重新跑了论文表中 strict 失败的 9 个 cell，共 27 个 seed-level DFU 任务，并重新计算 `nonmember=val` 的 exact MIA。

当前结果不能压到 1 个点以内：

- F1-only：`6/16` 在 1 点内，`15/16` 在 3 点内。
- MIA-only：`0/16` 在 1 点内，`6/16` 在 3 点内。
- F1+MIA 同时通过：`0/16` 在 1 点内，`6/16` 在 3 点内。

主要偏差来自 MIA，而不是 F1。最大 MIA 偏差如下：

| Dataset | Method | DSU | ΔF1 | ΔMIA |
|---|---|:---:|---:|---:|
| yahoo_subset | FedOSD | ✓ | -2.18 | +11.67 |
| 20newsgroups | Oblivionis | ✓ | -2.79 | +11.50 |
| yahoo_subset | FedOSD | × | +5.52 | +11.13 |
| yahoo_subset | Oblivionis | × | +2.40 | +10.91 |
| 20newsgroups | FedOSD | × | -0.15 | +9.87 |

## 已确认的问题

1. 历史里能找到接近论文表的结果，但那属于 best-so-far / candidate search 口径，不等于 strict seed-aligned clean rerun。
2. 本轮开始时发现一个错误清单混入了 `checkpoints_backdoor_audit` 路径：27 行里 11 行 `dfl_snapshot` 指向后门 DFL，配置里明确有 `backdoor_trigger=cf_trigger_xzq`。这批任务已停止，不能作为 clean 论文复现证据。
3. 重新生成的 clean 清单来自 `artifacts/llm_selected_runs_424344.csv` 和 `artifacts/repro_pass_status_424344_latest.csv`，27 行中 `history_path/dfu_dir/dfl_snapshot` 均不含 backdoor 路径。
4. `20newsgroups / FedEraser / base / seed42` 原配置 `batch_size=8` + 全 LoRA 参数在 24GB GPU 上原样重试仍 OOM；本轮使用 `batch_size=4, grad_accum_steps=4` 做 fallback，并在结果中标记为 `batch4_grad4_fallback_after_exact_oom`。

## 固定产物

- clean 失败清单：`artifacts/strict_fail_cells_27_manifest_current_clean_20260522.csv`
- DFU 重跑合并：`artifacts/strict_fail_rerun_current_clean_20260522_merged_with_fallback.csv`
- exact val-MIA 输入：`artifacts/strict_fail_rerun_current_clean_20260522_mia_input.csv`
- exact val-MIA 合并：`artifacts/strict_fail_rerun_current_clean_20260522_mia_merged.csv`
- 48 行 run override：`artifacts/strict_fail_rerun_current_clean_20260522_run_override_48.csv`
- 48 行 MIA override：`artifacts/strict_fail_rerun_current_clean_20260522_mia_override_48.csv`
- 新聚合表：`artifacts/strict_fail_rerun_current_clean_20260522_report/`
- pass/fail 对比：`artifacts/strict_fail_rerun_current_clean_20260522_pass_status.csv`

## 当前 pass/fail 表

| Dataset | Method | DSU | Local F1 | ΔF1 | Local MIA | ΔMIA | pass@1pt | pass@3pt |
|---|---|:---:|---:|---:|---:|---:|:---:|:---:|
| 20newsgroups | FedEraser | × | 39.44±1.05 | -0.59 | 52.63±2.96 | +4.27 | 0 | 0 |
| 20newsgroups | FedEraser | ✓ | 44.67±1.91 | -0.27 | 51.12±1.43 | +2.32 | 0 | 1 |
| 20newsgroups | FedOSD | × | 47.83±2.59 | -0.15 | 55.00±2.33 | +9.87 | 0 | 0 |
| 20newsgroups | FedOSD | ✓ | 61.07±1.61 | +2.59 | 53.49±2.23 | +7.76 | 0 | 0 |
| 20newsgroups | FedRecovery | × | 48.58±2.08 | +2.40 | 49.79±1.41 | +2.45 | 0 | 1 |
| 20newsgroups | FedRecovery | ✓ | 61.15±0.23 | -0.08 | 55.36±1.69 | +5.89 | 0 | 0 |
| 20newsgroups | Oblivionis | × | 48.55±4.92 | +2.18 | 45.85±4.09 | +2.24 | 0 | 1 |
| 20newsgroups | Oblivionis | ✓ | 59.22±1.42 | -2.79 | 55.08±2.44 | +11.50 | 0 | 0 |
| yahoo_subset | FedEraser | × | 66.57±0.71 | -0.92 | 50.19±6.10 | +1.19 | 0 | 1 |
| yahoo_subset | FedEraser | ✓ | 68.31±1.87 | -1.02 | 49.82±5.83 | +2.08 | 0 | 1 |
| yahoo_subset | FedOSD | × | 66.49±5.19 | +5.52 | 52.41±6.84 | +11.13 | 0 | 0 |
| yahoo_subset | FedOSD | ✓ | 73.16±3.05 | -2.18 | 56.85±7.97 | +11.67 | 0 | 0 |
| yahoo_subset | FedRecovery | × | 69.32±1.33 | -1.50 | 53.56±7.27 | +3.05 | 0 | 0 |
| yahoo_subset | FedRecovery | ✓ | 73.84±1.10 | -0.44 | 54.44±7.43 | +1.33 | 0 | 1 |
| yahoo_subset | Oblivionis | × | 70.22±2.15 | +2.40 | 55.11±4.86 | +10.91 | 0 | 0 |
| yahoo_subset | Oblivionis | ✓ | 71.97±1.46 | -1.67 | 54.92±8.77 | +6.02 | 0 | 0 |

## 下一步判断

如果目标是论文表 strict 复现，那么当前证据说明不能继续使用“历史结果里能找到接近值”作为复现完成依据；必须继续做针对 MIA 口径的定位。当前最可疑的是论文 MIA 目标值与本仓库固定的 exact `val` MIA 口径不一致，或者历史 candidate search 曾经混入非 clean / 非 strict 路径。
