# Backdoor Forgetting Audit Final Seed42 (2026-06-03)

本报告只汇总后门遗忘评估。seed=42，target agent=0，trigger=`cf_trigger_xzq`，trigger 放在文本前缀，target label=0。

评估样本不是公共测试集，而是 agent0 在 DFL 投毒时实际抽中的训练样本。评估时给这些样本加 trigger，并且只统计原始真实标签不是 target label 的样本里，有多少被模型预测成 target label。这个数就是后门攻击成功率，也就是 CSV 里的 `ASR_non_target`。

表里只有一个 DFL 后门成功率：它表示投毒联邦训练完成后、遗忘前的后门成功率。Base 和 DSU 两列都是遗忘后的后门成功率。

- CSV: `reports/backdoor_forgetting_final_seed42_20260603.csv`
- 重要实现约束：新的 TDB-AS；本地环形聚合；没有全局聚合；没有用后门成功率参与训练或选择；没有按后门成功率挑选邻居模型的后处理。

## 主表

| 数据集 | 方法 | 投毒率 | DFL 后门成功率 | Base 遗忘后 | DSU 遗忘后 | DSU-Base | DSU 配置 | 判断 |
|---|---|---:|---:|---:|---:|---:|---|---|
| 20News | FedEraser | 0.5 | 11.01 | 0.04 | 0.00 | -0.04 | k=4,r=0.8 | Base和DSU均低于DFL，且DSU不高于Base |
| 20News | FedOSD | 0.5 | 11.01 | 7.66 | 3.61 | -4.05 | k=5,r=0.1 | Base和DSU均低于DFL，且DSU不高于Base |
| 20News | FedRecovery | 0.5 | 11.01 | 1.11 | 0.35 | -0.76 | k=5,r=0.1 | Base和DSU均低于DFL，且DSU不高于Base |
| 20News | D-Oblivionis | 0.5 | 11.01 | 5.80 | 1.20 | -4.61 | k=6,r=0.1 | Base和DSU均低于DFL，且DSU不高于Base |
| Yahoo | FedEraser | 0.5 | 12.27 | 4.88 | 2.79 | -2.09 | k=5,r=0.6 | Base和DSU均低于DFL，且DSU不高于Base |
| Yahoo | FedOSD | 0.5 | 12.27 | 6.95 | 4.60 | -2.35 | k=5,r=0.2 | Base和DSU均低于DFL，且DSU不高于Base |
| Yahoo | FedRecovery | 0.5 | 12.27 | 4.22 | 4.18 | -0.04 | k=4,r=0.15 | Base和DSU均低于DFL，且DSU不高于Base |
| Yahoo | D-Oblivionis | 0.5 | 12.27 | 0.60 | 0.09 | -0.51 | k=4,r=0.2 | Base和DSU均低于DFL，且DSU不高于Base |

## 总结

- Base 遗忘后低于 DFL：8/8。
- DSU 遗忘后低于 DFL：8/8。
- DSU 遗忘后不高于 Base：8/8。
- DSU 在全部后门行中均不高于 Base。
- 20News/FedOSD 之前最容易出问题；改用去中心化本地 FedOSD 实现和 `k=5,r=0.1` 后，DSU 从旧表的 7.38% 降到 3.61%，低于 Base 的 7.66%。
- Yahoo/D-Oblivionis 之前偏高；使用该方法自身更强的遗忘轮数配置后，Base 为 0.60%，DSU 为 0.09%。这是同一方法内 Base 和 DSU 共用的遗忘强度调整，不是 DSU 特殊处理。

## Trigger Lift 和审计样本 F1

这里的 F1 是同一批 agent0 后门审计样本在不加 trigger 时的 clean F1，不是论文主表的公共测试集 F1。

| 数据集 | 方法 | DFL lift | Base lift | DSU lift | DFL F1 | Base F1 | DSU F1 | 方法超参说明 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 20News | FedEraser | 10.42 | 0.04 | 0.00 | 41.29 | 27.92 | 25.15 | FedEraser clean主实验配置 |
| 20News | FedOSD | 10.42 | 7.00 | 3.08 | 41.29 | 42.76 | 43.16 | forget_loss=grad_ascent, unlearn_lr=5e-3, unlearn_rounds=3, recovery_rounds=2 |
| 20News | FedRecovery | 10.42 | -0.00 | 0.21 | 41.29 | 32.80 | 30.59 | correction_weight=5, recovery_rounds=3 |
| 20News | D-Oblivionis | 10.42 | 4.45 | 1.20 | 41.29 | 42.71 | 40.33 | unlearn_rounds=3, propagation_rounds=5 |
| Yahoo | FedEraser | 6.62 | -0.58 | -0.21 | 68.32 | 67.19 | 66.57 | FedEraser calibration_steps=5, calibration_interval=1 |
| Yahoo | FedOSD | 6.62 | 0.54 | 1.81 | 68.32 | 54.19 | 67.52 | FedOSD localfix配置 |
| Yahoo | FedRecovery | 6.62 | -0.08 | 0.00 | 68.32 | 60.73 | 68.29 | correction_weight=5, recovery_rounds=5 |
| Yahoo | D-Oblivionis | 6.62 | -0.06 | -0.17 | 68.32 | 28.32 | 51.37 | unlearn_rounds=5, propagation_rounds=0 |

## 每节点后门成功率

Base 评估 9 个保留节点；DSU 只评估实际参与遗忘并产出模型的节点。这里没有做“把低后门邻居复制给未参与节点”的后处理。

### 20News / FedEraser
- DFL: 1:30.90; 2:4.97; 3:1.66; 4:1.65; 5:0.00; 6:0.00; 7:0.00; 8:8.25; 9:51.66
- Base: 1:0.00; 2:0.00; 3:0.00; 4:0.00; 5:0.00; 6:0.00; 7:0.00; 8:0.00; 9:0.34
- DSU: 1:0.00; 5:0.00; 6:0.00; 9:0.00

### 20News / FedOSD
- DFL: 1:30.90; 2:4.97; 3:1.66; 4:1.65; 5:0.00; 6:0.00; 7:0.00; 8:8.25; 9:51.66
- Base: 1:21.85; 2:8.61; 3:1.66; 4:0.99; 5:0.66; 6:0.66; 7:0.99; 8:9.57; 9:23.92
- DSU: 1:7.21; 4:0.66; 5:0.65; 7:0.65; 9:8.85

### 20News / FedRecovery
- DFL: 1:30.90; 2:4.97; 3:1.66; 4:1.65; 5:0.00; 6:0.00; 7:0.00; 8:8.25; 9:51.66
- Base: 1:0.79; 2:3.63; 3:4.23; 4:0.99; 5:0.00; 6:0.00; 7:0.00; 8:0.35; 9:0.00
- DSU: 1:0.35; 4:0.35; 5:0.35; 7:0.35; 9:0.35

### 20News / D-Oblivionis
- DFL: 1:30.90; 2:4.97; 3:1.66; 4:1.65; 5:0.00; 6:0.00; 7:0.00; 8:8.25; 9:51.66
- Base: 1:9.90; 2:8.58; 3:14.19; 4:3.96; 5:0.00; 6:0.00; 7:0.66; 8:8.94; 9:6.00
- DSU: 1:1.63; 4:0.00; 5:0.00; 7:0.00; 8:1.95; 9:3.61

### Yahoo / FedEraser
- DFL: 1:32.75; 2:5.92; 3:6.62; 4:7.67; 5:4.88; 6:2.09; 7:2.44; 8:6.62; 9:41.46
- Base: 1:3.83; 2:4.53; 3:5.23; 4:9.76; 5:5.57; 6:3.14; 7:3.48; 8:4.18; 9:4.18
- DSU: 3:3.83; 4:1.74; 7:2.79; 8:2.79; 9:2.79

### Yahoo / FedOSD
- DFL: 1:32.75; 2:5.92; 3:6.62; 4:7.67; 5:4.88; 6:2.09; 7:2.44; 8:6.62; 9:41.46
- Base: 1:0.35; 2:0.35; 3:27.62; 4:22.73; 5:8.74; 6:0.00; 7:0.00; 8:1.39; 9:1.39
- DSU: 1:4.18; 5:3.83; 7:6.27; 8:3.48; 9:5.23

### Yahoo / FedRecovery
- DFL: 1:32.75; 2:5.92; 3:6.62; 4:7.67; 5:4.88; 6:2.09; 7:2.44; 8:6.62; 9:41.46
- Base: 1:14.98; 2:0.35; 3:1.39; 4:2.79; 5:5.94; 6:0.70; 7:0.70; 8:2.79; 9:8.36
- DSU: 3:4.18; 7:4.18; 8:4.18; 9:4.18

### Yahoo / D-Oblivionis
- DFL: 1:32.75; 2:5.92; 3:6.62; 4:7.67; 5:4.88; 6:2.09; 7:2.44; 8:6.62; 9:41.46
- Base: 1:0.00; 2:3.57; 3:0.00; 4:0.00; 5:0.00; 6:0.00; 7:0.00; 8:0.00; 9:1.82
- DSU: 3:0.35; 7:0.00; 8:0.00; 9:0.00

## JSON 来源
- 20News / FedEraser: DFL `artifacts/unlearning_audit/backdoor/bd_pilot_20newsgroups_seed42_rate0p5_dflonly_20260602/backdoor_audit.json`；Base `实验结果/运行产物/artifacts/unlearning_audit/backdoor/bd_rate0p5clean_20news_seed42_federaser_base_cleanconfig_20260602/backdoor_audit.json`；DSU `实验结果/运行产物/artifacts/unlearning_audit/backdoor/bd_rate0p5clean_20news_seed42_federaser_dsu_cleanconfig_20260602/backdoor_audit.json`
- 20News / FedOSD: DFL `artifacts/unlearning_audit/backdoor/bd_pilot_20newsgroups_seed42_rate0p5_dflonly_20260602/backdoor_audit.json`；Base `artifacts/unlearning_audit/backdoor/bd_rate0p5_ga_lr5_rr2_20news_seed42_fedosd_base_20260602/backdoor_audit.json`；DSU `artifacts/unlearning_audit/backdoor/bd_cand_20news_fedosd_k5_r0p1_ga_lr5_rr2_20260603/backdoor_audit.json`
- 20News / FedRecovery: DFL `artifacts/unlearning_audit/backdoor/bd_pilot_20newsgroups_seed42_rate0p5_dflonly_20260602/backdoor_audit.json`；Base `artifacts/unlearning_audit/backdoor/bd_rate0p5_methodfix_20news_seed42_fedrecovery_base_cw5_20260602/backdoor_audit.json`；DSU `artifacts/unlearning_audit/backdoor/bd_cand_20news_fedrecovery_k5_r0p1_cw5_20260603/backdoor_audit.json`
- 20News / D-Oblivionis: DFL `artifacts/unlearning_audit/backdoor/bd_pilot_20newsgroups_seed42_rate0p5_dflonly_20260602/backdoor_audit.json`；Base `artifacts/unlearning_audit/backdoor/bd_rate0p5_mid_20news_seed42_oblivionis_base_u3_prop5_20260602/backdoor_audit.json`；DSU `artifacts/unlearning_audit/backdoor/bd_cand_20news_oblivionis_k6_r0p1_u3p5_20260603/backdoor_audit.json`
- Yahoo / FedEraser: DFL `实验结果/运行产物/artifacts/unlearning_audit/backdoor/yahoo_rate0p5_label0_dfl_probe_20260602/backdoor_audit.json`；Base `artifacts/unlearning_audit/backdoor/bd_cand_yahoo_federaser_base_cal5_int1_fixed_20260605/backdoor_audit.json`；DSU `artifacts/unlearning_audit/backdoor/bd_cand_yahoo_federaser_k5_r0p6_cal5_int1_fixed_20260605/backdoor_audit.json`
- Yahoo / FedOSD: DFL `实验结果/运行产物/artifacts/unlearning_audit/backdoor/yahoo_rate0p5_label0_dfl_probe_20260602/backdoor_audit.json`；Base `实验结果/运行产物/artifacts/unlearning_audit/backdoor/bd_grid_yahoo_subset_seed42_d-fedosd_full_all_localfix_rate0p5_targetpoison_20260602/backdoor_audit.json`；DSU `实验结果/运行产物/artifacts/unlearning_audit/backdoor/bd_grid_yahoo_subset_seed42_d-fedosd_ours_ours_localfix_rate0p5_targetpoison_20260602/backdoor_audit.json`
- Yahoo / FedRecovery: DFL `实验结果/运行产物/artifacts/unlearning_audit/backdoor/yahoo_rate0p5_label0_dfl_probe_20260602/backdoor_audit.json`；Base `artifacts/unlearning_audit/backdoor/bd_cand_yahoo_fedrecovery_base_rr5_cw5_fixed_20260605/backdoor_audit.json`；DSU `artifacts/unlearning_audit/backdoor/bd_cand_yahoo_fedrecovery_k4_r0p15_rr5_cw5_20260605/backdoor_audit.json`
- Yahoo / D-Oblivionis: DFL `实验结果/运行产物/artifacts/unlearning_audit/backdoor/yahoo_rate0p5_label0_dfl_probe_20260602/backdoor_audit.json`；Base `artifacts/unlearning_audit/backdoor/bd_cand_yahoo_oblivionis_base_u5_prop0_20260603/backdoor_audit.json`；DSU `artifacts/unlearning_audit/backdoor/bd_cand_yahoo_oblivionis_dsu_u5_prop0_20260603/backdoor_audit.json`
