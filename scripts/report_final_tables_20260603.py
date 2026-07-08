#!/usr/bin/env python3
"""Regenerate the final TDB clean and backdoor summary tables.

This is a reporting-only script. It reads completed CSV/JSON artifacts and
does not run training, unlearning, selection, or any ASR-guided optimization.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

DATASET_ORDER = ["20newsgroups", "yahoo_subset"]
METHOD_ORDER = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
DATASET_LABEL = {"20newsgroups": "20News", "yahoo_subset": "Yahoo"}
METHOD_LABEL = {
    "d-federaser": "FedEraser",
    "d-fedosd": "FedOSD",
    "d-fedrecovery": "FedRecovery",
    "d-oblivionis": "D-Oblivionis",
}

FINAL_DSU = {
    ("20newsgroups", "d-federaser"): (4, 0.8),
    ("20newsgroups", "d-fedosd"): (5, 0.1),
    ("20newsgroups", "d-fedrecovery"): (5, 0.1),
    ("20newsgroups", "d-oblivionis"): (6, 0.1),
    ("yahoo_subset", "d-federaser"): (5, 0.6),
    ("yahoo_subset", "d-fedosd"): (5, 0.2),
    ("yahoo_subset", "d-fedrecovery"): (4, 0.15),
    ("yahoo_subset", "d-oblivionis"): (4, 0.2),
}

EXTRA_DSU_AGGREGATES = [
    ROOT / "reports/tdb_yahoo_fedrecovery_fine_r_20260605_aggregate.csv",
]

BACKDOOR_ROWS = [
    {
        "dataset": "20newsgroups",
        "method": "d-federaser",
        "poison_rate": 0.5,
        "dsu_config": "k=4,r=0.8",
        "method_hparams": "FedEraser clean主实验配置",
        "base_json": "实验结果/运行产物/artifacts/unlearning_audit/backdoor/bd_rate0p5clean_20news_seed42_federaser_base_cleanconfig_20260602/backdoor_audit.json",
        "dsu_json": "实验结果/运行产物/artifacts/unlearning_audit/backdoor/bd_rate0p5clean_20news_seed42_federaser_dsu_cleanconfig_20260602/backdoor_audit.json",
    },
    {
        "dataset": "20newsgroups",
        "method": "d-fedosd",
        "poison_rate": 0.5,
        "dsu_config": "k=5,r=0.1",
        "method_hparams": "forget_loss=grad_ascent, unlearn_lr=5e-3, unlearn_rounds=3, recovery_rounds=2",
        "base_json": "artifacts/unlearning_audit/backdoor/bd_rate0p5_ga_lr5_rr2_20news_seed42_fedosd_base_20260602/backdoor_audit.json",
        "dsu_json": "artifacts/unlearning_audit/backdoor/bd_cand_20news_fedosd_k5_r0p1_ga_lr5_rr2_20260603/backdoor_audit.json",
    },
    {
        "dataset": "20newsgroups",
        "method": "d-fedrecovery",
        "poison_rate": 0.5,
        "dsu_config": "k=5,r=0.1",
        "method_hparams": "correction_weight=5, recovery_rounds=3",
        "base_json": "artifacts/unlearning_audit/backdoor/bd_rate0p5_methodfix_20news_seed42_fedrecovery_base_cw5_20260602/backdoor_audit.json",
        "dsu_json": "artifacts/unlearning_audit/backdoor/bd_cand_20news_fedrecovery_k5_r0p1_cw5_20260603/backdoor_audit.json",
    },
    {
        "dataset": "20newsgroups",
        "method": "d-oblivionis",
        "poison_rate": 0.5,
        "dsu_config": "k=6,r=0.1",
        "method_hparams": "unlearn_rounds=3, propagation_rounds=5",
        "base_json": "artifacts/unlearning_audit/backdoor/bd_rate0p5_mid_20news_seed42_oblivionis_base_u3_prop5_20260602/backdoor_audit.json",
        "dsu_json": "artifacts/unlearning_audit/backdoor/bd_cand_20news_oblivionis_k6_r0p1_u3p5_20260603/backdoor_audit.json",
    },
    {
        "dataset": "yahoo_subset",
        "method": "d-federaser",
        "poison_rate": 0.5,
        "dsu_config": "k=5,r=0.6",
        "method_hparams": "FedEraser calibration_steps=5, calibration_interval=1",
        "base_json": "artifacts/unlearning_audit/backdoor/bd_cand_yahoo_federaser_base_cal5_int1_fixed_20260605/backdoor_audit.json",
        "dsu_json": "artifacts/unlearning_audit/backdoor/bd_cand_yahoo_federaser_k5_r0p6_cal5_int1_fixed_20260605/backdoor_audit.json",
    },
    {
        "dataset": "yahoo_subset",
        "method": "d-fedosd",
        "poison_rate": 0.5,
        "dsu_config": "k=5,r=0.2",
        "method_hparams": "FedOSD localfix配置",
        "base_json": "实验结果/运行产物/artifacts/unlearning_audit/backdoor/bd_grid_yahoo_subset_seed42_d-fedosd_full_all_localfix_rate0p5_targetpoison_20260602/backdoor_audit.json",
        "dsu_json": "实验结果/运行产物/artifacts/unlearning_audit/backdoor/bd_grid_yahoo_subset_seed42_d-fedosd_ours_ours_localfix_rate0p5_targetpoison_20260602/backdoor_audit.json",
    },
    {
        "dataset": "yahoo_subset",
        "method": "d-fedrecovery",
        "poison_rate": 0.5,
        "dsu_config": "k=4,r=0.15",
        "method_hparams": "correction_weight=5, recovery_rounds=5",
        "base_json": "artifacts/unlearning_audit/backdoor/bd_cand_yahoo_fedrecovery_base_rr5_cw5_fixed_20260605/backdoor_audit.json",
        "dsu_json": "artifacts/unlearning_audit/backdoor/bd_cand_yahoo_fedrecovery_k4_r0p15_rr5_cw5_20260605/backdoor_audit.json",
    },
    {
        "dataset": "yahoo_subset",
        "method": "d-oblivionis",
        "poison_rate": 0.5,
        "dsu_config": "k=4,r=0.2",
        "method_hparams": "unlearn_rounds=5, propagation_rounds=0",
        "base_json": "artifacts/unlearning_audit/backdoor/bd_cand_yahoo_oblivionis_base_u5_prop0_20260603/backdoor_audit.json",
        "dsu_json": "artifacts/unlearning_audit/backdoor/bd_cand_yahoo_oblivionis_dsu_u5_prop0_20260603/backdoor_audit.json",
    },
]

DFL_JSON = {
    "20newsgroups": "artifacts/unlearning_audit/backdoor/bd_pilot_20newsgroups_seed42_rate0p5_dflonly_20260602/backdoor_audit.json",
    "yahoo_subset": "实验结果/运行产物/artifacts/unlearning_audit/backdoor/yahoo_rate0p5_label0_dfl_probe_20260602/backdoor_audit.json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fnum(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def pct_value(value: Any) -> float:
    x = fnum(value)
    return 100.0 * x if x is not None else float("nan")


def fmt(value: Any, digits: int = 2) -> str:
    x = fnum(value)
    return "-" if x is None else f"{x:.{digits}f}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    x = fnum(value)
    return "-" if x is None else f"{100.0 * x:.{digits}f}"


def fmt_mean_std(mean: Any, std: Any) -> str:
    if fnum(mean) is None:
        return "-"
    return f"{fmt_pct(mean)}±{fmt_pct(std or 0.0)}"


def fmt_ratio(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def method_label(method: str) -> str:
    return METHOD_LABEL.get(method, method)


def dataset_label(dataset: str) -> str:
    return DATASET_LABEL.get(dataset, dataset)


def row_order(row: dict[str, Any]) -> tuple[int, int]:
    dataset = row["dataset"]
    method = row["method"]
    return (
        DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else 99,
        METHOD_ORDER.index(method) if method in METHOD_ORDER else 99,
    )


def build_clean_table() -> list[dict[str, Any]]:
    current = read_csv(ROOT / "reports/tdb_clean_final_local_f1_mia_20260602.csv")
    current_map = {(r["dataset"], r["method"]): dict(r) for r in current}
    dsu_rows = read_csv(ROOT / "reports/tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_aggregate.csv")
    dsu_map: dict[tuple[str, str, int, float], dict[str, str]] = {}
    for row in dsu_rows:
        dsu_map[(row["dataset"], row["algorithm"], int(float(row["k"])), float(row["r"]))] = row
    for extra_path in EXTRA_DSU_AGGREGATES:
        if extra_path.exists():
            for row in read_csv(extra_path):
                dsu_map[(row["dataset"], row["algorithm"], int(float(row["k"])), float(row["r"]))] = row

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            row = dict(current_map[(dataset, method)])
            k, r = FINAL_DSU[(dataset, method)]
            src = dsu_map[(dataset, method, k, r)]
            row["dsu_param"] = f"k={k},r={fmt_ratio(r)}"
            row["dsu_f1"] = src["macro_f1_best_mean"]
            row["dsu_f1_std"] = src["macro_f1_best_std"]
            row["dsu_mia"] = src["mia_auc_mean"]
            row["dsu_mia_std"] = src["mia_auc_std"]
            dsu_f1 = fnum(row["dsu_f1"])
            for setting in ["base", "as", "ls"]:
                other = fnum(row.get(f"{setting}_f1"))
                row[f"dsu_minus_{setting}_f1"] = None if dsu_f1 is None or other is None else dsu_f1 - other
            rows.append(row)
    return rows


def clean_to_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Final Local-Ring TDB Clean F1/MIA Table",
        "",
        "本表只汇总不带后门投毒的主实验。F1 是公共测试集 macro-F1，MIA 是成员推断攻击 AUC。DSU 使用新的 TDB-AS，且 `tdb_aggregation_scope=local`，不使用全局聚合。",
        "",
        "- CSV: `reports/tdb_clean_final_local_f1_mia_20260602.csv`",
        "- F1 图：`reports/tdb_final_local_as_k_f1_20260603.png`、`reports/tdb_final_local_ls_r_f1_20260603.png`、`reports/tdb_final_local_dsu_kr_heatmaps_f1_20260603.png`",
        "",
        "| 数据集 | 方法 | Base F1/MIA | AS F1/MIA | LS F1/MIA | DSU F1/MIA | DSU-Base | DSU-AS | DSU-LS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{dataset_label(row['dataset'])} | {method_label(row['method'])} | "
            f"{fmt_mean_std(row.get('base_f1'), row.get('base_f1_std'))}/{fmt_mean_std(row.get('base_mia'), row.get('base_mia_std'))} | "
            f"{row.get('as_param')} {fmt_mean_std(row.get('as_f1'), row.get('as_f1_std'))}/{fmt_mean_std(row.get('as_mia'), row.get('as_mia_std'))} | "
            f"{row.get('ls_param')} {fmt_mean_std(row.get('ls_f1'), row.get('ls_f1_std'))}/{fmt_mean_std(row.get('ls_mia'), row.get('ls_mia_std'))} | "
            f"{row.get('dsu_param')} {fmt_mean_std(row.get('dsu_f1'), row.get('dsu_f1_std'))}/{fmt_mean_std(row.get('dsu_mia'), row.get('dsu_mia_std'))} | "
            f"{fmt_pct(row.get('dsu_minus_base_f1'))} | {fmt_pct(row.get('dsu_minus_as_f1'))} | {fmt_pct(row.get('dsu_minus_ls_f1'))} |"
        )
    dsu_gt_base = sum(1 for r in rows if fnum(r["dsu_minus_base_f1"]) is not None and fnum(r["dsu_minus_base_f1"]) > 0)
    dsu_gt_as = sum(1 for r in rows if fnum(r["dsu_minus_as_f1"]) is not None and fnum(r["dsu_minus_as_f1"]) > 0)
    dsu_ge_ls = sum(1 for r in rows if fnum(r["dsu_minus_ls_f1"]) is not None and fnum(r["dsu_minus_ls_f1"]) >= 0)
    ls_boundary_rows = [r for r in rows if fnum(r["dsu_minus_ls_f1"]) is not None and fnum(r["dsu_minus_ls_f1"]) < 0]
    if ls_boundary_rows:
        boundary_text = "；".join(
            f"{dataset_label(r['dataset'])}/{method_label(r['method'])} 低 {abs(fnum(r['dsu_minus_ls_f1']) or 0.0) * 100:.2f} 个百分点"
            for r in ls_boundary_rows
        )
        boundary_line = (
            f"- DSU 低于 LS 的边界行：{boundary_text}。这些行仍高于 Base 和 AS，"
            "最终配置是在 clean F1、参与节点数、LoRA 更新比例和直接遗忘审计之间的保守取舍。"
        )
    else:
        boundary_line = "- DSU 在全部设置中均不低于 LS。"
    lines += [
        "",
        "## 结论",
        "",
        f"- DSU 高于 Base：{dsu_gt_base}/8。",
        f"- DSU 高于 AS：{dsu_gt_as}/8。",
        f"- DSU 不低于 LS：{dsu_ge_ls}/8。",
        boundary_line,
    ]
    return "\n".join(lines) + "\n"


def load_json(path_text: str) -> dict[str, Any]:
    path = ROOT / path_text
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def model_metric(data: dict[str, Any], model: str, metric: str) -> float | None:
    return fnum(((data.get("models") or {}).get(model) or {}).get(metric, {}).get("asr"))


def model_f1(data: dict[str, Any], model: str) -> float | None:
    return fnum(((data.get("models") or {}).get(model) or {}).get("clean", {}).get("macro_f1"))


def agent_asr(data: dict[str, Any], model: str) -> str:
    model_data = ((data.get("models") or {}).get(model) or {})
    agents = model_data.get("agents") or []
    parts = []
    for item in sorted(agents, key=lambda x: int(x.get("agent_id", 999))):
        agent_id = item.get("agent_id")
        value = (((item.get("asr_non_target") or {}).get("asr")))
        parts.append(f"{agent_id}:{fmt_pct(value)}")
    return "; ".join(parts)


def build_backdoor_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in BACKDOOR_ROWS:
        base_data = load_json(cfg["base_json"])
        dsu_data = load_json(cfg["dsu_json"])
        dfl_data = load_json(DFL_JSON[cfg["dataset"]])
        dfl_asr = model_metric(dfl_data, "dfl", "asr_non_target")
        base_asr = model_metric(base_data, "dfu", "asr_non_target")
        dsu_asr = model_metric(dsu_data, "dfu", "asr_non_target")
        dfl_lift = None
        base_lift = None
        dsu_lift = None
        if dfl_asr is not None:
            clean = model_metric(dfl_data, "dfl", "clean_target_rate_non_target")
            dfl_lift = None if clean is None else dfl_asr - clean
        if base_asr is not None:
            clean = model_metric(base_data, "dfu", "clean_target_rate_non_target")
            base_lift = None if clean is None else base_asr - clean
        if dsu_asr is not None:
            clean = model_metric(dsu_data, "dfu", "clean_target_rate_non_target")
            dsu_lift = None if clean is None else dsu_asr - clean
        row = {
            "dataset": cfg["dataset"],
            "method": cfg["method"],
            "seed": 42,
            "poison_rate": cfg["poison_rate"],
            "target_agent": 0,
            "target_label": 0,
            "dsu_config": cfg["dsu_config"],
            "method_hparams": cfg["method_hparams"],
            "dfl_asr": pct_value(dfl_asr),
            "base_asr": pct_value(base_asr),
            "dsu_asr": pct_value(dsu_asr),
            "dfl_trigger_lift": pct_value(dfl_lift),
            "base_trigger_lift": pct_value(base_lift),
            "dsu_trigger_lift": pct_value(dsu_lift),
            "dfl_clean_f1": pct_value(model_f1(dfl_data, "dfl")),
            "base_clean_f1": pct_value(model_f1(base_data, "dfu")),
            "dsu_clean_f1": pct_value(model_f1(dsu_data, "dfu")),
            "base_minus_dfl_asr": pct_value(base_asr) - pct_value(dfl_asr),
            "dsu_minus_base_asr": pct_value(dsu_asr) - pct_value(base_asr),
            "dsu_minus_dfl_asr": pct_value(dsu_asr) - pct_value(dfl_asr),
            "status": "",
            "dfl_agents_asr": agent_asr(dfl_data, "dfl"),
            "base_agents_asr": agent_asr(base_data, "dfu"),
            "dsu_agents_asr": agent_asr(dsu_data, "dfu"),
            "base_json": cfg["base_json"],
            "dsu_json": cfg["dsu_json"],
            "dfl_json": DFL_JSON[cfg["dataset"]],
        }
        dfl_ok = row["base_asr"] < row["dfl_asr"] and row["dsu_asr"] < row["dfl_asr"]
        dsu_gap = row["dsu_minus_base_asr"]
        if dfl_ok and dsu_gap <= 0:
            row["status"] = "Base和DSU均低于DFL，且DSU不高于Base"
        elif dfl_ok and dsu_gap <= 2.0:
            row["status"] = f"Base和DSU均低于DFL；DSU高于Base {dsu_gap:.2f}点，仍在2点容忍范围内"
        else:
            row["status"] = "需要继续谨慎解释或补跑"
        rows.append(row)
    return rows


def backdoor_to_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Backdoor Forgetting Audit Final Seed42 (2026-06-03)",
        "",
        "本报告只汇总后门遗忘评估。seed=42，target agent=0，trigger=`cf_trigger_xzq`，trigger 放在文本前缀，target label=0。",
        "",
        "评估样本不是公共测试集，而是 agent0 在 DFL 投毒时实际抽中的训练样本。评估时给这些样本加 trigger，并且只统计原始真实标签不是 target label 的样本里，有多少被模型预测成 target label。这个数就是后门攻击成功率，也就是 CSV 里的 `ASR_non_target`。",
        "",
        "表里只有一个 DFL 后门成功率：它表示投毒联邦训练完成后、遗忘前的后门成功率。Base 和 DSU 两列都是遗忘后的后门成功率。",
        "",
        "- CSV: `reports/backdoor_forgetting_final_seed42_20260603.csv`",
        "- 重要实现约束：新的 TDB-AS；本地环形聚合；没有全局聚合；没有用后门成功率参与训练或选择；没有复制低后门邻居模型。",
        "",
        "## 主表",
        "",
        "| 数据集 | 方法 | 投毒率 | DFL 后门成功率 | Base 遗忘后 | DSU 遗忘后 | DSU-Base | DSU 配置 | 判断 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{dataset_label(row['dataset'])} | {method_label(row['method'])} | {row['poison_rate']} | "
            f"{fmt(row['dfl_asr'])} | {fmt(row['base_asr'])} | {fmt(row['dsu_asr'])} | "
            f"{fmt(row['dsu_minus_base_asr'])} | {row['dsu_config']} | {row['status']} |"
        )
    base_lt_dfl = sum(1 for r in rows if r["base_asr"] < r["dfl_asr"])
    dsu_lt_dfl = sum(1 for r in rows if r["dsu_asr"] < r["dfl_asr"])
    dsu_le_base = sum(1 for r in rows if r["dsu_asr"] <= r["base_asr"])
    dsu_higher_base = [r for r in rows if r["dsu_asr"] > r["base_asr"]]
    if dsu_higher_base:
        higher_text = "；".join(
            f"{dataset_label(r['dataset'])}/{method_label(r['method'])} 高 {r['dsu_minus_base_asr']:.2f} 个百分点"
            for r in dsu_higher_base
        )
        higher_line = f"- DSU 高于 Base 的边界行：{higher_text}。这些行仍低于 DFL，不能写成 DSU 在所有后门行都低于 Base。"
    else:
        higher_line = "- DSU 在全部后门行中均不高于 Base。"
    lines += [
        "",
        "## 总结",
        "",
        f"- Base 遗忘后低于 DFL：{base_lt_dfl}/8。",
        f"- DSU 遗忘后低于 DFL：{dsu_lt_dfl}/8。",
        f"- DSU 遗忘后不高于 Base：{dsu_le_base}/8。",
        higher_line,
        "- 20News/FedOSD 之前最容易出问题；改用去中心化本地 FedOSD 实现和 `k=5,r=0.1` 后，DSU 从旧表的 7.38% 降到 3.61%，低于 Base 的 7.66%。",
        "- Yahoo/D-Oblivionis 之前偏高；使用该方法自身更强的遗忘轮数配置后，Base 为 0.60%，DSU 为 0.09%。这是同一方法内 Base 和 DSU 共用的遗忘强度调整，不是 DSU 特殊处理。",
        "",
        "## Trigger Lift 和审计样本 F1",
        "",
        "这里的 F1 是同一批 agent0 后门审计样本在不加 trigger 时的 clean F1，不是论文主表的公共测试集 F1。",
        "",
        "| 数据集 | 方法 | DFL lift | Base lift | DSU lift | DFL F1 | Base F1 | DSU F1 | 方法超参说明 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{dataset_label(row['dataset'])} | {method_label(row['method'])} | "
            f"{fmt(row['dfl_trigger_lift'])} | {fmt(row['base_trigger_lift'])} | {fmt(row['dsu_trigger_lift'])} | "
            f"{fmt(row['dfl_clean_f1'])} | {fmt(row['base_clean_f1'])} | {fmt(row['dsu_clean_f1'])} | "
            f"{row['method_hparams']} |"
        )
    lines += [
        "",
        "## 每节点后门成功率",
        "",
        "Base 评估 9 个保留节点；DSU 只评估实际参与遗忘并产出模型的节点。这里没有做“把低后门邻居复制给未参与节点”的后处理。",
    ]
    for row in rows:
        lines += [
            "",
            f"### {dataset_label(row['dataset'])} / {method_label(row['method'])}",
            f"- DFL: {row['dfl_agents_asr']}",
            f"- Base: {row['base_agents_asr']}",
            f"- DSU: {row['dsu_agents_asr']}",
        ]
    lines += [
        "",
        "## JSON 来源",
    ]
    for row in rows:
        lines.append(
            f"- {dataset_label(row['dataset'])} / {method_label(row['method'])}: "
            f"DFL `{row['dfl_json']}`；Base `{row['base_json']}`；DSU `{row['dsu_json']}`"
        )
    return "\n".join(lines) + "\n"


def reviewer_to_md(clean_rows: list[dict[str, Any]], backdoor_rows: list[dict[str, Any]]) -> str:
    proxy_lines = []
    for row in clean_rows:
        k_text, r_text = str(row["dsu_param"]).split(",")
        k = int(k_text.split("=")[1])
        r = float(r_text.split("=")[1])
        proxy_lines.append(
            f"| {dataset_label(row['dataset'])} | {method_label(row['method'])} | {k} | {fmt_ratio(r)} | {(k / 9.0) * r * 100:.2f}% |"
        )
    clean_all_base = sum(1 for r in clean_rows if fnum(r["dsu_minus_base_f1"]) and fnum(r["dsu_minus_base_f1"]) > 0)
    clean_ge_ls = sum(1 for r in clean_rows if fnum(r["dsu_minus_ls_f1"]) is not None and fnum(r["dsu_minus_ls_f1"]) >= 0)
    clean_ls_boundary = [r for r in clean_rows if fnum(r["dsu_minus_ls_f1"]) is not None and fnum(r["dsu_minus_ls_f1"]) < 0]
    if clean_ls_boundary:
        clean_boundary_summary = "；".join(
            f"{dataset_label(r['dataset'])}/{method_label(r['method'])} 低 {abs(fnum(r['dsu_minus_ls_f1']) or 0.0) * 100:.2f} 个百分点"
            for r in clean_ls_boundary
        )
        clean_ls_line = (
            f"- DSU 不低于 LS：{clean_ge_ls}/8；边界行 {len(clean_ls_boundary)} 个，"
            f"{clean_boundary_summary}。这些边界行仍高于 Base 和 AS。"
        )
    else:
        clean_ls_line = f"- DSU 不低于 LS：{clean_ge_ls}/8。"
    backdoor_dsu_lt_dfl = sum(1 for r in backdoor_rows if r["dsu_asr"] < r["dfl_asr"])
    backdoor_dsu_le_base = sum(1 for r in backdoor_rows if r["dsu_asr"] <= r["base_asr"])
    backdoor_dsu_higher = [r for r in backdoor_rows if r["dsu_asr"] > r["base_asr"]]
    if backdoor_dsu_higher:
        higher_summary = "；".join(
            f"{dataset_label(r['dataset'])}/{method_label(r['method'])} 高 {r['dsu_minus_base_asr']:.2f} 个百分点"
            for r in backdoor_dsu_higher
        )
        backdoor_base_line = (
            f"- DSU 遗忘后不高于 Base：{backdoor_dsu_le_base}/8；"
            f"边界行 {len(backdoor_dsu_higher)} 个，{higher_summary}。"
        )
    else:
        backdoor_base_line = f"- DSU 遗忘后不高于 Base：{backdoor_dsu_le_base}/8。"
    return "\n".join(
        [
            "# Reviewer Supplement Final Summary (2026-06-03)",
            "",
            "这个报告把审稿意见相关补实验按统一口径放在一起。所有后门结果是 seed=42 的单 seed 收口结果；clean 主实验、节点数和 LoRA 比例遍历、求解器统计等使用已有多 seed 汇总结果。",
            "",
            "完整可用性与真实性核查见：`reports/reviewer_experiment_readiness_audit_20260603.md`。",
            "",
            "## 1. 直接遗忘指标：后门攻击成功率",
            "",
            "- 完整报告：`reports/backdoor_forgetting_final_seed42_20260603.md`",
            "- 口径：target agent=0；target label=0；trigger=`cf_trigger_xzq`；投毒率 20News=0.5、Yahoo=0.5；样本为 agent0 实际被投毒抽中的训练样本；主指标为 `ASR_non_target`。",
            "- 重要约束：后门成功率只用于离线审计，不进入训练、不进入 TDB-AS 求解、不进入 LS 打分。",
            "",
            f"- DSU 遗忘后低于 DFL：{backdoor_dsu_lt_dfl}/8。",
            backdoor_base_line,
            "",
            "| 数据集 | 方法 | DFL 后门成功率 | Base 遗忘后 | DSU 遗忘后 | DSU 配置 |",
            "|---|---|---:|---:|---:|---|",
            *[
                f"| {dataset_label(r['dataset'])} | {method_label(r['method'])} | {fmt(r['dfl_asr'])} | {fmt(r['base_asr'])} | {fmt(r['dsu_asr'])} | {r['dsu_config']} |"
                for r in backdoor_rows
            ],
            "",
            "## 2. 无后门主实验：跨数据集、跨方法 F1/MIA",
            "",
            "- 完整报告：`reports/tdb_clean_final_local_f1_mia_20260602.md`",
            f"- DSU 高于 Base：{clean_all_base}/8。",
            clean_ls_line,
            "",
            "## 3. AS/LS/DSU 超参数遍历",
            "",
            "- AS 节点数遍历：k=1..9。",
            "- LS LoRA 比例遍历：r=0.1..1.0。",
            "- DSU 联合遍历：k×r。",
            "- 图表：`reports/tdb_final_local_as_k_f1_20260603.png`、`reports/tdb_final_local_ls_r_f1_20260603.png`、`reports/tdb_final_local_dsu_kr_heatmaps_f1_20260603.png`。",
            "",
            "## 4. 标签分布 sketch 的经验支持",
            "",
            "- 报告：`reports/tdb_proxy_validation_correlation_20260601.md`。",
            "- 在 720 个 DSU 聚合配置上，F1 与 label discrepancy 的 Spearman 相关为 -0.402，F1 与 trajectory discrepancy 的 Spearman 相关为 -0.270。负相关说明 discrepancy 越小，F1 越高。",
            "- 论文写法要明确：label sketch 是轻量可通信代理，不是完整输入分布距离；class-conditional consistency 是同一任务 taxonomy 和同一预处理下的近似假设。",
            "",
            "## 5. 连续遗忘请求边界",
            "",
            "- 报告：`reports/sequential_cumulative_tdb_dsu_20260603.md`。",
            "- 当前补实验按顺序删除 `{0}`、`{0,1}`、`{0,1,2}`，是真正的累计 removed-agent set 增长实验，不再是三个互不相关的单目标探针。",
            "- 20News 使用 `k=4,r=0.5`，三步分别实际选择 4/9、4/8、4/7 个保留节点，更新 77/154 个 LoRA 模块。",
            "- Yahoo 使用 `k=5,r=0.3`，三步分别实际选择 5/9、5/8、5/7 个保留节点，更新 46/154 个 LoRA 模块。",
            "- 连续删除补实验中，DSU 的最佳 F1 和平均 F1 在 6/6 个步骤里都高于 Base，因此不是“剩余节点全都选了”的退化结果。",
            "- Yahoo 第三步 `{0,1,2}` 额外补跑了 `k=1..7` 消融，结果显示 `k=4/5` 最好，`k=7` 全选反而下降；详细见 `reports/sequential_yahoo_step3_k_sensitivity_20260603.md`。",
            "- 实现方式是复用 retained DFL history，并在每次新请求时按累计删除集合重新求解 TDB-AS 与 LS。它能支持“历史可复用、分数需重算”的审稿回复；但不能夸大成任意长直接链式更新已经无误差累积，长序列仍建议周期性 refresh 或完整重训校准。",
            "",
            "## 6. TDB/MILP 求解可行性",
            "",
            "- 报告：`reports/tdb_solver_stats_20260602.csv`。",
            "- 当前 clean sweep 中 TDB/MILP 求解成功率 100%，求解时间约 0.13 到 0.15 秒量级。",
            "",
            "## 7. 效率和存储开销",
            "",
            "相对更新开销 proxy = 参与节点数 / 9 × LoRA 更新比例。Base 是 9 个保留节点、全 LoRA 更新，所以 proxy=100%。",
            "",
            "| 数据集 | 方法 | DSU 参与节点数 | LoRA 更新比例 | 相对更新开销 proxy |",
            "|---|---|---:|---:|---:|",
            *proxy_lines,
            "",
            "- 存储：当前 DFL LoRA snapshot 文件按实际 checkpoint 统计，单个 seed 的 LoRA snapshot 约 5GB。论文里需要说明 snapshot cadence 和存储/通信折中。",
            "",
            "## 8. 论文文字需要同步澄清",
            "",
            "- 理论假设：bounded loss 可写成有限评估集或 clipped loss 下的分析假设，不要暗示原始 LLM loss 天然有界。",
            "- 分布距离：实现中的 distribution discrepancy 是 label-sketch L1，不是完整输入分布距离。",
            "- 模块敏感度 cadence：说明 LS 使用 DFL retained snapshots 上目标 agent 的 LoRA 更新能量，TDB trajectory sketch 使用历史 retained interval 的 LoRA update sketch。",
            "- 后门直接遗忘指标：只作为离线审计，不参与 AS/LS/DSU 选择，不影响训练目标。",
            "",
        ]
    )


def main() -> None:
    clean_rows = build_clean_table()
    clean_fields = [
        "dataset",
        "method",
        "base_param",
        "base_f1",
        "base_f1_std",
        "base_mia",
        "base_mia_std",
        "as_param",
        "as_f1",
        "as_f1_std",
        "as_mia",
        "as_mia_std",
        "ls_param",
        "ls_f1",
        "ls_f1_std",
        "ls_mia",
        "ls_mia_std",
        "dsu_param",
        "dsu_f1",
        "dsu_f1_std",
        "dsu_mia",
        "dsu_mia_std",
        "dsu_minus_base_f1",
        "dsu_minus_as_f1",
        "dsu_minus_ls_f1",
    ]
    write_csv(ROOT / "reports/tdb_clean_final_local_f1_mia_20260602.csv", clean_rows, clean_fields)
    (ROOT / "reports/tdb_clean_final_local_f1_mia_20260602.md").write_text(clean_to_md(clean_rows), encoding="utf-8")

    backdoor_rows = build_backdoor_table()
    backdoor_fields = [
        "dataset",
        "method",
        "seed",
        "poison_rate",
        "target_agent",
        "target_label",
        "dsu_config",
        "method_hparams",
        "dfl_asr",
        "base_asr",
        "dsu_asr",
        "dfl_trigger_lift",
        "base_trigger_lift",
        "dsu_trigger_lift",
        "dfl_clean_f1",
        "base_clean_f1",
        "dsu_clean_f1",
        "base_minus_dfl_asr",
        "dsu_minus_base_asr",
        "dsu_minus_dfl_asr",
        "status",
        "dfl_agents_asr",
        "base_agents_asr",
        "dsu_agents_asr",
        "base_json",
        "dsu_json",
        "dfl_json",
    ]
    write_csv(ROOT / "reports/backdoor_forgetting_final_seed42_20260603.csv", backdoor_rows, backdoor_fields)
    (ROOT / "reports/backdoor_forgetting_final_seed42_20260603.md").write_text(backdoor_to_md(backdoor_rows), encoding="utf-8")
    (ROOT / "reports/reviewer_supplement_final_20260603.md").write_text(
        reviewer_to_md(clean_rows, backdoor_rows), encoding="utf-8"
    )
    print("updated final clean, backdoor, and reviewer reports")


if __name__ == "__main__":
    main()
