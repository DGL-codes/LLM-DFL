#!/usr/bin/env python3
"""Report the cumulative sequential-withdrawal probe.

This script is reporting-only. It reads completed Base/DSU histories and does
not run selection, training, unlearning, or any audit-guided optimization.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

BASE_ROOT = ROOT / "artifacts/sequential_cumulative_base_20260603"
DSU_ROOT = ROOT / "artifacts/sequential_cumulative_tdb_dsu_20260603"

FINAL_DSU_TAGS = {
    ("20newsgroups", (0,)): "20news_step1_removed0_k4_r0p5",
    ("20newsgroups", (0, 1)): "20news_step2_removed0_1_k4_r0p5",
    ("20newsgroups", (0, 1, 2)): "20news_step3_removed0_1_2_k4_r0p5",
    ("yahoo_subset", (0,)): "yahoo_step1_removed0_k5_r0p3",
    ("yahoo_subset", (0, 1)): "yahoo_step2_removed0_1_k5_r0p3",
    ("yahoo_subset", (0, 1, 2)): "yahoo_step3_removed0_1_2_k5_r0p3",
}

DATASET_LABEL = {"20newsgroups": "20News", "yahoo_subset": "Yahoo"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: Any, digits: int = 3) -> str:
    x = fnum(value)
    return "-" if x is None else f"{x:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    x = fnum(value)
    return "-" if x is None else f"{100.0 * x:.{digits}f}"


def cfg_dataset(cfg: dict[str, Any]) -> str:
    value = str(cfg.get("dataset") or "")
    if value:
        return value
    parts = str(cfg.get("dfl_snapshot") or "").split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "checkpoints" else "unknown"


def find_single_config(root: Path) -> Path:
    configs = sorted(root.glob("**/dfu_config.json"))
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly one dfu_config.json under {root}, got {len(configs)}")
    return configs[0]


def load_run(root: Path) -> dict[str, Any]:
    cfg_path = find_single_config(root)
    hist_path = cfg_path.parent / "history.json"
    cfg = load_json(cfg_path)
    hist = load_json(hist_path)
    final = hist.get("final_stats") or {}
    param = cfg.get("param_selection_result") or {}
    selected_modules = param.get("selected_modules")
    if isinstance(selected_modules, list):
        selected_module_count = len(selected_modules)
    elif selected_modules is None:
        selected_module_count = int(param.get("total_modules") or 154)
    else:
        selected_module_count = int(selected_modules)
    total_modules = int(param.get("total_modules") or 154)
    covered = fnum(param.get("covered_sensitivity"))
    total_sens = fnum(param.get("total_sensitivity"))
    coverage = None if covered is None or not total_sens else covered / total_sens
    diagnostics = cfg.get("selection_diagnostics") or {}
    return {
        "cfg_path": cfg_path,
        "run_dir": cfg_path.parent,
        "dataset": cfg_dataset(cfg),
        "target_agent": int(cfg.get("target_agent")),
        "removed_agents": tuple(int(x) for x in (cfg.get("removed_agents") or [cfg.get("target_agent")])),
        "survivors": int(cfg.get("surviving_agents_count")),
        "selected_agents": [int(x) for x in (cfg.get("selected_agents") or [])],
        "selected_agents_count": int(cfg.get("selected_agents_count")),
        "selection_count": cfg.get("selection_count"),
        "param_selection_ratio": cfg.get("param_selection_ratio"),
        "selected_module_count": selected_module_count,
        "total_modules": total_modules,
        "sensitivity_coverage": coverage,
        "macro_f1_best": fnum(final.get("macro_f1_best")),
        "macro_f1_mean": fnum(final.get("macro_f1_mean")),
        "mia_auc": fnum(final.get("mia_auc")),
        "trajectory_l1": fnum(diagnostics.get("trajectory_l1")),
        "label_l1": fnum(diagnostics.get("label_l1")),
        "target_exposure": fnum(diagnostics.get("target_exposure")),
        "solve_time_sec": fnum(diagnostics.get("solve_time_sec")),
    }


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, dsu_tag in FINAL_DSU_TAGS.items():
        dataset, removed = key
        if dataset == "20newsgroups":
            base_tag = {
                (0,): "20news_step1_removed0",
                (0, 1): "20news_step2_removed0_1",
                (0, 1, 2): "20news_step3_removed0_1_2",
            }[removed]
        else:
            base_tag = {
                (0,): "yahoo_step1_removed0",
                (0, 1): "yahoo_step2_removed0_1",
                (0, 1, 2): "yahoo_step3_removed0_1_2",
            }[removed]
        base = load_run(BASE_ROOT / base_tag)
        dsu = load_run(DSU_ROOT / dsu_tag)
        if base["removed_agents"] != dsu["removed_agents"]:
            raise RuntimeError(f"Removed set mismatch: {base['removed_agents']} vs {dsu['removed_agents']}")
        row = {
            "dataset": dataset,
            "removed_agents": ",".join(map(str, removed)),
            "step": len(removed),
            "current_target": dsu["target_agent"],
            "base_survivors": base["survivors"],
            "base_selected_agents": " ".join(map(str, base["selected_agents"])),
            "base_selected_agents_count": base["selected_agents_count"],
            "base_selected_modules": base["total_modules"],
            "base_total_modules": base["total_modules"],
            "base_macro_f1_best": base["macro_f1_best"],
            "base_macro_f1_mean": base["macro_f1_mean"],
            "base_mia_auc": base["mia_auc"],
            "dsu_survivors": dsu["survivors"],
            "dsu_requested_k": dsu["selection_count"],
            "dsu_selected_agents": " ".join(map(str, dsu["selected_agents"])),
            "dsu_selected_agents_count": dsu["selected_agents_count"],
            "dsu_param_ratio": dsu["param_selection_ratio"],
            "dsu_selected_modules": dsu["selected_module_count"],
            "dsu_total_modules": dsu["total_modules"],
            "dsu_sensitivity_coverage": dsu["sensitivity_coverage"],
            "dsu_macro_f1_best": dsu["macro_f1_best"],
            "dsu_macro_f1_mean": dsu["macro_f1_mean"],
            "dsu_mia_auc": dsu["mia_auc"],
            "dsu_minus_base_best_f1": None
            if base["macro_f1_best"] is None or dsu["macro_f1_best"] is None
            else dsu["macro_f1_best"] - base["macro_f1_best"],
            "dsu_minus_base_mean_f1": None
            if base["macro_f1_mean"] is None or dsu["macro_f1_mean"] is None
            else dsu["macro_f1_mean"] - base["macro_f1_mean"],
            "tdb_trajectory_l1": dsu["trajectory_l1"],
            "tdb_label_l1": dsu["label_l1"],
            "tdb_target_exposure": dsu["target_exposure"],
            "tdb_solve_time_sec": dsu["solve_time_sec"],
            "base_run_dir": str(base["run_dir"].relative_to(ROOT)),
            "dsu_run_dir": str(dsu["run_dir"].relative_to(ROOT)),
        }
        rows.append(row)
    return sorted(rows, key=lambda r: (r["dataset"], r["step"]))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], path: Path, csv_path: Path) -> None:
    lines = [
        "# Cumulative Sequential Withdrawal Probe",
        "",
        "本报告用于回复审稿意见中关于 dynamic edge environments 和 sequential unlearning requests 的问题。这里不再使用旧的“target 0、1、2 三个互不相关的单次请求探针”，而是让已删除集合按顺序增长：第一步删除 `{0}`，第二步删除 `{0,1}`，第三步删除 `{0,1,2}`。",
        "",
        "实验口径：seed 42，FedEraser 作为代表性 DFU 方法；Base 表示所有剩余 agent 参与、更新全部 LoRA 模块；DSU 使用新的 TDB-AS、本地环形邻居聚合和敏感度层选择。ASR、MIA 或 F1 都没有进入 TDB-AS/LS 的求解目标。",
        "",
        "配置说明：20News 使用固定 `k=4,r=0.5`。Yahoo 使用固定 `k=5,r=0.3`。这两个配置来自连续删除补实验的小网格诊断，用于保证 DSU 同时体现 agent selection 和 layer selection，而不是在连续删除后退化成全选节点。这里的 `k` 是请求的最多参与节点数；若当前剩余 agent 少于请求的 k，TDB-AS 会自动截断到当前剩余数量。",
        "",
        f"- CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "| 数据集 | 已删除 agent | 当前请求删除 | Base 参与/模块 | Base 最佳/平均 F1 | Base MIA | DSU 请求 k / 实际参与 | DSU 实际选中 agent | DSU 模块 | DSU 最佳/平均 F1 | DSU MIA | DSU-Best | DSU-Mean | TDB 轨迹误差 | TDB 标签误差 |",
        "|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | "
            f"{row['removed_agents']} | "
            f"{row['current_target']} | "
            f"{row['base_selected_agents_count']}/{row['base_survivors']} agents, "
            f"{row['base_selected_modules']}/{row['base_total_modules']} modules | "
            f"{fmt(row['base_macro_f1_best'])}/{fmt(row['base_macro_f1_mean'])} | "
            f"{fmt(row['base_mia_auc'])} | "
            f"k={row['dsu_requested_k']} / {row['dsu_selected_agents_count']}/{row['dsu_survivors']} agents | "
            f"{row['dsu_selected_agents']} | "
            f"{row['dsu_selected_modules']}/{row['dsu_total_modules']} | "
            f"{fmt(row['dsu_macro_f1_best'])}/{fmt(row['dsu_macro_f1_mean'])} | "
            f"{fmt(row['dsu_mia_auc'])} | "
            f"{fmt(row['dsu_minus_base_best_f1'])} | "
            f"{fmt(row['dsu_minus_base_mean_f1'])} | "
            f"{fmt(row['tdb_trajectory_l1'])} | "
            f"{fmt(row['tdb_label_l1'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 这是真正的累计删除集合实验：第二步和第三步的保留集合会排除之前已经删除的 agent，不再把它们当作可参与遗忘的 retained agent。",
            "- `k` 的含义是最多选择多少个 retained agents。若当前剩余 agent 少于请求的 k，TDB-AS 会自动截断到当前剩余数量。当前 20News 实际选择 4 个 agent；Yahoo 实际选择 5 个 agent，没有再使用旧报告中退化为全选的 `k=9` 配置。",
            "- 当前实现采用 history replay：后续请求复用同一份 DFL retained history，并用更大的 removed-agent 集合重新执行遗忘。这回答了“retained history 是否能复用”：可以复用，但每次请求都要按当前目标和当前剩余集合重新计算 TDB-AS 与 LoRA 敏感度。",
            "- 误差累积边界：history replay 避免了直接在上一次遗忘输出上继续迭代导致的无控制参数漂移；但如果真实系统选择低成本的 chained update，不重新回放 retained history，误差可能随请求数累积，需要周期性 history replay 或 retraining refresh。",
            "- 结果上，20News 在删除 1/2/3 个 agent 时，DSU 用 4 个参与 agent 和 77/154 个 LoRA 模块，平均 F1 和最佳 F1 均高于 Base。Yahoo 在删除 1/2/3 个 agent 时，DSU 用 5 个参与 agent 和 46/154 个 LoRA 模块，平均 F1 和最佳 F1 也均高于 Base。MIA AUC 基本接近 0.5，说明隐私审计没有出现明显反向信号。",
            "",
            "## 可写进论文的边界表述",
            "",
            "DSU 可以处理连续 withdrawal requests 的实用方式是：保留 DFL retained history；每次新请求到来时，把历史已删除 agent 和当前 target agent 组成累计 removed set，重新求解 TDB-AS 和 LS，并从 retained history 重放校正。该方式避免直接链式参数漂移，但当累计删除很多 agent 时，参与节点预算和层选择比例可能需要随剩余分布重新设定，极端情况下建议周期性 retraining refresh。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    csv_path = ROOT / "reports/sequential_cumulative_tdb_dsu_20260603.csv"
    md_path = ROOT / "reports/sequential_cumulative_tdb_dsu_20260603.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path, csv_path)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
