#!/usr/bin/env python3
"""Summarize the 2026-06-02 local-DFU backdoor audit runs.

This is a reporting-only helper. It intentionally does not implement any
training-time selection, deployment copying, or ASR-based filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(
    r"^bd_grid_(?P<dataset>20newsgroups|yahoo_subset)_seed(?P<seed>\d+)_"
    r"(?P<method>d-federaser|d-fedosd|d-fedrecovery|d-oblivionis)_"
    r"(?P<strategy>full_all|ours_ours)"
    r"(?P<suffix>.*)$"
)

METHOD_ORDER = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
DATASET_ORDER = ["20newsgroups", "yahoo_subset"]
STRATEGY_NAME = {"full_all": "Base", "ours_ours": "DSU"}
FINAL_TAG_PREFERENCE = {
    # These DSU rows use the fixed LoRA ratios selected after the seed-42
    # local-DFU diagnosis. Base is unaffected by the ratio, but using the same
    # tagged run keeps each method row tied to one declared final configuration.
    ("20newsgroups", "d-fedrecovery"): "localfix_rate0p2_targetpoison_r1p0_20260602",
    ("20newsgroups", "d-oblivionis"): "localfix_rate0p2_targetpoison_r1p0_20260602",
    ("yahoo_subset", "d-fedrecovery"): "localfix_rate0p5_targetpoison_r0p5_20260602",
    ("yahoo_subset", "d-oblivionis"): "localfix_rate0p5_targetpoison_r0p5_20260602",
}


def pct(value: Any) -> float | None:
    try:
        return 100.0 * float(value)
    except Exception:
        return None


def fmt(value: Any, digits: int = 2) -> str:
    x = pct(value)
    return "-" if x is None else f"{x:.{digits}f}"


def fmt_raw_pct(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def metric(model: dict[str, Any], family: str, key: str = "asr") -> float | None:
    try:
        return float(((model.get(family) or {}).get(key)))
    except Exception:
        return None


def agent_map(model: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for agent in model.get("agents") or []:
        try:
            out[int(agent["agent_id"])] = agent
        except Exception:
            continue
    return out


def extract_rate(tag: str, data: dict[str, Any]) -> str:
    source_meta = data.get("source_meta") or {}
    raw = source_meta.get("backdoor_poison_rate")
    if raw is not None:
        try:
            return f"{float(raw):.3f}".rstrip("0").rstrip(".")
        except Exception:
            pass
    m = re.search(r"rate(?P<rate>\d+p\d+)", tag)
    if m:
        return m.group("rate").replace("p", ".")
    m = re.search(r"_bd(?P<rate>\d+p\d+)", tag)
    if m:
        return m.group("rate").replace("p", ".")
    return ""


def read_dfu_config(data: dict[str, Any]) -> dict[str, Any]:
    dfu_dir = Path(str(data.get("dfu_dir") or ""))
    cfg_path = dfu_dir / "dfu_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def collect(paths: Iterable[Path], tag_contains: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tag = str(data.get("tag") or path.parent.name)
        if tag_contains and not any(part in tag for part in tag_contains):
            continue
        m = TAG_RE.match(tag)
        if not m:
            continue
        if data.get("sample_source") != "target_poisoned_train":
            continue
        if data.get("eval_scope") != "all":
            continue
        if data.get("dfu_state_mode") != "participant":
            continue
        cfg = read_dfu_config(data)
        dfl = (data.get("models") or {}).get("dfl") or {}
        dfu = (data.get("models") or {}).get("dfu") or {}
        clean_target = metric(dfu, "clean_target_rate_non_target")
        dfu_asr = metric(dfu, "asr_non_target")
        dfl_clean_target = metric(dfl, "clean_target_rate_non_target")
        dfl_asr = metric(dfl, "asr_non_target")
        rows.append(
            {
                "path": str(path),
                "tag": tag,
                "dataset": m.group("dataset"),
                "seed": int(m.group("seed")),
                "method": m.group("method"),
                "strategy": m.group("strategy"),
                "setting": STRATEGY_NAME[m.group("strategy")],
                "rate": extract_rate(tag, data),
                "target_agent": data.get("source_meta", {}).get("source_agent_id"),
                "target_label": data.get("target_label"),
                "target_label_name": data.get("target_label_name"),
                "n_triggered": data.get("n_triggered"),
                "dfl_n_agents": dfl.get("n_agents"),
                "dfu_n_agents": dfu.get("n_agents"),
                "selected_agents": ",".join(str(x) for x in (cfg.get("selected_agents") or [])),
                "selection_count": cfg.get("selection_count"),
                "param_selection_ratio": cfg.get("param_selection_ratio"),
                "tdb_aggregation_scope": cfg.get("tdb_aggregation_scope"),
                "dfl_asr": dfl_asr,
                "dfl_clean_target_rate": dfl_clean_target,
                "dfl_trigger_lift": None if dfl_asr is None or dfl_clean_target is None else dfl_asr - dfl_clean_target,
                "dfl_clean_f1": metric(dfl, "clean", "macro_f1"),
                "dfu_asr": dfu_asr,
                "dfu_clean_target_rate": clean_target,
                "dfu_trigger_lift": None if dfu_asr is None or clean_target is None else dfu_asr - clean_target,
                "dfu_clean_f1": metric(dfu, "clean", "macro_f1"),
                "dfl_agents": agent_map(dfl),
                "dfu_agents": agent_map(dfu),
            }
        )
    return rows


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        DATASET_ORDER.index(row["dataset"]) if row["dataset"] in DATASET_ORDER else 99,
        METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else 99,
        row["seed"],
        0 if row["strategy"] == "full_all" else 1,
        row["tag"],
    )


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return None, None
    if len(arr) == 1:
        return arr[0], 0.0
    return mean(arr), pstdev(arr)


def fmt_mean(values: list[float | None]) -> str:
    m, s = mean_std(values)
    if m is None:
        return "-"
    return f"{100.0*m:.2f}±{100.0*(s or 0.0):.2f}"


def write_compact_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "method",
        "seed",
        "rate",
        "setting",
        "target_agent",
        "target_label",
        "target_label_name",
        "n_triggered",
        "dfl_n_agents",
        "dfu_n_agents",
        "selected_agents",
        "selection_count",
        "param_selection_ratio",
        "tdb_aggregation_scope",
        "dfl_asr",
        "dfl_clean_target_rate",
        "dfl_trigger_lift",
        "dfl_clean_f1",
        "dfu_asr",
        "dfu_clean_target_rate",
        "dfu_trigger_lift",
        "dfu_clean_f1",
        "tag",
        "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=row_sort_key):
            writer.writerow({field: row.get(field) for field in fields})


def write_agent_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "method",
        "seed",
        "rate",
        "setting",
        "stage",
        "mean_asr",
        "mean_clean_target_rate",
        "mean_trigger_lift",
        "mean_clean_f1",
    ] + [f"agent{i}_asr" for i in range(1, 10)] + [f"agent{i}_lift" for i in range(1, 10)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=row_sort_key):
            for stage, model_key, prefix in [
                ("DFL comparable scope", "dfl_agents", "dfl"),
                (f"{row['setting']} DFU", "dfu_agents", "dfu"),
            ]:
                agents = row[model_key]
                out: dict[str, Any] = {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "rate": row["rate"],
                    "setting": row["setting"],
                    "stage": stage,
                    "mean_asr": row[f"{prefix}_asr"],
                    "mean_clean_target_rate": row[f"{prefix}_clean_target_rate"],
                    "mean_trigger_lift": row[f"{prefix}_trigger_lift"],
                    "mean_clean_f1": row[f"{prefix}_clean_f1"],
                }
                for aid in range(1, 10):
                    agent = agents.get(aid)
                    if agent:
                        asr = metric(agent, "asr_non_target")
                        clean_target = metric(agent, "clean_target_rate_non_target")
                        out[f"agent{aid}_asr"] = asr
                        out[f"agent{aid}_lift"] = None if asr is None or clean_target is None else asr - clean_target
                    else:
                        out[f"agent{aid}_asr"] = None
                        out[f"agent{aid}_lift"] = None
                writer.writerow(out)


def pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["method"], row["seed"])
        setting = row["setting"]
        old = buckets[key].get(setting)
        if old is None or tag_preference_score(row) > tag_preference_score(old):
            buckets[key][setting] = row

    paired: list[dict[str, Any]] = []
    for key, by_setting in buckets.items():
        base = by_setting.get("Base")
        dsu = by_setting.get("DSU")
        if not base and not dsu:
            continue
        ref = base or dsu
        paired.append(
            {
                "dataset": key[0],
                "method": key[1],
                "seed": key[2],
                "rate": ref["rate"],
                "base": base,
                "dsu": dsu,
            }
        )
    return sorted(
        paired,
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"]) if row["dataset"] in DATASET_ORDER else 99,
            METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else 99,
            row["seed"],
        ),
    )


def dedupe_final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the same final-tag row that the markdown summary uses."""
    out: list[dict[str, Any]] = []
    for pair in pair_rows(rows):
        if pair.get("base"):
            out.append(pair["base"])
        if pair.get("dsu"):
            out.append(pair["dsu"])
    return sorted(out, key=row_sort_key)


def tag_preference_score(row: dict[str, Any]) -> int:
    preferred = FINAL_TAG_PREFERENCE.get((row["dataset"], row["method"]))
    if preferred and preferred in str(row.get("tag") or ""):
        return 2
    if "localfix_rate" in str(row.get("tag") or ""):
        return 1
    return 0


def cell(row: dict[str, Any] | None, prefix: str) -> str:
    if row is None:
        return "-"
    return (
        f"{fmt(row.get(f'{prefix}_asr'))}/"
        f"{fmt(row.get(f'{prefix}_trigger_lift'))}/"
        f"{fmt(row.get(f'{prefix}_clean_f1'))}"
    )


def agent_cell(agents: dict[int, dict[str, Any]], aid: int) -> str:
    agent = agents.get(aid)
    if not agent:
        return "-"
    asr = metric(agent, "asr_non_target")
    clean_target = metric(agent, "clean_target_rate_non_target")
    lift = None if asr is None or clean_target is None else asr - clean_target
    return f"{fmt(asr)}/{fmt(lift)}"


def write_md(path: Path, rows: list[dict[str, Any]], tag_contains: list[str], compact_csv: Path, agent_csv: Path) -> None:
    paired = pair_rows(rows)
    lines: list[str] = []
    lines.append("# Backdoor Local-DFU Final Audit")
    lines.append("")
    lines.append("本报告只汇总 `target_poisoned_train + participant + eval_scope=all` 口径的结果。")
    lines.append("数值格式是 `ASR_non_target / trigger lift / clean F1`，都是百分比；ASR 和 lift 越低越好，clean F1 只用于确认遗忘审计没有严重破坏干净任务。")
    lines.append("")
    lines.append(f"- tag filters: `{', '.join(tag_contains)}`")
    lines.append(f"- compact CSV: `{compact_csv}`")
    lines.append(f"- per-agent CSV: `{agent_csv}`")
    lines.append("")
    lines.append("## Seed-Level Summary")
    lines.append("")
    lines.append("| Dataset | Method | Seed | Rate | Base DFL | Base DFU | DSU comparable DFL | DSU DFU | DSU agents |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in paired:
        base = row["base"]
        dsu = row["dsu"]
        agents = "-"
        if dsu:
            agents = dsu.get("selected_agents") or ",".join(str(x) for x in sorted(dsu.get("dfu_agents", {})))
        lines.append(
            "| "
            f"{row['dataset']} | {row['method']} | {row['seed']} | {row['rate']} | "
            f"{cell(base, 'dfl')} | {cell(base, 'dfu')} | "
            f"{cell(dsu, 'dfl')} | {cell(dsu, 'dfu')} | {agents} |"
        )

    if paired:
        lines.append("")
        lines.append("## Mean Over Available Seeds")
        lines.append("")
        lines.append("| Dataset | Method | n | Base DFL ASR/lift/F1 | Base DFU ASR/lift/F1 | DSU DFL ASR/lift/F1 | DSU DFU ASR/lift/F1 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in paired:
            grouped[(row["dataset"], row["method"])].append(row)
        for (dataset, method), items in sorted(
            grouped.items(),
            key=lambda kv: (
                DATASET_ORDER.index(kv[0][0]) if kv[0][0] in DATASET_ORDER else 99,
                METHOD_ORDER.index(kv[0][1]) if kv[0][1] in METHOD_ORDER else 99,
            ),
        ):
            bases = [x["base"] for x in items if x["base"]]
            dsus = [x["dsu"] for x in items if x["dsu"]]

            def triple(source: list[dict[str, Any]], prefix: str) -> str:
                return (
                    f"{fmt_mean([x.get(f'{prefix}_asr') for x in source])}/"
                    f"{fmt_mean([x.get(f'{prefix}_trigger_lift') for x in source])}/"
                    f"{fmt_mean([x.get(f'{prefix}_clean_f1') for x in source])}"
                )

            lines.append(
                "| "
                f"{dataset} | {method} | {len(items)} | "
                f"{triple(bases, 'dfl')} | {triple(bases, 'dfu')} | "
                f"{triple(dsus, 'dfl')} | {triple(dsus, 'dfu')} |"
            )

    lines.append("")
    lines.append("## Per-Agent ASR/Lift")
    lines.append("")
    lines.append("下面的逐节点表只放 ASR/lift。Base 有 9 个保留节点；DSU 只列实际参与并落盘的节点，未参与节点为 `-`。")
    for row in paired:
        lines.append("")
        lines.append(f"### {row['dataset']} / {row['method']} / seed {row['seed']}")
        lines.append("")
        lines.append("| Stage | Mean ASR/lift | agent1 | agent2 | agent3 | agent4 | agent5 | agent6 | agent7 | agent8 | agent9 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        stages: list[tuple[str, dict[str, Any] | None, str, str]] = [
            ("DFL for Base scope", row["base"], "dfl", "dfl_agents"),
            ("Base DFU", row["base"], "dfu", "dfu_agents"),
            ("DFL for DSU scope", row["dsu"], "dfl", "dfl_agents"),
            ("DSU DFU", row["dsu"], "dfu", "dfu_agents"),
        ]
        for title, source, prefix, agents_key in stages:
            if source is None:
                continue
            mean_text = f"{fmt(source.get(f'{prefix}_asr'))}/{fmt(source.get(f'{prefix}_trigger_lift'))}"
            lines.append(
                "| "
                f"{title} | {mean_text} | "
                + " | ".join(agent_cell(source[agents_key], aid) for aid in range(1, 10))
                + " |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        default="实验结果/运行产物/artifacts/unlearning_audit/backdoor,artifacts/unlearning_audit/backdoor",
    )
    ap.add_argument(
        "--tag_contains",
        default="localfix_rate0p2_targetpoison_20260602,localfix_rate0p5_targetpoison_20260602,localfix_rate0p2_targetpoison_r1p0_20260602,localfix_rate0p5_targetpoison_r0p5_20260602",
    )
    ap.add_argument("--out_md", default="reports/backdoor_localfix_final_audit_20260602.md")
    ap.add_argument("--out_csv", default="reports/backdoor_localfix_final_audit_20260602.csv")
    ap.add_argument("--out_agent_csv", default="reports/backdoor_localfix_final_per_agent_20260602.csv")
    args = ap.parse_args()

    roots = [Path(x.strip()) for x in args.roots.split(",") if x.strip()]
    paths: list[Path] = []
    for root in roots:
        root = root if root.is_absolute() else ROOT / root
        if root.exists():
            paths.extend(root.glob("*/backdoor_audit.json"))
    tag_contains = [x.strip() for x in args.tag_contains.split(",") if x.strip()]
    rows = collect(paths, tag_contains)
    final_rows = dedupe_final_rows(rows)
    compact_csv = ROOT / args.out_csv
    agent_csv = ROOT / args.out_agent_csv
    out_md = ROOT / args.out_md
    write_compact_csv(compact_csv, final_rows)
    write_agent_csv(agent_csv, final_rows)
    write_md(out_md, final_rows, tag_contains, compact_csv.relative_to(ROOT), agent_csv.relative_to(ROOT))
    print(out_md)
    print(compact_csv)
    print(agent_csv)


if __name__ == "__main__":
    main()
