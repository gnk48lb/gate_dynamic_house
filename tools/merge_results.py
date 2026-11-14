"""Merge episode logs from several runs into one file for analyze.py.

Why this exists: the DeepSeek quota ran out mid-project, so its matrix cannot
be re-run. The MiniMax cells can. To compare them, both have to land in one
episodes.jsonl - but they were produced under different ledger_cost_gate
prompts, which the merged report has to disclose (see PROMPT_NOTE below).

Usage:
    python tools/merge_results.py --out results_final/episodes.jsonl \
        --part deepseek:results_backup_20261113/episodes.jsonl \
        --part minimax:results_minimax_v2/episodes.jsonl

Later parts win for the same cell: analyze.py keeps the newest row per
(model_label, disturbance_p, method, seed), so a re-run supersedes the row it
replaces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROMPT_NOTE = (
    "合并时必须核对每个 cell 来自哪次运行：ledger_cost_gate 的提示词在 2026-11-14 改过"
    "（修复无效 ask_human 死循环）。若一部分 cell 用新提示词、一部分用旧提示词，"
    "同一方法名就对应两套策略，合并后的统计不可直接解释——需按 model_label 分开披露。"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part", action="append", required=True,
                        help="label:path to an episodes.jsonl (repeatable, later wins)")
    args = parser.parse_args()

    merged: dict[tuple[str, float, str, int], dict[str, Any]] = {}
    order: list[tuple[str, float, str, int]] = []
    provenance: dict[tuple[str, float, str, int], str] = {}
    for part in args.part:
        label, _, raw_path = part.partition(":")
        path = Path(raw_path)
        rows = load_jsonl(path)
        for row in rows:
            key = (row["model_label"], row["disturbance_p"], row["method"], row["seed"])
            if key not in merged:
                order.append(key)
            merged[key] = row
            provenance[key] = label

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for key in order:
            handle.write(json.dumps(merged[key], ensure_ascii=False, separators=(",", ":")) + "\n")

    counts: dict[str, int] = {}
    for label in provenance.values():
        counts[label] = counts.get(label, 0) + 1
    print(f"Wrote {len(order)} episodes to {args.out}")
    print(f"Rows per source part: {counts}")
    print(PROMPT_NOTE)


if __name__ == "__main__":
    main()
