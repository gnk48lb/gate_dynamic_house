"""Recompute Dynamic House metrics, paired statistics, Pareto plots, and gate verdict."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics

METHODS = ("sliding_window", "append_only_memory", "always_observe", "ledger_cost_gate")
COLORS = {"sliding_window": "#69788c", "append_only_memory": "#d17a3a",
          "always_observe": "#4b8a91", "ledger_cost_gate": "#244b70"}
MARKERS = {0.0: "o", 0.2: "s", 0.4: "^"}


def quantile(values, q):
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def normal_sf(value):
    return 0.5 * math.erfc(value / math.sqrt(2))


def paired_cost_test(differences):
    """Exact/bootstrap mean CI + Wilcoxon signed-rank normal approximation with ties."""
    nonzero = [abs(x) for x in differences if x != 0]
    n = len(nonzero)
    if not differences:
        return {"n": 0, "mean_cost_reduction": None, "ci95": None, "wilcoxon_p": None}
    if not n:
        return {"n": len(differences), "mean_cost_reduction": 0.0,
                "ci95": [0.0, 0.0], "wilcoxon_p": 1.0}
    distinct = sorted(set(nonzero))
    rank = {value: 1 + sum(v < value for v in nonzero) + (nonzero.count(value) - 1) / 2 for value in distinct}
    w_plus = sum(rank[abs(x)] for x in differences if x > 0)
    mean_w, variance_w = n * (n + 1) / 4, n * (n + 1) * (2 * n + 1) / 24
    tie_correction = sum(count ** 3 - count for count in
                         (nonzero.count(value) for value in distinct))
    variance_w -= tie_correction / 48
    # Two-sided continuity-corrected signed-rank test; H0: median difference is zero.
    z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(variance_w) if variance_w else 0
    p_value = min(1.0, 2 * normal_sf(max(0, z)))
    rng = random.Random(20261113)
    boot = [sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(10000)]
    return {"n": len(differences), "mean_cost_reduction": statistics.mean(differences),
            "ci95": [quantile(boot, 0.025), quantile(boot, 0.975)], "wilcoxon_p": p_value}


def fmt(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/episodes.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No episodes to analyze")
    # A repaired episode is appended after the original; the latest record for a
    # given cell wins, so an upstream timeout never counts as a policy failure.
    latest = {}
    for row in rows:
        latest[(row["model_label"], row["disturbance_p"], row["method"], row["seed"])] = row
    rows = list(latest.values())
    repaired = sum(1 for row in rows if row.get("repair_of_infrastructure_failure"))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["disturbance_p"], row["method"])].append(row)
    model_labels = sorted({row["model_label"] for row in rows})
    ps = sorted({row["disturbance_p"] for row in rows})
    metrics = {}
    for key, values in sorted(grouped.items()):
        n = len(values)
        metrics[key] = {
            "episodes": n,
            "success_rate": 100 * sum(v["success"] for v in values) / n,
            "avg_cost": statistics.mean(v["total_cost"] for v in values),
            "avg_remaining_budget": statistics.mean(v["remaining_budget"] for v in values),
            "stale_pick_failure_rate": 100 * sum(v["stale_pick_count"] for v in values) / max(1, sum(v["pick_count"] for v in values)),
            "avg_human_queries": statistics.mean(v["ask_count"] for v in values),
            "safety_violation_rate": 100 * sum(v["safety_count"] > 0 for v in values) / n,
            "avg_steps": statistics.mean(v["steps"] for v in values),
            "avg_tokens": statistics.mean(v["token_usage"]["total_tokens"] for v in values),
            "infrastructure_failures": sum(v["infrastructure_failure"] for v in values),
            "failure_counts": {name: sum(v["failure_reason"] == name for v in values)
                               for name in ("stale_memory_pick_fail", "budget_exhausted", "safety_violation", "max_steps",
                                            "infrastructure_failure", "incomplete")},
            "terminal_counts": {name: sum(v.get("terminal_reason") == name for v in values)
                                for name in ("budget_exhausted", "safety_violation", "max_steps", None)},
        }
    # Plain CSV is useful for independent replotting / checking.
    csv_lines = ["model,disturbance_p,method,episodes,success_rate_pct,avg_action_cost,avg_remaining_budget,stale_pick_failure_rate_pct,avg_human_queries,safety_violation_rate_pct,avg_steps,avg_total_tokens,infrastructure_failures"]
    for (label, p, method), value in metrics.items():
        csv_lines.append(",".join(str(x) for x in (label, p, method, value["episodes"], value["success_rate"],
                         value["avg_cost"], value["avg_remaining_budget"], value["stale_pick_failure_rate"],
                         value["avg_human_queries"], value["safety_violation_rate"], value["avg_steps"],
                         value["avg_tokens"], value["infrastructure_failures"])))
    (output / "metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    serializable_metrics = [{"model_label": key[0], "disturbance_p": key[1], "method": key[2], **value}
                            for key, value in metrics.items()]
    (output / "metrics.json").write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        # Palette validated with the data-viz checker (light surface, all pairs):
        # passes lightness band, chroma floor, normal-vision floor and contrast.
        # The two CVD/contrast WARNs are covered by the secondary encoding below
        # (per-point direct labels plus distinct marker shapes per method).
        palette = {"sliding_window": "#2a78d6", "append_only_memory": "#1baf7a",
                   "always_observe": "#e87ba4", "ledger_cost_gate": "#008300"}
        shapes = {"sliding_window": "o", "append_only_memory": "s",
                  "always_observe": "^", "ledger_cost_gate": "D"}
        ink, ink_soft, grid = "#0b0b0b", "#52514e", "#e6e6e2"
        fig, axes = plt.subplots(1, len(ps), figsize=(5.4 * len(ps), 5.4), sharex=True, sharey=True, squeeze=False)
        for axis, p in zip(axes[0], ps):
            axis.set_title(f"Roommate disturbance p = {p}", fontsize=12.5, pad=10, color=ink)
            axis.set_xlabel("Average action cost (budget points)", fontsize=10.5, color=ink_soft)
            axis.grid(True, color=grid, linewidth=0.8)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                axis.spines[spine].set_color(grid)
            axis.tick_params(colors=ink_soft, labelsize=9.5)
            # Pareto frontier of the cost/success trade-off for this p.
            points = [(metrics[(label, p, method)]["avg_cost"], metrics[(label, p, method)]["success_rate"],
                       label) for label in model_labels for method in METHODS
                      if (label, p, method) in metrics]
            if len(points) > 1:
                best, frontier = -1, []
                for x, y, _ in sorted(points):
                    if y > best:
                        frontier.append((x, y))
                        best = y
                axis.plot([v[0] for v in frontier], [v[1] for v in frontier],
                          color="#c9c9c4", linewidth=1.2, zorder=1)
            # Collect points first so labels can be placed without collisions.
            plotted = []
            for method in METHODS:
                for label in model_labels:
                    if (label, p, method) not in metrics:
                        continue
                    m = metrics[(label, p, method)]
                    axis.scatter(m["avg_cost"], m["success_rate"], s=115, marker=shapes[method],
                                 color=palette[method], edgecolor="#fcfcfb", linewidth=1.6, zorder=3)
                    plotted.append({"label": label, "method": method, "x": m["avg_cost"],
                                    "y": m["success_rate"]})
            # Place labels without collisions: group by rounded y, then stagger
            # the labels inside each row so near-identical points stay readable.
            plotted.sort(key=lambda row: (round(row["y"], 1), row["x"]))
            label_groups: dict[float, list] = {}
            for row in plotted:
                label_groups.setdefault(round(row["y"], 1), []).append(row)
            for _, row_list in label_groups.items():
                for index, row in enumerate(row_list):
                    row["dx"] = 8
                    row["dy"] = 7
                # If labels in this row would overlap horizontally, alternate
                # them above and below the marker.
                if len(row_list) > 1:
                    for index, row in enumerate(row_list):
                        row["dy"] = 7 if index % 2 == 0 else -14
            for row in plotted:
                axis.annotate(row["label"], (row["x"], row["y"]),
                              xytext=(row["dx"], row["dy"]), textcoords="offset points",
                              fontsize=9, color=ink_soft, zorder=4)
            axis.set_xlim(left=0)
            axis.set_ylim(-4, 118)
        axes[0][0].set_ylabel("Task success rate (%)", fontsize=10.5, color=ink_soft)
        handles = [Line2D([], [], marker=shapes[m], linestyle="", color=palette[m], label=m,
                          markersize=8.5, markeredgecolor="#fcfcfb", markeredgewidth=1.4)
                   for m in METHODS]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.075), fontsize=10.5, labelcolor=ink)
        fig.suptitle("Dynamic House: cost–success Pareto comparison", fontsize=16.5, y=0.985, color=ink)
        fig.text(0.5, 0.018, "Lower cost and higher success are better. Point labels name the model; "
                             "marker shape and colour name the method.", ha="center", fontsize=9.5, color=ink_soft)
        fig.tight_layout(rect=(0, 0.135, 1, 0.945))
        fig.savefig(output / "pareto_frontier.png", dpi=180, facecolor="#fcfcfb")
        plt.close(fig)
    except ImportError as exc:
        raise SystemExit("matplotlib is required for the requested plot: " + str(exc)) from exc

    lines = ["# Dynamic House 纯文本闸门实验报告", "",
             f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
             "## 实验定义", "",
             "五个连通房间、18 件物品（6 件高风险）、初始预算 60、最多 30 步。每个回合先注入覆盖全屋的巡检记忆，再经过 8 个后台干扰步，最后按顺序归位 2–3 件物品。相同模型、seed 与扰动率下的四种方法共享相同环境随机种子。室友优先移动当前待办子目标的物品（均匀随机命中目标仅约 10%，无法产生过期记忆场景）。室友真实变更只进入审计日志，不进入 Agent 提示。", "",
             "成本为动作实际扣减的预算（60 − 剩余预算 = 各动作成本之和），抓空与安全违规的 -15 / -25 是分数惩罚、不计入成本，LLM token 也不计入成本。抓空率的分母为全部 pick 动作；高风险抓空同时计入安全违规。`always_observe` 在每次 pick 前由策略强制观察。`ledger_cost_gate` 使用覆盖式账本与显式成本门控提示；不读取真实位置。被问过位置的物品在抓住前不再被室友移动，因此问人的答案可用、10 点成本可被权衡。", "",
             "本实验检验特定提示与策略在此环境中的表现，不构成对 Append-only / Mem0 / A-Mem 全部实现的普遍结论。", ""]
    manifest_path = args.input.parent / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lines += ["## 实验配置", "",
                  f"运行 ID：`{manifest.get('run_id')}`；每格回合数：{manifest.get('episodes_per_cell')}；"
                  f"预算：{manifest.get('budget')}；最大步数：{manifest.get('max_steps')}；"
                  f"巡检间隔：{manifest.get('inspection_gap')} 步；temperature={manifest.get('temperature')}；"
                  f"max_tokens={manifest.get('max_tokens')}；创建时间（UTC）：{manifest.get('created_at')}", "",
                  "| 预设名 | 实际模型 ID | 供应商 |", "|---|---|---|"]
        for label, model in (manifest.get("models") or {}).items():
            lines.append(f"| {label} | `{model}` | {model.split('/')[0] if '/' in model else '—'} |")
        # Episodes re-run through a different route (the original provider ran
        # out of quota) must be disclosed: the model is the same, the route is not.
        routes = defaultdict(int)
        for row in rows:
            if row.get("actual_model"):
                routes[(row["model_label"], row["actual_model"])] += 1
        if routes:
            lines += ["", "**路由变更披露**：部分回合因原供应商额度耗尽而改由另一条路由重跑（同一模型、不同 provider）。"
                      "这些回合的 `actual_model` 字段保留了实际路由，可逐回合审计。"
                      "重跑同时把单次请求的 max_tokens 从 2000 提高到 4000（思考模型在长 prompt 下会把预算耗在推理链上、"
                      "导致正文为空），这一改动只影响被重跑的那几回合，且与被替换的路由一并披露：", "",
                      "| 模型标签 | 实际路由 | 回合数 |", "|---|---|---:|"]
            for (label, model), count in sorted(routes.items()):
                lines.append(f"| {label} | `{model}` | {count} |")
        lines.append("")
    lines += ["## 汇总结果", "",
              f"（共 {len(rows)} 个有效回合，其中 {repaired} 个为基础设施失败后的修复重跑；"
              "同一格重复记录以最新一次为准。）", "",
             "SR=成功率；Cost=平均动作成本；Remain=平均剩余预算；Stale=过期记忆抓空率；Ask=平均问人次数；Safety=安全违规率；Infra=模型请求/协议异常回合数。", "",
             "| Model | p | Method | N | SR % | Cost | Remain | Stale % | Ask | Safety % | Infra |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in model_labels:
        for p in ps:
            for method in METHODS:
                if (label, p, method) not in metrics:
                    continue
                m = metrics[(label, p, method)]
                lines.append(f"| {label} | {p} | `{method}` | {m['episodes']} | {fmt(m['success_rate'])} | "
                             f"{fmt(m['avg_cost'])} | {fmt(m['avg_remaining_budget'])} | {fmt(m['stale_pick_failure_rate'])} | "
                             f"{fmt(m['avg_human_queries'], 2)} | {fmt(m['safety_violation_rate'])} | {m['infrastructure_failures']} |")
    lines += ["", "## 闸门判定", "",
              "判据：在 p=0.2 和 p=0.4，`ledger_cost_gate` 相比 `append_only_memory` 的成功率提升至少 10 个百分点；且相对 `always_observe` 平均动作成本更低，配对成本差异达到统计显著（Wilcoxon p<0.05 且配对 bootstrap 均值差 95% CI 不跨 0）。检验为探索性、未做多重比较校正。",
              "",
              "基础设施失败（模型请求/协议异常）按**失败**计入成功率分母，是保守方向。它们是供应商故障而非策略结果，因此另做敏感性检查：剔除这些回合后重算提升幅度与成本降幅；若结论方向改变则拒绝判定，并在报告中一并披露占比。", "",
              "### 主判定（跨模型合并，按模型×seed 配对）", ""]
    verdicts = []
    pooled = []
    for p in (0.2, 0.4):
        needed = ("append_only_memory", "always_observe", "ledger_cost_gate")
        if any(not all((label, p, method) in grouped for label in model_labels) for method in needed):
            lines.append(f"- **p={p}：无法判定**（三组方法数据不全）。")
            pooled.append({"disturbance_p": p, "passed": None, "reason": "incomplete_cells"})
            continue
        pairs = []
        for label in model_labels:
            append = {v["seed"]: v for v in grouped[(label, p, "append_only_memory")]}
            always = {v["seed"]: v for v in grouped[(label, p, "always_observe")]}
            ours = {v["seed"]: v for v in grouped[(label, p, "ledger_cost_gate")]}
            for seed in sorted(set(append) & set(always) & set(ours)):
                pairs.append((label, seed, append[seed], always[seed], ours[seed]))
        reduction = [always_row["total_cost"] - ours_row["total_cost"] for _, _, _, always_row, ours_row in pairs]
        test = paired_cost_test(reduction)
        ours_rows = [row for _, _, _, _, row in pairs]
        append_rows = [row for _, _, row, _, _ in pairs]
        always_rows = [row for _, _, _, row, _ in pairs]
        delta = 100 * (sum(r["success"] for r in ours_rows) - sum(r["success"] for r in append_rows)) / len(pairs)
        cost_reduction = statistics.mean(r["total_cost"] for r in always_rows) - statistics.mean(r["total_cost"] for r in ours_rows)
        ci = test["ci95"]
        infra = sum(r.get("infrastructure_failure", False) for r in ours_rows + append_rows + always_rows)
        infra_share = infra / (3 * len(pairs))
        cost_significant = cost_reduction > 0 and test["wilcoxon_p"] < 0.05 and ci[0] > 0
        # Infrastructure failures are provider faults, not policy outcomes, and
        # they are counted as failures here (the conservative direction). The
        # verdict therefore also checks the direction of the conclusion with
        # those episodes removed, and only refuses when dropping them would
        # change it.
        clean = [(a, o, w) for a, o, w in
                 ((append_row, ours_row, always_row) for _, _, append_row, always_row, ours_row in pairs)
                 if not (a.get("infrastructure_failure") or o.get("infrastructure_failure")
                         or w.get("infrastructure_failure"))]
        if clean:
            clean_delta = 100 * (sum(r[1]["success"] for r in clean) - sum(r[0]["success"] for r in clean)) / len(clean)
            clean_cost = statistics.mean(r[2]["total_cost"] for r in clean) - statistics.mean(r[1]["total_cost"] for r in clean)
        else:
            clean_delta, clean_cost = delta, cost_reduction
        direction_holds = (clean_delta > 0 and clean_cost > 0) and (delta > 0 and cost_reduction > 0)
        if not direction_holds:
            passed, reason = False, "剔除基础设施失败回合后结论方向改变，拒绝判定"
        elif delta >= 10 and cost_significant:
            passed, reason = True, ""
        else:
            passed, reason = False, ("成功率提升不足 10 个百分点" if delta < 10 else "成本差异未达显著")
        pooled.append({"disturbance_p": p, "success_delta_pp": delta, "cost_reduction": cost_reduction,
                       "paired_test": test, "infrastructure_failure_share": infra_share,
                       "success_delta_pp_excluding_infra": clean_delta,
                       "cost_reduction_excluding_infra": clean_cost,
                       "passed": passed, "reason": reason, "pairs": len(pairs)})
        lines.append(f"- **p={p}：{'通过' if passed else '未通过'}**。成功率提升 **{delta:+.1f} pp**；"
                     f"相对 always_observe 平均成本降低 **{cost_reduction:.2f}** 点；"
                     f"配对成本差 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]，Wilcoxon p={test['wilcoxon_p']:.4g}，"
                     f"n={test['n']} 对（模型×seed）；基础设施失败占比 {infra_share:.1%}（已按失败计入分母）。"
                     f"敏感性检查：剔除这些回合后提升为 {clean_delta:+.1f} pp、成本降低 {clean_cost:.2f} 点，"
                     f"结论方向{'不变' if direction_holds else '改变'}。" +
                     (f" 未通过原因：{reason}。" if reason else ""))
    lines += ["", "### 分模型判定", ""]
    for label in model_labels:
        for p in (0.2, 0.4):
            if any((label, p, method) not in metrics for method in ("append_only_memory", "always_observe", "ledger_cost_gate")):
                lines.append(f"- **{label}, p={p}：无法判定**（三组方法数据不全）。")
                continue
            ledger = grouped[(label, p, "ledger_cost_gate")]
            append = {v["seed"]: v for v in grouped[(label, p, "append_only_memory")]}
            always = {v["seed"]: v for v in grouped[(label, p, "always_observe")]}
            seeds = sorted(set(append) & set(always) & {v["seed"] for v in ledger})
            ours = {v["seed"]: v for v in ledger}
            reduction = [always[s]["total_cost"] - ours[s]["total_cost"] for s in seeds]
            test = paired_cost_test(reduction)
            delta = metrics[(label, p, "ledger_cost_gate")]["success_rate"] - metrics[(label, p, "append_only_memory")]["success_rate"]
            cost_reduction = metrics[(label, p, "always_observe")]["avg_cost"] - metrics[(label, p, "ledger_cost_gate")]["avg_cost"]
            ci = test["ci95"]
            cost_significant = cost_reduction > 0 and test["wilcoxon_p"] < 0.05 and ci[0] > 0
            passed = delta >= 10 and cost_significant
            verdicts.append({"model_label": label, "disturbance_p": p, "success_delta_pp": delta,
                             "cost_reduction": cost_reduction, "paired_test": test, "passed": passed})
            lines.append(f"- **{label}, p={p}：{'通过' if passed else '未通过'}**。成功率提升 **{delta:+.1f} pp**；相对 always_observe 平均成本降低 **{cost_reduction:.2f}** 点；配对成本差 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]，Wilcoxon p={test['wilcoxon_p']:.4g}，n={test['n']}。")
    pooled_pass = bool(pooled) and all(v["passed"] for v in pooled)
    lines += ["", "**主闸门结论（p=0.2 与 p=0.4 合并判定均通过）：" +
              ("通过。**" if pooled_pass else "未通过。**") + "**", "",
              "分模型一致性（每个模型在两个动态扰动率均通过）：" +
              ("**通过**。" if verdicts and all(v["passed"] for v in verdicts) else "**未通过（或数据不完整）**。"), "",
              "![成本–成功率帕累托比较](pareto_frontier.png)", "",
              "## 失败原因与解释边界", "",
             "根因列为互斥归类：因过期记忆抓空而最终失败记为 `stale_memory_pick_fail`；终止列保留真实结束原因。", "",
             "| Model | p | Method | stale_memory_pick_fail | budget_exhausted | safety_violation | max_steps | infra | incomplete | 终止: budget | 终止: safety | 终止: max_steps |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in model_labels:
        for p in ps:
            for method in METHODS:
                if (label, p, method) not in metrics:
                    continue
                counts = metrics[(label, p, method)]["failure_counts"]
                terminal = metrics[(label, p, method)]["terminal_counts"]
                lines.append(f"| {label} | {p} | `{method}` | " + " | ".join(str(counts[k]) for k in
                             ("stale_memory_pick_fail", "budget_exhausted", "safety_violation", "max_steps",
                              "infrastructure_failure", "incomplete")) +
                             f" | {terminal.get('budget_exhausted')} | {terminal.get('safety_violation')} | {terminal.get('max_steps')} |")
    lines += ["", "注意：一个回合可能出现多次普通物品抓空后继续执行，因此“回合最终失败原因”不等于“发生过抓空”。`failure_reason` 为互斥根因：回合内发生过过期记忆抓空且最终以步数或预算耗尽结束时，根因记为 `stale_memory_pick_fail`；`terminal_reason` 保留真实结束原因（安全违规、预算耗尽、达到最大步数）。普通抓空本身不终止回合。基础设施异常单独列出，不解释为记忆问题，并计入成功率分母。", "",
              "图表由 `metrics.csv` 对应的原始回合日志自动计算。`episodes.jsonl` 可用于复算本表；`traces.jsonl` 保存逐步账本、动作、室友真实变更、预算及 token 用量；`raw_responses.jsonl` 保存原始模型输出。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "gate_verdicts.json").write_text(
        json.dumps({"primary_pooled_verdicts": pooled, "per_model_verdicts": verdicts}, indent=2), encoding="utf-8")
    print(f"Wrote {output / 'summary.md'} and {output / 'pareto_frontier.png'}")
    for verdict in pooled:
        if verdict.get("passed") is None:
            print(f"pooled p={verdict['disturbance_p']}: INCOMPLETE ({verdict.get('reason')})")
        else:
            print(f"pooled p={verdict['disturbance_p']}: {'PASS' if verdict['passed'] else 'FAIL'}; "
                  f"success delta={verdict['success_delta_pp']:+.1f}pp; "
                  f"cost reduction={verdict['cost_reduction']:.2f}; "
                  f"infra={verdict['infrastructure_failure_share']:.1%}")
    for verdict in verdicts:
        print(f"{verdict['model_label']} p={verdict['disturbance_p']}: "
              f"{'PASS' if verdict['passed'] else 'FAIL'}; success delta={verdict['success_delta_pp']:+.1f}pp")


if __name__ == "__main__":
    main()
