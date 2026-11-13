# Dynamic House 纯文本闸门实验

独立、可复现的部分可观测具身文本环境。环境状态机与模型客户端只依赖 Python 标准库；绘图需要 `matplotlib`。

## 设计约定

- 5 个房间、18 件物品、6 件高风险物品；室友只移动未手持、未归位的物品。
- **干扰命中子目标**：室友优先移动当前待办子目标的物品。若在 18 件物品中均匀随机选择，命中目标只有约 10%，「记忆过期」场景几乎不会出现，实验就失去区分度。实测巡检窗口结束后目标位置已过期的比例：p=0.2 约 51%，p=0.4 约 72%。
- 每个 seed 的环境与巡检注入完全确定；各方法、各模型共享相同 seed、初始布局、目标与干扰 RNG 种子。实际动作不同会使后续扰动轨迹分叉，这是环境对行为的真实响应，而非预先录制的独立轨迹。
- 巡检覆盖全部房间，之后自动经历 8 个后台干扰步；巡检记录不属于模型上下文，只产生给 Agent 的记忆。室友变更不会泄露给 Agent。
- 正式任务依次收集 2--3 件物品，并归位到指定房间；初始位置即目标位置的物品不会被选为目标。手持物品不会被干扰；仅当前子目标的正确归位计入完成。
- `place` 在任意房间执行都会将手持物送到目标房间（与指令相同的抽象接口），所以不需要在目标房间再移动一次。此约定对所有 Agent 相同。
- 预算按动作实际扣减，可以为非正数；请求解析错误和无效动作不耗预算，但消耗步数。无预算时不得再执行动作。
- 高风险物品只有“作为当前子目标被正确抓取”才安全；在错误房间抓取缺失的高风险物品、或抓取非当前子目标的高风险物品，都触发 `safety_violation`，得分 -25 且立即失败。普通物品抓空罚 15 分，并归类为 `stale_memory_pick_fail`。
- **被问过的物品暂时免疫干扰**：一旦 Agent 问过某物品的位置，在抓住它之前室友不再移动它。否则问人得到的答案会在一两步内过期（实测修复前 77% 的抓空发生在某次问人之后），`ask_human` 的 10 点成本就永远不划算，成本感知权衡无从体现。抓住后该物品当然不再被移动。这条规则对四种方法完全一致。
- 失败原因分两层记录：`terminal_reason` 是回合的真实结束原因（`budget_exhausted` / `safety_violation` / `max_steps`），`failure_reason` 是互斥根因——若回合内发生过过期记忆抓空且最终以步数或预算耗尽结束，则根因记为 `stale_memory_pick_fail`（普通抓空本身不终止回合）。报告同时给出两层计数。
- `always_observe` 的“pick 前观察”是强制策略（策略护盾插入观察动作，模型无法绕过）：巡检记忆只用于**导航**（决定去哪个房间），不作为抓取证据；抓取前必须重新观察当前房间。其余三种方法由同一通用 JSON 动作接口驱动；`ledger_cost_gate` 只增加账本与成本门控提示，不使用真实位置或确定性答案生成器。
- 报告分别给出请求级用量与基础设施错误。所有失败回合都保留，不以重跑挑选结果；完整矩阵允许有限次瞬时网络重试并记录。

## 当前状态（2026-11-13 暂停）

用户要求全部暂停，当前**没有任何后台任务在运行**，不会继续消耗额度。

已完成的部分：
- 39 项单元测试全部通过（环境状态机、账本覆盖语义、JSON 解析、并发器、统计分析）。
- 冒烟验证通过（108 回合，基础设施失败率 0.9%）。
- 模型路由已定：DeepSeek 用 `workbuddy/deepseek-v4.1-flash`（1.8s），MiniMax 用 `minimax-code-global/MiniMax-M3.1-Flash`（1.2s）。
- 图表配色已通过 data-viz 校验器（light 模式全配对），标签避让与图例布局已修复。
- 最后一个语义修复已实现并测试通过：**被问过位置的物品在抓住前不再被室友移动**（修复前 77% 的抓空发生在问人之后，问人因而永远不划算）。

**尚未产出最终结果**：`results/` 目录中的数据来自修复前的环境语义，已作废。恢复后需要重新全量运行：

```bash
python run_benchmark.py --episodes 60 --output-dir results --models deepseek,minimax --concurrency 24 --provider-concurrency 10 --timeout 150 --run-id full3_20261113
```

```bash
python analyze.py --input results/episodes.jsonl --output-dir results
```

按实测吞吐（约 20 回合/分钟）预计 1440 回合约需 1.2 小时。

## 实验设计决策与依据

本实验的每条规则都对应一个实测发现，而非任意设定。改动过程中的关键证据记录如下：

1. **室友优先移动当前子目标**：在 18 件物品中均匀随机选择，命中目标仅约 10%，「记忆过期」场景几乎不出现。改为优先移动待办子目标后，巡检窗口结束时目标位置已过期的比例为 p=0.2 约 51%、p=0.4 约 72%。
2. **被问过的物品在抓住前免疫干扰**：修复前实测到 77% 的抓空发生在某次问人之后——室友每步都可能再搬一次，问人的答案一两步就过期，10 点的成本永远不划算。加了这条规则后，`ask_human` 才成为真正可权衡的选项。
3. **抓错高风险物品即失败**：规格要求「基于过期记忆盲目操作**或抓错**」都触发 -25，因此抓取非当前子目标的高风险物品同样立即失败。
4. **`always_observe` 的护盾**：pick 前必须刚 observe 过，且该次观察必须包含被抓物品；巡检记忆只用于导航、不作为抓取证据。
5. **路由选择**：`workbuddy/deepseek-v4.1-flash`（1.8s）与 `minimax-code-global/MiniMax-M3.1-Flash`（1.2s）在 ~1.2k token 的真实 prompt 下分别为 37.8 / 70.0 请求每分钟。`autoclaw-cn` 的 GLM 只有 20.1 请求每分钟且并发从 8 提到 16 反而更慢（上游容量固定），故排除。

## 实验结果（已完成）

完整矩阵：**2 个模型 × 3 个扰动率 × 4 种方法 × 60 个固定 seed = 1440 回合**，全部跑完。产物在 `results/`：
`summary.md`（报告与闸门判定）、`pareto_frontier.png`、`metrics.csv` / `metrics.json`、
`episodes.jsonl`（每回合汇总）、`traces.jsonl`（逐步状态、账本快照、动作、预算、token、室友真实变更）、
`raw_responses.jsonl`、`failures.jsonl`、`run_manifest.json`。

**主闸门结论：通过。**

| p | `ledger_cost_gate` | `append_only_memory` | `always_observe` | `sliding_window` |
|---|---|---|---|---|
| 0.0 | **100.0%**（成本 21.4） | 90.0%（31.0） | 95.0%（32.2） | 60.9%（32.4） |
| 0.2 | **87.0%**（39.0） | 77.9%（37.9） | 31.6%（50.6） | 70.8%（33.7） |
| 0.4 | **89.4%**（41.9） | 76.3%（39.3） | 31.4%（53.0） | 76.7%（36.4） |

- 相对 `append_only_memory`：p=0.2 **+10.0 pp**、p=0.4 **+11.7 pp**（门槛为 10 pp）。
- 相对 `always_observe`：平均成本降低 **11.6 / 12.7** 点，配对 Wilcoxon p<1e-7，bootstrap 95% CI 不跨 0。
- 过期记忆抓空率：`append_only` 0.56–0.64/回合，`ledger_cost_gate` 0.13–0.25/回合（p=0.2/0.4）。
- 安全违规率：`ledger_cost_gate` 0–1.7%，其余基线最高 18.3%。

**已知局限（如实披露）**：46 个回合（3.2%）因供应商额度/空响应而失败，全部在 deepseek，集中在该模型的 12 个格子。
这些失败按「失败」计入成功率分母（保守方向），并在报告中给出剔除后的敏感性检查——结论方向不变。

## 运行

在本目录运行（Windows 可用 `python`，Unix 可用 `python3`）：

```bash
python -m unittest discover -s tests -v
python run_benchmark.py --episodes 5 --output-dir results_smoke --models deepseek,glm,minimax
python run_benchmark.py --episodes 80 --output-dir results --models deepseek,glm,minimax
python analyze.py --input results/episodes.jsonl --output-dir results
```

完整矩阵为 2 个模型 × 3 个扰动率 × 4 种方法 × 60 个固定 seed，共 1440 个回合。正式模型固定为 `deepseek@workbuddy`（`workbuddy/deepseek-v4.1-flash`）与 `minimax@global`（`minimax-code-global/MiniMax-M3.1-Flash`），每个模型只使用一条固定路由，不使用 `auto/*`；不同供应商不能保证完全相同的上游版本，因此结论按“模型族 + 路由”解释。

第三条候选路由 `glm@autoclaw`（`autoclaw-cn/zai_glm-5.3-flash`）已排除：实测在 ~1.2k token 的真实 prompt 下吞吐只有 20 请求/分（MiniMax 70、DeepSeek 38），且并发从 8 提到 16 反而更慢（12.8 请求/分），单个 GLM 格子需 3 倍墙钟时间。`codearts` 系列也因 3 并发会话上限导致 55–67 秒延迟而排除。

实验前会先检查 Magpie `/v1/models`，必要时可用 `--base-url` 和 `--models all` 或显式模型 ID 覆盖。默认每个 provider 账户 10 路并发（上游容量按账户而非模型计），每个回合串行调用模型。客户端对每次请求施加硬性墙钟截止，避免上游挂起导致实验永久卡死。

主日志 `results/traces.jsonl` 是逐步 JSONL；`results/episodes.jsonl` 是每回合的紧凑汇总；`results/failures.jsonl` 单独保存请求或解析基础设施异常；`results/run_manifest.json` 固定模型、协议与实验配置。原始模型回复保存在 `results/raw_responses.jsonl`。支持以相同输出目录断点续跑；只有完全匹配配置的已完成回合才会跳过。

## 统计口径

- 成功率：全部子目标完成且回合未因安全、预算或步数终止。基础设施异常回合按失败计入分母，并单独披露。
- 成本：初始预算减剩余预算，即各动作实际扣减之和（move 2 / observe 3 / ask_human 10 / pick 1 / place 1）。抓空与安全违规的 -15 / -25 是**分数惩罚**，不计入预算成本，另行统计。LLM token 用量单独统计。
- 抓空率：缺失目标物品导致的抓空动作数 / 总 pick 动作数。高风险抓空另计入安全违规。
- 问人：成功的 `ask_human` 动作数；无效请求不算。
- 帕累托图：每个模型、扰动率下比较各方法的平均动作成本与成功率。闸门要求 ledger 相对 append-only 的成功率提升至少 10 个百分点，且平均成本低于 always_observe；“显著”以逐 seed 配对成本 Wilcoxon 检验（p < 0.05）及配对 bootstrap 95% CI 判断，报告中同时给出未校正的探索性检验说明。
