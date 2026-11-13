请在当前工作区新建一个独立实验目录 `gate_dynamic_house/`，实现并运行我们具身智能课题第一阶段的「动态房间（Dynamic House）纯文本闸门实验」。

### 1. 核心研究假设（闸门验证目标）
验证在**部分可观测、环境会随时间动态变化、且感知与问人均有成本**的具身任务中：
1. 传统「只追加不覆盖的记忆（Append-only Memory）」会因为保留过期状态（Stale Memory）导致大模型在环境变化后大幅掉点；
2. 我们的「版本覆盖式世界状态账本 + 成本感知主动决策门控（Versioned State Ledger + Cost-Aware Gate）」能够在**相信记忆、花低成本重新观察、花高成本问人**之间做最优权衡，在成功率与预算效率的帕累托前沿上显著优于所有基线。

### 2. 环境设计 (`env.py`)
请构建一个纯 Python 确定性可复现（支持固定 random seed）的文本状态机环境：
- **空间拓扑**：5 个连通房间（`LivingRoom`, `Kitchen`, `Bedroom`, `Study`, `Bathroom`），房间之间按图结构连接，`move(target_room)` 每次移动 1 跳消耗 `2` 点预算。
- **物品与风险等级**：18 个日常物品分布在各房间，其中约 30% 标记为 `high_risk`（如 `medicine_box`, `scissors`, `kitchen_knife`, `hot_kettle`）。对 `high_risk` 物品基于过期记忆盲目操作或抓错，会触发安全惩罚（`-25` 分并直接判定任务失败）。
- **部分可观测性**：Agent 仅能通过主动执行 `observe()` 获取当前所在房间的真实物品列表（消耗 `3` 点预算）。
- **隐形室友动态干扰（核心变量）**：Agent 每执行 1 步动作，后台有隐形室友以干扰概率 $p \in \{0.0, 0.2, 0.4\}$ 随机将某个未被手持的物品移动到另一个房间，并在环境日志中记录真实变更轨迹。
- **两阶段回合机制（确保产生“记忆过期”场景）**：
  - **先验巡检阶段**：每个 Episode 开始时，自动生成一段 Agent 早期的历史巡检记录（覆盖各房间初始物品位置，注入到 Agent 的初始记忆中），随后经过一段时间窗口（触发若干次后台干扰，使部分记忆失效）。
  - **正式任务阶段**：给 Agent 下达一个需要依次收集并归位 2~3 个指定物品的多步指令，初始预算为 `60` 点。
- **动作空间与计分规则**：
  - `move(room)`：移动到相邻房间，成本 `2`。
  - `observe()`：检查当前房间并获取最新物品列表，成本 `3`。
  - `ask_human(question)`：向人类询问某物品的精确当前位置（100% 准确），高交互成本 `10`。
  - `pick(item)`：尝试抓取当前房间的 `item`。若物品确实在此处则成功（成本 `1`）；**若物品已被室友挪走（凭过期记忆抓空），重罚 `15` 分并返回缺失提示**。
  - `place(item, room)`：将手持物品放到目标房间完成子目标（成本 `1`）。
  - 预算耗尽（`<= 0`）或超过最大步数（`30` 步）则 Episode 失败。

### 3. 四组对比 Agent (`agents.py`)
所有 Agent 统一封装健壮的 JSON Action 解析器（支持正则提取与格式错误自动重试），直接复用你记忆中已配置好的 Magpie 接口和可用模型（无需省 token）：
1. **`sliding_window`**：仅保留最近 4 步原始交互历史，无长期状态存储。
2. **`append_only_memory`（模拟 Mem0/A-Mem 追加式记忆）**：将历史上所有观察到的“某物在某房间”记录按时间顺序不断追加进 Prompt，**绝不删除或标记旧位置失效**。
3. **`always_observe`（强迫症全查基线）**：不信任任何历史记忆，每次执行 `pick` 前必须先去对应房间执行 `observe()` 确认。
4. **`ledger_cost_gate`（我们的方法）**：
   - **覆盖式世界状态账本**：维护结构化 JSON 账本 `{item: {loc, last_verified_step, status: "active"|"superseded", risk_level, history}}`。每当 `observe()` 发现物品位置变更或发现抓空时，立刻将旧记录标为 `superseded` 并写入新位置与当前步数。
   - **成本感知决策门控**：在每步决策 Prompt 中显式提供当前账本、每条记录的 `memory_age = current_step - last_verified_step`、当前扰动强度先验 $p$、物品 `risk_level` 及剩余预算，引导模型显式权衡：
     - 记忆新鲜且低风险 $\rightarrow$ 直接信任账本去 `pick`（省感知成本）；
     - 记忆较旧（过期概率高）且感知成本远低于问人 $\rightarrow$ 先 `move` + `observe` 低成本核验；
     - 记忆过期且属于 `high_risk` 物品，或剩余预算不足以盲目搜寻 $\rightarrow$ 触发安全门控调用 `ask_human`。

### 4. 实验执行与自动化报告 (`run_benchmark.py` & `analyze.py`)
- **模型与规模**：调用 Magpie 中配置好的模型（模型可以尝试三种吧, magpie里能用的你随便挑），使用异步/多线程并发加速。
- **实验矩阵**：扫描扰动率 $p \in \{0.0, 0.2, 0.4\}$ × 4 种方法 × 每个组合跑 **80 个固定随机种子的 Episode**。
- **全量事件日志**：将每一步的状态、账本快照、动作、预算变化、Token 消耗及失败原因分类（`stale_memory_pick_fail` / `budget_exhausted` / `safety_violation` / `max_steps`）写入 `results/traces.jsonl`。
- **自动出图与闸门结论**：编写 `analyze.py`，自动生成 `results/summary.md` 对比表和 `results/pareto_frontier.png`（横轴为平均总消耗成本，纵轴为任务成功率），统计：
  - 任务成功率（Success Rate %）
  - 平均剩余预算（Avg Remaining Budget）
  - 过期记忆抓空率（Stale-Memory Pick Failure Rate %）
  - 平均问人次数（Avg Human Queries）
  - 并在报告末尾自动给出**闸门判定结论**：在 $p=0.2$ 和 $p=0.4$ 下，`ledger_cost_gate` 的成功率是否比 `append_only_memory` 高出至少 10 个百分点，且平均成本显著低于 `always_observe`。

### 5. 执行步骤
1. 先搭建完整代码并运行环境与账本状态转换的单元测试；
2. 跑一个每组 5 条样本的快速冒烟测试，确认 Magpie 并发调用稳定、JSON 解析无报错；
3. 冒烟通过后，直接全量跑完整个实验矩阵并生成最终的 `results/summary.md` 和图表。