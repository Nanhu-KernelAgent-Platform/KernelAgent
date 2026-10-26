# 设计文档 fix 6-2：LLM 输出瘦身（归类 6・C3）

状态：待审核（未实施）。本文档只覆盖 C3（输出瘦身）；同批的 C1（截断续写）见
`design_kernelagent_fix_6-1_truncation_continuation.md`。两方案独立可回滚，
组合次序：C3 从源头压低单次输出量（少截断），C1 兜住残余截断（截了能救回）。

---

## 1 问题陈述

细化阶段单次响应触发服务端 16384 token 输出上限（fix 6-1 已论证检测与
补救），本方案解决**为什么会写到 16k+**。三段证据：

1. **成功 bundle 只有 4.4k–15.6k token**（P45–P56 批次所有成功四文件
   bundle 的 token 区间，中位数 ~13k）——而截断的那几次响应把预算花在了
   代码之外的东西上。
2. **截断响应的前缀里可见大量非必要内容**：
   - P56 第 5 轮截断响应以约 1500 token 的推理前导开头，之后才开始
     `FILE: kernel.py`；
   - 多个 bundle 的 `binding.cpp`/`kernel.py` 带大段注释与未用到的
     dtype 实例化（float/double 各一份样板，每份约 30 行）。
3. **模板从未告知模型输出预算**：`triton_kernel_agent/templates/musa_kernel_generation.j2`
   （L40-58）与 `musa_kernel_refinement.j2`（L47-51）只规定了文件块格式
   （"Return exactly four fenced FILE blocks and no prose" / "Return no prose"），
   对**篇幅**、**注释量**、**dtype 实例化范围**、**setup.py 长度**零约束。
   模型在不自知预算的前提下自由发挥，是 16k+ 输出的直接成因。

配套的卫生问题：worker.py:606 请求 `max_tokens=20000`，服务端实际上限
16384——请求值与上限不一致；agent.py:566 `max_completion_tokens=20000`、
agent.py:391 `max_tokens=24000` 同类。虽不影响功能（服务端自行触顶），
但让日志读数失真，且若未来服务端放开上限，20000+ 的单次输出会更糟。

## 2 现状定位

### 2.1 两个模板的输出契约现状

**生成模板** `musa_kernel_generation.j2` L40-58：

> Return exactly four fenced FILE blocks and no prose:
> FILE: kernel.py
> ```...```
> FILE: binding.cpp
> ```...```
> FILE: kernel.mu
> FILE: setup.py

只约束**格式**（四个 FILE 块、无散文），不约束**量**。

**细化模板** `musa_kernel_refinement.j2` L47-51：

> Return the complete kernel.py, binding.cpp, kernel.mu, and setup.py
> using fenced FILE blocks in the same order. Return no prose.

同上。且细化阶段模型自带"解释修改思路"的倾向（P56 截断响应的 1500 token
推理前导即为实例），模板没有明令禁止。

### 2.2 单文件的可压缩空间（从成功 bundle 反推）

| 内容 | 现状典型量 | 可压缩到 | 依据 |
|---|---|---|---|
| 推理前导 | 0–1500+ token | 0 | 模板未禁；成功 bundle 中最长的前导也非必要（修复信息在 user 消息里） |
| C++/Python 注释 | 每 100 行代码 20–40 行注释 | ≤5 行/文件 | 注释不影响验证；`validate_musa_bundle_semantics` 不读注释 |
| dtype 实例化 | float32/double 各一份完整样板（~30 行/份） | 只留测试用到的 dtype | fix 4-1 后 `validate_test_autograd` 只跑测试声明的 dtype；多份样板是死代码 |
| setup.py | 30–60 行手写 | ≤20 行规范模板 | fix 2-2 已确定其必需结构（import torch_musa.utils.musa_extension 等），可给模型一份权威范本照抄 |
| kernel.mu | 业务相关 | 业务相关 | 无通用压缩空间 |

四项合计，对 13k 中位数的 bundle 预计可压 2000–4000 token；对正在冲向
16k 的响应，这 2–4k 正是截断与安全落地的差距带。

### 2.3 拆分请求方案的排除论证（选型需要，详见 §3）

若把四文件拆成两次请求（kernel.py+kernel.mu 一次、binding.cpp+setup.py
一次），单次输出减半、结构性防截断。但：

1. **跨请求签名一致性风险**：`bind_kernel_call`（fix 3-1）要求 binding.cpp
   的导出符号与 kernel.py 的调用约定严格一致；两次独立请求间模型可能
   漂移签名，引入一类新的、更难诊断的失败。
2. **失败面翻倍**：每半都有自己的 413/截断/malformed 风险，且两半之间
   无事务性——一半成功一半失败时需要新的一致性协议。
3. **收益已被 C3-1 覆盖**：模板瘦身把 16k+ 场景压回 ≤13k 安全区，
   拆分的边际收益只剩极端场景。

故拆分方案记为**推迟**（若 C3-1+C1 落地后日志仍现截断再启动）。

## 3 方案选型

| 选项 | 做法 | 评估 |
|---|---|---|
| 1 | **模板输出预算硬化**（C3-1，本方案主体） | 在两个模板中加入：显式 token 预算声明、首 token 必须是 `FILE:`、注释上限、dtype 按需实例化、setup.py 规范范本。零代码改动（纯模板），一次修改全局生效 |
| 2 | 拆分请求（C3-2） | §2.3 已论证推迟：签名一致性风险 + 失败面翻倍 + 边际收益小 |
| 3 | 服务端扩容 | 不可控，且与"压输出"正交 |
| 4 | 只做 C1（截断后救） | 不压源头：每次截断仍花 25–30 分钟生成 16k token；C1 续写再花 5–10 分钟。源头压掉则多数截断根本不发生 |

选型结论：**选项 1**，含一小项配套卫生修正（max_tokens 对齐）；选项 2 明确推迟。

## 4 方案设计

### 4.1 C3-1a：生成模板加输出预算段（`musa_kernel_generation.j2`）

在 L40-58 的文件块契约**之前**插入一段预算声明（Jinja 缩进与现有段落对齐）：

```
Output budget: the whole response must stay under 13000 tokens; the
server silently truncates anything longer and the run fails. To fit:

1. The first character of your reply must be "F" of the first "FILE:"
   header. No reasoning, no plan, no summary before or after the blocks.
2. Comments: at most one short header comment per file (<= 5 lines
   total per file). The code itself is the deliverable.
3. Instantiate only the dtypes the test file actually exercises. Do not
   add float64 (or any other dtype) support "just in case" — extra dtype
   instantiations are dead code that burns your output budget.
4. setup.py: copy the canonical template below verbatim, changing only
   the extension name if required. Do not hand-write a longer variant.

Canonical setup.py (<= 20 lines):
--------------------------------------------------
from setuptools import setup
from torch_musa.utils.musa_extension import BuildExtension

setup(
    name="musa_ext",
    ext_modules=[<keep the existing Extension(...) block>],
    cmdclass={"build_ext": BuildExtension},
)
--------------------------------------------------
```

第 4 条的范本嵌入 J2 时用真实围栏（模板变量/字面量按 fix 2-2 确定的
必需结构给出权威版本，实施时从现有成功 bundle 的 setup.py 提炼）。

### 4.2 C3-1b：细化模板加同款预算段 + 禁推理前导（`musa_kernel_refinement.j2`）

在 L47-51 的 "Return the complete ... Return no prose" 处扩充：

```
Return the complete kernel.py, binding.cpp, kernel.mu, and setup.py
using fenced FILE blocks in the same order.

Output budget: under 13000 tokens for the whole reply; the server
truncates beyond that and the run fails.
- Start your reply directly with the first "FILE:" header. Do NOT open
  with an explanation of what you changed — the diff is already
  described in the message above.
- Keep comments minimal (<= 5 lines per file).
- Do not add dtype instantiations that the test file does not exercise.
- Keep setup.py as-is from the previous bundle unless it must change.
```

细化模板比生成模板多两条特殊要求：**禁止"修改思路"前导**（P56 的
1500 token 前导即此形态——修改说明已在 user 消息里给出，模型复述纯属
浪费）；**setup.py 默认沿用上一版**（细化通常不动构建配置，照抄即可）。

### 4.3 C3-1c：max_tokens 卫生对齐

| 位置 | 现值 | 改为 | 理由 |
|---|---|---|---|
| `worker.py:606`（refine `_call_llm`） | 20000 | 16384 | 与服务端实际上限一致；请求值>上限会让 usage 读数与请求意图不符，且服务端放开上限后会真的允许 20k 输出 |
| `agent.py:566`（种子 `max_completion_tokens`） | 20000 | 16384 | 同上 |
| `agent.py:391`（种子生成 `max_tokens`） | 24000 | 16384 | 同上 |

注：fix 6-1 的 `_call_llm_with_usage` 沿用调用方传入的 max_tokens，
本项改的是调用点的字面值，两方案无耦合。

### 4.4 不引入代码强制（设计边界）

预算段是**提示层**约束，不新增响应长度校验代码。理由：超预算的响应
必然走向 malformed → 已有 fix 6-1 的续写/格式修复兜底，再叠加一个
"超 13000 即拒绝"的硬校验只会把可救的响应提前判死。提示层约束 +
C1 兜底的组合已经闭环。

## 5 为什么能解决（按日志回放）

| 原事件 | 预算构成 | fix 6-2 后 |
|---|---|---|
| P56 R5 截断（1500 token 前导 + 四文件写到一半） | 前导 1500 + 注释/dtype 样板 ~2000 + 有效代码 ~13k | 前导归零（C3-1b 首条）+ 样板压缩（dtype 按需）→ 预计 ≤12k，**不触发截断** |
| P49 两次截断（无 FILE 头的代码流写满 16384） | 模型在无预算意识下自由展开 | 预算段 + "首字符必须是 FILE:" 给出强锚点；不能保证根除（模型仍可能无视），但截断概率显著下降；残余截断由 fix 6-1 兜底 |
| P50 R1 截断（`from setuptools import setup` 开头的代码流） | 同上 | 同上；另 setup.py 范本使该文件输出从 30–60 行手写变为 ≤20 行照抄 |
| 成功但偏长的 bundle（13k–15.6k） | 注释与双 dtype 样板 | 压至 ≤13k，给未来更复杂的题目留出余量 |

整体：**触发截断的响应量从"无约束"变为"有 13000 token 目标 + 逐项
压缩指引"**；即使模型部分遵守（例如只做到无前导、未删 dtype 样板），
也足以把多数响应拉回安全区。与 fix 6-1 组合后形成完整防线：
C3 压源头（少截断）→ C1 兜残余（截了能续）。

## 6 文件变化表

| 文件 | 变化 |
|---|---|
| `triton_kernel_agent/templates/musa_kernel_generation.j2` | L40-58 文件块契约前插入 §4.1 预算段（约 25 行） |
| `triton_kernel_agent/templates/musa_kernel_refinement.j2` | L47-51 扩充为 §4.2 版本（约 12 行净增） |
| `triton_kernel_agent/worker.py` | 仅 L606：`max_tokens=20000` → `16384`（一行） |
| `triton_kernel_agent/agent.py` | 仅两处字面值：L566 `20000→16384`、L391 `24000→16384` |
| `tests/test_musa_generation_reliability.py` 或新增小文件 | 模板渲染冒烟测试：渲染两个模板，断言包含预算关键词（"13000"、"FILE:" 首位指令、dtype 按需条目）；max_tokens 断言用例 |
| 不改动 | `kernel_backend.py`、`test_validation.py`（无校验逻辑变化）；provider 层 |

## 7 验证要点

### 7.1 单元测试（全 Mock、无 GPU）

1. **模板渲染冒烟**：用 Jinja 渲染两个模板（给定最小合法上下文变量），
   断言渲染产物含 "13000 tokens"、含 "first character"、含 dtype 按需条目、
   含 canonical setup.py 范本标记；断言 refine 模板含 "Do NOT open with
   an explanation"。
2. **max_tokens 对齐断言**：Mock provider 捕获 `get_response` 的 kwargs，
   断言 worker refine 路径传入 `max_tokens == 16384`；agent 种子路径同理
   （该测试可能需在 `tests/test_musa_seed_generation*.py` 既有用例上改
   断言值——实施时定位）。
3. **回归**：`python -m pytest tests/ -o addopts=""` 全绿。重点回归
   `test_kernel_call_contract.py`（模板变更不得破坏 fix 3-1 的
   bind_kernel_call 契约段——本方案只**插入**预算段，不触碰契约原文）与
   `test_move_leaf_validation.py`。

### 7.2 实机复跑（用户侧验收）

- 重跑 P49/P50/P56（历史截断题），统计细化阶段：
  - 截断发生率（对比历史：P50 4 次、P56 3 次、P49 2 次）；
  - 成功响应的 completion_tokens 分布（期望峰值从 ~13k 移到 ~11k 或更低）；
  - refine 单轮耗时（截断一次 25–30 分钟 → 期望整轮 <15 分钟）。
- 观察 P49 类形态 B 响应（无 FILE 头）：预算段是否把"首字符必须是 FILE:"
  遵守率提上来；仍截断的由 fix 6-1 日志（`refinement_truncated_*.txt`）接管。

## 8 风险边界

### 8.1 提示遵守率不确定

预算段是软约束，模型可能部分或完全无视（尤其 P49 类自由展开型模型）。
缓解：① 首字符锚点（"FILE:" 开头）是最容易被模型遵守的指令形态；
② 无视预算的最坏结果 = 现状（截断 → C1 续写兜底），无新增风险面；
③ 13000 目标留了 3384 token 的缓冲带（服务端 16384），部分遵守即有效。

### 8.2 注释压缩的可维护性权衡

压注释可能让后续人工审阅 bundle 变难。边界：≤5 行/文件 保留文件头
说明（用途、关键约束），不是零注释；且诊断文件（`refinement_rejected_*`、
`refinement_truncated_*`）始终保留完整原文供回溯。

### 8.3 dtype 按需与未来多 dtype 题目

"只实例化测试用到的 dtype"以测试文件声明为准。若某题测试确实需要
float64，模型按需实例化不受影响；风险仅在模型**误判**需求（漏实例化
被测 dtype）→ 由既有验证（运行测试失败 → refine 循环）兜底，不引入
新的静默失败。

### 8.4 明确不做 / 推迟项

- **拆分请求（C3-2）**：推迟，重启条件 = C3-1+C1 落地后实机日志仍出现
  ≥2 次/题的截断（§2.3 排除论证）。
- **响应长度硬校验代码**：不做（§4.4）。
- **13000 数值**：实施时作为模板字面量；若实机数据显示中位成功 bundle
  已 >12k，可下调目标或进一步压缩注释上限——属实施期调参，不动结构。
