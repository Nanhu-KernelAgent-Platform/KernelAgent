# 设计文档 fix 6-1：LLM 输出截断续写自愈（归类 6・C1）—— v3 修订版

状态：v3 待审核（未实施）。

**v3 相对 v2 的修订**（应用户审核意见收窄改动面）：

1. **删除 `utils/providers/grok_provider.py` 与 `utils/providers/openai_provider.py` 的全部改动**。v2 改名（`get_max_tokens_limit` → `_model_max_tokens_limit`）的唯一目的是让环境变量检查进入基类方法而不被子类重写绕过；v3 改用 **`utils/providers/base.py` 模块级函数 `resolve_output_cap()`** 承载环境变量旁路，子类注册表一行不动；
2. **finish_reason 填充从三处构造点收窄为一处**：只改 `get_response`（`worker._call_llm_detailed` 的唯一调用路径）；`get_multiple_responses`（仅服务 agent 种子路径，6-1 范围外）与 `_get_responses_response`（Responses API，默认未启用）推迟，靠字段默认值 `None`（"未知"≠`'stop'`）保证语义安全；
3. **删除新模块 `triton_kernel_agent/output_budget.py`**。该模块在 v2 中从未包含 6-2 的瘦身内容（无 0.8 预算系数、无提示层规则），只有 cap 解析与阈值派生两个 6-1 自身需要的函数；v3 将 `resolve_output_cap()` 移入 `utils/providers/base.py`（与 `LLMResponse` 同文件），阈值派生内联进 worker.py。

**v2 相对 v1 的修订**（换模型免改代码，保留）：

- 删除 v1 全局常量 `TRUNCATION_COMPLETION_TOKENS = 16000`，改为按模型动态解析输出上限，阈值 = `cap − 512` 随模型伸缩；
- 截断检测升级为双信号：`finish_reason == "length"`（主信号，诚实服务器零配置）OR `completion_tokens ≥ cap−512`（兜底信号，专治 MusaCoder 触顶谎报 `'stop'`）；
- 修正 v1 §7.2 的错误论断：三个既有细化测试必须迁移 Mock（细化路径首次调用从 `_call_llm` 切换到新方法）。

本文档只覆盖 C1。C3（输出瘦身）见 `design_kernelagent_fix_6-2_output_budget_slimming.md`（待 6-1 审核通过后修订，届时**复用**本文的 `resolve_output_cap()` 派生提示预算 `int(cap×0.8)`，不另建解析逻辑）；C2（判死容错阈值）与 C4（现行格式修复，保留为第二道防线）不在本文档范围。

---

## 1 问题陈述

### 1.1 当前模型的截断失败（v1 已论证，保留）

KernelBench L1 批次（P45–P56）四个失败题的细化阶段，全部 10 次 malformed 响应的
`completion_tokens` 恰好等于 16384（MusaCoder-27B 服务端输出上限），且
`finish_reason` 全部谎报为 `'stop'`：

| 日志 | 问题 | 截断次数 | 直接后果 |
|---|---|---|---|
| `agent_20261022_205837.log` | P49 Max reduction | 2 | 第二次截断后 `Malformed LLM kernel response` → 问题判死于 3/8 轮 |
| `agent_20261022_230128.log` | P50 conv2D square | 4 | 最后两次连续截断 → 判死于 5/8 轮 |
| `agent_20261023_062317.log` | P54 conv3D square | 1 | 现行格式修复重试一次成功 |
| `agent_20261023_082923.log` | P56 conv2D asym/asym | 3 | 均被现行重试救回，但每次多花 25–30 分钟 |

伴随事实：截断响应有巨大残值（P56 第 5 轮截断段含完整 `kernel.py` 和大半
`binding.cpp`）；每次截断浪费 25–30 分钟（P49 22:02→23:00 的 58 分钟即两次截断）；
`finish_reason` 对 MusaCoder 不可信，token 数才是可靠判据（但未来诚实服务器上
`finish_reason` 是比阈值更强的判据——见双信号设计）。

### 1.2 设计目标：换模型免改代码（v2 引入，v3 保留）

MusaCoder-27B 之后会更换输出能力更大的模型（如 32k / 128k 输出）。方案必须满足：

- **G1**：换模型时，续写触发阈值、请求 `max_tokens` 随新模型输出上限**自动伸缩**，不改代码（最多加注册表一行或设一个环境变量）；
- **G2**：在诚实报告 `finish_reason=='length'` 的标准服务器（vLLM/SGLang/OpenAI 系）上，检测**零配置**正确，不依赖任何阈值猜测；
- **G3**：在谎报 `'stop'` 的服务器（MusaCoder 型）上，阈值兜底依然有效；
- **G4**：大输出模型下不产生误触发（合法长响应不被误判为截断而浪费续写调用）。

v1 的全局常量 16000 只满足 G3，不满足 G1/G2/G4。

## 2 现状定位

### 2.1 仓库已有"每模型输出上限注册表"，且已在生效（关键发现）

`utils/providers/openai_provider.py:32-43` 已存在 `get_max_tokens_limit(model_name)`：

```python
if "musacoder" in model_name.lower():   return 16384   # ← 当前模型的 16384 登记在此
if model_name.startswith("gpt-5.6"):    return 128000
if model_name.startswith(("gpt-5", "gpt-4", "o3", "o1")): return 32000
if model_name.startswith("gpt-3.5"):    return 16000
else:                                   return 8192    # ← 未登记模型的保守默认
```

provider 层请求构造已用它做钳制（`openai_base.py:216-222`）：

```python
max_tokens_value = min(kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name))
```

两个推论：

- `worker.py:606` 的 `max_tokens=20000` 今天不会到达线上（`min(20000, 16384)=16384`）。但换 128k 模型后 `min(20000, 128000)=20000`，调用点字面量反而变成新模型的紧箍咒——G1 要消灭的一类问题；
- 未登记模型走 else → 8192。若环境变量旁路不进入钳制点，`min(env=32768, 8192)=8192` 会把环境变量在它最该生效的场景（换未登记的新模型）完全击穿——这是 §4.1.3 钳制点改动的直接理由。

### 2.2 usage 与 finish_reason 在 worker 层被丢弃；finish_reason 连 provider 层都未透出

provider 层已返回 usage（`openai_base.py:162-168`；`LLMResponse` 定义于
`base.py:23-31`），但 **`finish_reason` 连 provider 层都没透出**：
`openai_base.py` 的 `LLMResponse(...)` 构造点只取 content/usage。
`worker.py:532-551` `_call_llm` 进一步丢弃 usage，只 `return response.content`。

三处构造点的路径地位（v3 收窄的依据）：

| 构造点 | 位置 | 服务路径 | finish_reason 填充 |
|---|---|---|---|
| `get_response` | L164-169（chat completions） | **`worker._call_llm_detailed` 的唯一调用路径**：细化、续写、格式修复全走这里；MusaCoder 实跑即此路径 | **本方案填充** |
| `get_multiple_responses` | L192-199 | 仅 `agent.py:570` 种子生成使用（6-1 不动种子路径，种子侧已有 3 次完整重输修复） | 推迟（None 兜底） |
| `_get_responses_response` | L138-144 | 仅 `OPENAI_WIRE_API=responses` 启用，默认 chat_completions（`openai_base.py:91`） | 推迟（None 兜底） |

`_call_llm` 的外部消费方：`worker.py:606`（细化路径，本方案改造）与
`opt_worker_component/orchestrator/optimization_orchestrator.py:1049`（**签名必须
保持不变**）。

### 2.3 现行 malformed 处理流程（worker.py:603-646）

```
messages = [user: refine prompt]
for format_attempt in range(2 if musa else 1):        # L605
    response_text = self._call_llm(messages, max_tokens=20000)   # L606
    try:
        refined = extract_kernel_bundle(response_text)            # L607-615
        malformed_reason = self._validate_kernel_candidate(refined)  # L616
    except ValueError as exc:
        refined, malformed_reason = None, str(exc)
    if not malformed_reason: break                                  # L620-621
    写 refinement_rejected_{len(history)}_{format_attempt}.txt      # L623-624
    if 非 musa 或 format_attempt == 1:
        raise ValueError("Malformed LLM kernel response: ...")      # L625-626
    messages += [assistant: response_text, user: "Repair only the
                 complete bundle output format..."]                 # L628-636
```

外层 `except`（L648-651）对含 "Malformed LLM kernel response" 的异常原样上抛，
其余异常落入 mock 细化回退（L654-663）。

### 2.4 截断的两种形态（v1 已论证，保留）

- **形态 A**（P56 第 5 轮）：推理前导 + 完整 `kernel.py` + `binding.cpp` 围栏未闭合
  （断于 `w_stride_cout`），后两文件缺失 → "missing required files"。**续写的目标
  场景**：拼接后四文件齐全。
- **形态 B**（P49 两次、P50 第 1 次）：无任何 `FILE:` 头，16384 token 全部耗在
  无头代码流上 → "no Python kernel code was extracted"。续写难以可靠恢复，但
  兜底逻辑保证不劣于现状（§4.4/§8.2）。

### 2.5 种子侧先例（不在本次范围）

`agent.py:612-670` 已有 `MUSA_SEED_REPAIR_ATTEMPTS = 3` 的有界完整重输式修复。
10 次截断全部发生在细化侧；种子侧是否引入续写见 §8.6 观察项。

## 3 方案选型

| 选项 | 做法 | 评估 |
|---|---|---|
| 1 | 服务端提高输出上限 | 不可控（部署参数），且不解决检测问题，排除 |
| 2 | **动态阈值 + 双信号续写**（本方案） | cap 按模型解析（注册表+环境变量），阈值=cap−512 自动伸缩；`finish_reason=='length'` 主信号在诚实服务器零配置；token 兜底信号覆盖谎报服务器；残值不浪费 |
| 3 | 完整重输式修复（现状，保留为第二道防线） | 每次重输全部四文件（10–16k token、25–30 分钟）；作为续写失败后的退路保留（即 C4） |
| 4 | 拆分请求 | 归入 fix 6-2 论证（推迟） |
| 5 | 静态全局阈值 16000（v1 方案） | 只满足 G3；换 32k/128k 模型后：合法 17k malformed 响应被误判截断、浪费 ≤2 次续写调用（G4 失败）；诚实服务器上本可零配置却仍靠猜（G2 失败）；换模型要改代码（G1 失败）。**被换模型需求否决** |

选型结论：**选项 2**。防线次序：截断续写（新，≤2 次）→ 现行格式修复（保留，1 次）→ raise（现状不变）。

## 4 方案设计

四个子方案：A0 动态输出上限解析；A1 usage/finish_reason 透出与双信号判定；
A2 续写协议；A3 两阶段状态机。

### 4.1 A0：动态输出上限解析（单一事实来源，零子类改动）

**4.1.1 resolver**（`utils/providers/base.py` 模块级函数，与 `LLMResponse` 同文件；
**不新建模块**）：

```python
def resolve_output_cap(model_name: str, provider=None, default: int = 16384) -> int:
    """Effective output-token cap for ``model_name`` (fix 6-1).

    Priority: KERNELAGENT_OUTPUT_CAP env (deployment assertion, zero-code
    model switch) > provider.get_max_tokens_limit registry > default
    (offline/test, provider unavailable).
    """
    override = os.getenv("KERNELAGENT_OUTPUT_CAP")
    if override and override.strip().isdigit() and int(override) > 0:
        return int(override)
    if provider is not None:
        try:
            cap = int(provider.get_max_tokens_limit(model_name))
        except Exception:
            cap = 0
        if cap > 0:
            return cap
    return default
```

要点：

- **子类零改动**：环境变量检查在模块级函数里，不进 `get_max_tokens_limit`
  实例方法，因此不存在"被子类重写绕过"问题——`openai_provider.py` /
  `grok_provider.py` 的注册表原样保留（musacoder→16384、gpt-5.6→128000、
  gpt-5/o 系→32000、grok→128000、未登记→8192）；
- **环境变量语义**：部署级断言，单模型部署形态使用（KernelAgent 每次运行只用
  一个 `--model`）；多模型混跑不要设它，靠注册表；
- **未登记模型**（registry else→8192）：行为与现状一致（保守钳制），可见性由
  worker 启动日志提供（4.1.2），不加注册表 WARNING 代码。

**4.1.2 worker 接线与阈值派生**（worker.py，阈值内联，无独立函数）：

```python
# worker.py 模块级
TRUNCATION_MARGIN_TOKENS = 512   # 吸收服务端上限漂移与 usage 抖动（绝对量，不按比例）

# __init__，provider 初始化（L215-218）之后：
self.output_cap = resolve_output_cap(self.openai_model, self.provider)
self._truncation_threshold = self.output_cap - TRUNCATION_MARGIN_TOKENS
self.logger.info(
    "Output cap resolved for %s: %d (truncation threshold %d; set "
    "KERNELAGENT_OUTPUT_CAP or extend the provider registry if the model "
    "supports more)",
    self.openai_model, self.output_cap, self._truncation_threshold,
)
```

margin 取绝对量 512 的依据：其职责是吸收上限漂移/计数抖动（绝对量），不随 cap
放大；musacoder 下阈值 15872 与 v1 的 16000 行为等价（观测到的成功 bundle 最大
15552，距两者都有余量）。

**4.1.3 钳制点同步**（openai_base.py 两处各一行）：

```python
# _build_api_params L216-218 与 _build_responses_params L115-118：
max_tokens_value = min(
    kwargs.get("max_tokens", 8192),
    resolve_output_cap(model_name, self),   # 原为 self.get_max_tokens_limit(model_name)
)
```

理由（具体的失效链）：未登记的新模型注册表返回 8192；若环境变量不进入钳制点，
则 `min(env=32768, 8192) = 8192`——环境变量恰好在它最该生效的场景（换未登记的
新模型）被完全击穿，请求被钳死 8192，阈值 32256 永远达不到、信号 B 死亡。两处
钳制点与 worker 消费**同一个 resolver**，保证"线上请求上限"与"检测阈值"永远
一致。无环境变量时 `resolve_output_cap` 退化为注册表值，行为与现状完全相同。

### 4.2 A1：usage/finish_reason 透出与双信号判定

**4.2.1 provider 层透出 finish_reason**：

- `base.py:23-31` `LLMResponse` 增加字段 `finish_reason: str | None = None`
  （默认值向后兼容：所有未填充的构造点——含 relay/anthropic 自建响应——语义为
  "未知"，而非 `'stop'`，走信号 B 或无信号路径，无行为空洞）；
- **仅 `openai_base.py` `get_response`（L164-169）一处填充**：
  `finish_reason=getattr(response.choices[0], "finish_reason", None)`。
  这是 `worker._call_llm_detailed` 的唯一调用路径，也是 MusaCoder 实跑路径；
- `get_multiple_responses`（种子路径）与 `_get_responses_response`（Responses
  API，默认未启用）**推迟**，见 §8.6。

**4.2.2 worker 层新方法**（`_call_llm` 保持不动，保住
`optimization_orchestrator.py:1049` 的外部调用与既有 agent 侧测试 Mock）：

```python
# worker.py 模块级
class LLMCallResult(NamedTuple):
    content: str
    completion_tokens: int | None = None
    finish_reason: str | None = None

# worker.py，紧邻 _call_llm（L532-551）之后
def _call_llm_detailed(self, messages: list, **kwargs) -> LLMCallResult:
    """Like _call_llm but also surfaces completion_tokens and finish_reason.

    Both feed the truncation detector (fix 6-1): 'length' is the universal
    cap-hit signal on well-behaved servers, while the token threshold covers
    servers that report 'stop' at the cap (all 10 truncated MusaCoder
    refinement responses in the P45-56 batch did exactly that).
    """
    if not self.provider:
        raise RuntimeError(f"No provider available for model {self.openai_model}")
    if self.high_reasoning_effort:
        kwargs["high_reasoning_effort"] = True
    response = self.provider.get_response(self.openai_model, messages, **kwargs)
    completion_tokens = None
    if isinstance(response.usage, dict):
        raw = response.usage.get("completion_tokens")
        if isinstance(raw, int):
            completion_tokens = raw
    return LLMCallResult(
        content=response.content,
        completion_tokens=completion_tokens,
        finish_reason=getattr(response, "finish_reason", None),
    )
```

**4.2.3 双信号判定谓词**（worker.py 模块级）：

```python
def _looks_truncated(result: LLMCallResult, malformed_reason: str | None,
                     threshold: int) -> bool:
    """True when a malformed response was cut at the model's output cap.

    Signal A (finish_reason == 'length'): universal, zero-config on
    well-behaved servers, fires at any cap size.
    Signal B (completion_tokens >= threshold): fallback for servers that
    report 'stop' at the cap (MusaCoder evidence: 10/10 truncated responses).
    """
    if malformed_reason is None:
        return False          # 只对抽取/校验失败的响应谈续写
    if result.finish_reason == "length":
        return True           # 信号 A
    return (result.completion_tokens is not None
            and result.completion_tokens >= threshold)   # 信号 B
```

设计边界：信号只参与 malformed 响应的路由；抽取成功路径不看信号（至多记一条
`finish_reason=='length'` 的 WARNING 日志，防"恰好解析成功实则静默截断"的罕见
情形逃逸无痕）。

### 4.3 A2：续写协议

**4.3.1 消息与拼接**。检测到截断时，在当前线程上追加（多轮对话语义）：

```python
CONTINUATION_PROMPT = (
    "Your previous reply was cut off at the model's output-token limit. "
    "Continue EXACTLY from the character where it stopped. Do not repeat any "
    "file you already emitted in full, do not restart, and do not add "
    "explanations. Emit only the remainder of the interrupted file block and "
    "then the remaining FILE blocks, ending with the last line of "
    "FILE: setup.py."
)
thread += [
    {"role": "assistant", "content": segment.content},
    {"role": "user", "content": CONTINUATION_PROMPT},
]
segment = self._call_llm_detailed(thread, max_tokens=self.output_cap)
accumulated = accumulated + segment.content     # 原文在前，续写在后
```

四条硬性指令的依据：不重复已完整文件（P56 R5 的 `kernel.py` 已完整）；不重启
（重启=退化成完整重输）；不加解释（防再吃预算）；以 `setup.py` 末行收尾（确定
终止锚点）。抽取对象是**拼接文本**；`extract_kernel_bundle` 对同名文件后见覆盖，
模型小量重复某文件时以续写中的完整版本为准（§8.2 详述误读兜底）。

**4.3.2 线程增长的边界**（与归类 7 的 413 风险交叉）：每次续写追加的 assistant
原文 ≤ cap（16k）；`MAX_CONTINUATIONS = 2` 封顶线程增长 ≤ 2×cap。实证支撑：
P56 R5 现行格式修复已回传过完整 16k 截断原文且请求成功。**格式修复阶段重启
线程**（见 4.4），不携带续写累积文本，避免 45k 级 assistant 回传触发 413。

### 4.4 A3：两阶段状态机（`_refine_kernel` L603-646 重写）

```
base_messages = [{"role": "user", "content": prompt}]

# ── Phase 1：截断续写（两个后端通用；线程随续写延长）──────────────
thread = list(base_messages)
accumulated = ""
segment = self._call_llm_detailed(thread, max_tokens=self.output_cap)
for cont in range(MAX_CONTINUATIONS + 1):        # MAX_CONTINUATIONS = 2 → 3 次判定
    candidate = accumulated + segment.content
    try:
        refined = extract(candidate)             # musa: extract_kernel_bundle
        reason = self._validate_kernel_candidate(refined)
    except ValueError as exc:
        refined, reason = None, str(exc)
    if reason is None:
        return refined                            # 成功（含一次续写恢复）
    if cont == MAX_CONTINUATIONS or not _looks_truncated(segment, reason, self._truncation_threshold):
        last_segment, last_reason = segment.content, reason
        break                                     # → Phase 2
    (self.log_dir / f"refinement_truncated_{len(self.history)}_{cont}.txt"
        ).write_text(segment.content, encoding="utf-8")
    thread += [{"role": "assistant", "content": segment.content},
               {"role": "user", "content": CONTINUATION_PROMPT}]
    accumulated = candidate
    segment = self._call_llm_detailed(thread, max_tokens=self.output_cap)

# ── Phase 2：现行格式修复（仅 MUSA；线程重启，行为与现状一致）──────
(self.log_dir / f"refinement_rejected_{len(self.history)}_0.txt"
    ).write_text(last_segment, encoding="utf-8")          # 阶段边界留档
if self._kernel_backend.name == "musa":
    repair_messages = base_messages + [
        {"role": "assistant", "content": last_segment},
        {"role": "user", "content":
            "Repair only the complete bundle output format. " + last_reason
            + ". Return all four files with literal FILE: kernel.py, FILE: binding.cpp, "
            "FILE: kernel.mu and FILE: setup.py headers immediately before fenced code. "
            "No prose, reference Model, input factories or test code in kernel.py. "
            "Preserve the public contract and do not weaken any test."},
    ]
    segment = self._call_llm_detailed(repair_messages, max_tokens=self.output_cap)
    try:
        refined = extract(segment.content)
        reason = self._validate_kernel_candidate(refined)
    except ValueError as exc:
        refined, reason = None, str(exc)
    if reason is None:
        return refined
    (self.log_dir / f"refinement_rejected_{len(self.history)}_1.txt"
        ).write_text(segment.content, encoding="utf-8")
raise ValueError(f"Malformed LLM kernel response: {reason}")
```

要点：

1. **Phase 2 完整保留现行行为**：消息文本（L630-635 原文照搬）、从原始
   `base_messages` 重启、一次修复、失败即 raise；raise 消息不变，外层
   `except`（L648-651）的 "Malformed LLM kernel response" 识别与 mock 回退
   逻辑均不受影响；保留现行 L643-646 的防御性返回语义。
2. **非 MUSA 后端**：Phase 1 续写同样适用（机制后端无关，triton 路径
   malformed+截断也获得 ≤2 次续写机会——纯增量），Phase 2 保持现状不适用
   （直接 raise），维持现有"修复重试仅 MUSA"的不对称设计。
3. **诊断文件族**：Phase 1 截断段 → `refinement_truncated_{round}_{cont}.txt`；
   Phase 1 收尾 malformed 与 Phase 2 malformed → 沿用
   `refinement_rejected_{round}_{attempt}.txt`（attempt 0/1 与现行编号兼容）。
4. **成功路径零开销**：首答合法 → 1 次调用、无诊断文件、无消息追加，与现状相同。

### 4.5 触发与退出条件汇总

| 条件 | 行为 |
|---|---|
| 抽取/校验成功 | 立即返回（现状同） |
| malformed 且（`finish_reason=='length'` 或 `completion_tokens ≥ cap−512`） | Phase 1 续写，最多 2 次 |
| malformed 但双信号都不满足（如 5000 token 的纯散文响应） | 跳过续写，直入 Phase 2（与现状等价，零浪费） |
| `completion_tokens is None` 且 `finish_reason` 非 'length' | 同上（usage 缺失不猜测） |
| 续写 2 次仍失败 | Phase 2 格式修复 1 次（仅 MUSA） |
| Phase 2 仍失败 | `raise ValueError("Malformed LLM kernel response: ...")`（现状不变；放宽判死属 C2） |

## 5 为什么能解决

### 5.1 按当前批次日志回放（musacoder，cap=16384，阈值 15872）

| 原事件 | 现状行为 | 本方案后 |
|---|---|---|
| P56 R5 截断（形态 A：`kernel.py` 完整、`binding.cpp` 断于 `w_stride_cout`，16384 token + 谎报 'stop'） | 丢弃 ~15k 残值，格式修复完整重输 8711 token 恰好成功（运气） | 信号 B 命中（16384 ≥ 15872）→ 续写只补 `binding.cpp` 尾部 + `kernel.mu` + `setup.py`（约 3–5k token，5–8 分钟），确定性恢复 |
| P49 22:31 截断（形态 B：无 FILE 头 C++ 流，16384） | 格式修复 → 又截断（16384） | 信号 B 命中 → 2 次续写机会；若续写仍无 FILE 头 → Phase 2 重输（不劣于现状） |
| P49 23:00 第二次截断后 | raise → 判死于 3/8 轮（剩 5 轮预算） | 每次截断先获 2 次续写，判死点显著后移 |
| P50 R1 截断（形态 B，16384） | 格式修复成功（15246 token） | 同 P49：先续写、不行退 Phase 2 |
| P54 R2→R3 截断 | 格式修复一次成功（4403 token） | 续写先行，行为不劣 |

收益下界：**所有场景不劣于现状**（续写只是插入的额外机会，Phase 2 与 raise
原样兜底）。收益上界：形态 A 从"25–30 分钟重输 + 再截断风险"变为"5–10 分钟
确定性恢复"；P49/P50 判死点后移。

### 5.2 换模型场景推演（对应 G1–G4）

| 场景 | 行为 | 满足 |
|---|---|---|
| 换 32k 诚实服务器（vLLM 系），响应在 32000 处被切、`finish_reason='length'` | 信号 A 零配置命中 → 续写恢复；阈值 31488 自动就位，无需任何配置 | G1/G2 |
| 同上服务器，一个 17000 token 的 malformed 响应（非截断，`finish_reason='stop'`） | 17000 < 31488 且信号 A 不命中 → 直入 Phase 2，零次浪费性续写 | G4 |
| 换 128k 模型（注册表 gpt-5.6/grok 已有 128000 条目，或设 `KERNELAGENT_OUTPUT_CAP=131072`） | 阈值 127488、请求 max_tokens=131072 自动派生；钳制点与 worker 同源，环境变量对未登记模型同样生效 | G1 |
| 新模型未登记且未设环境变量 | 注册表 else→8192（与现状一致的保守行为）；启动日志 `Output cap resolved for X: 8192` 提示按需配置 | 可见性 |
| MusaCoder 型谎报服务器（触顶报 'stop'） | 信号 B 兜底，行为同 5.1 | G3 |

## 6 文件变化表

| # | 文件 | 变化 | 锚点 |
|---|---|---|---|
| 1 | `utils/providers/base.py` | ① `LLMResponse` 增加 `finish_reason: str \| None = None`（1 行）；② 模块级新增 `resolve_output_cap()`（约 15 行） | L23-31；文件尾部 |
| 2 | `utils/providers/openai_base.py` | ① `get_response` 的 `LLMResponse(...)` 填充 `finish_reason`（1 行）；② 两个钳制点 `min(请求值, self.get_max_tokens_limit(...))` → `min(请求值, resolve_output_cap(model_name, self))`（2 行，理由见 §4.1.3） | L164-169；L115-118；L216-218 |
| 3 | `triton_kernel_agent/worker.py` | ① `__init__` 解析 `self.output_cap`/`self._truncation_threshold` + INFO 日志；② 模块级新增 `LLMCallResult`、`_looks_truncated()`、`CONTINUATION_PROMPT`、`MAX_CONTINUATIONS=2`、`TRUNCATION_MARGIN_TOKENS=512`；③ 新方法 `_call_llm_detailed()`（置于 `_call_llm` 之后，`_call_llm` 本身不改）；④ `_refine_kernel` L603-646 重写为 §4.4 两阶段状态机（调用点 `max_tokens=20000` → `self.output_cap`）；⑤ 诊断文件 `refinement_truncated_*.txt` | L215-218；L532-551；L603-646 |
| 4 | `tests/test_truncation_continuation.py` | **新增**单元测试（§7.1，约 15 个用例） | 新文件 |
| 5 | `tests/test_musa_generation_reliability.py` | **迁移**三个细化测试的 Mock：`w._call_llm = Mock(...)` → `w._call_llm_detailed = Mock(...)`，side_effect 条目从裸字符串改为 `LLMCallResult(text, None, None)`（无信号 → 不续写 → 语义与原断言完全一致，call_count 不变） | L268、L279、L289 |
| — | **明确不改动** | `utils/providers/openai_provider.py`（注册表原样保留，v2 的重命名设计已废弃）；`utils/providers/grok_provider.py`（同上）；`utils/providers/relay_provider.py`（自建 `LLMResponse`，无 usage/finish_reason → 无信号 → 走 Phase 2 = 现状；且它本就不做 max_tokens 钳制，`resolve_output_cap` 对其无影响）；`utils/providers/anthropic_provider.py`；`triton_kernel_agent/agent.py`（种子/测试生成路径，L391/L566 字面值属 6-2；种子侧已有 3 次重输修复）；`optimization_orchestrator.py:1049`（外部消费 `_call_llm`，签名未动）；模板文件（6-2）；`kernel_backend.py`；判死路径（C2） | — |

## 7 验证要点

### 7.1 新增单元测试（`tests/test_truncation_continuation.py`，全 Mock、无 GPU）

沿用 `tests/test_musa_generation_reliability.py` 的 `worker(tmp_path)` /
`bundle_text()` 夹具模式；测试对 `w.output_cap` 直接赋值钉死阈值。

**resolver / 钳制点（A0）**：

1. `resolve_output_cap` 优先级：`KERNELAGENT_OUTPUT_CAP` 环境变量 > 注册表
   （fake provider 返回 32000）> provider 缺省（None → default 16384）；环境
   变量非法值（负数/非数字）回退注册表。
2. 钳制点一致性：`_build_api_params` 无环境变量时 `min(请求值, 注册表值)` 与
   现状一致；设环境变量 32768 + 未登记模型（注册表 8192）→ 请求值为 32768
   （证明旁路未被 8192 击穿）。

**finish_reason 透出（A1 provider 层）**：

3. fake chat response（带 `choices[0].finish_reason='length'`）→
   `get_response` 返回的 `LLMResponse.finish_reason=='length'`；
   `'stop'`/缺失（None）两种形态各断言一次。
4. 推迟项钉死：`get_multiple_responses` 与 Responses API 路径当前不填
   `finish_reason`（断言为 None），使未来任何填充都是显式决策。

**双信号检测（A1 worker 层，cap 参数化跑 16384/32000/128000 三档）**：

5. 信号 B 独立：`(malformed, tokens=cap, finish='stop')` → 续写触发（MusaCoder
   谎报场景，三档 cap 全命中）。
6. 信号 A 独立：`(malformed, tokens=3000, finish='length')` → 续写触发（证明 A
   与阈值无关，任意 cap 零配置）。
7. 双信号皆无：`(malformed, tokens=5000, finish='stop')` → 零续写，直入 Phase 2。
8. usage 缺失：`(malformed, tokens=None, finish=None)` → 同 7。
9. 阈值边界：tokens = cap−512 → 触发；tokens = cap−513 → 不触发。

**续写协议与状态机（A2/A3）**：

10. 形态 A 一次续写恢复：side_effect = `[(前导+kernel.py+半截 binding.cpp, 16384, 'stop'), (binding.cpp 尾部+kernel.mu+setup.py, 5000, 'stop')]` → 断言 bundle 四文件齐全、kernel.py 来自第一段、setup.py 来自第二段、调用数==2、`refinement_truncated_0_0.txt` 落盘且内容为第一段原文。
11. 第二次续写才成功：三段拼接，调用数==3。
12. 续写额度用尽 → Phase 2：两次续写仍 malformed → 格式修复第 4 次调用成功 →
    总调用数==4；修复仍失败 → `raise ValueError("Malformed LLM kernel response")`，
    总调用数==4，`refinement_rejected_*_0/_1.txt` 两份落盘。
13. 形态 B：首答无 FILE 头 + 16384 → 2 次续写均无 FILE 头 → Phase 2（调用数
    1+2+1=4）；Phase 2 修复消息从 `base_messages` 重启（断言消息列表不含
    CONTINUATION_PROMPT 痕迹、assistant 回传为最后一段而非累积文本）。
14. 拼接同名文件后见覆盖：第一段残缺 `setup.py`、续写段完整 `setup.py` →
    bundle 取续写版本。
15. 成功路径零开销：首答合法 → 调用数==1，无 truncated/rejected 文件。
16. 非 MUSA 后端：malformed + 信号命中 → 续写适用；额度用尽 → 无 Phase 2 直接
    raise（调用数==3）。

### 7.2 既有用例回归（含 v1 错误论断的修正）

- **v1 §7.2 曾断言三个细化测试"不受影响"——这是错的**：细化路径首次调用从
  `_call_llm` 切到 `_call_llm_detailed`，`w._call_llm` 的 Mock 拦截不到首答。
  三个用例（`test_musa_refinement_format_repair_once` L265、
  `test_musa_refinement_format_failure_remains_failure` L276、
  `test_unsafe_refinement_does_not_fall_back_to_original` L286）**必须迁移**到
  `_call_llm_detailed`，side_effect 条目改为 `LLMCallResult(text, None, None)`
  （无信号 → 不续写 → Phase 2 路径，断言与 call_count 原样保持）。
- 不受影响的既有 Mock：`agent._call_llm` 系列（test_musa_generation_reliability
  L115/224/237、test_prompt_enrichment、test_kernel_call_contract、
  test_joint_generation_memory——都是另一个类的方法）；worker_util 自由函数
  `_call_llm`；orchestrator 对 `worker._call_llm` 的调用（签名未动）。
- 既有 provider 测试（如 `test_openai_responses_provider.py` 断言
  `max_output_tokens==128`）：无环境变量时 `resolve_output_cap` 退化为注册表
  值，行为不变，预期原样通过。
- 全套 `python -m pytest tests/ -o addopts=""` 保持绿。

### 7.3 实机复跑（用户侧验收）

- 重跑 P49 或 P50：启动日志出现 `Output cap resolved for <model>: 16384`；截断
  时出现 CONTINUATION_PROMPT（"cut off at the model's output-token limit"）与
  `refinement_truncated_*.txt`；该轮 refine 总耗时显著低于 25 分钟；
- 对比指标：单题细化阶段 LLM 调用次数、截断浪费分钟数、判死轮次。

## 8 风险边界

### 8.1 与 C2/C4 的组合次序

落地后单轮细化防线：**截断续写（新，≤2 次）→ 格式修复（现行 C4，1 次，仅
MUSA）→ raise（现行）**。C2（连续 malformed 阈值放宽）是后续独立方案；本方案
不放宽判死，保证单独回滚无行为空洞。

### 8.2 续写指令被误解的兜底

模型无视"不重启"而重输全部四文件：拼接文本 = 残缺前段 + 完整后段，
`extract_kernel_bundle` 后见覆盖使同名文件取后者，多数情况仍抽出正确 bundle；
另一误读方向（只补缺文件）正是期望行为。两种方向都被抽取器语义兜住，最坏退
化为 Phase 2。形态 B（无 FILE 头）续写成功率天然低——设计上不预先区分形态
（§4.2.3 不引入文本启发式），靠"续写失败 → Phase 2"自然兜底，最多浪费 2 次
续写调用（约 10–20 分钟）换形态 A 的确定性恢复。

### 8.3 续写线程增长与 413 的交互（归类 7 交叉）

每次续写追加 ≤cap 的 assistant 原文，`MAX_CONTINUATIONS=2` 封顶增长 ≤2×cap。
实证：P56 R5 现行修复已成功回传 16k 截断原文。风险残留：基础细化 prompt 本身
已接近上下文上限的题（P54：66927 token → 413），续写请求可能同样 413 →
外层 except → mock 回退（轮次推进、kernel 不变）——与现状同级的不理想结果，
不新增崩溃面。根治属归类 7（D 系列 prompt 预算），本方案通过 Phase 2 重启
线程保证不放大该风险。

### 8.4 finish_reason 的可用性光谱

| 服务器行为 | 信号 A | 信号 B | 结果 |
|---|---|---|---|
| 诚实（vLLM/OpenAI 系） | ✅ 'length' | 冗余 | 零配置精确检测 |
| 谎报（MusaCoder：触顶报 'stop'） | ✗ | ✅ 阈值兜底 | 等效 v1 行为 |
| 不返回 finish_reason（None） | ✗ | ✅ | 等效 v1 行为 |
| usage 与 finish_reason 都缺（如 relay 路径） | ✗ | ✗ | 不续写 → Phase 2 = 现状 |

任何组合都不劣于现状。anthropic 的 `stop_reason=='max_tokens'` → 'length'
映射列为可选后续（观察项），不阻塞本方案。

### 8.5 环境变量误配

- 设得比真实上限高：信号 B 阈值过高 → 谎报服务器上漏检（退化为现状的
  Phase 2 路径）；诚实服务器仍有信号 A 兜住。
- 设得比真实上限低：误触发续写（≤2 次，有界浪费），Phase 2 兜底。
- 多模型混跑：单值环境变量只对其中一个模型正确——文档化为"单模型部署专用"，
  多模型场景用注册表。

### 8.6 明确不做 / 观察项

- **`get_multiple_responses` / `_get_responses_response` 的 finish_reason 填充**：
  前者仅服务种子路径（6-1 范围外，种子侧已有完整重输修复），后者默认未启用。
  字段默认 None 语义安全；§7.1 第 4 条测试钉死推迟状态，未来填充是显式决策；
- **种子生成侧续写**（agent.py 已有 3 次完整重输修复）：10 次截断全部在细化
  侧；若后续日志出现种子侧频发截断，再评估复用 `_looks_truncated` + 续写协议；
- **Phase 2 修复响应的续写资格**：修复响应若也截断（P49 场景）直接 raise，
  不递归续写——保持状态机线性可审；C2 若放宽判死可一并考虑；
- **agent.py 的 `max_tokens` 字面值**（L391=24000、L566=20000）：属 fix 6-2 的
  C3-1c 动态化范围（与提示预算同批），本方案不动；
- **margin=512 / `MAX_CONTINUATIONS=2` / 环境变量名 `KERNELAGENT_OUTPUT_CAP`**：
  实施期可调常量，结构不受影响。
