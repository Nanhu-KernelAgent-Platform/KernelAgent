# KernelAgent 修复方案 3-1：kernel_function 调用契约统一 —— bind_kernel_call 单一事实源（设计备忘录）

> 状态：**已实施 B1+B2（B3 二期暂缓，待授权后执行）**
> 关联问题归类：**归类 3 —— 测试与 kernel 两个独立 LLM 生成物之间的隐式调用契约漂移**
> （与归类 2 的区别：归类 2 是单个生成物自爆；归类 3 是两个各自合法的生成物互相不匹配）
> 影响问题：KernelBench Level-1 的 **P50**（conv2D square，251min / 227k tokens，失败）、
> **P56**（conv2D asym/asym，292min / 266k tokens，失败）；
> 成功对照 **P55**（conv_transpose2d，最终 kernel 签名
> `def kernel_function(x, weight, stride, padding, dilation, groups):`）
> —— 45–56 题批次中 B 类失败 2/6，约 9 小时 wall / 493k tokens 消耗在
> 「确定性 case suite 已通过、仅主测试因签名猜错而 TypeError」的轮次上。

---

## 1. 问题陈述

P50 与 P56 的失败模式完全一致：**确定性 case suite（companion test）在多轮里全部通过，
kernel 功能上是对的，但 LLM 生成的主测试把 `kernel_function` 的调用签名硬编码错了**，
每轮都在进入数值比较之前抛 `TypeError`：

- P50（session_20261022_210103_783027）：R2/R3 的 case coverage 为
  `[('default', 'passed')]`、`test_extra_1` 通过，但主测试抛
  `TypeError: kernel_function() missing 2 required positional arguments: 'weight' and 'bias'`。
- P56（session_20261023_083607_939604）：seed_0/kernel.py 是 `def kernel_function(x):`，
  主测试硬编码 `kernel_function(x)`；refine 把 kernel 改成 `(x, weight)` 之后，
  R4/R5/R7 主测试反而开始抛 `TypeError: missing 1 required positional argument: 'weight'`
  —— **kernel 越修越对，主测试越错越明显**，而同时刻 case coverage 一直是通过的。
- 对照组 P55 成功，其最终签名是 `def kernel_function(x, weight, stride, padding, dilation, groups):`
  —— 说明正确的契约本来就**允许**（且要求）把模型状态与模块配置声明为显式参数；
  P50/P56 的主测试只是恰好没猜中这个形态。

根因是一个**先有鸡还是先有蛋**的结构性缺陷：`agent.generate_kernel` 的流程是
`_generate_test`（先生成主测试）→ `_generate_kernel_seeds`（后生成 kernel）
（agent.py L723-847）。**主测试生成时 kernel 还不存在**，所以主测试里任何
`kernel_function(x, ...)` 直接调用都是对「一个尚不存在的签名」的猜测。两个独立的
LLM 生成物必须对齐一个没有任何代码强制、且模板文案描述错误的隐式契约——对参数化
Model（conv 有 weight/bias）这类题目，猜错是大概率事件。

> 结论：缺的不是「让 LLM 猜得更准」，而是**让主测试根本不需要猜**。

---

## 2. 现状代码缺口的精确定位

### 2.1 系统里其实已经存在一个统一契约：`bind_kernel_call`（按名绑定）

`kernel_call_binding.py::bind_kernel_call`（L78-122）的语义：

1. 参数**优先按名**绑定到参考 Model 的状态：`named_parameters()`/`named_buffers()`
   的叶名（`weight`、`bias`，别名 `w`/`gamma`→weight、`beta`→bias、
   `module_weight` 式解析），以及常见模块配置属性
   （`stride`、`padding`、`dilation`、`output_padding`、`groups`、`eps`、
   `num_groups`、`normalized_shape`、`kernel_size`）；
2. 其余参数**按位**用 `get_inputs()` 返回的张量依序填充；
3. 带默认值的参数跳过；
4. 都匹配不上的必填参数 → 抛
   `KernelCallBindingError: Cannot bind required kernel parameter(s): <name>; available model values: ...`
   （L116-121，**报错点名缺失参数与全部可用值**，是自带修复指引的失败）；
5. `get_init_inputs()` 的值（如 `dim`）**永远不会**被绑定（docstring L83-84 明确）。

### 2.2 四个消费方，三种约定，只有主测试是「异类」

| 消费方 | 调用方式 | 约定 |
|---|---|---|
| 确定性 case suite | `case_suite.py::verify_cases` L224/L232：`candidate_model = copy.deepcopy(model)` 后 `bind_kernel_call(kernel, candidate_model, candidate_inputs)()` | **按名绑定** ✅ |
| KernelBench 口径评测 | bridge `run_kernelbench_eval` L342-353：把 `kernel_function` 包成 `ModelNew`，`forward` 内 `_KA_BIND(kernel_function, self._ka_state, args)()`（`_KA_BIND` 即 `bind_kernel_call`） | **按名绑定** ✅ |
| **LLM 主测试** | 硬编码 `kernel_function(x, ...)`，签名是生成时的猜测 | **硬编码** ❌ 唯一异类 |
| 模板文案 | 见 2.3，描述的既不是绑定语义、也不适用于参数化 Model | **第三种且错误** ❌ |

即：**评测链路的两个真实消费方早已统一在 `bind_kernel_call` 上**，只有主测试在猜，
而模板文案还在教 LLM 一个错误版本。

### 2.3 模板文案在主动教错（B1 要修的原文）

`musa_kernel_generation.j2` L21-26（`musa_kernel_refinement.j2` L28-34 同文，
仅结尾多一个反例）：

```
- Evaluation call convention: `kernel_function` is called with exactly the
  tensors returned by `get_inputs()`, positionally, in that order. Values from
  `get_init_inputs()` (e.g. `dim`) are fixed at reference-model construction, so
  bake them into kernel.py (module constant, closure, or `autograd.Function`
  argument) instead of adding an extra required parameter — otherwise the
  harness fails with "Cannot bind required kernel parameter(s)".
```

这段话对无参数 Model 成立，但对 conv/norm 类参数化 Model **是错的**：
weight/bias 的值活在 Model 实例里（每次评测还会重置种子重新初始化），
不可能「bake into kernel.py」。它只禁止了 `dim` 型参数，却没说
`weight`/`bias` 型参数**必须**显式声明、以及它们会被按名绑定——
这正是 P50/P56 的 kernel 侧与测试侧各自猜错的直接诱因之一。
（变体模板如 `musa_kernel_generation_reduction.j2` 通过
`{% include "musa_kernel_generation.j2" %}` 复用该段，改一处全家族生效。）

`musa_test_generation.j2` L45-46 只说 "Call kernel_function as a normal Python
function"，L73-76 说 "Pass the same logical inputs to the reference Model and
`kernel_function`" —— **对签名只字未提，等于鼓励硬编码**。而 kernel 种子
prompt 会原样嵌入主测试（`## Tests {{ test_code }}`），主测试的硬编码调用
反过来又把错误约定「教」给 kernel 生成端，形成双向强化。

### 2.4 基础设施已就绪，无需新增运行时依赖

`worker._write_files`（worker.py L386-388）**无条件**把 `kernel_case_suite.py`
和 `kernel_call_binding.py` 都拷进每个测试 workdir，且 `case_suite.py` 顶部
`from .kernel_call_binding import bind_kernel_call` 带 ImportError 回退为绝对导入
——主测试今天就可以 `from kernel_case_suite import bind_kernel_call`，
**零基础设施改动**。

`verify_cases` 还给出了权威的 backward 范式（L224-232 + L238-244）：
先 `copy.deepcopy(model)` 再绑定，原生路径的参数梯度落在副本上，
与参考路径的梯度各归各（`gradient_call` L167-168 采集 `named_parameters`
的梯度做对比）。主测试照抄该范式即可，无需发明。

---

## 3. 方案选型

| 选项 | 思路 | 评估 |
|---|---|---|
| 一：契约前置注入 | 从 Model AST 静态提取「规范签名」，注入两个 prompt | 否。与 `bind_kernel_call` 的绑定逻辑重复实现，必然漂移；状态叶名的静态提取要复刻 `_resolve_state_alias` 等启发式，成本高且脆。作为 B3 的辅助思想保留。 |
| 二：主测试签名无关化（**选定**） | 主测试强制经 `bind_kernel_call` 调用，模板契约文案改写为绑定语义 | 是。单一事实源已存在且 4 个消费方中 3 个在用；主测试不再编码任何签名，「猜签名」这一失败类别被机械性消灭。 |
| 三：仅增强 refine 反馈 | TypeError 时在 refine prompt 里写出正确调用形态 | 否（作为主方案）。被动、每错一次仍烧掉一整轮 build+test+refine（10-30min）。B 落地后该反馈通道自然由 `KernelCallBindingError` 的自解释报错承担。 |

选定：**选项二为核（B1+B2）**，选项三的价值并入 B3（可选二期）。

---

## 4. 方案设计

### 4.1 B1 —— 重写两个 kernel 模板的「Evaluation call convention」

以 `bind_kernel_call` 语义为唯一事实源。`musa_kernel_generation.j2` L21-26 替换为：

```
- Evaluation call convention: every consumer (the generated test, the
  deterministic case suite, and the KernelBench evaluator) invokes
  `kernel_function` through `bind_kernel_call`, which resolves each parameter
  in this priority: (1) BY NAME against the reference Model's state —
  named_parameters()/named_buffers() leaf names (e.g. `weight`, `bias`;
  aliases `w`, `gamma`, `beta`) and common module config attributes
  (`stride`, `padding`, `dilation`, `output_padding`, `groups`, `eps`,
  `num_groups`, `normalized_shape`, `kernel_size`); (2) otherwise POSITIONALLY
  with the tensors returned by `get_inputs()`, in order; (3) parameters with
  defaults are skipped. A required parameter that matches neither source fails
  every consumer with "Cannot bind required kernel parameter(s): <name>".
  Consequences you MUST follow: when the reference Model has
  parameters/buffers (e.g. conv or norm weights), declare them as explicit
  parameters named after the model state, e.g. `def kernel_function(x, weight,
  bias)` — their values live in the model instance and cannot be baked in.
  Values from `get_init_inputs()` (e.g. `dim`) are fixed at reference-model
  construction, so bake them into kernel.py (module constant or closure)
  instead of adding a required parameter.
```

`musa_kernel_refinement.j2` L28-34 同文替换，仅保留其原有的 `dim` 反例并改为
正面引导：

```
  ... instead of adding a required parameter — a signature such as
  `kernel_function(input, dim)` fails with "Cannot bind required kernel
  parameter(s): dim", while `kernel_function(x, weight, bias)` binds weight
  and bias from the reference model by name.
```

要点：
- 保留并强化原第二条规则（`get_init_inputs()` 值不可作为 kernel 参数）——
  这是原文案正确的部分；
- 新增参数化 Model 的**必须显式声明 weight/bias** 规则，把 P55 的成功形态
  写成文档化契约；
- 明确「按名 → 按位 → 默认值」优先级与失败报错形态，让 kernel 侧在生成时
  就知道评测会怎么调它。

### 4.2 B2 —— 主测试强制经 `bind_kernel_call` 调用（模板 + 静态校验器）

#### 4.2.1 模板改动：`musa_test_generation.j2`

替换 L45-46（"Call kernel_function as a normal Python function..."）为：

```
- MANDATORY CALL BINDING: the test is generated BEFORE the kernel exists, so it
  must NEVER hardcode a `kernel_function(...)` call signature. Import
  `bind_kernel_call` from `kernel_case_suite` and route every native call site
  through it, exactly like the deterministic case suite does:
      from kernel_case_suite import bind_kernel_call
      native_model = copy.deepcopy(ref_model)
      invoke_native = bind_kernel_call(kernel_function, native_model, inputs_musa)
      native_out = invoke_native()
  `bind_kernel_call` binds model state (weight/bias and module config such as
  stride/padding/groups) by name and the factory inputs positionally, so the
  same test is correct for every legal kernel signature. Deep-copy the
  reference model before binding so native-path parameter gradients never
  land on the reference parameters (the case suite does the same). The test
  receives the model only to source its state; the module itself is never
  passed into kernel_function. A test that calls `kernel_function` directly
  is rejected by validation. The extension build and launch details belong to
  kernel.py, binding.cpp, kernel.mu, and setup.py.
```

修订 L73-76 末句（消除「Pass the same logical inputs ... to kernel_function」
的硬编码暗示）：

```
- Instantiate `Model(*get_init_inputs())` (or `Model()` only when the factory is
  absent/empty), then move the returned tensor inputs to MUSA without changing
  non-tensor arguments. Pass the same logical inputs to the reference Model
  directly and to `kernel_function` via `bind_kernel_call(kernel_function,
  model_copy, inputs)`; use independently moved tensors when independent
  autograd state is required.
```

其余既有规则不动，且与新写法天然兼容：
- L34-36 `move` 导入与布局保持 —— 新增的 `bind_kernel_call` 与 `move` 同源
  （`kernel_case_suite`），同一 import 来源；
- L85-86 「不得把参考 Model / 预计算期望输出传给 kernel_function」—— 依然成立，
  `bind_kernel_call` 只从 Model **取**状态，绝不把模块本身传入（已在新增文案中说明）；
- L87-96 backward 规则（独立叶子输入、非均匀上游梯度）—— 照抄 `verify_cases`
  的 deepcopy+bind 范式后，参数梯度比较在副本上独立成立，输入叶子仍按 L91-94
  用 `move(..., gradients=True)` 独立构造。

#### 4.2.2 静态校验器：`test_validation.py::validate_test_kernel_call_binding`

与 2-1 规则同风格：纯 AST、无 torch 依赖、报错文本自带修复指引。
语义：

1. **拒绝对 `kernel_function` 的一切直接调用**。主测试先于 kernel 生成，
   直接调用永远是猜测，没有合法例外。需解析：
   - `from kernel import kernel_function`（含 `as` 别名）；
   - `import kernel` / `import kernel as k` 后的 `kernel.kernel_function(...)`
     点链调用（复用 `validate_kernel_globals` 已有的 dotted-name 解析模式）。
2. **测试必须真正行使 kernel**：只要文件引用了 `kernel_function`（任一形式），
   就要求至少出现一次 `bind_kernel_call(...)` 调用（接受
   `from kernel_case_suite import bind_kernel_call`、含别名、
   `kernel_case_suite.bind_kernel_call(...)` 点链），否则是「引用了却不测」的空测试。
   注意 `bind_kernel_call(kernel_function, ...)` 中 `kernel_function` 只是
   **Name 参数**而非 Call，规则 1 不会误伤。

实现骨架（示意，行号以最终代码为准）：

```python
def validate_test_kernel_call_binding(source: str) -> None:
    """The main test is generated before the kernel exists, so it must invoke
    kernel_function through bind_kernel_call instead of guessing a signature."""
    tree = ast.parse(source)
    kernel_names, bind_names = {"kernel_function"}, {"bind_kernel_call"}
    # ... 解析 import/from-import 的模块别名（kernel / kernel_case_suite），同
    #     validate_kernel_globals 的 dotted() 模式 ...
    referenced_kernel = False
    bind_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_or_name(node.func)   # 复用既有别名解析
        if name in kernel_names:            # 含 "kernel.kernel_function" 形态
            raise ValueError(
                "Invalid correctness test: kernel_function must be invoked "
                f"through bind_kernel_call(kernel_function, model_copy, inputs) "
                f"from kernel_case_suite, not called directly at line "
                f"{node.lineno}; the test is generated before the kernel, so a "
                "hardcoded call signature is a guess that cannot be checked. "
                "Repair the test, not the kernel.")
        if name in bind_names:
            bind_calls += 1
        # 引用检测（Name/Attribute 出现即置 referenced_kernel）
    if referenced_kernel and bind_calls == 0:
        raise ValueError(
            "Invalid correctness test: kernel_function is referenced but never "
            "exercised; call it via bind_kernel_call(kernel_function, "
            "model_copy, inputs) from kernel_case_suite.")
```

#### 4.2.3 接线：`agent.py::_generate_test`

挂进现有 MUSA gate（L395-396），**继承既有 2 次尝试修复循环**（L384-429），
无需改重试提示文本 —— 新校验器的 ValueError 本身就是修复指引：

```python
validate_test_autograd(test_code)
if getattr(getattr(self, "_platform_config", None), "name", None) == "musa":
    validate_test_layout(test_code)
    validate_test_kernel_call_binding(test_code)   # ← 新增，MUSA-gated
validate_generated_test_input_contract(problem_description, test_code)
```

MUSA-gated 的理由：B2 的模板强制目前只落在 MUSA 模板上（见 §8 范围），
校验器与模板承诺必须同步，避免 CUDA 侧测试因未获知新规则而被误杀。

**不触碰的部分**：用户提供的测试（`test_1.py` 路径）与 companion suite
（`suite_test` 产物）都不经过 `_generate_test` 校验，不受影响；
kb_eval 无需任何改动（它本来就用 bind，bridge L342-353）。

### 4.3 B3（可选，二期）—— 种子期可绑定性预检

动机：B2 落地后，签名不可绑定的 kernel 会在**运行期**由 case suite 与主测试
共同给出 `KernelCallBindingError`（点名缺失参数与可用值）并进入 refine，
这是可恢复的；但每错一次仍烧掉一整轮 build+test+refine（约 10-30min）。
预检把它提前到种子修复循环内，用既有的一次 LLM 修复换掉一整轮。

设计（挂在 `agent.py::_generate_kernel_seeds` 的
`invalid_musa_responses` / `MUSA_SEED_REPAIR_ATTEMPTS=3` 修复循环上）：

1. 从 kernel.py 的 AST 提取 `kernel_function` 签名，`exec` 一个
   `def kernel_function(<签名>): pass` 桩函数；
2. 参考源码在 **CPU 上** `_exec_source` 后实例化 `Model(*get_init_inputs())`
   （先例：bridge `run_kernelbench_eval` L321-322；companion suite 在 worker
   里同样 exec）；
3. 输入个数**不真跑** `get_inputs()`（可能分配 GB 级张量），而是从其 AST
   return 字面量静态计数；
4. `bind_kernel_call(桩, cpu_model, sentinel_inputs)`：抛错 → 该 seed 记为
   invalid，把 `KernelCallBindingError` 全文喂给既有修复 prompt；
5. **护栏**：预检自身任何异常（torch 不可用、exec 失败、输入个数非字面量）
   一律跳过预检放行 —— 预检只许省时间，不许添堵。

---

## 5. 为什么能解决 P50 / P56

- **失败类别被机械性消灭**：主测试不再编码任何签名。P50 的
  `missing 2 required positional arguments: 'weight' and 'bias'`、
  P56 的 `missing 1 required positional argument: 'weight'` 都源于硬编码调用；
  改为 `bind_kernel_call(kernel_function, model_copy, inputs)` 后，
  「主测试 vs kernel」在签名维度上**不再有可 disagreement 的东西** ——
  主测试、case suite、kb_eval 三个消费方对同一个 kernel 执行同一种绑定。
- **两个证据轮直接反转**：P50 R2/R3 与 P56 R4/R5/R7 都是
  「case coverage 全过、仅主测试 TypeError」—— 在新契约下这些轮的主测试
  与 suite 走同一绑定，这些轮即为主测试通过轮，P50/P56 大概率转为成功
  （残余风险只剩真正的数值错误，属 E 类，且有清晰的报错定位）。
- **kernel 侧契约被教对**：P50/P56 的 kernel 端（`def kernel_function(x):`
  不带 weight 的 conv kernel）同样源于模板只说「get_inputs() 位置传入」；
  B1 文案改写后，参数化 Model 必须显式声明 weight/bias 成为文档化契约，
  P55 的成功形态从「运气好」变成「按规定」。
- **失败变得可自愈**：残余的签名错误（如 `kernel_function(x, dim)`）在三个
  消费方处统一抛 `Cannot bind required kernel parameter(s): dim; available
  model values: ...` —— 点名缺谁、有什么 —— refine prompt 原样带入即可修复，
  而不是今天这种三处报错形态各异、主测试报错还烧掉整轮。
- **成功用例零回归**：P46/P48/P51-53 等无参数 Model 题目按位绑定即可命中全部
  参数，`bind_kernel_call` 行为与原硬编码完全一致；P55 形态被 B1 显式合法化。
  唯一行为变化是主测试多一次 deepcopy（权重是小张量；大张量输入本就独立
  `move`，不参与 deepcopy），`verify_cases` 在 MUSA 上每 case 都这么干，
  成本可忽略。

---

## 6. 关联代码 / 文件变化

| 文件 | 改动 | 阶段 |
|---|---|---|
| `triton_kernel_agent/templates/musa_kernel_generation.j2` | L21-26 「Evaluation call convention」重写为 bind 语义（含参数化 Model 必须显式声明 weight/bias；变体模板经 `{% include %}` 自动继承） | B1 |
| `triton_kernel_agent/templates/musa_kernel_refinement.j2` | L28-34 同步重写（保留 `dim` 反例并补正面示例） | B1 |
| `triton_kernel_agent/templates/musa_test_generation.j2` | L45-46 替换为 MANDATORY CALL BINDING 范式（deepcopy + bind + invoke）；L73-76 末句改为经 bind 传参 | B2 |
| `triton_kernel_agent/test_validation.py` | 新增 `validate_test_kernel_call_binding(source)`（纯 AST，别名/点链解析，拒绝直接调用 + 要求至少一次 bind 调用） | B2 |
| `triton_kernel_agent/agent.py` | `_generate_test` L395-396 的 MUSA gate 内接入新校验器（复用既有 2 次修复循环，重试文案不动） | B2 |
| `tests/test_musa_generation_reliability.py`（或新增 `tests/test_kernel_call_contract.py`） | 回归测试（见 §7） | B1/B2 |
| `tests/test_kernel_call_binding.py` | 扩展：参数化 Model 桩（`x, weight, bias` 可绑 / `x, dim` 报错文案）+ deepcopy 范式用例 | B2 |
| `triton_kernel_agent/agent.py` + `kernel_call_binding.py` | B3 种子期可绑定性预检（exec 桩 + CPU Model + 静态输入计数 + 全量护栏） | B3（二期，可选） |

明确**不改**的：`kernel_call_binding.py` 绑定逻辑本身、`case_suite.py`、
`worker.py`、bridge `run_kernelbench_eval`、用户提供测试路径、companion suite。

---

## 7. 验证要点

单元测试（无 GPU、无 torch 运行时依赖，与既有校验器测试同风格）：

1. **直接调用被拒**：`out = kernel_function(x)` → ValueError，报错含
   `bind_kernel_call` 修复指引与行号；
2. **别名与点链不漏**：`from kernel import kernel_function as kf; kf(x)`、
   `import kernel as k; k.kernel_function(x)` → 均被拒；
3. **合法范式通过**：`deepcopy` + `bind_kernel_call(kernel_function, m, xs)` +
   `invoke()` → 通过；`kernel_case_suite.bind_kernel_call(...)` 点链形式 → 通过；
   `bind_kernel_call(kernel_function, ...)` 中的 kernel_function Name 参数
   **不**触发规则 1（无 Call 节点）；
4. **空测试被拒**：引用 `kernel_function` 但全文无 bind 调用 → ValueError；
5. **模板渲染冒烟**：三个 MUSA 模板经 `PromptManager` 渲染含新文案；
   `musa_kernel_generation_reduction.j2` include 后含新约定段落；
6. **绑定行为**（扩展 `test_kernel_call_binding.py`）：参数化 Model 桩上
   `(x, weight, bias)` 绑定成功；`(x, dim)` 抛
   `Cannot bind required kernel parameter(s): dim; available model values: ...`
   且文案逐字稳定（refine 反馈依赖它）。

真跑验收：

7. 重跑 **P50、P56** 各一次，另取一个对照（如 P52 成功题）：
   - 主测试全程不再出现 kernel_function 签名类 `TypeError`；
   - 每轮主测试结论与 case suite 结论一致（同过或同败于同一数值原因）；
   - P50/P56 达成成功，或失败原因明确属于 E 类真实数值缺陷；
8. 全量 `pytest tests/` 保持绿。

---

## 8. 风险、边界与后续推广

- **LLM 无视强制范式** → 静态校验器在生成期即拒绝，1 次修复尝试内报错文本
  自带修复指引；仍失败则 `invalid_correctness_test` fail-fast，不进 8 轮循环 ——
  严格优于现状（现状是每轮运行期 TypeError 空转）。
- **先 bind 后 deepcopy 的顺序错误**（参数梯度落到参考参数上）→ 模板文案强制
  「Deep-copy ... before binding」；即使写反，梯度对比也会**响亮地失败**而非
  静默通过。
- **deepcopy 成本**：仅复制 Model 权重（小张量）；GB 级输入张量按既有规则
  独立 `move`，不参与 deepcopy。`verify_cases` 每个 case 都在 MUSA 设备上
  deepcopy，性能先例充分。
- **范围**：本期仅 MUSA（45–56 批次证据全部来自 MUSA；校验器与模板承诺
  同步 MUSA-gated）。CUDA 侧 `test_generation.j2` L69 存在同款歧义
  （"passing weights separately only if the kernel API requires it"），
  待 MUSA 验证后作为后续 PR 同步套用 —— `worker._write_files` 对两个后端
  无条件拷贝 `kernel_case_suite.py`/`kernel_call_binding.py`（L386-388），
  推广零基础设施成本。
- **与方案 A（2-1 规则误伤）的关系**：正交。A 修「测试被静态规则误杀」，
  B 修「测试与 kernel 契约漂移」；B 的新校验器规则无歧义可误伤
  （被拒模式的合法替代写法唯一且总是可表达），不会重蹈 2-1 误伤覆辙。
