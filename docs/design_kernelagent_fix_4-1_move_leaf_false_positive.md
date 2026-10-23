# KernelAgent 修复方案 4-1：2-1 规则对 move() 派生叶子的误伤 —— 让静态规则认识自家搬运语义（设计备忘录）

> 状态：**已实施 A1+A2+A3（含实施修订：`move_tensors` 两步式容器访问追踪，
> 见 §4.1.3）**
> 关联问题归类：**归类 4 —— 静态校验规则自身的假阳性（修复 2-1 引入的回归）**
> （与归类 2 的区别：归类 2 是 LLM 生成物自爆、校验器正确拦截；归类 4 是生成物本身
> 运行时合法、**校验器判死**——校验基础设施自己成了失败原因）
> 影响问题：KernelBench Level-1 的 **P45**（avg_pool2d，日志
> `agent_20261022_174918.log`）与 **P47**（日志 `agent_20261022_191431.log`）。
> 两题的共同点：模型的**全部修复尝试都产出了运行时合法的代码**，却全部被
> 修复 2-1 引入的静态规则拒绝，问题死在「测试生成阶段」——尚未进入 kernel
> 生成与验证轮次。45–56 题批次中该类失败 2/12。
> 前置依赖：无。与方案 B（fix 3-1，已实施 B1+B2）共享 `_generate_test`
> 修复循环但互相独立，可单独实施。

---

## 1. 问题陈述

### 1.1 P45：两次尝试、两种合法写法、两次被拒

`agent_20261022_174918.log`（avg_pool2d，测试生成阶段）：

**尝试 1**（18:01:38，7651 completion tokens）构造反向独立叶子：

```python
x_ref = move(x.detach(), 'musa', gradients=True)
...
x_ref.requires_grad = True      # ← 第 90 行被拒
```

> `WARNING - Generated test failed validity/contract checks; retrying once:
> Invalid correctness test: only a fresh leaf tensor (e.g. torch.randn(...),
> or a factory with requires_grad=True) may have .requires_grad set to True,
> not a derived tensor at line 90.`

**尝试 2**（18:05:24，修复重试）换了一种同样合法的写法——直接从工厂重新取输入再搬运：

```python
inputs_ref = move(get_inputs(), 'musa')
x_ref = inputs_ref[0]
x_ref.requires_grad = True      # ← 第 84 行再次被拒
```

> `ERROR - Error generating test with LLM API: Invalid correctness test:
> ... not a derived tensor at line 84.`

问题整题死亡。**注意**：尝试 2 的「修复」方向其实完全正确——两种写法在运行时
都是合法的（见 §2.1），只是静态规则不认识。

### 1.2 P47：唯一有效的一次尝试被拒

`agent_20261022_191431.log`：

- **尝试 1**（19:41:56）：16k token 的推理循环、无代码块 →
  `No valid code found in LLM response`（这是合法拦截，无争议）。
- **尝试 2**（19:56:24）：

```python
x_leaf = move(x.detach(), 'musa', gradients=True)
x_leaf.requires_grad = True     # ← 第 78 行被拒
```

> `ERROR - Error generating test with LLM API: ... not a derived tensor
> at line 78.`

问题整题死亡。

### 1.3 三个要点

1. **这不是模型能力问题**：move() 的返回值在运行时**必然是全新叶子张量**
   （§2.1），`X.requires_grad = True` 合法。被拒的代码如果进入运行时会
   正常工作。
2. **修复提示在把模型往沟里带**：报错文案（也是修复提示的正文）建议
   「e.g. torch.randn(...)」。但模型此时需要的不是「重新随机一组输入」，
   而是「保值的独立叶子」——torch.randn 会**改变输入取值来源**，模型正确地
   拒绝了这个建议、继续用 move，然后再次被同一规则判死。
   **提示词与规则互相矛盾，修复循环在原地打转。**
3. **只有一次重试**：`_generate_test` 共 2 次机会（`agent.py` L388
   `for attempt in range(2)`），两次都撞在同一堵墙上后直接抛异常、整题失败。
   P45 耗尽两次，P47 一次浪费在无代码块响应上、仅剩的一次也被误杀。

---

## 2. 现状代码缺口的精确定位

### 2.1 move() 的运行时语义：**全新叶子**（case_suite.py L122-137）

```python
def move(value, device, *, gradients=False):
    if isinstance(value, torch.Tensor):
        moved = torch.empty_strided(value.shape, value.stride(),
                                    dtype=value.dtype, device=device)
        with torch.no_grad():
            moved.copy_(value)
        moved.requires_grad_(value.requires_grad or (gradients and (...)))
        return moved
    if isinstance(value, (list, tuple)):   # dict 同理，递归映射
        return type(value)(move(item, device, gradients=gradients) for item in value)
```

关键事实：

- `torch.empty_strided(...)` 是**全新分配**，没有历史图；
- `copy_` 在 `no_grad` 下执行，梯度**不**穿过拷贝回溯到源张量；
- `requires_grad_(...)` 在无历史的张量上是合法操作，结果**就是叶子**。

所以：**move() 的返回值、以及 move(list/tuple) 的每个元素，运行时必然是
全新叶子张量**，且是独立求导根（backward 停在这里，.grad 落在这里）。
这正是测试模板（musa_test_generation.j2，fix 3-1 B2 已强化）强制要求的
「independent stride-preserving leaf」的实现载体。

### 2.2 2-1 规则只认 torch 工厂（test_validation.py）

`validate_test_autograd` 中的叶子白名单（L120-125）：

```python
FACTORY_CALLS = {
    "torch.randn", "torch.rand", "torch.randn_like", "torch.rand_like",
    "torch.zeros", "torch.ones", "torch.empty", "torch.full",
    "torch.zeros_like", "torch.ones_like", "torch.empty_like",
    "torch.arange", "torch.tensor",
}
```

叶子追踪（L216-221）：赋值目标是 `ast.Name` 且值是「工厂调用」才进
`leaf_tensors`，**其他任何赋值都将其移出**：

```python
for t in targets:
    if isinstance(t, ast.Name):
        if factory:
            leaf_tensors.add(t.id)
        else:
            leaf_tensors.discard(t.id)
```

两个撞墙的形态都不满足 `factory`：

| 写法 | AST 形态 | 判定 |
|---|---|---|
| `x_ref = move(x.detach(), 'musa', gradients=True)` | `Call(func=Name('move'))`，`move` 不在 FACTORY_CALLS | 非 factory → 移出 leaf_tensors |
| `inputs = move(get_inputs(), 'musa')` → `x_ref = inputs[0]` | 先是 `Call(move)`，后是 `Subscript` | 两次赋值都非 factory → 移出 |

随后 `x_ref.requires_grad = True`（L172-186）命中 `t.value.id not in
leaf_tensors` → 拒绝。

另一个缺口：`validate_test_autograd` 的 `dotted()`（L97-102）**不做 import
别名解析**（`from kernel_case_suite import move as mv` 或
`import kernel_case_suite as kcs; kcs.move(...)` 无法归一），
即使把 `move` 加进集合也会漏掉别名形态。

### 2.3 修复循环与修复提示（agent.py L388-436）

- 共 2 次尝试；第 2 次失败即 `raise contract_error`（L415-416），整题失败。
- 修复提示（L427-436）只有原则句「recompute the reference with grad enabled
  and independent leaf inputs; do not differentiate a no_grad result」，
  **没有给出合法叶子的具体构造范式**；具体建议全部来自报错文案，而文案
  正在教错（§1.3 第 2 点）。

### 2.4 2-1 规则的真阳性必须保住

修复 2-1 的初衷（见 design_kernelagent_fix_2-1_test_leaf_requires_grad.md）
是拦截 P18 那种「派生张量上设 requires_grad → RuntimeError → 8 轮空转」。
以下形态**必须继续拒绝**：

```python
x = y.clone()          # 克隆：非叶子
x.requires_grad = True # 运行时 RuntimeError: only leaf tensor...
x = y + 1              # 算术派生：非叶子
x.requires_grad = True
```

所以方向不是「放宽规则」而是「补全规则对 move 语义的认识」——move 是
**仓库自家代码**，其叶子性可以静态确证，不是启发式猜测。

---

## 3. 方案选型

| 选项 | 思路 | 评估 |
|---|---|---|
| 1. 删除 2-1 规则 | 回到纯运行时发现 | ❌ P18 的 8 轮空转回归，且 45–56 批次无此类失败，纯倒退 |
| 2. **把 move 的语义教给规则（本方案 A1）** | 白名单精确扩展：解析 import 别名后识别 `kernel_case_suite.move` 调用及其下标链；同时修修复提示（A2）与重试次数（A3） | ✅ 假阴性风险≈0（move 实现是仓库内固定代码，语义可静态确证）；真阳性完全保留 |
| 3. 运行时试探 | 测试生成阶段起个 torch 会话试跑 | ❌ 生成环境不保证有加速卡；CPU 试探语义不等价；违背「静态优先」原则 |

选型 2。三个子项 A1（规则补全）/ A2（修复提示给出具体范式）/ A3（第三次尝试）
独立成章，可分别审阅；A1 是核心，A2/A3 是廉价加固。

---

## 4. 方案设计

### 4.1 A1 —— 2-1 规则识别 move() 派生叶子（test_validation.py）

#### 4.1.1 import 别名解析（新增，镜像既有模式）

在 `validate_test_autograd` 开头构建别名表，模式与
`validate_kernel_globals`（L28-34）和 `validate_test_kernel_call_binding`
（fix 3-1）完全一致：

- `from kernel_case_suite import move` → `move → kernel_case_suite.move`
- `from kernel_case_suite import move as mv` → `mv → kernel_case_suite.move`
- `import kernel_case_suite as kcs` → `kcs → kernel_case_suite`（配合
  dotted 解析 `kcs.move(...)` → `kernel_case_suite.move`）
- `import kernel_case_suite` + `kernel_case_suite.move(...)`：现有 dotted
  天然可解析。

`dotted()` 保持不变，仅在 Name 分支查别名表（与 validate_kernel_globals
的 dotted 一致化）。

#### 4.1.2 本地遮蔽守卫（防假阴性）

若测试**本地重定义**了同名函数（`def move(...)` 或 `move = ...`），运行时它
就不再是 case_suite 的 move，不能再按「必为叶子」处理。实现：收集全树中
`FunctionDef.name` 与赋值目标 `Name` 构成 `rebound` 集合；别名键落在
`rebound` 中即从别名表剔除（保守回落为未知函数 → 不视为叶子来源）。
同时扫描**点链属性赋值**：`kernel_case_suite.move = ...` 或
`kcs.move = ...`（对导入模块别名的属性重绑定）同样视为遮蔽，整棵树不再
把任何调用识别为 suite 的 move。这堵住「自定义/篡改 move 返回派生张量」
绕过规则的两条口子（注意：`validate_kernel_globals` 的 framework-patch
保护只覆盖 torch/torch_musa 且只跑在 kernel.py 上，测试侧没有这层保护，
必须在守卫里自理）。

#### 4.1.3 `is_move_leaf(value)` 谓词（实施修订：补两步式容器访问）

设计稿伪代码只覆盖「直接下标链」`move(...)[0]`；实施时发现 P45 尝试 2 的
真实形态是**两步式**：`inputs_ref = move(...)` 之后 `x_ref = inputs_ref[0]`
（§2.2 表格第二行的形态）。仅识别直接链会漏掉它。最终实现增加一个与
`leaf_tensors` 同样按作用域线程化追踪的名字集合 **`move_tensors`**
（最新赋值为 move 派生值的名字），下标取自该集合的元素同样视为全新叶子：

```python
def move_leaf(value, move_tensors):
    if move_shadowed:
        return False
    while isinstance(value, ast.Subscript):
        value = value.value
        if isinstance(value, ast.Name) and value.id in move_tensors:
            return True          # inputs_ref = move(...); x = inputs_ref[0]
    if not (isinstance(value, ast.Call)
            and dotted(value.func) == "kernel_case_suite.move"):
        return False
    base = value.func
    while isinstance(base, ast.Attribute):
        base = base.value
    return not (isinstance(base, ast.Name) and base.id in rebound)
```

- ✅ `move(x.detach(), 'musa', gradients=True)` → Call → 叶子
- ✅ `move(get_inputs(), 'musa')[0]` → Subscript(Call) → 叶子
- ✅ 深层嵌套 `move(get_inputs(), 'musa')[0][1]` → 叶子（递归映射语义）
- ✅ **两步式** `inputs_ref = move(...); x_ref = inputs_ref[0]` →
  `move_tensors` 命中 → 叶子（P45 尝试 2 的真实形态）
- ❌ `move(...)[0].clone()` → Attribute 包裹，不匹配 → 非叶子（链在方法
  调用处正确截断，派生张量继续被拒）
- ❌ 别名被遮蔽后的本地 `move(...)` → 别名表无此键 → 非叶子

`move_tensors` 仅追踪 move 派生值，**不**扩展到 torch 工厂：对张量工厂
`x = torch.randn(3)` 而言 `x[0]` 是视图，其叶子性取决于 `x.requires_grad`
（基张量需要梯度时视图必为非叶子），语义与 move 的容器映射不同，且无失败
证据，不动。

#### 4.1.4 接入点一：叶子追踪（两处集合）

`leaf_tensors` 之外同步维护 `move_tensors`（工厂值从 `move_tensors` 移出，
move 派生值同时进入两个集合，其他赋值双双移出）；分支拷贝
（if/else/try-except）与嵌套作用域对两个集合做同样的线程化处理。

```python
moved = move_leaf(value, move_tensors)
...
for t in targets:
    if isinstance(t, ast.Name):
        if factory:
            leaf_tensors.add(t.id); move_tensors.discard(t.id)
        elif moved:
            leaf_tensors.add(t.id); move_tensors.add(t.id)
        else:
            leaf_tensors.discard(t.id); move_tensors.discard(t.id)
```

#### 4.1.5 接入点二：blocked/live 语义（move 结果是独立求导根）

blocked/live 判定（L199-213）中把 `moved` 与 `factory` 同等对待：

- move 结果**永不继承 blocked**：即使源值来自 no_grad 块，拷贝在新张量上
  完成、梯度不回溯——它是一个全新的求导根，对它 backward 合法。
  这正是测试语义想要的「independent leaf」。
- live 判定保持自然行为：`x = move(...)` 是 Call 形态，走既有
  「Call 且非 factory → live」分支（新张量可作为求导根，标记 live 正确）；
  `x = inputs[0]` 下标形态不标 live，但也**不 blocked**——后续
  `requires_grad = True`（叶子追踪放行）+ backward（目标不 blocked）均通过。

#### 4.1.6 报错文案修正（L179-186）

合法示例中加入 move 与方法形态，替换误导性的「e.g. torch.randn(...)」排他建议：

> only a fresh leaf tensor (a torch factory result, **or a
> `move(...)` result from kernel_case_suite, which always returns fresh
> leaf tensors**) may have `.requires_grad` set to True, not a derived
> tensor at line N. A value such as `.clone()` is not a leaf and fails at
> runtime; to enable grad on an existing fresh leaf, call
> `.requires_grad_()` instead of assigning the attribute.
> Repair the test, not the kernel.

文案是修复循环的实际载荷，这一处同时服务 A2。

### 4.2 A2 —— 修复提示给出具体合法范式（agent.py L427-436）

修复消息在既有原则句后**追加一句具体范式**（平台中立措辞，规则与提示均
不引入 MUSA gate——2-1 规则本来就是全平台的）：

> Build independent leaf inputs with
> `move(tensor.detach(), device, gradients=True)` from kernel_case_suite
> — the result is already a fresh leaf, so never assign `.requires_grad`
> afterwards; call `.requires_grad_()` on a fresh leaf instead if needed.

与 fix 3-1 B2 模板中的既有指导（「use `move(tensor.detach(), 'musa',
gradients=True)` for an independent stride-preserving leaf」）形成闭环：
模板教、校验器认、修复提示兜底，三处说的是同一句话。

### 4.3 A3 —— 第三次生成尝试（agent.py L388, L415-421）

- `for attempt in range(2)` → `range(3)`；`if attempt == 1` → `if attempt == 2`；
- 日志文案 `"retrying once: %s"` → 带计次（`"retrying (attempt %d): %s"`）。

**成本分析**：第 3 次尝试只在「前两次都失败」的分支发生——当前代码在该
分支直接抛异常、**整题死亡**。用一次测试生成调用（正常响应实测约 2–4 分钟，
P47 的 16k-token 畸形长推理响应偏上限约 15 分钟；token 上限 24k）换整题存活
的概率，期望值严格为正（对照 kernel 验证轮次 25–30 分钟/轮、整题 3–5 小时
的沉没成本）；正常路径（一次通过）零额外成本。P47 的失败模式
（第 1 次浪费在无代码块响应上）正是该余量兜底的对象。

**不触碰的部分**：

- kernel 生成/精修的重试结构不在本期范围（那是 worker 的独立循环）；
- `x = y.detach(); x.requires_grad = True`（detach 结果运行时也是叶子，
  规则同样误伤）——45–56 批次**无证据触发**，且 B2 模板已强制 move 范式
  引导生成方向，列为观察项（§8），不带证据不动规则。

---

## 5. 为什么能解决 P45 / P47

**P45 路径回放**：

| 步骤 | 现状 | 修复后 |
|---|---|---|
| 尝试 1 `move(x.detach(),'musa',gradients=True)` + `requires_grad=True` | 2-1 拒绝（L90） | `moved=True` → leaf_tensors 含 x_ref → **通过** |
| 若模型仍先产出别的形态被拒 | 修复提示教 torch.randn，误导 | 修复提示给出 move 范式（A2），模型按模板已有习惯（B2 已教）收敛 |
| 尝试 2 `move(get_inputs(),'musa')[0]` + `requires_grad=True` | 再次拒绝 → 整题死亡 | Subscript 链识别 → **通过**；即便前两次全废还有第 3 次（A3） |
| 测试通过后 | （未到达） | 进入 kernel 生成与验证轮次，问题获得正常求解机会 |

**P47 路径回放**：尝试 1 无代码块（合法拦截）→ 尝试 2 的 move 写法被
A1 放行 → 问题存活；即使再废一次，A3 还有第三次。

**真阳性不回潮**：`y.clone()` / 算术派生 / 方法链派生（§4.1.3 的截断规则）
全部继续被拒——P18 的保护完整保留（§7 回归测试 5–7 项锁定）。

---

## 6. 关联代码 / 文件变化

| 文件 | 改动 | 性质 |
|---|---|---|
| `triton_kernel_agent/test_validation.py` | validate_test_autograd：新增 import 别名解析 + rebound 遮蔽守卫 + `is_move_leaf()` 谓词；叶子追踪与 blocked/live 两处接入；报错文案加入 move/`.requires_grad_()` 指引 | 纯静态规则扩展，无运行时依赖 |
| `triton_kernel_agent/agent.py` | `_generate_test`：`range(2)`→`range(3)`、末次判定与日志计次、修复消息追加 move 具体范式一句 | 消息与循环参数 |
| `tests/test_move_leaf_validation.py`（新） | §7 的 1–12 项单元测试（纯 AST，无 torch） | 新增 |
| `tests/test_kernel_call_contract.py` | 追加一项：`_generate_test` 三次修复循环（2 废 1 好）成功返回 | 扩展 |

无模板改动（B2 已把 move 范式写进 musa_test_generation.j2）；无 worker /
bridge / case_suite.py 改动。

---

## 7. 验证要点

单元测试（纯 AST，无 torch；新文件 `tests/test_move_leaf_validation.py`）：

1. **P45 尝试 1 形态通过**：`x = move(y.detach(), 'musa', gradients=True);
   x.requires_grad = True` 不抛。
2. **P45 尝试 2 形态通过**：`inputs = move(get_inputs(), 'musa');
   x = inputs[0]; x.requires_grad = True` 不抛。
3. **P47 形态通过**：同 1。
4. 别名与点链全覆盖：`import move as mv` / `import kernel_case_suite as
   kcs; kcs.move(...)` / `import kernel_case_suite; kernel_case_suite.move(...)`
   三种导入形态均识别为叶子来源。
5. **真阳性回归锁 1**：`x = y.clone(); x.requires_grad = True` 仍拒绝。
6. **真阳性回归锁 2**：`x = y + 1; x.requires_grad = True` 仍拒绝。
7. 链截断：`x = move(...)[0].clone(); x.requires_grad = True` 拒绝；
   深层下标 `move(...)[0][1]` 通过。
8. **遮蔽守卫（含点链篡改）**：本地 `def move(v, d): return v.clone()` 后
   `x = move(y, 'musa'); x.requires_grad = True` 拒绝；
   `kernel_case_suite.move = fake` 属性重绑定后同样拒绝（无假阴性逃逸）。
9. **blocked 语义**：no_grad 块内 `x = move(y, 'musa')` 后对 x 的函数
   backward 不触发「derived from no_grad」拒绝（新求导根合法）。
10. 既有 no_grad/blocked 用例（test_musa_generation_reliability.py 的
    BAD_TEST/GOOD_TEST 族）全数保持原判定。
11. 报错文案契约：拒绝消息包含 `move(...)` 与 `.requires_grad_()`。
12. 修复循环：`_generate_test` 侧 `_call_llm` 三连
    `[废, 废, 好]` → 返回好代码、`call_count == 3`；`[废, 废, 废]` → 抛
    最后一次的 contract_error；修复消息文本含
    `move(tensor.detach(), device, gradients=True)`。

真跑验收（用户侧，不阻塞代码合并）：

13. 重跑 P45 / P47：测试生成阶段不再出现
    `only a fresh leaf tensor ... not a derived tensor` 误杀，问题进入
    kernel 生成与验证轮次。

---

## 8. 风险、边界与后续推广

1. **假阴性风险（规则放过会运行时失败的测试）**：move 的叶子性来自
   case_suite.py L122-137 的**仓库内固定实现**（empty_strided 全新分配 +
   no_grad 拷贝 + requires_grad_），可静态确证，非启发式。本地重定义
   （`def move`/`move = ...`）与模块属性篡改（`kernel_case_suite.move = ...`）
   两条遮蔽路径都由 §4.1.2 守卫拦截（测试侧没有 validate_kernel_globals
   的 framework-patch 保护，故在守卫内自理）。极端残余：测试用
   `setattr`/闭包等更绕的方式篡改 move——无日志证据、模板无此引导，
   且一旦发生会在运行时以同款 RuntimeError 暴露（fix 1-1 保证错误不被
   吞掉），损失有界。
2. **已知但明确不做的假阳性**：`x = y.detach(); x.requires_grad = True`
   运行时同样合法（detach 结果必为叶子）。45–56 批次无证据触发、B2 模板
   已把生成方向固定在 move 范式上，**观察不动**；若后续日志出现证据，
   按本方案同一模式（白名单 + 遮蔽守卫）扩展即可。
3. **A3 最坏成本**：三次全废比两次全废多一次 LLM 调用（约 2–4 分钟 /
   ≤24k tokens），且该分支现状本就是整题死亡，纯增量保险。
4. **平台范围**：2-1 规则与 move 修复均为全平台生效（kernel_case_suite
   在 CUDA/MUSA 都落盘，worker `_write_files` 无条件复制），与 2-1 原范围
   一致；不新增 MUSA gate。CUDA 模板若后续对齐 B2 文案，另行评审。
5. **与方案 B 的协同**：A1 与 fix 3-1 B2 校验器同处 `_generate_test` 的
   一个 try 块、共享同一修复循环；报错文案各自自解释，互不覆盖；A2 的
   move 范式句与 B2 模板指导是同一句话，无冲突。
6. **文案即载荷**：报错文案同时是修复提示正文（§2.3），本方案对文案的
   每处修改都会直接进入模型视野——§7 第 11 项以契约测试锁定，防止未来
   无意漂移。
