# KernelAgent 修复方案 2-1：测试自身对非叶子张量设 requires_grad 的静态拦截（设计备忘录）

> 状态：已实施（轻量 AST 规则版，无真实运行冒烟）
> 关联问题归类：**归类 2 —— LLM 自己生成的测试 / bundle 自爆，与算子正确性无关**
> 影响问题：KernelBench Level-1 的 P18（LLM 生成的梯度测试在每轮因 `A_ref.requires_grad = True`
> 对非叶子张量抛 `RuntimeError`，8 轮空转）

---

## 1. 问题陈述

P18 的失败不是 kernel 写错，而是 **LLM 生成的「主测试」自身非法**：

```python
def main():
    A = torch.rand(M, K)              # A 是叶子
    A_leaf = move(A.detach(), 'musa', gradients=True)
    A_ref = A_leaf.clone()            # clone 得到的 **非叶子**张量
    A_ref.requires_grad = True        # ← 对非叶子 set requires_grad → 运行时 RuntimeError
    ...
    ref_loss.backward()
```

对一个非叶子张量直接赋 `.requires_grad = True`，PyTorch 会在运行时抛：

```
RuntimeError: you can only change requires_grad flags of leaf variables.
```

现有静态校验器 `validate_test_autograd` 不认识 `.clone()`（不在 factory 集合里），
所以漏检 → worker 每次都在同一个运行时错误上空转 8 轮，kernel 从未有机会被真正验证。

> 结论：缺的是「在进入 8 轮循环之前，静态识别「测试把 requires_grad 设在非叶子派生张量上」」的规则。

---

## 2. 方案选型（经确认）

原方案建议「真实冒烟」：用一个桩 kernel 跑一遍测试，看测试是否在 kernel 被判对之前自行抛异常。
**经用户确认，放弃真跑冒烟**（太重、引入资源依赖），改用**轻量 AST 静态规则**。

核心语义：只有在「最近一次赋值来自 torch 工厂函数」的叶子张量上允许 `X.requires_grad = True`；
任何派生张量（如 `.clone()` 的结果、运算结果）都不是叶子 → 直接静态拒绝。

---

## 3. 方案设计

### 3.1 `test_validation.py::validate_test_autograd` 改动

新增 `leaf_tensors` 集合，在已有的 `blocked`/`live` 数据流之外追踪「叶子张量名」：

1. 定义 `FACTORY_CALLS`（与现有 factory 判定共用同一集合）：
   ```python
   FACTORY_CALLS = {
       "torch.randn", "torch.rand", "torch.randn_like", "torch.rand_like",
       "torch.zeros", "torch.ones", "torch.empty", "torch.full",
       "torch.zeros_like", "torch.ones_like", "torch.empty_like",
       "torch.arange", "torch.tensor",
   }
   ```

2. `check_block` 增加 `leaf_tensors` 参数，随函数/类/with/分支递归传递；
   分支与 except handler 用**并集传播**（`leaf_tensors.update(branch_leaves)`），避免误报。

3. 在 `Assign` 分支，**先做 2-1 规则判定**，再照旧维护 `blocked`/`live`：
   ```python
   # 2-1 rule: 只有当 X 是 torch-factory 叶子时才允许 X.requires_grad = True
   for t in targets:
       if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and t.attr == "requires_grad"
               and isinstance(value, ast.Constant) and value.value is True
               and t.value.id not in leaf_tensors):
           raise ValueError(
               "Invalid correctness test: only a fresh leaf tensor "
               "(e.g. torch.randn(...), or a factory with requires_grad=True) "
               f"may have .requires_grad set to True, not a derived tensor at "
               f"line {t.lineno}. A value such as .clone() is not a leaf and "
               "fails at runtime; repair the test, not the kernel.")
   ```

4. 维护叶子状态：对一个 `ast.Name` 目标，若本次赋值是 `factory` → `leaf_tensors.add(id)`，
   否则 `leaf_tensors.discard(id)`。

### 3.2 调用链复用现有 fail-fast

`worker.run()` 已调用 `_test_validation_error(test_code)`（对每个测试跑
`validate_test_autograd`），命中即 `failure_kind="invalid_correctness_test"` 且 `rounds=0` 返回。
因此新规则**自动获得** fail-fast + 不进入 8 轮循环，无需额外接线。

---

## 4. 为什么能解决 P18

- **合法写法不受影响**：
  - `A = torch.randn(...); A.requires_grad_()`（方法调用，非属性赋值）→ 不触发该规则；
  - `torch.randn(..., requires_grad=True)`（工厂参数）→ 不触发；
  - `A_ref = torch.rand(n,N); A_ref.requires_grad = True`（直接工厂叶子）→ `leaf_tensors` 含 `A_ref`，通过。
- **P18 的坏测试被拦截**：`A_ref = A_leaf.clone()` 后 `A_ref.requires_grad = True`——
  `A_ref` 的最近赋值来自 `.clone()`（非 factory）→ 不在 `leaf_tensors` → 静态 `ValueError`。
- 结果：P18 从「用一个坏测试空转 8 轮」变成「进入循环前即 `invalid_correctness_test`，
  kernel 不被 kernel 的错牵连」。

---

## 5. 关联文件

| 文件 | 改动 |
|---|---|
| `triton_kernel_agent/test_validation.py` | `validate_test_autograd`：新增 `leaf_tensors` 追踪 + 2-1 规则 |
| `tests/test_musa_generation_reliability.py` | 新增 2-1 规则测试 |
| `docs/design_kernelagent_fix_2-2_musa_bundle_semantics.md` | 相邻方案（bundle 体格 2-2） |

---

## 6. 验证要点

1. `A_ref = torch.rand(...); A_ref.requires_grad = True`（leaf）→ 通过。
2. `A_ref = A.clone(); A_ref.requires_grad = True`（非 leaf）→ `ValueError`。
3. 分支/函数作用域内 leaf 状态正确传播，不误伤合法分支。
4. P18 精确模式在 8 轮循环前即被拒。