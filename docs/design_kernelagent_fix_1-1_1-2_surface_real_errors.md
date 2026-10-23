# KernelAgent 修复方案 1-1 / 1-2：验证框架不再吞掉真实错误（设计备忘录）

> 状态：已实施
> 关联问题归类：**归类 A —— 验证框架「吞掉真实错误」，4 题死于脚手架而非算法**
> 影响问题：KernelBench Level-1 的 P2 / P5 / P7 / P14（真实错误被占位符掩盖后 8 轮空转）

---

## 1. 问题陈述

KernelAgent 在 MUSA 后端下，`worker` 会先生成并运行一个「确定性校验套件」companion 测试
（文件头带 `SUITE_MARKER = "# KERNELAGENT_EXPLICIT_CASE_SUITE_V1"`），它打印
`KERNELAGENT_CASE_RESULTS=<json>` 行来上报每个 case 的覆盖情况。

当这个套件测试**在打印结果行之前就崩溃/以非零退出**时，旧代码返回一个硬编码占位符：

- `"Explicit case suite did not report coverage"`
- `"Explicit case report is empty or invalid"`
- `"An explicit test case failed or was missing"`

而子进程**真实**的 `stderr`/`stdout`（里面带着真正的 `ImportError`、`ModuleNotFoundError`、
相对导入错误、编译 traceback）被整个丢弃。于是 LLM 的 refinement 循环在同一个「与算法
无关」的神秘占位符上反复空转 8 轮，永远碰不到真正的病灶。

四个已复现的真实底层错误（全被占位符遮住）：

| 问题 | 复现出的真实错误 | 归类 |
|---|---|---|
| P2 | `ImportError: cannot import name 'kernel_function' from 'kernel' (...kernel.so)`（扩展名 `kernel` 遮蔽 wrapper） | 脚手架 |
| P5 | `ModuleNotFoundError: No module named 'matmul_scalar'` | 脚手架 |
| P7 | 同 P2 `.so` 遮蔽 wrapper | 脚手架 |
| P14 | `ImportError: attempted relative import with no known parent package` | 脚手架 |

> 结论：这些题的 kernel 数学逻辑根本没机会被验证，全卡在 bundle 自带的 glue bug 上，
> 且真实报错被验证框架吞掉，才导致「8 轮在同一句占位符上空转」。

---

## 2. 现状代码缺口的精确定位

`worker.py` `_run_test()` 中，MUSA 路径用 `run_guarded_test(test_file, 'musa')` 跑每个测试，
随后从 `result.stdout` 里找 `RESULT_PREFIX` 开头的行解析 case 结果：

```python
for line in result.stdout.splitlines():
    if line.startswith(RESULT_PREFIX):
        rows = json.loads(line[len(RESULT_PREFIX):])
        if not isinstance(rows, list) or not rows:
            return False, result.stdout, "Explicit case report is empty or invalid"   # L475
        self.last_case_results = rows
        if any(row["status"] != "passed" for row in rows):
            return False, result.stdout, result.stderr or "An explicit test case failed or was missing"  # L479
from .case_suite import SUITE_MARKER
if SUITE_MARKER in test_file.read_text() and not self.last_case_results:
    return False, result.stdout, "Explicit case suite did not report coverage"       # L482
```

**缺口**：这三处 `return` 都返回静态字符串，把 `result.stderr` / `result.stdout`（真实 traceback）
丢掉。后续 `error_info` 给 LLM 的只有那个占位符。

---

## 3. 方案设计

### 3.1 新增辅助方法 `_failure_diagnostic`

在 `worker.py` 增加一个纯静态方法，提取子进程输出里「最有行动价值」的一段：

```python
@staticmethod
def _failure_diagnostic(result: subprocess.CompletedProcess) -> str:
    """返回一次失败测试输出里最有行动价值的一段（真实错误，而非占位符）。

    优先取 stderr（携带 Python/工具链 traceback），空则回退 stdout；
    裁剪到有界长度以便塞进 refinement prompt。
    """
    fragment = (result.stderr or "").strip()
    if not fragment:
        fragment = (result.stdout or "").strip()
    if not fragment:
        return ""
    return fragment[-2000:]
```

### 3.2 三处 return 改用真实诊断（属性 1-1/1-2 的「止血」）

```python
if not isinstance(rows, list) or not rows:
    return False, result.stdout, self._failure_diagnostic(result) or (
        "Explicit case report is empty or invalid")
...
if any(row["status"] != "passed" for row in rows):
    return False, result.stdout, self._failure_diagnostic(result) or (
        "An explicit test case failed or was missing")
...
if SUITE_MARKER in test_file.read_text() and not self.last_case_results:
    return False, result.stdout, self._failure_diagnostic(result) or (
        "Explicit case suite did not report coverage")
```

保留占位符仅作为 `_failure_diagnostic` 返回空串时（确无任何输出）的最后兜底。

---

## 4. 为什么能解决对应问题

- **P2/P5/P7/P14（第一轮即见真）**：core loop 第一轮 `_run_test` 失败时，`error_info.stderr`
  现在带上真实的 `ImportError` / `ModuleNotFoundError` / 相对导入 traceback，而非占位符。
  -> LLM 立即知道真正的脚手架问题，而不是反复面对一句无意义的话。

- **不误伤正常流程**：只有「套件失败且无 RESULT_PREFIX 行」时才走 `_failure_diagnostic`；
  正常 case 成功解析时逻辑原样不变。
  有输出就透传真实内容；确无任何输出才退回占位符。

---

## 5. 关联代码 / 文件变化

| 文件 | 改动 |
|---|---|
| `triton_kernel_agent/worker.py` | 新增 `_failure_diagnostic`；L475/L479/L482 三处 return 改为「真实诊断 or 占位符」 |
| `tests/test_case_suite.py` | 新增 **1-1/1-2 回归测试**（`test_suite_crash_surfaces_real_stderr_not_placeholder`、`test_failure_diagnostic_prefers_stderr_and_bounds_output`） |
| `docs/design_kernelagent_fix_2-2_musa_bundle_semantics.md` | （相邻方案，见该文档 bundle 体格 2-2） |

---

## 6. 验证要点

1. `_failure_diagnostic` 优先取 stderr、回退 stdout、空输出返回 `""`、且长度 ≤ 2000。
2. `_run_test` 在套件先崩溃时返回真实 traceback，而不是 `"did not report coverage"`。
3. 既有「case 全部 passed」路径行为不回归。