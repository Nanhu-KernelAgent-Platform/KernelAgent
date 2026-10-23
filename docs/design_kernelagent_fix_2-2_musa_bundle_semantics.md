# KernelAgent 修复方案 2-2：MUSA bundle 语义体检（设计备忘录）

> 状态：设计已定稿，等待审核 / **已实施（2026-xx）**
> 关联问题归类：**归类 2 —— bundle 自爆（glue 语义错误 / 工具链选错）**，不是算子正确性问题。
> 适用范围：KernelBench Level-1 已失败的 P19/P4/P2/P7/P14 这类"编译前就注定失败"的 bundle。

---

## 1. 问题陈述

KernelAgent 生成一个 Native MUSA kernel 时得到 4 个文件（bundle）：

```
kernel.py        # Python 包装，必须导出 kernel_function()
binding.cpp      # pybind + launch 封装
kernel.mu        # MUSA kernel 源码（由 mcc 编译 .mu）
setup.py         # 用 torch_musa.utils.musa_extension 的 MUSAExtension 构建
```

实测失败分两类，且**都不是 kernel 数学写错**：

| 问题 | 真实根因 | 已发生的题 |
|---|---|---|
| 工具链选错 | setup.py 用了 `torch.utils.cpp_extension.BuildExtension`（CUDA 路径），mcc 从不处理 `.mu` → `error: unknown file type '.mu'` | P19 |
| builtin 位置错 | `blockIdx.x` 等 MUSA/CUDA builtin 被写进 **host** `launch_*` 函数（而非 `__global__` 内核体）→ mcc 报 __device__ function in __host__ function | P4 |
| 扩展名遮蔽 | setup.py `name='kernel'` → 编译出 `kernel.<abi>.so` 与 wrapper `kernel.py` **同名**，Python 导入时 `.so` 优先，`from kernel import kernel_function` 实际导入 `.so` → ImportError | P2、P7 |
| 相对导入 | kernel.py 写 `from . import matmul_triu_ext`，bundle 以顶层脚本运行且无 package → `attempted relative import with no known parent package` | P14 |

这些全部发生在 **worker 8 轮验证循环开始之前或第一轮 build 时**，而当前流程对 bundle 只做“文件名是否齐全”的结构检查，**从不检查内容语义**，于是要么直接编译失败，要么一轮一轮反复失败/空转。

---

## 2. 现有校验的缺口（精确到代码）

### 2.1 种子生成侧 — `agent.py:_extract_kernel_payload`（L696-713）

只做三件事，全部是**结构检查**：

1. `extract_kernel_bundle(response_text)` — 只认 4 个 FILE 块是否被解析出来。
2. `bundle.require(self._kernel_backend)` — 只查文件名是否在 `required_files` 里。
3. `"def kernel_function" not in wrapper` — 只查 wrapper 里有没有该字符串。

**不查**：setup.py 是否用对了 `torch_musa` 构建类；`kernel.py` 是否有相对导入；
setup 的 extension 名是否与 `kernel.py` 冲突/遮蔽；binding.cpp 里 builtin 是否出现在 host 函数。

### 2.2 验证侧 — `worker.py:_validate_kernel_candidate`（L298-341）

对 `KernelBundle` 分支同样只做：
`require()` + `"def kernel_function"` 存在性 + 对 `.py` 文件跑 `validate_kernel_globals`（防止 framework 篡改全局）。

**不查** bundle 内 `binding.cpp` / `kernel.mu` / `setup.py` 的语义，**也不查** `kernel.py` 相对导入。

> 结论：两处都需要同一个“bundle 语义体检函数”，失败信息要可被现有的 “repair” 路径回喂给 LLM。

---

## 3. 方案设计

### 3.1 新增函数（放在 `triton_kernel_agent/kernel_backend.py`）

新增一个纯函数，输入 `KernelBundle`，输出 `str | None`（`None`=体检通过，`str`=失败原因）：

```python
def validate_musa_bundle_semantics(bundle: KernelBundle) -> str | None:
    """对 Native MUSA bundle 做*内容语义*体检（不是语法解析完整分析）。

    只做廉价、稳健的静态检查；命中任何一项即返回可读的失败原因，
    以便现有 “repair the complete format” 流程把原因回喂给 LLM。
    """
```

四个检查项（每项对应 §1 的一个失败）：

#### 检查①：setup.py 工具链必须用 torch_musa 构建类（对应 P19）

- 读取 `bundle.files["setup.py"]`。
- 必须**同时**满足：
  - 出现 `torch_musa.utils.musa_extension`；
  - **未**出现 `torch.utils.cpp_extension` / `torch.utils.cpp_extension.BuildExtension`（CUDA 工具链字样）。
- 失败原因示例：
  `setup.py uses {FoundCudaName} (CUDA toolchain) instead of torch_musa.utils.musa_extension.{MUSAExtension,BuildExtension}; mcc refuses .mu files otherwise`.

> 为什么能解决 P19：P19 的 setup.py 就是 `from torch.utils.cpp_extension import BuildExtension`。本检查在 build/编译之前就命中并返回明确原因，让 LLM 一次修对，而不是等 mcc 报 `unknown file type '.mu'` 后再去猜。

#### 检查②：extension 名不得遮蔽 kernel.py wrapper（对应 P2/P7）

- 从 `setup.py` 中解析出 `MUSAExtension(name=...)` 的 name（正则抓 `name=['\"]([^'\"]+)['\"]`）。
- **也扫描 `kernel.py` 里 `import <name>` / `from <name> import` 的顶层模块名**，找出 wrapper 实际依赖的扩展模块名。
- 只要满足任意一条就失败：
  - 解析出的 extension name 等于 `kernel`（与 wrapper `kernel.py` 同名 → `.so` 遮蔽）。
  - extension name 等于 `kernel.py` 里顶层 import 目标的模块名（wrapper 想 import 它，但它会被编译成同名 `.so`，导致 wrapper 拿到的是扩展而非 Python 包装）。
- 失败原因示例：
  `setup.py MUSAExtension name='kernel' matches the wrapper module kernel.py; the built .so shadows it and 'from kernel import kernel_function' will import the extension, not the wrapper`.

> 为什么能解决 P2/P7：P2/P7 的 setup name 均为 `kernel`，编译后生成 `kernel.<abi>.so` 遮蔽 `kernel.py`，导入 `kernel_function` 失败。本检查在生成种子后立刻识破，让 LLM 改为 `name='<other>'`。
>
> **实现备注**：检查②只拦截 `name == 'kernel'` 遮蔽 wrapper *文件* 的情形，**不**拦截「kernel.py 导入与扩展同名的模块」——因为那是 wrapper 调用编译扩展入口的**正确契约**（如 P5 的 `import matmul_scalar` 实为合法）。误报已消除。

#### 检查③：kernel.py 禁止相对导入（对应 P14）

- 读 `bundle.files["kernel.py"]`。
- 若出现 `from . import` 或 `from ..` 开头的相对导入 → 失败。
- 失败原因示例：
  `kernel.py 使用相对导入 (from . import ...)；bundle 以顶层脚本运行，无 parent package，会产生 ImportError；请改为绝对导入 (import <module> 或 from <module> import ...)`。

> 为什么能解决 P14：P14 的 kernel.py 是 `from . import matmul_triu_ext`，运行时必然 `attempted relative import with no known parent package`。本检查静态命中，不用等运行。

#### 检查④：binding.cpp 的 MUSA builtin 只能出现在 __global__ 内核体（对应 P4）

- 只做**廉价文本/行级判断**（不做完整 C++ 解析）：
  - 若 `binding.cpp` 或 `kernel.mu` 中出现 `blockIdx.` / `blockDim.` / `gridDim.` / `threadIdx.`，
  - 且该出现**不**位于某个 `__global__` 声明的块之后的 `loop/for` 段内——简化实现：**只要找到 builtin 出现在 `__global__` 关键字的行号之前**（即出现在 host 段），就判定为错误。
  - 更稳健的近似：把文件按行分割，维护「当前是否处于最近一个 `__global__` 函数体内」的状态。若 builtin 出现在非 `__global__` 函数体内 → 失败。
- 失败原因示例：
  `binding.cpp/kernel.mu 的 host 函数（launch_*）中出现了 MUSA builtin (blockIdx.)；builtin 只能在 __global__ 内核体内使用；请把运算体移入 __global__ 内核。`

> 为什么能解决 P4：P4 的 kernel.mu 把 `blockIdx.x` 写在 host `launch_matmul` 模板里（CUDA 习惯误迁移到 MUSA）。本检查在编译前识别出「builtin 未处于 __global__ 段」，让 LLM 一次性修正。

---

### 3.2 接线点（两处都调用同一函数）

#### 接线程 A：`worker.py:_validate_kernel_candidate`（拒绝无效 bundle 快速失败）

```python
if isinstance(kernel_code, KernelBundle):
    ...  # 现有 require() + def kernel_function
    reason = validate_musa_bundle_semantics(kernel_code)
    if reason:
        return reason      # 复用现有 "Malformed..." 失败信息回喂路径
    return None
```

#### 接线程 B：`agent.py:_extract_kernel_payload`（种子生成后立即体检）

```python
bundle.require(self._kernel_backend)
...
wrapper = bundle.files.get("kernel.py", "")
if "def kernel_function" not in wrapper:
    raise ValueError("kernel.py is missing required kernel_function definition")
reason = validate_musa_bundle_semantics(bundle)
if reason:
    raise ValueError(reason)   # 路由进现有 "repair complete format" 重试
```

> 现有代码注释已明确：`_extract_kernel_payload` 里 `raise ValueError(reason)` 会**自动进入** seed 的 “repair 完整格式” 重试循环（`_generate_kernel_seeds` 里对 invalid 响应做 repair）。因此只需抛异常、不必另写新循环。

---

## 4. 为什么这是"精简且有效"

- **新增面小**：只新增一个函数 + 两处各加约三行调用。
- **纯静态、无 IO**：不构建、不运行，不会被打断/超时，也不依赖 mcc/MUSA 环境。
- **失败即快速失败**：4 类问题（工具链选错、扩展名遮蔽、相对导入、host-builtin）在编译前被识别，LLM 拿到的是可读的**根因描述**而非一个玄学编译错误。
- **不误伤合法 bundle**：检查项都是“命中异常模式才返回”；正常用 `torch_musa`、扩展名非 `kernel`、绝对导入、builtin 在 `__global__` 内均通过。
- **与现有流程兼容**：两个接线点复用了已有 `raise ValueError -> repair` 与 `return reason -> failed_run` 机制，不需要新增错误类型。

---

## 5. 验证计划（与 2-1 一起跑）

在 `tests/test_musa_generation_reliability.py` 增加用例，构造四类病态 bundle 与一个正常 bundle，断言：
1. P19 型（CUDA BuildExtension）→ 命中检查①；
2. P2/P7 型（extension name == 'kernel'）→ 命中检查②；
3. P14 型（`from . import`）→ 命中检查③；
4. P4 型（builtin 出现在 host 段）→ 命中检查④；
5. 合法 bundle（torch_musa + 非 kernel 名 + 绝对导入 + builtin 在 __global__ 内）→ 返回 None。

并保证既有用例（如 `test_comment_headers_parse_complete_bundle`）不受影响。

---

## 6. 关联文件清单

| 文件 | 改动 |
|---|---|
| `triton_kernel_agent/kernel_backend.py` | 新增 `validate_musa_bundle_semantics` |
| `triton_kernel_agent/worker.py` | `_validate_kernel_candidate` 加一行调用 |
| `triton_kernel_agent/agent.py` | `_extract_kernel_payload` 加一行调用 + raise |
| `tests/test_musa_generation_reliability.py` | 新增 2-2 用例（并保留 2-1 用例） |