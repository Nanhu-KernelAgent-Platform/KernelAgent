# KernelAgent — Multi‑Agent GPU Kernel Synthesis and Optimization

KernelAgent turns PyTorch programs into verified Triton kernels and optimize its performance. It was designed around KernelBench workloads and combines:

- Static problem analysis to decide whether to run a lightweight path or a full pipeline
- LLM‑assisted refactoring that isolates fusable subgraphs
- Parallel Triton kernel generation with strict runtime verification
- End‑to‑end composition that rebuilds the original forward pass using only the synthesized kernels
- Hardware‑guided optimization pipeline that iteratively improves performance

GPU Kernel Synthesis Blog post: [PyTorch KernelFalcon](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/)

GPU Kernel Optimization Blog post: [PyTorch KernelAgent](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)

## Kernel Generation Pipeline Overview

![](./assets/kernelagent2.excalidraw.svg)

Every stage writes artifacts to a run directory under `.fuse/<run_id>/`, including the fused PyTorch code, `subgraphs.json`, individual KernelAgent sessions, and the final `compose_out/composed_kernel.py`.

## KernelAgent Multi-Worker Optimization Pipeline Overview
![](./assets/opt_agent.svg)
Every stage writes artifacts to a run directory under `.optimize/<run_id>/`, including the input Triton kernel, artifacts, individual optimization worker sessions, and the final `output/best_kernel.py`.


## Quickstart

### Requirements
- Python 3.8 – 3.12
- Linux or macOS
- **GPU Requirements (one of the following):**
  - **CUDA**: NVIDIA GPU with CUDA support
  - **XPU**: Intel GPU with oneAPI support (Arc, Data Center GPUs, or integrated Xe graphics)
- Triton (installed separately: `pip install triton` or nightly from source)
- PyTorch (https://pytorch.org/get-started/locally/)
- LLM provider ([OpenAI](https://openai.com/api/), [Anthropic](https://www.anthropic.com/), or a self-hosted relay)

### Install
```bash
pip install -e .
```

### Platform-Specific PyTorch Installation

#### Intel XPU (Intel GPUs)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

**Note:** Intel XPU support requires:
- Compatible Intel GPU (Arc series, Data Center GPUs, or integrated Xe graphics)
- Linux with appropriate Intel GPU drivers

Verify your XPU installation:
```python
import torch
print(torch.xpu.is_available())  # Should print True
print(torch.xpu.device_count())  # Number of Intel GPUs
```

#### (Optional) Install KernelBench for problem examples
```bash
git clone https://github.com/ScalingIntelligence/KernelBench.git
```
Note: By default, KernelAgent UI searches for KernelBench at the same level as `KernelAgent`. (i.e. `../KernelBench`)

### Configure
You can export keys directly or use an `.env` file that the CLIs load automatically.

```bash
OPENAI_MODEL=gpt-5            # default model for extraction
NUM_KERNEL_SEEDS=4            # parallel workers per kernel
MAX_REFINEMENT_ROUNDS=10      # retry budget per worker
KERNEL_AGENT_EXPERIENCE_DB=.kernelagent/experiences.sqlite3
LOG_LEVEL=INFO                # logging level
```

Kernel generation and optimization use a persistent cross-task experience
store by default. Correctness-verified kernels, measured optimization actions,
MCU/NCU summaries, and failed approaches are stored in the configured SQLite
database. Later tasks retrieve experiences only from the same platform and
kernel backend, then rank them by operator, dtype, shape, and problem
similarity. Set `enable_experience_memory: false` in an optimization YAML
config, or pass `enable_experience_memory=False` to the Python API, to disable
retrieval and recording.

Inspect the binary SQLite store with the read-only helper instead of opening it
as text:

```bash
python3 scripts/show_experiences.py --operator rmsnorm --platform musa
python3 scripts/show_experiences.py --outcome improved --json
```

Add `--show-code` to include the complete stored Kernel bundle. JSON output can
be saved with normal shell redirection when a human-readable snapshot is needed.

For MUSA, prompt enrichment remains inside the existing generation and
optimization calls: a static problem context builder extracts the operator
contract, the MUSA knowledge pack supplies bounded operator-family guidance,
PromptManager selects a specialized family template, and highly compatible
verified historical implementations may be included as generation seeds.

#### LLM Providers
KernelAgent currently supports OpenAI and Anthropic out-of-the-box. You can also use a custom OpenAI endpoint.
These can be configured in `.env` or via environment variables.
```bash
# OpenAI (models like `o4-mini`, `gpt-5`)
OPENAI_API_KEY=sk-...

# Anthropic (default; `claude-sonnet-4-20250514` is used when `OPENAI_MODEL` is unset)
ANTHROPIC_API_KEY=sk-ant-...

# Relay configuration for self-hosted gateways
LLM_RELAY_URL=http://127.0.0.1:11434
LLM_RELAY_TIMEOUT_S=120
```

More knobs live in `triton_kernel_agent/agent.py` and `Fuser/config.py`.

## End-to-End Kernel Generation Workflows

- **Auto-route a KernelBench problem** — static analysis picks between the direct KernelAgent path and the full Fuser pipeline, with automatic fallback if the first attempt fails:
  ```bash
  python -m Fuser.auto_agent \
    --problem /abs/path/to/KernelBench/level1/19_ReLU.py \
    --no-router-cache \     # avoid caching or using cached results
    --verify                # ensure final composition test runs
  ```
  `--no-router-cache` can be enabled to avoid utilizing any cached router results and prevent writing to the cache.

- **Manually run the pipeline (extract → dispatch → compose)** when you want explicit control over models or concurrency:
  ```bash
  python -m Fuser.pipeline \
    --problem /abs/path/to/problem.py \
    --extract-model gpt-5 \
    --dispatch-model o4-mini \
    --dispatch-jobs auto \
    --compose-model o4-mini \
    --workers 4 \
    --max-iters 5 \
    --verify

  # For Intel XPU
  python -m Fuser.pipeline \
    --problem /abs/path/to/problem.py \
    --target-platform xpu \
    --extract-model gpt-5 \
    --dispatch-model o4-mini \
    --dispatch-jobs auto \
    --compose-model o4-mini \
    --workers 4 \
    --max-iters 5 \
    --verify

  ```
  `dispatch-jobs auto` matches the number of discovered subgraphs; artifacts are placed under `.fuse/<run_id>/`.

- **Direct KernelAgent run** — bypass Fuser and provide a plain language problem description or a KernelBench snippet:
  ```python
  from triton_kernel_agent import TritonKernelAgent

  agent = TritonKernelAgent(num_workers=4, max_rounds=8, model_name="gpt-5")
  result = agent.generate_kernel(
      problem_description="Implement ReLU over a contiguous 1D tensor of length 1024"
  )

  if result["success"]:
      print("Kernel path:", result["kernel_path"])
      print("Session directory:", result["session_dir"])
  else:
      print("Failure:", result["message"])
  ```

- **UIs** — interactive runs with Gradio frontends:
  - Triton KernelAgent UI: `kernel-agent` or `python scripts/triton_ui.py`
  - Fuser orchestration UI: `fuser-ui` or `python scripts/fuser_ui`
  - Full pipeline UI: `pipeline-ui` or `python scripts/pipeline_ui`

## Component Details

- **AutoRouter (`Fuser/auto_agent.py`)**: parses the problem’s AST, looks for attention blocks, transposed convolutions, control flow, and long op chains. It caches decisions under `.fuse/router_cache.json` and can fall back to the other path if the first attempt fails. Use  `--no-router-cache` to ignore the existing cache and caching new routes. Use `--ignore-router-config` to ignore router-provided tuning and rely on CLI args.

- **Fuser Orchestrator (`Fuser/orchestrator.py`)**: rewrites the PyTorch module into fusable modules, executes them for validation, and packages a tarball of the fused code. Run IDs and directories are managed via `Fuser/paths.py`.

- **Subgraph Extractor (`Fuser/subgraph_extractor.py`)**: prompts the LLM to emit a JSON array describing each unique subgraph, including ops, shapes, dtypes, and parameter tensors. Entries are deduplicated by shape signature so the dispatcher can reuse kernels.

- **Dispatcher (`Fuser/dispatch_kernel_agent.py`)**: converts each JSON item into a precise Triton generation spec, then spins up `TritonKernelAgent` processes in parallel. Each worker writes its own session directory with the candidate kernel, test harness, and verification logs.

- **TritonKernelAgent (`triton_kernel_agent/`)**: manages a pool of verification workers (`worker.py`, `manager.py`). Each worker iteratively asks an LLM for improvements, executes unit tests under sandboxed subprocesses (`Fuser/runner.py`), and enforces strict bans on PyTorch fallbacks. A run succeeds only when the test prints `PASS` (or the sentinel string) and exits with status 0.

- **Composer (`Fuser/compose_end_to_end.py`)**: stitches the verified kernels back into a single Triton program. The composed file contains one or more `@triton.jit` kernels plus a `kernel_function(...)` wrapper and a self-test that replays the original PyTorch problem. With `--verify`, the test is executed immediately and must succeed.

## End-to-End Kernel Optimization Workflows

KernelAgent includes a hardware-guided optimization pipeline that iteratively improves a verified Triton kernel's performance using GPU profiling feedback.

1. **Profile** — NCU collects 28 hardware metrics (compute utilization, memory bandwidth, cache hit rates, occupancy, stall breakdowns)
2. **Roofline Analysis** — Classifies the kernel as memory-bound, compute-bound, or underutilized based on SOL (speed-of-light) percentages
3. **Bottleneck Diagnosis** — An LLM analyzes the NCU metrics + kernel code to identify root causes and recommend specific fixes
4. **Optimization** — An LLM generates an optimized kernel applying the recommended fixes
5. **Verification** — The optimized kernel is tested for numerical correctness against PyTorch reference
6. **Benchmarking** — CUDA event timing measures the new kernel, tracking best-so-far with divergence-based revert

The loop runs for up to N rounds, with early termination when the kernel reaches roofline (≥95% SOL) or when performance converges.

## Usage

### Gradio UI

```bash
python scripts/optimization_ui.py --port 8085
```


### Programmatic API

**Optimize a kernel using beam search** — parallel exploration with top-N kernels and M bottleneck directions:
```bash
cd examples && python run_opt_manager.py \
  --kernel-dir optimize_01_matvec/ \
  --strategy beam_search \
  --max-rounds 5
```

### Key Components

| Component | Location | Role |
|---|---|---|
| **OptimizationOrchestrator** | `triton_kernel_agent/opt_worker_component/orchestrator/` | Main optimization loop |
| **KernelProfiler** | `triton_kernel_agent/opt_worker_component/profiling/` | NCU hardware profiling |
| **BottleneckAnalyzer** | `triton_kernel_agent/opt_worker_component/prescribing/` | LLM-based bottleneck diagnosis |
| **RooflineAnalyzer** | `kernel_perf_agent/kernel_opt/roofline/` | SOL classification and early stopping |
| **Benchmark** | `triton_kernel_agent/opt_worker_component/benchmarking/` | CUDA event timing |

### Optimization Artifacts

```
.optimize/workers/<worker_id>/<run_id>/artifacts
  kernel_round_0.py          # baseline kernel
  kernel_round_N.py          # kernel after round N
  round001_opt_prompt.txt    # optimization prompt sent to LLM
  round001_opt_reply.txt     # LLM response
  round001_strategy.json     # bottleneck analysis result
  ...
```

## Platform Support

KernelAgent and Fuser support multiple GPU platforms and source backends:

| Platform | Device String | Flag | Status |
|----------|---------------|------|--------|
| NVIDIA CUDA | `cuda` | `--target-platform cuda` (default) | Fully supported |
| Moore Threads MUSA | `musa` | `--target-platform musa` | Triton and native MUSA backend |
| Intel XPU | `xpu` | `--target-platform xpu` | Supported |

Fuser keeps the source backend separate from the target platform. The default
`--kernel-backend triton` path is unchanged. Native MUSA generation is enabled
explicitly with `--target-platform musa --kernel-backend musa`; it produces a
validated bundle containing `kernel.py`, `binding.cpp`, `kernel.mu`, and
`setup.py`. The existing `--target-platform musa --kernel-backend triton`
combination remains available for MUSA-compatible Triton kernels.

### Native MUSA PPT Demo

The sigmoid presentation demo runs AutoRoute, native MUSA generation,
correctness verification, MCU-guided optimization, re-verification, benchmarking,
and experience persistence with one command:

```bash
bash examples/run_ppt_demo.sh \
  --problem examples/optimize_04_musa_sigmoid/problem.py \
  --allow-third-party-api \
  --reasoning-effort low
```

Each run is retained under `examples/results/run_<timestamp>_<pid>/` by default.
Use `--problem /abs/path/to/problem.py` for a custom KernelBench-style problem
and `--output-root /abs/path/to/results` to select another run root. The directory
contains `demo.log`, `demo_summary.json`, profiler logs, the generated bundle,
and a materialized `best_bundle/`. To retry optimization without paying the
generation cost again, pass a previous result directory:

```bash
bash examples/run_ppt_demo.sh \
  --allow-third-party-api \
  --reuse-generated examples/results/run_<id>
```

The final status distinguishes an improvement (`SUCCESS`), a valid candidate
that did not beat the baseline (`NO_GAIN`), profiler/worker degradation
(`DEGRADED`), and a hard failure (`FAILED`).

### Intel XPU Notes

When targeting Intel XPU, KernelAgent automatically:
- Uses `device='xpu'` for all tensor allocations
- Applies XPU-specific Triton optimizations (subgroup sizes, block sizes)
- Generates appropriate device availability checks
- Removes CUDA-specific patterns from generated code

### Verifying Platform Setup
```python
# Check CUDA availability
import torch
print("CUDA available:", torch.cuda.is_available())

# Check XPU availability
print("XPU available:", hasattr(torch, 'xpu') and torch.xpu.is_available())
```

## Run Artifacts

A successful pipeline run yields a structure similar to:

```
.fuse/<run_id>/
  orchestrator/code.py.tgz         # fused PyTorch refactor
  subgraphs.json                   # shape-specialized subgraph descriptions
  kernels_out/
    <subgraph_id>/*                # per-subgraph KernelAgent sessions
    summary.json                   # success/failure per subgraph
  compose_out/
    composed_kernel.py             # final Triton program + self-test
    summary.json                   # composition metadata
```

These artifacts are designed for reproducibility: you can re-run a single kernel session, inspect prompts/responses, or feed `composed_kernel.py` directly into downstream tooling.

## Example Artifacts

Looking for ready-to-browse **kernel generation** outputs? See the curated artifacts repo:

- https://github.com/Laurawly/kernelagent-artifacts

It includes selected L1/L2/L3 problems with:
- Original problems (PyTorch)
- Fused subgraphs (`subgraphs.json`) and per‑subgraph Triton kernels
- Composed end‑to‑end Triton programs and verification logs
- Minimal examples for quick scanning

Looking for ready-to-browse **kernel optimization** outputs? See the curated artifacts repo:
- https://github.com/kaiming-cheng/kernelagent-optimization-artifacts


It includes selected L1 problems with:
- Initial Triton kernel (generated by earlier KernelAgent) fed into the optimization loop
- Final optimized Triton kernel (output)
- Per-round artifacts from the beam-search optimization
## Repository Layout

- `triton_kernel_agent/` — KernelAgent core (agent, worker manager, provider adapters, prompt templates)
- `triton_kernel_agent/opt_worker_component/` — optimization pipeline (profiler, benchmarker, bottleneck analyzer, orchestrator)
- `kernel_perf_agent/kernel_opt` — roofline analysis, hardware specs, and benchmarking utilities
- `Fuser/` — auto-router, orchestration pipeline, CLIs, Gradio UIs
- `triton_kernel_agent/templates/` — Jinja templates used when prompting TritonKernelAgent
- `examples/` — sample problems and prompt snippets
- `tests/` — unit tests for agents and utilities
- `e2e_test.py` — example end-to-end kernel generation harness
- `scripts/` — coverage/benchmark tooling, profiling helpers, CLI entry points (e.g., autoroute coverage runners, Triton UI)

## Development

- Install in editable mode with `pip install -e .[dev]`
- Run the test suite with `pytest -v`
- Follow the contribution guidelines in `CONTRIBUTING.md`
- KernelAgent intentionally leaves Triton installation to the user so you can pin the version that matches your GPU driver/toolchain

## Documentation & Community

- Optimization pipeline docs: see [Kernel Optimization Pipeline](#kernel-optimization-pipeline) above
- Issues: https://github.com/pytorch-labs/KernelAgent/issues
- Blog post: https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/

## License

KernelAgent is released under the Apache License 2.0; see `LICENSE`.
