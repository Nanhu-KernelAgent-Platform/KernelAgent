# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Platform configuration registry for multi-backend support.

Usage:
    from triton_kernel_agent.platform_config import get_platform, get_platform_choices

    platform = get_platform("xpu")
    print(platform.device_string)  # "xpu"
    print(platform.guidance_block)  # Intel XPU-specific guidance
"""

from dataclasses import dataclass, field

DEFAULT_PLATFORM = "cuda"


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for a specific hardware platform/backend."""

    name: str
    device_string: str
    torch_namespace: str
    guidance_block: str
    kernel_guidance: str
    profiler_name: str = "NCU"
    profiler_command: str = "ncu"
    profiler_file_prefix: str = "ncu"
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)


# Platform-specific constants
_XPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR INTEL XPU:**
- Default tensor allocations to device='xpu' (never 'cuda'); CPU is allowed only when necessary.
- Check availability with: hasattr(torch, 'xpu') and torch.xpu.is_available()
- Do NOT monkey-patch torch.cuda or torch.device
- Do NOT set TRITON_BACKENDS environment variable
- Do NOT import or disable XPUDriver
- Use torch.xpu.synchronize() if synchronization is needed
- Intel XPU subgroup size is typically 16 (not 32 like CUDA warps)
- Preferred block sizes: 64, 128, 256, or 512"""

_XPU_KERNEL_GUIDANCE = """\
## Intel XPU-Specific Optimizations

You are generating a Triton kernel for Intel XPU (Xe GPUs). Follow these guidelines:

1. **Device Context**: Use 'xpu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Intel Xe has different cache sizes - optimize accordingly
3. **Thread Configuration**:
   - Subgroup size is typically 8, 16, or 32 (flexible)
   - num_warps: typically 4, 8, or 16 for Intel GPUs
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
4. **Optimal Block Sizes**: Start with 128-256 for most kernels
5. **Data Types**: Intel supports fp32, fp16, bf16 (fp8 varies by generation)"""

_MUSA_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR MUSA:**
- Allocate tensors on device='musa', never device='cuda'.
- Use torch.musa for synchronization, events, and device properties.
- Keep inputs, weights, intermediates, and outputs on the same MUSA device.
- Do not monkey-patch torch.cuda or torch.device."""

_MUSA_KERNEL_GUIDANCE = """\
## MUSA-Specific Requirements

Use torch.musa APIs and MUSA-compatible Triton or native MUSA source. Do not
leave torch.cuda calls in wrappers or tests. Native MUSA kernels must use the
project-managed MUSAExtension build scaffold:
- Import both `MUSAExtension` and `BuildExtension` from
  `torch_musa.utils.musa_extension`, never `BuildExtension` from
  `torch.utils.cpp_extension`.
- Define `ext_modules=[MUSAExtension(... sources=['binding.cpp', 'kernel.mu'],
  extra_compile_args={'cxx': [...], 'mcc': [...]})]`.
- Use `cmdclass={'build_ext': BuildExtension}` in setup.py. Do not use
  `torch.utils.cpp_extension.load`; it does not support native MUSA `.mu`
  sources in this environment.
- Keep the four required files exactly: kernel.py, binding.cpp, kernel.mu,
  setup.py. The worker runs `python setup.py build_ext --inplace`.
- kernel.py must obtain the compiled module with a plain `import <name>`, where
  `<name>` is exactly the `MUSAExtension(name=...)` value. The in-place build
  writes an ABI-tagged file (e.g. `name.cpython-310-x86_64-linux-gnu.so`), so
  never hard-code a `.so` filename or build a path to one, and never call
  `torch.utils.cpp_extension.load` or re-run the build at import time.
"""

_XPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
    "from triton.backends.intel.driver import XPUDriver",
    "XPUDriver.is_available = classmethod(lambda cls: False)",
)

_MUSA_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
)

# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        torch_namespace="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
    ),
    "musa": PlatformConfig(
        name="musa",
        device_string="musa",
        torch_namespace="musa",
        guidance_block=_MUSA_GUIDANCE,
        kernel_guidance=_MUSA_KERNEL_GUIDANCE,
        profiler_name="MCU",
        profiler_command="mcu",
        profiler_file_prefix="mcu",
        cuda_hacks_to_strip=_MUSA_CUDA_HACKS,
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        torch_namespace="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
    ),
}


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name."""
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())
