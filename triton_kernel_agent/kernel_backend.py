# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

"""Kernel code-generation backend definitions and multi-file payload helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

DEFAULT_KERNEL_BACKEND = "triton"
_FILE_BLOCK_RE = re.compile(
    r"(?:^|\n|(?<=[.!?]))[ \t]*(?:#{1,6}[ \t]+)?(?:\*{1,2})?"
    r"FILE[ \t]*:[ \t]*([^\r\n]+?)(?:\*{1,2})?[ \t]*\r?\n"
    r"[ \t]*```[^\r\n]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_FENCED_FILE_BLOCK_RE = re.compile(
    r"(?:^|\n)```FILE\s*:?[ \t]*([^\r\n]+)\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_MARKDOWN_HEADING_FILE_BLOCK_RE = re.compile(
    r"(?:^|\n)[ \t]*#{1,6}[ \t]+[`*_'\"]*"
    r"([A-Za-z0-9_.-]+\.(?:py|cpp|mu))[`*_'\"]*[ \t]*\r?\n"
    r"[ \t]*```[^\r\n]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_FILENAME_BLOCK_RE = re.compile(
    r"(?:^|\n)[ \t]*```[^\r\n]*?"
    r"(?:filename|file)[ \t]*[:=][ \t]*[`'\"]?"
    r"([A-Za-z0-9_.-]+\.(?:py|cpp|mu))[`'\"]?[^\r\n]*\r?\n"
    r"(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_COMMENT_FILE_BLOCK_RE = re.compile(
    r"(?:^|\n)[ \t]*```[^\r\n]*\r?\n[ \t]*"
    r"(?:\#|//)[ \t]*FILE[ \t]*:[ \t]*([^\r\n]+)\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
# Some models open the fence first and put a bare `FILE: name` marker on the
# first line inside it, instead of the documented `FILE: name` + fence order.
# Accept that ordering too; the comment prefix is optional here.
#
# The negative lookahead is essential: without it this pattern also matches the
# *closing* fence of the previous file followed by the next `FILE: name` line
# and that file's *opening* fence, yielding an empty body. Because
# extract_kernel_bundle merges matches into a dict, those empty matches would
# overwrite the correct bodies produced by _FILE_BLOCK_RE and silently reduce
# every non-first file to a blank placeholder.
_FENCE_THEN_FILE_BLOCK_RE = re.compile(
    r"(?:^|\n)[ \t]*```[^\r\n]*\r?\n[ \t]*"
    r"(?:\#|//)?[ \t]*FILE[ \t]*:[ \t]*([^\r\n]+?)[ \t]*\r?\n"
    r"(?![ \t]*```)(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_CANONICAL_MUSA_FILENAMES = {
    name.lower(): name for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
}


def _normalize_generated_filename(raw_name: str) -> str:
    """Normalize harmless Markdown decoration around a generated filename."""

    name = raw_name.strip()
    name = re.sub(r"^(?:filename|file)[ \t]*[:=][ \t]*", "", name, flags=re.I)
    name = name.strip(" \t`*_'\"")
    name = name.removeprefix("./")
    # Never turn a path into a harmless canonical basename during normalization.
    if "/" in name or "\\" in name or name == "..":
        return name

    # Models sometimes append a short explanation after the filename. Prefer a
    # known MUSA bundle filename when one is present instead of treating the
    # explanation as part of the artifact path.
    known = re.search(
        r"(?i)(?:^|[^A-Za-z0-9_.-])"
        r"(kernel\.py|binding\.cpp|kernel\.mu|setup\.py)"
        r"(?:$|[^A-Za-z0-9_.-])",
        name,
    )
    if known:
        return _CANONICAL_MUSA_FILENAMES[known.group(1).lower()]
    return name


@dataclass(frozen=True)
class KernelBackendConfig:
    name: str
    display_name: str
    generation_template: str
    refinement_template: str
    guidelines_template: str
    optimization_template: str
    supported_platforms: tuple[str, ...]
    required_files: tuple[str, ...] = ("kernel.py",)
    verification_timeout_s: int = 30
    test_generation_template: str = "test_generation.j2"

    def supports_platform(self, platform_name: str) -> bool:
        return platform_name in self.supported_platforms


@dataclass(frozen=True)
class KernelBundle:
    """A validated collection of generated kernel source files."""

    files: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized: dict[str, str] = {}
        for raw_name, content in self.files.items():
            name = raw_name.strip().replace("\\", "/")
            if not name or name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"Unsafe kernel artifact path: {raw_name!r}")
            if "/" in name:
                raise ValueError(
                    f"Kernel artifacts must be top-level files: {raw_name!r}"
                )
            normalized[name] = content.rstrip() + "\n"
        object.__setattr__(self, "files", normalized)

    def require(self, backend: KernelBackendConfig) -> None:
        missing = [name for name in backend.required_files if name not in self.files]
        if missing:
            raise ValueError(
                f"{backend.display_name} response is missing required files: "
                f"{', '.join(missing)}"
            )

    def render_for_prompt(self) -> str:
        return "\n".join(
            f"FILE: {name}\n```\n{content.rstrip()}\n```"
            for name, content in self.files.items()
        )


BACKENDS: dict[str, KernelBackendConfig] = {
    "triton": KernelBackendConfig(
        name="triton",
        display_name="Triton",
        generation_template="kernel_generation.j2",
        refinement_template="kernel_refinement.j2",
        guidelines_template="triton_guidelines.j2",
        optimization_template="kernel_optimization.j2",
        supported_platforms=("cuda", "musa", "xpu"),
    ),
    "musa": KernelBackendConfig(
        name="musa",
        display_name="MUSA",
        generation_template="musa_kernel_generation.j2",
        refinement_template="musa_kernel_refinement.j2",
        guidelines_template="musa_guidelines.j2",
        optimization_template="musa_kernel_optimization.j2",
        supported_platforms=("musa",),
        test_generation_template="musa_test_generation.j2",
        required_files=("kernel.py", "binding.cpp", "kernel.mu", "setup.py"),
        verification_timeout_s=180,
    ),
}


def get_kernel_backend(name: str) -> KernelBackendConfig:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown kernel backend {name!r}. Available: {', '.join(sorted(BACKENDS))}"
        ) from exc


def get_kernel_backend_choices() -> list[str]:
    return sorted(BACKENDS)


def extract_kernel_bundle(response_text: str) -> KernelBundle | None:
    """Extract safe fenced file blocks from a mildly variable LLM response."""

    text = response_text or ""
    matches = _FILE_BLOCK_RE.findall(text)
    matches.extend(_FENCED_FILE_BLOCK_RE.findall(text))
    matches.extend(_MARKDOWN_HEADING_FILE_BLOCK_RE.findall(text))
    matches.extend(_FENCE_FILENAME_BLOCK_RE.findall(text))
    matches.extend(_COMMENT_FILE_BLOCK_RE.findall(text))
    matches.extend(_FENCE_THEN_FILE_BLOCK_RE.findall(text))
    if not matches:
        return None
    return KernelBundle(
        {
            _normalize_generated_filename(name): content.strip()
            for name, content in matches
        }
    )


# Builtin identifiers that MUSA/CUDA kernels may use, permitted only inside a
# ``__global__`` kernel body (never in a host ``launch_*`` wrapper).
_MUSA_DEVICE_BUILTIN_TOKENS = (
    "blockIdx.", "blockDim.", "gridDim.", "threadIdx.",
)
_EXTENSION_NAME_RE = re.compile(r"name[ \t]*=[ \t]*['\"]([^'\"]+)['\"]")
_RELATIVE_IMPORT_RE = re.compile(
    r"^[ \t]*(?:from[ \t]+\.+|import[ \t]+\.)", re.MULTILINE
)


def validate_musa_bundle_semantics(bundle: KernelBundle) -> str | None:
    """Cheap *content-semantic* checks for a Native MUSA bundle.

    The structural checks (``require`` + ``def kernel_function`` presence) do
    not catch bundle-glue mistakes that are fatal before or at compile time:

    1. setup.py choosing the CUDA extension toolchain instead of torch_musa.
    2. a MUSAExtension whose module name shadows the kernel.py wrapper (or the
       module the wrapper imports), so ``from kernel import ...`` resolves to
       the compiled ``.so`` instead of the wrapper.
    3. a relative import in kernel.py, which fails when a bundle is run as a
       top-level script.
    4. MUSA/CUDA device builtins (``blockIdx.`` etc.) appearing inside host
       ``launch_*`` functions rather than a ``__global__`` kernel body.

    Returns ``None`` when the bundle looks semantically sound, otherwise a
    human-readable reason describing the defect so the existing "repair the
    complete format" flow can feed it back to the LLM.
    """
    setup_py = bundle.files.get("setup.py", "")
    kernel_py = bundle.files.get("kernel.py", "")
    binding_cpp = bundle.files.get("binding.cpp", "")
    kernel_mu = bundle.files.get("kernel.mu", "")

    # 1. Toolchain: native MUSA bundles must build with torch_musa.
    if "torch.utils.cpp_extension" in setup_py or "cpp_extension import" in setup_py:
        return (
            "setup.py uses the CUDA extension toolchain "
            "(torch.utils.cpp_extension) instead of "
            "torch_musa.utils.musa_extension.{MUSAExtension, BuildExtension}; "
            "mcc refuses kernel.mu files otherwise"
        )
    if "torch_musa.utils.musa_extension" not in setup_py:
        return (
            "setup.py must import torch_musa.utils.musa_extension "
            "MUSAExtension/BuildExtension to compile this MUSA bundle"
        )

    # 2. Extension-name shadowing of the kernel.py wrapper.
    # Only ``name == 'kernel'`` is genuinely fatal: the compiled extension is a
    # module named ``kernel`` that shadows the wrapper *file* ``kernel.py``, so
    # ``from kernel import kernel_function`` resolves to the extension (which
    # does not export ``kernel_function``) instead of the Python wrapper. An
    # extension whose name matches a module the wrapper imports is the normal,
    # correct contract (the wrapper calls extension entry points), so it is
    # deliberately NOT flagged here.
    setup_names = set(_EXTENSION_NAME_RE.findall(setup_py))
    for ext_name in setup_names:
        if ext_name == "kernel":
            return (
                f"setup.py MUSAExtension name='kernel' shadows the kernel.py "
                f"wrapper module: the built extension shadows kernel.py, so "
                f"'from kernel import kernel_function' resolves to the compiled "
                f"extension (which does not export kernel_function) instead of "
                f"the wrapper"
            )

    # 3. Relative imports cannot work when a bundle runs as a top-level script.
    if _RELATIVE_IMPORT_RE.search(kernel_py):
        return (
            "kernel.py uses a relative import (from . import ...), which fails "
            "because the bundle runs as a top-level script with no parent "
            "package; use an absolute import instead"
        )

    # 4. Device builtins belong inside a __global__ kernel body, not host glue.
    # Scope is tracked with true brace depth: a kernel body opens at the first
    # "{" seen at or after its "__global__" marker and closes only when that
    # same braces level is closed. Inner blocks (for/if) merely change the
    # running depth. The previous implementation treated every line starting
    # with "}" as a possible kernel exit, so any builtin referenced after the
    # first inner block of a __global__ kernel (the standard shared-memory
    # reduction pattern) was falsely rejected as host code.
    for source, filename in ((binding_cpp, "binding.cpp"), (kernel_mu, "kernel.mu")):
        pending_global = False  # saw "__global__", waiting for its body "{"
        global_base = None      # brace depth where the current kernel body lives
        depth = 0               # running open-brace depth
        for lineno, line in enumerate(source.splitlines(), start=1):
            if global_base is None and not pending_global and "__global__" in line:
                pending_global = True
            opens = line.count("{")
            closes = line.count("}")
            if pending_global and opens:
                # The first scope opened at/after a __global__ marker is the body.
                global_base = depth + 1
                pending_global = False
            depth += opens - closes
            if global_base is not None and depth < global_base:
                global_base = None  # the kernel body closed on this line
            if global_base is None and any(
                tok in line for tok in _MUSA_DEVICE_BUILTIN_TOKENS
            ):
                offending = next(tok for tok in _MUSA_DEVICE_BUILTIN_TOKENS if tok in line)
                return (
                    f"{filename}:{lineno}: MUSA/CUDA device builtin "
                    f"({offending.rstrip('.')}) appears outside any __global__ "
                    f"kernel body (e.g. in host launch_* glue); move the "
                    f"computation into a __global__ kernel function"
                )

    return None
