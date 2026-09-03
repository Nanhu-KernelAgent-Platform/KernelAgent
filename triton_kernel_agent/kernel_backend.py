# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

"""Kernel code-generation backend definitions and multi-file payload helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

DEFAULT_KERNEL_BACKEND = "triton"
_FILE_BLOCK_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:#{1,6}[ \t]+)?(?:\*{1,2})?"
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

_CANONICAL_MUSA_FILENAMES = {
    name.lower(): name
    for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
}


def _normalize_generated_filename(raw_name: str) -> str:
    """Normalize harmless Markdown decoration around a generated filename."""

    name = raw_name.strip()
    name = re.sub(r"^(?:filename|file)[ \t]*[:=][ \t]*", "", name, flags=re.I)
    name = name.strip(" \t`*_'\"")
    name = name.removeprefix("./")

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
    if not matches:
        return None
    return KernelBundle(
        {
            _normalize_generated_filename(name): content.strip()
            for name, content in matches
        }
    )
