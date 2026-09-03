"""Collect runtime metadata without initializing an accelerator context."""

from __future__ import annotations

import os
import platform as host_platform
import sys
from importlib.metadata import PackageNotFoundError, version


def collect_environment() -> dict[str, str]:
    result = {
        "python": sys.version.split()[0],
        "host": host_platform.platform(),
    }
    for package in ("torch", "torch-musa", "triton"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    for variable in ("MUSA_HOME", "MUSA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(variable)
        if value:
            result[variable.lower()] = value
    return result
