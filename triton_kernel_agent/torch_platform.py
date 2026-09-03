"""Dispatch common torch accelerator APIs through a platform configuration."""

from __future__ import annotations

from typing import Any

import torch

from triton_kernel_agent.platform_config import PlatformConfig, get_platform


def _config(platform: str | PlatformConfig) -> PlatformConfig:
    return get_platform(platform) if isinstance(platform, str) else platform


def get_backend_module(platform: str | PlatformConfig) -> Any:
    cfg = _config(platform)
    backend = getattr(torch, cfg.torch_namespace, None)
    if backend is None:
        raise RuntimeError(f"torch.{cfg.torch_namespace} is not available")
    return backend


def is_available(platform: str | PlatformConfig) -> bool:
    backend = getattr(torch, _config(platform).torch_namespace, None)
    return bool(backend and backend.is_available())


def synchronize(platform: str | PlatformConfig, device: Any = None) -> None:
    backend = get_backend_module(platform)
    backend.synchronize() if device is None else backend.synchronize(device=device)


def current_device(platform: str | PlatformConfig) -> Any:
    return get_backend_module(platform).current_device()


def get_device_name(platform: str | PlatformConfig, device: Any = None) -> str:
    backend = get_backend_module(platform)
    return (
        backend.get_device_name()
        if device is None
        else backend.get_device_name(device)
    )


def get_device_properties(platform: str | PlatformConfig, device: Any = None) -> Any:
    backend = get_backend_module(platform)
    return (
        backend.get_device_properties()
        if device is None
        else backend.get_device_properties(device)
    )


def empty_cache(platform: str | PlatformConfig) -> None:
    get_backend_module(platform).empty_cache()


def create_event(platform: str | PlatformConfig, **kwargs: Any) -> Any:
    return get_backend_module(platform).Event(**kwargs)


def device_context(platform: str | PlatformConfig, device: Any = None) -> Any:
    return get_backend_module(platform).device(device)
