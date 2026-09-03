from pathlib import Path

import pytest

from triton_kernel_agent.kernel_backend import (
    KernelBundle,
    extract_kernel_bundle,
    get_kernel_backend,
)
from triton_kernel_agent.worker import VerificationWorker


def test_extract_complete_musa_bundle():
    text = """
FILE: kernel.py
```python
def kernel_function(x): return x
```
FILE: binding.cpp
```cpp
void launch();
```
FILE: kernel.mu
```cpp
__global__ void kernel() {}
```
FILE: setup.py
```python
from setuptools import setup
```
"""
    bundle = extract_kernel_bundle(text)
    assert bundle is not None
    bundle.require(get_kernel_backend("musa"))


def test_extract_fenced_file_header_bundle():
    response = """```FILE kernel.py
def kernel_function():
    return 1
```
```FILE binding.cpp
// binding
```
```FILE kernel.mu
// kernel
```
```FILE setup.py
# setup
```
"""
    bundle = extract_kernel_bundle(response)
    assert bundle is not None
    bundle.require(get_kernel_backend("musa"))
    assert "def kernel_function" in bundle.files["kernel.py"]


@pytest.mark.parametrize(
    "kernel_header",
    [
        "### FILE: kernel.py",
        "FILE: `kernel.py`",
        "**FILE: kernel.py**",
        "### `kernel.py`",
    ],
)
def test_extract_musa_bundle_accepts_markdown_decorated_headers(kernel_header):
    text = f"""
{kernel_header}
```python
def kernel_function(x): return x
```
FILE: binding.cpp
```cpp
void launch();
```
FILE: kernel.mu
```cpp
__global__ void kernel() {{}}
```
FILE: setup.py
```python
from setuptools import setup
```
"""
    bundle = extract_kernel_bundle(text)
    assert bundle is not None
    bundle.require(get_kernel_backend("musa"))
    assert set(bundle.files) == {"kernel.py", "binding.cpp", "kernel.mu", "setup.py"}


def test_extract_musa_bundle_accepts_fence_filename_metadata():
    text = """
```python filename=kernel.py
def kernel_function(x): return x
```
```cpp filename=binding.cpp
void launch();
```
```cpp filename=kernel.mu
__global__ void kernel() {}
```
```python filename=setup.py
from setuptools import setup
```
"""
    bundle = extract_kernel_bundle(text)
    assert bundle is not None
    bundle.require(get_kernel_backend("musa"))


def test_native_musa_extension_forward_is_not_torch_fallback(tmp_path):
    (tmp_path / "work").mkdir()
    (tmp_path / "logs").mkdir()
    worker = VerificationWorker(
        worker_id=0,
        workdir=tmp_path / "work",
        log_dir=tmp_path / "logs",
        target_platform="musa",
        kernel_backend="musa",
    )
    bundle = KernelBundle(
        {
            "kernel.py": "def kernel_function(x):\n    return _extension().forward(x)\n",
            "binding.cpp": "// binding",
            "kernel.mu": "// native kernel",
            "setup.py": "# setup",
        }
    )
    assert worker._detect_pytorch_compute(bundle) is None
    assert set(bundle.files) == {"kernel.py", "binding.cpp", "kernel.mu", "setup.py"}


@pytest.mark.parametrize("name", ["../secret", "/absolute", "nested/kernel.py"])
def test_bundle_rejects_unsafe_paths(name):
    with pytest.raises(ValueError):
        KernelBundle({name: "content"})


def test_bundle_writes_are_safe_under_workdir(tmp_path: Path):
    bundle = KernelBundle({"kernel.py": "def kernel_function(): pass"})
    for name, content in bundle.files.items():
        (tmp_path / name).write_text(content)
    assert (tmp_path / "kernel.py").is_file()
