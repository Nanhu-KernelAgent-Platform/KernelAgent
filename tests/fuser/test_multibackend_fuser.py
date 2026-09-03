import json
from pathlib import Path

import pytest

from Fuser.compose_end_to_end import _load_kernels_from_summary
from Fuser.dispatch_kernel_agent import _synthesize_problem_description
from Fuser.pipeline import run_pipeline
from triton_kernel_agent.kernel_backend import KernelBundle, get_kernel_backend
from triton_kernel_agent.platform_config import get_platform


def test_backend_platform_contract():
    assert get_kernel_backend("triton").supports_platform("musa")
    assert get_kernel_backend("musa").supports_platform("musa")
    assert not get_kernel_backend("musa").supports_platform("cuda")


def test_musa_dispatch_prompt_requests_bundle():
    prompt = _synthesize_problem_description(
        {
            "id": "sg_1",
            "type": "elementwise",
            "input_shape": [8],
            "output_shape": [8],
            "ops": [{"op": "relu"}],
        },
        get_platform("musa"),
        kernel_backend="musa",
    )
    assert "binding.cpp" in prompt
    assert "kernel.mu" in prompt
    assert "MUSAExtension" in prompt
    assert "@triton.jit" not in prompt


def test_load_bundle_summary(tmp_path: Path):
    artifact_dir = tmp_path / "sg_1"
    artifact_dir.mkdir()
    files = {
        "kernel.py": "def kernel_function(x): return x\n",
        "binding.cpp": "void launch();\n",
        "kernel.mu": "__global__ void launch() {}\n",
        "setup.py": "from setuptools import setup\n",
    }
    file_paths = {}
    for name, content in files.items():
        path = artifact_dir / name
        path.write_text(content, encoding="utf-8")
        file_paths[name] = str(path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            [
                {
                    "id": "sg_1",
                    "success": True,
                    "kernel_backend": "musa",
                    "kernel_path": file_paths["kernel.py"],
                    "artifact_dir": str(artifact_dir),
                    "files": file_paths,
                }
            ]
        ),
        encoding="utf-8",
    )
    items = _load_kernels_from_summary(summary, expected_backend="musa")
    assert len(items) == 1
    assert set(items[0].files) == set(files)


def test_bundle_rejects_backend_mismatch(tmp_path: Path):
    summary = tmp_path / "summary.json"
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel_function(x): return x\n", encoding="utf-8")
    summary.write_text(
        json.dumps(
            [
                {
                    "id": "sg_1",
                    "success": True,
                    "kernel_backend": "triton",
                    "kernel_path": str(kernel),
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="backend mismatch"):
        _load_kernels_from_summary(summary, expected_backend="musa")


def test_pipeline_rejects_incompatible_backend(tmp_path: Path):
    problem = tmp_path / "problem.py"
    problem.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="does not support platform"):
        run_pipeline(
            problem_path=problem,
            extract_model="unused",
            dispatch_model="unused",
            compose_model="unused",
            dispatch_jobs=1,
            workers=1,
            max_iters=1,
            llm_timeout_s=1,
            run_timeout_s=1,
            target_platform="cuda",
            kernel_backend="musa",
        )


def test_bundle_file_contract():
    bundle = KernelBundle(
        {
            "kernel.py": "def kernel_function(x): return x",
            "binding.cpp": "void launch();",
            "kernel.mu": "__global__ void launch() {}",
            "setup.py": "from setuptools import setup",
        }
    )
    bundle.require(get_kernel_backend("musa"))
