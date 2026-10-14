import logging
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from triton_kernel_agent.opt_worker_component.profiling.ncu_wrapper_factory import (
    NCUWrapperFactory,
)


def test_musa_wrapper_uses_torch_musa(tmp_path):
    kernel = tmp_path / "kernel.py"
    problem = tmp_path / "problem.py"
    kernel.write_text("def kernel_function(x): return x")
    problem.write_text("def get_inputs(): return []\ndef get_init_inputs(): return []")

    wrapper = NCUWrapperFactory(logging.getLogger(__name__)).create_ncu_wrapper(
        kernel, problem, tmp_path, target_platform="musa"
    )
    code = wrapper.read_text()
    assert "import torch_musa" in code
    assert "torch.musa.synchronize()" in code
    assert ".cuda()" not in code


def test_wrapper_does_not_blindly_pass_model_init_args_to_kernel(tmp_path):
    kernel = tmp_path / "kernel.py"
    problem = tmp_path / "problem.py"
    kernel.write_text("def kernel_function(x): return x\n")
    problem.write_text(
        "def get_inputs(): return [object()]\n" "def get_init_inputs(): return [64]\n"
    )

    wrapper = NCUWrapperFactory(logging.getLogger(__name__)).create_ncu_wrapper(
        kernel, problem, tmp_path, target_platform="musa"
    )
    code = wrapper.read_text()
    assert "bind_kernel_call(kernel_function, model, cuda_inputs)" in code
    assert "kernel_function(*cuda_inputs, *init_inputs)" not in code
    assert (tmp_path / "kernel_call_binding.py").is_file()


@pytest.mark.parametrize('device', ['musa', 'cuda'])
def test_executed_wrapper_preserves_nested_input_layout_and_dtype(tmp_path, monkeypatch, device):
    kernel = tmp_path / 'layout_contract_kernel.py'
    problem = tmp_path / 'layout_contract_problem.py'
    kernel.write_text('# module supplied below\n')
    problem.write_text('# module supplied below\n')
    original = torch.randn(5, 14)[:, ::2]
    calls = []

    def kernel_function(payload):
        # float16 in source must not silently change float32 reference inputs.
        value = payload['x']
        assert value.shape == original.shape and value.stride() == (14, 2)
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, original)
        calls.append(value)
        return value

    fake_torch = ModuleType('torch')
    fake_torch.__dict__.update(torch.__dict__)
    fake_torch.empty_strided = lambda *args, **kwargs: torch.empty_strided(
        *args, **dict(kwargs, device='cpu'))
    setattr(fake_torch, device, SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'torch_musa', ModuleType('torch_musa'))
    monkeypatch.setitem(sys.modules, kernel.stem, SimpleNamespace(kernel_function=kernel_function))
    monkeypatch.setitem(sys.modules, problem.stem, SimpleNamespace(
        get_inputs=lambda: [{'x': original}], get_init_inputs=lambda: []))
    monkeypatch.syspath_prepend(str(tmp_path))
    wrapper = NCUWrapperFactory(logging.getLogger(__name__)).create_ncu_wrapper(
        kernel, problem, tmp_path, target_platform=device, dtype_inference=False)
    assert (tmp_path / 'kernel_case_suite.py').is_file()
    runpy.run_path(str(wrapper))
    assert len(calls) == (1 if device == 'musa' else 4)
