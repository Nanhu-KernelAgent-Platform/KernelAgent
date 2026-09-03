import logging

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
        "def get_inputs(): return [object()]\n"
        "def get_init_inputs(): return [64]\n"
    )

    wrapper = NCUWrapperFactory(logging.getLogger(__name__)).create_ncu_wrapper(
        kernel, problem, tmp_path, target_platform="musa"
    )
    code = wrapper.read_text()
    assert "required_kernel_positional_count" in code
    assert "missing = max(0, required_kernel_positional_count - len(call_args))" in code
    assert "kernel_function(*cuda_inputs, *init_inputs)" not in code
