from unittest.mock import Mock
import multiprocessing as mp

import pytest

from triton_kernel_agent.agent import TritonKernelAgent
from triton_kernel_agent.kernel_backend import extract_kernel_bundle, get_kernel_backend
from triton_kernel_agent.test_validation import validate_test_autograd, validate_test_layout, validate_kernel_globals
from triton_kernel_agent.worker import LLMCallResult, VerificationWorker
from triton_kernel_agent.prompt_manager import PromptManager
from triton_kernel_agent.platform_config import get_platform


BAD_TEST = '''
import torch
def main():
    x = torch.randn(3, requires_grad=True)
    with torch.no_grad():
        reference_output = model(x)
    upstream = torch.randn_like(reference_output)
    reference_loss = (reference_output * upstream).sum()
    reference_loss.backward()
'''
GOOD_TEST = BAD_TEST.replace("torch.no_grad()", "torch.enable_grad()")


def bundle_text(comment_headers=False):
    # setup.py must satisfy validate_musa_bundle_semantics (fix 2-2): the
    # MUSA toolchain import is mandatory, a CUDA cpp_extension import is
    # rejected.
    files = {"kernel.py": "def kernel_function(x): return extension.forward(x)",
             "binding.cpp": "void launch();", "kernel.mu": "void launch() {}",
             "setup.py": ("from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
                          "from setuptools import setup")}
    return "\n".join(
        (f"```\n{'#' if name.endswith('.py') else '//'} FILE: {name}\n{body}\n```"
         if comment_headers else f"FILE: {name}\n```\n{body}\n```")
        for name, body in files.items())


def worker(tmp_path):
    (tmp_path / "work").mkdir()
    (tmp_path / "logs").mkdir()
    return VerificationWorker(0, tmp_path / "work", tmp_path / "logs",
                              target_platform="musa", kernel_backend="musa")


@pytest.mark.parametrize('violation', [None, 'forbidden framework operation'])
def test_run_uses_same_refinement_path_without_losing_failure_evidence(tmp_path, violation):
    w = worker(tmp_path)
    bundle = extract_kernel_bundle(bundle_text())
    w._single_verification_pass = Mock(side_effect=[
        (False, 'diagnostic output', 'numerical mismatch', violation),
        (True, 'PASS', '', None)])
    w._refine_kernel = Mock(return_value=bundle)
    result = w.run(bundle, ['print("PASS")'], 'formula', mp.Event())
    assert result['success'] and result['rounds'] == 2
    assert len(result['history']) == 2
    failure = w._refine_kernel.call_args.args[1]
    assert failure['stdout'] == ('' if violation else 'diagnostic output')
    assert failure['stderr'] == (violation or 'numerical mismatch')
    assert len(failure['history']) == 1


def test_common_refinement_failure_still_rejects_malformed_result(tmp_path):
    w = worker(tmp_path)
    w._single_verification_pass = Mock(return_value=(False, '', 'mismatch', None))
    w._refine_kernel = Mock(return_value='not a kernel bundle')
    result = w.run(extract_kernel_bundle(bundle_text()), ['print("PASS")'], 'formula', mp.Event())
    # Fix P1 B: a malformed response is still rejected (never accepted as a
    # passing kernel), but it now consumes every round instead of terminating
    # the run after one attempt, so the max_rounds budget really applies.
    assert not result['success']
    assert result['rounds'] == w.max_rounds
    assert result.get('max_rounds_reached') is True
    # history is a deque capped at history_size (8 by default), not max_rounds.
    assert len(result['history']) == w.history_size


@pytest.mark.parametrize('source', [
    'import torch\ntorch.Tensor.to = replacement',
    'import torch as t\nt.Tensor.to = replacement',
    'from torch import Tensor as T\nT.to = replacement',
    'import torch\nT = torch.Tensor\nsetattr(T, "to", replacement)',
    'import torch\ndel torch.testing.assert_close',
    'import torch\ntorch.__dict__["allclose"] = replacement',
    'import torch\ntorch.__dict__.update(allclose=replacement)',
    'from unittest.mock import patch\npatch("torch.allclose", replacement)',
    'import sys\nsys.exit = replacement',
])
def test_visible_framework_global_modifications_rejected(source):
    with pytest.raises(ValueError, match='verification behavior'):
        validate_kernel_globals(source)


def test_normal_native_wrapper_does_not_trigger_global_guard():
    validate_kernel_globals('import torch\nimport extension\ndef kernel_function(x): return extension.forward(x)')


@pytest.mark.parametrize('transfer', [
    'inputs = [value.to(device="musa") for value in get_inputs()]',
    'factory_inputs = get_inputs()\ninputs = [value.to("musa") for value in factory_inputs]',
    'inputs = get_inputs()\nx = inputs[0].cuda()',
])
def test_factory_input_transfers_require_shared_layout_mover(transfer):
    with pytest.raises(ValueError, match='Repair the test, not the kernel'):
        validate_test_layout(transfer)
    validate_test_layout('from kernel_case_suite import move\ninputs = move(get_inputs(), "musa")\nmodel.to("musa")')


def test_layout_error_repairs_test_before_kernel_generation():
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.provider = object()
    agent.model_name = 'test-model'
    agent._platform_config = get_platform('musa')
    agent.prompt_manager = Mock()
    agent.logger = Mock()
    bad = 'inputs = [x.to("musa") for x in get_inputs()]'
    good = 'from kernel_case_suite import move\ninputs = move(get_inputs(), "musa")'
    agent._call_llm = Mock(side_effect=[bad, good])
    agent._extract_code_from_response = lambda text: text
    assert agent._generate_test('formula without factories') == good
    assert agent._call_llm.call_count == 2


def test_worker_rejects_bad_layout_test_without_kernel_refinement(tmp_path):
    w = worker(tmp_path)
    w._refine_kernel = Mock()
    result = w.run(extract_kernel_bundle(bundle_text()),
                   ['inputs = [x.to("musa") for x in get_inputs()]'], 'formula', mp.Event())
    assert result['failure_kind'] == 'invalid_correctness_test'
    w._refine_kernel.assert_not_called()


def test_worker_rejects_framework_patch_in_any_python_bundle_file(tmp_path):
    w = worker(tmp_path)
    for filename in ('kernel.py', 'setup.py'):
        bundle = extract_kernel_bundle(bundle_text())
        bundle.files[filename] += '\nimport torch\ntorch.Tensor.to = replacement\n'
        assert 'verification behavior' in w._validate_kernel_candidate(bundle)


def test_comment_headers_parse_complete_bundle():
    bundle = extract_kernel_bundle(bundle_text(True))
    bundle.require(get_kernel_backend("musa"))
    assert bundle.files["kernel.py"].startswith("def kernel_function")


@pytest.mark.parametrize("name", ["../kernel.py", "/tmp/kernel.py", "nested/kernel.py"])
def test_comment_headers_do_not_normalize_unsafe_paths(name):
    with pytest.raises(ValueError, match="Unsafe|top-level"):
        extract_kernel_bundle(f"```python\n# FILE: {name}\ndef kernel_function(): pass\n```")


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_reference_backward_disabled_grad_rejected(mode):
    with pytest.raises(ValueError, match="repair the test, not the kernel"):
        validate_test_autograd(BAD_TEST.replace("no_grad", mode))


def test_forward_no_grad_and_separate_backward_allowed():
    validate_test_autograd(GOOD_TEST)
    validate_test_autograd(BAD_TEST.replace("reference_loss.backward()", "print(reference_loss)"))
    validate_test_autograd('''
def main():
    with torch.no_grad():
        forward_only = model(x)
        with torch.enable_grad():
            reference = model(leaf)
    reference.sum().backward()
''')


def test_autograd_grad_disabled_reference_rejected():
    with pytest.raises(ValueError, match="Invalid correctness test"):
        validate_test_autograd(BAD_TEST.replace("reference_loss.backward()",
                                              "torch.autograd.grad(reference_loss, x)"))


def test_detached_target_in_differentiable_loss_is_valid():
    validate_test_autograd('''
def main():
    x = torch.randn(3, requires_grad=True)
    with torch.no_grad():
        target = model(x)
    native = kernel_function(x)
    loss = ((native - target) ** 2).sum()
    loss.backward()
''')


def test_upstream_shape_metadata_does_not_restore_reference_gradient():
    with pytest.raises(ValueError, match="Invalid correctness test"):
        validate_test_autograd('''
def main():
    x = torch.randn(3, requires_grad=True)
    with torch.no_grad():
        reference_output = model(x)
    native_output = kernel_function(x)
    upstream = torch.randn(native_output.shape, dtype=native_output.dtype)
    upstream = upstream + torch.where(
        torch.arange(native_output.numel()) % 2 == 0,
        torch.tensor(0.25, dtype=native_output.dtype), torch.tensor(-0.75))
    reference_loss = (reference_output * upstream).sum()
    reference_loss.backward()
''')


def test_reassignment_and_function_scope_do_not_leak():
    validate_test_autograd(BAD_TEST.replace("reference_loss.backward()",
                                          "reference_loss = model(x).sum()\n    reference_loss.backward()"))
    validate_test_autograd('''
def f():
    with torch.no_grad():
        y = model(x)
def g():
    y = model(x)
    y.backward()
''')


def test_invalid_test_repairs_before_kernel_generation():
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.provider = object()
    agent.model_name = "test-model"
    agent.prompt_manager = Mock()
    agent.prompt_manager.render_test_generation_prompt.return_value = "prompt"
    agent.logger = Mock()
    agent._call_llm = Mock(side_effect=[BAD_TEST, GOOD_TEST])
    agent._extract_code_from_response = lambda text: text
    assert agent._generate_test("formula without factories") == GOOD_TEST
    assert agent._call_llm.call_count == 2
    assert "repair the test, not the kernel" in agent._call_llm.call_args.args[0][-1]["content"]


def test_invalid_test_repair_is_bounded():
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.provider = object()
    agent.model_name = "test-model"
    agent.prompt_manager = Mock()
    agent.logger = Mock()
    agent._call_llm = Mock(return_value=BAD_TEST)
    agent._extract_code_from_response = lambda text: text
    with pytest.raises(ValueError, match="Invalid correctness test"):
        agent._generate_test("formula")
    # Fix 4-1 A3: the repair loop is bounded at three attempts.
    assert agent._call_llm.call_count == 3


def test_missing_provider_never_falls_back_to_mock_correctness_test():
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.provider = None
    with pytest.raises(RuntimeError, match='mock fallback disabled'):
        agent._generate_test('formula')


def test_worker_rejects_invalid_test_without_kernel_refinement(tmp_path):
    w = worker(tmp_path)
    w._refine_kernel = Mock()
    w._single_verification_pass = Mock()
    result = w.run(extract_kernel_bundle(bundle_text()), [BAD_TEST], "formula", mp.Event())
    assert result["failure_kind"] == "invalid_correctness_test"
    assert result["rounds"] == 0
    success, _, error = w.verify_with_refinement(extract_kernel_bundle(bundle_text()), [BAD_TEST], "formula")
    assert not success and "kernel refinement skipped" in error
    w._refine_kernel.assert_not_called()
    w._single_verification_pass.assert_not_called()


def test_musa_refinement_has_no_same_round_format_repair(tmp_path):
    # Fix P1 (A): a malformed first response no longer triggers a same-round
    # format repair. It raises so run()'s backstop (fix P1 B) regenerates next
    # round instead of emitting a second full bundle inside this round.
    w = worker(tmp_path)
    w.provider = object()
    w._call_llm_detailed = Mock(side_effect=[
        LLMCallResult("not code", None, None),
        LLMCallResult(bundle_text(True), None, None),
    ])
    with pytest.raises(ValueError, match="Malformed LLM kernel response"):
        w._refine_kernel(extract_kernel_bundle(bundle_text()),
                         {"stdout": "mismatch", "stderr": ""}, "formula", GOOD_TEST)
    assert w._call_llm_detailed.call_count == 1  # no repair second call
    assert (w.log_dir / "refinement_rejected_0_0.txt").read_text() == "not code"


def test_musa_refinement_format_failure_remains_failure(tmp_path):
    w = worker(tmp_path)
    w.provider = object()
    w._call_llm_detailed = Mock(return_value=LLMCallResult("not code", None, None))
    with pytest.raises(ValueError, match="Malformed LLM kernel response"):
        w._refine_kernel(extract_kernel_bundle(bundle_text()),
                         {"stdout": "mismatch", "stderr": ""}, "formula", GOOD_TEST)
    assert w._call_llm_detailed.call_count == 1  # no same-round repair (fix P1 A)


def test_unsafe_refinement_does_not_fall_back_to_original(tmp_path):
    w = worker(tmp_path)
    w.provider = object()
    unsafe = "```python\n# FILE: ../kernel.py\npass\n```"
    w._call_llm_detailed = Mock(return_value=LLMCallResult(unsafe, None, None))
    with pytest.raises(ValueError, match="Malformed LLM kernel response: Unsafe"):
        w._refine_kernel(extract_kernel_bundle(bundle_text()),
                         {"stdout": "mismatch", "stderr": ""}, "formula", GOOD_TEST)
    assert w._call_llm_detailed.call_count == 1  # no same-round repair (fix P1 A)


def test_musa_prompts_cover_generic_repeated_failures():
    manager = PromptManager(target_platform=get_platform("musa"), kernel_backend="musa")
    test_prompt = manager.render_test_generation_prompt("training formula")
    assert "torch.enable_grad()" in test_prompt
    assert "non-leaf" in test_prompt
    assert "nonuniform upstream" in test_prompt
    generation = manager.render_kernel_generation_prompt("formula", GOOD_TEST)
    assert "implementation-only" in generation
    assert "declarations and definitions" in generation
