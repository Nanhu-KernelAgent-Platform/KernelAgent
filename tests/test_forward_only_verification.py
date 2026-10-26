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

"""Fix 7-1: forward-only verification for KernelBench-style scoring.

KernelBench scores the forward output only. These tests pin that the generated
correctness test stops demanding a correct backward (the "Weight gradient
mismatch" failures happened after the forward case had already passed).
"""

from unittest.mock import Mock

import pytest

from triton_kernel_agent.agent import TritonKernelAgent
from triton_kernel_agent.kernel_backend import get_kernel_backend
from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_manager import PromptManager
from triton_kernel_agent.test_validation import validate_forward_only_test
from triton_kernel_agent.verification_policy import (
    is_forward_only,
    resolve_verify_direction,
)

PROBLEM = '''
import torch
class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)
def get_inputs():
    return [torch.randn(4, 8)]
def get_init_inputs():
    return []
'''

FORWARD_TEST = '''
import torch
import torch.nn as nn
from kernel_case_suite import move, bind_kernel_call
from kernel import kernel_function

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(4, 8)]

def get_init_inputs():
    return []

def main():
    inputs = move(get_inputs(), 'musa')
    model = Model(*get_init_inputs()).to('musa')
    with torch.no_grad():
        ref = model(inputs[0])
    native = bind_kernel_call(kernel_function, model, inputs)()
    torch.testing.assert_close(ref, native, rtol=1e-3, atol=1e-3)
    print("PASS")

main()
'''

BACKWARD_TEST = FORWARD_TEST.replace(
    '    native = bind_kernel_call(kernel_function, model, inputs)()',
    '    native = bind_kernel_call(kernel_function, model, inputs)()\n'
    '    (ref * 2.0).sum().backward()'
)


def _manager():
    return PromptManager(
        target_platform=get_platform("musa"), kernel_backend=get_kernel_backend("musa")
    )


def _agent(tmp_path):
    (tmp_path / "logs").mkdir(exist_ok=True)
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.provider = Mock()
    agent.model_name = "MusaCoder-27B"
    agent.high_reasoning_effort = False
    agent.logger = Mock()
    agent.prompt_manager = _manager()
    agent._platform_config = get_platform("musa")
    agent._kernel_backend = get_kernel_backend("musa")
    return agent


def test_forward_only_prompt_replaces_backward_contract():
    manager = _manager()
    forward = manager.render_test_generation_prompt(
        PROBLEM, None, verify_direction="forward"
    )
    legacy = manager.render_test_generation_prompt(PROBLEM, None)

    assert "FORWARD-ONLY CONTRACT" in forward
    assert "For training/backward" not in forward
    assert "runs under `torch.no_grad()`" in forward
    # legacy contract unchanged
    assert "FORWARD-ONLY CONTRACT" not in legacy
    assert "For training/backward" in legacy


def test_forward_only_test_guard():
    validate_forward_only_test(FORWARD_TEST)  # accepted
    for source in (
        BACKWARD_TEST,
        BACKWARD_TEST.replace(".backward()", ".backward(grad)"),
        "import torch\ntorch.autograd.grad(out, x)\n",
    ):
        with pytest.raises(ValueError, match="forward-only"):
            validate_forward_only_test(source)


def test_resolve_verify_direction(monkeypatch):
    monkeypatch.delenv("KERNELAGENT_VERIFY_DIRECTION", raising=False)
    assert resolve_verify_direction(None) is None
    assert resolve_verify_direction("forward") == "forward"
    assert is_forward_only("forward") and not is_forward_only("joint")
    monkeypatch.setenv("KERNELAGENT_VERIFY_DIRECTION", "forward")
    assert resolve_verify_direction(None) == "forward"
    monkeypatch.setenv("KERNELAGENT_VERIFY_DIRECTION", "auto")
    assert resolve_verify_direction(None) is None
    with pytest.raises(ValueError, match="invalid verification direction"):
        resolve_verify_direction("sideways")


def test_agent_rejects_backward_test_in_forward_only_mode(tmp_path):
    agent = _agent(tmp_path)
    responses = iter([
        "```python\n" + BACKWARD_TEST + "\n```",
        "```python\n" + FORWARD_TEST + "\n```",
    ])
    agent._call_llm = Mock(side_effect=lambda *_a, **_k: next(responses))
    agent._extract_code_from_response = Mock(
        side_effect=lambda text: text.split("```python\n")[1].rsplit("\n```", 1)[0]
    )

    test_code = agent._generate_test(PROBLEM, None, verify_direction="forward")

    assert ".backward()" not in test_code
    assert agent._call_llm.call_count == 2
    prompt = agent._call_llm.call_args_list[0].args[0][0]["content"]
    assert "FORWARD-ONLY CONTRACT" in prompt
    # the repair message tells the model to drop the grad steps
    repair = agent._call_llm.call_args_list[1].args[0][-1]["content"]
    assert "FORWARD-ONLY" in repair


def test_agent_forward_only_accepts_clean_test_first_try(tmp_path):
    agent = _agent(tmp_path)
    agent._call_llm = Mock(return_value="```python\n" + FORWARD_TEST + "\n```")
    agent._extract_code_from_response = Mock(
        side_effect=lambda text: text.split("```python\n")[1].rsplit("\n```", 1)[0]
    )

    test_code = agent._generate_test(PROBLEM, None, verify_direction="forward")

    assert "torch.no_grad()" in test_code
    assert agent._call_llm.call_count == 1
