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

"""Smoke tests for fix 6-2 (output budget slimming) prompts.

The budget contract is prompt-layer only (no response-length validation code),
so these tests pin the text the model actually receives: a token target, a
"start at the FILE: header" anchor, a comment cap, dtype-on-demand, and the
canonical setup.py shape.
"""

from triton_kernel_agent.kernel_backend import get_kernel_backend
from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_manager import PromptManager


RMSNORM_PROBLEM = """
import torch
M, N = 1024, 4096
EPS = 1e-6
class Model:
    def forward(self, x, weight):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + EPS) * weight
def get_inputs():
    return [
        torch.randn((M, N), dtype=torch.bfloat16),
        torch.ones((N,), dtype=torch.bfloat16),
    ]
def get_init_inputs():
    return [N, EPS]
"""


def _manager():
    return PromptManager(
        target_platform=get_platform("musa"), kernel_backend=get_kernel_backend("musa")
    )


def test_generation_prompt_carries_output_budget():
    prompt = _manager().render_kernel_generation_prompt(
        problem_description=RMSNORM_PROBLEM,
        test_code="from kernel import kernel_function",
    )
    assert "13000 tokens" in prompt
    # first-token anchor: no reasoning preamble before the blocks
    assert 'starts at the "F" of the first "FILE:" header' in prompt
    assert "5 comment lines" in prompt
    assert "float64" in prompt
    assert "MUSAExtension" in prompt  # canonical setup.py shape


def test_specialized_generation_template_inherits_budget():
    # Category templates ``include`` the generic one, so norm routing keeps the
    # budget contract (regression guard for future category templates).
    prompt = _manager().render_kernel_generation_prompt(
        problem_description=RMSNORM_PROBLEM,
        test_code="from kernel import kernel_function",
    )
    assert "Normalization generation strategy" in prompt
    assert "13000 tokens" in prompt


def test_refinement_prompt_carries_output_budget():
    prompt = _manager().render_kernel_refinement_prompt(
        problem_description=RMSNORM_PROBLEM,
        test_code="from kernel import kernel_function",
        kernel_code="FILE: kernel.py\n```python\ndef kernel_function(x): return x\n```\n",
        error_info={"stdout": "mismatch", "stderr": ""},
    )
    assert "13000 tokens" in prompt
    assert "Do NOT open with an" in prompt
    assert "<= 5 comment lines per file" in prompt
    assert "Keep setup.py as-is" in prompt
