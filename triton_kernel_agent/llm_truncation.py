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

"""Output-cap truncation detection and continuation (fix 6-1).

Shared by the worker refinement loop and the agent seed-generation path so
both react to a response cut off at the model's output-token cap the same way:
ask the model to continue from where it stopped instead of regenerating whole
bundles (which truncate again because the prompt is even longer).

A complete native MUSA bundle (kernel.py, binding.cpp, kernel.mu, setup.py) for
an operator with forward+backward easily exceeds 16k output tokens. Before this
module, only the worker refinement loop knew about truncation; the agent seed
path recorded the cut-off response as "invalid" and then issued a *full
regeneration repair* under the same cap, so it truncated again and the problem
died with "No valid MUSA bundle" / "no Python kernel code extracted".
"""

from __future__ import annotations

from typing import NamedTuple

# Absorbs server-side cap drift / usage jitter between the truncation threshold
# and the real cap. Absolute (not proportional): its job is to tolerate drift,
# which does not scale with the cap size.
TRUNCATION_MARGIN_TOKENS = 512

# Max continuation segments attempted per LLM call before falling back to the
# legacy full-regeneration format repair.
MAX_CONTINUATIONS = 2

CONTINUATION_PROMPT = (
    "Your previous reply was cut off at the model's output-token limit. "
    "Continue EXACTLY from the character where it stopped. Do not repeat any "
    "file you already emitted in full, do not restart, and do not add "
    "explanations. Emit only the remainder of the interrupted file block and "
    "then the remaining FILE blocks, ending with the last line of "
    "FILE: setup.py."
)


class LLMCallResult(NamedTuple):
    """LLM call outcome with the signals the truncation detector needs."""

    content: str
    completion_tokens: int | None = None
    finish_reason: str | None = None


def looks_truncated(
    result: LLMCallResult, malformed_reason: str | None, threshold: int
) -> bool:
    """True when a malformed response was cut at the model's output cap.

    Signal A (finish_reason == 'length'): universal, zero-config on
    well-behaved servers, fires at any cap size.
    Signal B (completion_tokens >= threshold): fallback for servers that
    report 'stop' at the cap (MusaCoder evidence: all 10 truncated refinement
    responses in the P45-56 batch did exactly that).
    """
    if malformed_reason is None:
        return False  # only malformed responses are continuation candidates
    if result.finish_reason == "length":
        return True  # signal A
    return result.completion_tokens is not None and result.completion_tokens >= threshold
