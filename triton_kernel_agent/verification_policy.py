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

"""Verification-direction policy (fix 7-1: KernelBench forward-only mode).

KernelBench's official metric scores the forward output only. The generated
correctness test used to demand a correct backward as well (dgrad/wgrad/bias
grad), so conv and matmul problems failed on a contract the benchmark never
measures -- observed as repeated "Weight gradient mismatch" while the forward
case had already passed.

``forward`` here means: the generated test verifies forward outputs only, and a
static guard rejects any leftover ``backward()`` / ``autograd.grad`` call so a
stale training-style test cannot reintroduce the stricter contract.
"""

from __future__ import annotations

import os

VERIFY_DIRECTIONS = ("forward", "backward", "joint")

_ENV_NAME = "KERNELAGENT_VERIFY_DIRECTION"
_AUTO_VALUES = {"", "auto", "unknown", "none", "any"}


def resolve_verify_direction(explicit: str | None = None) -> str | None:
    """Resolve the requested verification direction.

    Priority: explicit argument, then ``KERNELAGENT_VERIFY_DIRECTION``.
    ``None`` keeps the legacy behavior (the test template's own contract, which
    asks for backward whenever the reference is differentiable).
    """
    value = (explicit if explicit is not None else os.environ.get(_ENV_NAME, ""))
    value = (value or "").strip().lower()
    if value in _AUTO_VALUES:
        return None
    if value not in VERIFY_DIRECTIONS:
        raise ValueError(
            f"invalid verification direction {value!r}; expected one of "
            f"{', '.join(VERIFY_DIRECTIONS)} or unset"
        )
    return value


def is_forward_only(direction: str | None) -> bool:
    """True when the caller asked to score forward outputs only."""
    return direction == "forward"
