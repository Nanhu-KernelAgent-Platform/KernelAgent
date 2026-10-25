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

"""OpenAI provider implementation."""

from .openai_base import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider."""

    def __init__(self):
        import os
        base_url = os.getenv("OPENAI_BASE_URL")
        super().__init__(api_key_env="OPENAI_API_KEY", base_url=base_url)

    @property
    def name(self) -> str:
        return "openai"

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get the output-token limit used by KernelAgent requests."""
        if "musacoder" in model_name.lower():
            # A complete 4-file native MUSA bundle (kernel.py, binding.cpp,
            # kernel.mu, setup.py) for a forward+backward operator exceeds
            # 16384 output tokens; at that cap the reply is cut mid-file and the
            # bundle is rejected as "missing required files". The model supports
            # a 262144-token context and the server honours MAX_TOKENS_CAP, so
            # request 32768. KERNELAGENT_OUTPUT_CAP still overrides this value.
            return 32768
        if model_name.startswith("gpt-5.6"):
            return 128000
        if model_name.startswith(("gpt-5", "gpt-4", "o3", "o1")):
            return 32000
        elif model_name.startswith("gpt-3.5"):
            return 16000
        else:
            return 8192
