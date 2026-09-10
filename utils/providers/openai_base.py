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

"""Base provider for OpenAI-compatible APIs."""

import logging
import os
from typing import Any
from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


OPENAI_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
_REASONING_EFFORT_ALIASES = {
    "不推理": "none",
    "无": "none",
    "低": "low",
    "中": "medium",
    "中等": "medium",
    "高": "high",
    "极高": "xhigh",
    "最高": "max",
    "最大": "max",
}


def normalize_reasoning_effort(value: Any) -> str | None:
    """Normalize and validate OpenAI reasoning effort configuration."""

    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    normalized = _REASONING_EFFORT_ALIASES.get(normalized, normalized)
    if normalized not in OPENAI_REASONING_EFFORTS:
        raise ValueError(
            f"Unsupported OpenAI reasoning effort {value!r}. Expected one of: "
            f"{', '.join(OPENAI_REASONING_EFFORTS)}"
        )
    return normalized


class OpenAICompatibleProvider(BaseProvider):
    """Base provider for OpenAI-compatible APIs."""

    def __init__(self, api_key_env: str, base_url: str | None = None):
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        if not OPENAI_AVAILABLE:
            return

        api_key = self._get_api_key(self.api_key_env)
        if api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            # Initialize client (proxy configured via environment variables)
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": float(os.getenv("OPENAI_TIMEOUT_S", "600")),
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)

    def _uses_responses_api(self) -> bool:
        return os.getenv("OPENAI_WIRE_API", "chat_completions").lower() == "responses"

    @staticmethod
    def _usage_dict(response: Any) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "dict"):
            return usage.dict()
        return usage if isinstance(usage, dict) else None

    def _build_responses_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        effort = normalize_reasoning_effort(
            kwargs.get("reasoning_effort")
            or os.getenv("OPENAI_REASONING_EFFORT")
            or ("high" if kwargs.get("high_reasoning_effort") else None)
        )
        params: dict[str, Any] = {
            "model": model_name,
            "input": messages,
            "max_output_tokens": min(
                kwargs.get("max_tokens", 8192),
                self.get_max_tokens_limit(model_name),
            ),
            "store": False,
        }
        if effort:
            params["reasoning"] = {"effort": effort}
        return params

    def _get_responses_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        response = self.client.responses.create(
            **self._build_responses_params(model_name, messages, **kwargs)
        )
        logging.getLogger(__name__).info(
            "OpenAI Responses response: id=%s model=%s",
            getattr(response, "id", None),
            getattr(response, "model", model_name),
        )
        return LLMResponse(
            content=getattr(response, "output_text", "") or "",
            model=getattr(response, "model", model_name),
            provider=self.name,
            usage=self._usage_dict(response),
            response_id=getattr(response, "id", None),
        )

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Get single response."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        if self._uses_responses_api():
            return self._get_responses_response(model_name, messages, **kwargs)

        api_params = self._build_api_params(model_name, messages, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (single): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_name,
            provider=self.name,
            usage=self._usage_dict(response),
        )

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        """Get multiple responses using n parameter."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        if self._uses_responses_api():
            return [
                self._get_responses_response(model_name, messages, **kwargs)
                for _ in range(n)
            ]

        api_params = self._build_api_params(model_name, messages, n=n, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (multi): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return [
            LLMResponse(
                content=choice.message.content,
                model=model_name,
                provider=self.name,
                usage=self._usage_dict(response),
            )
            for choice in response.choices
        ]

    def _build_api_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        """Build API parameters for OpenAI-compatible call."""
        params = {
            "model": model_name,
            "messages": messages,
        }

        # GPT-5 and o-series models pin their own sampling behaviour
        if not (model_name.startswith("gpt-5") or model_name.startswith("o")):
            params["temperature"] = kwargs.get("temperature", 0.7)

        # Use max_completion_tokens for newer models like GPT-5, fallback to max_tokens
        max_tokens_value = min(
            kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
        )
        if model_name.startswith("gpt-5") or model_name.startswith("o"):
            params["max_completion_tokens"] = max_tokens_value
        else:
            params["max_tokens"] = max_tokens_value

        # Add n parameter if specified
        if "n" in kwargs:
            params["n"] = kwargs["n"]

        configured_effort = normalize_reasoning_effort(
            kwargs.get("reasoning_effort") or os.getenv("OPENAI_REASONING_EFFORT")
        )

        # Keep the historical high default unless explicitly configured.
        if model_name.startswith(("gpt-5", "grok-")):
            params["reasoning_effort"] = configured_effort or "high"
        elif kwargs.get("high_reasoning_effort") and model_name.startswith(
            ("o3", "o1")
        ):
            params["reasoning_effort"] = "high"

        return params

    def is_available(self) -> bool:
        """Check if provider is available."""
        return OPENAI_AVAILABLE and self.client is not None

    def supports_multiple_completions(self) -> bool:
        """OpenAI-compatible APIs support native multiple completions."""
        return True
