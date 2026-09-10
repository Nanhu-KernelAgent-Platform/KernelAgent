# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Grok provider for third-party OpenAI-compatible endpoints."""

import os

from .openai_base import OpenAICompatibleProvider


class GrokProvider(OpenAICompatibleProvider):
    """Route Grok through the configured OpenAI-compatible API and credential."""

    def __init__(self):
        super().__init__(
            api_key_env="OPENAI_API_KEY",
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    @property
    def name(self) -> str:
        # A distinct name keeps Fuser on the provider interface instead of its
        # OpenAI-only Responses streaming adapter.
        return "grok"

    def get_max_tokens_limit(self, model_name: str) -> int:
        return 128000

    def supports_multiple_completions(self) -> bool:
        """Third-party Grok endpoints may reject the OpenAI ``n`` parameter."""
        return False
