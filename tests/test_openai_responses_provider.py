from types import SimpleNamespace

import pytest

from utils.providers.openai_provider import OpenAIProvider
from utils.providers.openai_base import normalize_reasoning_effort


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_test",
            model=kwargs["model"],
            output_text="OK",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 3}),
        )


def test_responses_api_uses_configured_low_reasoning(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_WIRE_API", "responses")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    provider = OpenAIProvider()
    fake = _FakeResponses()
    provider.client.responses = fake

    response = provider.get_response(
        "gpt-5.6-sol",
        [{"role": "user", "content": "Reply OK"}],
        max_tokens=128,
        high_reasoning_effort=True,
    )

    assert response.content == "OK"
    assert response.response_id == "resp_test"
    assert response.usage == {"total_tokens": 3}
    assert fake.calls[0]["reasoning"] == {"effort": "low"}
    assert fake.calls[0]["store"] is False
    assert fake.calls[0]["max_output_tokens"] == 128


def test_responses_multiple_requests_are_sequential(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_WIRE_API", "responses")
    provider = OpenAIProvider()
    fake = _FakeResponses()
    provider.client.responses = fake

    responses = provider.get_multiple_responses(
        "gpt-5.6-sol", [{"role": "user", "content": "Reply OK"}], n=2
    )

    assert len(responses) == 2
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    "effort", ["none", "low", "medium", "high", "xhigh", "max"]
)
def test_gpt56_supports_all_official_reasoning_efforts(monkeypatch, effort):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_WIRE_API", "responses")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", effort)
    provider = OpenAIProvider()
    fake = _FakeResponses()
    provider.client.responses = fake

    provider.get_response(
        "gpt-5.6-sol", [{"role": "user", "content": "Reply OK"}]
    )
    assert fake.calls[0]["reasoning"] == {"effort": effort}


@pytest.mark.parametrize(
    ("localized", "expected"),
    [("低", "low"), ("中", "medium"), ("高", "high"), ("极高", "xhigh")],
)
def test_reasoning_effort_chinese_aliases(localized, expected):
    assert normalize_reasoning_effort(localized) == expected


def test_invalid_reasoning_effort_is_rejected():
    with pytest.raises(ValueError, match="Unsupported OpenAI reasoning effort"):
        normalize_reasoning_effort("extreme")
