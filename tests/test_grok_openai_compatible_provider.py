from utils.providers.available_models import AVAILABLE_MODELS
from utils.providers.grok_provider import GrokProvider


def test_grok_46_is_registered_on_openai_compatible_provider():
    config = next(model for model in AVAILABLE_MODELS if model.name == "grok-4.6")

    assert config.provider_classes == [GrokProvider]


def test_grok_chat_completions_parameters_include_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
    provider = GrokProvider()

    params = provider._build_api_params(
        "grok-4.6",
        [{"role": "user", "content": "Generate a MUSA kernel"}],
        max_tokens=50000,
    )

    assert params == {
        "model": "grok-4.6",
        "messages": [{"role": "user", "content": "Generate a MUSA kernel"}],
        "temperature": 0.7,
        "max_tokens": 50000,
        "reasoning_effort": "medium",
    }


def test_grok_uses_configured_third_party_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-zapi-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://zapi.deuo.top")

    provider = GrokProvider()

    assert provider.is_available()
    assert provider.name == "grok"
    assert provider.base_url == "https://zapi.deuo.top"


def test_grok_uses_sequential_seed_requests():
    provider = GrokProvider()

    assert not provider.supports_multiple_completions()
