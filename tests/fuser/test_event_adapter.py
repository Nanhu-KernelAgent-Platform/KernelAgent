import json
from types import SimpleNamespace

from Fuser.event_adapter import EventAdapter


class _Responses:
    def __init__(self, stream_error: Exception | None = None):
        self.stream_error = stream_error
        self.stream_calls = 0
        self.create_calls = 0

    def stream(self, **kwargs):
        self.stream_calls += 1
        if self.stream_error is not None:
            raise self.stream_error
        raise AssertionError("stream should not be called")

    def create(self, **kwargs):
        self.create_calls += 1
        return SimpleNamespace(
            output_text="```python\nprint('PASS')\n```",
            id="resp_test",
        )


def _adapter(tmp_path, responses):
    client = SimpleNamespace(responses=responses)
    return EventAdapter(
        model="gpt-5.6-sol",
        store_responses=False,
        timeout_s=30,
        jsonl_path=tmp_path / "events.jsonl",
        client=client,
    )


def test_non_streaming_mode_uses_responses_create(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_RESPONSES_STREAM", "0")
    responses = _Responses()
    result = _adapter(tmp_path, responses).stream("system", "user")

    assert result["error"] is None
    assert "print('PASS')" in result["output_text"]
    assert result["response_id"] == "resp_test"
    assert responses.stream_calls == 0
    assert responses.create_calls == 1


def test_relay_event_order_error_falls_back_to_non_streaming(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_RESPONSES_STREAM", "1")
    responses = _Responses(
        RuntimeError(
            "Expected to have received `response.created` before `codex.rate_limits`"
        )
    )
    adapter = _adapter(tmp_path, responses)
    result = adapter.stream("system", "user")

    assert result["error"] is None
    assert "print('PASS')" in result["output_text"]
    assert responses.stream_calls == 1
    assert responses.create_calls == 1
    events = [
        json.loads(line) for line in adapter.jsonl_path.read_text().splitlines()
    ]
    assert any(event["kind"] == "stream_fallback" for event in events)
