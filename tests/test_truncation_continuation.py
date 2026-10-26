"""Unit tests for fix 6-1: output-cap truncation continuation (C1).

Fully mocked -- no GPU, no live LLM calls.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from triton_kernel_agent.kernel_backend import extract_kernel_bundle, get_kernel_backend
from triton_kernel_agent.worker import (
    LLMCallResult,
    MAX_CONTINUATIONS,
    TRUNCATION_MARGIN_TOKENS,
    VerificationWorker,
    _looks_truncated,
)
from utils.providers.base import resolve_output_cap


# ------------------------------------------------------------------ helpers
def bundle_text(comment_headers=False):
    """Minimal MUSA bundle that passes validate_musa_bundle_semantics (fix 2-2)."""
    files = {"kernel.py": "def kernel_function(x): return extension.forward(x)",
             "binding.cpp": "void launch();", "kernel.mu": "void launch() {}",
             "setup.py": ("from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
                          "from setuptools import setup")}
    return "\n".join(
        (f"```\n{'#' if name.endswith('.py') else '//'} FILE: {name}\n{body}\n```"
         if comment_headers else f"FILE: {name}\n```\n{body}\n```")
        for name, body in files.items())


GOOD_TEST = "import torch\ndef main():\n    pass\n"


def file_block(name, body, close=True):
    text = f"FILE: {name}\n```\n{body}"
    return text + ("\n```" if close else "")


# Form-A segment 1: complete kernel.py, binding.cpp cut mid-fence.
SEGMENT_1 = (
    "Reasoning preamble that burns tokens.\n"
    + file_block("kernel.py", "def kernel_function(x): return extension.forward(x)") + "\n"
    + file_block("binding.cpp", "void launch();\nvoid w_stride_cout(", close=False)
)
# Continuation tail: remainder of binding.cpp + kernel.mu + setup.py.
SEGMENT_2 = (
    "int width);\n```\n"
    + file_block("kernel.mu", "void launch() {}") + "\n"
    + file_block("setup.py",
                 "from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
                 "from setuptools import setup")
)


def worker(tmp_path, backend="musa", platform="musa"):
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    w = VerificationWorker(0, tmp_path / "work", tmp_path / "logs",
                           target_platform=platform, kernel_backend=backend)
    w.provider = object()
    return w


def refine(w, test_code=GOOD_TEST):
    return w._refine_kernel(extract_kernel_bundle(bundle_text()),
                            {"stdout": "mismatch", "stderr": ""},
                            "formula", test_code)


# --------------------------------------------------- A0: resolver / clamp sync
@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("KERNELAGENT_OUTPUT_CAP", raising=False)


def test_resolve_output_cap_priority(monkeypatch):
    fake = Mock()
    fake.get_max_tokens_limit.return_value = 32000
    assert resolve_output_cap("some-model", fake) == 32000        # registry
    monkeypatch.setenv("KERNELAGENT_OUTPUT_CAP", "48000")
    assert resolve_output_cap("some-model", fake) == 48000        # env overrides registry
    monkeypatch.setenv("KERNELAGENT_OUTPUT_CAP", "-5")
    assert resolve_output_cap("some-model", fake) == 32000        # invalid env -> registry
    monkeypatch.setenv("KERNELAGENT_OUTPUT_CAP", "abc")
    assert resolve_output_cap("some-model", fake) == 32000
    monkeypatch.delenv("KERNELAGENT_OUTPUT_CAP")
    assert resolve_output_cap("any-model", None) == 16384         # offline default


def test_clamp_sites_use_same_resolver(monkeypatch):
    """Env override must not be defeated by the 8192 registry default."""
    from utils.providers.openai_base import OpenAICompatibleProvider

    class _Fake:
        get_max_tokens_limit = OpenAICompatibleProvider.get_max_tokens_limit

    fake = _Fake()  # unbound calls: methods only read env + the registry hook
    monkeypatch.setenv("KERNELAGENT_OUTPUT_CAP", "32768")
    params = OpenAICompatibleProvider._build_api_params(fake, "brand-new-model", [], max_tokens=32768)
    assert params["max_tokens"] == 32768
    params = OpenAICompatibleProvider._build_responses_params(fake, "brand-new-model", [], max_tokens=32768)
    assert params["max_output_tokens"] == 32768

    monkeypatch.delenv("KERNELAGENT_OUTPUT_CAP")
    params = OpenAICompatibleProvider._build_api_params(fake, "brand-new-model", [], max_tokens=32768)
    assert params["max_tokens"] == 8192  # unregistered -> registry default


# ------------------------------------------ A1: finish_reason surfacing
def _fake_chat_response(finish_reason):
    message = SimpleNamespace(content="FILE: kernel.py\n```\nx = 1\n```")
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(model_dump=lambda: {"completion_tokens": 42})
    choices = [choice]
    completions = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            choices=choices, usage=usage,
            model_dump=lambda: {"id": "chatcmpl-x"}))
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


@pytest.mark.parametrize("fr", ["length", "stop", None])
def test_get_response_surfaces_finish_reason(monkeypatch, fr):
    from utils.providers.openai_provider import OpenAIProvider
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_WIRE_API", "chat_completions")
    provider = OpenAIProvider()           # real __init__; we swap the client below
    provider.client = _fake_chat_response(fr)
    response = provider.get_response("musacoder-27b", [{"role": "user", "content": "hi"}])
    assert response.finish_reason == fr


def test_call_llm_detailed_surfaces_usage_and_finish_reason(tmp_path, monkeypatch):
    w = worker(tmp_path)
    resp = SimpleNamespace(
        content="FILE: kernel.py\n```\nx = 1\n```",
        usage={"completion_tokens": 128},
        finish_reason="length",
    )
    w.provider = Mock()
    w.provider.get_response.return_value = resp
    w.provider.__bool__ = lambda self: True  # truthy Mock already
    result = w._call_llm_detailed([{"role": "user", "content": "hi"}])
    assert result.content.startswith("FILE:")
    assert result.completion_tokens == 128
    assert result.finish_reason == "length"
    # and the legacy _call_llm keeps its plain-str contract
    assert w._call_llm([{"role": "user", "content": "hi"}]).startswith("FILE:")


# ------------------------------------------------- dual-signal detection
@pytest.mark.parametrize("cap", [16384, 32000, 128000])
def test_signal_b_scales_with_cap(cap):
    thr = cap - TRUNCATION_MARGIN_TOKENS
    assert _looks_truncated(LLMCallResult("x", cap, "stop"), "no files", thr) is True


@pytest.mark.parametrize("cap", [16384, 32000, 128000])
def test_signal_a_is_threshold_independent(cap):
    thr = cap - TRUNCATION_MARGIN_TOKENS
    assert _looks_truncated(LLMCallResult("x", 3000, "length"), "no files", thr) is True


def test_no_signal_no_continuation():
    thr = 16384 - TRUNCATION_MARGIN_TOKENS
    assert _looks_truncated(LLMCallResult("x", 5000, "stop"), "no files", thr) is False
    assert _looks_truncated(LLMCallResult("x", None, None), "no files", thr) is False


def test_threshold_boundary():
    thr = 16384 - TRUNCATION_MARGIN_TOKENS
    assert _looks_truncated(LLMCallResult("x", 16384 - TRUNCATION_MARGIN_TOKENS, "stop"),
                            "no files", thr) is True
    assert _looks_truncated(LLMCallResult("x", 16384 - TRUNCATION_MARGIN_TOKENS - 1, "stop"),
                            "no files", thr) is False


def test_clean_extraction_never_truncated():
    assert _looks_truncated(LLMCallResult("x", 99999, "length"), None, 15872) is False


# ------------------------------------------------ A2/A3: protocol & state machine
def test_form_a_one_continuation_recovers(tmp_path):
    w = worker(tmp_path)
    w._call_llm_detailed = Mock(side_effect=[
        LLMCallResult(SEGMENT_1, 16384, "stop"),
        LLMCallResult(SEGMENT_2, 5000, "stop"),
    ])
    result = refine(w)
    result.require(get_kernel_backend("musa"))
    assert "kernel_function" in result.files["kernel.py"]
    assert "MUSAExtension" in result.files["setup.py"]
    assert w._call_llm_detailed.call_count == 2
    diag = w.log_dir / "refinement_truncated_0_0.txt"
    assert diag.read_text() == SEGMENT_1


def test_second_continuation_recovers(tmp_path):
    w = worker(tmp_path)
    # seg1 truncated -> seg2 truncated again (only kernel.mu, no setup.py)
    # -> seg3 supplies setup.py.
    w._call_llm_detailed = Mock(side_effect=[
        LLMCallResult(SEGMENT_1, 16384, "stop"),
        LLMCallResult(SEGMENT_2.split("FILE: setup.py")[0], 16384, "stop"),
        LLMCallResult(file_block("setup.py",
                                 "from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
                                 "from setuptools import setup"), 3000, "stop"),
    ])
    result = refine(w)
    result.require(get_kernel_backend("musa"))
    assert w._call_llm_detailed.call_count == 1 + MAX_CONTINUATIONS
    assert "MUSAExtension" in result.files["setup.py"]


def test_continuation_exhausted_then_raises(tmp_path):
    # Fix P1 (A): when continuation is exhausted and the response is still
    # malformed, there is no same-round format repair -- raise so run()'s
    # backstop (fix P1 B) regenerates next round.
    w = worker(tmp_path)
    w._call_llm_detailed = Mock(side_effect=[
        LLMCallResult(SEGMENT_1, 16384, "stop"),
        LLMCallResult("still no FILE headers " * 400, 16384, "stop"),
        LLMCallResult("again no FILE headers " * 300, 16384, "stop"),
    ])
    with pytest.raises(ValueError, match="Malformed LLM kernel response"):
        refine(w)
    assert w._call_llm_detailed.call_count == 1 + MAX_CONTINUATIONS
    assert (w.log_dir / "refinement_rejected_0_0.txt").read_text()


def test_form_b_skips_continuation_when_no_signal(tmp_path):
    # No truncation signal (finish_reason 'stop', completion < threshold) means
    # no continuation, and under fix P1 (A) also no same-round repair: raise.
    w = worker(tmp_path)
    w._call_llm_detailed = Mock(side_effect=[
        LLMCallResult("from setuptools import setup\n" + "x = 0\n" * 400, 9000, "stop"),
    ])
    with pytest.raises(ValueError, match="Malformed LLM kernel response"):
        refine(w)
    assert w._call_llm_detailed.call_count == 1  # no continuation, no repair


def test_repeated_file_last_seen_wins(tmp_path):
    # The unclosed setup.py fence leaves the continuation candidate malformed,
    # so under fix P1 (A) it raises instead of falling into a same-round repair.
    # (The last-seen-wins extraction behavior is orthogonal to the removed
    # repair path and is covered by the extraction layer.)
    w = worker(tmp_path)
    broken = (file_block("kernel.py", "def kernel_function(x): return extension.forward(x)") + "\n"
              + file_block("binding.cpp", "void launch();") + "\n"
              + file_block("kernel.mu", "void launch() {}") + "\n"
              + file_block("setup.py", "BROKEN", close=False))
    good_setup = file_block(
        "setup.py",
        "from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
        "from setuptools import setup")
    w._call_llm_detailed = Mock(side_effect=[
        LLMCallResult(broken, 16384, "stop"),
        LLMCallResult(good_setup, 2000, "stop"),
    ])
    with pytest.raises(ValueError, match="Malformed LLM kernel response"):
        refine(w)
    assert w._call_llm_detailed.call_count == 2  # 1 main + 1 continuation, no repair


def test_success_first_call_zero_overhead(tmp_path):
    w = worker(tmp_path)
    w._call_llm_detailed = Mock(return_value=LLMCallResult(bundle_text(True), 9000, "stop"))
    result = refine(w)
    result.require(get_kernel_backend("musa"))
    assert w._call_llm_detailed.call_count == 1
    assert not list(w.log_dir.glob("refinement_truncated_*"))
    assert not list(w.log_dir.glob("refinement_rejected_*"))


def test_non_musa_backend_gets_continuation_but_raise_after(tmp_path):
    w = worker(tmp_path, backend="triton", platform="cuda")
    w._call_llm_detailed = Mock(side_effect=[
        LLMCallResult("def kernel_function(  # cut here", 16384, "length"),
        LLMCallResult("x):\n    return 1", 100, "stop"),
    ])
    out = w._refine_kernel("def kernel_function(x): return 0",
                           {"stdout": "", "stderr": "e"}, "formula", GOOD_TEST)
    assert "kernel_function" in out
    assert w._call_llm_detailed.call_count == 2
    # and still no format repair for non-MUSA backends
    w._call_llm_detailed = Mock(side_effect=[LLMCallResult("not code", 40, "stop")])
    with pytest.raises(ValueError, match="Malformed LLM kernel response"):
        w._refine_kernel("def kernel_function(x): return 0",
                         {"stdout": "", "stderr": "e"}, "formula", GOOD_TEST)
    assert w._call_llm_detailed.call_count == 1
