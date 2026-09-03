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
from __future__ import annotations
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

try:
    # OpenAI official SDK (Responses API)
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    OpenAI = None  # type: ignore


@dataclass
class StreamDelta:
    ts: float
    kind: str
    data: dict[str, Any]


class EventAdapter:
    """
    Wraps the OpenAI Responses streaming API and persists a buffered JSONL of events.

    - Streams output_text deltas to an optional callback for console mux
    - Writes JSONL with simple flush policy (>=8KB buffer or >=50ms since last flush)
    - Supports cooperative cancellation via a threading.Event
    """

    def __init__(
        self,
        model: str,
        store_responses: bool,
        timeout_s: int,
        jsonl_path: Path,
        stop_event: threading.Event | None = None,
        on_delta: Callable[[str, None]] | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.store_responses = store_responses
        self.timeout_s = timeout_s
        self.jsonl_path = jsonl_path
        self.stop_event = stop_event or threading.Event()
        self.on_delta = on_delta
        self._client = client
        self._buffer: list[str] = []
        self._buffer_bytes = 0
        self._last_flush = time.time()
        self._lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI SDK not available. Install openai>=1.40 and set OPENAI_API_KEY."
            )
        self._client = OpenAI()
        return self._client

    def _append_event(self, ev: StreamDelta) -> None:
        line = json.dumps(
            {"ts": ev.ts, "kind": ev.kind, "data": ev.data}, ensure_ascii=False
        )
        with self._lock:
            self._buffer.append(line)
            self._buffer_bytes += len(line) + 1

    def _should_flush(self) -> bool:
        now = time.time()
        if self._buffer_bytes >= 8 * 1024:
            return True
        if now - self._last_flush >= 0.050 and self._buffer:
            return True
        return False

    def _flush(self) -> None:
        if not self._buffer:
            return
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            for line in self._buffer:
                f.write(line)
                f.write("\n")
        self._buffer.clear()
        self._buffer_bytes = 0
        self._last_flush = time.time()

    def _flusher_thread(self, running_flag: threading.Event) -> None:
        # Periodically flush while streaming
        while not running_flag.is_set():
            time.sleep(0.025)
            with self._lock:
                if self._should_flush():
                    self._flush()
        # Final flush
        with self._lock:
            self._flush()

    def _create_non_streaming(
        self, client: Any, params: dict[str, Any]
    ) -> tuple[str, str | None]:
        """Use Responses.create for relays whose custom SSE events break the SDK."""

        response = client.responses.create(**params)
        output_text = getattr(response, "output_text", "") or ""
        response_id = getattr(response, "id", None)
        if output_text and self.on_delta:
            try:
                self.on_delta(output_text)
            except Exception:
                pass
        self._append_event(
            StreamDelta(
                time.time(),
                "response.completed",
                {
                    "response_id": response_id,
                    "transport": "non_streaming",
                    "output_chars": len(output_text),
                },
            )
        )
        return output_text, response_id

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Start streaming and persist events. Returns a dict with summary fields:
        {"output_text": <str>, "response_id": <str|None>, "error": <str|None>}
        """
        client = self._ensure_client()
        output_text_parts: list[str] = []
        response_id: str | None = None
        error_msg: str | None = None

        # Start background flusher
        done_flag = threading.Event()
        t = threading.Thread(
            target=self._flusher_thread, args=(done_flag,), daemon=True
        )
        t.start()

        start_ts = time.time()
        self._append_event(
            StreamDelta(start_ts, "stream_started", {"model": self.model})
        )

        params: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "timeout": self.timeout_s,
        }
        if extras:
            params.update(extras)
        # store_responses is typically a client-side behavior in our design; include as hint
        if self.store_responses:
            params["store"] = True

        try:
            stream_enabled = os.getenv("OPENAI_RESPONSES_STREAM", "1").strip().lower()
            if stream_enabled in {"0", "false", "no", "off"}:
                text, response_id = self._create_non_streaming(client, params)
                output_text_parts.append(text)
            else:
                with client.responses.stream(**params) as stream:  # type: ignore[attr-defined]
                    for event in stream:
                        if self.stop_event.is_set():
                            # cooperative cancel
                            self._append_event(StreamDelta(time.time(), "canceled", {}))
                            break
                        # Determine event kind/type
                        kind = (
                            getattr(event, "type", None)
                            or getattr(event, "event", None)
                            or "unknown"
                        )
                        data: dict[str, Any] = {}

                        # Handle textual deltas per Responses API
                        if kind == "response.output_text.delta":
                            delta = getattr(event, "delta", None)
                            if isinstance(delta, str) and delta:
                                output_text_parts.append(delta)
                                if self.on_delta:
                                    try:
                                        self.on_delta(delta)
                                    except Exception:
                                        pass
                                data["delta"] = delta

                        # Handle completed and IDs
                        if kind == "response.completed" and hasattr(event, "response"):
                            try:
                                response_id = getattr(
                                    getattr(event, "response"), "id", None
                                )
                                if response_id:
                                    data["response_id"] = response_id
                            except Exception:
                                pass

                        # Handle error events
                        if kind == "response.error" and hasattr(event, "error"):
                            try:
                                err_obj = getattr(event, "error")
                                msg = getattr(err_obj, "message", None)
                                data["error"] = (
                                    msg if isinstance(msg, str) else str(err_obj)
                                )
                            except Exception:
                                data["error"] = "unknown"

                        # Fallback: best-effort ID extraction on any event
                        if not data.get("response_id") and hasattr(event, "response"):
                            try:
                                rid = getattr(getattr(event, "response"), "id", None)
                                if rid:
                                    data["response_id"] = rid
                            except Exception:
                                pass

                        self._append_event(StreamDelta(time.time(), kind, data))
        except Exception as e:
            message = str(e)
            relay_event_before_created = (
                "Expected to have received `response.created` before" in message
            )
            if relay_event_before_created:
                self._append_event(
                    StreamDelta(
                        time.time(),
                        "stream_fallback",
                        {"reason": message, "transport": "non_streaming"},
                    )
                )
                try:
                    text, response_id = self._create_non_streaming(client, params)
                    output_text_parts.append(text)
                except Exception as fallback_error:
                    error_msg = (
                        "stream_fallback_error: "
                        f"{fallback_error.__class__.__name__}: {fallback_error}"
                    )
            else:
                error_msg = f"stream_error: {e.__class__.__name__}: {e}"
            if error_msg:
                self._append_event(
                    StreamDelta(time.time(), "exception", {"message": error_msg})
                )
        finally:
            done_flag.set()
            t.join(timeout=1.0)
            with self._lock:
                self._flush()

        return {
            "output_text": "".join(output_text_parts),
            "response_id": response_id,
            "error": error_msg,
        }
