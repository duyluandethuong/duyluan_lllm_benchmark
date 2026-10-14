"""Backend interface. To add an engine: subclass Backend, implement the three
methods, and register it in backends/__init__.py."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

from ..config import Model, Settings


class EngineError(RuntimeError):
    """An HTTP engine rejected the request; carries the server's error message."""

    def __init__(self, status: int, body: str):
        self.status, self.body = status, body
        super().__init__(f"HTTP {status}: {body[:300]}")


class TooLarge(Exception):
    """The engine refused or failed to load the model for lack of memory."""


@dataclass
class RunMetrics:
    tokens_in: int
    tokens_out: int
    ttft_s: float  # client-side: request sent -> first streamed token (incl. reasoning)
    prefill_tps: float | None  # engine-reported prompt processing speed
    decode_tps: float  # engine-reported generation speed (excludes prefill)
    total_s: float  # client-side wall time of the request
    text: str = field(default="", repr=False)  # generated output (reasoning + answer), saved for review


class Backend(ABC):
    id: str  # CLI name, e.g. "llamacpp"
    label: str  # report name, e.g. "llama.cpp"
    fmt: str  # artifact format it consumes: "gguf" | "mlx"

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(usable on this machine, version string or reason it is not)."""

    @abstractmethod
    def load(self, model: Model, path: Path, placement: str, s: Settings) -> float:
        """Load the model, return load time in seconds. Raise TooLarge on OOM."""

    @abstractmethod
    def run(self, prompt: str, s: Settings) -> RunMetrics:
        """One timed generation of `prompt` (already made unique per run)."""

    def run_batch(self, prompts: list[str], s: Settings) -> list[RunMetrics]:
        """All prompts at the same moment, as concurrent users (s.users slots were loaded).
        Default: one thread per request against an HTTP server."""
        if len(prompts) == 1:
            return [self.run(prompts[0], s)]
        start = threading.Barrier(len(prompts))

        def one(p: str) -> RunMetrics:
            start.wait()
            return self.run(p, s)

        # Ctrl-C must stop a round: a blocking wait (pool.map, or the exit of `with`) only sees the
        # signal once every request is done. Wait in short steps instead; on interrupt bench.py calls
        # unload(), which stops the server and ends the threads.
        pool = ThreadPoolExecutor(len(prompts))
        try:
            futures = [pool.submit(one, p) for p in prompts]
            while wait(futures, timeout=0.5).not_done:
                pass
            ms = [f.result() for f in futures]
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        # Per-slot engine prefill rates aren't comparable across engines (mlx-lm has none in a batch),
        # so with several users prefill is client-side for every engine: tokens in / TTFT.
        return [replace(m, prefill_tps=m.tokens_in / m.ttft_s if m.ttft_s else None) for m in ms]

    def unload(self) -> None:
        """Free the model. Must be safe to call twice or after a failed load."""

    def bench_prefill(self, path: Path, placement: str, n_tokens: int) -> float | None:
        """Optional engine-native prefill rate for an n_tokens prompt, measured after unload()."""
        return None


# ---- helpers shared by HTTP engines -------------------------------------------------

OOM_MARKERS = (
    "out of memory", "failed to allocate", "unable to allocate", "cudamalloc failed",
    "erroroutofdevicememory", "insufficient memory", "not enough memory",
    "insufficient system resources", "would overload",
)


def looks_like_oom(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in OOM_MARKERS)


def post_sse(url: str, body: dict, timeout: float = 3600) -> Iterator[tuple[float, str | None, dict]]:
    """POST JSON and yield (perf_counter time, event name, data) per SSE message."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise EngineError(e.code, e.read().decode("utf-8", errors="replace")) from None
    with resp:
        event = None
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                yield time.perf_counter(), event, json.loads(data)
            elif not line:
                event = None


def get_json(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())
