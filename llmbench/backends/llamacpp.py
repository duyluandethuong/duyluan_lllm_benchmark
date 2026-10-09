"""llama.cpp via llama-server (GGUF). Works on macOS, Linux and Windows.

Finds the binary from --llama-server, $LLAMA_SERVER, or PATH. Install:
  macOS: brew install llama.cpp      Windows: winget install llama.cpp
  Linux: brew / distro package, or a release zip from github.com/ggml-org/llama.cpp
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import urllib.error
from pathlib import Path

from .. import ROOT
from ..config import Model, Settings
from .base import Backend, RunMetrics, TooLarge, get_json, looks_like_oom, post_sse

LOG_DIR = ROOT / "results" / "logs"
LOAD_TIMEOUT_S = 900


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaCpp(Backend):
    id = "llamacpp"
    label = "llama.cpp"
    fmt = "gguf"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
        self.proc: subprocess.Popen | None = None
        self.url = ""

    def available(self) -> tuple[bool, str]:
        if not self.binary or (not Path(self.binary).exists() and not shutil.which(self.binary)):
            return False, "llama-server not found (install llama.cpp or pass --llama-server)"
        out = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=30)
        line = next((l for l in (out.stdout + out.stderr).splitlines() if l.startswith("version")), "unknown")
        return True, line.replace("version: ", "")

    def load(self, model: Model, path: Path, placement: str, s: Settings) -> float:
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        cmd = [
            self.binary, "-m", str(path), "--host", "127.0.0.1", "--port", str(port),
            "-c", str(s.ctx_size), "-np", "1",      # one slot gets the whole context
            "--cache-ram", "0", "--no-webui",        # no host-side prompt cache between runs
            "-rea", "on" if s.reasoning else "off",
        ]
        if placement == "CPU":
            cmd += ["-ngl", "0"]
        # GPU / GPU+CPU: leave -ngl unset; llama.cpp's --fit (default on) offloads as
        # many layers as fit and keeps the rest on CPU.
        cmd += list(model.llamacpp_args)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"llamacpp_{model.id}.log"
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        while time.perf_counter() - t0 < LOAD_TIMEOUT_S:
            if self.proc.poll() is not None:
                tail = self.log_path.read_text(errors="replace")[-4000:]
                self.proc = None
                if looks_like_oom(tail):
                    raise TooLarge(f"llama-server could not allocate memory (see {self.log_path})")
                raise RuntimeError(f"llama-server exited: {tail.strip().splitlines()[-1] if tail.strip() else '?'}")
            try:
                if get_json(f"{self.url}/health", timeout=2).get("status") == "ok":
                    return time.perf_counter() - t0
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.25)
        raise RuntimeError(f"llama-server not ready after {LOAD_TIMEOUT_S}s (see {self.log_path})")

    def run(self, prompt: str, s: Settings) -> RunMetrics:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True, "max_tokens": s.max_tokens, "temperature": s.temperature, "seed": s.seed,
            "cache_prompt": False, "stream_options": {"include_usage": True},
        }
        t0 = time.perf_counter()
        ttft = None
        timings, usage = {}, {}
        for t, _, d in post_sse(f"{self.url}/v1/chat/completions", body):
            delta = (d.get("choices") or [{}])[0].get("delta", {})
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = t - t0
            timings = d.get("timings", timings)
            usage = d.get("usage") or usage
        total = time.perf_counter() - t0
        return RunMetrics(
            tokens_in=usage.get("prompt_tokens", timings.get("prompt_n", 0)),
            tokens_out=usage.get("completion_tokens", timings.get("predicted_n", 0)),
            ttft_s=ttft if ttft is not None else total,
            prefill_tps=timings.get("prompt_per_second"),
            decode_tps=timings.get("predicted_per_second", 0.0),
            total_s=total,
        )

    def unload(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if getattr(self, "log", None):
            self.log.close()
            self.log = None
