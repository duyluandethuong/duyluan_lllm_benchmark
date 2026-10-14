"""vLLM in Docker (Linux + NVIDIA). Consumes a Hugging Face repo snapshot (format "vllm").

Each load starts a fresh container from the official vllm/vllm-openai image (override with
$LLMBENCH_VLLM_IMAGE) with the model directory mounted read-only, on every GPU (tensor parallel),
and removes it on unload. Needs Docker with the NVIDIA container toolkit.

Unlike llama.cpp, vLLM shares one KV-cache pool between all requests (paged attention), so
--users sets --max-num-seqs and the context stays --ctx per request; requests that don't fit in
the pool wait in vLLM's queue, which shows up as TTFT.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.error
from pathlib import Path

from .. import ROOT
from ..config import Model, Settings
from .base import Backend, RunMetrics, TooLarge, get_json, looks_like_oom, post_sse

LOG_DIR = ROOT / "results" / "logs"
IMAGE = os.environ.get("LLMBENCH_VLLM_IMAGE", "vllm/vllm-openai:v0.30.0")
COMPILE_CACHE = Path.home() / ".cache" / "llmbench-vllm"  # torch.compile / CUDA graph cache, reused across runs
LOAD_TIMEOUT_S = 1800  # first start of a model compiles kernels
VLLM_OOM = ("no available memory for the cache blocks", "larger than the available kv cache memory",
            "insufficient memory", "cuda out of memory")


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for l in out.splitlines() if l.startswith("GPU "))


class Vllm(Backend):
    id = "vllm"
    label = "vLLM"
    fmt = "vllm"
    shared_kv = True  # fit check counts one context, not one per user

    def __init__(self):
        self.container: str | None = None
        self.url = ""

    def available(self) -> tuple[bool, str]:
        if not shutil.which("docker"):
            return False, "docker not found"
        if not _gpu_count():
            return False, "no NVIDIA GPU (nvidia-smi)"
        r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"image {IMAGE} not pulled (docker pull {IMAGE})"
        return True, IMAGE.rsplit(":", 1)[-1] + f" (docker, {_gpu_count()} GPU tensor parallel)"

    def load(self, model: Model, path: Path, placement: str, s: Settings) -> float:
        if placement == "GPU+CPU":
            raise TooLarge("the model does not fit in VRAM and vLLM has no CPU offload")
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.container = f"llmbench-vllm-{os.getpid()}"
        COMPILE_CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
        cmd = [
            "docker", "run", "-d", "--name", self.container, "--gpus", "all", "--ipc=host",
            "-p", f"127.0.0.1:{port}:8000", "-v", f"{path.resolve()}:/model:ro",
            "-v", f"{COMPILE_CACHE}:/root/.cache/vllm", "-e", "HF_HUB_OFFLINE=1", IMAGE,
            "/model", "--served-model-name", "bench", "--port", "8000",
            "--tensor-parallel-size", str(_gpu_count()),
            "--max-model-len", str(s.ctx_size), "--max-num-seqs", str(s.users),
            "--gpu-memory-utilization", "0.92",
            "--no-enable-prefix-caching",  # like cache_prompt=false on llama.cpp
            "--seed", str(s.seed),
            *model.vllm_args,
        ]
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"vllm_{model.id}_{os.getpid()}.log"
        self.log_path.write_text("+ " + " ".join(cmd) + "\n")
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr.strip()[-300:]}")
        while time.perf_counter() - t0 < LOAD_TIMEOUT_S:
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.container],
                                   capture_output=True, text=True).stdout.strip()
            if state != "true":
                tail = self._save_log()[-6000:]
                if looks_like_oom(tail) or any(m in tail.lower() for m in VLLM_OOM):
                    raise TooLarge(f"vLLM ran out of GPU memory at startup (see {self.log_path})")
                errs = [l for l in tail.splitlines() if "Error" in l]
                raise RuntimeError(f"vLLM exited: {(errs or tail.splitlines() or ['?'])[-1][-200:]}")
            try:
                get_json(f"{self.url}/v1/models", timeout=2)
                return time.perf_counter() - t0
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError(f"vLLM not ready after {LOAD_TIMEOUT_S}s (see {self.log_path})")

    def run(self, prompt: str, s: Settings) -> RunMetrics:
        body = {
            "model": "bench", "messages": [{"role": "user", "content": prompt}],
            "stream": True, "max_tokens": s.max_tokens, "temperature": s.temperature, "seed": s.seed,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": s.reasoning},
        }
        t0 = time.perf_counter()
        ttft = None
        usage, parts = {}, []
        for t, _, d in post_sse(f"{self.url}/v1/chat/completions", body):
            delta = (d.get("choices") or [{}])[0].get("delta", {})
            piece = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
            if piece:
                parts.append(piece)
                if ttft is None:
                    ttft = t - t0
            usage = d.get("usage") or usage
        total = time.perf_counter() - t0
        ttft = ttft if ttft is not None else total
        n_in, n_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        # vLLM reports no per-request timings: both rates are client-side.
        return RunMetrics(
            tokens_in=n_in, tokens_out=n_out, ttft_s=ttft,
            prefill_tps=n_in / ttft if ttft else None,
            decode_tps=(n_out - 1) / (total - ttft) if n_out > 1 and total > ttft else 0.0,
            total_s=total, text="".join(parts),
        )

    def _save_log(self) -> str:
        if not self.container:
            return ""
        r = subprocess.run(["docker", "logs", self.container], capture_output=True, text=True, errors="replace")
        text = r.stdout + r.stderr
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(text)
        return text

    def unload(self) -> None:
        if self.container:
            self._save_log()
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
            self.container = None
