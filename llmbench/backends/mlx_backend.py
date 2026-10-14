"""MLX via mlx-lm (Apple Silicon only). The model runs in a child process
(mlx_worker.py) so its memory is returned to the OS before the next engine runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .. import ROOT
from ..config import Model, Settings
from .base import Backend, RunMetrics, TooLarge, looks_like_oom
from .llamacpp import LOG_DIR


class Mlx(Backend):
    id = "mlx"
    label = "MLX"
    fmt = "mlx"

    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def available(self) -> tuple[bool, str]:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            return False, "MLX needs Apple Silicon"
        try:
            from importlib.metadata import version
            return True, f"mlx-lm {version('mlx-lm')}, mlx {version('mlx')}"
        except Exception:
            return False, "mlx-lm not installed (run: uv sync)"

    def _recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            self.proc.wait()
            self.proc = None
            err = self.log_path.read_text(errors="replace")[-3000:]
            if looks_like_oom(err):
                raise TooLarge("MLX worker ran out of memory")
            raise RuntimeError(f"MLX worker died: {err.strip().splitlines()[-1] if err.strip() else '?'}")
        msg = json.loads(line)
        if "error" in msg:
            if looks_like_oom(msg["error"]):
                raise TooLarge(msg["error"])
            raise RuntimeError(msg["error"])
        return msg

    def load(self, model: Model, path: Path, placement: str, s: Settings) -> float:
        cfg = json.dumps({"path": str(path), "reasoning": s.reasoning})
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"mlx_{model.id}_{os.getpid()}.log"
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "llmbench.backends.mlx_worker", cfg], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
        )
        return self._recv()["loaded"]

    def _ask(self, req: dict, s: Settings) -> dict:
        req |= {"max_tokens": s.max_tokens, "temperature": s.temperature, "seed": s.seed}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return self._recv()

    @staticmethod
    def _metrics(m: dict) -> RunMetrics:
        return RunMetrics(m["tokens_in"], m["tokens_out"], m["ttft_s"], m["prefill_tps"], m["decode_tps"],
                          m["total_s"], m["text"])

    def run(self, prompt: str, s: Settings) -> RunMetrics:
        return self._metrics(self._ask({"prompt": prompt}, s))

    def run_batch(self, prompts: list[str], s: Settings) -> list[RunMetrics]:
        # mlx-lm has no server slots: the worker batches all prompts in one BatchGenerator.
        if len(prompts) == 1:
            return [self.run(prompts[0], s)]
        return [self._metrics(m) for m in self._ask({"prompts": prompts}, s)["results"]]

    def unload(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if getattr(self, "log", None):
            self.log.close()
            self.log = None
