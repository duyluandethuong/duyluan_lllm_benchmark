"""LM Studio via the `lms` CLI + its native REST API (/api/v1/chat).

Model files come from our ./models folder: they are hard-linked into LM Studio's
models folder (<lmstudio>/models/<publisher>/<repo>/...), so nothing is downloaded
twice and every engine reads byte-identical weights. LM Studio 0.4 does not index
symlinked files, hence hard links; on a different drive we fall back to copying.

One instance per format: LM Studio runs GGUF everywhere and MLX on Apple Silicon.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from ..config import Model, Settings
from .base import Backend, EngineError, RunMetrics, TooLarge, looks_like_oom, post_sse

IDENTIFIER = "llmbench"
HOME = Path.home()
EXE = "lms.exe" if platform.system() == "Windows" else "lms"
CANDIDATE_DIRS = [HOME / ".lmstudio", HOME / ".cache" / "lm-studio"]


def find_lms(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    if found := (os.environ.get("LMS") or shutil.which("lms")):
        return found
    for d in CANDIDATE_DIRS:
        if (d / "bin" / EXE).exists():
            return str(d / "bin" / EXE)
    return None


def models_root() -> Path:
    for d in CANDIDATE_DIRS:
        settings = d / "settings.json"
        if settings.exists():
            try:
                folder = json.loads(settings.read_text()).get("downloadsFolder")
                if folder:
                    return Path(folder)
            except (json.JSONDecodeError, OSError):
                pass
            return d / "models"
    return HOME / ".lmstudio" / "models"


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        if os.path.samefile(src, dst) or dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        print(f"  (hard link failed, copying {src.name} into LM Studio's folder)")
        shutil.copy2(src, dst)


class LmStudio(Backend):
    def __init__(self, fmt: str, lms: str | None = None):
        self.fmt = fmt
        self.id = f"lmstudio-{fmt}"
        self.label = f"LM Studio ({fmt.upper()})"
        self.lms = find_lms(lms)
        self.url = ""

    def _lms(self, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run([self.lms, *args], capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")

    def available(self) -> tuple[bool, str]:
        if not self.lms:
            return False, "LM Studio CLI `lms` not found (install LM Studio, run it once)"
        if self.fmt == "mlx" and not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            return False, "MLX needs Apple Silicon"
        return True, self._version()

    def _version(self) -> str:
        app = "LM Studio"
        if platform.system() == "Darwin":
            plist = Path("/Applications/LM Studio.app/Contents/Info.plist")
            if plist.exists():
                import plistlib
                app += " " + plistlib.loads(plist.read_bytes()).get("CFBundleShortVersionString", "?")
        # The selected runtime is the engine that actually runs, e.g. llama.cpp-...@2.55.0
        for line in self._lms("runtime", "ls").stdout.splitlines():
            cols = line.split()
            if "✓" in cols and cols[-1].lower() == self.fmt:
                return f"{app}, runtime {cols[0]}"
        return app

    def _ensure_server(self) -> None:
        status = self._lms("server", "status", "--json")
        try:
            st = json.loads(status.stdout)
        except json.JSONDecodeError:
            st = {}
        if not st.get("running"):
            self._lms("server", "start")
            st = json.loads(self._lms("server", "status", "--json").stdout or "{}")
        self.url = f"http://127.0.0.1:{st.get('port', 1234)}"

    def _import(self, path: Path) -> str:
        """Hard-link our files into LM Studio and return its model key."""
        root = models_root()
        repo_dir = path.parent if self.fmt == "gguf" else path
        rel_repo = repo_dir.relative_to(repo_dir.parent.parent)  # publisher/repo
        files = [path] if self.fmt == "gguf" else [p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")]
        if self.fmt == "gguf":  # include the other shards of a split GGUF
            files += [p for p in repo_dir.glob(path.name.split("-00001-of-")[0] + "-0*.gguf") if p != path]
        for f in files:
            _link(f, root / rel_repo / f.relative_to(repo_dir))
        want = (rel_repo / path.name).as_posix() if self.fmt == "gguf" else rel_repo.as_posix()
        for _ in range(60):  # LM Studio indexes new files asynchronously
            listing = json.loads(self._lms("ls", "--json").stdout or "[]")
            for m in listing:
                if m.get("path", "").lower() == want.lower():
                    return m["modelKey"]
            time.sleep(1)
        raise RuntimeError(f"LM Studio did not index {want} (check {root})")

    def load(self, model: Model, path: Path, placement: str, s: Settings) -> float:
        self._ensure_server()
        key = self._import(path)
        self._lms("unload", "--all")  # a clean slate: nothing else holding memory
        args = ["load", key, "-y", "--identifier", IDENTIFIER, "-c", str(s.ctx_size), "--parallel", "1"]
        if placement in ("GPU", "GPU (unified)"):
            args += ["--gpu", "max"]
        elif placement == "CPU":
            args += ["--gpu", "off"]
        t0 = time.perf_counter()
        r = self._lms(*args, timeout=1800)
        load_s = time.perf_counter() - t0
        out = r.stdout + r.stderr
        if r.returncode != 0 or "loaded successfully" not in out.lower():
            tail = " ".join(out.split()[-40:])
            if looks_like_oom(out):
                raise TooLarge(f"LM Studio refused to load: {tail}")
            raise RuntimeError(f"lms load failed: {tail}")
        return load_s

    def run(self, prompt: str, s: Settings) -> RunMetrics:
        body = {
            "model": IDENTIFIER, "input": prompt, "stream": True, "temperature": s.temperature,
            "max_output_tokens": s.max_tokens, "reasoning": "on" if s.reasoning else "off",
        }
        try:
            return self._run(body)
        except EngineError as e:
            if e.status == 400 and "reasoning" in e.body.lower():
                body.pop("reasoning")  # model has no reasoning toggle
                return self._run(body)
            raise

    def _run(self, body: dict) -> RunMetrics:
        t0 = time.perf_counter()
        ttft, stats, last = None, None, {}
        for t, event, d in post_sse(f"{self.url}/api/v1/chat", body):
            event = event or d.get("type")
            last = d
            if ttft is None and event in ("message.delta", "reasoning.delta"):
                ttft = t - t0
            elif event == "chat.end":
                stats = d["result"]["stats"]
        total = time.perf_counter() - t0
        if stats is None:  # stream ended on an error event instead of chat.end
            raise RuntimeError(f"LM Studio: {json.dumps(last)[:300]}")
        server_ttft = stats.get("time_to_first_token_seconds") or 0
        tin = stats.get("input_tokens", 0)
        return RunMetrics(
            tokens_in=tin, tokens_out=stats.get("total_output_tokens", 0),
            ttft_s=ttft if ttft is not None else total,
            prefill_tps=tin / server_ttft if server_ttft else None,  # LM Studio reports no prefill rate
            decode_tps=stats.get("tokens_per_second", 0.0), total_s=total,
        )

    def unload(self) -> None:
        if self.lms:
            self._lms("unload", IDENTIFIER)


def loaded_elsewhere(lms: str | None = None) -> list[str]:
    """Models LM Studio is holding in memory right now (they skew other engines)."""
    exe = find_lms(lms)
    if not exe:
        return []
    try:
        r = subprocess.run([exe, "ps", "--json"], capture_output=True, text=True, timeout=20)
        return [m.get("identifier") or m.get("modelKey") for m in json.loads(r.stdout or "[]")]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
