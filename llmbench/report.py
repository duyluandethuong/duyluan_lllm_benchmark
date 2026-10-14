"""Collect results, render Markdown tables, write results/<timestamp>_<host>.md/.json."""

from __future__ import annotations

import json
import platform
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean

from . import ROOT
from .backends import RunMetrics
from .config import Settings
from .hardware import Hardware

RESULTS_DIR = ROOT / "results"
TOO_LARGE = "Too large to run"


@dataclass
class Round:
    """One timed run: `users` requests sent at the same moment."""
    wall_s: float  # first request sent -> last request finished
    requests: list[RunMetrics]

    @property
    def agg_tps(self) -> float:  # output tokens of all users per second of wall time
        return sum(m.tokens_out for m in self.requests) / self.wall_s

    @property
    def ttft_max(self) -> float:
        return max(m.ttft_s for m in self.requests)


@dataclass
class Result:
    model: str
    engine: str
    quant: str
    status: str  # "OK", TOO_LARGE, "Skipped: ...", "Error: ..."
    users: int = 1
    placement: str = "—"
    est_gb: float | None = None
    load_s: float | None = None
    bench_prefill_tps: float | None = None  # llama.cpp only: llama-bench on a same-length prompt
    rounds: list[Round] = field(default_factory=list)

    @property
    def runs(self) -> list[RunMetrics]:  # every request of every round
        return [m for rd in self.rounds for m in rd.requests]

    def avg(self, attr: str) -> float | None:
        vals = [getattr(r, attr) for r in self.runs if getattr(r, attr) is not None]
        return mean(vals) if vals else None

    def round_avg(self, attr: str) -> float | None:
        return mean(getattr(rd, attr) for rd in self.rounds) if self.rounds else None


def _f(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v:,.{digits}f}"


def _i(v: float | None) -> str:
    return "—" if v is None else f"{round(v):,}"


SUMMARY_HEADER = ["Model", "Engine", "Users", "Quant", "Status", "Placement", "Est. mem (GB)", "Load (s)",
                  "Tokens in", "Tokens out", "TTFT (s)", "TTFT max (s)", "Prefill tok/s", "llama-bench prefill tok/s",
                  "Decode tok/s", "Agg. out tok/s", "Total (s)"]


def summary_rows(results: list[Result]) -> list[list[str]]:
    rows = []
    for r in results:
        rows.append([
            r.model, r.engine, str(r.users), r.quant, r.status, r.placement, _f(r.est_gb), _f(r.load_s),
            _i(r.avg("tokens_in")), _i(r.avg("tokens_out")), _f(r.avg("ttft_s"), 2), _f(r.round_avg("ttft_max"), 2),
            _f(r.avg("prefill_tps")), _f(r.bench_prefill_tps), _f(r.avg("decode_tps")), _f(r.round_avg("agg_tps")),
            _f(r.round_avg("wall_s")),
        ])
    return rows


def md_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(header)]
    def line(cells): return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, rows)])


def per_run_table(results: list[Result]) -> str:
    """One row per round; with several users, token counts are summed and rates averaged per user."""
    header = ["Model", "Engine", "Users", "Run", "Tokens in", "Tokens out", "TTFT (s)", "TTFT max (s)",
              "Prefill tok/s", "Decode tok/s", "Agg. out tok/s", "Total (s)"]
    def avg(rd, attr):
        vals = [getattr(m, attr) for m in rd.requests if getattr(m, attr) is not None]
        return mean(vals) if vals else None
    rows = [[r.model, r.engine, str(r.users), str(i), _i(sum(m.tokens_in for m in rd.requests)),
             _i(sum(m.tokens_out for m in rd.requests)), _f(avg(rd, "ttft_s"), 2), _f(rd.ttft_max, 2),
             _f(avg(rd, "prefill_tps")), _f(avg(rd, "decode_tps")), _f(rd.agg_tps), _f(rd.wall_s)]
            for r in results for i, rd in enumerate(r.rounds, 1)]
    return md_table(header, rows)


def write(results: list[Result], hw: Hardware, s: Settings, engines: dict[str, str], prompt_chars: int) -> str:
    now = datetime.now()
    host = re.sub(r"[^A-Za-z0-9_-]+", "-", platform.node().split(".")[0]) or "host"
    stem = f"{now:%Y-%m-%d_%H%M%S}_{host}"
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = md_table(SUMMARY_HEADER, summary_rows(results))
    engine_lines = "\n".join(f"- {k}: {v}" for k, v in engines.items())
    md = f"""# LLM benchmark — {host}, {now:%Y-%m-%d %H:%M}

**Machine:** {hw.describe()}

**Engines:**
{engine_lines}

**Settings:** prompt `prompts/broll_mapping.txt` ({prompt_chars:,} chars) · context {s.ctx_size:,} · max output {s.max_tokens:,} tokens · temperature {s.temperature} · reasoning {"on" if s.reasoning else "off"} · {s.runs} run(s) per model, averaged · concurrent users {", ".join(map(str, sorted({r.users for r in results})))}

## Summary

{summary}

## Per run

{per_run_table(results)}

### Metric definitions
- **Tokens in / out**: prompt and generated tokens as counted by the engine's own tokenizer (output includes reasoning tokens).
- **TTFT**: wall time from sending the request to receiving the first streamed token (client-side, includes prompt processing).
- **Prefill tok/s**: prompt processing speed reported by the engine (LM Studio: tokens in ÷ its own TTFT).
- **llama-bench prefill tok/s** (llama.cpp only): `llama-bench -p <tokens in> -n 0 -r 2` on the same GGUF, run after
  the server is stopped. A cross-check: if it is far above the server's prefill, the server path is stalling.
- **Decode tok/s**: generation speed reported by the engine, excluding prefill.
- **Users**: requests sent at the same moment to one loaded model (one server slot each, context {s.ctx_size:,} per user;
  mlx-lm: one continuous batch). TTFT, prefill and decode are per user, averaged over users; MLX with several users
  has no engine prefill rate, so its prefill is tokens in ÷ TTFT.
- **TTFT max**: the slowest user's TTFT (the last one to get a first token).
- **Agg. out tok/s**: output tokens of all users ÷ wall time of the run (first request sent to last one finished).
- **Total (s)**: wall time of the run.
- **Est. mem**: weights + KV cache at the configured context + 1.5 GB overhead; used for the "{TOO_LARGE}" check.
"""
    out_dir = RESULTS_DIR / "outputs" / stem
    for r in results:
        for i, rd in enumerate(r.rounds, 1):
            for u, m in enumerate(rd.requests, 1):
                out_dir.mkdir(parents=True, exist_ok=True)
                tag = f"run{i}" if r.users == 1 else f"u{r.users}__run{i}__user{u}"
                name = re.sub(r"[^A-Za-z0-9.]+", "-", f"{r.model}__{r.engine}__{tag}").strip("-")
                (out_dir / f"{name}.txt").write_text(m.text, encoding="utf-8")
    md += f"\nModel outputs (for answer-quality review): `results/outputs/{stem}/`\n"
    (RESULTS_DIR / f"{stem}.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps({
        "host": host, "time": now.isoformat(timespec="seconds"), "hardware": asdict(hw),
        "settings": asdict(s), "engines": engines,
        "results": [{**asdict(r), "rounds": [
            {"wall_s": rd.wall_s, "requests": [{k: v for k, v in asdict(m).items() if k != "text"} for m in rd.requests]}
            for rd in r.rounds]} for r in results],
    }, indent=2), encoding="utf-8")
    return str(RESULTS_DIR / f"{stem}.md")
