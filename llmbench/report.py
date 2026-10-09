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
class Result:
    model: str
    engine: str
    quant: str
    status: str  # "OK", TOO_LARGE, "Skipped: ...", "Error: ..."
    placement: str = "—"
    est_gb: float | None = None
    load_s: float | None = None
    runs: list[RunMetrics] = field(default_factory=list)

    def avg(self, attr: str) -> float | None:
        vals = [getattr(r, attr) for r in self.runs if getattr(r, attr) is not None]
        return mean(vals) if vals else None


def _f(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v:,.{digits}f}"


def _i(v: float | None) -> str:
    return "—" if v is None else f"{round(v):,}"


SUMMARY_HEADER = ["Model", "Engine", "Quant", "Status", "Placement", "Est. mem (GB)", "Load (s)",
                  "Tokens in", "Tokens out", "TTFT (s)", "Prefill tok/s", "Decode tok/s", "Total (s)"]


def summary_rows(results: list[Result]) -> list[list[str]]:
    rows = []
    for r in results:
        rows.append([
            r.model, r.engine, r.quant, r.status, r.placement, _f(r.est_gb), _f(r.load_s),
            _i(r.avg("tokens_in")), _i(r.avg("tokens_out")), _f(r.avg("ttft_s"), 2),
            _f(r.avg("prefill_tps")), _f(r.avg("decode_tps")), _f(r.avg("total_s")),
        ])
    return rows


def md_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(header)]
    def line(cells): return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, rows)])


def per_run_table(results: list[Result]) -> str:
    header = ["Model", "Engine", "Run", "Tokens in", "Tokens out", "TTFT (s)", "Prefill tok/s", "Decode tok/s", "Total (s)"]
    rows = [[r.model, r.engine, str(i), _i(m.tokens_in), _i(m.tokens_out), _f(m.ttft_s, 2),
             _f(m.prefill_tps), _f(m.decode_tps), _f(m.total_s)]
            for r in results for i, m in enumerate(r.runs, 1)]
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

**Settings:** prompt `prompts/broll_mapping.txt` ({prompt_chars:,} chars) · context {s.ctx_size:,} · max output {s.max_tokens:,} tokens · temperature {s.temperature} · reasoning {"on" if s.reasoning else "off"} · {s.runs} run(s) per model, averaged

## Summary

{summary}

## Per run

{per_run_table(results)}

### Metric definitions
- **Tokens in / out**: prompt and generated tokens as counted by the engine's own tokenizer (output includes reasoning tokens).
- **TTFT**: wall time from sending the request to receiving the first streamed token (client-side, includes prompt processing).
- **Prefill tok/s**: prompt processing speed reported by the engine (LM Studio: tokens in ÷ its own TTFT).
- **Decode tok/s**: generation speed reported by the engine, excluding prefill.
- **Est. mem**: weights + KV cache at the configured context + 1.5 GB overhead; used for the "{TOO_LARGE}" check.
"""
    out_dir = RESULTS_DIR / "outputs" / stem
    for r in results:
        for i, m in enumerate(r.runs, 1):
            out_dir.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^A-Za-z0-9.]+", "-", f"{r.model}__{r.engine}__run{i}").strip("-")
            (out_dir / f"{name}.txt").write_text(m.text, encoding="utf-8")
    md += f"\nModel outputs (for answer-quality review): `results/outputs/{stem}/`\n"
    (RESULTS_DIR / f"{stem}.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps({
        "host": host, "time": now.isoformat(timespec="seconds"), "hardware": asdict(hw),
        "settings": asdict(s), "engines": engines,
        "results": [{**asdict(r), "runs": [{k: v for k, v in asdict(m).items() if k != "text"} for m in r.runs]}
                    for r in results],
    }, indent=2), encoding="utf-8")
    return str(RESULTS_DIR / f"{stem}.md")
