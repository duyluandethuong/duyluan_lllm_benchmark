"""The benchmark prompt, identical for every model and engine."""

from __future__ import annotations

import uuid

from . import ROOT

PROMPT_FILE = ROOT / "prompts" / "broll_mapping.txt"


def load() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def for_run(text: str) -> str:
    """Prefix a unique id so no engine can reuse a cached prefix from a previous
    run (LM Studio and llama.cpp both cache prompts). Costs ~10 tokens."""
    return f"Request id: {uuid.uuid4().hex[:12]}\n\n{text}"
