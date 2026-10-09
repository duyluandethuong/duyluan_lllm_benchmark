"""Engine registry. Add a new engine by appending it in `all_backends`."""

from __future__ import annotations

from .base import Backend, RunMetrics, TooLarge
from .llamacpp import LlamaCpp
from .lmstudio import LmStudio
from .mlx_backend import Mlx


def all_backends(llama_server: str | None = None, lms: str | None = None) -> list[Backend]:
    return [
        LlamaCpp(llama_server),
        Mlx(),
        LmStudio("gguf", lms),
        LmStudio("mlx", lms),
    ]


# CLI shorthands: "--engine lmstudio" = both LM Studio formats
ALIASES = {"lmstudio": ["lmstudio-gguf", "lmstudio-mlx"], "llama.cpp": ["llamacpp"]}

__all__ = ["Backend", "RunMetrics", "TooLarge", "all_backends", "ALIASES"]
