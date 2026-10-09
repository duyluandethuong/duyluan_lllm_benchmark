"""Model registry (models.toml) and run settings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import ROOT

REGISTRY = ROOT / "models.toml"


@dataclass(frozen=True)
class Arch:
    attn_layers: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class Artifact:
    """One downloadable form of a model (a GGUF file set, or an MLX repo snapshot)."""

    fmt: str  # "gguf" | "mlx"
    repo: str
    quant: str
    files: tuple[str, ...] = ()  # gguf: explicit files; mlx: empty = whole repo


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    smoke: bool
    arch: Arch | None
    artifacts: dict[str, Artifact]  # keyed by fmt
    llamacpp_args: tuple[str, ...] = ()


@dataclass
class Settings:
    ctx_size: int = 32768
    max_tokens: int = 4096
    runs: int = 1
    reasoning: bool = False
    temperature: float = 0.0
    seed: int = 42
    extra: dict = field(default_factory=dict)


def load_models(path: Path = REGISTRY) -> list[Model]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    models = []
    for m in data.get("model", []):
        artifacts = {}
        for fmt in ("gguf", "mlx"):
            if fmt in m:
                a = m[fmt]
                artifacts[fmt] = Artifact(fmt, a["repo"], a.get("quant", "?"), tuple(a.get("files", ())))
        arch = Arch(**m["arch"]) if "arch" in m else None
        models.append(Model(
            id=m["id"], name=m.get("name", m["id"]), smoke=m.get("smoke", False), arch=arch,
            artifacts=artifacts, llamacpp_args=tuple(m.get("llamacpp_args", ())),
        ))
    return models
