"""Resolve model files: sizes (before download), local paths, downloads.

Layout:  <models_dir>/<publisher>/<repo>/<files>     e.g. models/unsloth/Qwen3.8-27B-GGUF/x.gguf
A file is only downloaded when no copy exists. Lookup order:
  1. ./models (copy this folder between machines to skip downloads)
  2. the Hugging Face cache (~/.cache/huggingface/hub, or $HF_HUB_CACHE)
  3. LM Studio's models folder (<lmstudio>/models/<publisher>/<repo>/)
Copies found in 2/3 are hard-linked into ./models (symlink, then copy, as fallbacks)
so every engine reads the same file and nothing is stored twice.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .config import Arch, Artifact

MANIFEST = ".llmbench_manifest.json"  # remote file sizes, cached so later runs work offline
MLX_SKIP = {".gitattributes", "README.md"}
GB = 1024**3


def repo_dir(models_dir: Path, art: Artifact) -> Path:
    return models_dir / art.repo


def _remote_sizes(art: Artifact, cache: Path) -> dict[str, int] | None:
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        from huggingface_hub import HfApi

        tree = HfApi().list_repo_tree(art.repo, recursive=True)
        sizes = {f.path: f.size for f in tree if getattr(f, "size", None) is not None}
    except Exception as e:  # offline, gated, renamed...
        print(f"  ! could not list {art.repo} on Hugging Face: {e}")
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sizes, indent=1))
    return sizes


def wanted_files(art: Artifact, models_dir: Path) -> dict[str, int | None]:
    """{relative file: expected size or None if unknown}."""
    d = repo_dir(models_dir, art)
    remote = _remote_sizes(art, d / MANIFEST)
    if art.fmt == "gguf":
        return {f: (remote or {}).get(f) for f in art.files}
    if remote is not None:
        return {f: s for f, s in remote.items() if f not in MLX_SKIP}
    # Offline and no manifest: trust whatever was copied in.
    return {str(p.relative_to(d)): p.stat().st_size for p in d.rglob("*")
            if p.is_file() and not p.name.startswith(".")} if d.exists() else {}


def weights_gb(art: Artifact, models_dir: Path) -> float | None:
    files = wanted_files(art, models_dir)
    d = repo_dir(models_dir, art)
    total = 0
    for f, size in files.items():
        if art.fmt == "mlx" and not f.endswith(".safetensors"):
            continue
        if size is None:
            p = d / f
            if not p.exists():
                return None
            size = p.stat().st_size
        total += size
    return total / GB if total else None


def kv_cache_gb(arch: Arch | None, ctx: int, weights: float) -> float:
    if arch is None:
        return weights * 0.15  # rough fallback when the registry has no arch info
    # K and V in fp16. Linear/recurrent layers (Qwen3.6+) hold a tiny fixed state: ignored.
    full = 2 * arch.attn_layers * arch.kv_heads * arch.head_dim * ctx * 2
    sliding = 2 * arch.sliding_layers * arch.sliding_kv_heads * arch.sliding_head_dim * min(ctx, arch.sliding_window) * 2
    return (full + sliding) / GB


OVERHEAD_GB = 1.5  # compute buffers, runtime, activations during a 20K prefill


def estimate_gb(art: Artifact, arch: Arch | None, ctx: int, models_dir: Path) -> float | None:
    w = weights_gb(art, models_dir)
    if w is None:
        return None
    return w + kv_cache_gb(arch, ctx, w) + OVERHEAD_GB


def missing(art: Artifact, models_dir: Path) -> list[str]:
    d = repo_dir(models_dir, art)
    out = []
    for f, size in wanted_files(art, models_dir).items():
        p = d / f
        if not p.exists() or (size is not None and p.stat().st_size != size):
            out.append(f)
    return out


def _existing_copy(art: Artifact, f: str, size: int | None) -> Path | None:
    """The same repo file already on disk elsewhere (HF cache, LM Studio)."""
    from huggingface_hub import try_to_load_from_cache

    from .backends.lmstudio import models_root

    candidates = []
    hit = try_to_load_from_cache(art.repo, f)
    if isinstance(hit, str):
        candidates.append(Path(hit))
    candidates.append(models_root() / art.repo / f)
    for c in candidates:
        if c.exists() and (size is None or c.stat().st_size == size):
            return c.resolve()  # HF cache snapshots are symlinks into blobs/
    return None


def _place(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    for how, fn in (("hard-linked", os.link), ("symlinked", os.symlink)):
        try:
            fn(src, dst)
            return how
        except OSError:  # other drive / no symlink rights (Windows without dev mode)
            pass
    shutil.copy2(src, dst)
    return "copied"


def ensure(art: Artifact, models_dir: Path) -> Path:
    """Reuse or download what is missing; return the path the engine should load
    (first GGUF shard, or the MLX directory)."""
    from huggingface_hub import hf_hub_download

    d = repo_dir(models_dir, art)
    sizes = wanted_files(art, models_dir)
    todo = missing(art, models_dir)
    for i, f in enumerate(todo, 1):
        if src := _existing_copy(art, f, sizes.get(f)):
            print(f"  reusing {src} ({_place(src, d / f)})")
            continue
        print(f"  downloading {art.repo}/{f} ({i}/{len(todo)})")
        hf_hub_download(art.repo, f, local_dir=d)
    return d / art.files[0] if art.fmt == "gguf" else d
