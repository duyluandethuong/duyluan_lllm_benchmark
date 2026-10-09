"""Benchmark local LLMs with one fixed ~20K-token prompt across engines.

    uv run bench.py                         # pick models interactively
    uv run bench.py -m qwen3.8-27b          # one model, every engine available here
    uv run bench.py --all -e llamacpp,mlx   # all models, chosen engines
    uv run bench.py --list                  # show models, engines, hardware
    uv run bench.py -m smoke                # quick end-to-end check (~0.5 GB)

See README.md for the full option list.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psutil

from llmbench import ROOT, fetch, prompt
from llmbench.backends import ALIASES, TooLarge, all_backends
from llmbench.backends.lmstudio import loaded_elsewhere
from llmbench.config import Model, Settings, load_models
from llmbench.hardware import detect
from llmbench.report import TOO_LARGE, SUMMARY_HEADER, Result, md_table, summary_rows, write


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", help="comma-separated model ids from models.toml")
    p.add_argument("--all", action="store_true", help="every non-smoke model")
    p.add_argument("-e", "--engine", help="comma-separated: llamacpp, mlx, lmstudio, lmstudio-gguf, lmstudio-mlx "
                                          "(default: all available)")
    p.add_argument("--list", action="store_true", help="list models/engines/hardware and exit")
    p.add_argument("--runs", type=int, default=1, help="timed runs per model+engine, averaged (default 1)")
    p.add_argument("--ctx", type=int, default=32768, help="context size (default 32768)")
    p.add_argument("--max-tokens", type=int, default=4096, help="max generated tokens (default 4096)")
    p.add_argument("--reasoning", choices=["on", "off"], default="off", help="thinking mode (default off)")
    p.add_argument("--models-dir", type=Path, default=Path(os.environ.get("LLMBENCH_MODELS_DIR", ROOT / "models")),
                   help="where model files live (default ./models, env LLMBENCH_MODELS_DIR)")
    p.add_argument("--download-only", action="store_true", help="fetch files for the selection and exit")
    p.add_argument("--vram-gb", type=float, help="override detected VRAM / GPU memory budget")
    p.add_argument("--ram-gb", type=float, help="override detected system RAM")
    p.add_argument("--force", action="store_true", help="skip the fits-in-memory check")
    p.add_argument("--llama-server", help="path to llama-server binary")
    p.add_argument("--lms", help="path to LM Studio `lms` CLI")
    return p.parse_args()


def pick_models(models: list[Model], args) -> list[Model]:
    by_id = {m.id: m for m in models}
    if args.model:
        ids = [x.strip() for x in args.model.split(",") if x.strip()]
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            sys.exit(f"Unknown model id(s): {', '.join(unknown)}. Known: {', '.join(by_id)}")
        return [by_id[i] for i in ids]
    if args.all:
        return [m for m in models if not m.smoke]
    print("Models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m.id:<20} {m.name}  [{', '.join(m.artifacts)}]")
    choice = input("Select (e.g. 1,2 or 'all'): ").strip().lower()
    if choice in ("all", "a", ""):
        return [m for m in models if not m.smoke]
    try:
        return [models[int(x) - 1] for x in choice.replace(" ", "").split(",")]
    except (ValueError, IndexError):
        sys.exit(f"Invalid selection: {choice!r}")


def pick_engines(args, backends):
    if not args.engine:
        return backends
    wanted = []
    for e in args.engine.split(","):
        wanted += ALIASES.get(e.strip(), [e.strip()])
    known = {b.id for b in backends}
    if bad := [w for w in wanted if w not in known]:
        sys.exit(f"Unknown engine(s): {', '.join(bad)}. Known: {', '.join(sorted(known | set(ALIASES)))}")
    return [b for b in backends if b.id in wanted]


def bench_one(model: Model, backend, hw, s: Settings, args, text: str) -> Result:
    art = model.artifacts[backend.fmt]
    res = Result(model.name, backend.label, art.quant, "OK")
    res.est_gb = fetch.estimate_gb(art, model.arch, s.ctx_size, args.models_dir)
    placement = hw.placement(res.est_gb) if res.est_gb is not None else "?"
    if res.est_gb is not None and placement is None and not args.force:
        res.status = TOO_LARGE
        print(f"  {TOO_LARGE}: needs ~{res.est_gb:.1f} GB")
        return res
    res.placement = placement or "forced"
    if res.est_gb is not None and res.est_gb > hw.ram_available_gb and hw.unified:
        print(f"  ! only {hw.ram_available_gb:.0f} GB free right now; close other apps for clean numbers")

    path = fetch.ensure(art, args.models_dir)
    if args.download_only:
        res.status = "Downloaded"
        return res
    try:
        print(f"  loading ({res.placement}, ~{res.est_gb or 0:.1f} GB)...", flush=True)
        res.load_s = backend.load(model, path, res.placement, s)
        for i in range(1, s.runs + 1):
            m = backend.run(prompt.for_run(text), s)
            res.runs.append(m)
            print(f"  run {i}: in {m.tokens_in:,} | out {m.tokens_out:,} | TTFT {m.ttft_s:.2f}s | "
                  f"prefill {m.prefill_tps or 0:,.1f} tok/s | decode {m.decode_tps:.1f} tok/s", flush=True)
    except TooLarge as e:
        res.status = TOO_LARGE
        print(f"  {TOO_LARGE}: {e}")
    except KeyboardInterrupt:
        backend.unload()
        raise
    except Exception as e:  # keep going with the next model/engine
        res.status = f"Error: {str(e)[:120]}"
        print(f"  {res.status}")
    finally:
        backend.unload()
    return res


LOCK = ROOT / "results" / ".bench.lock"


def acquire_lock() -> None:
    """Two benchmarks sharing one GPU silently halve each other's numbers: refuse."""
    try:
        pid = int(LOCK.read_text().strip())
        if pid != os.getpid() and psutil.pid_exists(pid) and "bench.py" in " ".join(psutil.Process(pid).cmdline()):
            sys.exit(f"Another benchmark is already running (pid {pid}). Wait for it, or stop it first; "
                     f"parallel runs skew both results. (Stale lock? delete {LOCK})")
    except (FileNotFoundError, ValueError, psutil.Error):
        pass
    LOCK.parent.mkdir(exist_ok=True)
    LOCK.write_text(str(os.getpid()))


def release_lock() -> None:
    try:
        if LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()
    except (FileNotFoundError, OSError):
        pass


def warn_if_busy(threshold: float = 25.0) -> None:
    """LLM speed is memory-bandwidth bound: builds, VMs or simulators running alongside
    lower the numbers. Warn (don't block) and name the biggest consumers."""
    procs = list(psutil.process_iter(["name"]))
    for p in procs:
        try:
            p.cpu_percent(None)  # prime per-process counters
        except psutil.Error:
            pass
    load = psutil.cpu_percent(interval=2.0)
    if load < threshold:
        return
    top = []
    for p in procs:
        try:
            top.append((p.cpu_percent(None), p.info["name"] or "?"))
        except psutil.Error:
            pass
    busiest = ", ".join(f"{n} {c:.0f}%" for c, n in sorted(top, reverse=True)[:5])
    print(f"! Machine is busy ({load:.0f}% CPU): {busiest}\n"
          f"  Close these for clean numbers; results will be lower than the machine can do.")


def main() -> None:
    # Windows consoles/redirects default to cp1252; never crash on a table character.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args()
    s = Settings(ctx_size=args.ctx, max_tokens=args.max_tokens, runs=args.runs, reasoning=args.reasoning == "on")
    models = load_models()
    hw = detect(args.vram_gb, args.ram_gb)
    backends = all_backends(args.llama_server, args.lms)
    status = {b.id: b.available() for b in backends}

    print(f"Machine: {hw.describe()}")
    for b in backends:
        ok, info = status[b.id]
        print(f"  [{'ok' if ok else '--'}] {b.label:<20} {info}")
    if args.list:
        print("\nModels (models.toml):")
        for m in models:
            print(f"  {m.id:<20} {m.name:<28} formats: {', '.join(m.artifacts)}{'  (smoke)' if m.smoke else ''}")
        return

    selected = pick_models(models, args)
    engines = [b for b in pick_engines(args, backends) if status[b.id][0]]
    if not engines:
        sys.exit("No usable engine on this machine; see the [--] lines above.")
    if busy := [x for x in loaded_elsewhere(args.lms) if x]:
        print(f"! LM Studio currently holds {', '.join(busy)} in memory; numbers may be skewed. "
              f"(Unload with: lms unload --all)")

    text = prompt.load()
    results: list[Result] = []
    if not args.download_only:
        acquire_lock()
        warn_if_busy()
    try:
        for model in selected:
            for b in engines:
                if b.fmt not in model.artifacts:
                    continue
                print(f"\n=== {model.name} · {b.label}")
                results.append(bench_one(model, b, hw, s, args, text))
    except KeyboardInterrupt:
        print("\nInterrupted; writing partial report.")
    finally:
        release_lock()

    if not results:
        sys.exit("Nothing was run (no model/engine combination matched).")
    print("\n" + md_table(SUMMARY_HEADER, summary_rows(results)))
    if not args.download_only:
        used = {b.label: status[b.id][1] for b in engines}
        print(f"\nReport: {write(results, hw, s, used, len(text))}")


if __name__ == "__main__":
    main()
