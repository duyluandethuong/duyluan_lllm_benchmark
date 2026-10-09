# Notes for agents working on this repo

Shared memory for any coding agent (Claude Code, Codex, ...) on any machine. Keep it current:
when you learn something non-obvious, add it here and commit it with your change.

## Owner preferences
- Luan runs this on several machines (macOS, Windows, Linux), including unreleased hardware.
  He changes things with agents, so keep the code small and obvious.
- Python via `uv` (`uv sync`, `uv run bench.py`). Don't use pip or conda.
- Don't commit or push unless asked.
- Results are data, not prose: tables in `results/*.md`, raw numbers in `results/*.json`.

## Layout
```
bench.py                     CLI: selection, fit check, loop, report
models.toml                  model registry (add models here only)
prompts/make_prompt.py       generator for prompts/broll_mapping.txt (fixed seed; don't regenerate casually)
llmbench/hardware.py         RAM/VRAM detection + placement() fit rule
llmbench/fetch.py            HF sizes (cached in models/<repo>/.llmbench_manifest.json), downloads, memory estimate
llmbench/report.py           tables + results files
llmbench/backends/base.py    Backend interface (available/load/run/unload), SSE helper, OOM markers
llmbench/backends/llamacpp.py   llama-server subprocess, /v1/chat/completions stream
llmbench/backends/mlx_backend.py + mlx_worker.py   mlx-lm in a child process (memory freed on exit)
llmbench/backends/lmstudio.py   lms CLI + /api/v1/chat stream
```
New engine: subclass `Backend`, add it to `all_backends()` in `backends/__init__.py`.

## Measurement rules (keep consistent across engines)
- TTFT = client-side, from sending the request to the first streamed content or reasoning token.
- Prefill/decode tok/s = the engine's own numbers (llama.cpp `timings`, mlx `prompt_tps`/`generation_tps`,
  LM Studio `tokens_per_second`; LM Studio has no prefill rate, so it is input_tokens / its TTFT).
- Every run starts with a random "Request id" line (`prompt.for_run`). Without it LM Studio
  reuses its prompt cache and TTFT drops from seconds to ~0.04 s. llama.cpp also sends `cache_prompt: false`
  and runs with `--cache-ram 0`.
- Reasoning is off by default, so output lengths stay comparable.
- The prompt is 20,004 tokens with the Qwen3.5/3.6/3.8 tokenizer. Characters/4 overestimated it badly
  (the first version came out at 34K) because timestamps and IDs tokenize densely. Measure with a real tokenizer.

## Gotchas found so far
- **LM Studio 0.4.x does not index symlinked models**, so they don't show up in `lms ls`. Hard links work.
  `lms import -l` also writes a *relative* symlink if you give it a relative path, which is broken.
  So `lmstudio.py` hard-links (or copies) files into `<lmstudio>/models/<publisher>/<repo>/`.
- LM Studio models folder: `downloadsFolder` in `~/.lmstudio/settings.json` or `~/.cache/lm-studio/settings.json`.
  The `lms` binary is in `<that dir>/bin/`.
- LM Studio model keys are derived (e.g. `unsloth/qwen3.5-0.8b`). The script finds the key by matching `path`
  in `lms ls --json`. `lms load <path>` does not work.
- `lms load` defaults to `--parallel 4`. We pass `--parallel 1`.
- llama-server (build 11429+): `--fit` is on by default and `-ngl` defaults to auto, so partial GPU offload
  is automatic. We only force `-ngl 0` for CPU placement.
- mlx-lm 0.32 loads the Qwen3.5/3.6/3.8 4-bit repos from mlx-community fine (they include vision files, which are ignored).
- Qwen3.6/3.8 are hybrid attention models: only 1 in 4 layers has a KV cache, so the KV estimate is small.
- Gemma 4 mixes a few global layers (num_global_key_value_heads x global_head_dim 512) with sliding-window
  layers (cache capped at sliding_window). `Arch` has sliding_* fields for this. It ignores KV sharing
  (E2B/E4B) and K=V (12B/26B), so the estimate is a slight overestimate, which is the safe direction.
- QAT Gemma checkpoints were trained for Q4_0, so the GGUF entries use Google's own Q4_0 files.
- File reuse order: ./models, then the HF cache (`try_to_load_from_cache`, resolved to blobs/), then LM Studio's folder.
  Matches are by exact repo + filename + size. An HF cache dir can exist with no snapshot (empty); that's a miss.
- **Don't run anything else (not even a smoke test) while a benchmark is running.** It skews the numbers.
  This happened once on 2026-10-09 and the affected combination had to be rerun.
- Every run's generated text goes to `results/outputs/<run>/`. Check it when tokens_out hits the
  --max-tokens cap (looping, or thinking despite reasoning off).

## Status
- 2026-10-09: built and verified on macOS (M5 Max, 128 GB) with all 4 engines. The smoke model passed on all of them.
  Full Qwen3.8 27B / Qwen3.6 35B A3B Mac run in progress; results land in `results/`.
- **Not yet run on Windows or Linux.** Expect to fix: GPU detection (`hardware.py`: nvidia-smi,
  /sys/class/drm for AMD, Windows registry qwMemorySize), llama-server discovery, and LM Studio paths on Windows
  (`%USERPROFILE%\.lmstudio`). If detection is wrong, the user can pass `--vram-gb` / `--ram-gb`; fix the code too.
- 2026-10-09: added Gemma 4 E2B / E4B / 12B QAT / 26B A4B QAT (not benchmarked yet).
