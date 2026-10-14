# Notes for agents working on this repo

Shared memory for any coding agent (Claude Code, Codex, ...) on any machine. Keep it current:
when you learn something non-obvious, add it here and commit it with your change.

## Owner preferences
- Luan runs this on several machines (macOS, Windows, Linux), including unreleased hardware.
  He changes things with agents, so keep the code small and obvious.
- Python via `uv` (`uv sync`, `uv run bench.py`). Don't use pip or conda.
- Don't commit or push unless asked.
- Results are data, not prose: tables in `results/*.md`, raw numbers in `results/*.json`.
- **Results are never committed** (`results/` is gitignored, the user's decision). They stay on the machine that ran them.

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
llmbench/backends/vllm_backend.py   vllm/vllm-openai container per load (docker run/rm), /v1/chat/completions stream
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

- `--users N` (2026-10-14): N copies of the prompt (each with its own Request id) sent at the same moment to one
  loaded model, reloaded per N. llama.cpp: `-np N -c ctx*N`. LM Studio: `--parallel N -c ctx*N` (assumes its unified
  KV cache shares -c across slots; not verified). mlx-lm has no slots: mlx_worker.py runs all prompts in one
  `BatchGenerator` (prefill batches of 8, so users 9+ wait). With 2+ users prefill is tokens in / TTFT for every engine:
  llama.cpp's per-slot `prompt_per_second` was ~2x the client-side rate and MLX has none in a batch.
  `run_batch` waits in 0.5 s steps: a blocking wait (`pool.map`, or the exit of `with ThreadPoolExecutor`) only sees
  Ctrl-C once the whole round is done (it hung for minutes).
  Testing Ctrl-C from an agent shell: `cmd &` in a non-interactive shell starts with SIGINT *ignored*, so `kill -INT`
  does nothing and looks like a hang. Launch via Popen(preexec_fn=lambda: signal.signal(SIGINT, SIG_DFL)).
  Fit check counts one ctx-sized KV cache per user; once a count is too large, higher counts are skipped.
  Smoke (Qwen3.5 0.8B, M5 Max) aggregate out tok/s with 128-token outputs is prefill-bound: 1 vs 4 users ~55 -> 62
  (llama.cpp), 85 -> 101 (MLX). Use the default --max-tokens for real decode scaling.
  A GPU+CPU placement with 2+ users counts as too large: on 2x RTX PRO 4500 Qwen3.8 27B at 40 users
  (est. 97 GB vs 63 GB VRAM) spilled to RAM and ran at ~0.15 tok/s per user for 4 hours before it was stopped.
- vLLM (2026-10-15): one container per (model, user count): `--max-model-len ctx --max-num-seqs users`,
  `--no-enable-prefix-caching`, TP = all GPUs, `--language-model-only`, KV cache dtype auto (bf16, like llama.cpp's f16),
  `chat_template_kwargs.enable_thinking` = --reasoning. vLLM shares one paged KV pool across requests, so the fit
  check counts one context (`shared_kv`); users that don't fit wait in vLLM's queue (shows as TTFT).
  vLLM gives no per-request timings: prefill = tokens in / TTFT, decode = (tokens out - 1) / (total - TTFT), client-side.
  Startup is ~90-100 s per load (torch.compile + CUDA graphs; cache in ~/.cache/llmbench-vllm).

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
  is automatic. We force `-ngl 0` for CPU placement and `-ngl 999` when our fit check says it fits on the GPU.
- mlx-lm 0.32 loads the Qwen3.5/3.6/3.8 4-bit repos from mlx-community fine (they include vision files, which are ignored).
- Qwen3.6/3.8 are hybrid attention models: only 1 in 4 layers has a KV cache, so the KV estimate is small.
- Gemma 4 mixes a few global layers (num_global_key_value_heads x global_head_dim 512) with sliding-window
  layers (cache capped at sliding_window). `Arch` has sliding_* fields for this. It ignores KV sharing
  (E2B/E4B) and K=V (12B/26B), so the estimate is a slight overestimate, which is the safe direction.
- QAT Gemma checkpoints were trained for Q4_0, so the GGUF entries use Google's own Q4_0 files.
- File reuse order: ./models, then the HF cache (`try_to_load_from_cache`, resolved to blobs/), then LM Studio's folder.
  Matches are by exact repo + filename + size. An HF cache dir can exist with no snapshot (empty); that's a miss.
- **Never run two benchmarks at once.** On 2026-10-09 an agent run and a run from the user's terminal
  overlapped, and both slowed down about 2x (Qwen3.6 A3B decode went from ~50 to 27 tok/s). bench.py now holds
  `results/.bench.lock` and refuses to start a second run (--download-only is exempt). Engine logs are named
  with the pid, so parallel processes can't overwrite each other's logs.
- **Background load matters a lot.** On 2026-10-10, Xcode builds, a VM and the iOS Simulator running alongside
  cut Qwen3.8 27B llama.cpp decode from 16.9 to 10.3 tok/s. bench.py now samples CPU load before starting and
  warns with the top processes (`warn_if_busy`); it warns but doesn't block. Ask the user to idle the machine.
- LM Studio launched from an agent shell quit by itself within seconds (no crash report). The user had to open it
  by hand. If `lms server status` says not running, ask the user to open LM Studio.
- Gemma 4 E4B and 26B A4B ignore "exactly 30 placements" and keep emitting valid entries until the 4096 cap.
  That's the model, not looping; Qwen3.8 usually returns exactly 30 (once on N1X/CUDA it also ran to the cap).
- Every run's generated text goes to `results/outputs/<run>/`. Check it when tokens_out hits the
  --max-tokens cap (looping, or thinking despite reasoning off).

## Status
- 2026-10-09: built and verified on macOS (M5 Max, 128 GB) with all 4 engines. The smoke model passed on all of them.
  No clean Mac numbers recorded yet (see below).
- 2026-10-13/15: first Linux machine, `duyluanai` (Ubuntu 26.04, Core Ultra 7 270K Plus, 123 GB, 2x RTX PRO 4500
  Blackwell 32 GB, PCIe 5.0 x8/x8, `iommu=pt`). Detection worked unchanged. llama.cpp: GitHub release b11583
  `ubuntu-cuda-13.4-x64` + the matching `cudart-...` tarball extracted into one dir (libggml-cuda has RUNPATH $ORIGIN),
  `~/.local/opt/llama-b11583/llama-b11583/llama-server`. The box normally runs a production vLLM container
  (~/duyluanai_llm_setup, holds ~30 GB per GPU): `docker compose stop vllm` before a run, `start` after.
  llama.cpp splits layers over both GPUs. Gemma 4 E2B is anomalous there: aggregate stays ~75 tok/s from 1 to 50 users
  (E4B scales to ~480); not investigated. Gemma 26B A4B at 50 users: llama.cpp OOM on GPU1 (KV) though est. 56 GB fits.
  Full llama.cpp matrix: results/2026-10-14_224211_duyluanai.
- 2026-10-10: first Windows run, on an ARM64 laptop with NVIDIA RTX Spark N1X (64 GB unified, 20-core CPU):
  - `nvidia-smi` also lists "NVIDIA NPU, [N/A]"; `_nvidia()` skips rows without a number. The GPU reports ~all of RAM,
    so `detect()` treats a single GPU with >=90% of RAM as unified (budget 75% of RAM, like Apple).
  - `winget install astral-sh.uv ggml.llamacpp` works; winget's llama.cpp is the **Vulkan arm64** build (sees the GPU).
    A fresh shell is needed for PATH. uv first picked an x86_64 Python (emulated); use
    `uv venv --python cpython-3.14-windows-aarch64-none`.
  - Vulkan compiles shaders on first use: the very first smoke run had TTFT 12.2 s vs 1.8 s once warm (cache persists).
    If a model's log shows the first 2048-token chunk much slower than the rest, rerun it.
  - **LM Studio has no GPU runtime on Windows ARM64.** The default `x86_64-nvidia-cuda` runtime exits with
    0xC0000135 (DLL not found); `x86_64-vulkan` fails on device selection; `llama.cpp-win-arm64` works but CPU-only
    (smoke prefill 347 tok/s vs 11K on llama.cpp). Left selected: `llama.cpp-win-arm64@2.55.0`.
  - Luan's call: on this machine benchmark **llama.cpp (GPU) only** (`--all -e llamacpp`); no CPU-only LM Studio runs.
  - **llama-server prefill stalls here**: the GPU shows 96% busy at ~25 W instead of ~75 W for most of the prompt, so
    Qwen3.8 27B TTFT varied 45–210 s across runs. `llama-bench` on the same model sustains 660–740 tok/s
    (pp2048 up to depth 16K). Decode numbers are stable; treat prefill/TTFT from this machine as unreliable.
    llama.cpp rows now also get a **llama-bench prefill** column (`bench_prefill()` in llamacpp.py, same GGUF,
    `-p <tokens in>`, run after the server stops). On 2026-10-13 that column stalled too (Qwen3.8 27B: 69 tok/s for one
    20K prompt, while pp2048 short bursts gave 738 the same hour). So it is not llama-server: any *long continuous*
    prefill drops into the ~25 W mode, short bursts don't. Likely GPU power management in the driver/firmware
    (fix pending from the OEM). Until then this machine has no reliable full-prompt prefill number; decode is fine.
  - **Fix (2026-10-14): use the CUDA build, not winget's Vulkan one.** GitHub releases ship
    `llama-bXXXX-bin-win-cuda-13.4-arm64.zip` + `cudart-llama-bin-win-cuda-13.4-arm64.zip` (extract both into one dir;
    here `%LOCALAPPDATA%\llama.cpp-cuda\b11592`) and run with `--llama-server <dir>\llama-server.exe`.
    Qwen3.8 27B llama-bench: pp20000 809 tok/s steady (Vulkan 64), tg 16.2 (Vulkan 12.0). The prefill stall and
    the `nvogla64.dll` crashes are Vulkan-path problems; the notes above apply to Vulkan only.
    Clean N1X result: `results/2026-10-14_025519_LAPTOP-LFUSQL4B` (all 6 models, server and llama-bench prefill agree).
    In that run Qwen3.8 27B wrote 73 placements to the 4096 cap (valid JSON, no looping) instead of 30.
  - 2026-10-14: Tài reported partial CPU offload on N1X and 3–7.5x prefill / +20–60% decode after `-ngl 999`.
    Not reproduced here: `llama-fit-params` gives `-ngl -1` (all layers) for Qwen3.8 27B @32K, and with
    `-ngl 999` decode stayed 12.0 tok/s and prefill still stalled (server 210, llama-bench 64 tok/s). llamacpp.py now
    passes `-ngl 999` for full-GPU placements anyway (explicit, harmless). If Tài's numbers are real, the difference is
    elsewhere: llama.cpp backend/build (CUDA vs our winget Vulkan b11515) or NVIDIA driver (ours 32.0.16.1662).
  - **NVIDIA driver crash**: llama-server sometimes dies with 0xc0000005 in `nvogla64.dll` (ASUS ARM64 driver
    32.0.16.1662), e.g. Qwen3.6 35B A3B mid-generation, or on a request that reuses the slot's prompt cache.
    Default flags (4 slots) crashed on the first request. `-np 1 --cache-ram 0` survived 20+ requests.
- 2026-10-09: added Gemma 4 E2B / E4B / 12B QAT / 26B A4B QAT (not benchmarked yet).
- **Fixed 2026-10-13: LM Studio + MLX thinking toggle.** LM Studio offers `reasoning: off` only for GGUF files
  whose template it recognises and for its own catalog models (defined by a model.yaml under <lmstudio>/hub/models).
  Raw mlx-community repos get no toggle, and `chat_template_kwargs`, `reasoning_effort: none` and `/no_think` are all
  ignored. lmstudio.py now writes <lmstudio>/hub/models/llmbench/<id>-mlx/{model.yaml,manifest.json} (base key =
  "<publisher>/<repo>".lower(), customField enableThinking -> setJinjaVariable enable_thinking) and loads that key.
  Verified: 0 reasoning tokens with reasoning off.
- 2026-10-13: first clean full Mac run (M5 Max 128 GB, idle except one light VM): all 6 models x 4 engines, 1 run each.
  Engine-reported speeds match client-side wall time on every row. Single runs vary: rerun with --runs 3 before
  drawing fine conclusions. Unexplained so far: LM Studio GGUF decodes Qwen3.8 27B at 34.8 tok/s vs llama.cpp 23.7
  on the same file, while it is slower than llama.cpp on every other model.
- The earlier mlx-lm Qwen3.8 4096-token output was a one-off; the clean run stopped at 1,800 tokens.
- **2026-10-14: direct-engine Mac rerun (no LM Studio)**, to compare with Tài's M4 Max and the N1X: bench code at
  2b6d377, `--engine llamacpp,mlx --runs 3`, one model per invocation with a 300 s cooldown between models,
  `LLAMA_ARG_N_GPU_LAYERS=999`, KV f16, flash attention auto (= on), llama.cpp b11429, mlx-lm 0.32.0 / mlx 0.32.3,
  AC power, **High Power Mode on** (`pmset -g` powermode 2), LM Studio fully quit. Local results:
  `results/2026-10-14_direct_M5Max/` (+ SUMMARY_M5_vs_M4.md, llama-bench/). M5 Max beats M4 Max on all 12
  model x engine pairs: decode 1.13–1.77x, prefill 2–4x. Qwen3.8 27B decode = 63% (llama.cpp) / 76% (MLX) of the
  614 GB/s ÷ 16.46 GB bandwidth ceiling. llama-bench (pp20000, tg128 @ d20000) agrees with the server within 1.0–1.2x.
  - **The "M5 slower than M4" numbers came from LM Studio**, not the hardware: LM Studio's own runtimes
    (llama.cpp-mac 2.55.0, mlx-llm 1.13.1) are 20–40% slower than llama-server / mlx-lm on the same files
    (Qwen3.6 A3B MLX: 77 vs 121 tok/s). Compare machines with direct engines only.
  - **High Power Mode matters on M5 Max.** Without it (2026-10-13): mlx-lm Qwen3.8 decode 22.9, Gemma E2B prefill
    6.1K tok/s; with it: 28.4 and 16.6K. Record `pmset -g | grep powermode` with every Mac run.
  - mlx-lm's first request after load pays Metal kernel compilation: prefill run 1 can be ~3x slower than runs 2–3
    (Qwen3.6 A3B: 1,093 vs ~3,900). The report averages runs, so also look at the median.
  - Back-to-back llama.cpp runs throttle a little (Gemma E4B decode 85 -> 81 -> 76 over 3 runs).
- 2026-10-13: first clean run on the Mac Studio (M5 Ultra, 256 GB): `--all -e llamacpp,mlx --runs 1`, idle machine,
  llama-server b11539 from `~/ai/macstudio_llm/tools/llama-b11539/` (pass `--llama-server`). Local results:
  results/2026-10-13_185604_duyluanM5Ultra. That Mac also runs a llama-swap server (`~/ai/macstudio_llm/serving/llama-swap.yaml`,
  port 8080, MLX models kept resident, ~100 GB). Fine to leave up while idle; the benchmark doesn't use it.
  **But it gets outside traffic** (clients come in over [::1], likely a tunnel): on 2026-10-14 a request stream to
  qwen3.8:8 pinned the GPU at 100% and a Gemma E4B `--users` run collapsed at 20 users (aggregate 299 -> 76 tok/s).
  `warn_if_busy` misses it (CPU stays idle; it's GPU load). Since 2026-10-15 every run records **Other GPU %**
  (`gpu_time_by_pid()` in hardware.py: IOKit AppUsage accumulatedGPUTime per pid, before/after each round, minus
  bench.py's own process tree and WindowServer) and prints a warning above 5%. Verified: a competing MLX matmul
  loop reads 48-92%, idle reads 0.0. Before a long run also check `lsof -nP -iTCP:8080 -sTCP:ESTABLISHED` is empty.
  The mlx-lm first-request penalty is bigger on small models there: Gemma 4 E2B MLX prefill 9.3K, 19.6K, 37.1K tok/s
  over three runs.
- 2026-10-14 Mac Studio `--users` (Gemma 4 E4B, 20K prompt, 4096 cap, GPU free of llama-swap): aggregate out tok/s
  llama.cpp peaks at 10 users (~300) then drops (20: 190, 40: ~205, 50: 216; TTFT max 264 s at 50) because prompts are
  prefilled one slot at a time. mlx-lm BatchGenerator keeps climbing (10: 287, 20: 336, 40: 343, 50: 429; TTFT max 72 s).
  Agg. tok/s at high counts depends on output lengths (wall = slowest user), so compare per-user decode and TTFT too.
- 2026-10-15 Mac Studio full `--users all` matrix (6 models x llama.cpp/MLX x 1..50 users = 108 cells, all OK, GPU
  sampled every ~30 s and clean for every round). Local results: results/FINAL_concurrency_duyluanM5Ultra.{md,json},
  built by results/console_logs_full_2026-10-14/consolidate.py (one bench.py per model+engine, timestamped log,
  2 idle GPU minutes before each). bench.py now does this itself: `--all -e llamacpp,mlx --users all` gives one
  report with a row for every cell ("Cannot run: <reason>" when unmeasured) and Other GPU % per run. Took ~8 h; Qwen3.8 27B llama.cpp alone ~1.5 h (TTFT max 1,123 s at 50 users).
  MLX beats llama.cpp at 50 users on every model (1.3-2.8x aggregate); llama.cpp peaks at 10-40 users.
