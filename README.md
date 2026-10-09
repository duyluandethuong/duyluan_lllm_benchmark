# duyluan_llm_benchmark

Benchmark local LLMs on any machine (macOS, Windows, Linux) with one fixed, realistic
~20K-token task: map ~90 B-roll clips (metadata) onto the transcript of a ~20-minute
A-roll video. Every model and engine gets byte-identical input.

| Engine | Format | Runs on |
|---|---|---|
| llama.cpp (`llama-server`) | GGUF | macOS, Windows, Linux |
| MLX (`mlx-lm`) | MLX | macOS, Apple Silicon |
| LM Studio (`lms`) | GGUF | macOS, Windows, Linux |
| LM Studio (`lms`) | MLX | macOS, Apple Silicon |

Each run reports **tokens in, tokens out, TTFT, prefill tok/s, decode tok/s** (plus load time
and estimated memory). The results are printed as a table and saved to `results/<date>_<host>.md` (and `.json`).
If the machine cannot hold a model, its row says **Too large to run**.
The generated text for every run is saved in `results/outputs/<run>/`, so you can compare answer quality as well as speed.

Models so far: Qwen3.8 27B, Qwen3.6 35B A3B, Gemma 4 E2B, Gemma 4 E4B, Gemma 4 12B QAT, Gemma 4 26B A4B QAT
(4-bit everywhere; QAT models use Google's Q4_0 GGUF).

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Install at least one engine:
   - **llama.cpp**:
     - macOS: `brew install llama.cpp`
     - Windows: `winget install llama.cpp`
     - Linux: `brew install llama.cpp`, or a release build (CUDA/Vulkan/ROCm) from
       <https://github.com/ggml-org/llama.cpp/releases>. Put `llama-server` on PATH or pass `--llama-server`.
   - **LM Studio**: install it and open it once, so the `lms` CLI exists.
   - **MLX**: installed by `uv sync` on Apple Silicon.
3. `uv sync`

## Run

```bash
uv run bench.py --list                      # hardware, engines found, models
uv run bench.py -m smoke                    # 0.5 GB end-to-end check on a new machine
uv run bench.py                             # choose models interactively
uv run bench.py -m qwen3.8-27b,qwen3.6-35b-a3b
uv run bench.py --all -e llamacpp,lmstudio  # every model, chosen engines only
```

| Option | Default | |
|---|---|---|
| `-m/--model` | interactive | ids from `models.toml`, comma-separated |
| `--all` | | every model except `smoke` |
| `-e/--engine` | all available | `llamacpp`, `mlx`, `lmstudio` (both formats), `lmstudio-gguf`, `lmstudio-mlx` |
| `--runs` | 1 | timed runs per model+engine, averaged |
| `--ctx` | 32768 | context size |
| `--max-tokens` | 4096 | output cap |
| `--reasoning` | off | `on` lets thinking models think (output tokens then include reasoning) |
| `--models-dir` | `./models` | also env `LLMBENCH_MODELS_DIR` |
| `--download-only` | | fetch files, don't benchmark |
| `--vram-gb`, `--ram-gb` | detected | override detection (unknown or new hardware) |
| `--force` | | skip the "Too large to run" check |

## Model files

Files go to `./models/<publisher>/<repo>/`, for example `models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`.
Before downloading anything, the script looks for the same repo file, at the right size, in this order:

1. `./models`. **To skip downloading on a new machine, copy the `models/` folder over.**
2. The Hugging Face cache (`~/.cache/huggingface/hub`, or `$HF_HUB_CACHE`).
3. LM Studio's models folder (`<lmstudio>/models/<publisher>/<repo>/`).

A file found in 2 or 3 is hard-linked into `./models` (no extra disk space), or symlinked or copied
if it's on another drive. The match has to be the same repo. A similar quant from a different publisher (e.g.
`lmstudio-community` instead of `unsloth`) is a different file and gets downloaded.

LM Studio uses the same files. They are hard-linked into LM Studio's models folder (no extra disk
space), or copied if that folder is on a different drive. LM Studio does not index symlinks.
Before each LM Studio load the script runs `lms unload --all`, so LM Studio models you have loaded get unloaded.

## Adding a model

Add a block to `models.toml`. That's all. See the comments at the top of the file. Use
`uv run bench.py --list` to check it parses.

## How "Too large to run" is decided

estimate = weight file size + KV cache at `--ctx` (from `arch` in models.toml) + 1.5 GB.

- **Apple Silicon**: must fit in the Metal GPU budget (~75% of RAM, or `iogpu.wired_limit_mb` if you raised it).
- **Discrete GPU**: fits fully in VRAM → `GPU`. Otherwise fits in VRAM + 85% of RAM → `GPU+CPU`
  (llama.cpp splits it automatically). Otherwise it doesn't fit.
- **No GPU**: must fit in 85% of RAM → `CPU`.

If an engine still fails to load with an out-of-memory error, the row also says "Too large to run".
Engine logs are in `results/logs/`.
