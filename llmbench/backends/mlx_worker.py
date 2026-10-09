"""Child process for the MLX backend. Holds one model in memory and answers
JSON-line requests on stdin, so memory is fully released when it exits.

Protocol (one JSON object per line):
  argv[1]  : {"path": ..., "reasoning": bool}
  stdout   : {"loaded": load_seconds}            once, or {"error": "..."}
  stdin    : {"prompt": ..., "max_tokens": N, "temperature": T, "seed": S}
  stdout   : {"tokens_in": ..., "tokens_out": ..., "ttft_s": ..., ...}
"""

from __future__ import annotations

import json
import sys
import time


def main() -> None:
    out = sys.stdout
    sys.stdout = sys.stderr  # keep library prints off the protocol channel
    cfg = json.loads(sys.argv[1])

    def send(obj: dict) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    try:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        t0 = time.perf_counter()
        model, tokenizer = load(cfg["path"])
        send({"loaded": time.perf_counter() - t0})
    except Exception as e:  # noqa: BLE001 - reported to the parent
        send({"error": f"{type(e).__name__}: {e}"})
        return

    for line in sys.stdin:
        req = json.loads(line)
        try:
            mx.random.seed(req["seed"])
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": req["prompt"]}], add_generation_prompt=True,
                tokenize=False, enable_thinking=cfg["reasoning"],
            )
            sampler = make_sampler(temp=req["temperature"])
            t0 = time.perf_counter()
            ttft, last, parts = None, None, []
            for r in stream_generate(model, tokenizer, prompt, max_tokens=req["max_tokens"], sampler=sampler):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(r.text)
                last = r
            total = time.perf_counter() - t0
            send({
                "tokens_in": last.prompt_tokens, "tokens_out": last.generation_tokens,
                "ttft_s": ttft, "prefill_tps": last.prompt_tps, "decode_tps": last.generation_tps,
                "total_s": total, "peak_memory_gb": last.peak_memory, "text": "".join(parts),
            })
        except Exception as e:  # noqa: BLE001
            send({"error": f"{type(e).__name__}: {e}"})
        mx.clear_cache()


if __name__ == "__main__":
    main()
