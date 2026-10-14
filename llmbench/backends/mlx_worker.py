"""Child process for the MLX backend. Holds one model in memory and answers
JSON-line requests on stdin, so memory is fully released when it exits.

Protocol (one JSON object per line):
  argv[1]  : {"path": ..., "reasoning": bool}
  stdout   : {"loaded": load_seconds}            once, or {"error": "..."}
  stdin    : {"prompt": ..., "max_tokens": N, "temperature": T, "seed": S}
  stdout   : {"tokens_in": ..., "tokens_out": ..., "ttft_s": ..., ...}
  stdin    : {"prompts": [...], ...}                concurrent users (continuous batching)
  stdout   : {"results": [{"tokens_in": ...}, ...]}
"""

from __future__ import annotations

import json
import sys
import time


def run_batch(model, tokenizer, texts: list[str], max_tokens: int, sampler) -> list[dict]:
    """Every prompt starts at once in one BatchGenerator, like N users hitting a server."""
    from mlx_lm.generate import BatchGenerator

    bos = tokenizer.bos_token
    toks = [tokenizer.encode(t, add_special_tokens=bos is None or not t.startswith(bos)) for t in texts]
    gen = BatchGenerator(model, max_tokens=max_tokens, sampler=sampler,
                         stop_tokens=[[t] for t in tokenizer.eos_token_ids], completion_batch_size=max(32, len(toks)))
    t0 = time.perf_counter()
    uids = gen.insert(toks, [max_tokens] * len(toks))
    first, last, count, out = {}, {}, dict.fromkeys(uids, 0), {u: [] for u in uids}
    while responses := gen.next_generated():
        now = time.perf_counter()
        for r in responses:
            first.setdefault(r.uid, now)
            count[r.uid] += 1
            if r.finish_reason != "stop":
                out[r.uid].append(r.token)
            if r.finish_reason is not None:
                last[r.uid] = now
    gen.close()
    results = []
    for u, tk in zip(uids, toks):
        ttft, decode_s = first[u] - t0, last[u] - first[u]
        results.append({
            "tokens_in": len(tk), "tokens_out": count[u], "ttft_s": ttft,
            "prefill_tps": len(tk) / ttft,  # per user, includes waiting for a prefill batch
            "decode_tps": (count[u] - 1) / decode_s if decode_s > 0 else 0.0,
            "total_s": last[u] - t0, "text": tokenizer.decode(out[u]),
        })
    return results


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
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                tokenize=False, enable_thinking=cfg["reasoning"],
            ) for p in req.get("prompts") or [req["prompt"]]]
            sampler = make_sampler(temp=req["temperature"])
            if "prompts" in req:
                send({"results": run_batch(model, tokenizer, texts, req["max_tokens"], sampler)})
                mx.clear_cache()
                continue
            prompt = texts[0]
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
