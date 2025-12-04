#!/usr/bin/env python3
"""
Solution: Async chat benchmark for vLLM. Measures TTFT, E2E, throughput.
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

async def run_single_request(session, url, prompt, max_tokens):
    t0 = time.perf_counter()
    async with session.post(url, json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": False}) as resp:
        t1 = time.perf_counter()
        data = await resp.json()
    t2 = time.perf_counter()

    text = ""
    if "choices" in data and data["choices"]:
        text = data["choices"][0].get("text", "") or data["choices"][0].get("message", {}).get("content", "")

    ttft_ms = (t1 - t0) * 1000
    e2e_ms = (t2 - t0) * 1000
    out_tokens = max(0, len(text.split()))
    return ttft_ms, e2e_ms, out_tokens

async def run_bench(url, prompt, n_requests, concurrency, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def worker():
        async with sem:
            async with aiohttp.ClientSession() as session:
                results.append(await run_single_request(session, url, prompt, max_tokens))

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(n_requests)])
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in results]
    e2es = [r[1] for r in results]
    toks = [r[2] for r in results]

    def pct(vals, p):
        if not vals: return 0.0
        s = sorted(vals)
        return s[min(int(len(s)*p), len(s)-1)]

    total_tokens = sum(toks)
    return {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "wall_clock_s": round(wall, 2),
        "p50_ttft_ms": round(statistics.median(ttfts), 2) if ttfts else 0,
        "p95_ttft_ms": round(pct(ttfts, 0.95), 2),
        "p50_e2e_ms": round(statistics.median(e2es), 2) if e2es else 0,
        "p95_e2e_ms": round(pct(e2es, 0.95), 2),
        "throughput_tok_s": round(total_tokens / wall, 2) if wall > 0 else 0,
        "total_tokens": total_tokens,
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--n-requests", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    prompt = "Explain the trade-offs between max_model_len and max_num_seqs in 3 sentences."
    out = asyncio.run(run_bench(args.url, prompt, args.n_requests, args.concurrency, args.max_tokens))
    print(json.dumps(out, indent=2))
