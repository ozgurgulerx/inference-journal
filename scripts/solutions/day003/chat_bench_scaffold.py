#!/usr/bin/env python3
"""
Scaffold: Async chat benchmark (vLLM/OpenAI-compatible)
Goal: measure p50/p95 TTFT, p50/p95 E2E, throughput_tok_s
Acceptance:
- Runs N requests at C concurrency
- Prints JSON with fields: p50_ttft_ms, p95_e2e_ms, throughput_tok_s
"""

import argparse
import asyncio
import json
import time
from typing import Tuple

import aiohttp

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

async def run_single(session, url: str, prompt: str, max_tokens: int) -> Tuple[float, float, int]:
    # TODO: implement request; return (ttft_ms, e2e_ms, output_tokens)
    t0 = time.perf_counter()
    async with session.post(url, json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": False}) as resp:
        _ = await resp.json()
    t1 = time.perf_counter()  # first byte (approx)
    t2 = time.perf_counter()
    return (t1 - t0) * 1000.0, (t2 - t0) * 1000.0, 0  # TODO: parse tokens

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--n-requests", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async def worker():
        async with sem:
            async with aiohttp.ClientSession() as session:
                results.append(await run_single(session, args.url, "Say hello.", args.max_tokens))

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(args.n_requests)])
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in results]
    e2es = [r[1] for r in results]
    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(int(len(xs)*p), max(len(xs)-1, 0))] if xs else 0

    out = {
        "n_requests": args.n_requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_clock_s": round(wall, 2),
        "p50_ttft_ms": round(pct(ttfts, 0.50), 2),
        "p95_e2e_ms": round(pct(e2es, 0.95), 2),
        "throughput_tok_s": 0.0,  # TODO: compute from tokens/second
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
