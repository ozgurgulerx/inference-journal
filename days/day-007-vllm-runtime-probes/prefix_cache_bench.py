#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import requests


def _p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, int(0.95 * (len(xs) - 1))))
    return xs[idx]


def _load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" in obj and isinstance(obj["prompt"], str):
                prompts.append(obj["prompt"])
    return prompts


def _call_once(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.time()

    if not stream:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        t1 = time.time()
        resp.raise_for_status()
        data = resp.json()
        return {
            "ttft_s": None,
            "e2e_s": t1 - t0,
            "total_tokens": data.get("usage", {}).get("total_tokens"),
        }

    payload["stream"] = True
    first_token_t: Optional[float] = None

    with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines():
            if not raw_line:
                continue

            line = raw_line.strip()
            if line.startswith(b"data:"):
                line = line[len(b"data:") :].strip()

            if line == b"[DONE]":
                break

            try:
                _ = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            if first_token_t is None:
                first_token_t = time.time()

    t_end = time.time()
    return {
        "ttft_s": (first_token_t - t0) if first_token_t is not None else None,
        "e2e_s": t_end - t0,
        "total_tokens": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--prompts", default="prefix_prompts.jsonl")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="prefix_cache_bench_results.jsonl")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    prompts = _load_prompts(Path(args.prompts))
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]

    results: list[dict[str, Any]] = []

    start = time.time()
    if args.concurrency <= 1:
        for i, p in enumerate(prompts):
            r = _call_once(
                args.url,
                args.model,
                p,
                args.max_tokens,
                args.temperature,
                args.timeout_s,
                args.stream,
            )
            r["i"] = i
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {
                ex.submit(
                    _call_once,
                    args.url,
                    args.model,
                    p,
                    args.max_tokens,
                    args.temperature,
                    args.timeout_s,
                    args.stream,
                ): i
                for i, p in enumerate(prompts)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                r = fut.result()
                r["i"] = i
                results.append(r)

    wall_s = time.time() - start

    ttfts = [float(r["ttft_s"]) for r in results if isinstance(r.get("ttft_s"), (int, float))]
    e2es = [float(r["e2e_s"]) for r in results if isinstance(r.get("e2e_s"), (int, float))]
    tokens = [int(r["total_tokens"]) for r in results if isinstance(r.get("total_tokens"), int)]

    summary = {
        "n": len(results),
        "concurrency": args.concurrency,
        "stream": args.stream,
        "wall_s": wall_s,
        "mean_ttft_s": statistics.mean(ttfts) if ttfts else None,
        "p95_ttft_s": _p95(ttfts),
        "mean_e2e_s": statistics.mean(e2es) if e2es else None,
        "p95_e2e_s": _p95(e2es),
        "tok_s": (sum(tokens) / wall_s) if tokens and wall_s > 0 else None,
    }

    with open(args.out, "w") as f:
        for r in sorted(results, key=lambda x: x["i"]):
            f.write(json.dumps(r) + "\n")

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
