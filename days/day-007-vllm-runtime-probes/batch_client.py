#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests


def _p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, int(0.95 * (len(xs) - 1))))
    return xs[idx]


def _call_once(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[float, int]:
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout_s)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    tokens = int(data.get("usage", {}).get("total_tokens") or 0)
    return (t1 - t0), tokens


def _run(url: str, payload: dict[str, Any], n: int, concurrency: int, timeout_s: float) -> dict[str, Any]:
    lats: list[float] = []
    toks: list[int] = []

    t0 = time.time()
    if concurrency <= 1:
        for _ in range(n):
            lat_s, t = _call_once(url, payload, timeout_s)
            lats.append(lat_s)
            toks.append(t)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_call_once, url, payload, timeout_s) for _ in range(n)]
            for fut in as_completed(futs):
                lat_s, t = fut.result()
                lats.append(lat_s)
                toks.append(t)
    t1 = time.time()

    wall_s = t1 - t0
    return {
        "n": n,
        "concurrency": concurrency,
        "wall_s": wall_s,
        "mean_e2e_s": statistics.mean(lats) if lats else None,
        "p95_e2e_s": _p95(lats),
        "qps": (n / wall_s) if wall_s > 0 else None,
        "tok_s": (sum(toks) / wall_s) if wall_s > 0 and toks else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--prompt", default="Explain memory fragmentation in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-list", default="1,2,4,8,16,32")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    if not args.sweep:
        seq = _run(args.url, payload, args.n, 1, args.timeout_s)
        conc = _run(args.url, payload, args.n, args.concurrency, args.timeout_s)
        result = {"sequential": seq, "concurrent": conc}
        print(json.dumps(result, indent=2))
        return

    sweep_conc = [int(x) for x in args.sweep_list.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for c in sweep_conc:
        rows.append(_run(args.url, payload, args.n, c, args.timeout_s))

    if args.out:
        with open(args.out, "w") as f:
            f.write("concurrency,mean_e2e_s,p95_e2e_s,qps,tok_s\n")
            for r in rows:
                f.write(
                    f"{r['concurrency']},{r['mean_e2e_s']},{r['p95_e2e_s']},{r['qps']},{r['tok_s']}\n"
                )

    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
