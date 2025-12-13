#!/usr/bin/env python3
import argparse
import json
import time
from typing import Any, Optional

import requests


def _p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, int(0.95 * (len(xs) - 1))))
    return xs[idx]


def _post_non_stream(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout_s)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    return {
        "ttft_s": None,
        "e2e_s": t1 - t0,
        "total_tokens": data.get("usage", {}).get("total_tokens"),
        "output_preview": (data.get("choices", [{}])[0].get("text") or "").strip()[:120],
    }


def _post_stream(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    t0 = time.time()
    first_token_t: Optional[float] = None
    last_text: str = ""

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
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            if first_token_t is None:
                first_token_t = time.time()

            choice0 = (obj.get("choices") or [{}])[0]
            delta_text = choice0.get("text")
            if isinstance(delta_text, str):
                last_text += delta_text

    t_end = time.time()

    return {
        "ttft_s": (first_token_t - t0) if first_token_t is not None else None,
        "e2e_s": t_end - t0,
        "total_tokens": None,
        "output_preview": last_text.strip()[:120],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--prompt", default="Explain TTFT in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    ttfts: list[float] = []
    e2es: list[float] = []

    for _ in range(args.runs):
        payload: dict[str, Any] = {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }

        if args.stream:
            payload["stream"] = True
            out = _post_stream(args.url, payload, args.timeout_s)
        else:
            out = _post_non_stream(args.url, payload, args.timeout_s)

        if isinstance(out.get("ttft_s"), (int, float)):
            ttfts.append(float(out["ttft_s"]))
        if isinstance(out.get("e2e_s"), (int, float)):
            e2es.append(float(out["e2e_s"]))

        print(json.dumps(out))

    if args.runs > 1:
        summary = {
            "runs": args.runs,
            "ttft_s": {
                "min": min(ttfts) if ttfts else None,
                "p95": _p95(ttfts),
                "max": max(ttfts) if ttfts else None,
            },
            "e2e_s": {
                "min": min(e2es) if e2es else None,
                "p95": _p95(e2es),
                "max": max(e2es) if e2es else None,
            },
        }
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
