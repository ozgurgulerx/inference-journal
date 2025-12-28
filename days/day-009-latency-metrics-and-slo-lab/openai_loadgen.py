#!/usr/bin/env python3
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai_stream_probe import stream_chat_completion


def build_prompts(mix: str) -> list[str]:
    short = [
        "Answer in one sentence: what is TTFT?",
        "Give a JSON object with keys: ttft, tpot, e2e (strings).",
        "Explain p99 in 2 bullets.",
    ]
    medium = [
        "Summarize the difference between prefill and decode in 5 bullets.",
        "Explain why mean latency can be misleading under load; keep it tight.",
        "Give a short checklist to report latency with load context.",
    ]
    long_blob = " ".join(["Context: user chat history."] * 400)
    long = [
        f"{long_blob}\n\nTask: produce a 6-bullet response, no preamble.",
        f"{long_blob}\n\nTask: output strict JSON with keys: risk, mitigation (strings).",
    ]

    if mix == "short":
        return short
    if mix == "long":
        return long
    if mix == "mixed":
        return short + medium + long
    raise ValueError(f"Unknown mix: {mix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny loadgen for OpenAI-compatible streaming chat completions.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--mix", choices=["short", "mixed", "long"], default="mixed")
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    prompts = build_prompts(args.mix)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "max_tokens": args.max_tokens,
        "mix": args.mix,
        "seed": args.seed,
        "started_at_unix_s": time.time(),
    }

    def one(i: int) -> dict:
        prompt = rng.choice(prompts)
        rec = stream_chat_completion(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
        )
        rec["request_index"] = i
        rec["mix"] = args.mix
        return rec

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"run": run_meta}) + "\n")
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(one, i) for i in range(args.requests)]
            for fut in as_completed(futures):
                rec = fut.result()
                f.write(json.dumps(rec) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

