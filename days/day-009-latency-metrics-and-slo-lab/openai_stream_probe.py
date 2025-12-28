#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.request


def _percent_ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def stream_chat_completion(*, base_url: str, api_key: str, model: str, prompt: str, max_tokens: int) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

    start = time.perf_counter()
    first_token_time = None
    token_times = []
    output_chars = 0
    usage = None

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:") :].strip()
                if not data:
                    continue
                if data == b"[DONE]":
                    break

                event = json.loads(data.decode("utf-8"))
                if usage is None and isinstance(event, dict) and event.get("usage"):
                    usage = event["usage"]

                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if not delta:
                    continue

                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                output_chars += len(delta)

    except Exception as exc:  # noqa: BLE001 - script-level diagnostic
        end = time.perf_counter()
        return {
            "ok": False,
            "error": repr(exc),
            "base_url": base_url,
            "model": model,
            "prompt_chars": len(prompt),
            "max_tokens": max_tokens,
            "ttft_ms": None if first_token_time is None else _percent_ms(first_token_time - start),
            "e2e_ms": _percent_ms(end - start),
            "tpot_mean_ms": None,
            "completion_events": len(token_times),
            "completion_chars": output_chars,
            "usage": usage,
        }

    end = time.perf_counter()
    if first_token_time is None:
        ttft_ms = None
        tpot_mean_ms = None
    else:
        ttft_ms = _percent_ms(first_token_time - start)
        if len(token_times) >= 2:
            deltas = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
            tpot_mean_ms = _percent_ms(sum(deltas) / len(deltas))
        else:
            tpot_mean_ms = None

    return {
        "ok": True,
        "base_url": base_url,
        "model": model,
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "ttft_ms": ttft_ms,
        "e2e_ms": _percent_ms(end - start),
        "tpot_mean_ms": tpot_mean_ms,
        "completion_events": len(token_times),
        "completion_chars": output_chars,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure TTFT/TPOT/E2E against an OpenAI-compatible streaming endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Write 3 bullet points explaining what TTFT is.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    record = stream_chat_completion(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    json.dump(record, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if record.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

