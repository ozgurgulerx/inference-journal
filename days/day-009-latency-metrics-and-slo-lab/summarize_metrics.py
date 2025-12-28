#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("empty")
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def collect(jsonl_path: Path) -> tuple[dict, list[dict]]:
    run = {}
    rows = []
    for i, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines()):
        obj = json.loads(line)
        if i == 0 and "run" in obj:
            run = obj["run"]
            continue
        rows.append(obj)
    return run, rows


def fmt_ms(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize JSONL metrics into a compact markdown table.")
    parser.add_argument("jsonl")
    parser.add_argument("--out", default="results.md")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    run, rows = collect(Path(args.jsonl))
    ok_rows = [r for r in rows if r.get("ok")]
    err_rows = [r for r in rows if not r.get("ok")]

    def series(key: str) -> list[float]:
        vals = [r.get(key) for r in ok_rows]
        return sorted([v for v in vals if isinstance(v, (int, float))])

    metrics = {
        "TTFT (ms)": series("ttft_ms"),
        "TPOT mean (ms)": series("tpot_mean_ms"),
        "E2E (ms)": series("e2e_ms"),
    }

    lines = []
    title = args.title or f"Latency summary: {run.get('model', '')}".strip()
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Load Context")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(run, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Percentiles")
    lines.append("")
    lines.append("| Metric | n | mean | p50 | p95 | p99 | error_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    total = len(rows)
    error_rate = (len(err_rows) / total) if total else 0.0

    for name, vals in metrics.items():
        if not vals:
            lines.append(f"| {name} | 0 | — | — | — | — | {error_rate:.2%} |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(len(vals)),
                    fmt_ms(mean(vals)),
                    fmt_ms(percentile(vals, 50)),
                    fmt_ms(percentile(vals, 95)),
                    fmt_ms(percentile(vals, 99)),
                    f"{error_rate:.2%}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Interpret p99 only with this load context; percentiles without load are meaningless.")
    lines.append("- If TTFT p99 jumps with concurrency while TPOT stays flat, you’re likely queueing/admission limited.")
    lines.append("- If TPOT degrades with prompt mix/seq length, you’re likely decode/KV-cache/bandwidth limited.")
    lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

