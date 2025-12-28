#!/usr/bin/env python3
import argparse
import math
import random
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


def poisson_next_arrival(rng: random.Random, qps: float) -> float:
    if qps <= 0:
        return math.inf
    return rng.expovariate(qps)


def simulate(*, service_s: float, mean_qps: float, burst_qps: float, burst_every_s: float, duration_s: float, seed: int) -> dict:
    rng = random.Random(seed)
    t = 0.0
    next_arrival = 0.0
    server_free = 0.0

    latencies = []
    queue_delays = []

    while True:
        if next_arrival > duration_s:
            break

        t = next_arrival

        in_burst = burst_every_s > 0 and (t % burst_every_s) < (burst_every_s / 5.0)
        qps = burst_qps if in_burst else mean_qps
        next_arrival = t + poisson_next_arrival(rng, qps)

        start_service = max(t, server_free)
        queue_delay = start_service - t
        finish = start_service + service_s
        server_free = finish

        latency = finish - t
        latencies.append(latency * 1000.0)
        queue_delays.append(queue_delay * 1000.0)

    lat_sorted = sorted(latencies)
    q_sorted = sorted(queue_delays)
    return {
        "n": len(latencies),
        "service_ms": service_s * 1000.0,
        "mean_qps": mean_qps,
        "burst_qps": burst_qps,
        "burst_every_s": burst_every_s,
        "duration_s": duration_s,
        "lat_ms": {
            "mean": mean(lat_sorted) if lat_sorted else None,
            "p50": percentile(lat_sorted, 50) if lat_sorted else None,
            "p95": percentile(lat_sorted, 95) if lat_sorted else None,
            "p99": percentile(lat_sorted, 99) if lat_sorted else None,
        },
        "queue_ms": {
            "mean": mean(q_sorted) if q_sorted else None,
            "p95": percentile(q_sorted, 95) if q_sorted else None,
            "p99": percentile(q_sorted, 99) if q_sorted else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Toy single-server queue simulator to build p50/p95/p99 intuition.")
    parser.add_argument("--service-ms", type=float, required=True)
    parser.add_argument("--mean-qps", type=float, required=True)
    parser.add_argument("--burst-qps", type=float, required=True)
    parser.add_argument("--burst-every-s", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = simulate(
        service_s=args.service_ms / 1000.0,
        mean_qps=args.mean_qps,
        burst_qps=args.burst_qps,
        burst_every_s=args.burst_every_s,
        duration_s=args.duration_s,
        seed=args.seed,
    )

    print("# Queue Simulation (toy)")
    print("")
    print(f"- n={result['n']} service_ms={result['service_ms']:.1f}")
    print(f"- mean_qps={result['mean_qps']} burst_qps={result['burst_qps']} burst_every_s={result['burst_every_s']}")
    print("")
    print("Latency (ms):")
    for k, v in result["lat_ms"].items():
        print(f"- {k}: {v:.1f}")
    print("")
    print("Queue delay (ms):")
    for k, v in result["queue_ms"].items():
        print(f"- {k}: {v:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

