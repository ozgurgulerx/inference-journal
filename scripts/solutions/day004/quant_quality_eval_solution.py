#!/usr/bin/env python3
"""
Day 004 – Quality Sanity Check: BF16 vs Quantized Model

Solution: Complete implementation of quality comparison script.

Usage:
    python day004_quant_quality_eval.py \
        --bf16-url http://127.0.0.1:8000/v1/completions \
        --quant-url http://127.0.0.1:8001/v1/completions \
        --output ~/benchmarks/day004_quant_quality_eval.json
"""
import argparse
import json
import requests
from pathlib import Path
from typing import Optional

# Diverse prompts to test quality across different task types
PROMPTS = [
    # Factual Q&A
    "What is the capital of France?",
    "Who wrote the play 'Hamlet'?",
    
    # Explanation / Teaching
    "Explain how a transformer attention mechanism works in 3 sentences.",
    "What is gradient descent in machine learning? Keep it under 50 words.",
    
    # Basic Reasoning / Math
    "If I have 3 apples and give away 2, how many do I have left?",
    "What comes next in this sequence: 2, 4, 8, 16, ?",
    
    # Summarization
    "Summarize in one sentence: The Industrial Revolution was a period of major industrialization and innovation during the late 1700s and early 1800s. It began in Great Britain and quickly spread throughout the world.",
    
    # Code
    "Write a Python function to reverse a string.",
    "Write a one-liner in Python to check if a number is even.",
    
    # Edge cases
    "What is 0 divided by 0?",
    "Complete this sentence: The quick brown fox",
    
    # Slightly harder reasoning
    "Alice is taller than Bob. Bob is taller than Charlie. Who is the shortest?",
    
    # Format following
    "List the first 5 prime numbers, separated by commas.",
]


def call_completion(
    url: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    timeout: int = 30,
) -> Optional[str]:
    """
    Call the vLLM completion endpoint and return the response text.
    """
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract text from vLLM response format
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "").strip()
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"    [!] Request error: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    [!] Parse error: {e}")
        return None


def run_comparison(
    bf16_url: str,
    quant_url: str,
    prompts: list,
    max_tokens: int = 100,
) -> list:
    """
    Run all prompts through both models and collect results.
    """
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Prompt: {prompt[:50]}...")
        
        bf16_output = call_completion(bf16_url, prompt, max_tokens=max_tokens)
        quant_output = call_completion(quant_url, prompt, max_tokens=max_tokens)
        
        result = {
            "prompt": prompt,
            "bf16_output": bf16_output,
            "quant_output": quant_output,
            "outputs_match": bf16_output == quant_output if bf16_output and quant_output else None,
        }
        results.append(result)
        
        # Quick visual comparison
        if bf16_output and quant_output:
            match_str = "✅ MATCH" if bf16_output == quant_output else "⚠️ DIFF"
            print(f"    {match_str}")
        else:
            print("    ⚠️ One or both outputs missing")
    
    return results


def print_summary(results: list):
    """Print a quick summary of the comparison."""
    total = len(results)
    matches = sum(1 for r in results if r.get("outputs_match") is True)
    diffs = sum(1 for r in results if r.get("outputs_match") is False)
    errors = sum(1 for r in results if r.get("outputs_match") is None)
    
    print("\n" + "=" * 50)
    print("QUALITY COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Total prompts: {total}")
    print(f"Exact matches: {matches} ({100*matches/total:.1f}%)")
    print(f"Differences: {diffs} ({100*diffs/total:.1f}%)")
    print(f"Errors: {errors} ({100*errors/total:.1f}%)")
    print("=" * 50)
    print("\n[*] Now manually review the JSON to assess quality differences.")
    print("    Look for: nonsense outputs, loss of nuance, repeated patterns, factual errors.")


def main():
    parser = argparse.ArgumentParser(description="Quality comparison: BF16 vs Quant")
    parser.add_argument("--bf16-url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--quant-url", default="http://127.0.0.1:8001/v1/completions")
    parser.add_argument("--output", default="day004_quant_quality_eval.json")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()
    
    if not PROMPTS:
        print("[!] No prompts defined. Add prompts to the PROMPTS list.")
        return
    
    print(f"[*] Running quality comparison with {len(PROMPTS)} prompts...")
    print(f"    BF16: {args.bf16_url}")
    print(f"    Quant: {args.quant_url}")
    print()
    
    results = run_comparison(
        args.bf16_url,
        args.quant_url,
        PROMPTS,
        max_tokens=args.max_tokens,
    )
    
    # Save results
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[✓] Results saved to {output_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
