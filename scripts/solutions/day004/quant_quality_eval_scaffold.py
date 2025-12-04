#!/usr/bin/env python3
"""
Day 004 – Quality Sanity Check: BF16 vs Quantized Model

Scaffold: Fill in the TODOs to complete the quality comparison.

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

# TODO 1: Add 10-15 diverse prompts to test quality
# Include:
# - Simple factual Q&A
# - Explanation/teaching prompts
# - Basic reasoning/math
# - Short summarization
# - Code snippets
PROMPTS = [
    # Example:
    # "What is the capital of France?",
    # "Explain how a hash table works in 2 sentences.",
    # TODO: Add your prompts here
]


def call_completion(url: str, prompt: str, model: str = "", max_tokens: int = 100) -> str:
    """
    Call the vLLM completion endpoint and return the response text.
    
    TODO 2: Implement this function
    - POST to the url with appropriate payload
    - Handle errors gracefully
    - Return the generated text
    """
    # Hint: Use requests.post with json payload
    # The vLLM /v1/completions endpoint expects:
    # {"prompt": "...", "max_tokens": N, "temperature": 0}
    pass


def run_comparison(bf16_url: str, quant_url: str, prompts: list) -> list:
    """
    Run all prompts through both models and collect results.
    
    TODO 3: Implement this function
    - Iterate through prompts
    - Call both endpoints for each
    - Store results in a list of dicts
    """
    results = []
    # Hint: For each prompt, call both endpoints and store:
    # {"prompt": prompt, "bf16_output": ..., "quant_output": ...}
    return results


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
    
    # TODO 4: Run comparison and save results
    # results = run_comparison(args.bf16_url, args.quant_url, PROMPTS)
    
    # TODO 5: Save results to JSON
    # output_path = Path(args.output).expanduser()
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(output_path, 'w') as f:
    #     json.dump(results, f, indent=2)
    
    print(f"[✓] Results saved to {args.output}")
    print("[*] Now manually review the outputs for quality differences.")


if __name__ == "__main__":
    main()
