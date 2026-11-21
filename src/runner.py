import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from prompts import PROMPT_TIERS

load_dotenv()


def extract_answer(raw_text: str, tier_name: str) -> tuple[bool, str]:
    """Checks strict JSON schema compliance and extracts the final answer."""
    cleaned = raw_text.strip()
    valid_json = False
    extracted = ""

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "final_answer" in parsed:
            valid_json = True
            extracted = str(parsed["final_answer"]).strip()
    except json.JSONDecodeError:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if numbers:
            extracted = numbers[-1]

    return valid_json, extracted


def check_accuracy(extracted: str, expected: str, tolerance: float) -> bool:
    """Evaluates objective correctness within numerical tolerance."""
    if not extracted:
        return False
    try:
        return abs(float(extracted) - float(expected)) <= tolerance
    except ValueError:
        return extracted.strip() == expected.strip()


def call_with_retry(client: OpenAI, model: str, sys_prompt: str, usr_prompt: str, max_retries: int = 5):
    """Calls the LLM API with automatic backoff on transient RPM/TPM 429 limits."""
    wait_sec = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt},
                ],
                temperature=0.2,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return response, latency_ms
        except RateLimitError as e:
            if "tokens per day" in str(e).lower() or attempt == max_retries:
                raise
            print(f"    [Rate limit 429] Pausing {wait_sec:.1f}s before retry {attempt}/{max_retries}...")
            time.sleep(wait_sec)
            wait_sec = min(wait_sec * 1.8, 30.0)


def run_benchmark(model: str, base_url: str, api_key: str, n_runs: int, pilot: bool):
    client = OpenAI(base_url=base_url, api_key=api_key)

    with open("data/raw/dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if pilot:
        dataset = [dataset[0], dataset[20], dataset[40]]
        print("Running in PILOT mode (3 tasks across domains)...")

    safe_model_name = model.replace("/", "_")
    tag = "pilot" if pilot else "full"
    out_file = Path(f"data/results/{safe_model_name}_{tag}.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoint if resuming an interrupted run
    completed_keys = set()
    if out_file.exists():
        existing_df = pd.read_csv(out_file)
        completed_keys = set(
            zip(existing_df["tier"], existing_df["task_id"], existing_df["run"])
        )
        print(f"Resuming from checkpoint: {len(completed_keys)} trials already saved in {out_file}")

    total_calls = len(PROMPT_TIERS) * len(dataset) * n_runs
    call_count = 0

    for tier_name, builder in PROMPT_TIERS.items():
        for item in dataset:
            for run_idx in range(1, n_runs + 1):
                call_count += 1
                if (tier_name, item["id"], run_idx) in completed_keys:
                    continue

                sys_prompt, usr_prompt = builder(item["task"])
                response, latency_ms = call_with_retry(client, model, sys_prompt, usr_prompt)
                raw_text = response.choices[0].message.content or ""
                usage = response.usage

                valid_json, extracted = extract_answer(raw_text, tier_name)
                is_correct = check_accuracy(extracted, item["expected_answer"], item["tolerance"])

                row = pd.DataFrame([{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "tier": tier_name,
                    "domain": item["domain"],
                    "task_id": item["id"],
                    "run": run_idx,
                    "valid_json": valid_json,
                    "is_correct": is_correct,
                    "extracted_answer": extracted,
                    "expected_answer": item["expected_answer"],
                    "latency_ms": round(latency_ms, 1),
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                }])

                # Append immediately to CSV so zero progress is ever lost
                row.to_csv(out_file, mode="a", header=not out_file.exists(), index=False)

                print(
                    f"[{call_count}/{total_calls}] {tier_name} | {item['id']} (run {run_idx}) "
                    f"-> JSON:{valid_json} | Correct:{is_correct} ({extracted} vs {item['expected_answer']})"
                )
                time.sleep(0.8)

    df = pd.read_csv(out_file)
    print(f"\nCompleted all {len(df)} trials in {out_file}\n")
    summary = df.groupby("tier").agg(
        json_compliance_pct=("valid_json", lambda x: round(x.mean() * 100, 1)),
        accuracy_pct=("is_correct", lambda x: round(x.mean() * 100, 1)),
        avg_latency_ms=("latency_ms", lambda x: round(x.mean(), 1)),
        avg_tokens=("completion_tokens", lambda x: round(x.mean(), 1)),
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Prompt Determinism Benchmark")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Model identifier")
    parser.add_argument("--base-url", default="https://api.groq.com/openai/v1", help="OpenAI-compatible endpoint")
    parser.add_argument("--runs", type=int, default=3, help="Repeated runs per task to measure determinism")
    parser.add_argument("--pilot", action="store_true", help="Run on a 3-task subset first")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "ollama")
    run_benchmark(args.model, args.base_url, api_key, args.runs, args.pilot)