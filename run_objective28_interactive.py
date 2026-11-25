"""
run_objective28_interactive.py
------------------------------
Phase 4 - Objective 28: Automatic LLM Evaluation Runner for Remaining Cases.

Provides a safe, resilient evaluation runner that automatically processes all
remaining cases in the 200-case Phase 4 evaluation dataset without requiring
per-case confirmation prompts.

Key Features & Safeguards:
1. Automatic Full-Pass Execution: Processes all remaining cases sequentially without
   manual confirmation prompts.
2. Strict Information Barrier: Passes only `problem` and `reasoning_text` to the detector.
   Never exposes gold labels, error categories, steps, or metadata to the LLM.
3. Safe Resumability: Checks existing predictions using `load_existing_predictions()`.
   Only records with `parsed_successfully == True` are counted as completed.
   Previously failed (`parsed_successfully == False`) or missing cases are retried.
4. Immediate Append & Flush: Each prediction is appended to the checkpoint file immediately.
   Note: while flush() pushes buffer data to the OS, it does not guarantee atomic writes;
   load_existing_predictions() safely ignores any truncated/malformed JSON lines if interrupted,
   and checkpoint_prediction() ensures new appends start on a fresh newline.
5. Zero 503 Auto-Retries: Disables automatic 503 retry loops (`max_retries=0`) in this runner,
   preserving immediate failure recording for persistent 503s.
   Note: Gemini 429 rate-limiting continues to use the existing SDK backoff logic in call_llm().
6. Pacing & Rate Safeguards: Configurable inter-request pacing (default 4.0s) to respect API quotas.
7. Graceful Exit on Ctrl+C: Catches KeyboardInterrupt, preserves all flushed records, and outputs
   a comprehensive session summary (successful, failed, skipped, and remaining cases).
"""

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_llm_improved import (
    load_dataset,
    load_existing_predictions,
    checkpoint_prediction,
    DEFAULT_DATASET_PATH,
    DEFAULT_PREDICTIONS_PATH,
)
from llm_improved_detector import detect_reasoning_error_improved, LLMImprovedErrorDetection
from main import get_llm_client


def run_automatic_evaluation(
    dataset_path: str = DEFAULT_DATASET_PATH,
    predictions_path: str = DEFAULT_PREDICTIONS_PATH,
    provider: str = "gemini",
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    delay_seconds: float = 4.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Executes the Objective 28 automatic evaluation workflow for remaining cases.

    Parameters
    ----------
    dataset_path : str
        Path to the 200-case evaluation dataset JSONL.
    predictions_path : str
        Path to the checkpoint file (appended only, never truncated).
    provider : str
        LLM provider, defaults to 'gemini'.
    client : Any, optional
        Pre-initialized LLM client (for dependency injection or testing).
    model_name : str, optional
        Model override (for testing or configuration).
    delay_seconds : float, optional
        Pacing delay in seconds between API requests (default 4.0s).

    Returns
    -------
    dict
        Summary of the evaluation session:
        - successful_predictions (int)
        - failed_predictions (int)
        - skipped_successful_predictions (int)
        - remaining_cases (int)
        - evaluated_this_session (int)
        - completed_total (int)
        - total_cases (int)
    """
    cases = load_dataset(dataset_path)
    completed_predictions = load_existing_predictions(predictions_path)

    # Initialize client if not injected
    if client is None or model_name is None:
        init_client, init_model = get_llm_client(provider)
        active_client = client if client is not None else init_client
        active_model = model_name if model_name is not None else init_model
    else:
        active_client = client
        active_model = model_name

    total_cases = len(cases)
    initially_completed = len(completed_predictions)
    skipped_successful_count = initially_completed
    evaluated_this_session = 0
    successful_count = 0
    failed_count = 0

    print("=" * 70)
    print("  Phase 4 Objective 28: Automatic LLM Evaluation Runner")
    print("=" * 70)
    print(f"Dataset:              {dataset_path}")
    print(f"Predictions File:     {predictions_path}")
    print(f"Provider:             {provider}")
    print(f"Model:                {active_model} (free tier, temperature=0)")
    print(f"Total Dataset Cases:  {total_cases}")
    print(f"Already Completed:    {initially_completed} (parsed_successfully == True)")
    print(f"Remaining to Audit:   {total_cases - initially_completed}")
    print(f"Pacing Delay:         {delay_seconds:.1f}s between requests")
    print("-" * 70)
    print("Safeguards Active:")
    print("  - Automatic Processing: Processing remaining cases sequentially.")
    print("  - Strict Information Barrier: Only 'problem' and 'reasoning_text' enter prompt.")
    print("  - 503 UNAVAILABLE Retries: Disabled (max_retries=0) for automatic control.")
    print("  - 429 RESOURCE_EXHAUSTED: Standard provider SDK backoff preserved in call_llm().")
    print("  - Checkpointing: Immediate append-and-flush; existing records never overwritten.")
    print("=" * 70)
    print()

    try:
        # If all cases already completed, terminate immediately without API calls
        if total_cases > 0 and initially_completed >= total_cases:
            print("\n[+] All cases in dataset have already been successfully completed.")
            print("[+] Zero API requests needed.")
        else:
            for idx, case in enumerate(cases, 1):
                cid = case["id"]

                # Requirement 3: Skip only predictions where parsed_successfully == True
                if cid in completed_predictions:
                    continue

                print("\n" + "-" * 70)
                print(f">>> Case: {cid} (Position: {idx}/{total_cases})")
                print(f"    Status: Processing Evaluation (Automatic)")
                print(f"    Provider: {provider} | Model: {active_model}")
                print(f"[*] Sending 1 API request for {cid} to {provider}/{active_model}...")
                print("-" * 70)

                # Requirement 9: Strict Information Barrier - ONLY problem and reasoning_text
                problem = case["problem"]
                reasoning_text = case["reasoning_text"]

                # Requirement 7: max_retries=0 disables automatic 503 retry loops
                detection = detect_reasoning_error_improved(
                    problem=problem,
                    reasoning_text=reasoning_text,
                    provider=provider,
                    client=active_client,
                    model_name=active_model,
                    max_retries=0,
                )

                # Requirement 5: Checkpoint immediately using existing record schema
                record = {
                    "id": cid,
                    "prediction": {
                        "classification": detection.classification,
                        "first_incorrect_step": detection.first_incorrect_step,
                        "confidence": detection.confidence,
                        "explanation": detection.explanation,
                    },
                    "raw_llm_response": detection.raw_response,
                    "parsed_successfully": detection.parsed_successfully,
                    "provider": detection.provider,
                    "model": detection.model,
                    "latency_seconds": detection.latency_seconds,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                checkpoint_prediction(predictions_path, record)
                evaluated_this_session += 1

                if detection.parsed_successfully:
                    completed_predictions[cid] = record
                    successful_count += 1
                    status_str = "SUCCESS (parsed_successfully = True)"
                else:
                    failed_count += 1
                    status_str = f"FAILED / UNPARSED (parsed_successfully = False: {detection.explanation})"

                print(f"\n[+] Prediction Recorded for {cid}:")
                print(f"    Classification:       {detection.classification}")
                print(f"    First Incorrect Step: {detection.first_incorrect_step}")
                print(f"    Confidence:           {detection.confidence}")
                print(f"    Latency:              {detection.latency_seconds:.2f}s")
                print(f"    Parsing Status:       {status_str}")
                print(f"    Explanation:          {detection.explanation}")
                print(f"    Checkpoint Updated:   {predictions_path} (flushed to disk)")

                # Pacing delay between requests if more cases remain
                if delay_seconds > 0 and idx < total_cases:
                    time.sleep(delay_seconds)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C detected] Automatic evaluation interrupted. Stopping safely.")

    # Session Summary (Requirement 11)
    current_completed = len(load_existing_predictions(predictions_path))
    remaining = total_cases - current_completed

    print("\n" + "=" * 70)
    print("  Objective 28: Automatic Evaluation Session Summary")
    print("=" * 70)
    print(f"Total Dataset Cases:            {total_cases}")
    print(f"Skipped (Already Completed):    {skipped_successful_count}")
    print(f"Evaluated This Session:         {evaluated_this_session}")
    print(f"  - Successful Predictions:     {successful_count}")
    print(f"  - Failed Predictions:         {failed_count}")
    print(f"Total Completed Checkpointed:   {current_completed}/{total_cases} cases")
    print(f"Remaining Cases to Complete:    {remaining}")
    print(f"Predictions File:               {predictions_path}")
    print("All progress has been safely saved to disk.")
    print("=" * 70)

    return {
        "evaluated_this_session": evaluated_this_session,
        "successful_predictions": successful_count,
        "failed_predictions": failed_count,
        "skipped_successful_predictions": skipped_successful_count,
        "completed_total": current_completed,
        "remaining_cases": remaining,
        "remaining_count": remaining,  # backward compatibility alias
        "successful_count": successful_count,
        "failed_count": failed_count,
        "skipped_successful_count": skipped_successful_count,
        "total_cases": total_cases,
        "predictions_path": predictions_path,
    }


# Backwards compatibility alias
run_interactive_evaluation = run_automatic_evaluation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Objective 28 Automatic Evaluation for Remaining Cases.")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "groq"])
    parser.add_argument("--predictions", type=str, default=DEFAULT_PREDICTIONS_PATH, help="Path to predictions file.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to dataset file.")
    parser.add_argument("--delay", type=float, default=4.0, help="Pacing delay in seconds between API requests.")

    args = parser.parse_args()

    run_automatic_evaluation(
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        provider=args.provider,
        delay_seconds=args.delay,
    )
