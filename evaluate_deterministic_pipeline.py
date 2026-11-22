"""
evaluate_deterministic_pipeline.py
----------------------------------
Phase 2, Objective 13: Independent Evaluation of the Deterministic,
Solver-Only Mathematical Problem-Solving Pipeline on GSM8K (300 questions).

Evaluates:
- Coverage rate across the GSM8K test set
- Solved / Unsupported / Invalid classification counts
- Overall accuracy and category-specific accuracy
- Detailed CSV logging to results/deterministic_solver_300_results.csv
- Concise error analysis of incorrect, unsupported, and invalid cases

100% offline, deterministic, and reproducible. Zero LLM or external API calls.
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deterministic_solver_pipeline import solve_problem, PipelineResult
from evaluate import extract_ground_truth, compare_answers


DATASET_PATH = os.path.join("data", "gsm8k_test_300.csv")
OUTPUT_PATH = os.path.join("results", "deterministic_solver_300_results.csv")

SUPPORTED_CATEGORIES = [
    "repeated_rate",
    "weighted_sum",
    "unknown_quantity",
    "additive_partition",
    "percentage_discount",
]


def run_evaluation() -> Dict[str, Any]:
    """Runs the deterministic solver pipeline on all 300 GSM8K questions."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    total_questions = len(df)
    print(f"Loaded {total_questions} questions from {DATASET_PATH}.\n")

    results_records: List[Dict[str, Any]] = []

    # Category tracking
    category_stats = {
        cat: {"matched": 0, "correct": 0, "incorrect": 0}
        for cat in SUPPORTED_CATEGORIES
    }

    # Pipeline outcome tracking
    solved_count = 0
    unsupported_count = 0
    invalid_count = 0
    correct_count = 0
    incorrect_count = 0

    incorrect_cases: List[Dict[str, Any]] = []
    unsupported_ids: List[int] = []
    invalid_cases: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        question_id = idx + 1
        question_text = str(row["question"])
        raw_answer = str(row["answer"])

        reference_answer = extract_ground_truth(raw_answer)

        # Run pipeline
        pipe_res: PipelineResult = solve_problem(question_text)

        status = pipe_res.status
        detected_category = pipe_res.detected_category
        constructed_equation = pipe_res.constructed_equation or ""
        predicted_answer = pipe_res.calculated_answer_str or ""
        explanation = pipe_res.explanation

        is_correct = False

        if status == "solved":
            solved_count += 1
            if compare_answers(predicted_answer, reference_answer):
                is_correct = True
                correct_count += 1
                if detected_category in category_stats:
                    category_stats[detected_category]["matched"] += 1
                    category_stats[detected_category]["correct"] += 1
            else:
                incorrect_count += 1
                if detected_category in category_stats:
                    category_stats[detected_category]["matched"] += 1
                    category_stats[detected_category]["incorrect"] += 1
                incorrect_cases.append({
                    "question_id": question_id,
                    "category": detected_category,
                    "equation": constructed_equation,
                    "predicted": predicted_answer,
                    "reference": reference_answer,
                    "explanation": explanation,
                })
        elif status == "unsupported":
            unsupported_count += 1
            unsupported_ids.append(question_id)
        else:  # 'invalid' or 'error'
            invalid_count += 1
            invalid_cases.append({
                "question_id": question_id,
                "category": detected_category,
                "status": status,
                "explanation": explanation,
            })

        results_records.append({
            "question_id": question_id,
            "question": question_text,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "detected_category": detected_category,
            "constructed_equation": constructed_equation,
            "pipeline_status": status,
            "is_correct": is_correct,
            "explanation": explanation,
        })

    # Save results to CSV safely
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df = pd.DataFrame(results_records)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved detailed results to {OUTPUT_PATH}")

    # Compute metrics
    coverage_rate = (solved_count / total_questions * 100) if total_questions > 0 else 0.0
    accuracy_on_supported = (correct_count / solved_count * 100) if solved_count > 0 else 0.0
    overall_accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0.0

    # Print Report
    print("\n" + "=" * 70)
    print("DETERMINISTIC PIPELINE EVALUATION REPORT (GSM8K 300)")
    print("=" * 70)
    print(f"Total Questions:               {total_questions}")
    print(f"Successfully Solved Questions: {solved_count}")
    print(f"Unsupported Questions:         {unsupported_count}")
    print(f"Invalid / Error Questions:     {invalid_count}")
    print(f"Correct Predictions:           {correct_count}")
    print(f"Incorrect Predictions:         {incorrect_count}")
    print("-" * 70)
    print(f"Coverage Rate:                 {coverage_rate:.2f}% ({solved_count}/{total_questions})")
    print(f"Accuracy on Solved Questions:  {accuracy_on_supported:.2f}% ({correct_count}/{solved_count})")
    print(f"Overall Accuracy:              {overall_accuracy:.2f}% ({correct_count}/{total_questions})")
    print("=" * 70)

    print("\nCATEGORY-LEVEL METRICS:")
    print("-" * 70)
    print(f"{'Category':<24} | {'Matched':<8} | {'Correct':<8} | {'Incorrect':<10} | {'Accuracy':<8}")
    print("-" * 70)
    for cat in SUPPORTED_CATEGORIES:
        stats = category_stats[cat]
        m = stats["matched"]
        c = stats["correct"]
        inc = stats["incorrect"]
        acc = (c / m * 100) if m > 0 else 0.0
        print(f"{cat:<24} | {m:<8} | {c:<8} | {inc:<10} | {acc:.1f}%")
    print("-" * 70)

    # Error Analysis
    print("\nERROR ANALYSIS:")
    print("-" * 70)
    print(f"1. Incorrect Predictions ({len(incorrect_cases)} questions):")
    if incorrect_cases:
        for err in incorrect_cases:
            print(f"   - QID {err['question_id']} [{err['category']}]: Pred={err['predicted']}, Ref={err['reference']}")
            print(f"     Eq: {err['equation']}")
            print(f"     Note: {err['explanation']}")
    else:
        print("   None! All solved questions were 100% correct.")

    print(f"\n2. Invalid / Error Cases ({len(invalid_cases)} questions):")
    if invalid_cases:
        for inv in invalid_cases:
            print(f"   - QID {inv['question_id']} [{inv['category']}]: status={inv['status']}")
            print(f"     Note: {inv['explanation']}")
    else:
        print("   None.")

    print(f"\n3. Unsupported Questions ({len(unsupported_ids)} questions):")
    # Show first 20 and count
    preview = unsupported_ids[:20]
    print(f"   Count: {len(unsupported_ids)} / {total_questions} ({len(unsupported_ids)/total_questions*100:.1f}%)")
    print(f"   Sample QIDs (first 20): {preview} ...")

    return {
        "total_questions": total_questions,
        "solved_count": solved_count,
        "unsupported_count": unsupported_count,
        "invalid_count": invalid_count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "coverage_rate": coverage_rate,
        "accuracy_on_supported": accuracy_on_supported,
        "overall_accuracy": overall_accuracy,
        "category_stats": category_stats,
        "incorrect_cases": incorrect_cases,
        "unsupported_ids": unsupported_ids,
        "invalid_cases": invalid_cases,
    }


if __name__ == "__main__":
    run_evaluation()
