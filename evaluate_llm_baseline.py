"""
evaluate_llm_baseline.py
------------------------
Phase 4 - Objective 27: Evaluation Runner for LLM-Only Error Detection Baseline.

Orchestrates batch evaluation on the 200-case Phase 4 evaluation dataset, implements
immediate checkpointing and resumability, keeps model predictions strictly separated
from gold labels, and calculates comprehensive evaluation metrics according to the
strict Objective 27 specification.

Evaluation Metrics Specification:
- Positive class = incorrect reasoning.
- Explicit incorrect on gold incorrect = True Positive (TP).
- Explicit incorrect on gold correct = False Positive (FP).
- Correct on gold incorrect = False Negative (FN_explicit).
- Uncertain on gold incorrect = False Negative for conservative recall (FN_uncertain).
- Uncertain on gold correct = Abstention (NOT a False Positive).
- Strict accuracy: All uncertain predictions count as incorrect.
- Decisive accuracy & coverage: Evaluated strictly over decisive predictions (correct or incorrect).
- Error precision, conservative recall, and F1.
- First-incorrect-step localization accuracy among true positives.
- Cohort-separated reporting: Sound, Unsupported, Conditional Warning, Incorrect (6 categories),
  and the 10 symbolic limitation cases.
"""

from collections import Counter
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_error_detector import detect_reasoning_error_llm, LLMErrorDetection


DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reasoning_eval_dataset.jsonl")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_PREDICTIONS_PATH = os.path.join(DEFAULT_RESULTS_DIR, "llm_baseline_predictions.jsonl")
DEFAULT_METRICS_PATH = os.path.join(DEFAULT_RESULTS_DIR, "llm_baseline_metrics.json")
DEFAULT_REPORT_PATH = os.path.join(DEFAULT_RESULTS_DIR, "llm_baseline_report.md")


def load_dataset(dataset_path: str = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    """Loads all records from the evaluation dataset JSONL file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                cases.append(json.loads(line_str))
    return cases


def load_existing_predictions(predictions_path: str = DEFAULT_PREDICTIONS_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Loads completed predictions from the checkpoint file.
    Only records where parsed_successfully is True are treated as completed.
    Returns a dictionary mapping case ID -> prediction record.
    """
    if not os.path.exists(predictions_path):
        return {}
    completed = {}
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    record = json.loads(line_str)
                    case_id = record.get("id")
                    if case_id and record.get("parsed_successfully") is True:
                        completed[case_id] = record
                except Exception:
                    continue
    return completed


def checkpoint_prediction(predictions_path: str, record: Dict[str, Any]) -> None:
    """Appends a single prediction record immediately to disk and flushes."""
    os.makedirs(os.path.dirname(os.path.abspath(predictions_path)), exist_ok=True)
    with open(predictions_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def compute_metrics_for_subset(
    subset_name: str,
    gold_cases: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes rigorous Objective 27 evaluation metrics for a specific subset of test cases.

    Rules:
    - Positive class = incorrect reasoning.
    - TP: Gold is incorrect AND Model predicts 'incorrect'.
    - FP: Gold is correct AND Model predicts 'incorrect'.
    - FN_explicit: Gold is incorrect AND Model predicts 'correct'.
    - FN_uncertain: Gold is incorrect AND Model predicts 'uncertain'.
    - TN: Gold is correct AND Model predicts 'correct'.
    - Abstention: Gold is correct AND Model predicts 'uncertain' (NOT FP).
    - Strict Accuracy: Count correct if (Gold is correct and Model is 'correct')
      OR (Gold is incorrect and Model is 'incorrect'). Uncertain predictions count as incorrect.
    - Decisive Coverage: Fraction of cases with decisive prediction ('correct' or 'incorrect').
    - Decisive Accuracy: Accuracy evaluated strictly over decisive predictions.
    - Localization Accuracy: Step accuracy among TPs where gold error step is specified.
    """
    total = len(gold_cases)
    if total == 0:
        return {
            "subset_name": subset_name,
            "total_cases": 0,
            "gold_correct_count": 0,
            "gold_incorrect_count": 0,
            "confusion_matrix": {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives_explicit": 0,
                "false_negatives_uncertain": 0,
                "total_false_negatives": 0,
                "abstentions_on_correct": 0,
                "total_uncertain": 0,
            },
            "accuracy_metrics": {
                "strict_accuracy": 0.0,
                "decisive_accuracy": 0.0,
                "decisive_coverage": 0.0,
                "uncertainty_rate": 0.0,
            },
            "error_detection_metrics": {
                "precision": 0.0,
                "conservative_recall": 0.0,
                "f1_score": 0.0,
            },
            "step_localization": {
                "step_match_count": 0,
                "tp_with_gold_step_count": 0,
                "localization_accuracy": 0.0,
            },
        }

    gold_correct_count = 0
    gold_incorrect_count = 0

    tp = 0
    fp = 0
    fn_explicit = 0
    fn_uncertain = 0
    tn = 0
    abstentions_on_correct = 0

    strict_correct_count = 0
    decisive_correct_count = 0
    decisive_total_count = 0

    step_match_count = 0
    tp_with_gold_step_count = 0

    for case in gold_cases:
        cid = case["id"]
        gt = case["ground_truth"]
        is_gold_correct = gt["is_correct"]
        gold_error_step = gt.get("error_step")

        pred_rec = predictions.get(cid, {})
        pred_dict = pred_rec.get("prediction", {})
        model_class = pred_dict.get("classification", "uncertain")
        model_step = pred_dict.get("first_incorrect_step")

        if is_gold_correct:
            gold_correct_count += 1
            if model_class == "correct":
                tn += 1
                strict_correct_count += 1
                decisive_correct_count += 1
                decisive_total_count += 1
            elif model_class == "incorrect":
                fp += 1
                decisive_total_count += 1
            else:  # uncertain
                abstentions_on_correct += 1
        else:
            gold_incorrect_count += 1
            if model_class == "incorrect":
                tp += 1
                strict_correct_count += 1
                decisive_correct_count += 1
                decisive_total_count += 1

                # Step localization evaluation
                if gold_error_step is not None:
                    tp_with_gold_step_count += 1
                    if model_step == gold_error_step:
                        step_match_count += 1
            elif model_class == "correct":
                fn_explicit += 1
                decisive_total_count += 1
            else:  # uncertain
                fn_uncertain += 1

    total_uncertain = abstentions_on_correct + fn_uncertain
    uncertainty_rate = total_uncertain / total if total > 0 else 0.0

    strict_accuracy = strict_correct_count / total if total > 0 else 0.0
    decisive_coverage = decisive_total_count / total if total > 0 else 0.0
    decisive_accuracy = decisive_correct_count / decisive_total_count if decisive_total_count > 0 else 0.0

    # Error detection metrics (Positive class = incorrect)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    conservative_recall = tp / gold_incorrect_count if gold_incorrect_count > 0 else 0.0
    f1 = (2 * precision * conservative_recall) / (precision + conservative_recall) if (precision + conservative_recall) > 0 else 0.0

    step_localization_accuracy = (
        step_match_count / tp_with_gold_step_count if tp_with_gold_step_count > 0 else 0.0
    )

    return {
        "subset_name": subset_name,
        "total_cases": total,
        "gold_correct_count": gold_correct_count,
        "gold_incorrect_count": gold_incorrect_count,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives_explicit": fn_explicit,
            "false_negatives_uncertain": fn_uncertain,
            "total_false_negatives": fn_explicit + fn_uncertain,
            "abstentions_on_correct": abstentions_on_correct,
            "total_uncertain": total_uncertain,
        },
        "accuracy_metrics": {
            "strict_accuracy": round(strict_accuracy, 4),
            "decisive_accuracy": round(decisive_accuracy, 4),
            "decisive_coverage": round(decisive_coverage, 4),
            "uncertainty_rate": round(uncertainty_rate, 4),
        },
        "error_detection_metrics": {
            "precision": round(precision, 4),
            "conservative_recall": round(conservative_recall, 4),
            "f1_score": round(f1, 4),
        },
        "step_localization": {
            "step_match_count": step_match_count,
            "tp_with_gold_step_count": tp_with_gold_step_count,
            "localization_accuracy": round(step_localization_accuracy, 4),
        },
    }


def compute_all_evaluation_metrics(
    gold_cases: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes comprehensive metrics overall, across four cohorts, and for symbolic limitations."""
    overall_metrics = compute_metrics_for_subset("Overall Dataset", gold_cases, predictions)

    # 1. Four explicit cohorts
    sound_cases = [c for c in gold_cases if c["ground_truth"]["is_correct"] and c["ground_truth"]["error_category"] == "none"]
    unsupported_cases = [c for c in gold_cases if c["ground_truth"]["is_correct"] and c["ground_truth"]["error_category"] == "unsupported"]
    warning_cases = [c for c in gold_cases if c["ground_truth"]["is_correct"] and c["ground_truth"]["error_category"] == "conditional_warning"]
    incorrect_cases = [c for c in gold_cases if not c["ground_truth"]["is_correct"]]

    cohort_metrics = {
        "sound_reasoning_none": compute_metrics_for_subset("Cohort 1: Sound Reasoning (none)", sound_cases, predictions),
        "unsupported_conceptual": compute_metrics_for_subset("Cohort 2: Unsupported Conceptual Equations", unsupported_cases, predictions),
        "conditional_warnings": compute_metrics_for_subset("Cohort 3: Conditional Warning Cases", warning_cases, predictions),
        "all_incorrect_cases": compute_metrics_for_subset("Cohort 4: All Injected Errors", incorrect_cases, predictions),
    }

    # Breakdown by 6 incorrect categories
    incorrect_category_metrics = {}
    for cat in [
        "invalid_transformation",
        "arithmetic_error",
        "domain_violation",
        "incorrect_setup",
        "lost_solution",
        "division_by_zero",
    ]:
        cat_cases = [c for c in gold_cases if c["ground_truth"]["error_category"] == cat]
        incorrect_category_metrics[cat] = compute_metrics_for_subset(f"Category: {cat}", cat_cases, predictions)

    # 2. Symbolic Limitation boundary cases (eval_191 to eval_200)
    limitation_cases = [c for c in gold_cases if c["metadata"].get("symbolic_limitation")]
    limitation_metrics = compute_metrics_for_subset("Symbolic Limitation Boundary Cases", limitation_cases, predictions)

    return {
        "overall": overall_metrics,
        "cohorts": cohort_metrics,
        "incorrect_categories": incorrect_category_metrics,
        "symbolic_limitations": limitation_metrics,
    }


def generate_markdown_report(metrics_data: Dict[str, Any], output_path: str = DEFAULT_REPORT_PATH) -> str:
    """Renders a comprehensive human-readable Markdown evaluation report."""
    overall = metrics_data["overall"]
    cm = overall["confusion_matrix"]
    acc = overall["accuracy_metrics"]
    det = overall["error_detection_metrics"]
    step = overall["step_localization"]

    cohorts = metrics_data["cohorts"]
    cats = metrics_data["incorrect_categories"]
    lim = metrics_data["symbolic_limitations"]

    lines = [
        "# Phase 4 – Objective 27: LLM-Only Error Detection Baseline Report",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **Total Cases** | {overall['total_cases']} | Complete Phase 4 evaluation dataset |",
        f"| **Strict Accuracy** | {acc['strict_accuracy'] * 100:.1f}% | Uncertain predictions counted as incorrect |",
        f"| **Decisive Accuracy** | {acc['decisive_accuracy'] * 100:.1f}% | Accuracy evaluated only on decisive calls |",
        f"| **Decisive Coverage** | {acc['decisive_coverage'] * 100:.1f}% | Percentage of cases model made a decisive call |",
        f"| **Uncertainty Rate** | {acc['uncertainty_rate'] * 100:.1f}% | Abstention frequency across dataset |",
        f"| **Error Precision** | {det['precision'] * 100:.1f}% | True positives / total model incorrect flags |",
        f"| **Conservative Recall** | {det['conservative_recall'] * 100:.1f}% | True positives / total gold errors (uncertain = missed) |",
        f"| **Error F1-Score** | {det['f1_score'] * 100:.1f}% | Harmonic mean of precision & conservative recall |",
        f"| **Step Localization Accuracy** | {step['localization_accuracy'] * 100:.1f}% | Correct first error step among True Positives |",
        "",
        "---",
        "",
        "## 2. Confusion Matrix (Positive Class = Incorrect)",
        "",
        "| Matrix Category | Count | Percentage | Definition |",
        "| :--- | :--- | :--- | :--- |",
        f"| **True Positives (TP)** | {cm['true_positives']} | {cm['true_positives']/overall['total_cases']*100:.1f}% | Explicit incorrect on gold incorrect |",
        f"| **False Positives (FP)** | {cm['false_positives']} | {cm['false_positives']/overall['total_cases']*100:.1f}% | Explicit incorrect on gold correct |",
        f"| **True Negatives (TN)** | {cm['true_negatives']} | {cm['true_negatives']/overall['total_cases']*100:.1f}% | Explicit correct on gold correct |",
        f"| **False Negatives (Explicit)** | {cm['false_negatives_explicit']} | {cm['false_negatives_explicit']/overall['total_cases']*100:.1f}% | Explicit correct on gold incorrect |",
        f"| **False Negatives (Uncertain)** | {cm['false_negatives_uncertain']} | {cm['false_negatives_uncertain']/overall['total_cases']*100:.1f}% | Uncertain on gold incorrect (missed error) |",
        f"| **Abstentions on Correct** | {cm['abstentions_on_correct']} | {cm['abstentions_on_correct']/overall['total_cases']*100:.1f}% | Uncertain on gold correct (not counted as FP) |",
        "",
        "---",
        "",
        "## 3. Performance Across Dataset Cohorts",
        "",
        "| Cohort | N | Strict Acc | Decisive Acc | Uncertainty | Precision | Recall |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for cohort_key, label in [
        ("sound_reasoning_none", "Sound Cases (`none`)"),
        ("unsupported_conceptual", "Unsupported Cases (`unsupported`)"),
        ("conditional_warnings", "Conditional Warnings (`conditional_warning`)"),
        ("all_incorrect_cases", "All Incorrect Cases (Injected Errors)"),
    ]:
        cd = cohorts[cohort_key]
        lines.append(
            f"| **{label}** | {cd['total_cases']} | "
            f"{cd['accuracy_metrics']['strict_accuracy']*100:.1f}% | "
            f"{cd['accuracy_metrics']['decisive_accuracy']*100:.1f}% | "
            f"{cd['accuracy_metrics']['uncertainty_rate']*100:.1f}% | "
            f"{cd['error_detection_metrics']['precision']*100:.1f}% | "
            f"{cd['error_detection_metrics']['conservative_recall']*100:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Breakdown by Injected Error Category (N=120)",
        "",
        "| Error Category | N | Detected (TP) | Missed (Explicit FN) | Missed (Uncertain FN) | Recall |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for cat_name, cd in cats.items():
        cm_c = cd["confusion_matrix"]
        recall = cd["error_detection_metrics"]["conservative_recall"] * 100
        lines.append(
            f"| **`{cat_name}`** | {cd['total_cases']} | "
            f"{cm_c['true_positives']} | {cm_c['false_negatives_explicit']} | {cm_c['false_negatives_uncertain']} | {recall:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Symbolic Limitation Boundary Cases (`eval_191` to `eval_200`, N=10)",
        "",
        f"- **Total Cases**: {lim['total_cases']}",
        f"- **Detected Errors (TP)**: {lim['confusion_matrix']['true_positives']} / {lim['total_cases']}",
        f"- **Conservative Recall**: {lim['error_detection_metrics']['conservative_recall'] * 100:.1f}%",
        f"- **Decisive Accuracy**: {lim['accuracy_metrics']['decisive_accuracy'] * 100:.1f}%",
        f"- **Uncertainty Rate**: {lim['accuracy_metrics']['uncertainty_rate'] * 100:.1f}%",
        "",
    ])

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text


def run_llm_baseline_evaluation(
    dataset_path: str = DEFAULT_DATASET_PATH,
    predictions_path: str = DEFAULT_PREDICTIONS_PATH,
    metrics_path: str = DEFAULT_METRICS_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    provider: str = "gemini",
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    delay_seconds: float = 4.0,
    max_cases: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Executes the LLM-only error detection evaluation pipeline.

    Checkpoints each successful prediction immediately to predictions_path.
    Resumes seamlessly if partially completed by skipping existing IDs.
    """
    cases = load_dataset(dataset_path)
    if max_cases is not None:
        cases = cases[:max_cases]

    completed_predictions = load_existing_predictions(predictions_path)
    print(f"Loaded {len(cases)} cases from {dataset_path}.")
    print(f"Found {len(completed_predictions)} existing checkpointed predictions.")

    if not dry_run:
        for idx, case in enumerate(cases, 1):
            cid = case["id"]
            if cid in completed_predictions:
                continue

            # Strict Information Barrier: Extract ONLY problem and reasoning_text
            problem = case["problem"]
            reasoning_text = case["reasoning_text"]

            print(f"[{idx}/{len(cases)}] Querying LLM baseline for {cid}...")
            detection = detect_reasoning_error_llm(
                problem=problem,
                reasoning_text=reasoning_text,
                provider=provider,
                client=client,
                model_name=model_name,
            )

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
            if detection.parsed_successfully:
                completed_predictions[cid] = record

            if delay_seconds > 0 and idx < len(cases):
                time.sleep(delay_seconds)

    # Compute comprehensive metrics
    metrics = compute_all_evaluation_metrics(cases, completed_predictions)

    # Save metrics JSON
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Generate Markdown Report
    generate_markdown_report(metrics, report_path)

    print(f"Evaluation complete. Metrics saved to {metrics_path}. Report saved to {report_path}.")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-Only Error Detection Baseline.")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "groq"])
    parser.add_argument("--delay", type=float, default=4.0, help="Delay between calls in seconds.")
    parser.add_argument("--max-cases", type=int, default=None, help="Maximum number of cases to evaluate.")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics from existing predictions without API calls.")

    args = parser.parse_args()
    run_llm_baseline_evaluation(
        provider=args.provider,
        delay_seconds=args.delay,
        max_cases=args.max_cases,
        dry_run=args.dry_run,
    )
