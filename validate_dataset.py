"""
validate_dataset.py
-------------------
Phase 4 - Objective 26: Labeled Evaluation Dataset Validator.

Validates the integrity, schema compliance, label validity, and statistical distribution
of the reasoning evaluation dataset (data/reasoning_eval_dataset.jsonl).

Validation criteria:
1. File existence and JSONL parseability.
2. Exactly 200 total entries.
3. Unique and properly formatted case IDs (e.g. 'eval_001').
4. Schema conformity for top-level, ground_truth, and metadata fields.
5. Allowed enumerations for reasoning_type, error_category, and expected_report_status.
6. Error step reference bounds check (1 <= error_step <= total_steps).
7. Compatibility with step_extractor.py (extracted step count matches step_labels count).
8. Absence of null or empty essential fields.
9. Dataset balance and distribution statistics.
"""

import json
import os
import re
import sys
from collections import Counter
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_extractor import extract_reasoning_steps


ALLOWED_REASONING_TYPES = {"single_step", "multi_step"}
ALLOWED_ERROR_CATEGORIES = {
    "none",
    "arithmetic_error",
    "invalid_transformation",
    "lost_solution",
    "division_by_zero",
    "domain_violation",
    "incorrect_setup",
    "unsupported",
    "conditional_warning",
}
ALLOWED_REPORT_STATUSES = {
    "verified",
    "errors_detected",
    "warnings",
    "partially_verified",
    "verification_failed",
}
ALLOWED_DOMAINS = {
    "arithmetic",
    "algebra",
    "word_problem",
    "geometry",
    "number_theory",
    "calculus",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


def validate_case_schema(case: Dict[str, Any], index: int) -> List[str]:
    """Validates schema and field types for a single test case entry."""
    errors = []
    case_id = case.get("id", f"entry_{index}")

    # Top-level required fields
    for field in ["id", "problem", "reference_final_answer", "reasoning_text", "reasoning_type", "ground_truth", "metadata"]:
        if field not in case:
            errors.append(f"[{case_id}] Missing required top-level field '{field}'")

    if errors:
        return errors

    # Check ID format
    if not re.match(r"^eval_\d{3}$", case["id"]):
        errors.append(f"[{case_id}] Invalid ID format '{case['id']}'; expected 'eval_XXX'")

    # Check non-empty strings
    if not case["problem"].strip():
        errors.append(f"[{case_id}] 'problem' cannot be empty")
    if not str(case["reference_final_answer"]).strip():
        errors.append(f"[{case_id}] 'reference_final_answer' cannot be empty")
    if not case["reasoning_text"].strip():
        errors.append(f"[{case_id}] 'reasoning_text' cannot be empty")

    # Reasoning type
    if case["reasoning_type"] not in ALLOWED_REASONING_TYPES:
        errors.append(f"[{case_id}] Invalid reasoning_type '{case['reasoning_type']}'; allowed: {ALLOWED_REASONING_TYPES}")

    # Ground truth sub-fields
    gt = case["ground_truth"]
    for field in ["is_correct", "error_category", "error_step", "error_sub_type", "expected_report_status", "step_labels"]:
        if field not in gt:
            errors.append(f"[{case_id}] Missing ground_truth field '{field}'")

    if not isinstance(gt.get("is_correct"), bool):
        errors.append(f"[{case_id}] 'is_correct' must be a boolean")

    if gt.get("error_category") not in ALLOWED_ERROR_CATEGORIES:
        errors.append(f"[{case_id}] Invalid error_category '{gt.get('error_category')}'; allowed: {ALLOWED_ERROR_CATEGORIES}")

    if gt.get("expected_report_status") not in ALLOWED_REPORT_STATUSES:
        errors.append(f"[{case_id}] Invalid expected_report_status '{gt.get('expected_report_status')}'; allowed: {ALLOWED_REPORT_STATUSES}")

    step_labels = gt.get("step_labels", [])
    if not isinstance(step_labels, list) or len(step_labels) == 0:
        errors.append(f"[{case_id}] 'step_labels' must be a non-empty list")
    else:
        for idx, sl in enumerate(step_labels):
            for sl_field in ["step_number", "is_correct", "step_status", "error_category", "notes"]:
                if sl_field not in sl:
                    errors.append(f"[{case_id}] Step label {idx+1} missing field '{sl_field}'")

    # Error step reference integrity
    err_step = gt.get("error_step")
    is_correct = gt.get("is_correct")
    error_cat = gt.get("error_category")
    rep_status = gt.get("expected_report_status")

    if is_correct is True:
        if err_step is not None:
            errors.append(f"[{case_id}] Correct case must have error_step=None, got {err_step}")
        if error_cat not in ("none", "unsupported", "conditional_warning"):
            errors.append(f"[{case_id}] Correct case has invalid error_category '{error_cat}'")
        if error_cat == "none" and rep_status != "verified":
            errors.append(f"[{case_id}] Correct case with category 'none' must have status 'verified', got '{rep_status}'")
        elif error_cat == "unsupported" and rep_status != "partially_verified":
            errors.append(f"[{case_id}] Correct case with category 'unsupported' must have status 'partially_verified', got '{rep_status}'")
        elif error_cat == "conditional_warning" and rep_status != "warnings":
            errors.append(f"[{case_id}] Correct case with category 'conditional_warning' must have status 'warnings', got '{rep_status}'")
        for sl in step_labels:
            if not sl.get("is_correct"):
                errors.append(f"[{case_id}] Correct case contains incorrect step {sl.get('step_number')}")
    elif is_correct is False:
        if err_step is None:
            errors.append(f"[{case_id}] Incorrect case must have non-null error_step")
        elif not isinstance(err_step, int) or err_step < 1 or err_step > len(step_labels):
            errors.append(f"[{case_id}] 'error_step' ({err_step}) out of bounds (1..{len(step_labels)})")
        if error_cat == "none":
            errors.append(f"[{case_id}] Incorrect case cannot have error_category 'none'")
        if rep_status != "errors_detected":
            errors.append(f"[{case_id}] Incorrect case must have expected_report_status 'errors_detected', got '{rep_status}'")
        
        # Verify step labels match error step
        if isinstance(err_step, int) and 1 <= err_step <= len(step_labels):
            flagged_step = step_labels[err_step - 1]
            if flagged_step.get("is_correct") is not False:
                errors.append(f"[{case_id}] Step {err_step} marked by error_step is not labeled is_correct=False")
            if flagged_step.get("error_category") != error_cat:
                errors.append(f"[{case_id}] Step {err_step} error_category '{flagged_step.get('error_category')}' does not match case category '{error_cat}'")

    # Metadata sub-fields
    meta = case.get("metadata", {})
    if meta.get("domain") not in ALLOWED_DOMAINS:
        errors.append(f"[{case_id}] Invalid domain '{meta.get('domain')}'; allowed: {ALLOWED_DOMAINS}")

    if meta.get("difficulty") not in ALLOWED_DIFFICULTIES:
        errors.append(f"[{case_id}] Invalid difficulty '{meta.get('difficulty')}'; allowed: {ALLOWED_DIFFICULTIES}")

    if not isinstance(meta.get("symbolic_limitation"), bool):
        errors.append(f"[{case_id}] 'symbolic_limitation' must be a boolean")

    return errors


def validate_dataset_file(file_path: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Runs complete validation across all records in the dataset file."""
    if not os.path.exists(file_path):
        return False, [f"Dataset file does not exist: {file_path}"], {}

    all_errors = []
    cases = []
    seen_ids = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                cases.append(record)
            except Exception as e:
                all_errors.append(f"Line {line_num}: Invalid JSON syntax ({e})")

    if all_errors:
        return False, all_errors, {}

    # Total count check
    if len(cases) != 200:
        all_errors.append(f"Expected exactly 200 cases, found {len(cases)}")

    # Record-level validation
    seen_problems = {}
    seen_traces = {}
    for idx, case in enumerate(cases):
        case_id = case.get("id")
        if case_id in seen_ids:
            all_errors.append(f"Duplicate case ID '{case_id}' at record {idx+1}")
        seen_ids.add(case_id)

        prob_key = case.get("problem", "").strip().lower()
        if prob_key in seen_problems:
            all_errors.append(f"Duplicate problem found in '{case_id}' and '{seen_problems[prob_key]}'")
        seen_problems[prob_key] = case_id

        trace_key = case.get("reasoning_text", "").strip()
        if trace_key in seen_traces:
            all_errors.append(f"Duplicate reasoning trace found in '{case_id}' and '{seen_traces[trace_key]}'")
        seen_traces[trace_key] = case_id

        # Validate schema
        errs = validate_case_schema(case, idx + 1)
        all_errors.extend(errs)

        # Step Extractor Integration check
        raw_text = case.get("reasoning_text", "")
        extracted = extract_reasoning_steps(raw_text)
        expected_step_count = len(case.get("ground_truth", {}).get("step_labels", []))
        if len(extracted.steps) != expected_step_count:
            all_errors.append(
                f"[{case_id}] step_extractor extracted {len(extracted.steps)} steps, but step_labels has {expected_step_count}"
            )

    # Compute Statistics
    stats = {
        "total_cases": len(cases),
        "correct_count": sum(1 for c in cases if c["ground_truth"]["is_correct"]),
        "incorrect_count": sum(1 for c in cases if not c["ground_truth"]["is_correct"]),
        "category_distribution": dict(Counter(c["ground_truth"]["error_category"] for c in cases)),
        "reasoning_type_distribution": dict(Counter(c["reasoning_type"] for c in cases)),
        "domain_distribution": dict(Counter(c["metadata"]["domain"] for c in cases)),
        "difficulty_distribution": dict(Counter(c["metadata"]["difficulty"] for c in cases)),
        "expected_report_status_distribution": dict(Counter(c["ground_truth"]["expected_report_status"] for c in cases)),
        "symbolic_limitation_count": sum(1 for c in cases if c["metadata"]["symbolic_limitation"]),
    }

    is_valid = len(all_errors) == 0
    return is_valid, all_errors, stats


def print_validation_report(file_path: str) -> bool:
    """Validates the dataset file and prints a comprehensive human-readable report."""
    print("=" * 70)
    print("PHASE 4 - OBJECTIVE 26: EVALUATION DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"Target File: {file_path}")

    is_valid, errors, stats = validate_dataset_file(file_path)

    if not is_valid:
        print("\n[VALIDATION FAILED] The following errors were found:")
        for e in errors[:30]:
            print(f"  - {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more errors.")
        return False

    print("\n[STATUS: PASS] All schema, integrity, and compatibility checks passed!\n")
    print(f"Total Cases: {stats['total_cases']}")
    print(f"Correct Reasoning: {stats['correct_count']} ({stats['correct_count'] / stats['total_cases'] * 100:.1f}%)")
    print(f"Incorrect Reasoning: {stats['incorrect_count']} ({stats['incorrect_count'] / stats['total_cases'] * 100:.1f}%)")
    print(f"Symbolic Limitation Cases: {stats['symbolic_limitation_count']}")

    print("\n--- Error Category Distribution ---")
    for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {count:3d} ({count / stats['total_cases'] * 100:.1f}%)")

    print("\n--- Expected Report Status Distribution ---")
    for st, count in sorted(stats["expected_report_status_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {st:25s}: {count:3d} ({count / stats['total_cases'] * 100:.1f}%)")

    print("\n--- Reasoning Type Distribution ---")
    for rt, count in sorted(stats["reasoning_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {rt:25s}: {count:3d} ({count / stats['total_cases'] * 100:.1f}%)")

    print("\n--- Mathematical Domain Distribution ---")
    for dom, count in sorted(stats["domain_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {dom:25s}: {count:3d} ({count / stats['total_cases'] * 100:.1f}%)")

    print("\n--- Difficulty Distribution ---")
    for diff, count in sorted(stats["difficulty_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {diff:25s}: {count:3d} ({count / stats['total_cases'] * 100:.1f}%)")

    print("=" * 70)
    return True


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reasoning_eval_dataset.jsonl")
    path_to_validate = sys.argv[1] if len(sys.argv) > 1 else default_path
    success = print_validation_report(path_to_validate)
    sys.exit(0 if success else 1)
