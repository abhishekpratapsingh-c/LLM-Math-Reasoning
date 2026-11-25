"""
test_solver_only_evaluation.py
------------------------------
Phase 4 — Objective 29: Unit and Integration Tests for Solver-Only Evaluation.

Verifies:
1. Correct classification of supported valid reasoning traces.
2. Detection of supported arithmetic and algebraic transformation errors.
3. Detection of division-by-zero errors.
4. Handling of unsupported conceptual equations without guessing.
5. Robust handling of missing, empty, or malformed reasoning.
6. Zero gold-label leakage into solver inputs or prediction outputs.
7. Zero LLM or external API calls (Gemini/Groq).
8. Exactly 200 output prediction records with sequential ID correspondence.
9. Correct metric calculations, formulas, and denominators.
10. Markdown report generation and completeness.
"""

import json
import os
import unittest
from unittest.mock import patch

from evaluate_solver_only import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_REPORT_PATH,
    compute_all_solver_only_metrics,
    evaluate_single_case,
    run_pipeline,
)


class TestSolverOnlyEvaluation(unittest.TestCase):
    """Test suite for Objective 29 deterministic solver-only reasoning evaluation."""

    def test_01_evaluate_supported_valid_reasoning(self):
        """Supported sound reasoning is verified as correct with no errors."""
        problem = "Calculate the total cost of 3 books at $15 each and 2 pens at $5 each."
        reasoning = (
            "Step 1: Cost of books = 3 * 15 = 45\n\n"
            "Step 2: Cost of pens = 2 * 5 = 10\n\n"
            "Step 3: Total cost = 45 + 10 = 55\n\n"
            "Final Answer: 55"
        )
        res = evaluate_single_case(problem, reasoning, case_id="test_sound")

        self.assertEqual(res["solver_classification"], "correct")
        self.assertEqual(res["verification_status"], "verified")
        self.assertIsNone(res["detected_error_category"])
        self.assertIsNone(res["first_incorrect_step"])
        self.assertEqual(res["detected_errors_count"], 0)
        self.assertIn("verified", res["evidence"].lower())

    def test_02_detect_supported_arithmetic_error(self):
        """Supported arithmetic error is detected and classified as incorrect at Step 1."""
        problem = "Evaluate 25 * 4."
        reasoning = (
            "Step 1: Compute product\n"
            "25 * 4 = 110\n\n"
            "Final Answer: 110"
        )
        res = evaluate_single_case(problem, reasoning, case_id="test_arith_err")

        self.assertEqual(res["solver_classification"], "incorrect")
        self.assertEqual(res["verification_status"], "errors_detected")
        self.assertEqual(res["first_incorrect_step"], 1)
        self.assertEqual(res["detected_error_category"], "incorrect_assertion")
        self.assertEqual(res["detected_errors_count"], 1)

    def test_03_detect_supported_algebraic_error(self):
        """Supported invalid algebraic transformation is detected and localized."""
        problem = "Solve 2*x + 4 = 10."
        reasoning = (
            "Step 1: State equation\n"
            "2*x + 4 = 10\n\n"
            "Step 2: Subtract 4 from both sides\n"
            "2*x = 14\n\n"
            "Step 3: Divide by 2\n"
            "x = 7\n\n"
            "Final Answer: 7"
        )
        res = evaluate_single_case(problem, reasoning, case_id="test_alg_err")

        self.assertEqual(res["solver_classification"], "incorrect")
        self.assertEqual(res["verification_status"], "errors_detected")
        self.assertEqual(res["first_incorrect_step"], 2)
        self.assertEqual(res["detected_error_category"], "invalid_transformation")

    def test_04_detect_division_by_zero(self):
        """Division by zero is detected deterministically."""
        problem = "Evaluate 10 / (5 - 5)."
        reasoning = (
            "Step 1: Simplify denominator\n"
            "5 - 5 = 0\n\n"
            "Step 2: Divide numerator by zero\n"
            "10 / 0 = 10\n\n"
            "Final Answer: 10"
        )
        res = evaluate_single_case(problem, reasoning, case_id="test_div_zero")

        self.assertEqual(res["solver_classification"], "incorrect")
        self.assertEqual(res["verification_status"], "errors_detected")
        self.assertEqual(res["detected_error_category"], "division_by_zero")

    def test_05_unsupported_conceptual_case(self):
        """Conceptual formulas or unparsed syntax are held as unsupported abstentions."""
        problem = "A company sells units for $20 each with total cost $500 for 40 units. Find profit."
        reasoning = (
            "Step 1: Apply economic definition of profit\n"
            "Profit = Total Revenue - Total Cost\n\n"
            "Step 2: Calculate revenue and subtract cost\n"
            "Profit = 40 * 20 - 500 = 800 - 500 = 300\n\n"
            "Final Answer: 300"
        )
        res = evaluate_single_case(problem, reasoning, case_id="test_unsupp")

        self.assertEqual(res["solver_classification"], "unsupported")
        self.assertEqual(res["verification_status"], "partially_verified")
        self.assertIsNotNone(res["unsupported_reason"])

    def test_06_missing_or_malformed_reasoning(self):
        """Handles empty, whitespace, non-string, or unparseable input safely."""
        res_empty_prob = evaluate_single_case("", "Step 1: 1 + 1 = 2", case_id="test_empty_prob")
        self.assertEqual(res_empty_prob["solver_classification"], "unsupported")
        self.assertEqual(res_empty_prob["verification_status"], "verification_failed")

        res_empty_trace = evaluate_single_case("Problem", "", case_id="test_empty_trace")
        self.assertEqual(res_empty_trace["solver_classification"], "unsupported")
        self.assertEqual(res_empty_trace["verification_status"], "verification_failed")

        res_no_math = evaluate_single_case(
            "What is color of sky?",
            "Step 1: Observe sky\nThe sky is blue today.\n\nFinal Answer: Blue",
            case_id="test_no_math",
        )
        self.assertEqual(res_no_math["solver_classification"], "unsupported")
        self.assertEqual(res_no_math["verification_status"], "verification_failed")

    def test_07_no_gold_label_leakage(self):
        """Ensure solver inputs and prediction outputs contain no ground truth fields."""
        if not os.path.exists(DEFAULT_PREDICTIONS_PATH):
            run_pipeline()

        with open(DEFAULT_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.assertNotIn("ground_truth", rec)
                self.assertNotIn("is_correct", rec)
                self.assertNotIn("gold_error_step", rec)
                self.assertNotIn("reference_final_answer", rec)

    def test_08_no_llm_api_calls_made(self):
        """Assert zero LLM client creations or API calls occur during evaluation."""
        with patch("main.get_llm_client", side_effect=AssertionError("Live LLM client requested!")):
            res = evaluate_single_case("Problem 1", "Step 1: 2 + 2 = 4\n\nFinal Answer: 4", "case_test")
            self.assertEqual(res["solver_classification"], "correct")

    def test_09_exactly_200_output_records_and_case_correspondence(self):
        """Verify exactly 200 records with sequential ID matching in predictions JSONL."""
        if not os.path.exists(DEFAULT_PREDICTIONS_PATH):
            run_pipeline()

        with open(DEFAULT_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(records), 200)
        expected_ids = [f"eval_{i:03d}" for i in range(1, 201)]
        actual_ids = [r["id"] for r in records]
        self.assertEqual(actual_ids, expected_ids)

        # Check required fields in each record
        required_fields = {
            "id",
            "solver_classification",
            "detected_error_category",
            "first_incorrect_step",
            "verification_status",
            "evidence",
            "unsupported_reason",
            "total_steps",
            "total_expressions",
            "detected_errors_count",
            "detected_warnings_count",
            "unsupported_count",
        }
        for r in records:
            self.assertTrue(required_fields.issubset(set(r.keys())))
            self.assertIn(r["solver_classification"], ("correct", "incorrect", "unsupported"))

    def test_10_metric_calculations_and_denominators(self):
        """Verify metric calculation formulas, denominators, and values in results JSON."""
        if not os.path.exists(DEFAULT_METRICS_PATH):
            run_pipeline()

        with open(DEFAULT_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        ov = metrics["overall"]
        cm = ov["confusion_matrix"]
        acc = ov["accuracy_metrics"]
        edm = ov["error_detection_metrics"]
        loc = ov["step_localization"]

        self.assertEqual(ov["total_cases"], 200)
        self.assertEqual(ov["supported_cases"], 191)
        self.assertEqual(ov["unsupported_cases"], 9)

        # Confusion matrix checks
        self.assertEqual(cm["true_positives"], 59)
        self.assertEqual(cm["false_positives"], 6)
        self.assertEqual(cm["true_negatives"], 66)
        self.assertEqual(cm["false_negatives_explicit"], 60)
        self.assertEqual(cm["false_negatives_unsupported"], 1)
        self.assertEqual(cm["abstentions_on_correct"], 8)
        self.assertEqual(cm["total_unsupported"], 9)

        # Formula and denominator verifications
        expected_strict_acc = round((59 + 66) / 200, 4)
        expected_supp_acc = round((59 + 66) / 191, 4)
        expected_prec = round(59 / (59 + 6), 4)
        expected_rec = round(59 / 120, 4)
        expected_loc_acc = round(50 / 59, 4)

        self.assertEqual(acc["strict_accuracy"], expected_strict_acc)
        self.assertEqual(acc["supported_case_accuracy"], expected_supp_acc)
        self.assertEqual(edm["precision"], expected_prec)
        self.assertEqual(edm["recall"], expected_rec)
        self.assertEqual(loc["step_match_count"], 50)
        self.assertEqual(loc["tp_with_gold_step_count"], 59)
        self.assertEqual(loc["localization_accuracy"], expected_loc_acc)

        # Cohort completeness
        self.assertEqual(len(metrics["cohorts"]), 4)
        self.assertIn("sound_reasoning_none", metrics["cohorts"])
        self.assertIn("unsupported_conceptual", metrics["cohorts"])
        self.assertIn("conditional_warnings", metrics["cohorts"])
        self.assertIn("all_incorrect_cases", metrics["cohorts"])

        # Category completeness
        self.assertEqual(len(metrics["incorrect_categories"]), 6)
        self.assertEqual(metrics["symbolic_limitations"]["total_cases"], 10)


if __name__ == "__main__":
    unittest.main()
