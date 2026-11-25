"""
test_hybrid_error_detector.py
-----------------------------
Phase 4 — Objective 30: Automated Test Suite for Hybrid LLM + Solver Architecture.

Verifies:
1. LLM-only error with solver unsupported.
2. Solver-detected arithmetic error where LLM missed.
3. Full agreement on sound reasoning.
4. Full agreement on detected error.
5. Disagreements between systems.
6. Incorrect problem setup (LLM catches semantic flaw, solver algebra holds).
7. Division-by-zero detection and priority.
8. Unsupported reasoning and preservation of unsupported caveats.
9. Complete preservation of evidence from both systems.
10. Zero gold-label leakage into hybrid inputs or prediction records.
11. Exactly 200 hybrid output prediction records with sequential ID correspondence.
12. Correct metric calculations, formulas, and denominators.
13. Common 187-case comparison metrics and markdown report integrity.
"""

import json
import os
import unittest
from typing import Any, Dict

from evaluate_hybrid import (
    DEFAULT_COMMON_187_METRICS_PATH,
    DEFAULT_COMMON_187_REPORT_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_HYBRID_METRICS_PATH,
    DEFAULT_HYBRID_PREDICTIONS_PATH,
    DEFAULT_HYBRID_REPORT_PATH,
    compute_all_hybrid_metrics,
    run_pipeline,
)
from hybrid_error_detector import HybridPrediction, fuse_predictions


class TestHybridErrorDetector(unittest.TestCase):
    """Test suite for Objective 30 hybrid error detector and evaluation pipeline."""

    def test_01_agreement_sound_reasoning(self):
        """When both LLM and solver agree on correct, hybrid emits confirmed_correct."""
        llm = {"classification": "correct", "first_incorrect_step": None, "confidence": "high", "explanation": "Sound math"}
        solver = {"solver_classification": "correct", "verification_status": "verified", "evidence": "All steps verified"}
        fused = fuse_predictions(llm, solver, case_id="test_01")

        self.assertEqual(fused.hybrid_classification, "correct")
        self.assertEqual(fused.hybrid_verdict_type, "confirmed_correct")
        self.assertFalse(fused.disagreement)
        self.assertTrue(fused.symbolic_supported)
        self.assertIsNone(fused.first_incorrect_step)

    def test_02_agreement_error_detected(self):
        """When both LLM and solver agree on error, hybrid emits confirmed_error."""
        llm = {"classification": "incorrect", "first_incorrect_step": 2, "confidence": "high", "explanation": "Wrong sign"}
        solver = {"solver_classification": "incorrect", "verification_status": "errors_detected", "first_incorrect_step": 2, "detected_error_category": "invalid_transformation", "evidence": "LHS != RHS"}
        fused = fuse_predictions(llm, solver, case_id="test_02")

        self.assertEqual(fused.hybrid_classification, "incorrect")
        self.assertEqual(fused.hybrid_verdict_type, "confirmed_error")
        self.assertFalse(fused.disagreement)
        self.assertTrue(fused.step_agreement)
        self.assertEqual(fused.first_incorrect_step, 2)

    def test_03_solver_detected_arithmetic_error_llm_missed(self):
        """Solver mathematical proof overrides LLM false negative."""
        llm = {"classification": "correct", "first_incorrect_step": None, "confidence": "high", "explanation": "Looks fine"}
        solver = {"solver_classification": "incorrect", "verification_status": "errors_detected", "first_incorrect_step": 1, "detected_error_category": "incorrect_assertion", "evidence": "15 + 7 != 21"}
        fused = fuse_predictions(llm, solver, case_id="test_03")

        self.assertEqual(fused.hybrid_classification, "incorrect")
        self.assertEqual(fused.hybrid_verdict_type, "solver_overruled_llm")
        self.assertTrue(fused.disagreement)
        self.assertEqual(fused.first_incorrect_step, 1)
        self.assertEqual(fused.detected_error_category, "incorrect_assertion")
        self.assertIn("overrides", fused.fusion_rationale)

    def test_04_solver_detected_division_by_zero(self):
        """Solver division by zero establishes mathematical flaw."""
        llm = {"classification": "correct", "first_incorrect_step": None, "confidence": "high", "explanation": "Seems ok"}
        solver = {"solver_classification": "incorrect", "verification_status": "errors_detected", "first_incorrect_step": 2, "detected_error_category": "division_by_zero", "evidence": "Division by zero is undefined"}
        fused = fuse_predictions(llm, solver, case_id="test_04")

        self.assertEqual(fused.hybrid_classification, "incorrect")
        self.assertEqual(fused.hybrid_verdict_type, "solver_overruled_llm")
        self.assertEqual(fused.detected_error_category, "division_by_zero")

    def test_05_incorrect_problem_setup_llm_detected_solver_missed(self):
        """LLM semantic detection catches problem setup error where solver algebra is internally valid."""
        llm = {"classification": "incorrect", "first_incorrect_step": 1, "confidence": "high", "explanation": "Equations do not match the word problem text"}
        solver = {"solver_classification": "correct", "verification_status": "verified", "evidence": "Equations are algebraically self-consistent"}
        fused = fuse_predictions(llm, solver, case_id="test_05")

        self.assertEqual(fused.hybrid_classification, "incorrect")
        self.assertEqual(fused.hybrid_verdict_type, "llm_semantic_error")
        self.assertTrue(fused.disagreement)
        self.assertEqual(fused.first_incorrect_step, 1)

    def test_06_llm_only_error_with_solver_unsupported(self):
        """LLM catches error when solver is unsupported on natural language assertion."""
        llm = {"classification": "incorrect", "first_incorrect_step": 2, "confidence": "high", "explanation": "15 is composite, not prime"}
        solver = {"solver_classification": "unsupported", "verification_status": "verification_failed", "unsupported_reason": "No expressions could be parsed"}
        fused = fuse_predictions(llm, solver, case_id="test_06")

        self.assertEqual(fused.hybrid_classification, "incorrect")
        self.assertEqual(fused.hybrid_verdict_type, "unsupported_symbolic_llm_error")
        self.assertFalse(fused.symbolic_supported)
        self.assertEqual(fused.first_incorrect_step, 2)

    def test_07_unsupported_reasoning_handling(self):
        """Unsupported solver with sound LLM decision is marked correct with caveat."""
        llm = {"classification": "correct", "first_incorrect_step": None, "confidence": "high", "explanation": "Conceptual formula used properly"}
        solver = {"solver_classification": "unsupported", "verification_status": "partially_verified", "unsupported_reason": "Conceptual formula skipped"}
        fused = fuse_predictions(llm, solver, case_id="test_07")

        self.assertEqual(fused.hybrid_classification, "correct")
        self.assertEqual(fused.hybrid_verdict_type, "llm_accepted_symbolic_unsupported")
        self.assertFalse(fused.symbolic_supported)
        self.assertIn("symbolic proof was unavailable", fused.fusion_rationale)

    def test_08_disagreement_detection_and_preservation(self):
        """Both original predictions and evidence strings are strictly preserved."""
        llm = {"classification": "correct", "first_incorrect_step": None, "confidence": "low", "explanation": "Original LLM text"}
        solver = {"solver_classification": "incorrect", "verification_status": "errors_detected", "first_incorrect_step": 3, "detected_error_category": "arithmetic_error", "evidence": "Original solver text"}
        fused = fuse_predictions(llm, solver, case_id="test_08")

        self.assertTrue(fused.disagreement)
        self.assertEqual(fused.llm_prediction["explanation"], "Original LLM text")
        self.assertEqual(fused.solver_prediction["evidence"], "Original solver text")

    def test_09_step_disagreement_on_common_error(self):
        """When both detect error but differ on step, step_agreement is False and primary step recorded."""
        llm = {"classification": "incorrect", "first_incorrect_step": 1, "confidence": "high", "explanation": "Initial failure"}
        solver = {"solver_classification": "incorrect", "verification_status": "errors_detected", "first_incorrect_step": 3, "detected_error_category": "invalid_transformation", "evidence": "Downstream syntax"}
        fused = fuse_predictions(llm, solver, case_id="test_09")

        self.assertEqual(fused.hybrid_classification, "incorrect")
        self.assertFalse(fused.step_agreement)
        self.assertEqual(fused.first_incorrect_step, 1)

    def test_10_no_gold_label_leakage(self):
        """Ensure hybrid prediction records contain zero ground truth labels."""
        if not os.path.exists(DEFAULT_HYBRID_PREDICTIONS_PATH):
            run_pipeline()

        with open(DEFAULT_HYBRID_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.assertNotIn("ground_truth", rec)
                self.assertNotIn("is_correct", rec)
                self.assertNotIn("reference_final_answer", rec)

    def test_11_exactly_200_hybrid_output_records_and_correspondence(self):
        """Verify exactly 200 records in hybrid predictions with sequential IDs."""
        if not os.path.exists(DEFAULT_HYBRID_PREDICTIONS_PATH):
            run_pipeline()

        with open(DEFAULT_HYBRID_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(records), 200)
        expected_ids = [f"eval_{i:03d}" for i in range(1, 201)]
        actual_ids = [r["id"] for r in records]
        self.assertEqual(actual_ids, expected_ids)

    def test_12_metrics_calculations_and_denominators(self):
        """Verify 200-case hybrid metrics, formulas, and denominators."""
        if not os.path.exists(DEFAULT_HYBRID_METRICS_PATH):
            run_pipeline()

        with open(DEFAULT_HYBRID_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        ov = metrics["overall"]
        cm = ov["confusion_matrix"]
        acc = ov["accuracy_metrics"]
        edm = ov["error_detection_metrics"]
        loc = ov["step_localization"]

        self.assertEqual(ov["total_cases"], 200)
        self.assertEqual(cm["true_positives"], 120)
        self.assertEqual(cm["false_positives"], 8)
        self.assertEqual(cm["true_negatives"], 72)
        self.assertEqual(cm["total_false_negatives"], 0)

        # 100% Recall
        self.assertEqual(edm["recall"], 1.0)
        self.assertEqual(edm["precision"], round(120 / 128, 4))
        self.assertEqual(acc["strict_accuracy"], 0.96)
        self.assertEqual(loc["step_match_count"], 117)
        self.assertEqual(loc["tp_with_gold_step_count"], 120)
        self.assertEqual(loc["localization_accuracy"], 0.975)

        # Cohorts & categories completeness
        self.assertEqual(len(metrics["cohorts"]), 4)
        self.assertEqual(len(metrics["incorrect_categories"]), 6)
        self.assertEqual(metrics["symbolic_limitations"]["total_cases"], 10)

        # Fusion analysis counts
        fa = metrics["fusion_analysis"]
        self.assertEqual(fa["detected_by_hybrid_llm_missed_count"], 3)
        self.assertEqual(fa["detected_by_hybrid_solver_missed_count"], 61)
        self.assertEqual(fa["introduced_false_positives_count"], 5)

    def test_13_common_187_comparison_file_integrity(self):
        """Verify common 187-case comparison metrics and markdown report exist."""
        self.assertTrue(os.path.exists(DEFAULT_COMMON_187_METRICS_PATH))
        self.assertTrue(os.path.exists(DEFAULT_COMMON_187_REPORT_PATH))

        with open(DEFAULT_COMMON_187_METRICS_PATH, "r", encoding="utf-8") as f:
            comp = json.load(f)

        systems = comp["systems"]
        self.assertIn("objective_27", systems)
        self.assertIn("objective_28", systems)
        self.assertIn("objective_29", systems)
        self.assertIn("objective_30", systems)

        obj30 = systems["objective_30"]
        self.assertEqual(obj30["total_cases"], 187)
        self.assertEqual(obj30["confusion_matrix"]["tp"], 107)
        self.assertEqual(obj30["confusion_matrix"]["fn_exp"], 0)
        self.assertEqual(obj30["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
