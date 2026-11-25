"""
test_step_localization.py
-------------------------
Unit and integration tests for Phase 4 Objective 32: Step-Level Localization Evaluation.

Tests:
1. Correct exact localization calculations.
2. Mislocalized true positives handling.
3. False negatives with no predicted step.
4. False positives on sound cases.
5. Category-level denominators (including zero-denominator handling).
6. Step-depth denominators.
7. Mean Absolute Error (MAE) step distance calculation.
8. Objective 28 common-subset handling (187 cases).
9. Invalid and duplicate prediction records handling.
10. Confusion-matrix row and column totals reconciliation.
11. End-to-end verification against verified benchmark expected values.
"""

import json
import os
import unittest
from typing import Any, Dict, List

from evaluate_step_localization import (
    load_canonical_dataset,
    load_and_validate_predictions,
    calculate_step_localization_metrics,
    run_evaluation,
    DATASET_PATH,
    LLM_BASELINE_PATH,
    LLM_IMPROVED_PATH,
    SOLVER_ONLY_PATH,
    HYBRID_PATH,
)


class TestStepLocalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_canonical_dataset(DATASET_PATH)
        cls.b_preds = load_and_validate_predictions(LLM_BASELINE_PATH, "llm_baseline")
        cls.imp_preds = load_and_validate_predictions(LLM_IMPROVED_PATH, "llm_improved")
        cls.s_preds = load_and_validate_predictions(SOLVER_ONLY_PATH, "solver_only")
        cls.h_preds = load_and_validate_predictions(HYBRID_PATH, "hybrid")

        cls.common_ids = set(cls.imp_preds.keys())
        cls.common_cases = [c for c in cls.dataset if c["id"] in cls.common_ids]

    def test_01_synthetic_exact_localization(self):
        """Test exact match yields 100% conditional accuracy and 0 MAE."""
        gold = [
            {"id": "c1", "ground_truth": {"is_correct": False, "error_step": 1, "error_category": "arithmetic_error"}},
            {"id": "c2", "ground_truth": {"is_correct": False, "error_step": 2, "error_category": "invalid_transformation"}},
        ]
        preds = {
            "c1": {"id": "c1", "classification": "incorrect", "predicted_step": 1},
            "c2": {"id": "c2", "classification": "incorrect", "predicted_step": 2},
        }
        res = calculate_step_localization_metrics(gold, preds)
        loc = res["overall_localization"]
        self.assertEqual(loc["exact_matches"], 2)
        self.assertEqual(loc["tp_count"], 2)
        self.assertEqual(loc["conditional_localization_accuracy"], 1.0)
        self.assertEqual(loc["unconditional_localization_accuracy"], 1.0)
        self.assertEqual(loc["mean_absolute_step_distance"], 0.0)
        self.assertEqual(loc["distance_distribution"]["exact_match_0"], 2)

    def test_02_synthetic_mislocalized_true_positives(self):
        """Test mislocalized TPs appear in predicted column, contributing to MAE."""
        gold = [
            {"id": "c1", "ground_truth": {"is_correct": False, "error_step": 1, "error_category": "arithmetic_error"}},
            {"id": "c2", "ground_truth": {"is_correct": False, "error_step": 3, "error_category": "invalid_transformation"}},
        ]
        preds = {
            "c1": {"id": "c1", "classification": "incorrect", "predicted_step": 2}, # dist 1
            "c2": {"id": "c2", "classification": "incorrect", "predicted_step": 1}, # dist 2
        }
        res = calculate_step_localization_metrics(gold, preds)
        loc = res["overall_localization"]
        self.assertEqual(loc["exact_matches"], 0)
        self.assertEqual(loc["tp_count"], 2)
        self.assertEqual(loc["conditional_localization_accuracy"], 0.0)
        self.assertEqual(loc["unconditional_localization_accuracy"], 0.0)
        self.assertEqual(loc["mean_absolute_step_distance"], 1.5)  # (1 + 2) / 2 = 1.5
        self.assertEqual(loc["distance_distribution"]["distance_1"], 1)
        self.assertEqual(loc["distance_distribution"]["distance_2"], 1)

        # Check confusion matrix placement
        cm = res["first_error_confusion_matrix"]
        self.assertEqual(cm["Step 1"]["Pred Step 2"], 1)
        self.assertEqual(cm["Step 3"]["Pred Step 1"], 1)

    def test_03_synthetic_false_negatives_no_predicted_step(self):
        """Test false negatives are placed in 'No Error Pred' and penalize unconditional accuracy."""
        gold = [
            {"id": "c1", "ground_truth": {"is_correct": False, "error_step": 1, "error_category": "arithmetic_error"}},
            {"id": "c2", "ground_truth": {"is_correct": False, "error_step": 2, "error_category": "domain_violation"}},
        ]
        preds = {
            "c1": {"id": "c1", "classification": "incorrect", "predicted_step": 1},
            "c2": {"id": "c2", "classification": "correct", "predicted_step": None}, # FN
        }
        res = calculate_step_localization_metrics(gold, preds)
        loc = res["overall_localization"]
        self.assertEqual(loc["exact_matches"], 1)
        self.assertEqual(loc["tp_count"], 1)
        self.assertEqual(loc["error_case_count"], 2)
        self.assertEqual(loc["conditional_localization_accuracy"], 1.0)
        self.assertEqual(loc["unconditional_localization_accuracy"], 0.5)

        cm = res["first_error_confusion_matrix"]
        self.assertEqual(cm["Step 2"]["No Error Pred"], 1)
        self.assertEqual(cm["Step 1"]["Pred Step 1"], 1)

    def test_04_synthetic_false_positives_on_sound_cases(self):
        """Test false positives on sound cases appear in the Sound row and their predicted column."""
        gold = [
            {"id": "s1", "ground_truth": {"is_correct": True, "error_step": None, "error_category": "none"}},
            {"id": "s2", "ground_truth": {"is_correct": True, "error_step": None, "error_category": "none"}},
        ]
        preds = {
            "s1": {"id": "s1", "classification": "correct", "predicted_step": None}, # TN
            "s2": {"id": "s2", "classification": "incorrect", "predicted_step": 2}, # FP at step 2
        }
        res = calculate_step_localization_metrics(gold, preds)
        cm = res["first_error_confusion_matrix"]
        self.assertEqual(cm["Sound"]["No Error Pred"], 1) # TN
        self.assertEqual(cm["Sound"]["Pred Step 2"], 1)   # FP at step 2
        self.assertEqual(cm["Sound"]["Pred Step 1"], 0)

        counts = res["confusion_matrix_counts"]
        self.assertEqual(counts["false_positives"], 1)
        self.assertEqual(counts["true_negatives"], 1)

    def test_05_category_denominators_and_zero_denominator_handling(self):
        """Test category-level metrics explicitly return None (not 0%) when TP is 0."""
        gold = [
            {"id": "c1", "ground_truth": {"is_correct": False, "error_step": 1, "error_category": "cat_a"}},
            {"id": "c2", "ground_truth": {"is_correct": False, "error_step": 1, "error_category": "cat_b"}},
        ]
        preds = {
            "c1": {"id": "c1", "classification": "incorrect", "predicted_step": 1},
            "c2": {"id": "c2", "classification": "correct", "predicted_step": None}, # 0 TP for cat_b
        }
        res = calculate_step_localization_metrics(gold, preds)
        cats = res["by_error_category"]

        # cat_a has 1 TP
        self.assertEqual(cats["cat_a"]["conditional_accuracy"], 1.0)
        self.assertEqual(cats["cat_a"]["unconditional_accuracy"], 1.0)

        # cat_b has 0 TP
        self.assertIsNone(cats["cat_b"]["conditional_accuracy"])
        self.assertEqual(cats["cat_b"]["unconditional_accuracy"], 0.0)

    def test_06_step_depth_denominators(self):
        """Test step-depth metrics correctly partition cases by error depth."""
        gold = [
            {"id": "c1", "ground_truth": {"is_correct": False, "error_step": 1, "error_category": "cat_a"}},
            {"id": "c2", "ground_truth": {"is_correct": False, "error_step": 2, "error_category": "cat_a"}},
            {"id": "c3", "ground_truth": {"is_correct": False, "error_step": 2, "error_category": "cat_a"}},
        ]
        preds = {
            "c1": {"id": "c1", "classification": "incorrect", "predicted_step": 1},
            "c2": {"id": "c2", "classification": "incorrect", "predicted_step": 2},
            "c3": {"id": "c3", "classification": "correct", "predicted_step": None},
        }
        res = calculate_step_localization_metrics(gold, preds)
        depths = res["by_error_step_depth"]

        self.assertEqual(depths["step_1"]["total_cases"], 1)
        self.assertEqual(depths["step_1"]["conditional_accuracy"], 1.0)

        self.assertEqual(depths["step_2"]["total_cases"], 2)
        self.assertEqual(depths["step_2"]["true_positives"], 1)
        self.assertEqual(depths["step_2"]["conditional_accuracy"], 1.0)
        self.assertEqual(depths["step_2"]["unconditional_accuracy"], 0.5)

    def test_07_confusion_matrix_row_and_column_totals(self):
        """Test that confusion matrix row totals equal gold cohort sizes and grand total equals dataset size."""
        res = calculate_step_localization_metrics(self.dataset, self.b_preds)
        cm = res["first_error_confusion_matrix"]

        # Row totals
        sound_total = sum(cm["Sound"].values())
        step1_total = sum(cm["Step 1"].values())
        step2_total = sum(cm["Step 2"].values())
        step3_total = sum(cm["Step 3"].values())
        step4_total = sum(cm["Step 4"].values())

        self.assertEqual(sound_total, 80)
        self.assertEqual(step1_total, 85)
        self.assertEqual(step2_total, 27)
        self.assertEqual(step3_total, 8)
        self.assertEqual(step4_total, 0)

        grand_total = sound_total + step1_total + step2_total + step3_total + step4_total
        self.assertEqual(grand_total, 200)

    def test_08_invalid_and_duplicate_prediction_handling(self):
        """Test loader handles deduplication and filters out unparsed records for Objective 28."""
        self.assertEqual(len(self.imp_preds), 187)
        self.assertEqual(len(self.b_preds), 200)
        self.assertEqual(len(self.s_preds), 200)
        self.assertEqual(len(self.h_preds), 200)

        # Check all predicted steps are valid
        for p in self.b_preds.values():
            s = p["predicted_step"]
            self.assertTrue(s is None or (isinstance(s, int) and s >= 1))

    def test_09_verified_expected_values_full_200(self):
        """Verify the exact audited validation checks for the full 200-case results."""
        # Objective 27: 114 exact matches, 117 TP, 3 FN, conditional 97.44%, unconditional 95.00%, MAE 0.0256
        res_27 = calculate_step_localization_metrics(self.dataset, self.b_preds)
        loc_27 = res_27["overall_localization"]
        self.assertEqual(loc_27["exact_matches"], 114)
        self.assertEqual(loc_27["tp_count"], 117)
        self.assertEqual(res_27["confusion_matrix_counts"]["false_negatives"], 3)
        self.assertAlmostEqual(loc_27["conditional_localization_accuracy"], 0.9743589, places=4)
        self.assertAlmostEqual(loc_27["unconditional_localization_accuracy"], 0.9500000, places=4)
        self.assertAlmostEqual(loc_27["mean_absolute_step_distance"], 0.0256410, places=4)

        # Objective 29: 50 exact matches, 59 TP, 61 FN, conditional 84.75%, unconditional 41.67%, MAE 0.2034
        res_29 = calculate_step_localization_metrics(self.dataset, self.s_preds)
        loc_29 = res_29["overall_localization"]
        self.assertEqual(loc_29["exact_matches"], 50)
        self.assertEqual(loc_29["tp_count"], 59)
        self.assertEqual(res_29["confusion_matrix_counts"]["false_negatives"], 61)
        self.assertAlmostEqual(loc_29["conditional_localization_accuracy"], 0.8474576, places=4)
        self.assertAlmostEqual(loc_29["unconditional_localization_accuracy"], 0.4166666, places=4)
        self.assertAlmostEqual(loc_29["mean_absolute_step_distance"], 0.2033898, places=4)

        # Objective 30: 117 exact matches, 120 TP, 0 FN, conditional 97.50%, unconditional 97.50%, MAE 0.0250
        res_30 = calculate_step_localization_metrics(self.dataset, self.h_preds)
        loc_30 = res_30["overall_localization"]
        self.assertEqual(loc_30["exact_matches"], 117)
        self.assertEqual(loc_30["tp_count"], 120)
        self.assertEqual(res_30["confusion_matrix_counts"]["false_negatives"], 0)
        self.assertAlmostEqual(loc_30["conditional_localization_accuracy"], 0.9750000, places=4)
        self.assertAlmostEqual(loc_30["unconditional_localization_accuracy"], 0.9750000, places=4)
        self.assertAlmostEqual(loc_30["mean_absolute_step_distance"], 0.0250000, places=4)

    def test_10_verified_expected_values_common_187(self):
        """Verify the exact audited validation checks for the common 187-case subset."""
        self.assertEqual(len(self.common_cases), 187)
        gold_err_187 = [c for c in self.common_cases if not c["ground_truth"]["is_correct"]]
        gold_sound_187 = [c for c in self.common_cases if c["ground_truth"]["is_correct"]]
        self.assertEqual(len(gold_err_187), 107)
        self.assertEqual(len(gold_sound_187), 80)

        # Objective 27 (187): 101 exact matches, 104 TP, 3 FN, conditional 97.12%, unconditional 94.39%, MAE 0.0288
        res_27 = calculate_step_localization_metrics(self.common_cases, self.b_preds)
        loc_27 = res_27["overall_localization"]
        self.assertEqual(loc_27["exact_matches"], 101)
        self.assertEqual(loc_27["tp_count"], 104)
        self.assertEqual(res_27["confusion_matrix_counts"]["false_negatives"], 3)
        self.assertAlmostEqual(loc_27["conditional_localization_accuracy"], 0.9711538, places=4)
        self.assertAlmostEqual(loc_27["unconditional_localization_accuracy"], 0.9439252, places=4)
        self.assertAlmostEqual(loc_27["mean_absolute_step_distance"], 0.0288461, places=4)

        # Objective 28 (187): 100 exact matches, 104 TP, 3 FN, conditional 96.15%, unconditional 93.46%, MAE 0.0385
        res_28 = calculate_step_localization_metrics(self.common_cases, self.imp_preds)
        loc_28 = res_28["overall_localization"]
        self.assertEqual(loc_28["exact_matches"], 100)
        self.assertEqual(loc_28["tp_count"], 104)
        self.assertEqual(res_28["confusion_matrix_counts"]["false_negatives"], 3)
        self.assertAlmostEqual(loc_28["conditional_localization_accuracy"], 0.9615384, places=4)
        self.assertAlmostEqual(loc_28["unconditional_localization_accuracy"], 0.9345794, places=4)
        self.assertAlmostEqual(loc_28["mean_absolute_step_distance"], 0.0384615, places=4)

        # Objective 29 (187): 49 exact matches, 58 TP, 49 FN, conditional 84.48%, unconditional 45.79%, MAE 0.2069
        res_29 = calculate_step_localization_metrics(self.common_cases, self.s_preds)
        loc_29 = res_29["overall_localization"]
        self.assertEqual(loc_29["exact_matches"], 49)
        self.assertEqual(loc_29["tp_count"], 58)
        self.assertEqual(res_29["confusion_matrix_counts"]["false_negatives"], 49)
        self.assertAlmostEqual(loc_29["conditional_localization_accuracy"], 0.8448275, places=4)
        self.assertAlmostEqual(loc_29["unconditional_localization_accuracy"], 0.4579439, places=4)
        self.assertAlmostEqual(loc_29["mean_absolute_step_distance"], 0.2068965, places=4)

        # Objective 30 (187): 104 exact matches, 107 TP, 0 FN, conditional 97.20%, unconditional 97.20%, MAE 0.0280
        res_30 = calculate_step_localization_metrics(self.common_cases, self.h_preds)
        loc_30 = res_30["overall_localization"]
        self.assertEqual(loc_30["exact_matches"], 104)
        self.assertEqual(loc_30["tp_count"], 107)
        self.assertEqual(res_30["confusion_matrix_counts"]["false_negatives"], 0)
        self.assertAlmostEqual(loc_30["conditional_localization_accuracy"], 0.9719626, places=4)
        self.assertAlmostEqual(loc_30["unconditional_localization_accuracy"], 0.9719626, places=4)
        self.assertAlmostEqual(loc_30["mean_absolute_step_distance"], 0.0280373, places=4)


if __name__ == "__main__":
    unittest.main()
