"""
Unit Tests for Objective 28 vs Objective 27 Comparative Evaluation
==================================================================
Verifies:
1. Exact 187 common case count between Objective 27 and Objective 28.
2. Strict exclusion of the 13 unresolved Objective 28 IDs from both systems.
3. No data leakage of gold labels or reference answers into predictions.
4. Completeness and validity of results/objective28_comparison_187_metrics.json.
5. Completeness and validity of results/objective28_comparison_187_report.md.
6. Deterministic computation of summary deltas, cohorts, and category breakdowns.
"""

import json
import os
import unittest
from typing import Any, Dict

from generate_objective28_comparison import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_IMPROVED_PATH,
    DEFAULT_OUTPUT_METRICS_PATH,
    DEFAULT_OUTPUT_REPORT_PATH,
    EXCLUDED_CASE_IDS,
    build_comparison_analysis,
    load_and_validate_common_cases,
)


class TestObjective28Comparison(unittest.TestCase):
    """Test suite for Objective 28 vs Objective 27 187-case comparative evaluation."""

    def test_common_cases_count_and_exclusions(self):
        """Verify exactly 187 common cases and strict exclusion of the 13 IDs."""
        common_cases, base_preds, imp_preds = load_and_validate_common_cases()

        self.assertEqual(len(common_cases), 187, "Common cases must be exactly 187.")
        self.assertEqual(len(EXCLUDED_CASE_IDS), 13, "Excluded IDs must be exactly 13.")

        common_ids = {c["id"] for c in common_cases}

        # Check no excluded IDs in common subset
        intersection = common_ids.intersection(EXCLUDED_CASE_IDS)
        self.assertEqual(
            len(intersection),
            0,
            f"Common cases should not contain excluded IDs: {intersection}",
        )

        # Check that both prediction dicts contain all 187 common IDs
        for cid in common_ids:
            self.assertIn(cid, base_preds, f"Base preds missing common ID: {cid}")
            self.assertIn(cid, imp_preds, f"Improved preds missing common ID: {cid}")
            self.assertTrue(base_preds[cid].get("parsed_successfully"))
            self.assertTrue(imp_preds[cid].get("parsed_successfully"))

    def test_no_gold_label_leakage(self):
        """Ensure prediction records do not leak ground_truth or gold references."""
        common_cases, base_preds, imp_preds = load_and_validate_common_cases()

        for c in common_cases:
            cid = c["id"]
            b_rec = base_preds[cid]
            i_rec = imp_preds[cid]

            # Neither prediction object should contain gold fields
            self.assertNotIn("ground_truth", b_rec.get("prediction", {}))
            self.assertNotIn("ground_truth", i_rec.get("prediction", {}))
            self.assertNotIn("reference_final_answer", b_rec.get("prediction", {}))
            self.assertNotIn("reference_final_answer", i_rec.get("prediction", {}))

    def test_json_metrics_structure_and_completeness(self):
        """Verify results/objective28_comparison_187_metrics.json exists and is valid."""
        self.assertTrue(
            os.path.exists(DEFAULT_OUTPUT_METRICS_PATH),
            f"Metrics file missing: {DEFAULT_OUTPUT_METRICS_PATH}",
        )

        with open(DEFAULT_OUTPUT_METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        self.assertEqual(meta.get("canonical_dataset_size"), 200)
        self.assertEqual(meta.get("evaluated_cases_count"), 187)
        self.assertEqual(meta.get("excluded_cases_count"), 13)
        self.assertEqual(sorted(meta.get("excluded_cases", [])), sorted(list(EXCLUDED_CASE_IDS)))

        deltas = data.get("summary_deltas", {})
        expected_delta_keys = {
            "strict_accuracy_delta",
            "decisive_accuracy_delta",
            "decisive_coverage_delta",
            "uncertainty_rate_delta",
            "precision_delta",
            "conservative_recall_delta",
            "f1_delta",
            "step_localization_accuracy_delta",
            "false_positive_reduction",
            "false_negative_reduction",
        }
        self.assertTrue(expected_delta_keys.issubset(set(deltas.keys())))

        # Verify case-level breakdown
        bd = data.get("case_level_breakdown", {})
        self.assertEqual(bd.get("both_correct_count"), 181)
        self.assertEqual(bd.get("both_incorrect_count"), 6)
        self.assertEqual(bd.get("classification_disagreements_count"), 0)
        self.assertEqual(bd.get("step_localization_disagreements_on_tp_count"), 1)

        # Verify cohort coverage in both baseline and improved
        b_cohorts = data.get("baseline_metrics", {}).get("cohorts", {})
        i_cohorts = data.get("improved_metrics", {}).get("cohorts", {})
        expected_cohorts = {
            "sound_reasoning_none",
            "unsupported_conceptual",
            "conditional_warnings",
            "all_incorrect_cases",
        }
        self.assertEqual(set(b_cohorts.keys()), expected_cohorts)
        self.assertEqual(set(i_cohorts.keys()), expected_cohorts)

        # Verify category coverage
        b_cats = data.get("baseline_metrics", {}).get("incorrect_categories", {})
        i_cats = data.get("improved_metrics", {}).get("incorrect_categories", {})
        expected_cats = {
            "invalid_transformation",
            "arithmetic_error",
            "domain_violation",
            "incorrect_setup",
            "lost_solution",
            "division_by_zero",
        }
        self.assertEqual(set(b_cats.keys()), expected_cats)
        self.assertEqual(set(i_cats.keys()), expected_cats)

        # Verify symbolic limitations
        b_lim = data.get("baseline_metrics", {}).get("symbolic_limitations", {})
        i_lim = data.get("improved_metrics", {}).get("symbolic_limitations", {})
        self.assertEqual(b_lim.get("total_cases"), 10)
        self.assertEqual(i_lim.get("total_cases"), 10)

    def test_markdown_report_content(self):
        """Verify results/objective28_comparison_187_report.md exists and contains required sections."""
        self.assertTrue(
            os.path.exists(DEFAULT_OUTPUT_REPORT_PATH),
            f"Report file missing: {DEFAULT_OUTPUT_REPORT_PATH}",
        )

        with open(DEFAULT_OUTPUT_REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        required_sections = [
            "Executive Summary",
            "Evaluation Setup & Rigorous Controls",
            "Confusion Matrix Comparison",
            "Cohort Performance Comparison",
            "Injected Error Category Breakdown",
            "Symbolic Limitation Boundary Cases",
            "Deep Discrepancy & Failure Analysis",
            "Limitations & Scope Constraints",
            "Conclusion",
        ]
        for sec in required_sections:
            self.assertIn(sec, content, f"Report missing section: {sec}")

        # Ensure all 13 excluded IDs are mentioned
        for eid in EXCLUDED_CASE_IDS:
            self.assertIn(eid, content, f"Report should mention excluded ID: {eid}")

        # Ensure eval_196 discrepancy is analyzed
        self.assertIn("eval_196", content)
        self.assertIn("variable shadowing", content.lower())

    def test_mock_discrepancy_and_delta_computation(self):
        """Test build_comparison_analysis with synthetic cases for robust edge cases."""
        mock_cases = [
            {
                "id": "mock_001",
                "ground_truth": {"is_correct": True, "error_category": "none", "error_step": None},
            },
            {
                "id": "mock_002",
                "ground_truth": {"is_correct": False, "error_category": "arithmetic_error", "error_step": 1},
            },
        ]
        mock_base = {
            "mock_001": {"prediction": {"classification": "correct", "first_incorrect_step": None}},
            "mock_002": {"prediction": {"classification": "incorrect", "first_incorrect_step": 1}},
        }
        mock_imp = {
            "mock_001": {"prediction": {"classification": "incorrect", "first_incorrect_step": 1}},
            "mock_002": {"prediction": {"classification": "incorrect", "first_incorrect_step": 1}},
        }

        res = build_comparison_analysis(mock_cases, mock_base, mock_imp)
        bd = res["case_level_breakdown"]
        self.assertEqual(bd["both_correct_count"], 1)
        self.assertEqual(bd["baseline_correct_improved_incorrect_count"], 1)
        self.assertEqual(bd["classification_disagreements_count"], 1)
        self.assertEqual(res["summary_deltas"]["strict_accuracy_delta"], -0.5)


if __name__ == "__main__":
    unittest.main()
