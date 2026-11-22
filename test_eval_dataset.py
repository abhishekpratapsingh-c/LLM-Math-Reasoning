"""
test_eval_dataset.py
--------------------
Phase 4 - Objective 26: Automated Test Suite for Labeled Evaluation Dataset.

Performs rigorous unit tests validating:
1. Dataset file presence and valid JSONL structure.
2. Exact case count (200 cases) and sequential ID integrity.
3. Category presence and distribution balance (80 correct / 120 incorrect).
4. Error category representation across all required types:
   - arithmetic_error
   - invalid_transformation
   - lost_solution
   - division_by_zero
   - domain_violation
   - incorrect_setup
   - unsupported
   - conditional_warning
5. Schema compliance and non-null required fields.
6. Error step reference bounds (1 <= error_step <= total_steps).
7. Full compatibility with step_extractor.py (step count and headers).
8. Presence of symbolic limitation boundary cases.
9. Domain and difficulty distribution completeness.
"""

import json
import os
import re
import sys
import unittest
from collections import Counter

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_extractor import extract_reasoning_steps
from validate_dataset import (
    validate_dataset_file,
    ALLOWED_REASONING_TYPES,
    ALLOWED_ERROR_CATEGORIES,
    ALLOWED_REPORT_STATUSES,
    ALLOWED_DOMAINS,
    ALLOWED_DIFFICULTIES,
)


class TestEvalDataset(unittest.TestCase):
    """Test suite for the Phase 4 Objective 26 mathematical reasoning evaluation dataset."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "reasoning_eval_dataset.jsonl",
        )
        cls.cases = []
        if os.path.exists(cls.dataset_path):
            with open(cls.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        cls.cases.append(json.loads(line_str))

    def test_01_dataset_file_exists_and_parseable(self):
        """Verifies the dataset file exists and contains valid JSON lines."""
        self.assertTrue(os.path.exists(self.dataset_path), f"Dataset not found at {self.dataset_path}")
        self.assertEqual(len(self.cases), 200, f"Expected 200 cases, found {len(self.cases)}")

    def test_02_unique_sequential_ids(self):
        """Verifies all IDs are unique, formatted as eval_XXX, and span eval_001 to eval_200."""
        seen_ids = set()
        expected_ids = [f"eval_{i:03d}" for i in range(1, 201)]
        actual_ids = []

        for case in self.cases:
            case_id = case.get("id")
            self.assertNotIn(case_id, seen_ids, f"Duplicate ID found: {case_id}")
            seen_ids.add(case_id)
            actual_ids.append(case_id)

        self.assertEqual(actual_ids, expected_ids, "Dataset IDs must match exact sequence eval_001 through eval_200")

    def test_03_balanced_correct_and_incorrect_counts(self):
        """Verifies balance: 80 correct cases (40%) and 120 incorrect cases (60%)."""
        correct_cases = [c for c in self.cases if c["ground_truth"]["is_correct"]]
        incorrect_cases = [c for c in self.cases if not c["ground_truth"]["is_correct"]]

        self.assertEqual(len(correct_cases), 80, f"Expected 80 correct cases, found {len(correct_cases)}")
        self.assertEqual(len(incorrect_cases), 120, f"Expected 120 incorrect cases, found {len(incorrect_cases)}")
        self.assertEqual(len(correct_cases) + len(incorrect_cases), 200)

    def test_04_error_categories_representation(self):
        """Verifies all 6 core error categories plus unsupported and warnings are well-represented."""
        categories = Counter(c["ground_truth"]["error_category"] for c in self.cases)

        # Correct & boundary
        self.assertGreaterEqual(categories["none"], 50)
        self.assertGreaterEqual(categories["unsupported"], 10)
        self.assertGreaterEqual(categories["conditional_warning"], 10)

        # Injected error categories
        self.assertGreaterEqual(categories["arithmetic_error"], 20)
        self.assertGreaterEqual(categories["invalid_transformation"], 20)
        self.assertGreaterEqual(categories["lost_solution"], 15)
        self.assertGreaterEqual(categories["division_by_zero"], 15)
        self.assertGreaterEqual(categories["domain_violation"], 15)
        self.assertGreaterEqual(categories["incorrect_setup"], 15)

    def test_05_error_step_reference_validity(self):
        """Verifies that for every injected error, error_step points to an existing step number."""
        for case in self.cases:
            case_id = case["id"]
            gt = case["ground_truth"]
            err_step = gt["error_step"]
            step_labels = gt["step_labels"]
            num_steps = len(step_labels)

            if gt["is_correct"]:
                self.assertIsNone(err_step, f"[{case_id}] Correct case should have error_step=None")
            else:
                self.assertIsNotNone(err_step, f"[{case_id}] Incorrect case must have non-null error_step")
                self.assertIsInstance(err_step, int, f"[{case_id}] error_step must be an int")
                self.assertTrue(1 <= err_step <= num_steps, f"[{case_id}] error_step ({err_step}) out of range (1..{num_steps})")

    def test_06_step_extractor_compatibility(self):
        """Verifies that step_extractor extracts the exact number of steps declared in ground truth."""
        for case in self.cases:
            case_id = case["id"]
            raw_text = case["reasoning_text"]
            extracted = extract_reasoning_steps(raw_text)

            expected_steps = len(case["ground_truth"]["step_labels"])
            self.assertEqual(
                len(extracted.steps),
                expected_steps,
                f"[{case_id}] step_extractor found {len(extracted.steps)} steps, expected {expected_steps}",
            )
            self.assertIsNotNone(
                extracted.final_answer,
                f"[{case_id}] Final answer must be extracted from reasoning_text",
            )

    def test_07_reasoning_types_distribution(self):
        """Verifies both single-step and multi-step reasoning are present."""
        types = Counter(c["reasoning_type"] for c in self.cases)
        self.assertIn("single_step", types)
        self.assertIn("multi_step", types)
        self.assertGreater(types["single_step"], 30, "Single-step cases should be >= 30")
        self.assertGreater(types["multi_step"], 100, "Multi-step cases should be >= 100")

    def test_08_symbolic_limitation_cases(self):
        """Verifies exactly 10 cases explicitly target limitations of symbolic verification."""
        limitation_cases = [c for c in self.cases if c["metadata"]["symbolic_limitation"]]
        self.assertEqual(len(limitation_cases), 10, f"Expected 10 limitation cases, found {len(limitation_cases)}")

        for lc in limitation_cases:
            case_id = lc["id"]
            reason = lc["metadata"].get("limitation_reason")
            self.assertIsNotNone(reason, f"[{case_id}] Limitation case must provide limitation_reason")
            self.assertTrue(len(str(reason).strip()) > 10, f"[{case_id}] limitation_reason too brief")

    def test_09_domain_and_difficulty_distribution(self):
        """Verifies diverse coverage of domains and difficulty levels."""
        domains = Counter(c["metadata"]["domain"] for c in self.cases)
        difficulties = Counter(c["metadata"]["difficulty"] for c in self.cases)

        for d in ["arithmetic", "algebra", "word_problem", "geometry", "number_theory"]:
            self.assertIn(d, domains, f"Domain '{d}' must be present in dataset")
            self.assertGreater(domains[d], 0, f"Domain '{d}' count should be > 0")

        for diff in ["easy", "medium", "hard"]:
            self.assertIn(diff, difficulties, f"Difficulty '{diff}' must be present in dataset")
            self.assertGreater(difficulties[diff], 0, f"Difficulty '{diff}' count should be > 0")

    def test_10_full_validator_integration(self):
        """Runs the standalone validate_dataset_file function and asserts zero errors."""
        is_valid, errors, stats = validate_dataset_file(self.dataset_path)
        self.assertTrue(is_valid, f"Validator returned errors:\n" + "\n".join(errors[:20]))
        self.assertEqual(stats["total_cases"], 200)

    def test_11_case_and_step_level_consistency(self):
        """Verifies strict consistency between case-level correctness, step-level labels, and error category."""
        for case in self.cases:
            case_id = case["id"]
            gt = case["ground_truth"]
            is_corr = gt["is_correct"]
            err_cat = gt["error_category"]
            err_step = gt["error_step"]
            step_labels = gt["step_labels"]

            if is_corr:
                self.assertIsNone(err_step, f"[{case_id}] Correct case must have error_step=None")
                self.assertIn(err_cat, ("none", "unsupported", "conditional_warning"))
                for s in step_labels:
                    self.assertTrue(s["is_correct"], f"[{case_id}] Step {s['step_number']} marked incorrect in correct case")
            else:
                self.assertIsNotNone(err_step, f"[{case_id}] Incorrect case must have error_step")
                self.assertNotEqual(err_cat, "none", f"[{case_id}] Incorrect case cannot have error_category 'none'")
                first_incorrect = next((s for s in step_labels if not s["is_correct"]), None)
                self.assertIsNotNone(first_incorrect, f"[{case_id}] Incorrect case has no step marked is_correct=False")
                self.assertEqual(
                    err_step,
                    first_incorrect["step_number"],
                    f"[{case_id}] error_step ({err_step}) does not match first incorrect step ({first_incorrect['step_number']})"
                )
                flagged_step = step_labels[err_step - 1]
                self.assertEqual(
                    err_cat,
                    flagged_step["error_category"],
                    f"[{case_id}] Case error_category '{err_cat}' != Step {err_step} error_category '{flagged_step['error_category']}'"
                )

    def test_12_expected_report_status_mapping(self):
        """Verifies that expected_report_status strictly maps to correctness and error category."""
        for case in self.cases:
            case_id = case["id"]
            gt = case["ground_truth"]
            is_corr = gt["is_correct"]
            err_cat = gt["error_category"]
            status = gt["expected_report_status"]

            if not is_corr:
                self.assertEqual(status, "errors_detected", f"[{case_id}] Incorrect case must have status 'errors_detected'")
            else:
                if err_cat == "none":
                    self.assertEqual(status, "verified", f"[{case_id}] Sound case must have status 'verified'")
                elif err_cat == "unsupported":
                    self.assertEqual(status, "partially_verified", f"[{case_id}] Unsupported case must have status 'partially_verified'")
                elif err_cat == "conditional_warning":
                    self.assertEqual(status, "warnings", f"[{case_id}] Conditional case must have status 'warnings'")

    def test_13_no_duplicate_problems_or_reasoning(self):
        """Verifies that all problems and reasoning traces in the dataset are distinct and unique."""
        seen_problems = {}
        seen_traces = {}

        for case in self.cases:
            case_id = case["id"]
            prob = case["problem"].strip().lower()
            trace = case["reasoning_text"].strip()

            self.assertNotIn(prob, seen_problems, f"Duplicate problem found in {case_id} and {seen_problems.get(prob)}")
            seen_problems[prob] = case_id

            self.assertNotIn(trace, seen_traces, f"Duplicate reasoning trace found in {case_id} and {seen_traces.get(trace)}")
            seen_traces[trace] = case_id

    def test_14_eval_191_circular_reasoning_distinction(self):
        """Verifies that eval_191 distinguishes algebraic transformation validity from proof logic and circular setup."""
        case_191 = next((c for c in self.cases if c["id"] == "eval_191"), None)
        self.assertIsNotNone(case_191, "eval_191 must be present in dataset")
        gt = case_191["ground_truth"]

        # 1. Overall proof is logically invalid
        self.assertFalse(gt["is_correct"], "eval_191 overall proof is logically invalid")
        self.assertEqual(gt["expected_report_status"], "errors_detected")

        # 2. Flaw is in the setup (begging the question: assuming what needs to be proved)
        self.assertEqual(gt["error_step"], 1)
        self.assertEqual(gt["error_category"], "incorrect_setup", "eval_191 must be classified as incorrect_setup, not invalid_transformation")
        self.assertEqual(gt["error_sub_type"], "circular_reasoning_begging_the_question")
        self.assertIn("circular reasoning", gt["step_labels"][0]["notes"].lower())
        self.assertFalse(gt["step_labels"][0]["is_correct"])

        # 3. Subsequent transitions (Steps 2 & 3) are algebraically valid transformations
        self.assertTrue(gt["step_labels"][1]["is_correct"], "Step 2 (x=5 -> x+1=6) is an algebraically valid transformation")
        self.assertIsNone(gt["step_labels"][1]["error_category"])
        self.assertIn("algebraically valid transformation", gt["step_labels"][1]["notes"])

        self.assertTrue(gt["step_labels"][2]["is_correct"], "Step 3 (x+1=6 -> x=5) is an algebraically valid transformation")
        self.assertIsNone(gt["step_labels"][2]["error_category"])
        self.assertIn("algebraically valid transformation", gt["step_labels"][2]["notes"])


if __name__ == "__main__":
    unittest.main()
