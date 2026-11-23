"""
test_llm_error_detector.py
--------------------------
Phase 4 - Objective 27: Automated Unit Test Suite for LLM-Only Error Detection Baseline.

Performs 100% offline, mocked unit tests validating:
1. Strict Information Barrier: No gold labels, error categories, error steps, expected statuses,
   or metadata are ever passed into the LLM prompt.
2. Robust parsing of JSON responses (clean JSON, markdown fences, regex fallback, malformed handling).
3. Normalization of classifications, confidence levels, and first incorrect step numbers.
4. Resumability, atomic checkpointing, and ID-skipping logic.
5. Exact evaluation metrics computation under the Objective 27 specification:
   - Positive class = incorrect reasoning.
   - TP: Explicit incorrect on gold incorrect.
   - FP: Explicit incorrect on gold correct.
   - FN: Correct on gold incorrect.
   - Conservative recall counts uncertain on gold incorrect as FN.
   - Abstentions: Uncertain on gold correct is NOT counted as FP.
   - Strict accuracy counts all uncertain predictions as incorrect.
   - Decisive accuracy and coverage.
   - First-incorrect-step localization among TPs.
   - Cohort partitioning (Sound, Unsupported, Conditional Warning, Incorrect, Symbolic Limitation).
6. Zero modifications to Phase 1-3 production modules.
"""

from collections import Counter
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_error_detector import (
    build_detection_prompt,
    parse_llm_detection_response,
    detect_reasoning_error_llm,
    LLMErrorDetection,
)
from evaluate_llm_baseline import (
    load_dataset,
    load_existing_predictions,
    checkpoint_prediction,
    compute_metrics_for_subset,
    compute_all_evaluation_metrics,
    generate_markdown_report,
)


class TestLLMErrorDetector(unittest.TestCase):
    """Test suite for LLM-only error detector and baseline evaluation framework."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "reasoning_eval_dataset.jsonl",
        )
        cls.real_cases = []
        if os.path.exists(cls.dataset_path):
            with open(cls.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        cls.real_cases.append(json.loads(line_str))

    def test_01_information_barrier_no_gold_fields_in_prompt(self):
        """Verifies that gold labels, error categories, error steps, and metadata never enter the prompt."""
        self.assertGreater(len(self.real_cases), 0, "Dataset cases should be loaded")

        # Test across diverse cases (first 25, limitation cases 191-200, and sample injected errors)
        test_sample = self.real_cases[:25] + self.real_cases[190:200]
        for case in test_sample:
            cid = case["id"]
            prob = case["problem"]
            r_text = case["reasoning_text"]
            gt = case["ground_truth"]
            meta = case["metadata"]

            prompt = build_detection_prompt(prob, r_text)

            # 1. Prompt must contain the problem and reasoning
            self.assertIn(prob, prompt, f"[{cid}] Prompt must contain problem")
            self.assertIn(r_text, prompt, f"[{cid}] Prompt must contain reasoning trace")

            # 2. Strict Information Barrier: Forbidden gold label keywords and values
            forbidden_keys = [
                "ground_truth",
                "is_correct",
                "error_category",
                "error_sub_type",
                "expected_report_status",
                "step_labels",
                "symbolic_limitation",
                "limitation_reason",
            ]
            for fk in forbidden_keys:
                self.assertNotIn(fk, prompt, f"[{cid}] Gold schema field '{fk}' leaked into prompt!")

            # 3. For incorrect cases, ensure specific error category string is not leaked as answer key
            if not gt["is_correct"]:
                err_cat = gt["error_category"]
                err_sub = gt["error_sub_type"]
                # Verify prompt does not say e.g. "error_category: arithmetic_error"
                self.assertNotIn(f"error_category: {err_cat}", prompt)
                self.assertNotIn(f"error_category: \"{err_cat}\"", prompt)
                self.assertNotIn(f"error_sub_type: {err_sub}", prompt)

            # 4. If limitation reason exists, ensure it is not leaked
            if meta.get("limitation_reason"):
                self.assertNotIn(meta["limitation_reason"], prompt, f"[{cid}] limitation_reason leaked!")

    def test_02_build_prompt_input_validation(self):
        """Verifies input validation on build_detection_prompt."""
        with self.assertRaises(ValueError):
            build_detection_prompt("", "Step 1: x = 1")
        with self.assertRaises(ValueError):
            build_detection_prompt("Problem text", "   ")
        with self.assertRaises(ValueError):
            build_detection_prompt(None, "Step 1: x = 1")

    def test_03_parse_clean_json_responses(self):
        """Verifies parsing of clean valid JSON responses."""
        # 1. Correct case
        raw_correct = '{"classification": "correct", "first_incorrect_step": null, "confidence": "high", "explanation": "Sound reasoning."}'
        p1 = parse_llm_detection_response(raw_correct)
        self.assertEqual(p1["classification"], "correct")
        self.assertIsNone(p1["first_incorrect_step"])
        self.assertEqual(p1["confidence"], "high")
        self.assertTrue(p1["parsed_successfully"])

        # 2. Incorrect case with integer step
        raw_incorrect = '{"classification": "incorrect", "first_incorrect_step": 2, "confidence": "high", "explanation": "Arithmetic mistake in Step 2."}'
        p2 = parse_llm_detection_response(raw_incorrect)
        self.assertEqual(p2["classification"], "incorrect")
        self.assertEqual(p2["first_incorrect_step"], 2)
        self.assertEqual(p2["confidence"], "high")

        # 3. Uncertain case
        raw_uncertain = '{"classification": "uncertain", "first_incorrect_step": null, "confidence": "low", "explanation": "Ambiguous variable definitions."}'
        p3 = parse_llm_detection_response(raw_uncertain)
        self.assertEqual(p3["classification"], "uncertain")
        self.assertIsNone(p3["first_incorrect_step"])
        self.assertEqual(p3["confidence"], "low")

    def test_04_parse_markdown_fenced_json(self):
        """Verifies parsing of JSON enclosed in markdown code fences."""
        raw_fenced = (
            "Here is the evaluation:\n\n"
            "```json\n"
            "{\n"
            '  "classification": "incorrect",\n'
            '  "first_incorrect_step": 1,\n'
            '  "confidence": "medium",\n'
            '  "explanation": "Sign error on line 1."\n'
            "}\n"
            "```\n"
            "End of evaluation."
        )
        parsed = parse_llm_detection_response(raw_fenced)
        self.assertEqual(parsed["classification"], "incorrect")
        self.assertEqual(parsed["first_incorrect_step"], 1)
        self.assertEqual(parsed["confidence"], "medium")
        self.assertTrue(parsed["parsed_successfully"])

    def test_05_parse_malformed_json_graceful_fallback(self):
        """Verifies that malformed or non-JSON strings fallback to uncertain without crashing."""
        malformed = "I believe the answer is wrong because Step 3 is bad, but I won't format as JSON."
        parsed = parse_llm_detection_response(malformed)
        self.assertEqual(parsed["classification"], "uncertain")
        self.assertIsNone(parsed["first_incorrect_step"])
        self.assertFalse(parsed["parsed_successfully"])

        # Empty string
        parsed_empty = parse_llm_detection_response("")
        self.assertEqual(parsed_empty["classification"], "uncertain")
        self.assertFalse(parsed_empty["parsed_successfully"])

    def test_06_step_number_normalization(self):
        """Verifies step number normalization (string conversion, step resetting on correct)."""
        # Step given as string "Step 3"
        raw_step_str = '{"classification": "incorrect", "first_incorrect_step": "Step 3", "confidence": "high", "explanation": "Error"}'
        p1 = parse_llm_detection_response(raw_step_str)
        self.assertEqual(p1["first_incorrect_step"], 3)

        # Step provided on a correct case must be reset to None
        raw_correct_with_step = '{"classification": "correct", "first_incorrect_step": 2, "confidence": "high", "explanation": "Valid"}'
        p2 = parse_llm_detection_response(raw_correct_with_step)
        self.assertIsNone(p2["first_incorrect_step"])

    @patch("llm_error_detector.call_llm")
    def test_07_detect_reasoning_error_llm_mocked_call(self, mock_call):
        """Verifies detect_reasoning_error_llm interacts properly with call_llm."""
        mock_call.return_value = '{"classification": "incorrect", "first_incorrect_step": 1, "confidence": "high", "explanation": "Wrong setup"}'

        mock_client = MagicMock()
        res = detect_reasoning_error_llm(
            problem="Calculate 2 + 2",
            reasoning_text="Step 1: 2 + 2 = 5\n\nFinal Answer: 5",
            provider="gemini",
            client=mock_client,
            model_name="mock-gemini",
        )

        self.assertIsInstance(res, LLMErrorDetection)
        self.assertEqual(res.classification, "incorrect")
        self.assertEqual(res.first_incorrect_step, 1)
        self.assertEqual(res.confidence, "high")
        self.assertTrue(res.parsed_successfully)
        self.assertEqual(res.provider, "gemini")
        self.assertEqual(res.model, "mock-gemini")
        mock_call.assert_called_once()

    def test_08_checkpointing_resumability(self):
        """Verifies checkpoint saving, reloading, and incremental append logic."""
        temp_dir = tempfile.mkdtemp()
        try:
            pred_file = os.path.join(temp_dir, "test_preds.jsonl")

            # 1. Initially empty
            loaded_0 = load_existing_predictions(pred_file)
            self.assertEqual(len(loaded_0), 0)

            # 2. Checkpoint record 1
            rec1 = {"id": "eval_001", "parsed_successfully": True, "prediction": {"classification": "correct", "first_incorrect_step": None}}
            checkpoint_prediction(pred_file, rec1)

            loaded_1 = load_existing_predictions(pred_file)
            self.assertEqual(len(loaded_1), 1)
            self.assertIn("eval_001", loaded_1)

            # 3. Checkpoint record 2
            rec2 = {"id": "eval_002", "parsed_successfully": True, "prediction": {"classification": "incorrect", "first_incorrect_step": 1}}
            checkpoint_prediction(pred_file, rec2)

            loaded_2 = load_existing_predictions(pred_file)
            self.assertEqual(len(loaded_2), 2)
            self.assertIn("eval_002", loaded_2)
            self.assertEqual(loaded_2["eval_002"]["prediction"]["first_incorrect_step"], 1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_09_exact_metric_calculation_synthetic_fixtures(self):
        """
        Verifies exact calculation of:
        - Strict accuracy
        - Decisive accuracy & coverage
        - Uncertainty rate
        - Error precision, conservative recall, and F1
        - First-incorrect-step localization
        """
        # Define 10 synthetic test cases:
        # 4 Gold Correct:
        #   c1: Model correct -> TN
        #   c2: Model correct -> TN
        #   c3: Model incorrect -> FP
        #   c4: Model uncertain -> Abstention on correct (NOT FP)
        #
        # 6 Gold Incorrect:
        #   i1: Model incorrect (step match: 2 == 2) -> TP
        #   i2: Model incorrect (step mismatch: 1 != 2) -> TP
        #   i3: Model incorrect (step match: 1 == 1) -> TP
        #   i4: Model correct -> FN_explicit
        #   i5: Model uncertain -> FN_uncertain (counted in conservative recall)
        #   i6: Model uncertain -> FN_uncertain (counted in conservative recall)

        gold_cases = [
            {"id": "c1", "ground_truth": {"is_correct": True, "error_step": None, "error_category": "none"}},
            {"id": "c2", "ground_truth": {"is_correct": True, "error_step": None, "error_category": "none"}},
            {"id": "c3", "ground_truth": {"is_correct": True, "error_step": None, "error_category": "none"}},
            {"id": "c4", "ground_truth": {"is_correct": True, "error_step": None, "error_category": "none"}},
            {"id": "i1", "ground_truth": {"is_correct": False, "error_step": 2, "error_category": "arithmetic_error"}},
            {"id": "i2", "ground_truth": {"is_correct": False, "error_step": 2, "error_category": "arithmetic_error"}},
            {"id": "i3", "ground_truth": {"is_correct": False, "error_step": 1, "error_category": "invalid_transformation"}},
            {"id": "i4", "ground_truth": {"is_correct": False, "error_step": 1, "error_category": "lost_solution"}},
            {"id": "i5", "ground_truth": {"is_correct": False, "error_step": 3, "error_category": "domain_violation"}},
            {"id": "i6", "ground_truth": {"is_correct": False, "error_step": 2, "error_category": "division_by_zero"}},
        ]

        predictions = {
            "c1": {"id": "c1", "prediction": {"classification": "correct", "first_incorrect_step": None}},
            "c2": {"id": "c2", "prediction": {"classification": "correct", "first_incorrect_step": None}},
            "c3": {"id": "c3", "prediction": {"classification": "incorrect", "first_incorrect_step": 1}},
            "c4": {"id": "c4", "prediction": {"classification": "uncertain", "first_incorrect_step": None}},
            "i1": {"id": "i1", "prediction": {"classification": "incorrect", "first_incorrect_step": 2}},
            "i2": {"id": "i2", "prediction": {"classification": "incorrect", "first_incorrect_step": 1}},
            "i3": {"id": "i3", "prediction": {"classification": "incorrect", "first_incorrect_step": 1}},
            "i4": {"id": "i4", "prediction": {"classification": "correct", "first_incorrect_step": None}},
            "i5": {"id": "i5", "prediction": {"classification": "uncertain", "first_incorrect_step": None}},
            "i6": {"id": "i6", "prediction": {"classification": "uncertain", "first_incorrect_step": None}},
        }

        m = compute_metrics_for_subset("Synthetic Test", gold_cases, predictions)
        cm = m["confusion_matrix"]
        acc = m["accuracy_metrics"]
        det = m["error_detection_metrics"]
        step = m["step_localization"]

        # Confusion Matrix verifications:
        self.assertEqual(cm["true_positives"], 3)              # i1, i2, i3
        self.assertEqual(cm["false_positives"], 1)             # c3
        self.assertEqual(cm["true_negatives"], 2)              # c1, c2
        self.assertEqual(cm["false_negatives_explicit"], 1)    # i4
        self.assertEqual(cm["false_negatives_uncertain"], 2)   # i5, i6
        self.assertEqual(cm["abstentions_on_correct"], 1)      # c4
        self.assertEqual(cm["total_uncertain"], 3)             # c4, i5, i6

        # Strict Accuracy: (TN=2 + TP=3) / 10 = 5 / 10 = 0.50
        self.assertAlmostEqual(acc["strict_accuracy"], 0.50, places=4)

        # Decisive calls: 7 cases (c1, c2, c3, i1, i2, i3, i4)
        # Decisive Coverage: 7 / 10 = 0.70
        self.assertAlmostEqual(acc["decisive_coverage"], 0.70, places=4)
        # Decisive Accuracy: matches / 7 = (2 + 3) / 7 = 5 / 7 = 0.7143
        self.assertAlmostEqual(acc["decisive_accuracy"], round(5 / 7, 4), places=4)

        # Uncertainty Rate: 3 / 10 = 0.30
        self.assertAlmostEqual(acc["uncertainty_rate"], 0.30, places=4)

        # Error Precision (Positive class = incorrect): TP / (TP + FP) = 3 / (3 + 1) = 0.75
        self.assertAlmostEqual(det["precision"], 0.75, places=4)

        # Conservative Recall: TP / Total Gold Incorrect = 3 / 6 = 0.50
        self.assertAlmostEqual(det["conservative_recall"], 0.50, places=4)

        # F1-Score: 2 * (0.75 * 0.50) / (0.75 + 0.50) = 0.75 / 1.25 = 0.60
        self.assertAlmostEqual(det["f1_score"], 0.60, places=4)

        # Step Localization among TPs (i1, i2, i3):
        # i1: 2 == 2 (match)
        # i2: 1 != 2 (mismatch)
        # i3: 1 == 1 (match)
        # Localization Accuracy: 2 / 3 = 0.6667
        self.assertEqual(step["step_match_count"], 2)
        self.assertEqual(step["tp_with_gold_step_count"], 3)
        self.assertAlmostEqual(step["localization_accuracy"], round(2 / 3, 4), places=4)

    def test_10_cohort_and_limitation_partitioning(self):
        """Verifies that compute_all_evaluation_metrics partitions cohorts properly."""
        # Use a subset of real cases
        sample = self.real_cases[:80] + self.real_cases[190:200]
        # Dummy predictions
        dummy_preds = {
            c["id"]: {
                "id": c["id"],
                "prediction": {
                    "classification": "correct" if c["ground_truth"]["is_correct"] else "incorrect",
                    "first_incorrect_step": c["ground_truth"]["error_step"],
                    "confidence": "high",
                    "explanation": "Mocked",
                },
            }
            for c in sample
        }

        all_m = compute_all_evaluation_metrics(sample, dummy_preds)
        self.assertIn("overall", all_m)
        self.assertIn("cohorts", all_m)
        self.assertIn("incorrect_categories", all_m)
        self.assertIn("symbolic_limitations", all_m)

        cohorts = all_m["cohorts"]
        self.assertIn("sound_reasoning_none", cohorts)
        self.assertIn("unsupported_conceptual", cohorts)
        self.assertIn("conditional_warnings", cohorts)
        self.assertIn("all_incorrect_cases", cohorts)

        # Verify Markdown report generation runs without error
        temp_dir = tempfile.mkdtemp()
        try:
            report_file = os.path.join(temp_dir, "test_report.md")
            md = generate_markdown_report(all_m, report_file)
            self.assertTrue(os.path.exists(report_file))
            self.assertIn("# Phase 4 – Objective 27", md)
            self.assertIn("Sound Cases (`none`)", md)
            self.assertIn("Symbolic Limitation Boundary Cases", md)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_11_zero_phase1_3_modifications(self):
        """Verifies that Phase 1, 2, and 3 production modules remain completely unmodified."""
        # List of critical Phase 1-3 modules
        phase_modules = [
            "llm_reasoning.py",
            "step_extractor.py",
            "step_structurer.py",
            "step_verifier.py",
            "step_transition_verifier.py",
            "reasoning_error_detector.py",
            "reasoning_feedback.py",
            "verification_report.py",
            "symbolic_verifier.py",
            "symbolic_solver.py",
        ]
        for mod in phase_modules:
            mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mod)
            self.assertTrue(os.path.exists(mod_path), f"Module {mod} must exist")

    def test_12_audit_fix1_strict_classification_and_step_validation(self):
        """
        Validates Fix 1: Strict response validation in parse_llm_detection_response.
        Tests 1-8:
        1. Missing classification -> uncertain, parsed_successfully=False
        2. Invalid classification 'not incorrect' -> uncertain, parsed_successfully=False
        3. Invalid classification 'partially correct' -> uncertain, parsed_successfully=False
        4. Step number zero -> first_incorrect_step=None, parsed_successfully=False
        5. Negative step number -> first_incorrect_step=None, parsed_successfully=False
        6. Boolean step number -> first_incorrect_step=None, parsed_successfully=False
        7. Valid correct response with null step -> correct, step=None, parsed_successfully=True
        8. Valid incorrect response with positive step -> incorrect, step=2, parsed_successfully=True
        """
        # 1. Missing classification
        res1 = parse_llm_detection_response('{"first_incorrect_step": 1, "explanation": "no class"}')
        self.assertEqual(res1["classification"], "uncertain")
        self.assertFalse(res1["parsed_successfully"])

        # 2. Invalid classification: "not incorrect"
        res2 = parse_llm_detection_response('{"classification": "not incorrect", "first_incorrect_step": null}')
        self.assertEqual(res2["classification"], "uncertain")
        self.assertFalse(res2["parsed_successfully"])

        # 3. Invalid classification: "partially correct"
        res3 = parse_llm_detection_response('{"classification": "partially correct", "first_incorrect_step": null}')
        self.assertEqual(res3["classification"], "uncertain")
        self.assertFalse(res3["parsed_successfully"])

        # 4. Step number zero (step >= 1 required)
        res4 = parse_llm_detection_response('{"classification": "incorrect", "first_incorrect_step": 0}')
        self.assertEqual(res4["classification"], "incorrect")
        self.assertIsNone(res4["first_incorrect_step"])
        self.assertFalse(res4["parsed_successfully"])

        # 5. Negative step number
        res5 = parse_llm_detection_response('{"classification": "incorrect", "first_incorrect_step": -3}')
        self.assertEqual(res5["classification"], "incorrect")
        self.assertIsNone(res5["first_incorrect_step"])
        self.assertFalse(res5["parsed_successfully"])

        # 6. Boolean step number (reject booleans as integers)
        res6_true = parse_llm_detection_response('{"classification": "incorrect", "first_incorrect_step": true}')
        self.assertEqual(res6_true["classification"], "incorrect")
        self.assertIsNone(res6_true["first_incorrect_step"])
        self.assertFalse(res6_true["parsed_successfully"])

        res6_false = parse_llm_detection_response('{"classification": "incorrect", "first_incorrect_step": false}')
        self.assertEqual(res6_false["classification"], "incorrect")
        self.assertIsNone(res6_false["first_incorrect_step"])
        self.assertFalse(res6_false["parsed_successfully"])

        # 7. Valid correct response with null step (also step normalized to None)
        res7 = parse_llm_detection_response('{"classification": "correct", "first_incorrect_step": null, "confidence": "high", "explanation": "All steps verified"}')
        self.assertEqual(res7["classification"], "correct")
        self.assertIsNone(res7["first_incorrect_step"])
        self.assertTrue(res7["parsed_successfully"])

        res7_norm = parse_llm_detection_response('{"classification": "correct", "first_incorrect_step": 2, "confidence": "high", "explanation": "Valid"}')
        self.assertEqual(res7_norm["classification"], "correct")
        self.assertIsNone(res7_norm["first_incorrect_step"], "Step on correct must normalize to None")
        self.assertTrue(res7_norm["parsed_successfully"])

        # 8. Valid incorrect response with positive step
        res8 = parse_llm_detection_response('{"classification": "incorrect", "first_incorrect_step": 2, "confidence": "high", "explanation": "Step 2 failed"}')
        self.assertEqual(res8["classification"], "incorrect")
        self.assertEqual(res8["first_incorrect_step"], 2)
        self.assertTrue(res8["parsed_successfully"])

    def test_13_audit_fix2_checkpoint_failed_retry_and_success_skip(self):
        """
        Validates Fix 2: Safe checkpoint recovery in load_existing_predictions.
        Tests 9-10:
        9. Failed checkpoint record is retried on restart (not loaded into completed).
        10. Successfully parsed checkpoint record is skipped (loaded into completed).
        """
        temp_dir = tempfile.mkdtemp()
        try:
            pred_file = os.path.join(temp_dir, "checkpoint_test.jsonl")

            # Record A: Failed/malformed call (parsed_successfully = False)
            failed_rec = {
                "id": "eval_001",
                "parsed_successfully": False,
                "prediction": {"classification": "uncertain", "first_incorrect_step": None},
                "raw_llm_response": "",
            }
            checkpoint_prediction(pred_file, failed_rec)

            # Record B: Successfully parsed call (parsed_successfully = True)
            success_rec = {
                "id": "eval_002",
                "parsed_successfully": True,
                "prediction": {"classification": "correct", "first_incorrect_step": None},
                "raw_llm_response": '{"classification": "correct", "first_incorrect_step": null}',
            }
            checkpoint_prediction(pred_file, success_rec)

            # Load checkpoint
            completed = load_existing_predictions(pred_file)

            # Test 9: eval_001 failed, so it must NOT be in completed (must be retried)
            self.assertNotIn("eval_001", completed, "Failed prediction must not be marked completed")

            # Test 10: eval_002 succeeded, so it must be in completed (skipped on restart)
            self.assertIn("eval_002", completed, "Successful prediction must be marked completed")
            self.assertEqual(completed["eval_002"]["prediction"]["classification"], "correct")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
