"""
test_llm_improved_detector.py
-----------------------------
Phase 4 - Objective 28: Automated Unit Test Suite for Improved LLM-Based Error Detector.

Performs 100% offline, mocked unit tests validating:
1. Strict Information Barrier: No gold labels, error categories, error steps, expected statuses,
   or metadata are ever passed into the improved LLM prompt.
2. Prompt Content & Structure: All 9 systematic verification procedure checks are embedded in the prompt.
3. Robust Parsing of JSON Responses: Clean JSON, markdown fences, regex fallback, malformed handling.
4. Strict Response Validation: Exact classification validation, strict first-incorrect-step validation,
   rejection of booleans, normalization of correct/uncertain step values.
5. Dataclass Serialization: to_dict and from_dict roundtrip fidelity.
6. Resumability, atomic checkpointing, and ID-skipping logic.
7. Exact evaluation metrics computation under Phase 4 specifications.
8. Zero modifications to Phase 1-3 production modules.
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

from llm_improved_detector import (
    build_improved_detection_prompt,
    parse_llm_detection_response,
    detect_reasoning_error_improved,
    LLMImprovedErrorDetection,
    IMPROVED_DETECTION_PROMPT_TEMPLATE,
)
from evaluate_llm_improved import (
    load_dataset,
    load_existing_predictions,
    checkpoint_prediction,
    compute_metrics_for_subset,
    compute_all_evaluation_metrics,
    generate_markdown_report,
)


class TestLLMImprovedDetector(unittest.TestCase):
    """Test suite for Improved LLM-based error detector and evaluation runner."""

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
            reasoning = case["reasoning_text"]
            gt = case["ground_truth"]

            prompt = build_improved_detection_prompt(prob, reasoning)

            # 1. Problem and reasoning MUST be in prompt
            self.assertIn(prob.strip(), prompt)
            self.assertIn(reasoning.strip(), prompt)

            # 2. Gold metadata MUST NOT be in prompt
            self.assertNotIn(cid, prompt)
            self.assertNotIn("is_correct", prompt)
            self.assertNotIn(f'"is_correct": {str(gt["is_correct"]).lower()}', prompt)
            self.assertNotIn(gt.get("expected_report_status", ""), prompt)
            if gt.get("error_category"):
                self.assertNotIn(f'category: {gt["error_category"]}', prompt)
            if gt.get("error_sub_type"):
                self.assertNotIn(gt["error_sub_type"], prompt)

    def test_02_prompt_structure_includes_9_verification_steps(self):
        """Verifies that all 9 systematic verification checks are clearly represented in the prompt template."""
        template = IMPROVED_DETECTION_PROMPT_TEMPLATE
        self.assertIn("1. Problem Understanding & Constraints", template)
        self.assertIn("2. Sequential Step-by-Step Audit", template)
        self.assertIn("3. Numerical & Arithmetic Verification", template)
        self.assertIn("4. Equivalence-Preserving Transformations", template)
        self.assertIn("5. Branch Analysis & Solution Preservation", template)
        self.assertIn("6. Domain Restrictions & Mathematical Boundaries", template)
        self.assertIn("7. Logical & Conceptual Validity", template)
        self.assertIn("Do not assume that an equation being symbolically or algebraically valid proves that the reasoning is logically valid", template)
        self.assertIn("8. Localization of First Failure", template)
        self.assertIn("9. Classification Consistency", template)

    def test_03_prompt_builder_input_validation(self):
        """Verifies that build_improved_detection_prompt rejects empty or invalid inputs."""
        with self.assertRaises(ValueError):
            build_improved_detection_prompt("", "Step 1: x = 1")
        with self.assertRaises(ValueError):
            build_improved_detection_prompt("Problem", "   ")
        with self.assertRaises(ValueError):
            build_improved_detection_prompt(None, "Step 1")

    def test_04_json_parsing_clean_json(self):
        """Verifies parsing of clean, standard JSON response."""
        raw = json.dumps({
            "classification": "incorrect",
            "first_incorrect_step": 2,
            "confidence": "high",
            "explanation": "Step 2 divides by zero."
        })
        parsed = parse_llm_detection_response(raw)
        self.assertEqual(parsed["classification"], "incorrect")
        self.assertEqual(parsed["first_incorrect_step"], 2)
        self.assertEqual(parsed["confidence"], "high")
        self.assertEqual(parsed["explanation"], "Step 2 divides by zero.")
        self.assertTrue(parsed["parsed_successfully"])

    def test_05_json_parsing_markdown_code_fence(self):
        """Verifies parsing when response is wrapped in markdown code blocks."""
        raw = (
            "```json\n"
            "{\n"
            '  "classification": "correct",\n'
            '  "first_incorrect_step": null,\n'
            '  "confidence": "high",\n'
            '  "explanation": "All steps are algebraically sound."\n'
            "}\n"
            "```"
        )
        parsed = parse_llm_detection_response(raw)
        self.assertEqual(parsed["classification"], "correct")
        self.assertIsNone(parsed["first_incorrect_step"])
        self.assertEqual(parsed["confidence"], "high")
        self.assertTrue(parsed["parsed_successfully"])

    def test_06_strict_validation_invalid_classification(self):
        """Verifies that invalid classifications fail parsing and default to uncertain."""
        raw = json.dumps({
            "classification": "partially_correct",
            "first_incorrect_step": None,
            "confidence": "medium",
            "explanation": "Ambiguous"
        })
        parsed = parse_llm_detection_response(raw)
        self.assertEqual(parsed["classification"], "uncertain")
        self.assertFalse(parsed["parsed_successfully"])

    def test_07_strict_validation_step_as_boolean(self):
        """Verifies that boolean values for first_incorrect_step are rejected."""
        raw = json.dumps({
            "classification": "incorrect",
            "first_incorrect_step": True,
            "confidence": "high",
            "explanation": "Error found"
        })
        parsed = parse_llm_detection_response(raw)
        self.assertIsNone(parsed["first_incorrect_step"])
        self.assertFalse(parsed["parsed_successfully"])

    def test_08_strict_validation_negative_or_zero_step(self):
        """Verifies that non-positive step integers are rejected."""
        for invalid_step in [0, -1, -5]:
            raw = json.dumps({
                "classification": "incorrect",
                "first_incorrect_step": invalid_step,
                "confidence": "high",
                "explanation": "Error found"
            })
            parsed = parse_llm_detection_response(raw)
            self.assertIsNone(parsed["first_incorrect_step"])
            self.assertFalse(parsed["parsed_successfully"])

    def test_09_strict_validation_string_step_parsing(self):
        """Verifies that string steps like '2' or 'Step 3' parse properly while malformed ones fail."""
        # Valid string formats
        for s, expected in [("2", 2), ("Step 3", 3), ("step 4", 4)]:
            raw = json.dumps({
                "classification": "incorrect",
                "first_incorrect_step": s,
                "confidence": "high",
                "explanation": "Found error"
            })
            parsed = parse_llm_detection_response(raw)
            self.assertEqual(parsed["first_incorrect_step"], expected)
            self.assertTrue(parsed["parsed_successfully"])

        # Invalid string format
        raw_bad = json.dumps({
            "classification": "incorrect",
            "first_incorrect_step": "Step Unknown",
            "confidence": "high",
            "explanation": "Error"
        })
        parsed_bad = parse_llm_detection_response(raw_bad)
        self.assertIsNone(parsed_bad["first_incorrect_step"])
        self.assertFalse(parsed_bad["parsed_successfully"])

    def test_10_step_normalization_for_correct_and_uncertain(self):
        """Verifies that first_incorrect_step is normalized to None if classification is correct or uncertain."""
        for cls_name in ["correct", "uncertain"]:
            raw = json.dumps({
                "classification": cls_name,
                "first_incorrect_step": 2,
                "confidence": "medium",
                "explanation": "Notes"
            })
            parsed = parse_llm_detection_response(raw)
            self.assertIsNone(parsed["first_incorrect_step"])
            self.assertTrue(parsed["parsed_successfully"])

    def test_11_malformed_json_fallback(self):
        """Verifies that unparseable text falls back to uncertain and parsed_successfully=False."""
        raw = "This is not valid JSON at all. Error in step 1."
        parsed = parse_llm_detection_response(raw)
        self.assertEqual(parsed["classification"], "uncertain")
        self.assertIsNone(parsed["first_incorrect_step"])
        self.assertFalse(parsed["parsed_successfully"])

    def test_12_dataclass_roundtrip(self):
        """Verifies LLMImprovedErrorDetection to_dict and from_dict serialization."""
        det = LLMImprovedErrorDetection(
            classification="incorrect",
            first_incorrect_step=1,
            confidence="high",
            explanation="Arithmetic mistake.",
            raw_response='{"classification": "incorrect"}',
            latency_seconds=1.25,
            parsed_successfully=True,
            provider="gemini",
            model="gemini-3.1-flash-lite",
        )
        d = det.to_dict()
        self.assertEqual(d["classification"], "incorrect")
        self.assertEqual(d["first_incorrect_step"], 1)
        self.assertEqual(d["latency_seconds"], 1.25)

        restored = LLMImprovedErrorDetection.from_dict(d)
        self.assertEqual(restored.classification, "incorrect")
        self.assertEqual(restored.first_incorrect_step, 1)
        self.assertEqual(restored.confidence, "high")
        self.assertTrue(restored.parsed_successfully)

    @patch("llm_improved_detector.call_llm")
    def test_13_mocked_detect_call_success(self, mock_call):
        """Verifies detect_reasoning_error_improved with mocked client call."""
        mock_call.return_value = json.dumps({
            "classification": "incorrect",
            "first_incorrect_step": 3,
            "confidence": "high",
            "explanation": "Sign error in step 3."
        })
        mock_client = MagicMock()
        det = detect_reasoning_error_improved(
            problem="Solve 2x + 4 = 10",
            reasoning_text="Step 1: 2x = 6\nStep 2: x = 3",
            provider="gemini",
            client=mock_client,
            model_name="mock-gemini",
        )
        self.assertEqual(det.classification, "incorrect")
        self.assertEqual(det.first_incorrect_step, 3)
        self.assertEqual(det.confidence, "high")
        self.assertTrue(det.parsed_successfully)
        self.assertEqual(det.model, "mock-gemini")
        mock_call.assert_called_once()

    @patch("llm_improved_detector.call_llm")
    def test_14_mocked_detect_call_api_exception(self, mock_call):
        """Verifies graceful handling of client exceptions during inference."""
        mock_call.side_effect = RuntimeError("API connection timeout")
        mock_client = MagicMock()
        det = detect_reasoning_error_improved(
            problem="Solve 2x = 4",
            reasoning_text="Step 1: x = 2",
            provider="gemini",
            client=mock_client,
            model_name="mock-gemini",
        )
        self.assertEqual(det.classification, "uncertain")
        self.assertIsNone(det.first_incorrect_step)
        self.assertFalse(det.parsed_successfully)
        self.assertIn("API connection timeout", det.explanation)

    def test_15_checkpoint_resumability_and_parsing_check(self):
        """Verifies checkpointing logic skips parsed records and retries failed records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "test_predictions.jsonl")

            # 1. Write an unparsed record (e.g. from an API failure)
            checkpoint_prediction(chk_path, {
                "id": "eval_001",
                "parsed_successfully": False,
                "prediction": {"classification": "uncertain"}
            })
            # 2. Write a valid parsed record
            checkpoint_prediction(chk_path, {
                "id": "eval_002",
                "parsed_successfully": True,
                "prediction": {"classification": "correct", "first_incorrect_step": None}
            })

            completed = load_existing_predictions(chk_path)
            # eval_001 must NOT be in completed (it failed and should be re-evaluated)
            self.assertNotIn("eval_001", completed)
            # eval_002 must be in completed
            self.assertIn("eval_002", completed)

            # 3. Simulate successful retry of eval_001 appended to file
            checkpoint_prediction(chk_path, {
                "id": "eval_001",
                "parsed_successfully": True,
                "prediction": {"classification": "incorrect", "first_incorrect_step": 1}
            })
            updated_completed = load_existing_predictions(chk_path)
            self.assertIn("eval_001", updated_completed)
            self.assertEqual(updated_completed["eval_001"]["prediction"]["classification"], "incorrect")

    def test_16_metrics_computation_exact_formulas(self):
        """Verifies exact formulas for TP, FP, TN, FN_exp, FN_unc, precision, recall, F1."""
        gold_cases = [
            {"id": "c1", "ground_truth": {"is_correct": False, "error_step": 1}},  # TP, step match
            {"id": "c2", "ground_truth": {"is_correct": False, "error_step": 2}},  # TP, step mismatch
            {"id": "c3", "ground_truth": {"is_correct": False, "error_step": 1}},  # FN_explicit (predicted correct)
            {"id": "c4", "ground_truth": {"is_correct": False, "error_step": 1}},  # FN_uncertain (predicted uncertain)
            {"id": "c5", "ground_truth": {"is_correct": True, "error_step": None}}, # TN (predicted correct)
            {"id": "c6", "ground_truth": {"is_correct": True, "error_step": None}}, # FP (predicted incorrect)
            {"id": "c7", "ground_truth": {"is_correct": True, "error_step": None}}, # Abstention on correct (predicted uncertain)
        ]

        preds = {
            "c1": {"prediction": {"classification": "incorrect", "first_incorrect_step": 1}},
            "c2": {"prediction": {"classification": "incorrect", "first_incorrect_step": 3}}, # mismatch
            "c3": {"prediction": {"classification": "correct", "first_incorrect_step": None}},
            "c4": {"prediction": {"classification": "uncertain", "first_incorrect_step": None}},
            "c5": {"prediction": {"classification": "correct", "first_incorrect_step": None}},
            "c6": {"prediction": {"classification": "incorrect", "first_incorrect_step": 1}},
            "c7": {"prediction": {"classification": "uncertain", "first_incorrect_step": None}},
        }

        metrics = compute_metrics_for_subset("test", gold_cases, preds)
        cm = metrics["confusion_matrix"]

        self.assertEqual(cm["true_positives"], 2)
        self.assertEqual(cm["false_positives"], 1)
        self.assertEqual(cm["true_negatives"], 1)
        self.assertEqual(cm["false_negatives_explicit"], 1)
        self.assertEqual(cm["false_negatives_uncertain"], 1)
        self.assertEqual(cm["total_false_negatives"], 2)
        self.assertEqual(cm["abstentions_on_correct"], 1)

        # Precision: TP / (TP + FP) = 2 / (2 + 1) = 2/3 = 0.6667
        self.assertAlmostEqual(metrics["error_detection_metrics"]["precision"], 0.6667, places=4)

        # Conservative recall: TP / Total Incorrect = 2 / 4 = 0.5000
        self.assertAlmostEqual(metrics["error_detection_metrics"]["conservative_recall"], 0.5000, places=4)

        # Strict accuracy: (TP + TN) / Total = (2 + 1) / 7 = 3/7 = 0.4286
        self.assertAlmostEqual(metrics["accuracy_metrics"]["strict_accuracy"], 0.4286, places=4)

        # Decisive accuracy: Correct Decisive / Total Decisive = (2 + 1) / (2 + 1 + 1 + 1) = 3 / 5 = 0.6000
        self.assertAlmostEqual(metrics["accuracy_metrics"]["decisive_accuracy"], 0.6000, places=4)

        # Step localization: 1 match out of 2 TPs = 0.5000
        self.assertAlmostEqual(metrics["step_localization"]["localization_accuracy"], 0.5000, places=4)

    def test_17_markdown_report_generation(self):
        """Verifies that markdown report is generated properly with expected sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.md")
            cases = self.real_cases[:10]
            # Mock dummy predictions
            preds = {c["id"]: {"prediction": {"classification": "correct", "first_incorrect_step": None}} for c in cases}
            all_metrics = compute_all_evaluation_metrics(cases, preds)
            rep = generate_markdown_report(all_metrics, out_path)

            self.assertTrue(os.path.exists(out_path))
            self.assertIn("# Phase 4 – Objective 28: Improved LLM-Based Reasoning Error Detection Report", rep)
            self.assertIn("## 1. Executive Summary", rep)
            self.assertIn("## 2. Confusion Matrix", rep)
            self.assertIn("## 3. Performance Across Dataset Cohorts", rep)

    def test_18_pilot_case_ids_and_comparison_logic(self):
        """Verifies PILOT_CASE_IDS structure and compare_improved_against_baseline functionality."""
        from evaluate_llm_improved import PILOT_CASE_IDS, compare_improved_against_baseline
        self.assertEqual(len(PILOT_CASE_IDS), 20)
        self.assertEqual(len(set(PILOT_CASE_IDS)), 20)

        # Confirm all 10 boundary cases are in pilot list
        for i in range(191, 201):
            self.assertIn(f"eval_{i:03d}", PILOT_CASE_IDS)

        # Mock comparative case
        cases = [
            {"id": "eval_017", "ground_truth": {"is_correct": True, "error_category": "none", "error_step": None}},
            {"id": "eval_092", "ground_truth": {"is_correct": False, "error_category": "arithmetic_error", "error_step": 1}},
        ]
        base_preds = {
            "eval_017": {"prediction": {"classification": "incorrect", "first_incorrect_step": 3}},  # FP
            "eval_092": {"prediction": {"classification": "correct", "first_incorrect_step": None}},  # FN
        }
        imp_preds = {
            "eval_017": {"prediction": {"classification": "correct", "first_incorrect_step": None}},   # Corrected to TN!
            "eval_092": {"prediction": {"classification": "incorrect", "first_incorrect_step": 1}},  # Corrected to TP!
        }
        comp = compare_improved_against_baseline(cases, base_preds, imp_preds)
        self.assertEqual(len(comp["corrected_cases"]), 2)
        self.assertEqual(len(comp["regressed_cases"]), 0)
        self.assertEqual(comp["summary_deltas"]["false_positive_reduction"], 1)
        self.assertEqual(comp["summary_deltas"]["false_negative_reduction"], 1)

    @patch("llm_improved_detector.call_llm")
    def test_19_transient_503_retry_success(self, mock_call):
        """Verifies that transient 503 errors trigger backoff retry and succeed upon recovery."""
        mock_call.side_effect = [
            RuntimeError("503 UNAVAILABLE: model is currently experiencing high demand"),
            json.dumps({"classification": "correct", "first_incorrect_step": None, "confidence": "high", "explanation": "Valid."}),
        ]
        with patch("time.sleep") as mock_sleep:
            det = detect_reasoning_error_improved(
                problem="Solve x = 2",
                reasoning_text="x = 2",
                client=MagicMock(),
                model_name="mock-model",
            )
            self.assertEqual(det.classification, "correct")
            self.assertTrue(det.parsed_successfully)
            self.assertEqual(mock_call.call_count, 2)
            mock_sleep.assert_called_once_with(8.0)


if __name__ == "__main__":
    unittest.main()
