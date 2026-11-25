"""
test_objective28_interactive.py
-------------------------------
Phase 4 - Objective 28: Automated Unit Tests for Automatic LLM Evaluation Runner.

Performs 100% offline, mocked unit tests validating:
1. Automatic Processing: Processes all cases sequentially without any input() prompt.
2. Skipping Successful Predictions: Completed cases with `parsed_successfully == True` are skipped.
3. Retrying Failed Predictions: Cases with `parsed_successfully == False` are retried.
4. All Cases Already Completed: Immediate exit with zero API calls.
5. Dedicated Checkpoint Preservation & Append: Existing records preserved intact; new records appended.
6. Malformed / Interrupted Line Ignored: Truncated JSON lines safely skipped; subsequent appends work.
7. Zero 503 Auto-Retries: Disables automatic 503 retry loops (`max_retries=0`) in automatic mode.
8. Strict Information Barrier: Only `problem` and `reasoning_text` enter prompt; zero gold metadata.
9. Graceful Exit on KeyboardInterrupt (Ctrl+C): Catches interrupt and returns summary of progress.
10. Mixed Success and Failure Summary: Correctly reports successful, failed, skipped, and remaining counts.
11. Backwards Compatible Alias: `run_interactive_evaluation` callable with legacy kwargs.
"""

import builtins
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_objective28_interactive import run_automatic_evaluation, run_interactive_evaluation
from evaluate_llm_improved import load_existing_predictions, checkpoint_prediction


class TestObjective28AutomaticRunner(unittest.TestCase):
    """Test suite for run_objective28_interactive.py (Automatic evaluation runner)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset.jsonl")
        self.predictions_path = os.path.join(self.temp_dir, "test_predictions.jsonl")

        # Create a small 3-case synthetic dataset
        self.test_cases = [
            {
                "id": "eval_001",
                "problem": "Solve 2x = 6",
                "reasoning_text": "Step 1: x = 3",
                "ground_truth": {
                    "is_correct": True,
                    "error_category": "none",
                    "error_step": None,
                    "expected_report_status": "verified",
                },
                "metadata": {"topic": "linear_algebra"}
            },
            {
                "id": "eval_002",
                "problem": "Solve x^2 = 4",
                "reasoning_text": "Step 1: x = 2",
                "ground_truth": {
                    "is_correct": False,
                    "error_category": "lost_solution",
                    "error_step": 1,
                    "expected_report_status": "errors_detected",
                },
                "metadata": {"topic": "quadratic"}
            },
            {
                "id": "eval_003",
                "problem": "Calculate 3 + 5",
                "reasoning_text": "Step 1: 3 + 5 = 8",
                "ground_truth": {
                    "is_correct": True,
                    "error_category": "none",
                    "error_step": None,
                    "expected_report_status": "verified",
                },
                "metadata": {"topic": "arithmetic"}
            },
        ]
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            for c in self.test_cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("llm_improved_detector.call_llm")
    def test_01_automatic_processing_no_input_prompts(self, mock_call):
        """Verifies that all cases are processed sequentially and automatically without input() prompts."""
        mock_call.return_value = json.dumps({
            "classification": "correct",
            "first_incorrect_step": None,
            "confidence": "high",
            "explanation": "Sound reasoning.",
        })

        # Ensure builtins.input is NEVER called
        with patch("builtins.input", side_effect=AssertionError("input() should not be called in automatic mode")):
            res = run_automatic_evaluation(
                dataset_path=self.dataset_path,
                predictions_path=self.predictions_path,
                client=MagicMock(),
                model_name="mock-gemini",
                delay_seconds=0.0,
            )

        # All 3 cases should have been processed
        self.assertEqual(mock_call.call_count, 3)
        self.assertEqual(res["evaluated_this_session"], 3)
        self.assertEqual(res["successful_predictions"], 3)
        self.assertEqual(res["failed_predictions"], 0)
        self.assertEqual(res["skipped_successful_predictions"], 0)
        self.assertEqual(res["completed_total"], 3)
        self.assertEqual(res["remaining_cases"], 0)

        # Checkpoint file should have 3 records
        completed = load_existing_predictions(self.predictions_path)
        self.assertEqual(len(completed), 3)
        self.assertIn("eval_001", completed)
        self.assertIn("eval_002", completed)
        self.assertIn("eval_003", completed)

    @patch("llm_improved_detector.call_llm")
    def test_02_skip_successful_predictions(self, mock_call):
        """Verifies that cases already completed with parsed_successfully == True are skipped."""
        # Pre-populate eval_001 as completed
        checkpoint_prediction(self.predictions_path, {
            "id": "eval_001",
            "prediction": {"classification": "correct", "first_incorrect_step": None, "confidence": "high"},
            "parsed_successfully": True,
        })

        mock_call.return_value = json.dumps({
            "classification": "incorrect",
            "first_incorrect_step": 1,
            "confidence": "high",
            "explanation": "Lost negative solution x = -2 in Step 1."
        })

        res = run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        # eval_001 is skipped; eval_002 and eval_003 are evaluated
        self.assertEqual(mock_call.call_count, 2)
        self.assertEqual(res["skipped_successful_predictions"], 1)
        self.assertEqual(res["evaluated_this_session"], 2)
        self.assertEqual(res["successful_predictions"], 2)
        self.assertEqual(res["completed_total"], 3)
        self.assertEqual(res["remaining_cases"], 0)

    @patch("llm_improved_detector.call_llm")
    def test_03_retry_failed_predictions(self, mock_call):
        """Verifies that cases with parsed_successfully == False are not skipped and get retried."""
        # Pre-populate eval_001 with parsed_successfully == False
        checkpoint_prediction(self.predictions_path, {
            "id": "eval_001",
            "prediction": {"classification": "uncertain", "first_incorrect_step": None, "confidence": "low"},
            "parsed_successfully": False,
        })

        mock_call.return_value = json.dumps({
            "classification": "correct",
            "first_incorrect_step": None,
            "confidence": "high",
            "explanation": "Verified sound on retry."
        })

        res = run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        # eval_001 is retried, plus eval_002 and eval_003
        self.assertEqual(mock_call.call_count, 3)
        self.assertEqual(res["skipped_successful_predictions"], 0)
        self.assertEqual(res["evaluated_this_session"], 3)
        self.assertEqual(res["successful_predictions"], 3)
        self.assertEqual(res["completed_total"], 3)

        # Checkpoint file should have 4 lines total: 1 failure + 3 successes
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0]["id"], "eval_001")
        self.assertFalse(lines[0]["parsed_successfully"])
        self.assertEqual(lines[1]["id"], "eval_001")
        self.assertTrue(lines[1]["parsed_successfully"])

    @patch("llm_improved_detector.call_llm")
    def test_04_all_cases_already_completed(self, mock_call):
        """Verifies that if all cases are already completed, runner exits with zero API requests."""
        # Pre-populate all 3 cases as completed
        for case in self.test_cases:
            checkpoint_prediction(self.predictions_path, {
                "id": case["id"],
                "prediction": {"classification": "correct", "first_incorrect_step": None, "confidence": "high"},
                "parsed_successfully": True,
            })

        res = run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        mock_call.assert_not_called()
        self.assertEqual(res["evaluated_this_session"], 0)
        self.assertEqual(res["successful_predictions"], 0)
        self.assertEqual(res["failed_predictions"], 0)
        self.assertEqual(res["skipped_successful_predictions"], 3)
        self.assertEqual(res["completed_total"], 3)
        self.assertEqual(res["remaining_cases"], 0)

    @patch("llm_improved_detector.call_llm")
    def test_05_checkpoint_preservation_and_append(self, mock_call):
        """Verifies that existing checkpoint records are preserved untouched and new records are cleanly appended."""
        rec1 = {
            "id": "eval_001",
            "prediction": {"classification": "correct", "first_incorrect_step": None, "confidence": "high"},
            "parsed_successfully": True,
            "custom_marker": "keep_intact_1",
        }
        rec2 = {
            "id": "eval_002",
            "prediction": {"classification": "incorrect", "first_incorrect_step": 1, "confidence": "high"},
            "parsed_successfully": True,
            "custom_marker": "keep_intact_2",
        }
        checkpoint_prediction(self.predictions_path, rec1)
        checkpoint_prediction(self.predictions_path, rec2)

        with open(self.predictions_path, "r", encoding="utf-8") as f:
            lines_before = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines_before), 2)

        mock_call.return_value = json.dumps({
            "classification": "correct",
            "first_incorrect_step": None,
            "confidence": "high",
            "explanation": "Sound reasoning.",
        })

        res = run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        self.assertEqual(res["evaluated_this_session"], 1)
        self.assertEqual(res["completed_total"], 3)

        with open(self.predictions_path, "r", encoding="utf-8") as f:
            lines_after = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines_after), 3)

        # Confirm first two lines are byte-for-byte identical
        self.assertEqual(lines_after[0], lines_before[0])
        self.assertEqual(lines_after[1], lines_before[1])

        # Confirm line 3 is the new eval_003 record
        parsed_rec3 = json.loads(lines_after[2])
        self.assertEqual(parsed_rec3["id"], "eval_003")
        self.assertTrue(parsed_rec3["parsed_successfully"])

    @patch("llm_improved_detector.call_llm")
    def test_06_malformed_truncated_checkpoint_line_ignored(self, mock_call):
        """Verifies that a truncated/malformed JSON line from an interrupted write is safely ignored, and subsequent appends work."""
        # Write valid record
        checkpoint_prediction(self.predictions_path, {
            "id": "eval_001",
            "prediction": {"classification": "correct", "first_incorrect_step": None, "confidence": "high"},
            "parsed_successfully": True,
        })

        # Simulate interrupted write: write a truncated line without proper JSON closure
        with open(self.predictions_path, "a", encoding="utf-8") as f:
            f.write('{"id": "eval_002", "prediction": {"class\n')

        # Verify load_existing_predictions ignores the malformed line
        completed_before = load_existing_predictions(self.predictions_path)
        self.assertEqual(len(completed_before), 1)
        self.assertIn("eval_001", completed_before)
        self.assertNotIn("eval_002", completed_before)

        mock_call.return_value = json.dumps({
            "classification": "incorrect",
            "first_incorrect_step": 1,
            "confidence": "high",
            "explanation": "Missing root.",
        })

        res = run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        # Evaluates eval_002 and eval_003
        self.assertEqual(res["evaluated_this_session"], 2)

        # Verify reload now successfully finds all 3 cases
        completed_after = load_existing_predictions(self.predictions_path)
        self.assertIn("eval_001", completed_after)
        self.assertIn("eval_002", completed_after)
        self.assertIn("eval_003", completed_after)
        self.assertEqual(len(completed_after), 3)

    @patch("llm_improved_detector.call_llm")
    def test_07_zero_retries_on_503_in_automatic_mode(self, mock_call):
        """Verifies that 503 errors do NOT trigger automatic retries in automatic mode (max_retries=0)."""
        mock_call.side_effect = RuntimeError("503 UNAVAILABLE: high demand")

        res = run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        # Exactly 1 call per case (3 calls total, no retry loops)
        self.assertEqual(mock_call.call_count, 3)
        self.assertEqual(res["evaluated_this_session"], 3)
        self.assertEqual(res["successful_predictions"], 0)
        self.assertEqual(res["failed_predictions"], 3)
        self.assertEqual(res["completed_total"], 0)
        self.assertEqual(res["remaining_cases"], 3)

        with open(self.predictions_path, "r", encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(records), 3)
        for r in records:
            self.assertFalse(r["parsed_successfully"])
            self.assertIn("503 UNAVAILABLE", r["prediction"]["explanation"])

    @patch("llm_improved_detector.call_llm")
    def test_08_information_barrier_no_gold_labels_in_prompt(self, mock_call):
        """Verifies that prompt sent to call_llm contains only problem & reasoning_text, never gold metadata."""
        mock_call.return_value = json.dumps({
            "classification": "correct",
            "first_incorrect_step": None,
            "confidence": "high",
            "explanation": "Sound.",
        })

        run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        for call_item in mock_call.call_args_list:
            prompt = call_item[0][3]  # call_llm(client, provider, model, prompt)
            # Gold metadata must NOT be present
            self.assertNotIn("is_correct", prompt)
            self.assertNotIn("error_category", prompt)
            self.assertNotIn("expected_report_status", prompt)
            self.assertNotIn("verified", prompt)
            self.assertNotIn("linear_algebra", prompt)

    @patch("llm_improved_detector.call_llm")
    def test_09_keyboard_interrupt_graceful_exit(self, mock_call):
        """Verifies that KeyboardInterrupt exits gracefully with summary and preserves completed records."""
        # Case 1 succeeds, Case 2 raises KeyboardInterrupt
        mock_call.side_effect = [
            json.dumps({
                "classification": "correct",
                "first_incorrect_step": None,
                "confidence": "high",
                "explanation": "Sound.",
            }),
            KeyboardInterrupt(),
        ]

        res = run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        self.assertEqual(res["evaluated_this_session"], 1)
        self.assertEqual(res["successful_predictions"], 1)
        self.assertEqual(res["completed_total"], 1)
        self.assertEqual(res["remaining_cases"], 2)

        # Checkpoint has eval_001 preserved
        completed = load_existing_predictions(self.predictions_path)
        self.assertEqual(len(completed), 1)
        self.assertIn("eval_001", completed)

    @patch("llm_improved_detector.call_llm")
    def test_10_mixed_success_and_failure_reporting(self, mock_call):
        """Verifies that summary dictionary accurately distinguishes successful and failed predictions."""
        # Case 1 succeeds, Case 2 fails with 503, Case 3 succeeds
        mock_call.side_effect = [
            json.dumps({
                "classification": "correct",
                "first_incorrect_step": None,
                "confidence": "high",
                "explanation": "Sound 1.",
            }),
            RuntimeError("503 UNAVAILABLE"),
            json.dumps({
                "classification": "correct",
                "first_incorrect_step": None,
                "confidence": "high",
                "explanation": "Sound 3.",
            }),
        ]

        res = run_automatic_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
        )

        self.assertEqual(res["evaluated_this_session"], 3)
        self.assertEqual(res["successful_predictions"], 2)
        self.assertEqual(res["failed_predictions"], 1)
        self.assertEqual(res["completed_total"], 2)
        self.assertEqual(res["remaining_cases"], 1)

    @patch("llm_improved_detector.call_llm")
    def test_11_backwards_compatible_aliases(self, mock_call):
        """Verifies that run_interactive_evaluation alias is importable and accepts legacy kwargs."""
        mock_call.return_value = json.dumps({
            "classification": "correct",
            "first_incorrect_step": None,
            "confidence": "high",
            "explanation": "Sound.",
        })

        # Passing legacy input_func should not raise any TypeError
        res = run_interactive_evaluation(
            dataset_path=self.dataset_path,
            predictions_path=self.predictions_path,
            client=MagicMock(),
            model_name="mock-gemini",
            delay_seconds=0.0,
            input_func=lambda prompt: "y",
        )

        self.assertEqual(res["evaluated_this_session"], 3)
        self.assertEqual(res["completed_total"], 3)


if __name__ == "__main__":
    unittest.main()
