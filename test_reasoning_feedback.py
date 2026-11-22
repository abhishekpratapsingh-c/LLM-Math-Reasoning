"""
test_reasoning_feedback.py
--------------------------
Unit tests for Phase 3, Objective 24: Pass solver feedback to the LLM so it
can explain why a mathematical reasoning step is incorrect.

All tests operate strictly on static data with mocked LLM clients.
Zero external API calls are made during test execution.

Covers all 13 required testing scenarios:
1. test_incorrect_arithmetic_explanation
2. test_incorrect_algebraic_transformation_explanation
3. test_lost_solution_explanation
4. test_division_by_zero_explanation
5. test_conditional_warning_explanation
6. test_unsupported_expression_handling
7. test_no_api_call_when_no_errors_or_warnings (verify mock called 0 times)
8. test_multiple_detected_issues
9. test_preservation_of_solver_evidence
10. test_empty_llm_response_handling
11. test_api_error_handling
12. test_invalid_input_handling (empty problem, invalid provider, missing data)
13. test_provider_selection (gemini vs groq)
"""

import unittest
from unittest.mock import patch, MagicMock

from reasoning_feedback import (
    StepFeedback,
    ReasoningFeedbackResult,
    generate_reasoning_feedback,
    FEEDBACK_PROMPT_TEMPLATE,
)
from reasoning_error_detector import (
    DetectedIssue,
    StepErrorReport,
    ReasoningErrorReport,
    detect_reasoning_errors,
)


class TestReasoningFeedback(unittest.TestCase):
    """Test suite for reasoning feedback generation in reasoning_feedback.py."""

    # --------------------------------------------------------------------------
    # 1. Incorrect arithmetic assertion explanation
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_incorrect_arithmetic_explanation(self, mock_call_llm):
        """Test that an incorrect arithmetic assertion generates proper solver evidence and LLM explanation."""
        mock_call_llm.return_value = (
            "Step 1 is incorrect because 3 * 40 equals 120, not 130. "
            "To correct it, replace 130 with 120."
        )
        problem = "A shopkeeper buys 3 notebooks for ₹40 each. What is the total cost?"
        reasoning = (
            "**Step 1: Calculate total cost**\n"
            "Cost of notebooks = 3 * 40\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        self.assertTrue(result.has_issues)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.total_feedbacks, 1)

        fb = result.feedbacks[0]
        self.assertEqual(fb.step_number, 1)
        self.assertEqual(fb.error_category, "incorrect_assertion")
        self.assertEqual(fb.severity, "definite_error")
        self.assertIn("120", fb.solver_evidence)
        self.assertIn("130", fb.solver_evidence)
        self.assertIn("3 * 40 equals 120", fb.llm_explanation)
        self.assertEqual(fb.status, "success")
        mock_call_llm.assert_called_once()

    # --------------------------------------------------------------------------
    # 2. Incorrect algebraic transformation explanation
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_incorrect_algebraic_transformation_explanation(self, mock_call_llm):
        """Test that an invalid algebraic leap produces transition solver evidence and LLM explanation."""
        mock_call_llm.return_value = (
            "Step 1 contains an algebraic error. Subtracting 4 from both sides of 2*x + 4 = 10 "
            "gives 2*x = 6, but the step states 2*x = 14."
        )
        problem = "Solve the equation 2*x + 4 = 10 for x."
        reasoning = (
            "**Step 1: Subtract 4 from both sides**\n"
            "2*x + 4 = 10\n"
            "2*x = 14\n\n"
            "Final Answer: 7"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        self.assertTrue(result.has_issues)
        self.assertEqual(result.total_feedbacks, 1)
        fb = result.feedbacks[0]
        self.assertEqual(fb.step_number, 1)
        self.assertEqual(fb.error_category, "invalid_transformation")
        self.assertEqual(fb.severity, "definite_error")
        self.assertIn("solution", fb.solver_evidence.lower())
        self.assertIn("algebraic error", fb.llm_explanation.lower())

    # --------------------------------------------------------------------------
    # 3. Lost solution explanation
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_lost_solution_explanation(self, mock_call_llm):
        """Test that division by a variable losing roots produces lost_solution feedback."""
        mock_call_llm.return_value = (
            "Dividing both sides by x ignores the case where x = 0, causing the solution x = 0 to be lost."
        )
        problem = "Find all real solutions to x**2 = 2*x."
        reasoning = (
            "**Step 1: Divide both sides by x**\n"
            "x**2 = 2*x\n"
            "x = 2\n\n"
            "Final Answer: 2"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        self.assertTrue(result.has_issues)
        self.assertEqual(result.total_feedbacks, 1)
        fb = result.feedbacks[0]
        self.assertEqual(fb.error_category, "lost_solution")
        self.assertEqual(fb.severity, "definite_error")
        self.assertTrue("losing" in fb.solver_evidence.lower() or "lost" in fb.solver_evidence.lower())
        self.assertIn("x = 0", fb.llm_explanation)

    # --------------------------------------------------------------------------
    # 4. Division by zero explanation
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_division_by_zero_explanation(self, mock_call_llm):
        """Test that division by zero is flagged and explained as undefined."""
        mock_call_llm.return_value = (
            "Division by zero is undefined in mathematics; 10 / 0 does not equal 5."
        )
        problem = "Evaluate 10 / 0."
        reasoning = (
            "**Step 1: Perform division**\n"
            "10 / 0 = 5\n\n"
            "Final Answer: 5"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        self.assertTrue(result.has_issues)
        self.assertEqual(result.total_feedbacks, 1)
        fb = result.feedbacks[0]
        self.assertEqual(fb.error_category, "division_by_zero")
        self.assertEqual(fb.severity, "definite_error")
        self.assertIn("division by zero", fb.solver_evidence.lower())
        self.assertIn("undefined", fb.llm_explanation.lower())

    # --------------------------------------------------------------------------
    # 5. Conditional warning explanation
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_conditional_warning_explanation(self, mock_call_llm):
        """Test that squaring both sides generates a conditional warning feedback with restrictions."""
        mock_call_llm.return_value = (
            "Squaring both sides is a one-way implication that can introduce extraneous roots. "
            "Candidate solutions must be checked in the original equation."
        )
        problem = "Solve the equation x = 3."
        reasoning = (
            "**Step 1: Square both sides**\n"
            "x = 3\n"
            "x**2 = 9\n\n"
            "Final Answer: 3"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        self.assertTrue(result.has_issues)
        self.assertEqual(result.total_feedbacks, 1)
        fb = result.feedbacks[0]
        self.assertEqual(fb.error_category, "conditional_warning")
        self.assertEqual(fb.severity, "potential_issue")
        self.assertTrue("square" in fb.solver_evidence.lower() or "squared" in fb.solver_evidence.lower())
        self.assertTrue(len(fb.conditions) > 0)
        self.assertIn("extraneous roots", fb.llm_explanation.lower())

    # --------------------------------------------------------------------------
    # 6. Unsupported expression handling
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_unsupported_expression_handling(self, mock_call_llm):
        """Test that natural language or unsupported formulas are handled gracefully."""
        mock_call_llm.return_value = (
            "This expression represents the physics relation Distance = Speed * Time. "
            "To be solved symbolically, it should be written with standard algebraic symbols."
        )
        problem = "A car travels at 60 km/h for 2 hours."
        reasoning = (
            "**Step 1: State formula**\n"
            "Distance = Speed * Time\n\n"
            "Final Answer: 120"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        self.assertTrue(result.has_issues)
        self.assertEqual(result.total_feedbacks, 1)
        fb = result.feedbacks[0]
        self.assertEqual(fb.error_category, "unsupported")
        self.assertEqual(fb.severity, "unsupported")
        self.assertTrue("natural-language" in fb.solver_evidence.lower() or "conceptual" in fb.solver_evidence.lower())

    # --------------------------------------------------------------------------
    # 7. Zero API call optimization when no errors or warnings
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.get_llm_client")
    @patch("reasoning_feedback.call_llm")
    def test_no_api_call_when_no_errors_or_warnings(self, mock_call_llm, mock_get_client):
        """Verify that when reasoning is completely sound, ZERO API calls and ZERO client initializations occur."""
        problem = "A shopkeeper buys 3 notebooks for ₹40 each and 2 pens for ₹15 each. What is the total cost?"
        reasoning = (
            "**Step 1: Calculate notebook cost**\n"
            "Cost of notebooks = 3 * 40\n"
            "Cost of notebooks = 120\n\n"
            "**Step 2: Calculate pen cost**\n"
            "Cost of pens = 2 * 15\n"
            "Cost of pens = 30\n\n"
            "**Step 3: Calculate total cost**\n"
            "Total cost = 120 + 30\n"
            "Total cost = 150\n\n"
            "Final Answer: 150"
        )

        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
        )

        # STRICT VERIFICATION: Exactly 0 API calls and 0 client inits
        mock_call_llm.assert_not_called()
        mock_get_client.assert_not_called()

        self.assertFalse(result.has_issues)
        self.assertEqual(result.status, "no_issues")
        self.assertEqual(result.total_feedbacks, 0)
        self.assertEqual(len(result.feedbacks), 0)
        self.assertIn("mathematically sound", result.summary.lower())

    # --------------------------------------------------------------------------
    # 8. Multiple detected issues
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_multiple_detected_issues(self, mock_call_llm):
        """Test that multiple detected errors across steps are each individually explained in order."""
        mock_call_llm.side_effect = [
            "Arithmetic error: 3 * 40 is 120, not 130.",
            "Arithmetic error: 2 * 15 is 30, not 40.",
        ]
        problem = "Calculate total items."
        reasoning = (
            "**Step 1: Calculate notebook cost**\n"
            "3 * 40 = 130\n\n"
            "**Step 2: Calculate pen cost**\n"
            "2 * 15 = 40\n\n"
            "Final Answer: 170"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        self.assertTrue(result.has_issues)
        self.assertEqual(result.total_feedbacks, 2)
        self.assertEqual(mock_call_llm.call_count, 2)
        self.assertEqual(result.feedbacks[0].step_number, 1)
        self.assertEqual(result.feedbacks[0].error_category, "incorrect_assertion")
        self.assertEqual(result.feedbacks[1].step_number, 2)
        self.assertEqual(result.feedbacks[1].error_category, "incorrect_assertion")

    # --------------------------------------------------------------------------
    # 9. Preservation of solver evidence (never overwritten by LLM)
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_preservation_of_solver_evidence(self, mock_call_llm):
        """Test that solver mathematical evidence is strictly preserved and not modified by LLM output."""
        # Simulated LLM output that claims something completely different
        mock_call_llm.return_value = "The student forgot to carry the one."

        problem = "Check calculation."
        reasoning = (
            "**Step 1: Arithmetic**\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        fb = result.feedbacks[0]
        # Solver evidence must contain exact mathematical verification, unmodified
        self.assertIn("120", fb.solver_evidence)
        self.assertIn("130", fb.solver_evidence)
        self.assertEqual(fb.error_category, "incorrect_assertion")
        # LLM explanation must be in its separate field
        self.assertEqual(fb.llm_explanation, "The student forgot to carry the one.")

    # --------------------------------------------------------------------------
    # 10. Empty LLM response handling
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_empty_llm_response_handling(self, mock_call_llm):
        """Test that empty or whitespace LLM response is handled gracefully without crashing."""
        mock_call_llm.return_value = "   "

        problem = "Check calculation."
        reasoning = (
            "**Step 1: Arithmetic**\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
        )

        fb = result.feedbacks[0]
        self.assertEqual(fb.status, "failed")
        self.assertIn("empty", fb.llm_explanation.lower())
        self.assertIn("120", fb.solver_evidence)
        self.assertEqual(result.status, "failed")

    # --------------------------------------------------------------------------
    # 11. API error handling
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_api_error_handling(self, mock_call_llm):
        """Test that an API exception during inference is caught gracefully and status marked failed."""
        mock_call_llm.side_effect = RuntimeError("503 Service Unavailable: Network timeout")

        problem = "Check calculation."
        reasoning = (
            "**Step 1: Arithmetic**\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        mock_client = MagicMock()
        result = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            client=mock_client,
            model_name="mock-model",
            raise_on_error=False,
        )

        self.assertEqual(result.total_feedbacks, 1)
        fb = result.feedbacks[0]
        self.assertEqual(fb.status, "failed")
        self.assertIn("503 Service Unavailable", fb.llm_explanation)
        self.assertIn("120", fb.solver_evidence)
        self.assertEqual(result.status, "failed")

    # --------------------------------------------------------------------------
    # 12. Invalid input handling
    # --------------------------------------------------------------------------
    def test_invalid_input_handling(self):
        """Test rejection of empty problems, invalid providers, or missing data."""
        # Empty problem
        with self.assertRaises(ValueError) as ctx:
            generate_reasoning_feedback(problem="", structured_reasoning="3 * 40 = 130")
        self.assertIn("empty", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            generate_reasoning_feedback(problem="   ", structured_reasoning="3 * 40 = 130")
        self.assertIn("empty", str(ctx.exception).lower())

        # Unsupported provider
        with self.assertRaises(ValueError) as ctx:
            generate_reasoning_feedback(problem="Solve", structured_reasoning="3*40=130", provider="openai")
        self.assertIn("unsupported provider", str(ctx.exception).lower())

        # Missing both error_report and structured_reasoning
        with self.assertRaises(ValueError) as ctx:
            generate_reasoning_feedback(problem="Solve", structured_reasoning=None, error_report=None)
        self.assertIn("either 'error_report' or 'structured_reasoning'", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 13. Provider selection (gemini vs groq)
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    @patch("reasoning_feedback.get_llm_client")
    def test_provider_selection(self, mock_get_client, mock_call_llm):
        """Test that provider parameter properly selects gemini and groq clients."""
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, "test-model")
        mock_call_llm.return_value = "Step 1 is wrong."

        problem = "Check calculation."
        reasoning = (
            "**Step 1: Arithmetic**\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )

        # Test Gemini
        generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            provider="gemini",
        )
        mock_get_client.assert_called_with("gemini")

        # Test Groq
        generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=reasoning,
            provider="groq",
        )
        mock_get_client.assert_called_with("groq")


if __name__ == "__main__":
    unittest.main()
