"""
test_verification_report.py
---------------------------
Unit tests for Phase 3, Objective 25: Generate a Final Verification Report.

All tests operate strictly on static data and existing dataclasses.
Zero external API calls are made during test execution.

Covers all 15 required testing scenarios:
1. test_completely_correct_reasoning
2. test_incorrect_arithmetic_assertion
3. test_invalid_algebraic_transformation
4. test_lost_solution
5. test_division_by_zero
6. test_conditional_warning
7. test_unsupported_expression
8. test_multiple_errors_in_one_solution
9. test_missing_llm_feedback
10. test_failed_llm_feedback
11. test_correct_calculation_of_report_metrics
12. test_overall_status_precedence
13. test_markdown_report_generation
14. test_empty_or_partially_available_inputs
15. test_preservation_of_original_reasoning_and_solver_evidence
"""

import unittest
from verification_report import (
    ProblemInfo,
    ExpressionRecord,
    StepRecord,
    TransitionRecord,
    DetectedIssueRecord,
    LLMExplanationRecord,
    VerificationSummary,
    VerificationReport,
    generate_verification_report,
    format_markdown_report,
)
from reasoning_feedback import (
    StepFeedback,
    ReasoningFeedbackResult,
)
from reasoning_error_detector import (
    DetectedIssue,
    StepErrorReport,
    ReasoningErrorReport,
)


class TestVerificationReport(unittest.TestCase):
    """Test suite for final verification report generation in verification_report.py."""

    # --------------------------------------------------------------------------
    # 1. Completely correct reasoning
    # --------------------------------------------------------------------------
    def test_completely_correct_reasoning(self):
        """Test report generation for completely correct mathematical reasoning."""
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
        report = generate_verification_report(problem=problem, raw_response=reasoning)

        self.assertEqual(report.summary.overall_status, "verified")
        self.assertEqual(report.summary.total_detected_errors, 0)
        self.assertEqual(report.summary.total_warnings, 0)
        self.assertEqual(report.summary.total_unsupported_expressions, 0)
        self.assertEqual(len(report.detected_errors), 0)
        self.assertEqual(len(report.detected_warnings), 0)
        self.assertEqual(report.problem_info.final_answer, "150")
        self.assertEqual(len(report.steps), 3)

    # --------------------------------------------------------------------------
    # 2. Incorrect arithmetic assertion
    # --------------------------------------------------------------------------
    def test_incorrect_arithmetic_assertion(self):
        """Test report generation when an arithmetic calculation is false."""
        problem = "A shopkeeper buys 3 notebooks for ₹40 each. What is the total cost?"
        reasoning = (
            "**Step 1: Calculate total cost**\n"
            "Cost of notebooks = 3 * 40\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)

        self.assertEqual(report.summary.overall_status, "errors_detected")
        self.assertEqual(report.summary.total_detected_errors, 1)
        self.assertEqual(len(report.detected_errors), 1)

        err = report.detected_errors[0]
        self.assertEqual(err.step_number, 1)
        self.assertEqual(err.error_category, "incorrect_assertion")
        self.assertTrue(err.is_definite_error)
        self.assertIn("120", err.explanation)
        self.assertIn("130", err.explanation)

    # --------------------------------------------------------------------------
    # 3. Invalid algebraic transformation
    # --------------------------------------------------------------------------
    def test_invalid_algebraic_transformation(self):
        """Test report generation when an algebraic transformation changes the solution set."""
        problem = "Solve the equation 2*x + 4 = 10 for x."
        reasoning = (
            "**Step 1: Subtract 4 from both sides**\n"
            "2*x + 4 = 10\n"
            "2*x = 14\n\n"
            "**Step 2: Divide by 2**\n"
            "x = 7\n\n"
            "Final Answer: 7"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)

        self.assertEqual(report.summary.overall_status, "errors_detected")
        self.assertTrue(report.summary.total_detected_errors >= 1)
        self.assertTrue(any(e.error_category == "invalid_transformation" for e in report.detected_errors))

    # --------------------------------------------------------------------------
    # 4. Lost solution
    # --------------------------------------------------------------------------
    def test_lost_solution(self):
        """Test report generation when division by a variable loses a valid root."""
        problem = "Find all real solutions to x**2 = 2*x."
        reasoning = (
            "**Step 1: Divide both sides by x**\n"
            "x**2 = 2*x\n"
            "x = 2\n\n"
            "Final Answer: 2"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)

        self.assertEqual(report.summary.overall_status, "errors_detected")
        self.assertTrue(any(e.error_category == "lost_solution" for e in report.detected_errors))
        lost_err = next(e for e in report.detected_errors if e.error_category == "lost_solution")
        self.assertTrue(lost_err.is_definite_error)
        self.assertIn("0", lost_err.explanation)

    # --------------------------------------------------------------------------
    # 5. Division by zero
    # --------------------------------------------------------------------------
    def test_division_by_zero(self):
        """Test report generation when division by zero occurs."""
        problem = "Evaluate 10 / 0."
        reasoning = (
            "**Step 1: Compute division**\n"
            "10 / 0 = 5\n\n"
            "Final Answer: 5"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)

        self.assertEqual(report.summary.overall_status, "errors_detected")
        self.assertTrue(any(e.error_category == "division_by_zero" for e in report.detected_errors))

    # --------------------------------------------------------------------------
    # 6. Conditional warning
    # --------------------------------------------------------------------------
    def test_conditional_warning(self):
        """Test report generation when squaring both sides introduces extraneous roots."""
        problem = "Solve the equation x = 3."
        reasoning = (
            "**Step 1: Square both sides**\n"
            "x = 3\n"
            "x**2 = 9\n\n"
            "Final Answer: 3"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)

        self.assertEqual(report.summary.overall_status, "warnings")
        self.assertEqual(report.summary.total_detected_errors, 0)
        self.assertEqual(report.summary.total_warnings, 1)
        self.assertEqual(len(report.detected_warnings), 1)
        self.assertEqual(report.detected_warnings[0].error_category, "conditional_warning")
        self.assertTrue(len(report.detected_warnings[0].conditions) > 0)

    # --------------------------------------------------------------------------
    # 7. Unsupported expression
    # --------------------------------------------------------------------------
    def test_unsupported_expression(self):
        """Test report generation when an expression is conceptual/natural language and not incorrect."""
        problem = "State the physics relation for distance."
        reasoning = (
            "**Step 1: State formula**\n"
            "Distance = Speed * Time\n\n"
            "Final Answer: Distance = Speed * Time"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)

        self.assertEqual(report.summary.overall_status, "partially_verified")
        self.assertEqual(report.summary.total_detected_errors, 0)
        self.assertEqual(report.summary.total_warnings, 0)
        self.assertEqual(report.summary.total_unsupported_expressions, 1)
        self.assertEqual(len(report.detected_errors), 0)
        self.assertEqual(len(report.unsupported_issues), 1)

    # --------------------------------------------------------------------------
    # 8. Multiple errors in one solution
    # --------------------------------------------------------------------------
    def test_multiple_errors_in_one_solution(self):
        """Test report generation when multiple errors occur in the reasoning."""
        problem = "Calculate total items."
        reasoning = (
            "**Step 1: Calculate notebook cost**\n"
            "3 * 40 = 130\n\n"
            "**Step 2: Calculate pen cost**\n"
            "2 * 15 = 40\n\n"
            "Final Answer: 170"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)

        self.assertEqual(report.summary.overall_status, "errors_detected")
        self.assertEqual(report.summary.total_detected_errors, 2)
        self.assertEqual(len(report.detected_errors), 2)
        self.assertEqual(report.detected_errors[0].step_number, 1)
        self.assertEqual(report.detected_errors[1].step_number, 2)

    # --------------------------------------------------------------------------
    # 9. Missing LLM feedback
    # --------------------------------------------------------------------------
    def test_missing_llm_feedback(self):
        """Test report generation when feedback_result is omitted."""
        problem = "A shopkeeper buys 3 notebooks for ₹40 each."
        reasoning = (
            "**Step 1: Calculate total cost**\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        # Calling without feedback_result
        report = generate_verification_report(problem=problem, raw_response=reasoning, feedback_result=None)

        self.assertEqual(len(report.llm_explanations), 0)
        md = report.to_markdown()
        self.assertIn("LLM explanations were not requested or were unavailable", md)

    # --------------------------------------------------------------------------
    # 10. Failed LLM feedback
    # --------------------------------------------------------------------------
    def test_failed_llm_feedback(self):
        """Test that failed LLM feedback is documented cleanly without crashing or inventing text."""
        problem = "A shopkeeper buys 3 notebooks for ₹40 each."
        reasoning = (
            "**Step 1: Calculate total cost**\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        failed_feedback = ReasoningFeedbackResult(
            problem=problem,
            feedbacks=[
                StepFeedback(
                    step_number=1,
                    expression="3 * 40 = 130",
                    error_category="incorrect_assertion",
                    severity="definite_error",
                    solver_evidence="LHS (120) != RHS (130)",
                    llm_explanation="504 Gateway Timeout: API unreachable",
                    status="failed",
                )
            ],
            total_feedbacks=1,
            has_issues=True,
            status="failed",
        )
        report = generate_verification_report(
            problem=problem,
            raw_response=reasoning,
            feedback_result=failed_feedback,
        )

        self.assertEqual(len(report.llm_explanations), 1)
        rec = report.llm_explanations[0]
        self.assertEqual(rec.status, "failed")
        self.assertEqual(rec.llm_explanation, "504 Gateway Timeout: API unreachable")

        md = report.to_markdown()
        self.assertIn("LLM explanation failed or returned empty", md)
        self.assertIn("504 Gateway Timeout", md)

    # --------------------------------------------------------------------------
    # 11. Correct calculation of report metrics
    # --------------------------------------------------------------------------
    def test_correct_calculation_of_report_metrics(self):
        """Test that all summary metrics match the underlying steps and expressions."""
        problem = "A shopkeeper buys items."
        reasoning = (
            "**Step 1: Calculate notebook cost**\n"
            "Cost of notebooks = 3 * 40\n"
            "Cost of notebooks = 120\n\n"
            "**Step 2: Calculate pen cost**\n"
            "Cost of pens = 2 * 15\n"
            "Cost of pens = 30\n\n"
            "Final Answer: 150"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)
        s = report.summary

        self.assertEqual(s.total_steps, 2)
        self.assertEqual(s.total_expressions, 4)
        self.assertEqual(
            s.total_verified_expressions
            + s.total_incorrect_expressions
            + s.total_invalid_expressions
            + s.total_unsupported_expressions,
            s.total_expressions,
        )

    # --------------------------------------------------------------------------
    # 12. Overall status precedence
    # --------------------------------------------------------------------------
    def test_overall_status_precedence(self):
        """Verify strict precedence: errors > warnings > unsupported > verified, and empty -> failed."""
        prob = "Test math problem"

        # Case 1: Empty reasoning -> verification_failed
        rep_empty = generate_verification_report(problem=prob, raw_response="")
        self.assertEqual(rep_empty.summary.overall_status, "verification_failed")

        # Case 2: Definite error + warning + unsupported -> errors_detected
        err_rep = ReasoningErrorReport(
            total_steps=3,
            total_expressions=3,
            total_definite_errors=1,
            total_potential_issues=1,
            total_unsupported=1,
            total_verified=0,
            status="has_errors",
        )
        rep_err = generate_verification_report(
            problem=prob,
            raw_response="Step 1: 1=1",
            error_report=err_rep,
        )
        self.assertEqual(rep_err.summary.overall_status, "errors_detected")

        # Case 3: Warning + unsupported (0 errors) -> warnings
        warn_rep = ReasoningErrorReport(
            total_steps=2,
            total_expressions=2,
            total_definite_errors=0,
            total_potential_issues=1,
            total_unsupported=1,
            total_verified=0,
            status="has_warnings",
        )
        rep_warn = generate_verification_report(
            problem=prob,
            raw_response="Step 1: 1=1",
            error_report=warn_rep,
        )
        self.assertEqual(rep_warn.summary.overall_status, "warnings")

        # Case 4: Unsupported only (0 errors, 0 warnings) -> partially_verified
        unsup_rep = ReasoningErrorReport(
            total_steps=1,
            total_expressions=1,
            total_definite_errors=0,
            total_potential_issues=0,
            total_unsupported=1,
            total_verified=0,
            status="unsupported",
        )
        # We also need an expression with unsupported status
        rep_unsup = generate_verification_report(
            problem=prob,
            raw_response="**Step 1**\nDistance = Speed * Time",
            error_report=unsup_rep,
        )
        self.assertEqual(rep_unsup.summary.overall_status, "partially_verified")

    # --------------------------------------------------------------------------
    # 13. Markdown report generation
    # --------------------------------------------------------------------------
    def test_markdown_report_generation(self):
        """Test that Markdown output contains all required sections from A to G."""
        problem = "A shopkeeper buys 3 notebooks for ₹40 each. What is the total cost?"
        reasoning = (
            "**Step 1: Calculate total cost**\n"
            "Cost of notebooks = 3 * 40\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        report = generate_verification_report(problem=problem, raw_response=reasoning)
        md = report.to_markdown()

        self.assertIn("# Mathematical Reasoning Verification Report", md)
        self.assertIn("## A. Problem Information", md)
        self.assertIn("## B. Generated Reasoning Steps", md)
        self.assertIn("## C. Mathematical Verification of Expressions", md)
        self.assertIn("## D. Step Transition Verification", md)
        self.assertIn("## E. Detected Errors and Warnings", md)
        self.assertIn("## F. LLM Explanations", md)
        self.assertIn("## G. Overall Verification Summary", md)
        self.assertIn("errors_detected", md)
        self.assertIn("3 * 40 = 130", md)

    # --------------------------------------------------------------------------
    # 14. Empty or partially available inputs
    # --------------------------------------------------------------------------
    def test_empty_or_partially_available_inputs(self):
        """Test robust handling of invalid, whitespace, or empty inputs."""
        # Empty problem raises ValueError
        with self.assertRaises(ValueError):
            generate_verification_report(problem="", raw_response="Step 1: 1=1")

        with self.assertRaises(ValueError):
            generate_verification_report(problem="   ", raw_response="Step 1: 1=1")

        with self.assertRaises(ValueError):
            generate_verification_report(problem=None, raw_response="Step 1: 1=1")

        # Empty reasoning produces verification_failed report
        rep = generate_verification_report(problem="Valid problem", raw_response=None)
        self.assertEqual(rep.summary.overall_status, "verification_failed")
        self.assertEqual(rep.summary.total_steps, 0)

    # --------------------------------------------------------------------------
    # 15. Preservation of original reasoning and solver evidence
    # --------------------------------------------------------------------------
    def test_preservation_of_original_reasoning_and_solver_evidence(self):
        """Test that verbatim original reasoning text and exact SymPy solver findings are preserved."""
        problem = "Verify equation."
        raw_text = (
            "**Step 1: Compute product**\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        report = generate_verification_report(problem=problem, raw_response=raw_text)

        # Original text preserved
        self.assertEqual(report.steps[0].step_header, "**Step 1: Compute product**")
        self.assertIn("3 * 40 = 130", report.steps[0].original_text)

        # Solver evidence preserved verbatim
        self.assertTrue(len(report.detected_errors) > 0)
        self.assertIn("120", report.detected_errors[0].explanation)
        self.assertIn("130", report.detected_errors[0].explanation)


if __name__ == "__main__":
    unittest.main()
