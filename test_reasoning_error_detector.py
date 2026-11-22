"""
Unit tests for reasoning_error_detector.py.

All tests operate strictly on static data without making any external API calls.
Covers all required testing scenarios for Phase 3, Objective 23:
- Correct reasoning with no errors.
- Incorrect arithmetic assertion.
- Incorrect algebraic transformation.
- Division-by-zero error.
- Lost solution after division by a variable.
- Domain restriction violation.
- Squaring both sides with a potential extraneous-root warning.
- Unsupported natural-language formula.
- Unrelated consecutive statements.
- Multiple errors in one response.
- Duplicate issue consolidation.
- Preservation of original step order.
- Empty or malformed reasoning.
"""

import unittest
from step_structurer import (
    StructuredReasoning,
    StructuredStep,
    StructuredExpression,
    structure_reasoning_steps,
)
from step_verifier import (
    VerifiedReasoning,
    VerifiedStep,
    VerifiedExpression,
    verify_structured_reasoning,
)
from step_transition_verifier import (
    StepTransition,
    TransitionVerificationResult,
    verify_step_transitions,
)
from reasoning_error_detector import (
    DetectedIssue,
    StepErrorReport,
    ReasoningErrorReport,
    detect_reasoning_errors,
)


class TestReasoningErrorDetector(unittest.TestCase):
    """Test suite for reasoning error detection and consolidation."""

    # --------------------------------------------------------------------------
    # 1. Correct reasoning with no errors
    # --------------------------------------------------------------------------
    def test_correct_reasoning_no_errors(self):
        text = (
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
        report = detect_reasoning_errors(text)
        self.assertEqual(report.status, "correct")
        self.assertEqual(report.total_definite_errors, 0)
        self.assertEqual(report.total_potential_issues, 0)
        self.assertEqual(len(report.flagged_error_steps), 0)
        self.assertEqual(len(report.flagged_warning_steps), 0)
        self.assertEqual(len(report.step_reports), 3)

    # --------------------------------------------------------------------------
    # 2. Incorrect arithmetic assertion
    # --------------------------------------------------------------------------
    def test_incorrect_arithmetic_assertion(self):
        text = (
            "**Step 1: Calculate total cost**\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        report = detect_reasoning_errors(text)
        self.assertEqual(report.status, "has_errors")
        self.assertEqual(report.total_definite_errors, 1)
        self.assertEqual(report.flagged_error_steps, [1])

        issue = report.step_reports[0].issues[0]
        self.assertEqual(issue.error_category, "incorrect_assertion")
        self.assertTrue(issue.is_definite_error)
        self.assertEqual(issue.severity, "definite_error")
        self.assertEqual(issue.verification_status, "incorrect")
        self.assertIn("120", issue.explanation)
        self.assertIn("130", issue.explanation)

    # --------------------------------------------------------------------------
    # 3. Incorrect algebraic transformation
    # --------------------------------------------------------------------------
    def test_incorrect_algebraic_transformation(self):
        text = (
            "**Step 1: Solve equation**\n"
            "2 * x + 4 = 10\n"
            "2 * x = 14\n\n"
            "Final Answer: 7"
        )
        report = detect_reasoning_errors(text)
        self.assertEqual(report.status, "has_errors")
        self.assertIn(1, report.flagged_error_steps)

        # Check the transformation issue on the second statement
        trans_issue = report.step_reports[0].issues[1]
        self.assertEqual(trans_issue.error_category, "invalid_transformation")
        self.assertTrue(trans_issue.is_definite_error)
        self.assertEqual(trans_issue.severity, "definite_error")
        self.assertIn("3", trans_issue.explanation)
        self.assertIn("7", trans_issue.explanation)

    # --------------------------------------------------------------------------
    # 4. Division-by-zero error
    # --------------------------------------------------------------------------
    def test_division_by_zero_error(self):
        text = (
            "**Step 1: Compute fraction**\n"
            "10 / 0 = 5\n\n"
            "Final Answer: 5"
        )
        report = detect_reasoning_errors(text)
        self.assertEqual(report.status, "has_errors")
        self.assertEqual(report.total_definite_errors, 1)
        self.assertEqual(report.flagged_error_steps, [1])

        issue = report.step_reports[0].issues[0]
        self.assertEqual(issue.error_category, "division_by_zero")
        self.assertTrue(issue.is_definite_error)
        self.assertEqual(issue.verification_status, "invalid")
        self.assertTrue(any("zero" in c.lower() for c in issue.conditions))

    # --------------------------------------------------------------------------
    # 5. Lost solution after division by a variable
    # --------------------------------------------------------------------------
    def test_lost_solution_division_by_variable(self):
        text = (
            "**Step 1: Divide by variable**\n"
            "x**2 = 2 * x\n"
            "x = 2\n\n"
            "Final Answer: 2"
        )
        report = detect_reasoning_errors(text)
        self.assertEqual(report.status, "has_errors")
        self.assertIn(1, report.flagged_error_steps)

        # The second expression loses the root x = 0
        issue = report.step_reports[0].issues[1]
        self.assertEqual(issue.error_category, "lost_solution")
        self.assertTrue(issue.is_definite_error)
        self.assertEqual(issue.severity, "definite_error")
        self.assertTrue(any("x != 0" in c for c in issue.conditions))

    # --------------------------------------------------------------------------
    # 6. Domain restriction violation
    # --------------------------------------------------------------------------
    def test_domain_restriction_violation(self):
        text = (
            "**Step 1: Extraneous solution violating restriction**\n"
            "(x**2 - 1)/(x - 1) = 2\n"
            "x = 1\n\n"
            "Final Answer: 1"
        )
        report = detect_reasoning_errors(text)
        self.assertEqual(report.status, "has_errors")
        self.assertIn(1, report.flagged_error_steps)

        issue = report.step_reports[0].issues[1]
        self.assertEqual(issue.error_category, "domain_violation")
        self.assertTrue(issue.is_definite_error)
        self.assertEqual(issue.severity, "definite_error")
        self.assertTrue(any("x != 1" in c for c in issue.conditions))

    # --------------------------------------------------------------------------
    # 7. Squaring both sides with potential extraneous-root warning
    # --------------------------------------------------------------------------
    def test_squaring_both_sides_warning(self):
        text = (
            "**Step 1: Square both sides**\n"
            "x = 3\n"
            "x**2 = 9\n\n"
            "Final Answer: 9"
        )
        report = detect_reasoning_errors(text)
        # Should be has_warnings, NOT has_errors (no definite error)
        self.assertEqual(report.status, "has_warnings")
        self.assertEqual(report.total_definite_errors, 0)
        self.assertEqual(report.total_potential_issues, 1)
        self.assertEqual(report.flagged_error_steps, [])
        self.assertEqual(report.flagged_warning_steps, [1])

        issue = report.step_reports[0].issues[1]
        self.assertEqual(issue.error_category, "conditional_warning")
        self.assertFalse(issue.is_definite_error)
        self.assertEqual(issue.severity, "potential_issue")
        self.assertEqual(issue.verification_status, "warning")
        self.assertTrue(any("extraneous" in c.lower() for c in issue.conditions))

    # --------------------------------------------------------------------------
    # 8. Unsupported natural-language formula
    # --------------------------------------------------------------------------
    def test_unsupported_natural_language_formula(self):
        text = (
            "**Step 1: Conceptual formula**\n"
            "Total cost = Unit cost * Quantity\n"
            "Total cost = 120\n\n"
            "Final Answer: 120"
        )
        report = detect_reasoning_errors(text)
        # Unsupported expressions are not classified as definite errors
        self.assertFalse(report.step_reports[0].has_definite_errors)
        self.assertTrue(report.step_reports[0].has_unsupported)

        issue = report.step_reports[0].issues[0]
        self.assertEqual(issue.error_category, "unsupported")
        self.assertFalse(issue.is_definite_error)
        self.assertEqual(issue.severity, "unsupported")

    # --------------------------------------------------------------------------
    # 9. Unrelated consecutive statements
    # --------------------------------------------------------------------------
    def test_unrelated_consecutive_statements(self):
        text = (
            "**Step 1: Notebook calculation**\n"
            "Cost of notebooks = 120\n\n"
            "**Step 2: Pen calculation**\n"
            "Cost of pens = 30\n\n"
            "Final Answer: 150"
        )
        report = detect_reasoning_errors(text)
        self.assertEqual(report.total_definite_errors, 0)
        self.assertEqual(len(report.flagged_error_steps), 0)

        # Step 2 expression should be verified (not flagged as an error merely because it's distinct)
        step2_issue = report.step_reports[1].issues[0]
        self.assertFalse(step2_issue.is_definite_error)

    # --------------------------------------------------------------------------
    # 10. Multiple errors in one response
    # --------------------------------------------------------------------------
    def test_multiple_errors_in_one_response(self):
        text = (
            "**Step 1: First error**\n"
            "3 * 40 = 130\n\n"
            "**Step 2: Second error**\n"
            "10 / 0 = 5\n\n"
            "Final Answer: 5"
        )
        report = detect_reasoning_errors(text)
        self.assertEqual(report.status, "has_errors")
        self.assertEqual(report.total_definite_errors, 2)
        self.assertEqual(report.flagged_error_steps, [1, 2])
        self.assertTrue(report.step_reports[0].has_definite_errors)
        self.assertTrue(report.step_reports[1].has_definite_errors)

    # --------------------------------------------------------------------------
    # 11. Duplicate issue consolidation
    # --------------------------------------------------------------------------
    def test_duplicate_issue_consolidation(self):
        # A statement that is both an arithmetic error AND an invalid incoming transition
        text = (
            "**Step 1: Arithmetic sequence**\n"
            "3 * 40 = 120\n"
            "3 * 40 = 130\n\n"
            "Final Answer: 130"
        )
        report = detect_reasoning_errors(text)
        step = report.step_reports[0]
        self.assertEqual(len(step.issues), 2)

        # The second expression has both assertion error and transition error,
        # but is reported as exactly ONE consolidated issue
        issue2 = step.issues[1]
        self.assertEqual(issue2.error_category, "incorrect_assertion")
        self.assertEqual(issue2.individual_status, "incorrect")
        self.assertEqual(issue2.transition_status, "invalid")
        self.assertTrue(issue2.is_definite_error)
        self.assertIn("120", issue2.explanation)
        self.assertIn("130", issue2.explanation)

    # --------------------------------------------------------------------------
    # 12. Preservation of original step order
    # --------------------------------------------------------------------------
    def test_preservation_of_original_step_order(self):
        text = (
            "**Step 1: First**\n"
            "a = 10\n\n"
            "**Step 2: Second**\n"
            "b = 20\n\n"
            "**Step 3: Third**\n"
            "c = 30\n\n"
            "Final Answer: 30"
        )
        report = detect_reasoning_errors(text)
        self.assertEqual(len(report.step_reports), 3)
        self.assertEqual(report.step_reports[0].step_number, 1)
        self.assertEqual(report.step_reports[1].step_number, 2)
        self.assertEqual(report.step_reports[2].step_number, 3)
        self.assertIn("First", report.step_reports[0].step_header)
        self.assertIn("Second", report.step_reports[1].step_header)
        self.assertIn("Third", report.step_reports[2].step_header)

    # --------------------------------------------------------------------------
    # 13. Empty or malformed reasoning
    # --------------------------------------------------------------------------
    def test_empty_or_malformed_reasoning(self):
        # Empty text
        rep_empty = detect_reasoning_errors("")
        self.assertEqual(rep_empty.status, "empty")
        self.assertEqual(rep_empty.total_steps, 0)
        self.assertEqual(rep_empty.total_expressions, 0)

        # No steps found
        rep_no_steps = detect_reasoning_errors("Just plain conversational text without steps.")
        self.assertEqual(rep_no_steps.status, "empty")
        self.assertEqual(rep_no_steps.total_steps, 0)


if __name__ == "__main__":
    unittest.main()
