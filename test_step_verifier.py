"""
Unit tests for step_verifier.py.

All tests operate strictly on static data without making any external API calls.
Covers all 14 required verification scenarios for Phase 3, Objective 21.
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


class TestStepVerifier(unittest.TestCase):
    """Test suite for SymPy-based reasoning verification."""

    # --------------------------------------------------------------------------
    # Scenario 1: Correct arithmetic equality
    # --------------------------------------------------------------------------
    def test_correct_arithmetic_equality(self):
        text = "**Step 1: Calculate cost**\n3 * 40 = 120\n\nFinal Answer: 120"
        result = verify_structured_reasoning(text)

        self.assertEqual(result.status, "verified")
        self.assertEqual(result.total_expressions, 1)
        self.assertEqual(result.verified_count, 1)
        self.assertEqual(result.incorrect_count, 0)

        expr = result.steps[0].expressions[0]
        self.assertEqual(expr.verification_status, "verified")
        self.assertEqual(expr.lhs_value, "120")
        self.assertEqual(expr.rhs_value, "120")
        self.assertEqual(expr.calculated_value, "120")
        self.assertEqual(expr.expression_type, "arithmetic_equality")

    # --------------------------------------------------------------------------
    # Scenario 2: Incorrect arithmetic equality
    # --------------------------------------------------------------------------
    def test_incorrect_arithmetic_equality(self):
        text = "**Step 1: Calculate cost**\n3 * 40 = 130\n\nFinal Answer: 130"
        result = verify_structured_reasoning(text)

        self.assertEqual(result.status, "has_errors")
        self.assertEqual(result.total_expressions, 1)
        self.assertEqual(result.verified_count, 0)
        self.assertEqual(result.incorrect_count, 1)

        expr = result.steps[0].expressions[0]
        self.assertEqual(expr.verification_status, "incorrect")
        self.assertEqual(expr.lhs_value, "120")
        self.assertEqual(expr.rhs_value, "130")
        self.assertIn("120", expr.explanation)
        self.assertIn("130", expr.explanation)

    # --------------------------------------------------------------------------
    # Scenario 3: Labeled equality with a correct RHS
    # --------------------------------------------------------------------------
    def test_labeled_equality_correct_rhs(self):
        text = "**Step 1: Calculate cost**\nCost of notebooks = 3 * 40\n\nFinal Answer: 120"
        result = verify_structured_reasoning(text)

        self.assertEqual(result.status, "verified")
        self.assertEqual(result.total_expressions, 1)
        self.assertEqual(result.verified_count, 1)

        expr = result.steps[0].expressions[0]
        self.assertEqual(expr.verification_status, "verified")
        self.assertEqual(expr.expression_type, "labeled_equality")
        self.assertEqual(expr.label, "Cost of notebooks")
        self.assertEqual(expr.calculated_value, "120")
        self.assertEqual(expr.rhs_value, "120")
        self.assertIn("Cost of notebooks", expr.explanation)

    # --------------------------------------------------------------------------
    # Scenario 4: Labeled equality with an incorrect numerical assertion
    # --------------------------------------------------------------------------
    def test_labeled_equality_incorrect_assertion(self):
        # 1. End-to-end parsed via text
        text = "**Step 1: Calculate cost**\nCost of notebooks = 3 * 40 = 130\n\nFinal Answer: 130"
        result = verify_structured_reasoning(text)

        self.assertEqual(result.status, "has_errors")
        self.assertEqual(result.incorrect_count, 1)
        # Find the incorrect expression
        incorrect_expr = [e for e in result.steps[0].expressions if e.verification_status == "incorrect"][0]
        self.assertEqual(incorrect_expr.lhs_value, "120")
        self.assertEqual(incorrect_expr.rhs_value, "130")

        # 2. Directly structured labeled equality with chained incorrect assertion
        sr = StructuredReasoning(
            raw_response="test",
            steps=[
                StructuredStep(
                    step_number=1,
                    step_header="Step 1",
                    original_text="Cost = 3 * 40 = 130",
                    expressions=[
                        StructuredExpression(
                            step_number=1,
                            expression_type="labeled_equality",
                            raw_text="Cost = 3 * 40 = 130",
                            label="Cost",
                            rhs_normalized="3 * 40 = 130",
                            is_equality=True,
                            is_valid_syntax=True,
                        )
                    ],
                )
            ],
            status="success",
        )
        res_direct = verify_structured_reasoning(sr)
        self.assertEqual(res_direct.status, "has_errors")
        self.assertEqual(res_direct.steps[0].expressions[0].verification_status, "incorrect")
        self.assertEqual(res_direct.steps[0].expressions[0].label, "Cost")

    # --------------------------------------------------------------------------
    # Scenario 5: Algebraic equality
    # --------------------------------------------------------------------------
    def test_algebraic_equality(self):
        # Solvable consistent equation
        sr_solvable = StructuredReasoning(
            raw_response="test",
            steps=[
                StructuredStep(
                    step_number=1,
                    step_header="Step 1",
                    original_text="x + 5 = 12",
                    expressions=[
                        StructuredExpression(
                            step_number=1,
                            expression_type="algebraic_equality",
                            raw_text="x + 5 = 12",
                            lhs_normalized="x + 5",
                            rhs_normalized="12",
                            normalized_equation="x + 5 = 12",
                            is_equality=True,
                            is_valid_syntax=True,
                            label="x",
                        )
                    ],
                )
            ],
            status="success",
        )
        res_sol = verify_structured_reasoning(sr_solvable)
        self.assertEqual(res_sol.status, "verified")
        expr_sol = res_sol.steps[0].expressions[0]
        self.assertEqual(expr_sol.verification_status, "verified")
        self.assertEqual(expr_sol.calculated_value, "7")
        self.assertEqual(expr_sol.label, "x")

        # Inconsistent equation (no solution)
        sr_inconsistent = StructuredReasoning(
            raw_response="test",
            steps=[
                StructuredStep(
                    step_number=1,
                    step_header="Step 1",
                    original_text="x + 1 = x + 2",
                    expressions=[
                        StructuredExpression(
                            step_number=1,
                            expression_type="algebraic_equality",
                            raw_text="x + 1 = x + 2",
                            lhs_normalized="x + 1",
                            rhs_normalized="x + 2",
                            normalized_equation="x + 1 = x + 2",
                            is_equality=True,
                            is_valid_syntax=True,
                            label="x",
                        )
                    ],
                )
            ],
            status="success",
        )
        res_incon = verify_structured_reasoning(sr_inconsistent)
        self.assertEqual(res_incon.status, "has_errors")
        expr_incon = res_incon.steps[0].expressions[0]
        self.assertEqual(expr_incon.verification_status, "incorrect")
        self.assertIn("inconsistent", expr_incon.explanation.lower())

    # --------------------------------------------------------------------------
    # Scenario 6: Standalone arithmetic expression
    # --------------------------------------------------------------------------
    def test_standalone_arithmetic_expression(self):
        text = "**Step 1: Compute total**\n3 * 40 + 2 * 15\n\nFinal Answer: 150"
        result = verify_structured_reasoning(text)

        self.assertEqual(result.status, "verified")
        self.assertEqual(result.total_expressions, 1)
        expr = result.steps[0].expressions[0]
        self.assertEqual(expr.verification_status, "verified")
        self.assertEqual(expr.expression_type, "standalone_expression")
        self.assertEqual(expr.calculated_value, "150")

    # --------------------------------------------------------------------------
    # Scenario 7: Fractions and decimal calculations
    # --------------------------------------------------------------------------
    def test_fractions_and_decimals(self):
        text = (
            "**Step 1: Decimals and Fractions**\n"
            "0.1 + 0.2 = 0.3\n"
            "1/2 + 1/4 = 3/4\n\n"
            "Final Answer: 0.3"
        )
        result = verify_structured_reasoning(text)

        self.assertEqual(result.status, "verified")
        self.assertEqual(result.total_expressions, 2)
        self.assertEqual(result.verified_count, 2)

        expr_dec = result.steps[0].expressions[0]
        self.assertEqual(expr_dec.verification_status, "verified")

        expr_frac = result.steps[0].expressions[1]
        self.assertEqual(expr_frac.verification_status, "verified")

    # --------------------------------------------------------------------------
    # Scenario 8: Negative numbers
    # --------------------------------------------------------------------------
    def test_negative_numbers(self):
        # Valid negative multiplication
        text_correct = "**Step 1: Negative numbers**\n-2 * 10 = -20\n\nFinal Answer: -20"
        res_corr = verify_structured_reasoning(text_correct)
        self.assertEqual(res_corr.status, "verified")
        self.assertEqual(res_corr.steps[0].expressions[0].verification_status, "verified")

        # Invalid negative multiplication
        text_incorrect = "**Step 1: Negative numbers**\n-2 * 10 = 20\n\nFinal Answer: 20"
        res_inc = verify_structured_reasoning(text_incorrect)
        self.assertEqual(res_inc.status, "has_errors")
        self.assertEqual(res_inc.steps[0].expressions[0].verification_status, "incorrect")

    # --------------------------------------------------------------------------
    # Scenario 9: Division by zero
    # --------------------------------------------------------------------------
    def test_division_by_zero(self):
        # Direct zero denominator in equality
        text_eq = "**Step 1: Div zero**\n10 / 0 = 5\n\nFinal Answer: 5"
        res_eq = verify_structured_reasoning(text_eq)
        self.assertEqual(res_eq.status, "invalid")
        self.assertEqual(res_eq.invalid_count, 1)
        self.assertEqual(res_eq.steps[0].expressions[0].verification_status, "invalid")
        self.assertIn("division by zero", res_eq.steps[0].expressions[0].explanation.lower())

        # Zero denominator in standalone expression
        text_expr = "**Step 1: Zero expr**\n5 / (3 - 3)\n\nFinal Answer: 0"
        res_expr = verify_structured_reasoning(text_expr)
        self.assertEqual(res_expr.status, "invalid")
        self.assertEqual(res_expr.steps[0].expressions[0].verification_status, "invalid")

    # --------------------------------------------------------------------------
    # Scenario 10: Domain restrictions
    # --------------------------------------------------------------------------
    def test_domain_restrictions(self):
        # Identity preserving domain restriction x != 0
        sr_id = StructuredReasoning(
            raw_response="test",
            steps=[
                StructuredStep(
                    step_number=1,
                    step_header="Step 1",
                    original_text="x / x = 1",
                    expressions=[
                        StructuredExpression(
                            step_number=1,
                            expression_type="algebraic_equality",
                            raw_text="x / x = 1",
                            lhs_normalized="x / x",
                            rhs_normalized="1",
                            normalized_equation="x / x = 1",
                            is_equality=True,
                            is_valid_syntax=True,
                            label="x",
                        )
                    ],
                )
            ],
            status="success",
        )
        res_id = verify_structured_reasoning(sr_id)
        self.assertEqual(res_id.status, "verified")
        expr_id = res_id.steps[0].expressions[0]
        self.assertEqual(expr_id.verification_status, "verified")
        self.assertTrue(any("x != 0" in r for r in expr_id.domain_restrictions))

        # Extraneous root violating domain restriction: (x**2 - 1)/(x - 1) = 2
        # Candidate root is x=1, but x != 1, so no solution -> incorrect
        sr_ext = StructuredReasoning(
            raw_response="test",
            steps=[
                StructuredStep(
                    step_number=1,
                    step_header="Step 1",
                    original_text="(x**2 - 1)/(x - 1) = 2",
                    expressions=[
                        StructuredExpression(
                            step_number=1,
                            expression_type="algebraic_equality",
                            raw_text="(x**2 - 1)/(x - 1) = 2",
                            lhs_normalized="(x**2 - 1)/(x - 1)",
                            rhs_normalized="2",
                            normalized_equation="(x**2 - 1)/(x - 1) = 2",
                            is_equality=True,
                            is_valid_syntax=True,
                            label="x",
                        )
                    ],
                )
            ],
            status="success",
        )
        res_ext = verify_structured_reasoning(sr_ext)
        self.assertEqual(res_ext.status, "has_errors")
        expr_ext = res_ext.steps[0].expressions[0]
        self.assertEqual(expr_ext.verification_status, "incorrect")
        self.assertTrue(any("x != 1" in r for r in expr_ext.domain_restrictions))

    # --------------------------------------------------------------------------
    # Scenario 11: Unsupported expressions (NL formulas and syntax errors)
    # --------------------------------------------------------------------------
    def test_unsupported_expressions(self):
        text = (
            "**Step 1: Formula**\n"
            "Cost = Number of items * Unit price\n"
            "3 * (40 + = 120\n\n"
            "Final Answer: 120"
        )
        result = verify_structured_reasoning(text)

        self.assertEqual(result.unsupported_count, 2)
        for expr in result.steps[0].expressions:
            self.assertEqual(expr.verification_status, "unsupported")
            self.assertIsNotNone(expr.unsupported_reason)

    # --------------------------------------------------------------------------
    # Scenario 12: Preservation of original step numbers and ordering
    # --------------------------------------------------------------------------
    def test_preservation_of_step_numbers_and_order(self):
        text = (
            "**Step 1: First step**\n"
            "a = 10\n\n"
            "**Step 2: Second step**\n"
            "b = 20\n\n"
            "**Step 3: Third step**\n"
            "c = 30\n\n"
            "Final Answer: 30"
        )
        result = verify_structured_reasoning(text)

        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.steps[0].step_number, 1)
        self.assertEqual(result.steps[1].step_number, 2)
        self.assertEqual(result.steps[2].step_number, 3)

        self.assertIn("First step", result.steps[0].step_header)
        self.assertIn("Second step", result.steps[1].step_header)
        self.assertIn("Third step", result.steps[2].step_header)

    # --------------------------------------------------------------------------
    # Scenario 13: Multiple expressions within one step
    # --------------------------------------------------------------------------
    def test_multiple_expressions_within_one_step(self):
        text = (
            "**Step 1: Compute notebook and pen prices**\n"
            "3 * 40 = 120\n"
            "2 * 15 = 30\n"
            "120 + 30 = 150\n\n"
            "Final Answer: 150"
        )
        result = verify_structured_reasoning(text)

        self.assertEqual(len(result.steps), 1)
        self.assertEqual(len(result.steps[0].expressions), 3)
        self.assertEqual(result.total_expressions, 3)
        self.assertEqual(result.verified_count, 3)

        self.assertEqual(result.steps[0].expressions[0].raw_text, "3 * 40 = 120")
        self.assertEqual(result.steps[0].expressions[1].raw_text, "2 * 15 = 30")
        self.assertEqual(result.steps[0].expressions[2].raw_text, "120 + 30 = 150")

    # --------------------------------------------------------------------------
    # Scenario 14: Empty or malformed structured reasoning
    # --------------------------------------------------------------------------
    def test_empty_or_malformed_reasoning(self):
        # Empty string
        res_empty = verify_structured_reasoning("")
        self.assertEqual(res_empty.status, "empty_response")
        self.assertEqual(res_empty.total_expressions, 0)
        self.assertEqual(len(res_empty.steps), 0)

        # No steps found
        res_nosteps = verify_structured_reasoning("Just some unstructured chat text without steps.")
        self.assertEqual(res_nosteps.status, "no_steps_found")
        self.assertEqual(res_nosteps.total_expressions, 0)
        self.assertEqual(len(res_nosteps.steps), 0)

        # Directly passed StructuredReasoning with no steps
        empty_sr = StructuredReasoning(
            raw_response="",
            steps=[],
            final_answer=None,
            final_answer_raw=None,
            status="empty_response",
            explanation="Empty input",
        )
        res_direct = verify_structured_reasoning(empty_sr)
        self.assertEqual(res_direct.status, "empty_response")
        self.assertEqual(res_direct.total_expressions, 0)


if __name__ == "__main__":
    unittest.main()
