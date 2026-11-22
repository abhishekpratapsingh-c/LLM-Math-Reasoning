"""
Unit tests for step_transition_verifier.py.

All tests operate strictly on static data without making any external API calls.
Covers all required verification scenarios for Phase 3, Objective 22:
- Valid arithmetic simplification.
- Valid algebraic transformations.
- Incorrect algebraic transformations.
- Adding or subtracting the same quantity from both sides.
- Multiplying or dividing both sides.
- Division by zero.
- Squaring both sides and extraneous roots.
- Fraction transformations.
- Negative numbers.
- Variable/domain restrictions.
- Unsupported expressions.
- Unrelated consecutive statements.
- Preservation of original step order.
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


class TestStepTransitionVerifier(unittest.TestCase):
    """Test suite for mathematical transformation and step transition verification."""

    # --------------------------------------------------------------------------
    # 1. Valid arithmetic simplification
    # --------------------------------------------------------------------------
    def test_valid_arithmetic_simplification(self):
        text = (
            "**Step 1: Simplify arithmetic**\n"
            "3 * 40 + 2 * 15\n"
            "120 + 30\n"
            "100 + 50\n\n"
            "Final Answer: 150"
        )
        res = verify_step_transitions(text)
        self.assertEqual(res.status, "valid")
        self.assertEqual(res.total_transitions, 2)
        self.assertEqual(res.valid_transitions, 2)
        self.assertEqual(res.invalid_transitions, 0)
        self.assertEqual(res.transitions[0].transition_status, "valid")
        self.assertEqual(res.transitions[0].transformation_type, "arithmetic_simplification")
        self.assertEqual(res.transitions[1].transition_status, "valid")
        self.assertEqual(res.transitions[1].transformation_type, "arithmetic_simplification")

    # --------------------------------------------------------------------------
    # 2. Valid algebraic transformations
    # --------------------------------------------------------------------------
    def test_valid_algebraic_transformations(self):
        text = (
            "**Step 1: Solve equation**\n"
            "2 * x + 4 = 10\n"
            "2 * x = 6\n"
            "x = 3\n\n"
            "Final Answer: 3"
        )
        res = verify_step_transitions(text)
        self.assertEqual(res.status, "valid")
        self.assertEqual(res.total_transitions, 2)
        self.assertEqual(res.valid_transitions, 2)
        self.assertEqual(res.invalid_transitions, 0)
        self.assertEqual(res.transitions[0].transition_status, "valid")
        self.assertEqual(res.transitions[1].transition_status, "valid")

    # --------------------------------------------------------------------------
    # 3. Incorrect algebraic transformations
    # --------------------------------------------------------------------------
    def test_incorrect_algebraic_transformation(self):
        text = (
            "**Step 1: Calculation error**\n"
            "2 * x + 4 = 10\n"
            "2 * x = 14\n\n"
            "Final Answer: 7"
        )
        res = verify_step_transitions(text)
        self.assertEqual(res.status, "has_invalid_transitions")
        self.assertEqual(res.invalid_transitions, 1)
        self.assertEqual(res.transitions[0].transition_status, "invalid")
        self.assertEqual(res.transitions[0].transformation_type, "incorrect_transformation")
        self.assertIn("3", res.transitions[0].explanation)
        self.assertIn("7", res.transitions[0].explanation)

    # --------------------------------------------------------------------------
    # 4. Adding or subtracting the same quantity from both sides
    # --------------------------------------------------------------------------
    def test_adding_or_subtracting_both_sides(self):
        # Adding 5 to both sides: 3 * x - 5 = 10 -> 3 * x = 15
        text_add = (
            "**Step 1: Add 5**\n"
            "3 * x - 5 = 10\n"
            "3 * x = 15\n\n"
            "Final Answer: 5"
        )
        res_add = verify_step_transitions(text_add)
        self.assertEqual(res_add.transitions[0].transition_status, "valid")
        self.assertIn(res_add.transitions[0].transformation_type, ("both_sides_operation", "algebraic_rearrangement"))

        # Subtracting 8 from both sides: y + 8 = 20 -> y = 12
        text_sub = (
            "**Step 1: Subtract 8**\n"
            "y + 8 = 20\n"
            "y = 12\n\n"
            "Final Answer: 12"
        )
        res_sub = verify_step_transitions(text_sub)
        self.assertEqual(res_sub.transitions[0].transition_status, "valid")
        self.assertIn(res_sub.transitions[0].transformation_type, ("both_sides_operation", "algebraic_rearrangement"))

    # --------------------------------------------------------------------------
    # 5. Multiplying or dividing both sides
    # --------------------------------------------------------------------------
    def test_multiplying_or_dividing_both_sides(self):
        # Dividing both sides by 3: 3 * x = 15 -> x = 5
        text_div = (
            "**Step 1: Divide by 3**\n"
            "3 * x = 15\n"
            "x = 5\n\n"
            "Final Answer: 5"
        )
        res_div = verify_step_transitions(text_div)
        self.assertEqual(res_div.transitions[0].transition_status, "valid")
        self.assertEqual(res_div.transitions[0].transformation_type, "both_sides_scaling")
        self.assertIn("divided both sides by 3", res_div.transitions[0].explanation.lower())

        # Multiplying both sides by 4: x / 4 = 5 -> x = 20
        text_mul = (
            "**Step 1: Multiply by 4**\n"
            "x / 4 = 5\n"
            "x = 20\n\n"
            "Final Answer: 20"
        )
        res_mul = verify_step_transitions(text_mul)
        self.assertEqual(res_mul.transitions[0].transition_status, "valid")
        self.assertEqual(res_mul.transitions[0].transformation_type, "both_sides_scaling")

    # --------------------------------------------------------------------------
    # 6. Division by zero
    # --------------------------------------------------------------------------
    def test_division_by_zero(self):
        text_divzero = (
            "**Step 1: Zero error**\n"
            "10 = 10\n"
            "10 / 0 = 5\n\n"
            "Final Answer: 5"
        )
        res = verify_step_transitions(text_divzero)
        self.assertEqual(res.status, "has_invalid_transitions")
        self.assertEqual(res.invalid_transitions, 1)
        self.assertEqual(res.transitions[0].transition_status, "invalid")
        self.assertIn(res.transitions[0].transformation_type, ("division_by_zero", "invalid_expression"))

    # --------------------------------------------------------------------------
    # 7. Squaring both sides and extraneous roots
    # --------------------------------------------------------------------------
    def test_squaring_both_sides_and_extraneous_roots(self):
        text_sq = (
            "**Step 1: Square both sides**\n"
            "x = 3\n"
            "x**2 = 9\n\n"
            "Final Answer: 9"
        )
        res = verify_step_transitions(text_sq)
        self.assertEqual(res.status, "valid")
        self.assertEqual(res.transitions[0].transition_status, "valid")
        self.assertEqual(res.transitions[0].transformation_type, "squaring_both_sides")
        self.assertTrue(any("extraneous" in c.lower() for c in res.transitions[0].conditions))

    # --------------------------------------------------------------------------
    # 8. Fraction transformations
    # --------------------------------------------------------------------------
    def test_fraction_transformations(self):
        # Valid fraction arithmetic simplification
        text_frac = (
            "**Step 1: Fractions**\n"
            "1/2 + 1/4\n"
            "3/4\n\n"
            "Final Answer: 3/4"
        )
        res = verify_step_transitions(text_frac)
        self.assertEqual(res.status, "valid")
        self.assertEqual(res.transitions[0].transition_status, "valid")
        self.assertEqual(res.transitions[0].transformation_type, "arithmetic_simplification")

        # Invalid fraction arithmetic simplification
        text_frac_err = (
            "**Step 1: Fraction error**\n"
            "1/2 + 1/4\n"
            "2/6\n\n"
            "Final Answer: 2/6"
        )
        res_err = verify_step_transitions(text_frac_err)
        self.assertEqual(res_err.status, "has_invalid_transitions")
        self.assertEqual(res_err.transitions[0].transition_status, "invalid")

    # --------------------------------------------------------------------------
    # 9. Negative numbers
    # --------------------------------------------------------------------------
    def test_negative_numbers(self):
        # Multiply both sides by -1: -x = 5 -> x = -5
        text_neg_scale = (
            "**Step 1: Multiply by -1**\n"
            "-x = 5\n"
            "x = -5\n\n"
            "Final Answer: -5"
        )
        res_scale = verify_step_transitions(text_neg_scale)
        self.assertEqual(res_scale.transitions[0].transition_status, "valid")
        self.assertIn(res_scale.transitions[0].transformation_type, ("both_sides_scaling", "algebraic_rearrangement"))

        # Valid negative arithmetic simplification
        text_neg_arith = (
            "**Step 1: Negative arithmetic**\n"
            "-2 * 10\n"
            "0 - 20\n\n"
            "Final Answer: -20"
        )
        res_arith = verify_step_transitions(text_neg_arith)
        self.assertEqual(res_arith.transitions[0].transition_status, "valid")
        self.assertEqual(res_arith.transitions[0].transformation_type, "arithmetic_simplification")

        # Incorrect negative sign arithmetic
        text_neg_err = (
            "**Step 1: Negative error**\n"
            "-2 * 10\n"
            "0 + 20\n\n"
            "Final Answer: 20"
        )
        res_err = verify_step_transitions(text_neg_err)
        self.assertEqual(res_err.transitions[0].transition_status, "invalid")
        self.assertEqual(res_err.transitions[0].transformation_type, "incorrect_transformation")

    # --------------------------------------------------------------------------
    # 10. Variable and domain restrictions
    # --------------------------------------------------------------------------
    def test_variable_and_domain_restrictions(self):
        # Division by variable losing solution 0: x**2 = 2 * x -> x = 2
        text_div_var = (
            "**Step 1: Divide by variable**\n"
            "x**2 = 2 * x\n"
            "x = 2\n\n"
            "Final Answer: 2"
        )
        res_var = verify_step_transitions(text_div_var)
        self.assertEqual(res_var.transitions[0].transition_status, "invalid")
        self.assertEqual(res_var.transitions[0].transformation_type, "division_by_variable")
        self.assertTrue(any("x != 0" in c for c in res_var.transitions[0].conditions))

        # Extraneous root violating domain restriction: (x**2 - 1)/(x - 1) = 2
        # Transitioning to x = 1 directly violates x != 1
        text_domain = (
            "**Step 1: Extraneous root**\n"
            "(x**2 - 1)/(x - 1) = 2\n"
            "x = 1\n\n"
            "Final Answer: 1"
        )
        res_dom = verify_step_transitions(text_domain)
        self.assertEqual(res_dom.transitions[0].transition_status, "invalid")
        self.assertEqual(res_dom.transitions[0].transformation_type, "domain_violation")
        self.assertTrue(any("x != 1" in c for c in res_dom.transitions[0].conditions))

    # --------------------------------------------------------------------------
    # 11. Unsupported expressions
    # --------------------------------------------------------------------------
    def test_unsupported_expressions(self):
        text_unsupp = (
            "**Step 1: Conceptual formula**\n"
            "Total cost = Unit cost * Quantity\n"
            "3 * 40 = 120\n\n"
            "Final Answer: 120"
        )
        res = verify_step_transitions(text_unsupp)
        self.assertEqual(res.transitions[0].transition_status, "unsupported")
        self.assertEqual(res.transitions[0].transformation_type, "unsupported")

    # --------------------------------------------------------------------------
    # 12. Unrelated consecutive statements
    # --------------------------------------------------------------------------
    def test_unrelated_consecutive_statements(self):
        text_unrelated = (
            "**Step 1: Calculate notebooks**\n"
            "Cost of notebooks = 120\n\n"
            "**Step 2: Calculate pens**\n"
            "Cost of pens = 30\n\n"
            "Final Answer: 150"
        )
        res = verify_step_transitions(text_unrelated)
        self.assertEqual(res.transitions[0].transition_status, "unsupported")
        self.assertEqual(res.transitions[0].transformation_type, "unrelated")
        self.assertIn("distinct", res.transitions[0].explanation.lower())

    # --------------------------------------------------------------------------
    # 13. Preservation of original step order and step numbers
    # --------------------------------------------------------------------------
    def test_preservation_of_original_step_order(self):
        text_multi = (
            "**Step 1: Step one**\n"
            "2 * x + 4 = 10\n\n"
            "**Step 2: Step two**\n"
            "2 * x = 6\n\n"
            "**Step 3: Step three**\n"
            "x = 3\n\n"
            "Final Answer: 3"
        )
        res = verify_step_transitions(text_multi)
        self.assertEqual(res.total_transitions, 2)
        # Transition 1: Step 1 -> Step 2
        self.assertEqual(res.transitions[0].prev_step_number, 1)
        self.assertEqual(res.transitions[0].curr_step_number, 2)
        self.assertEqual(res.transitions[0].prev_expression, "2 * x + 4 = 10")
        self.assertEqual(res.transitions[0].curr_expression, "2 * x = 6")
        self.assertEqual(res.transitions[0].transition_status, "valid")

        # Transition 2: Step 2 -> Step 3
        self.assertEqual(res.transitions[1].prev_step_number, 2)
        self.assertEqual(res.transitions[1].curr_step_number, 3)
        self.assertEqual(res.transitions[1].prev_expression, "2 * x = 6")
        self.assertEqual(res.transitions[1].curr_expression, "x = 3")
        self.assertEqual(res.transitions[1].transition_status, "valid")


if __name__ == "__main__":
    unittest.main()
