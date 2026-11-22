"""
test_step_structurer.py
-----------------------
Unit tests for Phase 3, Objective 20: Structured Mathematical Representation.

Verifies:
- Simple arithmetic equalities (3 * 40 = 120)
- Multiple equations within a single reasoning step
- Equations spread across multiple sequential steps
- Arithmetic operations: multiplication, division, addition, subtraction, fractions, decimals
- Handling of currency symbols ($100, ₹40) and unit words (mi/day, days)
- Natural-language labels on the left-hand side (Cost of notebooks = 3 * 40)
- Standalone arithmetic expressions without equality signs
- Unsupported, malformed, or ambiguous mathematical syntax
- Exact preservation of original text, LHS/RHS components, and equation ordering
- Empty or malformed extracted reasoning inputs

Uses 100% static test data. Zero external API calls.
"""

import unittest
from step_extractor import ExtractedReasoning, ReasoningStep
from step_structurer import structure_reasoning_steps, StructuredReasoning, StructuredStep, StructuredExpression


class TestStepStructurer(unittest.TestCase):
    """Test suite for structure_reasoning_steps."""

    def test_simple_arithmetic_equalities(self):
        """Test extraction of simple arithmetic equations with numeric-only LHS and RHS."""
        text = (
            "Step 1: Calculate basic operations\n"
            "3 * 40 = 120\n"
            "120 + 30 = 150\n"
            "Final Answer: 150"
        )
        res = structure_reasoning_steps(text)
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.steps), 1)
        step = res.steps[0]
        self.assertEqual(len(step.expressions), 2)

        eq1 = step.expressions[0]
        self.assertEqual(eq1.expression_type, "arithmetic_equality")
        self.assertEqual(eq1.lhs_raw, "3 * 40")
        self.assertEqual(eq1.rhs_raw, "120")
        self.assertEqual(eq1.lhs_normalized, "3 * 40")
        self.assertEqual(eq1.rhs_normalized, "120")
        self.assertEqual(eq1.normalized_equation, "3 * 40 = 120")
        self.assertTrue(eq1.is_equality)
        self.assertTrue(eq1.is_valid_syntax)

        eq2 = step.expressions[1]
        self.assertEqual(eq2.expression_type, "arithmetic_equality")
        self.assertEqual(eq2.lhs_normalized, "120 + 30")
        self.assertEqual(eq2.rhs_normalized, "150")

    def test_multiple_equations_within_single_step(self):
        """Test that multiple assertions within one step preserve their source step and ordering."""
        text = (
            "Step 1: Notebook calculations\n"
            "Rate = 40\n"
            "Count = 3\n"
            "3 * 40 = 120\n"
            "Subtotal = 120\n"
            "Final Answer: 120"
        )
        res = structure_reasoning_steps(text)
        self.assertEqual(len(res.steps), 1)
        step = res.steps[0]
        self.assertEqual(len(step.expressions), 4)

        for expr in step.expressions:
            self.assertEqual(expr.step_number, 1)

        # Verify ordering
        self.assertEqual(step.expressions[0].lhs_raw, "Rate")
        self.assertEqual(step.expressions[1].lhs_raw, "Count")
        self.assertEqual(step.expressions[2].lhs_raw, "3 * 40")
        self.assertEqual(step.expressions[3].lhs_raw, "Subtotal")

    def test_equations_spread_across_multiple_steps(self):
        """Test extraction across multiple sequential steps retains the correct step number for each expression."""
        text = (
            "Step 1: First item\n"
            "3 * 40 = 120\n\n"
            "Step 2: Second item\n"
            "2 * 15 = 30\n\n"
            "Step 3: Total\n"
            "120 + 30 = 150\n\n"
            "Final Answer: 150"
        )
        res = structure_reasoning_steps(text)
        self.assertEqual(len(res.steps), 3)
        self.assertEqual(res.steps[0].expressions[0].step_number, 1)
        self.assertEqual(res.steps[1].expressions[0].step_number, 2)
        self.assertEqual(res.steps[2].expressions[0].step_number, 3)

    def test_arithmetic_operations_fractions_and_decimals(self):
        """Test operations: multiplication (×), division (÷), addition, subtraction, decimals, fractions."""
        text = (
            "Step 1: Diverse arithmetic operations\n"
            "5 × 4 = 20\n"
            "100 ÷ 4 = 25\n"
            "50 - 10 = 40\n"
            "4 * 1.5 = 6.0\n"
            "1/2 * 10 = 5\n"
            "Final Answer: 5"
        )
        res = structure_reasoning_steps(text)
        exprs = res.steps[0].expressions
        self.assertEqual(len(exprs), 5)

        # 5 × 4 = 20 -> 5 * 4 = 20
        self.assertEqual(exprs[0].lhs_normalized, "5 * 4")
        self.assertEqual(exprs[0].rhs_normalized, "20")

        # 100 ÷ 4 = 25 -> 100 / 4 = 25
        self.assertEqual(exprs[1].lhs_normalized, "100 / 4")
        self.assertEqual(exprs[1].rhs_normalized, "25")

        # 50 - 10 = 40
        self.assertEqual(exprs[2].lhs_normalized, "50 - 10")
        self.assertEqual(exprs[2].rhs_normalized, "40")

        # 4 * 1.5 = 6.0
        self.assertEqual(exprs[3].lhs_normalized, "4 * 1.5")
        self.assertEqual(exprs[3].rhs_normalized, "6.0")

        # 1/2 * 10 = 5
        self.assertEqual(exprs[4].lhs_normalized, "1/2 * 10")
        self.assertEqual(exprs[4].rhs_normalized, "5")

    def test_currency_symbols_and_units_cleaning(self):
        """Test that currency symbols ($100, ₹40) and unit words (mi/day, days) are stripped in normalized form."""
        text = (
            "Step 1: Currency and unit handling\n"
            "Cost = 3 × ₹40\n"
            "Distance = 15 mi/day * 5 days = 75 mi\n"
            "Final Answer: 75"
        )
        res = structure_reasoning_steps(text)
        exprs = res.steps[0].expressions

        # Currency stripped from RHS: 3 × ₹40 -> 3 * 40
        self.assertEqual(exprs[0].rhs_normalized, "3 * 40")
        self.assertEqual(exprs[0].label, "Cost")

        # Chained equation with units: Distance = 15 mi/day * 5 days = 75 mi
        # Pair 1: Distance = 15 mi/day * 5 days
        self.assertEqual(exprs[1].label, "Distance")
        self.assertEqual(exprs[1].rhs_normalized, "15 * 5")

        # Pair 2: 15 mi/day * 5 days = 75 mi
        self.assertEqual(exprs[2].expression_type, "arithmetic_equality")
        self.assertEqual(exprs[2].lhs_normalized, "15 * 5")
        self.assertEqual(exprs[2].rhs_normalized, "75")

    def test_natural_language_labels_on_lhs(self):
        """Test distinction between multi-word English labels and algebraic variables."""
        text = (
            "Step 1: Natural language label\n"
            "Cost of notebooks = 3 * 40\n"
            "x = 3 * 40\n"
            "Final Answer: 120"
        )
        res = structure_reasoning_steps(text)
        exprs = res.steps[0].expressions

        # Multi-word label
        nl_expr = exprs[0]
        self.assertEqual(nl_expr.expression_type, "labeled_equality")
        self.assertEqual(nl_expr.label, "Cost of notebooks")
        self.assertIsNone(nl_expr.lhs_normalized)
        self.assertEqual(nl_expr.rhs_normalized, "3 * 40")
        self.assertIn("LHS is a multi-word natural-language label", nl_expr.unsupported_reason)

        # Single variable identifier
        var_expr = exprs[1]
        self.assertEqual(var_expr.expression_type, "labeled_equality")
        self.assertEqual(var_expr.label, "x")
        self.assertEqual(var_expr.lhs_normalized, "x")
        self.assertEqual(var_expr.rhs_normalized, "3 * 40")
        self.assertIsNone(var_expr.unsupported_reason)

    def test_natural_language_formula_without_numbers(self):
        """Test conceptual formulas containing natural language text on both sides are marked unsupported."""
        text = (
            "Step 1: Formula definition\n"
            "Cost of notebooks = Number of notebooks × Cost per notebook\n"
            "Final Answer: 0"
        )
        res = structure_reasoning_steps(text)
        expr = res.steps[0].expressions[0]
        self.assertEqual(expr.expression_type, "natural_language_formula")
        self.assertFalse(expr.is_valid_syntax)
        self.assertIsNone(expr.lhs_normalized)
        self.assertIsNone(expr.rhs_normalized)
        self.assertIn("Natural-language formula", expr.unsupported_reason)

    def test_standalone_arithmetic_expressions(self):
        """Test standalone calculations that do not have an equality sign."""
        text = (
            "Step 1: Standalone arithmetic calculation\n"
            "3 * 40 + 2 * 15\n"
            "120 + 30\n"
            "Final Answer: 150"
        )
        res = structure_reasoning_steps(text)
        exprs = res.steps[0].expressions
        self.assertEqual(len(exprs), 2)

        self.assertEqual(exprs[0].expression_type, "standalone_expression")
        self.assertFalse(exprs[0].is_equality)
        self.assertEqual(exprs[0].rhs_normalized, "3 * 40 + 2 * 15")
        self.assertTrue(exprs[0].is_valid_syntax)

        self.assertEqual(exprs[1].expression_type, "standalone_expression")
        self.assertEqual(exprs[1].rhs_normalized, "120 + 30")

    def test_unsupported_or_malformed_mathematical_text(self):
        """Test unparseable syntax (e.g. broken parentheses) is preserved and marked unsupported."""
        text = (
            "Step 1: Malformed syntax\n"
            "3 * (40 + = 120\n"
            "120 = 3 * (40 +\n"
            "Final Answer: 120"
        )
        res = structure_reasoning_steps(text)
        expr_lhs_err = res.steps[0].expressions[0]
        self.assertEqual(expr_lhs_err.expression_type, "unsupported")
        self.assertFalse(expr_lhs_err.is_valid_syntax)
        self.assertIn("LHS contains invalid or unparseable", expr_lhs_err.unsupported_reason)

        expr_rhs_err = res.steps[0].expressions[1]
        self.assertEqual(expr_rhs_err.expression_type, "unsupported")
        self.assertFalse(expr_rhs_err.is_valid_syntax)
        self.assertIn("RHS contains invalid or unparseable", expr_rhs_err.unsupported_reason)

    def test_preservation_of_original_text_and_ordering(self):
        """Test that original raw clauses and order are 100% preserved."""
        text = (
            "Step 1: Order verification\n"
            "A = 10\n"
            "B = 20\n"
            "C = A + B = 30\n"
            "Final Answer: 30"
        )
        res = structure_reasoning_steps(text)
        exprs = res.steps[0].expressions
        self.assertEqual(len(exprs), 4)

        self.assertEqual(exprs[0].raw_text, "A = 10")
        self.assertEqual(exprs[1].raw_text, "B = 20")
        self.assertEqual(exprs[2].raw_text, "C = A + B")
        self.assertEqual(exprs[3].raw_text, "A + B = 30")

    def test_empty_or_malformed_extracted_reasoning(self):
        """Test graceful handling of empty or non-string inputs."""
        res_empty = structure_reasoning_steps("")
        self.assertEqual(res_empty.status, "empty_response")
        self.assertEqual(res_empty.steps, [])
        self.assertEqual(res_empty.total_expressions, 0)

        res_none = structure_reasoning_steps(None)
        self.assertEqual(res_none.status, "empty_response")

        res_no_steps = structure_reasoning_steps("Just plain narrative text without numbered steps.")
        self.assertEqual(res_no_steps.status, "no_steps_found")

    def test_actual_gemini_objective_18_demonstration_structuring(self):
        """Test structuring against the exact Gemini output from Objective 18/19."""
        actual_gemini_output = (
            "To find the total cost of the items purchased by the shopkeeper, we follow these steps:\n\n"
            "**Step 1: Calculate the total cost of the notebooks.**\n"
            "The shopkeeper buys 3 notebooks at ₹40 each.\n"
            "Cost of notebooks = Number of notebooks × Cost per notebook\n"
            "Cost of notebooks = 3 × ₹40\n"
            "Cost of notebooks = ₹120\n\n"
            "**Step 2: Calculate the total cost of the pens.**\n"
            "The shopkeeper buys 2 pens at ₹15 each.\n"
            "Cost of pens = Number of pens × Cost per pen\n"
            "Cost of pens = 2 × ₹15\n"
            "Cost of pens = ₹30\n\n"
            "**Step 3: Calculate the total cost of all items.**\n"
            "Total cost = Cost of notebooks + Cost of pens\n"
            "Total cost = ₹120 + ₹30\n"
            "Total cost = ₹150\n\n"
            "Final Answer: 150"
        )
        res = structure_reasoning_steps(actual_gemini_output)
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.steps), 3)
        self.assertEqual(res.total_expressions, 9)
        self.assertEqual(res.unsupported_expressions, 3)

        # Step 1 checks
        s1 = res.steps[0]
        self.assertEqual(s1.expressions[0].expression_type, "natural_language_formula")
        self.assertEqual(s1.expressions[1].expression_type, "labeled_equality")
        self.assertEqual(s1.expressions[1].rhs_normalized, "3 * 40")
        self.assertEqual(s1.expressions[2].expression_type, "labeled_equality")
        self.assertEqual(s1.expressions[2].rhs_normalized, "120")

        # Step 2 checks
        s2 = res.steps[1]
        self.assertEqual(s2.expressions[0].expression_type, "natural_language_formula")
        self.assertEqual(s2.expressions[1].expression_type, "labeled_equality")
        self.assertEqual(s2.expressions[1].rhs_normalized, "2 * 15")
        self.assertEqual(s2.expressions[2].expression_type, "labeled_equality")
        self.assertEqual(s2.expressions[2].rhs_normalized, "30")

        # Step 3 checks
        s3 = res.steps[2]
        self.assertEqual(s3.expressions[0].expression_type, "natural_language_formula")
        self.assertEqual(s3.expressions[1].expression_type, "labeled_equality")
        self.assertEqual(s3.expressions[1].rhs_normalized, "120 + 30")
        self.assertEqual(s3.expressions[2].expression_type, "labeled_equality")
        self.assertEqual(s3.expressions[2].rhs_normalized, "150")

        # Final answer check
        self.assertEqual(res.final_answer, "150")


if __name__ == "__main__":
    unittest.main()
