"""
test_solver_edge_cases.py
-------------------------
Phase 2, Objective 14: Challenging Mathematical Edge Cases to Identify Solver Limitations.

Independent test suite evaluating the solver and verifier across five critical mathematical areas:
1. Fractions: fractional coefficients, fraction arithmetic, fractional solutions.
2. Negative numbers: negative coefficients, negative solutions, double-negatives.
3. Division by zero: direct zero denominators, undefined expressions, no-solution rational forms.
4. Squaring both sides & extraneous solutions: quadratic roots, radical equations, extraneous verification.
5. Variable restrictions & solver limitations: denominators with variables, removable singularities,
   domain restriction preservation.

Each test explicitly documents:
- Mathematical input
- Expected mathematical behavior
- Actual behavior of the existing solver / verifier
- Whether the case matches ideal expectations or reveals an architectural limitation.
"""

import unittest
from symbolic_solver import solve_equation, verify_solution, simplify_expression
from symbolic_verifier import verify_equation


class TestSolverEdgeCases(unittest.TestCase):

    # ==========================================================================
    # 1. Fractions
    # ==========================================================================
    def test_frac_1_simple_fraction_coefficient(self):
        """Mathematical Input: 1/2 * x = 5
        Expected Behavior: Solves to exact integer x = 10.
        """
        res = solve_equation("1/2 * x = 5")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["10"])

    def test_frac_2_fraction_addition_in_equation(self):
        """Mathematical Input: x + 1/3 = 1
        Expected Behavior: Solves to exact rational x = 2/3.
        """
        res = solve_equation("x + 1/3 = 1")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["2/3"])

    def test_frac_3_fractional_solution_from_integers(self):
        """Mathematical Input: 3 * x = 2
        Expected Behavior: Solves to exact non-integer rational x = 2/3.
        """
        res = solve_equation("3 * x = 2")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["2/3"])

    def test_frac_4_complex_fraction_simplification(self):
        """Mathematical Input: (1/3 + 1/6) / (1/2)
        Expected Behavior: (2/6 + 1/6) / (1/2) = (1/2) / (1/2) = 1.
        """
        res = simplify_expression("(1/3 + 1/6) / (1/2)")
        self.assertEqual(res.status, "simplified")
        self.assertEqual(res.simplified_expression, "1")

    def test_frac_5_fractions_on_both_sides(self):
        """Mathematical Input: 2/3 * x - 1/4 = 1/6 * x + 1/2
        Expected Behavior: (2/3 - 1/6)x = 1/2 + 1/4 -> 1/2 * x = 3/4 -> x = 3/2.
        """
        res = solve_equation("2/3 * x - 1/4 = 1/6 * x + 1/2")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["3/2"])

    # ==========================================================================
    # 2. Negative Numbers
    # ==========================================================================
    def test_neg_1_negative_coefficient(self):
        """Mathematical Input: -2 * x = 10
        Expected Behavior: Solves to negative integer x = -5.
        """
        res = solve_equation("-2 * x = 10")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["-5"])

    def test_neg_2_negative_coefficient_and_constant(self):
        """Mathematical Input: -3 * x - 5 = 10
        Expected Behavior: -3*x = 15 -> x = -5.
        """
        res = solve_equation("-3 * x - 5 = 10")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["-5"])

    def test_neg_3_negative_solution_from_positive_coefficients(self):
        """Mathematical Input: x + 10 = 3
        Expected Behavior: x = 3 - 10 = -7.
        """
        res = solve_equation("x + 10 = 3")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["-7"])

    def test_neg_4_subtraction_of_negative_value(self):
        """Mathematical Input: x - (-5) = 12
        Expected Behavior: x + 5 = 12 -> x = 7.
        """
        res = solve_equation("x - (-5) = 12")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["7"])

    def test_neg_5_negative_fraction_multiplication(self):
        """Mathematical Input: -1/2 * x = -3/4
        Expected Behavior: x = (-3/4) / (-1/2) = 3/2.
        """
        res = solve_equation("-1/2 * x = -3/4")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["3/2"])

    def test_neg_6_verify_negative_solution(self):
        """Mathematical Input: Verify x = -5 in -2 * x = 10
        Expected Behavior: Correct (-2 * -5 = 10 == 10).
        """
        res = verify_solution("-2 * x = 10", "-5")
        self.assertEqual(res.status, "correct")

    def test_neg_7_verify_rejects_wrong_sign(self):
        """Mathematical Input: Verify x = 5 in -2 * x = 10
        Expected Behavior: Incorrect (-10 != 10).
        """
        res = verify_solution("-2 * x = 10", "5")
        self.assertEqual(res.status, "incorrect")

    # ==========================================================================
    # 3. Division by Zero
    # ==========================================================================
    def test_zero_1_direct_division_by_zero_verifier(self):
        """Mathematical Input: 10 / 0 = 5
        Expected Behavior: Invalid equation (division by zero is undefined).
        """
        res = verify_equation("10 / 0 = 5")
        self.assertEqual(res.status, "invalid")
        self.assertIn("Division by zero", res.explanation)

    def test_zero_2_zero_denominator_in_simplification(self):
        """Mathematical Input: simplify 5 / (3 - 3)
        Expected Behavior: Invalid expression (division by zero).
        """
        res = simplify_expression("5 / (3 - 3)")
        self.assertEqual(res.status, "invalid")
        self.assertIn("Division by zero", res.explanation)

    def test_zero_3_variable_divided_by_zero_in_solver(self):
        """Mathematical Input: x / 0 = 5
        Expected Behavior: Invalid equation (division by zero).
        """
        res = solve_equation("x / 0 = 5")
        self.assertEqual(res.status, "invalid")
        self.assertIn("Division by zero", res.explanation)

    def test_zero_4_rational_equation_no_solution(self):
        """Mathematical Input: 1 / (x - 2) = 0
        Expected Behavior: No solution (1 / (x - 2) is never 0).
        """
        res = solve_equation("1 / (x - 2) = 0")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "no_solution")
        self.assertEqual(res.solutions, [])

    def test_zero_5_zero_divided_by_zero_verifier(self):
        """Mathematical Input: 0 / 0 = 0
        Expected Behavior: Invalid equation (indeterminate / undefined).
        """
        res = verify_equation("0 / 0 = 0")
        self.assertEqual(res.status, "invalid")
        self.assertIn("Division by zero", res.explanation)

    # ==========================================================================
    # 4. Squaring Both Sides & Extraneous Solutions
    # ==========================================================================
    def test_sq_1_quadratic_two_valid_roots(self):
        """Mathematical Input: x**2 = 16
        Expected Behavior: Two real roots: -4 and 4.
        """
        res = solve_equation("x**2 = 16")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "multiple")
        self.assertEqual(res.solutions, ["-4", "4"])

    def test_sq_2_verifier_rejects_false_quadratic_candidate(self):
        """Mathematical Input: Verify x = 5 in x**2 = 16
        Expected Behavior: Incorrect (5**2 = 25 != 16).
        """
        res = verify_solution("x**2 = 16", "5")
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.evaluated_lhs, "25")
        self.assertEqual(res.evaluated_rhs, "16")

    def test_sq_3_radical_equation_without_real_solution(self):
        """Mathematical Input: x**(1/2) = -2
        Expected Behavior: No solution (principal square root is non-negative).
        Squaring both sides naively would give extraneous root x = 4.
        """
        res = solve_equation("x**(1/2) = -2")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "no_solution")
        self.assertEqual(res.solutions, [])

    def test_sq_4_verification_catches_extraneous_root_from_squaring(self):
        """Mathematical Input: Verify x = 4 in x**(1/2) = -2
        Expected Behavior: Incorrect (LHS evaluates to 2 != -2).
        """
        res = verify_solution("x**(1/2) = -2", "4")
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.evaluated_lhs, "2")
        self.assertEqual(res.evaluated_rhs, "-2")

    def test_sq_5_radical_equation_eliminating_extraneous_root(self):
        """Mathematical Input: (x + 5)**(1/2) = x - 1
        Squaring yields x + 5 = x^2 - 2x + 1 -> x^2 - 3x - 4 = 0 -> (x-4)(x+1) = 0.
        Candidate roots are 4 and -1.
        At x = -1: sqrt(4) = 2, RHS = -2 (extraneous).
        At x = 4: sqrt(9) = 3, RHS = 3 (valid).
        Expected Behavior: Unique solution x = 4.
        """
        res = solve_equation("(x + 5)**(1/2) = x - 1")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["4"])

    def test_sq_6_rational_extraneous_root_elimination(self):
        """Mathematical Input: 1 / (x - 1) = 1 / (x**2 - 1)
        Cross multiplying yields x^2 - 1 = x - 1 -> x(x - 1) = 0.
        At x = 1: both denominators are 0 (extraneous singularity).
        At x = 0: 1/(-1) = 1/(-1) = -1 (valid).
        Expected Behavior: Unique solution x = 0.
        """
        res = solve_equation("1 / (x - 1) = 1 / (x**2 - 1)")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["0"])

    # ==========================================================================
    # 5. Variable Restrictions & Solver Limitations
    # ==========================================================================
    def test_rest_1_variable_in_denominator_valid_root(self):
        """Mathematical Input: 1 / x = 2
        Expected Behavior: Unique solution x = 1/2.
        """
        res = solve_equation("1 / x = 2")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["1/2"])

    def test_rest_2_domain_restriction_removable_singularity(self):
        """Mathematical Input: (x**2 - 1) / (x - 1) = 2
        Expected Behavior: No solution because x = 1 makes the original denominator x - 1 = 0.
        The solver identifies the domain restriction x != 1 and discards the candidate root x = 1.
        """
        res = solve_equation("(x**2 - 1) / (x - 1) = 2")
        self.assertEqual(res.solution_type, "no_solution")
        self.assertEqual(res.solutions, [])
        self.assertIn("x != 1", res.domain_restrictions)

    def test_rest_3_verifier_catches_removable_singularity(self):
        """Mathematical Input: Verify proposed solution x = 1 in (x**2 - 1) / (x - 1) = 2
        Expected Behavior: Verification must fail because x = 1 produces nan/zero-division.
        """
        res = verify_solution("(x**2 - 1) / (x - 1) = 2", "1")
        self.assertEqual(res.status, "incorrect")
        self.assertEqual(res.evaluated_lhs, "nan")

    def test_rest_4_rational_identity_domain_preservation(self):
        """Mathematical Input: x / x = 1
        Expected Behavior: Identity for all x != 0, explicitly preserving the domain restriction.
        """
        res = solve_equation("x / x = 1")
        self.assertEqual(res.solution_type, "infinite_solutions")
        self.assertIn("x != 0", res.domain_restrictions)
        self.assertIn("x != 0", res.explanation)

    def test_rest_5_expression_simplification_domain_preservation(self):
        """Mathematical Input: simplify x / x
        Expected Behavior: Simplifies to 1 with explicit domain restriction x != 0.
        """
        res = simplify_expression("x / x")
        self.assertEqual(res.simplified_expression, "1")
        self.assertIn("x != 0", res.domain_restrictions)
        self.assertIn("x != 0", res.explanation)

    def test_rest_6_complex_roots_polynomial(self):
        """Mathematical Input: x**2 = -4
        Expected Behavior: Two imaginary roots: -2*I and 2*I.
        """
        res = solve_equation("x**2 = -4")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "multiple")
        self.assertEqual(res.solutions, ["-2*I", "2*I"])

    def test_rest_7_multiple_denominator_restrictions(self):
        """Mathematical Input: 1/x + 1/(x - 1) = 0
        Expected Behavior: Denominators require x != 0 and x != 1.
        Solves to (x - 1 + x)/(x(x - 1)) = (2x - 1)/(x(x - 1)) = 0 -> x = 1/2.
        1/2 is non-zero and != 1, so unique solution is 1/2.
        """
        res = solve_equation("1/x + 1/(x - 1) = 0")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["1/2"])
        self.assertIn("x != 0", res.domain_restrictions)
        self.assertIn("x != 1", res.domain_restrictions)

    def test_rest_8_constant_denominators_no_restrictions(self):
        """Mathematical Input: x/2 + x/3 = 5
        Expected Behavior: Constant denominators 2 and 3 introduce no variable domain restrictions.
        Solves to 5/6 * x = 5 -> x = 6.
        """
        res = solve_equation("x/2 + x/3 = 5")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["6"])
        self.assertEqual(res.domain_restrictions, [])

    def test_rest_9_simplification_cancellation_preserves_restriction(self):
        """Mathematical Input: simplify (x**2 - 4) / (x - 2)
        Expected Behavior: Simplifies to x + 2 while recording domain restriction x != 2.
        """
        res = simplify_expression("(x**2 - 4) / (x - 2)")
        self.assertEqual(res.status, "simplified")
        self.assertEqual(res.simplified_expression, "x + 2")
        self.assertIn("x != 2", res.domain_restrictions)
        self.assertIn("x != 2", res.explanation)

    def test_rest_10_rational_equation_valid_cancellation_solution(self):
        """Mathematical Input: (x**2 - 4) / (x - 2) = 5
        Expected Behavior: (x - 2)(x + 2)/(x - 2) = 5 -> x + 2 = 5 -> x = 3.
        Since 3 != 2, x = 3 is valid.
        """
        res = solve_equation("(x**2 - 4) / (x - 2) = 5")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "unique")
        self.assertEqual(res.solutions, ["3"])
        self.assertIn("x != 2", res.domain_restrictions)

    def test_rest_11_rational_equation_extraneous_cancellation_solution(self):
        """Mathematical Input: (x**2 - 4) / (x - 2) = 4
        Expected Behavior: (x - 2)(x + 2)/(x - 2) = 4 -> x + 2 = 4 -> x = 2.
        Candidate x = 2 violates domain restriction x != 2 (division by zero), so no solution.
        """
        res = solve_equation("(x**2 - 4) / (x - 2) = 4")
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.solution_type, "no_solution")
        self.assertEqual(res.solutions, [])
        self.assertIn("x != 2", res.domain_restrictions)


# ==============================================================================
# Standalone Audit Runner & Report Generator
# ==============================================================================
def run_edge_case_audit():
    """Runs all 26 mathematical edge cases and generates a structured audit report."""
    print("=" * 75)
    print("PHASE 2, OBJECTIVE 14: MATHEMATICAL SOLVER LIMITATIONS & EDGE-CASE AUDIT")
    print("=" * 75)

    cases = [
        # 1. Fractions
        ("Fractions", "1/2 * x = 5", "Exact integer root x = 10",
         lambda: solve_equation("1/2 * x = 5").solutions == ["10"]),
        ("Fractions", "x + 1/3 = 1", "Exact fractional root x = 2/3",
         lambda: solve_equation("x + 1/3 = 1").solutions == ["2/3"]),
        ("Fractions", "3 * x = 2", "Fractional solution x = 2/3",
         lambda: solve_equation("3 * x = 2").solutions == ["2/3"]),
        ("Fractions", "(1/3 + 1/6) / (1/2)", "Simplifies to exact 1",
         lambda: simplify_expression("(1/3 + 1/6) / (1/2)").simplified_expression == "1"),
        ("Fractions", "2/3 * x - 1/4 = 1/6 * x + 1/2", "Fractions on both sides -> x = 3/2",
         lambda: solve_equation("2/3 * x - 1/4 = 1/6 * x + 1/2").solutions == ["3/2"]),

        # 2. Negative Numbers
        ("Negatives", "-2 * x = 10", "Negative root x = -5",
         lambda: solve_equation("-2 * x = 10").solutions == ["-5"]),
        ("Negatives", "-3 * x - 5 = 10", "Negative root x = -5",
         lambda: solve_equation("-3 * x - 5 = 10").solutions == ["-5"]),
        ("Negatives", "x + 10 = 3", "Negative root x = -7",
         lambda: solve_equation("x + 10 = 3").solutions == ["-7"]),
        ("Negatives", "x - (-5) = 12", "Double-negative subtraction -> x = 7",
         lambda: solve_equation("x - (-5) = 12").solutions == ["7"]),
        ("Negatives", "-1/2 * x = -3/4", "Negative fraction division -> x = 3/2",
         lambda: solve_equation("-1/2 * x = -3/4").solutions == ["3/2"]),
        ("Negatives", "verify -2 * x = 10 with x = -5", "Validates true negative solution",
         lambda: verify_solution("-2 * x = 10", "-5").status == "correct"),
        ("Negatives", "verify -2 * x = 10 with x = 5", "Rejects wrong sign candidate",
         lambda: verify_solution("-2 * x = 10", "5").status == "incorrect"),

        # 3. Division by Zero
        ("DivZero", "10 / 0 = 5", "Rejects division by zero as invalid",
         lambda: verify_equation("10 / 0 = 5").status == "invalid"),
        ("DivZero", "simplify 5 / (3 - 3)", "Rejects zero denominator expression",
         lambda: simplify_expression("5 / (3 - 3)").status == "invalid"),
        ("DivZero", "x / 0 = 5", "Rejects zero divisor in equation",
         lambda: solve_equation("x / 0 = 5").status == "invalid"),
        ("DivZero", "1 / (x - 2) = 0", "Correctly reports no_solution",
         lambda: solve_equation("1 / (x - 2) = 0").solution_type == "no_solution"),
        ("DivZero", "0 / 0 = 0", "Rejects indeterminate 0/0 form",
         lambda: verify_equation("0 / 0 = 0").status == "invalid"),

        # 4. Squaring / Extraneous Solutions
        ("Squaring", "x**2 = 16", "Finds both roots [-4, 4]",
         lambda: solve_equation("x**2 = 16").solutions == ["-4", "4"]),
        ("Squaring", "verify x = 5 in x**2 = 16", "Rejects non-root candidate",
         lambda: verify_solution("x**2 = 16", "5").status == "incorrect"),
        ("Squaring", "x**(1/2) = -2", "Detects no real solution for radical",
         lambda: solve_equation("x**(1/2) = -2").solution_type == "no_solution"),
        ("Squaring", "verify x = 4 in x**(1/2) = -2", "Rejects extraneous squared candidate",
         lambda: verify_solution("x**(1/2) = -2", "4").status == "incorrect"),
        ("Squaring", "(x + 5)**(1/2) = x - 1", "Eliminates extraneous -1, keeps 4",
         lambda: solve_equation("(x + 5)**(1/2) = x - 1").solutions == ["4"]),
        ("Squaring", "1/(x-1) = 1/(x**2-1)", "Eliminates singularity x=1, keeps 0",
         lambda: solve_equation("1 / (x - 1) = 1 / (x**2 - 1)").solutions == ["0"]),

        # 5. Variable Restrictions & Domain Preservation
        ("Restrictions", "1 / x = 2", "Solves non-zero denominator -> x = 1/2",
         lambda: solve_equation("1 / x = 2").solutions == ["1/2"]),
        ("Restrictions", "(x**2 - 1) / (x - 1) = 2", "Discards extraneous singularity x=1 -> no_solution",
         lambda: solve_equation("(x**2 - 1) / (x - 1) = 2").solution_type == "no_solution"),
        ("Restrictions", "verify x=1 in (x**2-1)/(x-1)=2", "Verifier catches nan / zero division",
         lambda: verify_solution("(x**2 - 1) / (x - 1) = 2", "1").status == "incorrect"),
        ("Restrictions", "x / x = 1", "Preserves identity domain restriction x != 0",
         lambda: "x != 0" in solve_equation("x / x = 1").explanation and "x != 0" in solve_equation("x / x = 1").domain_restrictions),
        ("Restrictions", "simplify x / x", "Preserves simplifier domain restriction x != 0",
         lambda: "x != 0" in simplify_expression("x / x").explanation and "x != 0" in simplify_expression("x / x").domain_restrictions),
        ("Restrictions", "1/x + 1/(x - 1) = 0", "Multiple restrictions: x != 0, x != 1 -> solves to 1/2",
         lambda: solve_equation("1/x + 1/(x - 1) = 0").solutions == ["1/2"] and "x != 0" in solve_equation("1/x + 1/(x - 1) = 0").domain_restrictions and "x != 1" in solve_equation("1/x + 1/(x - 1) = 0").domain_restrictions),
        ("Restrictions", "x/2 + x/3 = 5", "Constant denominators introduce no restrictions -> x = 6",
         lambda: solve_equation("x/2 + x/3 = 5").solutions == ["6"] and solve_equation("x/2 + x/3 = 5").domain_restrictions == []),
        ("Restrictions", "simplify (x**2 - 4) / (x - 2)", "Simplifies to x + 2 for x != 2",
         lambda: simplify_expression("(x**2 - 4) / (x - 2)").simplified_expression == "x + 2" and "x != 2" in simplify_expression("(x**2 - 4) / (x - 2)").domain_restrictions),
        ("Restrictions", "(x**2 - 4) / (x - 2) = 5", "Valid cancellation solution x = 3 != 2",
         lambda: solve_equation("(x**2 - 4) / (x - 2) = 5").solutions == ["3"] and "x != 2" in solve_equation("(x**2 - 4) / (x - 2) = 5").domain_restrictions),
        ("Restrictions", "(x**2 - 4) / (x - 2) = 4", "Discards extraneous cancellation root x = 2 -> no_solution",
         lambda: solve_equation("(x**2 - 4) / (x - 2) = 4").solution_type == "no_solution" and solve_equation("(x**2 - 4) / (x - 2) = 4").solutions == []),
    ]

    passed_count = 0
    failed_count = 0
    results_by_area = {}

    for area, inp, exp, check_fn in cases:
        if area not in results_by_area:
            results_by_area[area] = {"passed": 0, "failed": 0, "total": 0}
        results_by_area[area]["total"] += 1

        try:
            ok = check_fn()
        except Exception:
            ok = False

        if ok:
            passed_count += 1
            results_by_area[area]["passed"] += 1
            status = "PASS (Correct)"
        else:
            failed_count += 1
            results_by_area[area]["failed"] += 1
            status = "FAIL (Limitation)"

        print(f"[{status:<17}] {area:<14} | Input: {inp:<30} | Expected: {exp}")

    print("\n" + "=" * 75)
    print("AUDIT SUMMARY:")
    print("=" * 75)
    print(f"Total Edge Cases Tested:           {len(cases)}")
    print(f"Cases Matching Ideal Expectations: {passed_count} ({passed_count/len(cases)*100:.1f}%)")
    print(f"Identified Solver Limitations:     {failed_count} ({failed_count/len(cases)*100:.1f}%)")
    print("-" * 75)
    print("Breakdown by Mathematical Area:")
    for area, st in results_by_area.items():
        print(f"  - {area:<14}: {st['passed']}/{st['total']} ideal ({st['failed']} limitations identified)")
    print("=" * 75)


if __name__ == "__main__":
    unittest.main()
