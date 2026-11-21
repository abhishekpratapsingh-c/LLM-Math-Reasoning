"""
test_deterministic_solver_pipeline.py
------------------------------------
Test suite for Phase 2, Objective 12: Deterministic, Offline Problem Solving Pipeline.

Tests:
1. Category A: Repeated rate / multiplicative total (Reference Q25 + variants)
2. Category B: Multi-item shopping / weighted sum (Reference Q20, Q27 + variants)
3. Category C: Unknown quantity with known total (Reference Q29 + variants)
4. Category D: Additive partition / segmented difference (Reference Q28 + variants)
5. Category E: Percentage discount / original price (Reference Q26 + variants)
6. False-Positive Tests: Questions containing keywords/numbers that must NOT match
7. Numeric Robustness: Decimals, commas, boundary discounts, negative/zero prices, stop overlaps
8. Direct unit tests for problem_patterns, equation_builder, and PipelineResult
"""

import unittest
from problem_patterns import (
    PatternMatchResult,
    _parse_num,
    match_repeated_rate,
    match_weighted_sum,
    match_unknown_quantity,
    match_additive_partition,
    match_percentage_discount,
    classify_problem,
)
from equation_builder import build_equation, _fmt_num
from deterministic_solver_pipeline import solve_problem, PipelineResult


class TestDeterministicPipeline(unittest.TestCase):

    # ==========================================================================
    # Category A: Repeated rate / multiplicative total
    # ==========================================================================
    def test_category_a_reference_q25(self):
        """James runs 3 sprints 3 times a week, 60 meters each sprint. Total meters = 540."""
        q = (
            "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. "
            "How many total meters does he run a week?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.detected_category, "repeated_rate")
        self.assertEqual(res.calculated_answer, 540)
        self.assertEqual(res.calculated_answer_str, "540")
        self.assertIn("3 * 3 * 60", res.constructed_equation)

    def test_category_a_variants(self):
        """Variant with 2 sprints 4 times a week, 50 meters each -> 400."""
        q = (
            "Alex decides to do 2 sprints 4 times a week. He runs 50 meters each sprint. "
            "How many total meters does he run a week?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.calculated_answer, 400)

        # 2-factor simple rate: "5 times a day. She drinks 2 cups each" -> 10
        q2 = "She drinks water 5 times a day. She drinks 2 cups each time. How many total cups does she drink a day?"
        res2 = solve_problem(q2)
        self.assertEqual(res2.status, "solved")
        self.assertEqual(res2.calculated_answer, 10)

    def test_category_a_decimals_and_commas(self):
        """Repeated rate with decimal factors and comma-separated thousands."""
        q1 = "James decides to run 2.5 sprints 4 times a week. He runs 60.5 meters each sprint. How many total meters does he run a week?"
        res1 = solve_problem(q1)
        self.assertEqual(res1.status, "solved")
        self.assertEqual(res1.calculated_answer, 605)

        q2 = "James decides to run 3 sprints 3 times a week. He runs 1,000 meters each sprint. How many total meters does he run a week?"
        res2 = solve_problem(q2)
        self.assertEqual(res2.status, "solved")
        self.assertEqual(res2.calculated_answer, 9000)

    # ==========================================================================
    # Category B: Multi-item shopping / weighted sum
    # ==========================================================================
    def test_category_b_reference_q20(self):
        """3 dozen donuts at $68, 2 dozen cupcakes at $80, 6 dozen cheesecakes at $55 -> 694."""
        q = (
            "She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, "
            "and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.detected_category, "weighted_sum")
        self.assertEqual(res.calculated_answer, 694)
        self.assertEqual(res.calculated_answer_str, "694")
        self.assertIn("3 * 68", res.constructed_equation)
        self.assertIn("2 * 80", res.constructed_equation)
        self.assertIn("6 * 55", res.constructed_equation)

    def test_category_b_variant_q27(self):
        """Mishka bought 3 pairs of shorts, 2 pairs of pants. Shorts cost $16.50, pants cost $22.50. Total = 94.5."""
        q = (
            "Mishka bought 3 pairs of shorts and 2 pairs of pants. "
            "One pair of shorts costs $16.50 and one pair of pants costs $22.50. "
            "How much did Mishka spend on all the clothes in total?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.detected_category, "weighted_sum")
        self.assertAlmostEqual(res.calculated_answer, 94.5)

    def test_category_b_decimals_and_commas(self):
        """Shopping with decimals and comma-separated prices."""
        q1 = "She bought 2.5 dozen donuts which cost $10 per dozen, and 1.5 dozen mini cupcakes which cost $20 per dozen. How much was the total cost?"
        res1 = solve_problem(q1)
        self.assertEqual(res1.status, "solved")
        self.assertEqual(res1.calculated_answer, 55)

        q2 = "She bought 2 laptops which cost $1,200 each, and 3 phones which cost $800 each. How much was the total cost?"
        res2 = solve_problem(q2)
        self.assertEqual(res2.status, "solved")
        self.assertEqual(res2.calculated_answer, 4800)

    # ==========================================================================
    # Category C: Unknown quantity with known total
    # ==========================================================================
    def test_category_c_reference_q29(self):
        """Chicken meal $12, 5 packs milk at $3, 4 apples at $1.50, total $50, pizza boxes at $8.50 -> 2."""
        q = (
            "Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each, "
            "4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. "
            "How many boxes of pizza did Marie order if each box costs $8.50?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.detected_category, "unknown_quantity")
        self.assertEqual(res.calculated_answer, 2)
        self.assertEqual(res.calculated_answer_str, "2")
        self.assertIn("8.5 * x", res.constructed_equation)
        self.assertIn("= 50", res.constructed_equation)

    def test_category_c_variant(self):
        """1 meal $10, 2 sodas at $2 each, total $30, unknown tickets at $8 each -> (30 - 14) / 8 = 2."""
        q = (
            "Bob ordered one ticket meal that costs $10, 2 packs of soda that costs $2 each, "
            "and some movie passes. Bob paid a total of $30. "
            "How many movie passes did Bob order if each pass costs $8?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.calculated_answer, 2)

    def test_category_c_commas(self):
        """Unknown quantity with comma-separated numbers: $1,000 meal, 2 gifts at $500, total $3,000, unknown $250 -> 4."""
        q = (
            "Marie ordered one meal that costs $1,000, 2 gifts that cost $500 each, "
            "and some paintings. Marie paid a total of $3,000. "
            "How many paintings did Marie order if each painting costs $250?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.calculated_answer, 4)

    # ==========================================================================
    # Category D: Additive partition / segmented difference
    # ==========================================================================
    def test_category_d_reference_q28(self):
        """Henry 60-mile bike trip. Stopped after 20 miles. 2nd stop was 15 miles before end -> 25."""
        q = (
            "Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. "
            "His second stop was 15 miles before the end of the trip. "
            "How many miles did he travel between his first and second stops?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.detected_category, "additive_partition")
        self.assertEqual(res.calculated_answer, 25)
        self.assertEqual(res.calculated_answer_str, "25")
        self.assertIn("20 + x + 15 = 60", res.constructed_equation)

    def test_category_d_variant(self):
        """100-mile trip, first stop 30 miles, second stop 25 miles before end -> 45."""
        q = (
            "Sarah made two stops during her 100-mile bike trip. She first stopped after 30 miles. "
            "Her second stop was 25 miles before the end of the trip. "
            "How many miles did she travel between his first and second stops?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.calculated_answer, 45)

    def test_category_d_overlapping_or_zero_segment(self):
        """Stops that exceed or equal the total trip length must be rejected as unsupported."""
        # Overlapping: 40 + 30 = 70 > 60 -> impossible segment
        q1 = (
            "Henry made two stops during his 60-mile bike trip. He first stopped after 40 miles. "
            "His second stop was 30 miles before the end of the trip. "
            "How many miles did he travel between his first and second stops?"
        )
        res1 = solve_problem(q1)
        self.assertEqual(res1.status, "unsupported")

        # Zero distance: 35 + 25 = 60 -> intermediate distance 0
        q2 = (
            "Henry made two stops during his 60-mile bike trip. He first stopped after 35 miles. "
            "His second stop was 25 miles before the end of the trip. "
            "How many miles did he travel between his first and second stops?"
        )
        res2 = solve_problem(q2)
        self.assertEqual(res2.status, "unsupported")

    # ==========================================================================
    # Category E: Percentage discount / original price
    # ==========================================================================
    def test_category_e_reference_q26(self):
        """Book bought for $19.50 with 25% discount. Original price = 26."""
        q = (
            "Kyle bought last year's best-selling book for $19.50. "
            "This is with a 25% discount from the original price. "
            "What was the original price of the book?"
        )
        res = solve_problem(q)
        self.assertEqual(res.status, "solved")
        self.assertEqual(res.detected_category, "percentage_discount")
        self.assertEqual(res.calculated_answer, 26)
        self.assertEqual(res.calculated_answer_str, "26")
        self.assertIn("1 - 25/100", res.constructed_equation)
        self.assertIn("19.5", res.constructed_equation)

    def test_category_e_variant_and_commas(self):
        """Item bought for $80 with a 20% discount -> Original price = 100."""
        q1 = (
            "Ann purchased a winter coat for $80. "
            "This is with a 20% discount from the original price. "
            "What was the original price of the coat?"
        )
        res1 = solve_problem(q1)
        self.assertEqual(res1.status, "solved")
        self.assertEqual(res1.calculated_answer, 100)

        # Designer watch for $1,200 with 20% discount -> 1500
        q2 = (
            "Kyle bought a designer watch for $1,200. "
            "This is with a 20% discount from the original price. "
            "What was the original price of the watch?"
        )
        res2 = solve_problem(q2)
        self.assertEqual(res2.status, "solved")
        self.assertEqual(res2.calculated_answer, 1500)

    # ==========================================================================
    # False-Positive Audit Tests
    # ==========================================================================
    def test_false_positive_shopping_remaining_items(self):
        """A shopping question asking how many items remain must NOT match weighted sum."""
        q = "She bought 3 donuts for $5 each and ate 1. How many donuts remain?"
        res = solve_problem(q)
        self.assertEqual(res.status, "unsupported")

    def test_false_positive_running_duration(self):
        """A running question asking for duration/time must NOT match repeated rate."""
        q = "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many hours does it take?"
        res = solve_problem(q)
        self.assertEqual(res.status, "unsupported")

    def test_false_positive_discount_amount(self):
        """A discount question asking for discount savings must NOT match percentage discount."""
        q = "Kyle bought last year's best-selling book for $19.50. This is with a 25% discount from the original price. What was the discount amount in dollars?"
        res = solve_problem(q)
        self.assertEqual(res.status, "unsupported")

    def test_false_positive_distance_speed(self):
        """A distance question asking for speed rather than intermediate segment must NOT match."""
        q = "Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How fast was he cycling on average?"
        res = solve_problem(q)
        self.assertEqual(res.status, "unsupported")

    def test_false_positive_multiple_prices_single_item(self):
        """A question with multiple prices asking for a single item price must NOT match."""
        q = "She bought 3 donuts for $6 and 2 cupcakes for $8. What was the price of one cupcake?"
        res = solve_problem(q)
        self.assertEqual(res.status, "unsupported")

    # ==========================================================================
    # Boundary & Numeric Robustness Tests
    # ==========================================================================
    def test_boundary_zero_and_hundred_percent_discount(self):
        """0% and 100% discounts are invalid boundaries for percentage_discount."""
        q0 = "Kyle bought a book for $20. This is with a 0% discount from the original price. What was the original price of the book?"
        self.assertEqual(solve_problem(q0).status, "unsupported")

        q100 = "Kyle got a book for $0. This is with a 100% discount from the original price. What was the original price of the book?"
        self.assertEqual(solve_problem(q100).status, "unsupported")

    def test_zero_or_negative_price_rejection(self):
        """Zero or negative prices must be rejected."""
        q_neg = "She bought 2 books which cost $-10 each, and 3 pens which cost $5 each. How much was the total cost?"
        self.assertEqual(solve_problem(q_neg).status, "unsupported")

        q_zero = "She bought 2 books which cost $0 each, and 3 pens which cost $5 each. How much was the total cost?"
        self.assertEqual(solve_problem(q_zero).status, "unsupported")

    def test_missing_or_ambiguous_quantities(self):
        """Incomplete or ambiguous shopping/rates must return unsupported."""
        q_missing = "She bought 3 donuts and 2 cupcakes. How much was the total cost?"
        self.assertEqual(solve_problem(q_missing).status, "unsupported")

        q_ambig = "Marie ordered some boxes of pizza and paid $50. How many boxes of pizza did Marie order?"
        self.assertEqual(solve_problem(q_ambig).status, "unsupported")

    # ==========================================================================
    # General Unsupported and Invalid Input Tests
    # ==========================================================================
    def test_unsupported_complex_reasoning(self):
        """Questions that do not fit templates should be marked unsupported, not crash."""
        q = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with 4. How many does she have left?"
        res = solve_problem(q)
        self.assertEqual(res.status, "unsupported")
        self.assertIsNone(res.calculated_answer)
        self.assertIsNone(res.constructed_equation)

    def test_unsupported_general_question(self):
        """Non-mathematical question."""
        q = "What is the capital of France?"
        res = solve_problem(q)
        self.assertEqual(res.status, "unsupported")

    def test_invalid_empty_input(self):
        """Empty or non-string inputs should return invalid status."""
        self.assertEqual(solve_problem("").status, "invalid")
        self.assertEqual(solve_problem("   ").status, "invalid")
        self.assertEqual(solve_problem(None).status, "invalid")
        self.assertEqual(solve_problem(12345).status, "invalid")

    # ==========================================================================
    # Unit Tests for Helper Modules
    # ==========================================================================
    def test_number_formatting(self):
        self.assertEqual(_fmt_num(10.0), "10")
        self.assertEqual(_fmt_num(10), "10")
        self.assertEqual(_fmt_num(12.5), "12.5")
        self.assertEqual(_fmt_num(0.0), "0")
        self.assertEqual(_fmt_num(1200.0), "1200")

    def test_parse_num(self):
        self.assertEqual(_parse_num("five"), 5.0)
        self.assertEqual(_parse_num("dozen"), 12.0)
        self.assertEqual(_parse_num("12.5"), 12.5)
        self.assertEqual(_parse_num("1,200"), 1200.0)
        self.assertEqual(_parse_num("3,000.50"), 3000.5)
        self.assertIsNone(_parse_num("foobar"))
        self.assertIsNone(_parse_num(""))

    def test_builder_invalid_inputs(self):
        """build_equation should return None on invalid or empty match results."""
        self.assertIsNone(build_equation(None))
        self.assertIsNone(build_equation(PatternMatchResult(category="unknown", matched=False)))
        self.assertIsNone(build_equation(PatternMatchResult(category="repeated_rate", matched=True, extracted_data={})))
        self.assertIsNone(build_equation(PatternMatchResult(category="weighted_sum", matched=True, extracted_data={"items": []})))
        self.assertIsNone(build_equation(PatternMatchResult(category="unknown_quantity", matched=True, extracted_data={})))
        self.assertIsNone(build_equation(PatternMatchResult(category="additive_partition", matched=True, extracted_data={})))
        self.assertIsNone(build_equation(PatternMatchResult(category="percentage_discount", matched=True, extracted_data={})))

    def test_pipeline_result_to_dict(self):
        """PipelineResult should correctly serialize to dictionary."""
        res = PipelineResult(
            status="solved",
            original_question="test question",
            detected_category="repeated_rate",
            extracted_data={"factors": [2, 3]},
            constructed_equation="x = 2 * 3",
            calculated_answer=6,
            calculated_answer_str="6",
            explanation="test explanation"
        )
        d = res.to_dict()
        self.assertEqual(d["status"], "solved")
        self.assertEqual(d["calculated_answer"], 6)
        self.assertEqual(d["constructed_equation"], "x = 2 * 3")


if __name__ == "__main__":
    unittest.main()
