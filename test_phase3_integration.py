"""
test_phase3_integration.py
--------------------------
Phase 3 End-to-End Integration Test: Automatic Error Detection in LLM Reasoning.

Validates the complete 10-stage mathematical reasoning error detection pipeline:
1. Original mathematical problem input.
2. Static, mocked LLM reasoning response.
3. Intermediate step extraction (step_extractor.py).
4. Structured mathematical representation (step_structurer.py).
5. Individual SymPy statement verification (step_verifier.py).
6. Consecutive step transition verification (step_transition_verifier.py).
7. Consolidated error detection and issue classification (reasoning_error_detector.py).
8. Feedback prompt formatting and mocked LLM explanation generation (reasoning_feedback.py).
9. Final comprehensive verification report generation (verification_report.py).
10. Metric calculations, status precedence, and Markdown report rendering.

Covers all four required end-to-end scenarios:
- Scenario A: Completely Correct Reasoning (verified, 0 errors, 0 warnings, 0 API calls).
- Scenario B: Incorrect Arithmetic Assertion (errors_detected, incorrect_assertion, solver evidence preserved).
- Scenario C: Lost Solution via Variable Division (errors_detected, division_by_variable, lost root x=0).
- Scenario D: Conditional Warning after Squaring Both Sides (warnings, squaring_both_sides, extraneous roots).

Zero live API calls. Zero external network overhead.
"""

import unittest
from unittest.mock import MagicMock, patch

from step_extractor import extract_reasoning_steps
from step_structurer import structure_reasoning_steps
from step_verifier import verify_structured_reasoning
from step_transition_verifier import verify_step_transitions
from reasoning_error_detector import detect_reasoning_errors
from reasoning_feedback import generate_reasoning_feedback
from verification_report import generate_verification_report, format_markdown_report


class TestPhase3Integration(unittest.TestCase):
    """End-to-end integration tests verifying data flow across all Phase 3 modules."""

    # --------------------------------------------------------------------------
    # Scenario A: Completely Correct Reasoning
    # --------------------------------------------------------------------------
    def test_scenario_a_correct_reasoning(self):
        """
        Scenario A:
        Problem: A shopkeeper buys 3 notebooks for ₹40 each and 2 pens for ₹15 each. Find the total cost.
        Reasoning:
          Step 1: Cost of notebooks = 3 * 40 = 120
          Step 2: Cost of pens = 2 * 15 = 30
          Step 3: Total cost = 120 + 30 = 150
          Final Answer: 150
        Expected:
          - No definite errors.
          - No conditional warnings.
          - No LLM feedback API call.
          - Final report status: verified.
        """
        problem = "A shopkeeper buys 3 notebooks for ₹40 each and 2 pens for ₹15 each. Find the total cost."
        raw_reasoning = (
            "Step 1: Cost of notebooks = 3 * 40 = 120\n\n"
            "Step 2: Cost of pens = 2 * 15 = 30\n\n"
            "Step 3: Total cost = 120 + 30 = 150\n\n"
            "Final Answer: 150"
        )

        # 1. Step Extraction
        extracted = extract_reasoning_steps(raw_reasoning)
        self.assertEqual(len(extracted.steps), 3)
        self.assertEqual(extracted.final_answer, "150")

        # 2. Step Structuring
        structured = structure_reasoning_steps(extracted)
        self.assertEqual(len(structured.steps), 3)
        self.assertEqual(structured.total_expressions, 6)

        # 3. Individual Statement Verification
        verified = verify_structured_reasoning(structured)
        self.assertEqual(verified.total_expressions, 6)
        self.assertEqual(verified.verified_count, 6)
        self.assertEqual(verified.incorrect_count, 0)
        self.assertEqual(verified.invalid_count, 0)
        self.assertEqual(verified.unsupported_count, 0)

        # 4. Step Transition Verification
        transitions = verify_step_transitions(structured, verified)
        self.assertTrue(len(transitions.transitions) >= 2)
        # All transitions within steps are valid; transitions across steps are independent calculations
        self.assertEqual(transitions.invalid_transitions, 0)

        # 5. Consolidated Error Detection
        errors = detect_reasoning_errors(structured, verified, transitions)
        self.assertEqual(errors.total_definite_errors, 0)
        self.assertEqual(errors.total_potential_issues, 0)
        self.assertEqual(errors.status, "correct")
        self.assertEqual(len(errors.flagged_error_steps), 0)
        self.assertEqual(len(errors.flagged_warning_steps), 0)

        # 6. LLM Feedback Generation (Verify ZERO API Calls)
        mock_client = MagicMock()
        feedback = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
            client=mock_client,
        )
        # STRICT VERIFICATION: Exactly 0 API calls made for sound reasoning
        self.assertEqual(len(mock_client.method_calls), 0)
        self.assertFalse(feedback.has_issues)
        self.assertEqual(feedback.status, "no_issues")
        self.assertEqual(feedback.total_feedbacks, 0)

        # 7. Final Verification Report
        report = generate_verification_report(
            problem=problem,
            raw_response=raw_reasoning,
            extracted_reasoning=extracted,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
            feedback_result=feedback,
        )

        # 8. Report Validation
        self.assertEqual(report.summary.overall_status, "verified")
        self.assertEqual(report.summary.total_steps, 3)
        self.assertEqual(report.summary.total_expressions, 6)
        self.assertEqual(report.summary.total_verified_expressions, 6)
        self.assertEqual(report.summary.total_detected_errors, 0)
        self.assertEqual(report.summary.total_warnings, 0)
        self.assertEqual(len(report.detected_errors), 0)
        self.assertEqual(len(report.detected_warnings), 0)
        self.assertEqual(report.problem_info.final_answer, "150")

        # 9. Markdown Report Validation
        md_report = report.to_markdown()
        self.assertIn("verified", md_report)
        self.assertIn("No mathematical errors, invalid transformations, or warnings were detected.", md_report)
        self.assertIn("Reasoning is mathematically sound. Zero LLM explanations required.", md_report)

    # --------------------------------------------------------------------------
    # Scenario B: Incorrect Arithmetic Assertion
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_scenario_b_incorrect_arithmetic(self, mock_call_llm):
        """
        Scenario B:
        Problem: A shopkeeper buys 3 notebooks for ₹40 each. Find the total cost.
        Reasoning:
          Step 1: Cost of notebooks = 3 * 40 = 130
          Final Answer: 130
        Expected:
          - Incorrect arithmetic assertion detected.
          - Solver evidence identifies discrepancy (LHS 120 != RHS 130).
          - Mocked LLM explains the error.
          - Final report status: errors_detected.
        """
        mock_call_llm.return_value = (
            "Step 1 is incorrect because the arithmetic computation 3 * 40 equals 120, not 130. "
            "To correct it, replace 130 with 120."
        )
        problem = "A shopkeeper buys 3 notebooks for ₹40 each. Find the total cost."
        raw_reasoning = (
            "Step 1: Cost of notebooks = 3 * 40 = 130\n\n"
            "Final Answer: 130"
        )

        # Pipeline stages 1 through 7
        extracted = extract_reasoning_steps(raw_reasoning)
        structured = structure_reasoning_steps(extracted)
        verified = verify_structured_reasoning(structured)
        transitions = verify_step_transitions(structured, verified)
        errors = detect_reasoning_errors(structured, verified, transitions)

        self.assertEqual(errors.total_definite_errors, 1)
        self.assertEqual(errors.status, "has_errors")
        self.assertIn(1, errors.flagged_error_steps)

        # Feedback generation with mock client
        mock_client = MagicMock()
        feedback = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
            client=mock_client,
        )

        self.assertTrue(feedback.has_issues)
        self.assertEqual(feedback.total_feedbacks, 1)
        fb_item = feedback.feedbacks[0]
        self.assertEqual(fb_item.step_number, 1)
        self.assertEqual(fb_item.error_category, "incorrect_assertion")
        self.assertEqual(fb_item.severity, "definite_error")
        self.assertIn("120", fb_item.solver_evidence)
        self.assertIn("130", fb_item.solver_evidence)
        self.assertIn("3 * 40 equals 120, not 130", fb_item.llm_explanation)

        # Final Verification Report
        report = generate_verification_report(
            problem=problem,
            raw_response=raw_reasoning,
            extracted_reasoning=extracted,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
            feedback_result=feedback,
        )

        self.assertEqual(report.summary.overall_status, "errors_detected")
        self.assertEqual(report.summary.total_detected_errors, 1)
        self.assertEqual(len(report.detected_errors), 1)

        # Strict evidence separation check
        err_rec = report.detected_errors[0]
        self.assertEqual(err_rec.error_category, "incorrect_assertion")
        self.assertTrue(err_rec.is_definite_error)
        self.assertIn("120", err_rec.explanation)

        llm_rec = report.llm_explanations[0]
        self.assertEqual(llm_rec.error_category, "incorrect_assertion")
        self.assertEqual(llm_rec.llm_explanation, fb_item.llm_explanation)
        self.assertNotEqual(llm_rec.solver_evidence, llm_rec.llm_explanation)

        # Markdown verification
        md_report = report.to_markdown()
        self.assertIn("errors_detected", md_report)
        self.assertIn("Definite Mathematical Errors", md_report)
        self.assertIn("3 * 40 = 130", md_report)

    # --------------------------------------------------------------------------
    # Scenario C: Lost Solution via Variable Division
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_scenario_c_lost_solution(self, mock_call_llm):
        """
        Scenario C:
        Problem: Find all solutions to x**2 = 2*x.
        Reasoning:
          Step 1: x**2 = 2*x
          Step 2: x = 2
          Final Answer: 2
        Expected:
          - Invalid transformation detected.
          - Lost solution x = 0 identified.
          - Solver domain/condition evidence preserved.
          - Mocked LLM explains why division by x loses a solution.
          - Final report status: errors_detected.
        """
        mock_call_llm.return_value = (
            "Dividing both sides by the variable x eliminates the valid root x = 0. "
            "The equation should be factored as x*(x - 2) = 0 to yield both x = 0 and x = 2."
        )
        problem = "Find all solutions to x**2 = 2*x."
        raw_reasoning = (
            "Step 1: x**2 = 2*x\n\n"
            "Step 2: x = 2\n\n"
            "Final Answer: 2"
        )

        # Pipeline execution
        extracted = extract_reasoning_steps(raw_reasoning)
        structured = structure_reasoning_steps(extracted)
        verified = verify_structured_reasoning(structured)
        transitions = verify_step_transitions(structured, verified)
        errors = detect_reasoning_errors(structured, verified, transitions)

        self.assertEqual(errors.total_definite_errors, 1)
        self.assertEqual(errors.status, "has_errors")
        self.assertIn(2, errors.flagged_error_steps)

        mock_client = MagicMock()
        feedback = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
            client=mock_client,
        )

        self.assertTrue(feedback.has_issues)
        self.assertEqual(feedback.total_feedbacks, 1)
        fb_item = feedback.feedbacks[0]
        self.assertEqual(fb_item.step_number, 2)
        self.assertEqual(fb_item.error_category, "lost_solution")
        self.assertEqual(fb_item.severity, "definite_error")
        self.assertIn("0", fb_item.solver_evidence)
        self.assertIn("losing solution", fb_item.solver_evidence.lower())
        self.assertTrue(any("x != 0" in c for c in fb_item.conditions))
        self.assertIn("eliminates the valid root x = 0", fb_item.llm_explanation)

        report = generate_verification_report(
            problem=problem,
            raw_response=raw_reasoning,
            extracted_reasoning=extracted,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
            feedback_result=feedback,
        )

        self.assertEqual(report.summary.overall_status, "errors_detected")
        self.assertEqual(report.summary.total_detected_errors, 1)
        self.assertTrue(any(e.error_category == "lost_solution" for e in report.detected_errors))

        # Check conditions preserved
        err_item = next(e for e in report.detected_errors if e.error_category == "lost_solution")
        self.assertTrue(any("x != 0" in c for c in err_item.conditions))

        # Markdown report check
        md_report = report.to_markdown()
        self.assertIn("errors_detected", md_report)
        self.assertIn("lost_solution", md_report)
        self.assertIn("x != 0", md_report)

    # --------------------------------------------------------------------------
    # Scenario D: Conditional Warning after Squaring Both Sides
    # --------------------------------------------------------------------------
    @patch("reasoning_feedback.call_llm")
    def test_scenario_d_conditional_warning(self, mock_call_llm):
        """
        Scenario D:
        Problem: Solve x = 3.
        Reasoning:
          Step 1: x = 3
          Step 2: x**2 = 9
          Final Answer: 3
        Expected:
          - Squaring transformation recognized.
          - Conditional warning about possible extraneous roots.
          - No false definite-error classification solely because of squaring.
          - Final report status: warnings.
        """
        mock_call_llm.return_value = (
            "Squaring both sides is a one-way implication that introduces extraneous roots (x = -3). "
            "All candidate solutions must be checked against the original equation."
        )
        problem = "Solve x = 3."
        raw_reasoning = (
            "Step 1: x = 3\n\n"
            "Step 2: x**2 = 9\n\n"
            "Final Answer: 3"
        )

        # Pipeline execution
        extracted = extract_reasoning_steps(raw_reasoning)
        structured = structure_reasoning_steps(extracted)
        verified = verify_structured_reasoning(structured)
        transitions = verify_step_transitions(structured, verified)
        errors = detect_reasoning_errors(structured, verified, transitions)

        # Must NOT be classified as definite error
        self.assertEqual(errors.total_definite_errors, 0)
        self.assertEqual(errors.total_potential_issues, 1)
        self.assertEqual(errors.status, "has_warnings")
        self.assertEqual(len(errors.flagged_error_steps), 0)
        self.assertIn(2, errors.flagged_warning_steps)

        mock_client = MagicMock()
        feedback = generate_reasoning_feedback(
            problem=problem,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
            client=mock_client,
        )

        self.assertTrue(feedback.has_issues)
        self.assertEqual(feedback.total_feedbacks, 1)
        fb_item = feedback.feedbacks[0]
        self.assertEqual(fb_item.step_number, 2)
        self.assertEqual(fb_item.error_category, "conditional_warning")
        self.assertEqual(fb_item.severity, "potential_issue")
        self.assertTrue("square" in fb_item.solver_evidence.lower() or "squared" in fb_item.solver_evidence.lower())
        self.assertTrue(len(fb_item.conditions) > 0)
        self.assertIn("extraneous roots", fb_item.llm_explanation.lower())

        report = generate_verification_report(
            problem=problem,
            raw_response=raw_reasoning,
            extracted_reasoning=extracted,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
            feedback_result=feedback,
        )

        self.assertEqual(report.summary.overall_status, "warnings")
        self.assertEqual(report.summary.total_detected_errors, 0)
        self.assertEqual(report.summary.total_warnings, 1)
        self.assertEqual(len(report.detected_warnings), 1)
        self.assertEqual(len(report.detected_errors), 0)

        # Markdown report check
        md_report = report.to_markdown()
        self.assertIn("warnings", md_report)
        self.assertIn("Conditional Warnings", md_report)
        self.assertIn("extraneous roots", md_report)

    # --------------------------------------------------------------------------
    # Data Flow & Single-Count Consolidation Validation
    # --------------------------------------------------------------------------
    def test_data_flow_and_no_double_counting(self):
        """
        Validates that intermediate data structures flow end-to-end without loss,
        and that overlapping assertion and transition findings are never double-counted.
        """
        problem = "Evaluate 3 * 40 = 130 and 2 * 15 = 40."
        raw_reasoning = (
            "Step 1: Calculate first product\n"
            "3 * 40 = 130\n\n"
            "Step 2: Calculate second product\n"
            "2 * 15 = 40\n\n"
            "Final Answer: 170"
        )

        # Execute end-to-end
        extracted = extract_reasoning_steps(raw_reasoning)
        structured = structure_reasoning_steps(extracted)
        verified = verify_structured_reasoning(structured)
        transitions = verify_step_transitions(structured, verified)
        errors = detect_reasoning_errors(structured, verified, transitions)

        # Exactly 2 errors detected, exactly 1 per step, no double-counting
        self.assertEqual(errors.total_definite_errors, 2)
        self.assertEqual(len(errors.step_reports), 2)
        self.assertEqual(len(errors.step_reports[0].issues), 1)
        self.assertEqual(len(errors.step_reports[1].issues), 1)

        report = generate_verification_report(
            problem=problem,
            raw_response=raw_reasoning,
            extracted_reasoning=extracted,
            structured_reasoning=structured,
            verified_reasoning=verified,
            transition_verification=transitions,
            error_report=errors,
        )

        self.assertEqual(report.summary.total_detected_errors, 2)
        self.assertEqual(len(report.detected_errors), 2)
        # Original text preserved
        self.assertEqual(report.steps[0].original_text, extracted.steps[0].text)
        self.assertEqual(report.steps[1].original_text, extracted.steps[1].text)

    # --------------------------------------------------------------------------
    # Focused Audit: Cross-Step Transition Classification (Cases A, B, C, D)
    # --------------------------------------------------------------------------
    def test_cross_step_case_a_independent_calculations(self):
        """Case A: Independent calculations across steps should be unrelated, not an error."""
        prob = "Calculate costs."
        raw = "Step 1: 3 * 40 = 120\n\nStep 2: 2 * 15 = 30\n\nFinal Answer: 150"
        ext = extract_reasoning_steps(raw)
        struct = structure_reasoning_steps(ext)
        ver = verify_structured_reasoning(struct)
        trans = verify_step_transitions(struct, ver)

        self.assertEqual(len(trans.transitions), 1)
        t = trans.transitions[0]
        self.assertEqual(t.transition_status, "unsupported")
        self.assertEqual(t.transformation_type, "unrelated")
        self.assertIn("independent numerical calculations", t.explanation.lower())

        err = detect_reasoning_errors(struct, ver, trans)
        self.assertEqual(err.total_definite_errors, 0)
        self.assertEqual(err.total_potential_issues, 0)
        self.assertEqual(err.status, "correct")

        rep = generate_verification_report(prob, raw_response=raw, error_report=err)
        self.assertEqual(rep.summary.overall_status, "verified")

    def test_cross_step_case_b_invalid_transformation(self):
        """Case B: Invalid algebraic transformation across steps must be flagged as invalid, not unrelated."""
        prob = "Solve x + 4 = 10."
        raw = "Step 1: x + 4 = 10\n\nStep 2: x = 20\n\nFinal Answer: 20"
        ext = extract_reasoning_steps(raw)
        struct = structure_reasoning_steps(ext)
        ver = verify_structured_reasoning(struct)
        trans = verify_step_transitions(struct, ver)

        self.assertEqual(len(trans.transitions), 1)
        t = trans.transitions[0]
        self.assertEqual(t.transition_status, "invalid")
        self.assertEqual(t.transformation_type, "incorrect_transformation")
        self.assertIn("6", t.explanation)
        self.assertIn("20", t.explanation)

        err = detect_reasoning_errors(struct, ver, trans)
        self.assertEqual(err.total_definite_errors, 1)
        self.assertEqual(err.status, "has_errors")

        rep = generate_verification_report(prob, raw_response=raw, error_report=err)
        self.assertEqual(rep.summary.overall_status, "errors_detected")

    def test_cross_step_case_c_valid_transformation(self):
        """Case C: Valid algebraic transformation across steps must be valid."""
        prob = "Solve 2*x + 4 = 10."
        raw = "Step 1: 2*x + 4 = 10\n\nStep 2: 2*x = 6\n\nFinal Answer: 3"
        ext = extract_reasoning_steps(raw)
        struct = structure_reasoning_steps(ext)
        ver = verify_structured_reasoning(struct)
        trans = verify_step_transitions(struct, ver)

        self.assertEqual(len(trans.transitions), 1)
        t = trans.transitions[0]
        self.assertEqual(t.transition_status, "valid")
        self.assertTrue(t.transformation_type in ("algebraic_rearrangement", "both_sides_operation"))

        err = detect_reasoning_errors(struct, ver, trans)
        self.assertEqual(err.total_definite_errors, 0)
        self.assertEqual(err.total_potential_issues, 0)

        rep = generate_verification_report(prob, raw_response=raw, error_report=err)
        self.assertEqual(rep.summary.overall_status, "verified")

    def test_cross_step_case_d_lost_solution(self):
        """Case D: Dividing by a variable across steps must detect lost solution x = 0."""
        prob = "Solve x**2 = 2*x."
        raw = "Step 1: x**2 = 2*x\n\nStep 2: x = 2\n\nFinal Answer: 2"
        ext = extract_reasoning_steps(raw)
        struct = structure_reasoning_steps(ext)
        ver = verify_structured_reasoning(struct)
        trans = verify_step_transitions(struct, ver)

        self.assertEqual(len(trans.transitions), 1)
        t = trans.transitions[0]
        self.assertEqual(t.transition_status, "invalid")
        self.assertEqual(t.transformation_type, "division_by_variable")
        self.assertIn("0", t.explanation)

        err = detect_reasoning_errors(struct, ver, trans)
        self.assertEqual(err.total_definite_errors, 1)
        self.assertEqual(err.step_reports[1].issues[0].error_category, "lost_solution")

        rep = generate_verification_report(prob, raw_response=raw, error_report=err)
        self.assertEqual(rep.summary.overall_status, "errors_detected")

    # --------------------------------------------------------------------------
    # Focused Audit: Step Prefix Cleaning Safety
    # --------------------------------------------------------------------------
    def test_step_prefix_cleaning_audit_categories(self):
        """
        Verifies that step prefix cleaning does not damage:
        1. Variable names (s, step, steps, x)
        2. Exponentiation (x**2, x^2)
        3. Negative numbers (-x, x = -3, -5 = -5)
        4. Parentheses (2*(x + 3) = 10, (x + 1)**2 = 4)
        5. Chained equalities (Cost = 3 * 40 = 120)
        6. Mathematical labels (Step 1: Cost of notebooks = 3 * 40)
        """
        # 1. Variables starting with 's' or named 'step'
        s1 = structure_reasoning_steps("Step 1: s = 5\nFinal Answer: 5")
        self.assertEqual(s1.steps[0].expressions[0].lhs_normalized, "s")

        s2 = structure_reasoning_steps("Step 1: step = 5\nFinal Answer: 5")
        self.assertEqual(s2.steps[0].expressions[0].lhs_normalized, "step")

        s3 = structure_reasoning_steps("Step 1: steps = 10\nFinal Answer: 10")
        self.assertEqual(s3.steps[0].expressions[0].lhs_normalized, "steps")

        # 2. Exponentiation preserved
        s4 = structure_reasoning_steps("Step 1: x**2 = 4\nFinal Answer: 2")
        self.assertEqual(s4.steps[0].expressions[0].lhs_normalized, "x**2")
        self.assertEqual(s4.steps[0].expressions[0].rhs_normalized, "4")

        s5 = structure_reasoning_steps("Step 1: x^2 = 9\nFinal Answer: 3")
        self.assertEqual(s5.steps[0].expressions[0].lhs_normalized, "x**2")

        # 3. Negative numbers preserved
        s6 = structure_reasoning_steps("Step 1: x = -3\nFinal Answer: -3")
        self.assertEqual(s6.steps[0].expressions[0].rhs_normalized, "-3")

        s7 = structure_reasoning_steps("Step 1: -x = 5\nFinal Answer: -5")
        self.assertEqual(s7.steps[0].expressions[0].lhs_normalized, "-x")

        # 4. Parentheses preserved
        s8 = structure_reasoning_steps("Step 1: 2*(x + 3) = 10\nFinal Answer: 2")
        self.assertIn("x + 3", s8.steps[0].expressions[0].lhs_normalized)

        s9 = structure_reasoning_steps("Step 1: (x + 1)**2 = 4\nFinal Answer: 1")
        self.assertIn("x + 1", s9.steps[0].expressions[0].lhs_normalized)

        # 5. Chained equalities preserved
        s10 = structure_reasoning_steps("Step 1: Cost = 3 * 40 = 120\nFinal Answer: 120")
        self.assertEqual(len(s10.steps[0].expressions), 2)
        self.assertEqual(s10.steps[0].expressions[0].label, "Cost")
        self.assertEqual(s10.steps[0].expressions[1].lhs_normalized, "3 * 40")
        self.assertEqual(s10.steps[0].expressions[1].rhs_normalized, "120")

        # 6. Labels with step prefix preserved cleanly
        s11 = structure_reasoning_steps("Step 1: Cost of notebooks = 3 * 40\nFinal Answer: 120")
        self.assertEqual(s11.steps[0].expressions[0].label, "Cost of notebooks")


if __name__ == "__main__":
    unittest.main()
