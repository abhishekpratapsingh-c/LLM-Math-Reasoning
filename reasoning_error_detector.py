"""
reasoning_error_detector.py
----------------------------
Phase 3: Automatic Error Detection in LLM Reasoning.
Objective 23: Identify and flag incorrect mathematical reasoning steps.

Combines:
1. Structured reasoning representations from Objective 20 (step_structurer.py).
2. Individual SymPy-based mathematical assertions from Objective 21 (step_verifier.py).
3. Mathematical transformation and step transitions from Objective 22 (step_transition_verifier.py).

Produces a comprehensive, consolidated mathematical error detection report:
- Identifies and flags incorrect reasoning steps.
- Distinguishes definite errors from conditional warnings and unsupported expressions.
- Categorizes issues:
    * 'incorrect_assertion': false mathematical assertion (e.g. 3 * 40 = 130).
    * 'invalid_transformation': false algebraic leap (e.g. 2x + 4 = 10 -> 2x = 14).
    * 'division_by_zero': undefined operation (e.g. 10 / 0 = 5).
    * 'lost_solution': division by a variable that eliminates roots (e.g. x^2 = 2x -> x = 2).
    * 'domain_violation': candidate solution violating an accumulated restriction (e.g. x = 1 when x != 1).
    * 'conditional_warning': one-way implication (e.g. squaring both sides).
    * 'unsupported': natural-language formula or unparseable syntax.
    * 'verified': mathematically sound step with no detected issues.
- Consolidates overlapping assertion and transition findings without duplicate reporting.
- Preserves original step ordering, step numbers, and mathematical expressions.
"""

from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_extractor import ExtractedReasoning, ReasoningStep, extract_reasoning_steps
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


# ==============================================================================
# Error Detection Dataclasses
# ==============================================================================
@dataclass
class DetectedIssue:
    """Represents a specific mathematical issue, error, warning, or verification finding for an expression."""
    step_number: int
    expression: str
    error_category: str                 # 'incorrect_assertion', 'invalid_transformation',
                                        # 'division_by_zero', 'lost_solution', 'domain_violation',
                                        # 'conditional_warning', 'unsupported', 'verified'
    verification_status: str            # 'verified', 'incorrect', 'invalid', 'unsupported', 'warning'
    is_definite_error: bool             # True for definite errors, False for warnings/unsupported/verified
    severity: str                       # 'definite_error', 'potential_issue', 'unsupported', 'none'
    explanation: str
    conditions: List[str] = field(default_factory=list)
    individual_status: Optional[str] = None
    transition_status: Optional[str] = None
    incoming_from_expression: Optional[str] = None
    incoming_from_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "expression": self.expression,
            "error_category": self.error_category,
            "verification_status": self.verification_status,
            "is_definite_error": self.is_definite_error,
            "severity": self.severity,
            "explanation": self.explanation,
            "conditions": self.conditions,
            "individual_status": self.individual_status,
            "transition_status": self.transition_status,
            "incoming_from_expression": self.incoming_from_expression,
            "incoming_from_step": self.incoming_from_step,
        }


@dataclass
class StepErrorReport:
    """Consolidated error report for a single reasoning step."""
    step_number: int
    step_header: str
    original_text: str
    has_definite_errors: bool
    has_potential_issues: bool
    has_unsupported: bool
    issues: List[DetectedIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_header": self.step_header,
            "original_text": self.original_text,
            "has_definite_errors": self.has_definite_errors,
            "has_potential_issues": self.has_potential_issues,
            "has_unsupported": self.has_unsupported,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class ReasoningErrorReport:
    """Top-level error detection report across all reasoning steps."""
    step_reports: List[StepErrorReport] = field(default_factory=list)
    total_steps: int = 0
    total_expressions: int = 0
    total_definite_errors: int = 0
    total_potential_issues: int = 0
    total_unsupported: int = 0
    total_verified: int = 0
    status: str = "correct"             # 'correct', 'has_errors', 'has_warnings', 'unsupported', 'empty'
    flagged_error_steps: List[int] = field(default_factory=list)
    flagged_warning_steps: List[int] = field(default_factory=list)
    summary_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_reports": [sr.to_dict() for sr in self.step_reports],
            "total_steps": self.total_steps,
            "total_expressions": self.total_expressions,
            "total_definite_errors": self.total_definite_errors,
            "total_potential_issues": self.total_potential_issues,
            "total_unsupported": self.total_unsupported,
            "total_verified": self.total_verified,
            "status": self.status,
            "flagged_error_steps": self.flagged_error_steps,
            "flagged_warning_steps": self.flagged_warning_steps,
            "summary_explanation": self.summary_explanation,
        }


# ==============================================================================
# Issue Classification and Consolidation
# ==============================================================================
def _classify_expression_issue(
    v_expr: Union[VerifiedExpression, StructuredExpression],
    incoming_trans: Optional[StepTransition],
    step_number: int,
) -> DetectedIssue:
    """
    Consolidates individual verification findings and incoming transition verification
    into a single, clear DetectedIssue, avoiding duplicate reporting.
    """
    expr_text = v_expr.raw_text
    indiv_status = getattr(v_expr, "verification_status", "verified")
    indiv_expl = getattr(v_expr, "explanation", "")
    indiv_conds = getattr(v_expr, "domain_restrictions", [])

    trans_status = incoming_trans.transition_status if incoming_trans else None
    trans_type = incoming_trans.transformation_type if incoming_trans else None
    trans_expl = incoming_trans.explanation if incoming_trans else ""
    trans_conds = incoming_trans.conditions if incoming_trans else []
    prev_expr_text = incoming_trans.prev_expression if incoming_trans else None
    prev_step_num = incoming_trans.prev_step_number if incoming_trans else None

    # --------------------------------------------------------------------------
    # 1. Division by Zero (Definite Error)
    # --------------------------------------------------------------------------
    is_indiv_divzero = indiv_status == "invalid" and ("zero" in indiv_expl.lower() or "division" in indiv_expl.lower())
    is_trans_divzero = trans_status == "invalid" and trans_type == "division_by_zero"

    if is_indiv_divzero or is_trans_divzero:
        expl = indiv_expl if is_indiv_divzero else trans_expl
        return DetectedIssue(
            step_number=step_number,
            expression=expr_text,
            error_category="division_by_zero",
            verification_status="invalid",
            is_definite_error=True,
            severity="definite_error",
            explanation=f"Mathematical error: division by zero is undefined ({expl}).",
            conditions=["Division by zero is undefined."],
            individual_status=indiv_status,
            transition_status=trans_status,
            incoming_from_expression=prev_expr_text,
            incoming_from_step=prev_step_num,
        )

    # --------------------------------------------------------------------------
    # 2. Domain Restriction Violation (Definite Error)
    # --------------------------------------------------------------------------
    if trans_type == "domain_violation":
        all_conds = list(dict.fromkeys(indiv_conds + trans_conds))
        return DetectedIssue(
            step_number=step_number,
            expression=expr_text,
            error_category="domain_violation",
            verification_status="invalid",
            is_definite_error=True,
            severity="definite_error",
            explanation=trans_expl,
            conditions=all_conds,
            individual_status=indiv_status,
            transition_status=trans_status,
            incoming_from_expression=prev_expr_text,
            incoming_from_step=prev_step_num,
        )

    # --------------------------------------------------------------------------
    # 3. Lost Solution after Division by Variable (Definite Error)
    # --------------------------------------------------------------------------
    if trans_type == "division_by_variable" and trans_status == "invalid":
        all_conds = list(dict.fromkeys(indiv_conds + trans_conds))
        return DetectedIssue(
            step_number=step_number,
            expression=expr_text,
            error_category="lost_solution",
            verification_status="invalid",
            is_definite_error=True,
            severity="definite_error",
            explanation=trans_expl,
            conditions=all_conds,
            individual_status=indiv_status,
            transition_status=trans_status,
            incoming_from_expression=prev_expr_text,
            incoming_from_step=prev_step_num,
        )

    # --------------------------------------------------------------------------
    # 4. Incorrect Mathematical Assertion (Definite Error)
    # --------------------------------------------------------------------------
    # Consolidates cases where individual assertion is false (e.g. 3 * 40 = 130)
    # Even if transition also caught it, report as one consolidated assertion error
    if indiv_status == "incorrect":
        consolidated_expl = f"Mathematical assertion is incorrect: {indiv_expl}"
        if trans_status == "invalid" and trans_type != "division_by_variable":
            consolidated_expl += f" (Incoming transition from '{prev_expr_text}' is also invalid)."

        all_conds = list(dict.fromkeys(indiv_conds + trans_conds))
        return DetectedIssue(
            step_number=step_number,
            expression=expr_text,
            error_category="incorrect_assertion",
            verification_status="incorrect",
            is_definite_error=True,
            severity="definite_error",
            explanation=consolidated_expl,
            conditions=all_conds,
            individual_status=indiv_status,
            transition_status=trans_status,
            incoming_from_expression=prev_expr_text,
            incoming_from_step=prev_step_num,
        )

    # --------------------------------------------------------------------------
    # 5. Invalid Transformation (Definite Error)
    # --------------------------------------------------------------------------
    # Expression itself might be a valid formula/equation, but doesn't follow from previous step
    if trans_status == "invalid":
        all_conds = list(dict.fromkeys(indiv_conds + trans_conds))
        return DetectedIssue(
            step_number=step_number,
            expression=expr_text,
            error_category="invalid_transformation",
            verification_status="invalid",
            is_definite_error=True,
            severity="definite_error",
            explanation=trans_expl,
            conditions=all_conds,
            individual_status=indiv_status,
            transition_status=trans_status,
            incoming_from_expression=prev_expr_text,
            incoming_from_step=prev_step_num,
        )

    # --------------------------------------------------------------------------
    # 6. One-Way Transformation / Conditional Warning (Potential Issue)
    # --------------------------------------------------------------------------
    # e.g. Squaring both sides: A = B => A^2 = B^2 (may introduce extraneous roots)
    # Not definitely incorrect, but preserved as a conditional warning
    if trans_type == "squaring_both_sides" or any("extraneous" in str(c).lower() for c in trans_conds):
        all_conds = list(dict.fromkeys(indiv_conds + trans_conds))
        return DetectedIssue(
            step_number=step_number,
            expression=expr_text,
            error_category="conditional_warning",
            verification_status="warning",
            is_definite_error=False,
            severity="potential_issue",
            explanation=trans_expl,
            conditions=all_conds,
            individual_status=indiv_status,
            transition_status=trans_status,
            incoming_from_expression=prev_expr_text,
            incoming_from_step=prev_step_num,
        )

    # --------------------------------------------------------------------------
    # 7. Unsupported Expression or Transition (Unsupported)
    # --------------------------------------------------------------------------
    # Natural-language formula, unparseable math, or unrelated independent statement
    if indiv_status == "unsupported" or (trans_status == "unsupported" and trans_type != "unrelated"):
        expl = indiv_expl if indiv_status == "unsupported" else trans_expl
        return DetectedIssue(
            step_number=step_number,
            expression=expr_text,
            error_category="unsupported",
            verification_status="unsupported",
            is_definite_error=False,
            severity="unsupported",
            explanation=expl,
            conditions=[],
            individual_status=indiv_status,
            transition_status=trans_status,
            incoming_from_expression=prev_expr_text,
            incoming_from_step=prev_step_num,
        )

    # --------------------------------------------------------------------------
    # 8. Verified (No Error or Warning Detected)
    # --------------------------------------------------------------------------
    all_conds = list(dict.fromkeys(indiv_conds + trans_conds))
    return DetectedIssue(
        step_number=step_number,
        expression=expr_text,
        error_category="verified",
        verification_status="verified",
        is_definite_error=False,
        severity="none",
        explanation=indiv_expl or "Statement verified with no detected mathematical or transformation errors.",
        conditions=all_conds,
        individual_status=indiv_status,
        transition_status=trans_status,
        incoming_from_expression=prev_expr_text,
        incoming_from_step=prev_step_num,
    )


# ==============================================================================
# Public API: Reasoning Error Detection
# ==============================================================================
def detect_reasoning_errors(
    structured_reasoning: Union[StructuredReasoning, ExtractedReasoning, str, Dict[str, Any]],
    verified_reasoning: Optional[VerifiedReasoning] = None,
    transition_verification: Optional[TransitionVerificationResult] = None,
) -> ReasoningErrorReport:
    """
    Identifies, consolidates, and flags mathematical reasoning errors across an LLM's solution.

    Combines:
    - Structured reasoning representations (Objective 20)
    - Individual SymPy assertion verifications (Objective 21)
    - Mathematical transition verifications (Objective 22)

    Parameters
    ----------
    structured_reasoning : StructuredReasoning, ExtractedReasoning, str, or dict
        Extracted or structured reasoning steps, or raw LLM response.
    verified_reasoning : VerifiedReasoning, optional
        Pre-computed individual verification results. Generated automatically if omitted.
    transition_verification : TransitionVerificationResult, optional
        Pre-computed transition verification results. Generated automatically if omitted.

    Returns
    -------
    ReasoningErrorReport
        Consolidated report detailing all flagged error steps, potential issues,
        unsupported statements, and individual expression assessments.
    """
    # 1. Standardize structured reasoning
    if not isinstance(structured_reasoning, StructuredReasoning):
        sr = structure_reasoning_steps(structured_reasoning)
    else:
        sr = structured_reasoning

    if sr.status in ("empty_response", "no_steps_found"):
        return ReasoningErrorReport(
            step_reports=[],
            total_steps=0,
            total_expressions=0,
            total_definite_errors=0,
            total_potential_issues=0,
            total_unsupported=0,
            total_verified=0,
            status="empty",
            flagged_error_steps=[],
            flagged_warning_steps=[],
            summary_explanation=f"Cannot detect reasoning errors: reasoning status is '{sr.status}'.",
        )

    # 2. Standardize verified reasoning
    if verified_reasoning is None or not isinstance(verified_reasoning, VerifiedReasoning):
        vr = verify_structured_reasoning(sr)
    else:
        vr = verified_reasoning

    # 3. Standardize transition verification
    if transition_verification is None or not isinstance(transition_verification, TransitionVerificationResult):
        tr = verify_step_transitions(sr, vr)
    else:
        tr = transition_verification

    # 4. Map incoming transitions to expressions
    # Flatten expressions across steps to associate each expression with its incoming transition
    ordered_expr_records: List[Tuple[int, int, Union[VerifiedExpression, StructuredExpression]]] = []
    if vr and vr.steps:
        for s_idx, v_step in enumerate(vr.steps):
            for e_idx, expr in enumerate(v_step.expressions):
                ordered_expr_records.append((s_idx, v_step.step_number, expr))
    else:
        for s_idx, s_step in enumerate(sr.steps):
            for e_idx, expr in enumerate(s_step.expressions):
                ordered_expr_records.append((s_idx, s_step.step_number, expr))

    # Match each expression to its incoming transition:
    # ordered_expr_records[0] has no incoming transition.
    # ordered_expr_records[i] has incoming transition tr.transitions[i - 1] (if available).
    transitions_map: Dict[int, Optional[StepTransition]] = {}
    for i in range(len(ordered_expr_records)):
        if i == 0:
            transitions_map[i] = None
        elif i - 1 < len(tr.transitions):
            transitions_map[i] = tr.transitions[i - 1]
        else:
            transitions_map[i] = None

    # 5. Build consolidated StepErrorReports preserving original step order
    step_reports: List[StepErrorReport] = []
    flagged_error_steps: Set[int] = set()
    flagged_warning_steps: Set[int] = set()

    total_definite_errors = 0
    total_potential_issues = 0
    total_unsupported = 0
    total_verified = 0
    total_expressions = len(ordered_expr_records)

    source_steps = vr.steps if vr and vr.steps else sr.steps

    expr_idx_counter = 0
    for s_idx, step_obj in enumerate(source_steps):
        step_num = step_obj.step_number
        step_header = getattr(step_obj, "step_header", f"Step {step_num}")
        orig_text = getattr(step_obj, "original_text", "")
        step_issues: List[DetectedIssue] = []

        step_has_definite = False
        step_has_potential = False
        step_has_unsupported = False

        for expr in step_obj.expressions:
            incoming_trans = transitions_map.get(expr_idx_counter, None)
            expr_idx_counter += 1

            issue = _classify_expression_issue(
                v_expr=expr,
                incoming_trans=incoming_trans,
                step_number=step_num,
            )
            step_issues.append(issue)

            if issue.is_definite_error:
                step_has_definite = True
                total_definite_errors += 1
            elif issue.severity == "potential_issue":
                step_has_potential = True
                total_potential_issues += 1
            elif issue.severity == "unsupported":
                step_has_unsupported = True
                total_unsupported += 1
            else:
                total_verified += 1

        if step_has_definite:
            flagged_error_steps.add(step_num)
        if step_has_potential:
            flagged_warning_steps.add(step_num)

        step_reports.append(
            StepErrorReport(
                step_number=step_num,
                step_header=step_header,
                original_text=orig_text,
                has_definite_errors=step_has_definite,
                has_potential_issues=step_has_potential,
                has_unsupported=step_has_unsupported,
                issues=step_issues,
            )
        )

    # 6. Overall trajectory status
    if total_definite_errors > 0:
        overall_status = "has_errors"
    elif total_potential_issues > 0:
        overall_status = "has_warnings"
    elif total_unsupported > 0 and total_verified == 0:
        overall_status = "unsupported"
    elif total_expressions == 0:
        overall_status = "empty"
    else:
        overall_status = "correct"

    summary_explanation = (
        f"Analyzed {total_expressions} mathematical statement(s) across {len(step_reports)} step(s): "
        f"{total_definite_errors} definite error(s), {total_potential_issues} potential issue(s)/warning(s), "
        f"{total_unsupported} unsupported statement(s), {total_verified} verified statement(s)."
    )

    return ReasoningErrorReport(
        step_reports=step_reports,
        total_steps=len(step_reports),
        total_expressions=total_expressions,
        total_definite_errors=total_definite_errors,
        total_potential_issues=total_potential_issues,
        total_unsupported=total_unsupported,
        total_verified=total_verified,
        status=overall_status,
        flagged_error_steps=sorted(list(flagged_error_steps)),
        flagged_warning_steps=sorted(list(flagged_warning_steps)),
        summary_explanation=summary_explanation,
    )


# ==============================================================================
# Standalone Demonstrations
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3, OBJECTIVE 23: REASONING ERROR DETECTOR DEMONSTRATIONS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Demonstration 1: Completely Correct Reasoning Example
    # --------------------------------------------------------------------------
    print("\nDEMONSTRATION 1: COMPLETELY CORRECT REASONING EXAMPLE")
    print("-" * 80)
    d1_text = (
        "**Step 1: Calculate cost of notebooks**\n"
        "Cost of notebooks = 3 * 40\n"
        "Cost of notebooks = 120\n\n"
        "**Step 2: Calculate cost of pens**\n"
        "Cost of pens = 2 * 15\n"
        "Cost of pens = 30\n\n"
        "**Step 3: Calculate total cost**\n"
        "Total cost = 120 + 30\n"
        "Total cost = 150\n\n"
        "Final Answer: 150"
    )
    rep1 = detect_reasoning_errors(d1_text)
    print(f"Overall Status:        {rep1.status.upper()}")
    print(f"Summary:               {rep1.summary_explanation}")
    print(f"Flagged Error Steps:   {rep1.flagged_error_steps}")
    print(f"Flagged Warning Steps: {rep1.flagged_warning_steps}")
    for sr in rep1.step_reports:
        print(f"\n[Step {sr.step_number}] Header: {sr.step_header}")
        for idx, iss in enumerate(sr.issues, 1):
            print(f"  {idx}. [{iss.error_category.upper()}] '{iss.expression}' -> Severity: {iss.severity}")
            print(f"     Explanation: {iss.explanation}")

    # --------------------------------------------------------------------------
    # Demonstration 2: Incorrect Arithmetic Assertion Example
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMONSTRATION 2: INCORRECT ARITHMETIC ASSERTION EXAMPLE")
    print("-" * 80)
    d2_text = (
        "**Step 1: Calculate total cost**\n"
        "3 * 40 = 130\n\n"
        "Final Answer: 130"
    )
    rep2 = detect_reasoning_errors(d2_text)
    print(f"Overall Status:        {rep2.status.upper()}")
    print(f"Summary:               {rep2.summary_explanation}")
    print(f"Flagged Error Steps:   {rep2.flagged_error_steps}")
    for sr in rep2.step_reports:
        for iss in sr.issues:
            print(f"Expression:            '{iss.expression}'")
            print(f"Error Category:        {iss.error_category.upper()}")
            print(f"Is Definite Error:     {iss.is_definite_error}")
            print(f"Severity:              {iss.severity}")
            print(f"Explanation:           {iss.explanation}")

    # --------------------------------------------------------------------------
    # Demonstration 3: Incorrect Algebraic Transformation Example
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMONSTRATION 3: INCORRECT ALGEBRAIC TRANSFORMATION EXAMPLE")
    print("-" * 80)
    d3_text = (
        "**Step 1: Solve equation**\n"
        "2 * x + 4 = 10\n"
        "2 * x = 14\n\n"
        "Final Answer: 7"
    )
    rep3 = detect_reasoning_errors(d3_text)
    print(f"Overall Status:        {rep3.status.upper()}")
    print(f"Summary:               {rep3.summary_explanation}")
    print(f"Flagged Error Steps:   {rep3.flagged_error_steps}")
    for sr in rep3.step_reports:
        for iss in sr.issues:
            print(f"Expression:            '{iss.expression}'")
            print(f"Error Category:        {iss.error_category.upper()}")
            print(f"Is Definite Error:     {iss.is_definite_error}")
            print(f"Severity:              {iss.severity}")
            print(f"Explanation:           {iss.explanation}")

    # --------------------------------------------------------------------------
    # Demonstration 4: Variable-Division Example that Loses a Solution
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMONSTRATION 4: VARIABLE-DIVISION EXAMPLE THAT LOSES A SOLUTION")
    print("-" * 80)
    d4_text = (
        "**Step 1: Divide by variable**\n"
        "x**2 = 2 * x\n"
        "x = 2\n\n"
        "Final Answer: 2"
    )
    rep4 = detect_reasoning_errors(d4_text)
    print(f"Overall Status:        {rep4.status.upper()}")
    print(f"Summary:               {rep4.summary_explanation}")
    print(f"Flagged Error Steps:   {rep4.flagged_error_steps}")
    for sr in rep4.step_reports:
        for iss in sr.issues:
            print(f"Expression:            '{iss.expression}'")
            print(f"Error Category:        {iss.error_category.upper()}")
            print(f"Is Definite Error:     {iss.is_definite_error}")
            print(f"Conditions:            {iss.conditions}")
            print(f"Explanation:           {iss.explanation}")

    # --------------------------------------------------------------------------
    # Demonstration 5: Squaring Example that Produces a Conditional Warning
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMONSTRATION 5: SQUARING EXAMPLE THAT PRODUCES A CONDITIONAL WARNING")
    print("-" * 80)
    d5_text = (
        "**Step 1: Square both sides**\n"
        "x = 3\n"
        "x**2 = 9\n\n"
        "Final Answer: 9"
    )
    rep5 = detect_reasoning_errors(d5_text)
    print(f"Overall Status:        {rep5.status.upper()}")
    print(f"Summary:               {rep5.summary_explanation}")
    print(f"Flagged Error Steps:   {rep5.flagged_error_steps}")
    print(f"Flagged Warning Steps: {rep5.flagged_warning_steps}")
    for sr in rep5.step_reports:
        for iss in sr.issues:
            print(f"Expression:            '{iss.expression}'")
            print(f"Error Category:        {iss.error_category.upper()}")
            print(f"Is Definite Error:     {iss.is_definite_error}")
            print(f"Severity:              {iss.severity}")
            print(f"Conditions:            {iss.conditions}")
            print(f"Explanation:           {iss.explanation}")
    print("=" * 80)
