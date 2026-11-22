"""
step_verifier.py
----------------
Phase 3: Automatic Error Detection in LLM Reasoning.
Objective 21: SymPy-Based Verification of Structured Reasoning.

Verifies the mathematical correctness of individual structured mathematical statements
extracted from LLM reasoning steps using the existing symbolic verifier (symbolic_verifier.py)
and symbolic solver (symbolic_solver.py).
Categorizes and verifies:
- Arithmetic equalities (pure numeric equations)
- Labeled equalities (descriptive label = arithmetic RHS)
- Algebraic equalities (equations containing variables)
- Standalone arithmetic expressions (evaluating and preserving results)
- Unsupported expressions (preserving diagnostic reasons without guessing)
"""

from dataclasses import dataclass, field
import os
import sys
from typing import List, Optional, Dict, Any, Union

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_extractor import ExtractedReasoning, extract_reasoning_steps
from step_structurer import (
    StructuredReasoning,
    StructuredStep,
    StructuredExpression,
    structure_reasoning_steps,
)
from symbolic_verifier import verify_equation, VerificationResult
from symbolic_solver import simplify_expression, solve_equation, SimplifyResult, SolveResult


@dataclass
class VerifiedExpression:
    """Outcome of verifying an individual mathematical statement."""
    step_number: int                          # Source step number
    expression_type: str                      # 'arithmetic_equality', 'labeled_equality', 'algebraic_equality',
                                              # 'standalone_expression', 'unsupported'
    raw_text: str                             # Original expression text
    normalized_text: Optional[str] = None     # Normalized mathematical expression (if available)
    verification_status: str = "unsupported"  # 'verified', 'incorrect', 'unsupported', 'invalid'
    lhs_value: Optional[str] = None           # Evaluated LHS value (where applicable)
    rhs_value: Optional[str] = None           # Evaluated RHS value (where applicable)
    calculated_value: Optional[str] = None    # Evaluated single value or solution(s)
    explanation: str = ""                     # Detailed verification explanation
    unsupported_reason: Optional[str] = None  # Reason if marked unsupported
    label: Optional[str] = None               # Natural-language label or variable name
    domain_restrictions: List[str] = field(default_factory=list) # Preserved variable domain restrictions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "expression_type": self.expression_type,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "verification_status": self.verification_status,
            "lhs_value": self.lhs_value,
            "rhs_value": self.rhs_value,
            "calculated_value": self.calculated_value,
            "explanation": self.explanation,
            "unsupported_reason": self.unsupported_reason,
            "label": self.label,
            "domain_restrictions": self.domain_restrictions,
        }


@dataclass
class VerifiedStep:
    """Verification results for all statements within a single reasoning step."""
    step_number: int
    step_header: str
    original_text: str
    expressions: List[VerifiedExpression] = field(default_factory=list)
    has_errors: bool = False
    has_unsupported: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_header": self.step_header,
            "original_text": self.original_text,
            "expressions": [e.to_dict() for e in self.expressions],
            "has_errors": self.has_errors,
            "has_unsupported": self.has_unsupported,
        }


@dataclass
class VerifiedReasoning:
    """Complete verification report across all steps in the reasoning sequence."""
    raw_response: str
    steps: List[VerifiedStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_answer_raw: Optional[str] = None
    total_expressions: int = 0
    verified_count: int = 0
    incorrect_count: int = 0
    unsupported_count: int = 0
    invalid_count: int = 0
    status: str = "success"                   # 'success', 'empty_response', 'no_steps_found', 'malformed'
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "final_answer_raw": self.final_answer_raw,
            "total_expressions": self.total_expressions,
            "verified_count": self.verified_count,
            "incorrect_count": self.incorrect_count,
            "unsupported_count": self.unsupported_count,
            "invalid_count": self.invalid_count,
            "status": self.status,
            "explanation": self.explanation,
        }


def _verify_single_expression(expr: StructuredExpression) -> VerifiedExpression:
    """
    Independently verifies a single StructuredExpression using the appropriate
    symbolic engine (symbolic_verifier or symbolic_solver).
    """
    # --------------------------------------------------------------------------
    # A. Arithmetic Equality (pure numeric equations e.g. "3 * 40 = 120")
    # --------------------------------------------------------------------------
    if expr.expression_type == "arithmetic_equality":
        # Prefer raw_text if it is a clean equation preserving signs (e.g. leading unary '-')
        v_res = None
        eq_str = expr.raw_text
        if expr.raw_text and "=" in expr.raw_text:
            try:
                candidate_v = verify_equation(expr.raw_text)
                if candidate_v.status in ("correct", "incorrect"):
                    v_res = candidate_v
                    eq_str = expr.raw_text
            except Exception:
                pass

        if v_res is None:
            eq_str = expr.normalized_equation or f"{expr.lhs_normalized} = {expr.rhs_normalized}"
            v_res = verify_equation(eq_str)

        if v_res.status == "correct":
            v_status = "verified"
        elif v_res.status == "incorrect":
            v_status = "incorrect"
        else:
            v_status = "invalid"

        return VerifiedExpression(
            step_number=expr.step_number,
            expression_type="arithmetic_equality",
            raw_text=expr.raw_text,
            normalized_text=eq_str,
            verification_status=v_status,
            lhs_value=v_res.lhs_value,
            rhs_value=v_res.rhs_value,
            calculated_value=v_res.lhs_value if v_status == "verified" else None,
            explanation=v_res.explanation,
            unsupported_reason=None,
            label=None,
            domain_restrictions=[],
        )

    # --------------------------------------------------------------------------
    # B. Labeled Equality (e.g. "Cost of notebooks = 3 * 40")
    # --------------------------------------------------------------------------
    elif expr.expression_type == "labeled_equality":
        # Check if RHS itself contains an explicit equality (e.g. "3 * 40 = 130")
        if expr.rhs_normalized and "=" in expr.rhs_normalized:
            v_inner = verify_equation(expr.rhs_normalized)
            v_status = "verified" if v_inner.status == "correct" else v_inner.status
            return VerifiedExpression(
                step_number=expr.step_number,
                expression_type="labeled_equality",
                raw_text=expr.raw_text,
                normalized_text=f"{expr.label} = {expr.rhs_normalized}",
                verification_status=v_status,
                lhs_value=v_inner.lhs_value,
                rhs_value=v_inner.rhs_value,
                calculated_value=v_inner.lhs_value if v_status == "verified" else None,
                explanation=f"Labeled equation for '{expr.label}': {v_inner.explanation}",
                unsupported_reason=None,
                label=expr.label,
                domain_restrictions=[],
            )

        # Numerical RHS evaluation
        rhs_to_eval = expr.rhs_normalized if expr.rhs_normalized else expr.rhs_raw
        simp_res: SimplifyResult = simplify_expression(rhs_to_eval or "")

        if simp_res.status == "simplified":
            return VerifiedExpression(
                step_number=expr.step_number,
                expression_type="labeled_equality",
                raw_text=expr.raw_text,
                normalized_text=f"{expr.label} = {simp_res.simplified_expression}",
                verification_status="verified",
                lhs_value=expr.label,
                rhs_value=simp_res.simplified_expression,
                calculated_value=simp_res.simplified_expression,
                explanation=f"Labeled numerical assertion '{expr.label}': RHS evaluates to {simp_res.simplified_expression}.",
                unsupported_reason=None,
                label=expr.label,
                domain_restrictions=simp_res.domain_restrictions,
            )
        else:
            return VerifiedExpression(
                step_number=expr.step_number,
                expression_type="labeled_equality",
                raw_text=expr.raw_text,
                normalized_text=expr.normalized_equation,
                verification_status="invalid",
                lhs_value=expr.label,
                rhs_value=None,
                calculated_value=None,
                explanation=f"Invalid calculation in labeled equality '{expr.label}': {simp_res.explanation}",
                unsupported_reason=simp_res.explanation,
                label=expr.label,
                domain_restrictions=[],
            )

    # --------------------------------------------------------------------------
    # C. Algebraic Equality (equations containing variables e.g. "x + 5 = 12")
    # --------------------------------------------------------------------------
    elif expr.expression_type == "algebraic_equality":
        eq_str = expr.normalized_equation or f"{expr.lhs_normalized} = {expr.rhs_normalized}"
        sol_res: SolveResult = solve_equation(eq_str)

        if sol_res.status == "solved":
            if sol_res.solution_type == "no_solution":
                v_status = "incorrect"
                restr_info = f" (domain restrictions: {', '.join(sol_res.domain_restrictions)})" if sol_res.domain_restrictions else ""
                explanation = f"Algebraic equation is mathematically inconsistent (no solution){restr_info}."
            else:
                v_status = "verified"
                sol_str = ", ".join(sol_res.solutions) if sol_res.solutions else sol_res.solution_type
                explanation = f"Algebraic equation solved ({sol_res.solution_type}): {sol_res.variable} = {sol_str}."

            return VerifiedExpression(
                step_number=expr.step_number,
                expression_type="algebraic_equality",
                raw_text=expr.raw_text,
                normalized_text=eq_str,
                verification_status=v_status,
                lhs_value=expr.lhs_normalized,
                rhs_value=expr.rhs_normalized,
                calculated_value=", ".join(sol_res.solutions) if sol_res.solutions else sol_res.solution_type,
                explanation=explanation,
                unsupported_reason=None,
                label=sol_res.variable,
                domain_restrictions=sol_res.domain_restrictions,
            )
        else:
            return VerifiedExpression(
                step_number=expr.step_number,
                expression_type="algebraic_equality",
                raw_text=expr.raw_text,
                normalized_text=eq_str,
                verification_status="invalid",
                lhs_value=expr.lhs_normalized,
                rhs_value=expr.rhs_normalized,
                calculated_value=None,
                explanation=f"Invalid algebraic equation: {sol_res.explanation}",
                unsupported_reason=sol_res.explanation,
                label=None,
                domain_restrictions=[],
            )

    # --------------------------------------------------------------------------
    # D. Standalone Arithmetic Expression (e.g. "3 * 40 + 2 * 15")
    # --------------------------------------------------------------------------
    elif expr.expression_type == "standalone_expression":
        expr_to_eval = expr.rhs_normalized or expr.raw_text
        simp_res = simplify_expression(expr_to_eval)

        if simp_res.status == "simplified":
            return VerifiedExpression(
                step_number=expr.step_number,
                expression_type="standalone_expression",
                raw_text=expr.raw_text,
                normalized_text=simp_res.simplified_expression,
                verification_status="verified",
                lhs_value=None,
                rhs_value=simp_res.simplified_expression,
                calculated_value=simp_res.simplified_expression,
                explanation=f"Standalone expression evaluates to {simp_res.simplified_expression}.",
                unsupported_reason=None,
                label=None,
                domain_restrictions=simp_res.domain_restrictions,
            )
        else:
            return VerifiedExpression(
                step_number=expr.step_number,
                expression_type="standalone_expression",
                raw_text=expr.raw_text,
                normalized_text=expr.rhs_normalized,
                verification_status="invalid",
                lhs_value=None,
                rhs_value=None,
                calculated_value=None,
                explanation=f"Invalid standalone expression: {simp_res.explanation}",
                unsupported_reason=simp_res.explanation,
                label=None,
                domain_restrictions=[],
            )

    # --------------------------------------------------------------------------
    # E. Unsupported Expression / Natural-Language Formula
    # --------------------------------------------------------------------------
    else:
        unsupp_reason = expr.unsupported_reason or "Expression cannot be safely evaluated with symbolic verifier."
        return VerifiedExpression(
            step_number=expr.step_number,
            expression_type=expr.expression_type,
            raw_text=expr.raw_text,
            normalized_text=expr.normalized_equation,
            verification_status="unsupported",
            lhs_value=expr.lhs_raw,
            rhs_value=expr.rhs_raw,
            calculated_value=None,
            explanation=unsupp_reason,
            unsupported_reason=unsupp_reason,
            label=expr.label,
            domain_restrictions=[],
        )


def verify_structured_reasoning(
    structured_reasoning: Union[StructuredReasoning, ExtractedReasoning, str, Dict[str, Any]]
) -> VerifiedReasoning:
    """
    Verifies every structured mathematical statement in the reasoning steps.

    Parameters
    ----------
    structured_reasoning : StructuredReasoning, ExtractedReasoning, str, or dict
        Structured reasoning from step_structurer.py, or extracted reasoning, or raw response.

    Returns
    -------
    VerifiedReasoning
        Comprehensive verification report containing verified steps, individual expression
        statuses ('verified', 'incorrect', 'unsupported', 'invalid'), and calculated values.
    """
    # 1. Ensure input is a StructuredReasoning instance
    if not isinstance(structured_reasoning, StructuredReasoning):
        sr = structure_reasoning_steps(structured_reasoning)
    else:
        sr = structured_reasoning

    if sr.status in ("empty_response", "no_steps_found"):
        return VerifiedReasoning(
            raw_response=sr.raw_response,
            steps=[],
            final_answer=sr.final_answer,
            final_answer_raw=sr.final_answer_raw,
            total_expressions=0,
            status=sr.status,
            explanation=sr.explanation,
        )

    verified_steps: List[VerifiedStep] = []
    total_expressions = 0
    verified_count = 0
    incorrect_count = 0
    unsupported_count = 0
    invalid_count = 0

    # 2. Iterate through each step and its mathematical expressions in order
    for s_step in sr.steps:
        step_verified_exprs: List[VerifiedExpression] = []
        step_has_errors = False
        step_has_unsupported = False

        for expr in s_step.expressions:
            total_expressions += 1
            ver_res = _verify_single_expression(expr)
            step_verified_exprs.append(ver_res)

            if ver_res.verification_status == "verified":
                verified_count += 1
            elif ver_res.verification_status == "incorrect":
                incorrect_count += 1
                step_has_errors = True
            elif ver_res.verification_status == "invalid":
                invalid_count += 1
                step_has_errors = True
            else:  # 'unsupported'
                unsupported_count += 1
                step_has_unsupported = True

        verified_steps.append(
            VerifiedStep(
                step_number=s_step.step_number,
                step_header=s_step.step_header,
                original_text=s_step.original_text,
                expressions=step_verified_exprs,
                has_errors=step_has_errors,
                has_unsupported=step_has_unsupported,
            )
        )

    summary_explanation = (
        f"Verified {total_expressions} mathematical statements: "
        f"{verified_count} verified, {incorrect_count} incorrect, "
        f"{unsupported_count} unsupported, {invalid_count} invalid."
    )

    if sr.status in ("empty_response", "no_steps_found"):
        overall_status = sr.status
    elif incorrect_count > 0:
        overall_status = "has_errors"
    elif invalid_count > 0:
        overall_status = "invalid"
    elif unsupported_count > 0 and verified_count == 0:
        overall_status = "unsupported"
    else:
        overall_status = "verified"

    return VerifiedReasoning(
        raw_response=sr.raw_response,
        steps=verified_steps,
        final_answer=sr.final_answer,
        final_answer_raw=sr.final_answer_raw,
        total_expressions=total_expressions,
        verified_count=verified_count,
        incorrect_count=incorrect_count,
        unsupported_count=unsupported_count,
        invalid_count=invalid_count,
        status=overall_status,
        explanation=summary_explanation,
    )


if __name__ == "__main__":
    demo_gemini_response = (
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

    print("=" * 80)
    print("PHASE 3, OBJECTIVE 21: SYMPY-BASED REASONING VERIFICATION DEMONSTRATION")
    print("=" * 80)
    print("DEMONSTRATION 1: ACTUAL OBJECTIVE 20 GEMINI REASONING")
    print("-" * 80)

    res = verify_structured_reasoning(demo_gemini_response)
    print(f"Summary:     {res.explanation}")
    print(f"Total:       {res.total_expressions}")
    print(f"Verified:    {res.verified_count}")
    print(f"Incorrect:   {res.incorrect_count}")
    print(f"Unsupported: {res.unsupported_count}")
    print(f"Invalid:     {res.invalid_count}")
    print("-" * 80)

    for step in res.steps:
        print(f"\n[Step {step.step_number}] Header: {step.step_header}")
        for i, expr in enumerate(step.expressions, 1):
            status_tag = f"[{expr.verification_status.upper()}]"
            print(f"  {i}. {status_tag:<14} Raw: '{expr.raw_text}'")
            if expr.calculated_value:
                print(f"     Calculated Value: {expr.calculated_value}")
            if expr.lhs_value or expr.rhs_value:
                print(f"     Evaluated: LHS={repr(expr.lhs_value)}, RHS={repr(expr.rhs_value)}")
            print(f"     Explanation: {expr.explanation}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION 2: DELIBERATELY INCORRECT ARITHMETIC ASSERTION ('3 * 40 = 130')")
    print("-" * 80)
    err_text = (
        "**Step 1: Deliberate Calculation Error**\n"
        "3 * 40 = 130\n\n"
        "Final Answer: 130"
    )
    err_res = verify_structured_reasoning(err_text)
    for step in err_res.steps:
        for expr in step.expressions:
            print(f"Raw Text:            '{expr.raw_text}'")
            print(f"Verification Status: {expr.verification_status.upper()}")
            print(f"Evaluated LHS:       {expr.lhs_value}")
            print(f"Evaluated RHS:       {expr.rhs_value}")
            print(f"Explanation:         {expr.explanation}")
    print("=" * 80)
