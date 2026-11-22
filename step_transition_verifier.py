"""
step_transition_verifier.py
----------------------------
Phase 3: Automatic Error Detection in LLM Reasoning.
Objective 22: Mathematical Transformation and Step Transition Verification.

Examines consecutive mathematical statements in an LLM's step-by-step reasoning
and determines whether each transformation is mathematically valid, invalid, or unsupported.
Considers mathematical conditions including:
- Domain and denominator restrictions (e.g. division by zero, x != 0).
- Extraneous solutions and one-way implications from squaring both sides.
- Loss of solutions and zero-division risks from dividing both sides by variable expressions.
- Balanced operations applied to both sides (adding, subtracting, multiplying, dividing).
- Exact arithmetic and algebraic simplifications.
- Distinguishes valid transformations from unrelated independent calculations.
"""

from dataclasses import dataclass, field
import ast
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import sympy

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
from symbolic_solver import (
    _parse_expression_with_denominators,
    _compute_domain_restrictions,
    _parse_expression_safely,
    solve_equation,
    simplify_expression,
    SolveResult,
    SimplifyResult,
)
from symbolic_verifier import verify_equation, VerificationResult


# ==============================================================================
# Transition Verification Dataclasses
# ==============================================================================
@dataclass
class StepTransition:
    """Represents the verification of a transition between two consecutive mathematical statements."""
    prev_step_number: int
    curr_step_number: int
    prev_expression: str
    curr_expression: str
    transition_status: str              # 'valid', 'invalid', 'unsupported'
    transformation_type: str            # 'algebraic_rearrangement', 'both_sides_operation',
                                        # 'both_sides_scaling', 'squaring_both_sides',
                                        # 'arithmetic_simplification', 'substitution',
                                        # 'division_by_variable', 'incorrect_transformation',
                                        # 'domain_violation', 'invalid_expression',
                                        # 'unrelated', 'unsupported'
    explanation: str
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prev_step_number": self.prev_step_number,
            "curr_step_number": self.curr_step_number,
            "prev_expression": self.prev_expression,
            "curr_expression": self.curr_expression,
            "transition_status": self.transition_status,
            "transformation_type": self.transformation_type,
            "explanation": self.explanation,
            "conditions": self.conditions,
        }


@dataclass
class TransitionVerificationResult:
    """Top-level container holding verified transitions across all reasoning steps."""
    transitions: List[StepTransition] = field(default_factory=list)
    total_transitions: int = 0
    valid_transitions: int = 0
    invalid_transitions: int = 0
    unsupported_transitions: int = 0
    status: str = "valid"               # 'valid', 'has_invalid_transitions', 'unsupported', 'empty'
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transitions": [t.to_dict() for t in self.transitions],
            "total_transitions": self.total_transitions,
            "valid_transitions": self.valid_transitions,
            "invalid_transitions": self.invalid_transitions,
            "unsupported_transitions": self.unsupported_transitions,
            "status": self.status,
            "explanation": self.explanation,
        }


# ==============================================================================
# Internal Mathematical Transition Verification Helpers
# ==============================================================================
def _split_equation_sides(eq_str: str) -> Optional[Tuple[str, str]]:
    """Splits an equation string on '=' if exactly one '=' operator is present."""
    if not eq_str or "=" not in eq_str:
        return None
    parts = eq_str.split("=")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None


def _check_domain_violation(
    candidate_solutions: List[str],
    variable: Optional[str],
    domain_restrictions: List[str]
) -> Optional[str]:
    """Checks whether any candidate solution violates active domain restrictions (e.g. x != 1)."""
    if not candidate_solutions or not variable or not domain_restrictions:
        return None

    for sol in candidate_solutions:
        sol_clean = sol.strip()
        for restr in domain_restrictions:
            if "!=" in restr:
                r_var, r_val = restr.split("!=")
                if r_var.strip() == variable.strip() and r_val.strip() == sol_clean:
                    return f"Candidate solution '{variable} = {sol_clean}' violates domain restriction '{restr}'."
    return None


def _verify_equation_to_equation_transition(
    prev_eq: str,
    curr_eq: str,
    prev_restrictions: Optional[List[str]] = None
) -> Tuple[str, str, str, List[str]]:
    """
    Verifies the mathematical transition between two consecutive equations.
    Returns: (transition_status, transformation_type, explanation, conditions)
    """
    prev_sides = _split_equation_sides(prev_eq)
    curr_sides = _split_equation_sides(curr_eq)

    if not prev_sides or not curr_sides:
        return (
            "unsupported",
            "unsupported",
            "One or both expressions are not standard bipartite equations.",
            [],
        )

    p_lhs_raw, p_rhs_raw = prev_sides
    c_lhs_raw, c_rhs_raw = curr_sides

    # Parse LHS and RHS with safe denominator tracking
    l1, d1_l, err1_l = _parse_expression_with_denominators(p_lhs_raw, allow_variables=True)
    r1, d1_r, err1_r = _parse_expression_with_denominators(p_rhs_raw, allow_variables=True)
    l2, d2_l, err2_l = _parse_expression_with_denominators(c_lhs_raw, allow_variables=True)
    r2, d2_r, err2_r = _parse_expression_with_denominators(c_rhs_raw, allow_variables=True)

    # Check for division by zero or AST errors
    all_errs = [e for e in (err1_l, err1_r, err2_l, err2_r) if e]
    if all_errs:
        div_zero = any("division by zero" in str(e).lower() for e in all_errs)
        if div_zero:
            return (
                "invalid",
                "division_by_zero",
                f"Transition involves undefined division by zero: {all_errs[0]}",
                ["Division by zero is undefined."],
            )
        return (
            "unsupported",
            "unsupported",
            f"Unable to parse equation sides: {all_errs[0]}",
            [],
        )

    # Accumulate domain restrictions from denominators and prior steps
    all_denoms = d1_l + d1_r + d2_l + d2_r
    domain_restrs = _compute_domain_restrictions(all_denoms)
    if prev_restrictions:
        for r in prev_restrictions:
            if r not in domain_restrs:
                domain_restrs.append(r)

    f1 = l1 - r1
    f2 = l2 - r2

    # Check for domain violations in curr_eq candidate roots
    curr_sol = solve_equation(curr_eq)
    if curr_sol.status == "solved" and curr_sol.solutions and curr_sol.variable:
        violation = _check_domain_violation(curr_sol.solutions, curr_sol.variable, domain_restrs)
        if violation:
            return (
                "invalid",
                "domain_violation",
                f"Invalid algebraic transformation: {violation}",
                domain_restrs,
            )

    # 1. Algebraic Equivalence / Rearrangement (f1 - f2 == 0)
    if sympy.simplify(f1 - f2) == 0:
        return (
            "valid",
            "algebraic_rearrangement",
            "Valid algebraic rearrangement: equations are mathematically equivalent.",
            domain_restrs,
        )

    # Both sides multiplied by -1 (f1 + f2 == 0)
    if sympy.simplify(f1 + f2) == 0:
        return (
            "valid",
            "both_sides_scaling",
            "Valid algebraic transformation: multiplied both sides by -1.",
            domain_restrs,
        )

    # 2. Add / Subtract the same quantity on both sides
    delta_l = sympy.simplify(l2 - l1)
    delta_r = sympy.simplify(r2 - r1)
    if sympy.simplify(delta_l - delta_r) == 0:
        is_sub = hasattr(delta_l, "is_number") and delta_l < 0
        op = f"subtracted {-delta_l}" if is_sub else f"added {delta_l}"
        return (
            "valid",
            "both_sides_operation",
            f"Valid algebraic transformation: {op} on both sides.",
            domain_restrs,
        )

    # 3. Squaring both sides: L2 = L1^2 and R2 = R1^2
    is_squared = (
        (sympy.simplify(l2 - l1**2) == 0 and sympy.simplify(r2 - r1**2) == 0)
        or (sympy.simplify(r2 - l1**2) == 0 and sympy.simplify(l2 - r1**2) == 0)
    )
    if is_squared:
        conds = list(domain_restrs) + [
            "Squaring both sides is a one-way implication (A = B => A^2 = B^2) that may introduce extraneous roots; roots must be checked in original equation."
        ]
        return (
            "valid",
            "squaring_both_sides",
            "Valid algebraic transformation: squared both sides of the equation.",
            conds,
        )

    # 4. Constant scaling (multiply / divide both sides by non-zero constant)
    ratio = sympy.simplify(f1 / f2)
    if ratio.is_number and ratio != 0:
        op = f"divided both sides by {ratio}" if ratio != 1 else "simplified both sides"
        return (
            "valid",
            "both_sides_scaling",
            f"Valid algebraic transformation: {op}.",
            domain_restrs,
        )

    # 5. Division by variable expression (ratio contains free variables)
    free_syms = f1.free_symbols.union(f2.free_symbols)
    if free_syms:
        is_poly_ratio = False
        try:
            poly = sympy.Poly(ratio, *free_syms)
            if poly.total_degree() >= 1:
                is_poly_ratio = True
        except Exception:
            pass

        if is_poly_ratio:
            prev_sol = solve_equation(prev_eq)
            lost_roots = set(prev_sol.solutions) - set(curr_sol.solutions)
            conds = [f"{ratio} != 0 (division by variable expression)"] + domain_restrs

            if lost_roots:
                return (
                    "invalid",
                    "division_by_variable",
                    f"Invalid algebraic transformation: divided both sides by variable expression '{ratio}', losing solution(s) {sorted(list(lost_roots))} and risking division by zero if {ratio} = 0.",
                    conds,
                )
            else:
                return (
                    "valid",
                    "division_by_variable",
                    f"Algebraic transformation divides both sides by variable expression '{ratio}' (requires condition {ratio} != 0).",
                    conds,
                )

    # 6. General solution-set comparison
    prev_sol = solve_equation(prev_eq)
    if prev_sol.status == "solved" and curr_sol.status == "solved":
        if prev_sol.solutions and curr_sol.solutions and set(prev_sol.solutions) == set(curr_sol.solutions):
            return (
                "valid",
                "algebraic_transformation",
                f"Valid algebraic transformation: preserves solution set ({', '.join(prev_sol.solutions)}).",
                domain_restrs,
            )
        elif prev_sol.solutions != curr_sol.solutions:
            return (
                "invalid",
                "incorrect_transformation",
                f"Invalid algebraic transformation: original equation has solution(s) {prev_sol.solutions}, but transformed equation has solution(s) {curr_sol.solutions}.",
                domain_restrs,
            )

    return (
        "unsupported",
        "unsupported",
        "Unable to establish mathematical transformation relationship between equations.",
        domain_restrs,
    )


def _verify_single_transition(
    prev_expr: Union[VerifiedExpression, StructuredExpression],
    curr_expr: Union[VerifiedExpression, StructuredExpression],
    prev_step_num: int,
    curr_step_num: int,
    accumulated_restrictions: Optional[List[str]] = None,
) -> StepTransition:
    """
    Examines two consecutive mathematical statements and verifies whether the transition
    between them is mathematically valid, invalid, or unsupported.
    """
    prev_raw = prev_expr.raw_text.strip()
    curr_raw = curr_expr.raw_text.strip()

    # Extract verification statuses if available from Objective 21
    prev_v_status = getattr(prev_expr, "verification_status", "verified")
    curr_actual_status = getattr(curr_expr, "verification_status", "verified")

    # A. Check if either expression is already flagged as invalid (e.g. division by zero in either)
    if prev_v_status == "invalid" or curr_actual_status == "invalid":
        inv_text = prev_raw if prev_v_status == "invalid" else curr_raw
        return StepTransition(
            prev_step_number=prev_step_num,
            curr_step_number=curr_step_num,
            prev_expression=prev_raw,
            curr_expression=curr_raw,
            transition_status="invalid",
            transformation_type="invalid_expression",
            explanation=f"Transition involves an invalid mathematical statement ('{inv_text}').",
            conditions=["Mathematically invalid expression"],
        )

    # B. Try equation-to-equation transition on raw text if both contain '='
    prev_sides = _split_equation_sides(prev_raw)
    curr_sides = _split_equation_sides(curr_raw)

    if prev_sides and curr_sides:
        prev_label = getattr(prev_expr, "label", None)
        curr_label = getattr(curr_expr, "label", None)

        # Check for distinct multi-word labels (unrelated independent calculations)
        if prev_label and curr_label and prev_label != curr_label:
            # If multi-word conceptual labels like 'Cost of notebooks' vs 'Cost of pens'
            if len(prev_label.split()) > 1 and len(curr_label.split()) > 1:
                return StepTransition(
                    prev_step_number=prev_step_num,
                    curr_step_number=curr_step_num,
                    prev_expression=prev_raw,
                    curr_expression=curr_raw,
                    transition_status="unsupported",
                    transformation_type="unrelated",
                    explanation=f"Consecutive statements represent distinct, independent calculations ('{prev_label}' vs '{curr_label}'); no direct mathematical transformation.",
                    conditions=[],
                )

        # Verify equation-to-equation transition
        st, tt, expl, conds = _verify_equation_to_equation_transition(
            prev_raw, curr_raw, accumulated_restrictions
        )
        if st in ("valid", "invalid"):
            return StepTransition(
                prev_step_number=prev_step_num,
                curr_step_number=curr_step_num,
                prev_expression=prev_raw,
                curr_expression=curr_raw,
                transition_status=st,
                transformation_type=tt,
                explanation=expl,
                conditions=conds,
            )

    # C. Labeled Equalities (e.g. "Cost of notebooks = 3 * 40")
    prev_label = getattr(prev_expr, "label", None)
    curr_label = getattr(curr_expr, "label", None)

    if prev_label and curr_label:
        if prev_label == curr_label:
            prev_rhs = getattr(prev_expr, "rhs_normalized", None) or getattr(prev_expr, "rhs_raw", None) or ""
            curr_rhs = getattr(curr_expr, "rhs_normalized", None) or getattr(curr_expr, "rhs_raw", None) or ""

            if "=" not in prev_rhs and "=" not in curr_rhs:
                p_sym, _, err1 = _parse_expression_with_denominators(prev_rhs, allow_variables=True)
                c_sym, _, err2 = _parse_expression_with_denominators(curr_rhs, allow_variables=True)
                if not err1 and not err2:
                    diff = sympy.simplify(p_sym - c_sym)
                    if diff == 0:
                        return StepTransition(
                            prev_step_number=prev_step_num,
                            curr_step_number=curr_step_num,
                            prev_expression=prev_raw,
                            curr_expression=curr_raw,
                            transition_status="valid",
                            transformation_type="arithmetic_simplification",
                            explanation=f"Valid arithmetic simplification for '{prev_label}': RHS evaluates to {c_sym}.",
                            conditions=[],
                        )
                    else:
                        return StepTransition(
                            prev_step_number=prev_step_num,
                            curr_step_number=curr_step_num,
                            prev_expression=prev_raw,
                            curr_expression=curr_raw,
                            transition_status="invalid",
                            transformation_type="incorrect_transformation",
                            explanation=f"Invalid calculation for '{prev_label}': previous RHS ({prev_rhs}) evaluates to {p_sym}, but current RHS ({curr_rhs}) evaluates to {c_sym}.",
                            conditions=[],
                        )
        else:
            return StepTransition(
                prev_step_number=prev_step_num,
                curr_step_number=curr_step_num,
                prev_expression=prev_raw,
                curr_expression=curr_raw,
                transition_status="unsupported",
                transformation_type="unrelated",
                explanation=f"Consecutive statements represent distinct, independent calculations ('{prev_label}' vs '{curr_label}'); no direct mathematical transformation.",
                conditions=[],
            )

    # D. Both are Standalone Expressions (e.g. "3 * 40 + 2 * 15" -> "120 + 30" -> "150")
    p_sym, d1, err1 = _parse_expression_with_denominators(prev_raw, allow_variables=True)
    c_sym, d2, err2 = _parse_expression_with_denominators(curr_raw, allow_variables=True)
    if not err1 and not err2 and "=" not in prev_raw and "=" not in curr_raw:
        all_d = d1 + d2
        restrs = _compute_domain_restrictions(all_d)
        diff = sympy.simplify(p_sym - c_sym)
        if diff == 0:
            return StepTransition(
                prev_step_number=prev_step_num,
                curr_step_number=curr_step_num,
                prev_expression=prev_raw,
                curr_expression=curr_raw,
                transition_status="valid",
                transformation_type="arithmetic_simplification",
                explanation=f"Valid arithmetic simplification: expressions are mathematically equivalent ({c_sym}).",
                conditions=restrs,
            )
        else:
            return StepTransition(
                prev_step_number=prev_step_num,
                curr_step_number=curr_step_num,
                prev_expression=prev_raw,
                curr_expression=curr_raw,
                transition_status="invalid",
                transformation_type="incorrect_transformation",
                explanation=f"Invalid simplification: '{prev_raw}' evaluates to {p_sym}, but '{curr_raw}' evaluates to {c_sym}.",
                conditions=restrs,
            )

    # Fallback: unsupported transition
    return StepTransition(
        prev_step_number=prev_step_num,
        curr_step_number=curr_step_num,
        prev_expression=prev_raw,
        curr_expression=curr_raw,
        transition_status="unsupported",
        transformation_type="unsupported",
        explanation="No direct mathematical transformation established between consecutive statements.",
        conditions=[],
    )


# ==============================================================================
# Public API: Step Transition Verification
# ==============================================================================
def verify_step_transitions(
    structured_reasoning: Union[StructuredReasoning, ExtractedReasoning, str, Dict[str, Any]],
    verified_reasoning: Optional[VerifiedReasoning] = None,
) -> TransitionVerificationResult:
    """
    Examines consecutive mathematical statements across reasoning steps and verifies
    whether each transformation is mathematically valid, invalid, or unsupported.

    Parameters
    ----------
    structured_reasoning : StructuredReasoning, ExtractedReasoning, str, or dict
        Structured reasoning steps from step_structurer.py or raw reasoning text.
    verified_reasoning : VerifiedReasoning, optional
        Pre-computed verification results from step_verifier.py. If omitted, will be
        automatically generated.

    Returns
    -------
    TransitionVerificationResult
        Summary of all step transitions, including status ('valid', 'has_invalid_transitions',
        'unsupported', 'empty'), counts, and individual transition assessments.
    """
    # 1. Standardize structured reasoning input
    if not isinstance(structured_reasoning, StructuredReasoning):
        sr = structure_reasoning_steps(structured_reasoning)
    else:
        sr = structured_reasoning

    if sr.status in ("empty_response", "no_steps_found"):
        return TransitionVerificationResult(
            transitions=[],
            total_transitions=0,
            valid_transitions=0,
            invalid_transitions=0,
            unsupported_transitions=0,
            status="empty",
            explanation=f"Cannot verify transitions: reasoning status is '{sr.status}'.",
        )

    # 2. Standardize verified reasoning input
    if verified_reasoning is None or not isinstance(verified_reasoning, VerifiedReasoning):
        vr = verify_structured_reasoning(sr)
    else:
        vr = verified_reasoning

    # 3. Collect ordered sequence of expressions across steps
    ordered_expressions: List[Tuple[Union[VerifiedExpression, StructuredExpression], int]] = []

    # Prefer VerifiedExpressions from vr.steps if available, as they contain domain restrictions & evaluations
    if vr and vr.steps:
        for v_step in vr.steps:
            for expr in v_step.expressions:
                ordered_expressions.append((expr, v_step.step_number))
    else:
        for s_step in sr.steps:
            for expr in s_step.expressions:
                ordered_expressions.append((expr, s_step.step_number))

    if len(ordered_expressions) < 2:
        return TransitionVerificationResult(
            transitions=[],
            total_transitions=0,
            valid_transitions=0,
            invalid_transitions=0,
            unsupported_transitions=0,
            status="empty",
            explanation="Insufficient mathematical statements (less than 2) to evaluate transitions.",
        )

    # 4. Verify transitions between consecutive expressions
    transitions: List[StepTransition] = []
    valid_count = 0
    invalid_count = 0
    unsupported_count = 0
    accumulated_restrictions: List[str] = []

    for i in range(len(ordered_expressions) - 1):
        prev_expr, prev_step_num = ordered_expressions[i]
        curr_expr, curr_step_num = ordered_expressions[i + 1]

        # Accumulate domain restrictions from previous expressions
        prev_restrs = getattr(prev_expr, "domain_restrictions", [])
        if prev_restrs:
            for r in prev_restrs:
                if r not in accumulated_restrictions:
                    accumulated_restrictions.append(r)

        transition = _verify_single_transition(
            prev_expr=prev_expr,
            curr_expr=curr_expr,
            prev_step_num=prev_step_num,
            curr_step_num=curr_step_num,
            accumulated_restrictions=accumulated_restrictions,
        )

        transitions.append(transition)

        # Update running domain restrictions with conditions discovered in transition
        for c in transition.conditions:
            if "!=" in c and c not in accumulated_restrictions:
                accumulated_restrictions.append(c)

        if transition.transition_status == "valid":
            valid_count += 1
        elif transition.transition_status == "invalid":
            invalid_count += 1
        else:
            unsupported_count += 1

    # 5. Determine overall trajectory status
    if invalid_count > 0:
        overall_status = "has_invalid_transitions"
    elif valid_count > 0:
        overall_status = "valid"
    elif unsupported_count > 0:
        overall_status = "unsupported"
    else:
        overall_status = "empty"

    summary_explanation = (
        f"Evaluated {len(transitions)} mathematical transitions: "
        f"{valid_count} valid, {invalid_count} invalid, {unsupported_count} unsupported."
    )

    return TransitionVerificationResult(
        transitions=transitions,
        total_transitions=len(transitions),
        valid_transitions=valid_count,
        invalid_transitions=invalid_count,
        unsupported_transitions=unsupported_count,
        status=overall_status,
        explanation=summary_explanation,
    )


# ==============================================================================
# Standalone Demonstrations
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3, OBJECTIVE 22: MATHEMATICAL TRANSITION VERIFIER DEMONSTRATIONS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Demonstration 1: Valid Algebraic Transformation
    # --------------------------------------------------------------------------
    print("\nDEMONSTRATION 1: VALID ALGEBRAIC TRANSFORMATION")
    print("-" * 80)
    d1_text = (
        "**Step 1: Solve equation**\n"
        "2 * x + 4 = 10\n"
        "2 * x = 6\n"
        "x = 3\n\n"
        "Final Answer: 3"
    )
    r1 = verify_step_transitions(d1_text)
    print(f"Summary: {r1.explanation}")
    for idx, t in enumerate(r1.transitions, 1):
        print(f"Transition {idx}: [Step {t.prev_step_number}] '{t.prev_expression}' -> [Step {t.curr_step_number}] '{t.curr_expression}'")
        print(f"  Status: {t.transition_status.upper()} | Type: {t.transformation_type}")
        print(f"  Explanation: {t.explanation}")
        if t.conditions:
            print(f"  Conditions:  {t.conditions}")

    # --------------------------------------------------------------------------
    # Demonstration 2: Intentionally Incorrect Transformation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMONSTRATION 2: INTENTIONALLY INCORRECT TRANSFORMATION")
    print("-" * 80)
    d2_text = (
        "**Step 1: Subtraction calculation error**\n"
        "2 * x + 4 = 10\n"
        "2 * x = 14\n\n"
        "Final Answer: 7"
    )
    r2 = verify_step_transitions(d2_text)
    print(f"Summary: {r2.explanation}")
    for idx, t in enumerate(r2.transitions, 1):
        print(f"Transition {idx}: [Step {t.prev_step_number}] '{t.prev_expression}' -> [Step {t.curr_step_number}] '{t.curr_expression}'")
        print(f"  Status: {t.transition_status.upper()} | Type: {t.transformation_type}")
        print(f"  Explanation: {t.explanation}")
        if t.conditions:
            print(f"  Conditions:  {t.conditions}")

    # --------------------------------------------------------------------------
    # Demonstration 3: Transformation Involving Division by a Variable
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMONSTRATION 3: TRANSFORMATION INVOLVING DIVISION BY A VARIABLE")
    print("-" * 80)
    d3_text = (
        "**Step 1: Divide by variable**\n"
        "x**2 = 2 * x\n"
        "x = 2\n\n"
        "Final Answer: 2"
    )
    r3 = verify_step_transitions(d3_text)
    print(f"Summary: {r3.explanation}")
    for idx, t in enumerate(r3.transitions, 1):
        print(f"Transition {idx}: [Step {t.prev_step_number}] '{t.prev_expression}' -> [Step {t.curr_step_number}] '{t.curr_expression}'")
        print(f"  Status: {t.transition_status.upper()} | Type: {t.transformation_type}")
        print(f"  Explanation: {t.explanation}")
        if t.conditions:
            print(f"  Conditions:  {t.conditions}")

    # --------------------------------------------------------------------------
    # Demonstration 4: Transformation Involving Squaring Both Sides
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMONSTRATION 4: TRANSFORMATION INVOLVING SQUARING BOTH SIDES")
    print("-" * 80)
    d4_text = (
        "**Step 1: Square both sides**\n"
        "x = 3\n"
        "x**2 = 9\n\n"
        "Final Answer: 9"
    )
    r4 = verify_step_transitions(d4_text)
    print(f"Summary: {r4.explanation}")
    for idx, t in enumerate(r4.transitions, 1):
        print(f"Transition {idx}: [Step {t.prev_step_number}] '{t.prev_expression}' -> [Step {t.curr_step_number}] '{t.curr_expression}'")
        print(f"  Status: {t.transition_status.upper()} | Type: {t.transformation_type}")
        print(f"  Explanation: {t.explanation}")
        if t.conditions:
            print(f"  Conditions:  {t.conditions}")
    print("=" * 80)
