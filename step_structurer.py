"""
step_structurer.py
------------------
Phase 3: Automatic Error Detection in LLM Reasoning.
Objective 20: Structured Mathematical Representation.

Converts intermediate reasoning steps extracted by step_extractor.py into
structured mathematical representations prepared for symbolic verification (Objective 21).
Distinguishes arithmetic equalities, labeled equalities, algebraic equations,
natural-language formulas, and standalone arithmetic expressions.
Preserves original text, LHS/RHS components, step order, and unparseable/ambiguous expressions.
"""

from dataclasses import dataclass, field
import ast
import os
import re
import sys
from typing import List, Optional, Dict, Any, Union

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_extractor import ExtractedReasoning, ReasoningStep, extract_reasoning_steps
from equation_extractor import _preprocess_text, _clean_token, _strip_units
from symbolic_verifier import _clean_side


@dataclass
class StructuredExpression:
    """Represents a single structured mathematical statement extracted from a reasoning step."""
    step_number: int                          # Source step number (1, 2, 3...)
    expression_type: str                      # 'arithmetic_equality', 'labeled_equality', 'algebraic_equality',
                                              # 'natural_language_formula', 'standalone_expression', 'unsupported'
    raw_text: str                             # Original raw snippet / clause
    lhs_raw: Optional[str] = None             # Raw left-hand side if equality
    rhs_raw: Optional[str] = None             # Raw right-hand side if equality
    lhs_normalized: Optional[str] = None      # Normalized LHS expression for verification
    rhs_normalized: Optional[str] = None      # Normalized RHS expression for verification
    normalized_equation: Optional[str] = None # Full normalized equation (e.g. "3 * 40 = 120") if applicable
    is_equality: bool = False                 # True if an explicit '=' operator is present
    is_valid_syntax: bool = False             # True if mathematical portion parses with valid AST
    label: Optional[str] = None               # Natural-language label or variable name on LHS
    unsupported_reason: Optional[str] = None  # Reason why expression cannot be normalized or verified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "expression_type": self.expression_type,
            "raw_text": self.raw_text,
            "lhs_raw": self.lhs_raw,
            "rhs_raw": self.rhs_raw,
            "lhs_normalized": self.lhs_normalized,
            "rhs_normalized": self.rhs_normalized,
            "normalized_equation": self.normalized_equation,
            "is_equality": self.is_equality,
            "is_valid_syntax": self.is_valid_syntax,
            "label": self.label,
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass
class StructuredStep:
    """Represents an extracted reasoning step with its ordered mathematical expressions."""
    step_number: int
    step_header: str
    original_text: str
    expressions: List[StructuredExpression] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_header": self.step_header,
            "original_text": self.original_text,
            "expressions": [e.to_dict() for e in self.expressions],
        }


@dataclass
class StructuredReasoning:
    """Top-level container holding structured mathematical representations of all reasoning steps."""
    raw_response: str
    steps: List[StructuredStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_answer_raw: Optional[str] = None
    total_expressions: int = 0
    unsupported_expressions: int = 0
    status: str = "success"                   # 'success', 'empty_response', 'no_steps_found', 'malformed'
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "final_answer_raw": self.final_answer_raw,
            "total_expressions": self.total_expressions,
            "unsupported_expressions": self.unsupported_expressions,
            "status": self.status,
            "explanation": self.explanation,
        }


def _check_ast_validity(expr_str: str) -> bool:
    """Checks whether an expression string can be safely parsed by Python's eval AST."""
    if not expr_str or not expr_str.strip():
        return False
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
        # Ensure it contains supported expression nodes
        return isinstance(tree, ast.Expression)
    except Exception:
        return False


def _is_single_identifier(text: str) -> bool:
    """Checks whether text is a valid single Python identifier/variable name."""
    s = text.strip()
    return bool(re.match(r"^[a-zA-Z_]\w*$", s))


def _clean_token_preserving_math(token: str) -> str:
    """Strips markdown bold, italics, backticks, leading bullets from edges while preserving math operators like '**'."""
    t = token.strip()
    while t.startswith("**") and t.endswith("**") and len(t) >= 4:
        t = t[2:-2].strip()
    while t.startswith("__") and t.endswith("__") and len(t) >= 4:
        t = t[2:-2].strip()
    while t.startswith("`") and t.endswith("`") and len(t) >= 2:
        t = t[1:-1].strip()
    t = re.sub(r"^[\*\•\–\—\s]+", "", t).strip()
    t = re.sub(r"^\-\s+", "", t).strip()
    while t and t[-1] in ".,;:!?":
        t = t[:-1].strip()
    return t.strip()


def _clean_math_expression(raw_expr: str) -> str:
    """
    Cleans arithmetic expressions: strips currency symbols, normalizes operators,
    unescapes LaTeX, removes unit words, and formats caret exponents.
    """
    if not raw_expr:
        return ""
    prep = _preprocess_text(raw_expr)
    side_clean = _clean_side(prep)
    # If the expression already forms a syntactically valid AST expression (e.g. algebraic variables, powers),
    # preserve it directly without stripping variables as if they were units.
    if _check_ast_validity(side_clean):
        return re.sub(r"\s+", " ", side_clean).strip()
    # Otherwise strip unit words while preserving operators and numbers
    units_clean = _strip_units(side_clean)
    # Final cleanup of extraneous spaces
    cleaned = re.sub(r"\s+", " ", units_clean).strip()
    return cleaned


def _structure_equality_clause(step_num: int, clause: str) -> List[StructuredExpression]:
    """
    Parses an equality clause (e.g. 'Cost = 3 * 40 = 120') into structured expressions.
    Decomposes chained equalities into adjacent pairwise relationships.
    """
    expressions: List[StructuredExpression] = []
    # Split on '='
    parts = clause.split("=")
    if len(parts) < 2:
        return expressions

    for i in range(len(parts) - 1):
        raw_lhs = parts[i].strip()
        raw_rhs = parts[i + 1].strip()

        # Clean token wrappers while preserving math operators (like '**')
        cand_lhs = _clean_token_preserving_math(raw_lhs)
        cand_rhs = _clean_token_preserving_math(raw_rhs)

        # Remove leading conjunctions or step headings from LHS (e.g. 'Step 1:', 'So', 'Therefore', 'Let')
        cand_lhs = re.sub(r"^(?:step\s*\d+\s*[:.)-]?|let|so|then|and|therefore|we\s+have)\s+", "", cand_lhs, flags=re.IGNORECASE).strip()
        pair_raw_text = f"{raw_lhs} = {raw_rhs}"

        # 1. Check if both sides are purely natural-language text (no digits on either side)
        has_digit_lhs = bool(re.search(r"\d", cand_lhs))
        has_digit_rhs = bool(re.search(r"\d", cand_rhs))

        # Check if LHS is a single identifier (e.g. 'x', 'total', 'Rate')
        if _is_single_identifier(cand_lhs):
            norm_lhs = cand_lhs
            lhs_is_ast = True
        else:
            norm_lhs = _clean_math_expression(cand_lhs)
            lhs_is_ast = _check_ast_validity(norm_lhs)

        norm_rhs = _clean_math_expression(cand_rhs)
        rhs_is_ast = _check_ast_validity(norm_rhs)

        # Case A: Both sides have no digits and are conceptual / natural-language phrases
        if not has_digit_lhs and not has_digit_rhs and not (_is_single_identifier(cand_lhs) and _is_single_identifier(cand_rhs)):
            expressions.append(
                StructuredExpression(
                    step_number=step_num,
                    expression_type="natural_language_formula",
                    raw_text=pair_raw_text,
                    lhs_raw=raw_lhs,
                    rhs_raw=raw_rhs,
                    lhs_normalized=None,
                    rhs_normalized=None,
                    normalized_equation=None,
                    is_equality=True,
                    is_valid_syntax=False,
                    label=cand_lhs,
                    unsupported_reason="Natural-language formula: terms are conceptual phrases rather than numeric or algebraic expressions.",
                )
            )
            continue

        # Case B: LHS is not valid AST
        if not lhs_is_ast:
            # Check if cand_lhs was an attempted arithmetic expression that failed parsing (e.g. "3 * (40 +")
            if re.search(r"[+\-*/^×÷()]", cand_lhs):
                expressions.append(
                    StructuredExpression(
                        step_number=step_num,
                        expression_type="unsupported",
                        raw_text=pair_raw_text,
                        lhs_raw=raw_lhs,
                        rhs_raw=raw_rhs,
                        lhs_normalized=None,
                        rhs_normalized=norm_rhs if rhs_is_ast else None,
                        normalized_equation=None,
                        is_equality=True,
                        is_valid_syntax=False,
                        label=cand_lhs,
                        unsupported_reason="LHS contains invalid or unparseable mathematical syntax.",
                    )
                )
                continue

            # Otherwise, LHS is a descriptive natural-language label like "Cost of notebooks" or "Total cost"
            if rhs_is_ast:
                expressions.append(
                    StructuredExpression(
                        step_number=step_num,
                        expression_type="labeled_equality",
                        raw_text=pair_raw_text,
                        lhs_raw=raw_lhs,
                        rhs_raw=raw_rhs,
                        lhs_normalized=None,
                        rhs_normalized=norm_rhs,
                        normalized_equation=f"{cand_lhs} = {norm_rhs}",
                        is_equality=True,
                        is_valid_syntax=True,
                        label=cand_lhs,
                        unsupported_reason="LHS is a multi-word natural-language label; RHS is valid mathematical expression.",
                    )
                )
            else:
                expressions.append(
                    StructuredExpression(
                        step_number=step_num,
                        expression_type="unsupported",
                        raw_text=pair_raw_text,
                        lhs_raw=raw_lhs,
                        rhs_raw=raw_rhs,
                        lhs_normalized=None,
                        rhs_normalized=None,
                        normalized_equation=None,
                        is_equality=True,
                        is_valid_syntax=False,
                        label=cand_lhs,
                        unsupported_reason="Neither LHS nor RHS could be parsed as valid mathematical syntax.",
                    )
                )
            continue

        # Case C: LHS is a valid AST expression (either single variable 'x' or arithmetic '3 * 40')
        if lhs_is_ast and rhs_is_ast:
            lhs_has_letters = bool(re.search(r"[a-zA-Z]", norm_lhs))
            rhs_has_letters = bool(re.search(r"[a-zA-Z]", norm_rhs))

            if not lhs_has_letters and not rhs_has_letters:
                # Pure arithmetic equality: e.g. "3 * 40 = 120"
                expressions.append(
                    StructuredExpression(
                        step_number=step_num,
                        expression_type="arithmetic_equality",
                        raw_text=pair_raw_text,
                        lhs_raw=raw_lhs,
                        rhs_raw=raw_rhs,
                        lhs_normalized=norm_lhs,
                        rhs_normalized=norm_rhs,
                        normalized_equation=f"{norm_lhs} = {norm_rhs}",
                        is_equality=True,
                        is_valid_syntax=True,
                        label=None,
                        unsupported_reason=None,
                    )
                )
            elif _is_single_identifier(norm_lhs) and not rhs_has_letters:
                # Variable assignment: e.g. "x = 120"
                expressions.append(
                    StructuredExpression(
                        step_number=step_num,
                        expression_type="labeled_equality",
                        raw_text=pair_raw_text,
                        lhs_raw=raw_lhs,
                        rhs_raw=raw_rhs,
                        lhs_normalized=norm_lhs,
                        rhs_normalized=norm_rhs,
                        normalized_equation=f"{norm_lhs} = {norm_rhs}",
                        is_equality=True,
                        is_valid_syntax=True,
                        label=norm_lhs,
                        unsupported_reason=None,
                    )
                )
            else:
                # Algebraic equation: e.g. "2 * x + 5 = 15"
                expressions.append(
                    StructuredExpression(
                        step_number=step_num,
                        expression_type="algebraic_equality",
                        raw_text=pair_raw_text,
                        lhs_raw=raw_lhs,
                        rhs_raw=raw_rhs,
                        lhs_normalized=norm_lhs,
                        rhs_normalized=norm_rhs,
                        normalized_equation=f"{norm_lhs} = {norm_rhs}",
                        is_equality=True,
                        is_valid_syntax=True,
                        label=norm_lhs if _is_single_identifier(norm_lhs) else None,
                        unsupported_reason=None,
                    )
                )
            continue

        # Case D: LHS is valid AST but RHS is not
        expressions.append(
            StructuredExpression(
                step_number=step_num,
                expression_type="unsupported",
                raw_text=pair_raw_text,
                lhs_raw=raw_lhs,
                rhs_raw=raw_rhs,
                lhs_normalized=norm_lhs if lhs_is_ast else None,
                rhs_normalized=norm_rhs if rhs_is_ast else None,
                normalized_equation=None,
                is_equality=True,
                is_valid_syntax=False,
                label=cand_lhs,
                unsupported_reason="RHS contains invalid or unparseable mathematical syntax.",
            )
        )

    return expressions


def _detect_standalone_expression(step_num: int, line_text: str) -> Optional[StructuredExpression]:
    """
    Detects whether a non-equality line contains a standalone arithmetic/algebraic expression
    (e.g., '3 * 40 + 2 * 15' or '(120 + 30) / 2').
    """
    clean_line = line_text.strip()
    if not clean_line or "=" in clean_line:
        return None

    # Line must have digits and arithmetic operators
    has_digit = bool(re.search(r"\d", clean_line))
    has_operator = bool(re.search(r"[+\-*/^×÷]", clean_line))
    if not (has_digit and has_operator):
        return None

    # Check for sentence-like narrative patterns that happen to mention a number and operator
    # e.g. "The shopkeeper buys 3 notebooks at ₹40 each."
    words = re.findall(r"\b[a-zA-Z]{2,}\b", clean_line)
    if len(words) >= 4:
        # High word count indicates descriptive narrative prose
        return None

    norm_expr = _clean_math_expression(clean_line)
    if _check_ast_validity(norm_expr):
        return StructuredExpression(
            step_number=step_num,
            expression_type="standalone_expression",
            raw_text=clean_line,
            lhs_raw=None,
            rhs_raw=clean_line,
            lhs_normalized=None,
            rhs_normalized=norm_expr,
            normalized_equation=None,
            is_equality=False,
            is_valid_syntax=True,
            label=None,
            unsupported_reason=None,
        )

    return None


def structure_reasoning_steps(extracted_reasoning: Union[ExtractedReasoning, str, Dict[str, Any]]) -> StructuredReasoning:
    """
    Converts intermediate reasoning steps into structured mathematical representations.

    Parameters
    ----------
    extracted_reasoning : ExtractedReasoning, str, or dict
        Extracted reasoning output from step_extractor.py, or a raw textual response.

    Returns
    -------
    StructuredReasoning
        Dataclass containing the structured steps, mathematical expressions,
        LHS/RHS separations, and classification summaries.
    """
    # 1. Coerce input to ExtractedReasoning
    if isinstance(extracted_reasoning, str):
        er = extract_reasoning_steps(extracted_reasoning)
    elif isinstance(extracted_reasoning, dict):
        raw_resp = extracted_reasoning.get("raw_response", "")
        er = extract_reasoning_steps(raw_resp)
    elif isinstance(extracted_reasoning, ExtractedReasoning):
        er = extracted_reasoning
    else:
        return StructuredReasoning(
            raw_response="",
            steps=[],
            status="empty_response",
            explanation="Invalid input type provided to structure_reasoning_steps.",
        )

    if er.status == "empty_response":
        return StructuredReasoning(
            raw_response=er.raw_response,
            steps=[],
            final_answer=er.final_answer,
            final_answer_raw=er.final_answer_raw,
            status="empty_response",
            explanation=er.explanation,
        )

    if er.status == "no_steps_found":
        return StructuredReasoning(
            raw_response=er.raw_response,
            steps=[],
            final_answer=er.final_answer,
            final_answer_raw=er.final_answer_raw,
            status="no_steps_found",
            explanation=er.explanation,
        )

    structured_steps: List[StructuredStep] = []
    total_expr_count = 0
    unsupported_expr_count = 0

    for step in er.steps:
        step_expressions: List[StructuredExpression] = []

        # Process lines in the step
        for line in step.text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            # Skip pure markdown headers like '**Step 1: ...**' unless they contain explicit equalities
            if re.match(r"^\s*(?:#{1,6}\s*)?(?:\*\*|\*|__)?\s*Step\s+\d+", line_str, re.IGNORECASE) and "=" not in line_str:
                continue

            if "=" in line_str:
                # Equality detected: parse into pairwise expressions
                eqs = _structure_equality_clause(step.step_number, line_str)
                step_expressions.extend(eqs)
            else:
                # Check for standalone arithmetic expression
                standalone = _detect_standalone_expression(step.step_number, line_str)
                if standalone:
                    step_expressions.append(standalone)

        total_expr_count += len(step_expressions)
        unsupported_expr_count += sum(1 for e in step_expressions if e.expression_type in ("unsupported", "natural_language_formula"))

        structured_steps.append(
            StructuredStep(
                step_number=step.step_number,
                step_header=step.step_header,
                original_text=step.text,
                expressions=step_expressions,
            )
        )

    explanation = (
        f"Successfully structured {len(structured_steps)} steps containing "
        f"{total_expr_count} mathematical expressions "
        f"({total_expr_count - unsupported_expr_count} valid/verifiable, "
        f"{unsupported_expr_count} natural-language/unsupported)."
    )

    return StructuredReasoning(
        raw_response=er.raw_response,
        steps=structured_steps,
        final_answer=er.final_answer,
        final_answer_raw=er.final_answer_raw,
        total_expressions=total_expr_count,
        unsupported_expressions=unsupported_expr_count,
        status="success" if er.status == "success" else er.status,
        explanation=explanation,
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

    extracted = extract_reasoning_steps(demo_gemini_response)
    structured = structure_reasoning_steps(extracted)

    print("=" * 80)
    print("PHASE 3, OBJECTIVE 20: STRUCTURED MATHEMATICAL STEP REPRESENTATION DEMONSTRATION")
    print("=" * 80)
    print(f"Extraction Status: {structured.status.upper()}")
    print(f"Summary:           {structured.explanation}")
    print(f"Total Expressions: {structured.total_expressions}")
    print(f"Unsupported/NL:    {structured.unsupported_expressions}")
    print(f"Final Answer:      {structured.final_answer} ({structured.final_answer_raw})")
    print("=" * 80)

    for step in structured.steps:
        print(f"\n[STEP {step.step_number}] Header: {step.step_header}")
        print(f"Expressions ({len(step.expressions)}):")
        for i, expr in enumerate(step.expressions, 1):
            print(f"  {i}. Type:   {expr.expression_type}")
            print(f"     Raw:    '{expr.raw_text}'")
            if expr.is_equality:
                print(f"     LHS:    raw='{expr.lhs_raw}', norm={repr(expr.lhs_normalized)}")
                print(f"     RHS:    raw='{expr.rhs_raw}', norm={repr(expr.rhs_normalized)}")
            else:
                print(f"     Norm:   '{expr.rhs_normalized}'")
            print(f"     Valid:  {expr.is_valid_syntax}")
            if expr.label:
                print(f"     Label:  '{expr.label}'")
            if expr.unsupported_reason:
                print(f"     Note:   {expr.unsupported_reason}")
    print("\n" + "=" * 80)
