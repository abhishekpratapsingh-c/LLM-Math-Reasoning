"""
deterministic_solver_pipeline.py
--------------------------------
Phase 2, Objective 12: Deterministic, Offline Problem-Solving Pipeline.

Connects word problems directly to symbolic equations and solves them:
1. Pattern Matching: Classifies word problem into a recognized template (problem_patterns.py).
2. Equation Construction: Converts extracted numeric components into an algebraic equation (equation_builder.py).
3. Symbolic Solving: Solves for target variable x using secure AST and SymPy (symbolic_solver.py).
4. Solution Verification: Verifies that the computed solution satisfies the equation (symbolic_solver.py).
5. Result Formatting: Cleans and formats the final answer.

Features:
- 100% offline and deterministic.
- Zero external API costs, zero LLM calls.
- Enforces existing safety limits via symbolic_solver.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import sympy

from problem_patterns import PatternMatchResult, classify_problem
from equation_builder import build_equation
from symbolic_solver import solve_equation, verify_solution


@dataclass
class PipelineResult:
    """Structured result of the deterministic problem-solving pipeline."""
    status: str                         # 'solved', 'unsupported', 'invalid', 'error'
    original_question: str
    detected_category: str              # Category name or 'unsupported' / 'none'
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    constructed_equation: Optional[str] = None
    calculated_answer: Optional[Union[int, float]] = None
    calculated_answer_str: Optional[str] = None
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "original_question": self.original_question,
            "detected_category": self.detected_category,
            "extracted_data": self.extracted_data,
            "constructed_equation": self.constructed_equation,
            "calculated_answer": self.calculated_answer,
            "calculated_answer_str": self.calculated_answer_str,
            "explanation": self.explanation,
        }


def solve_problem(question: str) -> PipelineResult:
    """End-to-end pipeline to solve structured mathematical word problems deterministically.

    Parameters
    ----------
    question : str
        The raw word problem text.

    Returns
    -------
    PipelineResult
        Structured outcome containing status ('solved', 'unsupported', 'invalid'),
        detected category, extracted data, constructed equation, calculated answer, and explanation.
    """
    if not isinstance(question, str) or not question.strip():
        return PipelineResult(
            status="invalid",
            original_question=str(question) if question is not None else "",
            detected_category="none",
            explanation="Input question is empty or not a string.",
        )

    # Step 1: Deterministic Pattern Classification
    match_result = classify_problem(question)
    if not match_result.matched:
        return PipelineResult(
            status="unsupported",
            original_question=question,
            detected_category=match_result.category,
            extracted_data={},
            explanation=match_result.explanation or "Question does not match any supported deterministic category.",
        )

    category = match_result.category
    extracted_data = match_result.extracted_data

    # Step 2: Equation Construction
    equation_str = build_equation(match_result)
    if not equation_str:
        return PipelineResult(
            status="invalid",
            original_question=question,
            detected_category=category,
            extracted_data=extracted_data,
            explanation=f"Failed to construct valid equation for category '{category}'.",
        )

    # Step 3: Symbolic Equation Solving
    solve_res = solve_equation(equation_str, variable="x")
    if solve_res.status != "solved" or not solve_res.solutions:
        return PipelineResult(
            status="invalid",
            original_question=question,
            detected_category=category,
            extracted_data=extracted_data,
            constructed_equation=equation_str,
            explanation=f"Symbolic solver could not solve equation: {solve_res.explanation}",
        )

    # Choose primary solution: for word problems, prefer positive real numbers
    selected_solution_str = solve_res.solutions[0]
    if len(solve_res.solutions) > 1:
        for sol in solve_res.solutions:
            try:
                val = float(sympy.Rational(sol))
                if val > 0:
                    selected_solution_str = sol
                    break
            except Exception:
                pass

    # Step 4: Verification of Proposed Solution
    verif_res = verify_solution(equation_str, selected_solution_str)
    if verif_res.status != "correct":
        return PipelineResult(
            status="invalid",
            original_question=question,
            detected_category=category,
            extracted_data=extracted_data,
            constructed_equation=equation_str,
            explanation=f"Computed solution failed symbolic verification: {verif_res.explanation}",
        )

    # Step 5: Format Answer
    try:
        rational_val = sympy.Rational(selected_solution_str)
        if rational_val.q == 1:
            numeric_ans: Union[int, float] = int(rational_val.p)
            ans_str = str(numeric_ans)
        else:
            numeric_ans = float(rational_val)
            ans_str = f"{numeric_ans:g}"
    except Exception:
        try:
            numeric_ans = float(selected_solution_str)
            if numeric_ans.is_integer():
                numeric_ans = int(numeric_ans)
                ans_str = str(numeric_ans)
            else:
                ans_str = f"{numeric_ans:g}"
        except Exception:
            numeric_ans = None
            ans_str = selected_solution_str

    if numeric_ans is not None and isinstance(numeric_ans, (int, float)) and numeric_ans <= 0:
        return PipelineResult(
            status="invalid",
            original_question=question,
            detected_category=category,
            extracted_data=extracted_data,
            constructed_equation=equation_str,
            explanation=f"Calculated non-positive answer ({numeric_ans}) which is invalid for word problem context.",
        )

    explanation = (
        f"Category: {category}. Equation: {equation_str}. "
        f"Solved {solve_res.variable} = {ans_str}. Verified: {verif_res.explanation}"
    )

    return PipelineResult(
        status="solved",
        original_question=question,
        detected_category=category,
        extracted_data=extracted_data,
        constructed_equation=equation_str,
        calculated_answer=numeric_ans,
        calculated_answer_str=ans_str,
        explanation=explanation,
    )
