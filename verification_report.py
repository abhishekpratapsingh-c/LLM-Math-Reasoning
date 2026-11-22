"""
verification_report.py
----------------------
Phase 3: Automatic Error Detection in LLM Reasoning.
Objective 25: Generate a Final Verification Report.

Combines the outputs of the hybrid LLM + SymPy reasoning error-detection pipeline:
1. Problem information (original problem, LLM final answer).
2. Generated reasoning (steps, identified expressions, normalized forms).
3. Mathematical verification (SymPy evaluation, LHS/RHS values, solutions, domain restrictions).
4. Transition verification (consecutive steps, algebraic equivalence, conditions).
5. Detected errors and warnings (definite errors, invalid transformations, lost solutions,
   division by zero, domain violations, conditional warnings, unsupported expressions).
6. LLM explanations from Objective 24 (separate from deterministic solver evidence).
7. Overall verification summary and metrics.

Generates:
- Structured Python dataclasses (VerificationReport).
- Human-readable Markdown reports (format_markdown_report).
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
from reasoning_error_detector import (
    DetectedIssue,
    StepErrorReport,
    ReasoningErrorReport,
    detect_reasoning_errors,
)
from reasoning_feedback import (
    StepFeedback,
    ReasoningFeedbackResult,
)


# ==============================================================================
# Verification Report Dataclasses
# ==============================================================================
@dataclass
class ProblemInfo:
    """Problem statement and LLM final answer information."""
    problem: str
    final_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "final_answer": self.final_answer,
        }


@dataclass
class ExpressionRecord:
    """Detailed record of an individual mathematical expression and its verification."""
    step_number: int
    raw_text: str
    expression_type: str
    lhs: Optional[str] = None
    rhs: Optional[str] = None
    normalized_lhs: Optional[str] = None
    normalized_rhs: Optional[str] = None
    label: Optional[str] = None
    # Verification details
    verification_status: Optional[str] = None       # 'verified', 'incorrect', 'invalid', 'unsupported'
    solver_explanation: Optional[str] = None
    solutions: List[str] = field(default_factory=list)
    domain_restrictions: List[str] = field(default_factory=list)
    lhs_value: Optional[str] = None
    rhs_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "raw_text": self.raw_text,
            "expression_type": self.expression_type,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "normalized_lhs": self.normalized_lhs,
            "normalized_rhs": self.normalized_rhs,
            "label": self.label,
            "verification_status": self.verification_status,
            "solver_explanation": self.solver_explanation,
            "solutions": self.solutions,
            "domain_restrictions": self.domain_restrictions,
            "lhs_value": self.lhs_value,
            "rhs_value": self.rhs_value,
        }


@dataclass
class StepRecord:
    """Record of an extracted reasoning step with its expressions."""
    step_number: int
    step_header: str
    original_text: str
    expressions: List[ExpressionRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_header": self.step_header,
            "original_text": self.original_text,
            "expressions": [e.to_dict() for e in self.expressions],
        }


@dataclass
class TransitionRecord:
    """Record of a transition check between consecutive mathematical expressions."""
    prev_step_number: Optional[int]
    prev_expression: Optional[str]
    curr_step_number: int
    curr_expression: str
    is_valid: bool
    transition_status: str               # 'valid', 'invalid', 'warning', 'unsupported'
    transformation_type: str
    explanation: str
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prev_step_number": self.prev_step_number,
            "prev_expression": self.prev_expression,
            "curr_step_number": self.curr_step_number,
            "curr_expression": self.curr_expression,
            "is_valid": self.is_valid,
            "transition_status": self.transition_status,
            "transformation_type": self.transformation_type,
            "explanation": self.explanation,
            "conditions": self.conditions,
        }


@dataclass
class DetectedIssueRecord:
    """Record of a detected mathematical issue, error, warning, or unsupported statement."""
    step_number: int
    expression: str
    error_category: str                 # 'incorrect_assertion', 'invalid_transformation', etc.
    severity: str                       # 'definite_error', 'potential_issue', 'unsupported'
    is_definite_error: bool
    explanation: str
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "expression": self.expression,
            "error_category": self.error_category,
            "severity": self.severity,
            "is_definite_error": self.is_definite_error,
            "explanation": self.explanation,
            "conditions": self.conditions,
        }


@dataclass
class LLMExplanationRecord:
    """Record of an explanation generated by the LLM for a specific step."""
    step_number: int
    expression: str
    error_category: str
    severity: str
    solver_evidence: str
    llm_explanation: str
    status: str                         # 'success', 'failed', 'unavailable'
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "expression": self.expression,
            "error_category": self.error_category,
            "severity": self.severity,
            "solver_evidence": self.solver_evidence,
            "llm_explanation": self.llm_explanation,
            "status": self.status,
            "conditions": self.conditions,
        }


@dataclass
class VerificationSummary:
    """Summary metrics and overall status of the mathematical verification."""
    total_steps: int = 0
    total_expressions: int = 0
    total_verified_expressions: int = 0
    total_incorrect_expressions: int = 0
    total_invalid_expressions: int = 0
    total_unsupported_expressions: int = 0
    total_detected_errors: int = 0
    total_warnings: int = 0
    overall_status: str = "verified"     # 'verified', 'errors_detected', 'warnings', 'partially_verified', 'verification_failed'
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_expressions": self.total_expressions,
            "total_verified_expressions": self.total_verified_expressions,
            "total_incorrect_expressions": self.total_incorrect_expressions,
            "total_invalid_expressions": self.total_invalid_expressions,
            "total_unsupported_expressions": self.total_unsupported_expressions,
            "total_detected_errors": self.total_detected_errors,
            "total_warnings": self.total_warnings,
            "overall_status": self.overall_status,
            "summary_text": self.summary_text,
        }


@dataclass
class VerificationReport:
    """Comprehensive mathematical reasoning verification report."""
    problem_info: ProblemInfo
    steps: List[StepRecord] = field(default_factory=list)
    transitions: List[TransitionRecord] = field(default_factory=list)
    detected_errors: List[DetectedIssueRecord] = field(default_factory=list)
    detected_warnings: List[DetectedIssueRecord] = field(default_factory=list)
    unsupported_issues: List[DetectedIssueRecord] = field(default_factory=list)
    llm_explanations: List[LLMExplanationRecord] = field(default_factory=list)
    summary: VerificationSummary = field(default_factory=VerificationSummary)
    raw_response: Optional[str] = None

    def to_markdown(self) -> str:
        return format_markdown_report(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_info": self.problem_info.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "transitions": [t.to_dict() for t in self.transitions],
            "detected_errors": [e.to_dict() for e in self.detected_errors],
            "detected_warnings": [w.to_dict() for w in self.detected_warnings],
            "unsupported_issues": [u.to_dict() for u in self.unsupported_issues],
            "llm_explanations": [x.to_dict() for x in self.llm_explanations],
            "summary": self.summary.to_dict(),
            "raw_response": self.raw_response,
        }


# ==============================================================================
# Markdown Report Formatting
# ==============================================================================
def format_markdown_report(report: VerificationReport) -> str:
    """
    Renders a human-readable, beautifully structured Markdown verification report
    from a VerificationReport object.
    """
    lines: List[str] = []

    # Title & Overall Badge
    status_emojis = {
        "verified": "✅ Verified",
        "errors_detected": "❌ Errors Detected",
        "warnings": "⚠️ Warnings Present",
        "partially_verified": "ℹ️ Partially Verified",
        "verification_failed": "⛔ Verification Failed",
    }
    status_str = status_emojis.get(report.summary.overall_status, report.summary.overall_status)

    lines.append("# Mathematical Reasoning Verification Report")
    lines.append("")
    lines.append(f"**Overall Verification Status:** `{report.summary.overall_status}` ({status_str})")
    lines.append("")

    # Section A: Problem Information
    lines.append("## A. Problem Information")
    lines.append(f"- **Problem Statement:** {report.problem_info.problem.strip()}")
    final_ans = report.problem_info.final_answer if report.problem_info.final_answer else "None identified"
    lines.append(f"- **LLM Final Answer:** `{final_ans}`")
    lines.append("")

    # Section B: Generated Reasoning
    lines.append("## B. Generated Reasoning Steps")
    if not report.steps:
        lines.append("*No reasoning steps were extracted or provided.*")
    else:
        for step in report.steps:
            lines.append(f"### {step.step_header}")
            lines.append(f"> {step.original_text.strip()}")
            lines.append("")
            if step.expressions:
                lines.append("**Identified Mathematical Expressions:**")
                for e_idx, expr in enumerate(step.expressions, start=1):
                    expr_details = [f"Type: `{expr.expression_type}`"]
                    if expr.lhs is not None and expr.rhs is not None:
                        norm_str = f"Normalized: `{expr.normalized_lhs} = {expr.normalized_rhs}`"
                        expr_details.append(norm_str)
                    if expr.label:
                        expr_details.append(f"Label: *{expr.label}*")
                    lines.append(f"{e_idx}. `{expr.raw_text}` ({', '.join(expr_details)})")
                lines.append("")
            else:
                lines.append("*No mathematical expressions identified in this step.*")
                lines.append("")

    # Section C: Mathematical Verification
    lines.append("## C. Mathematical Verification of Expressions")
    all_expressions = [expr for step in report.steps for expr in step.expressions]
    if not all_expressions:
        lines.append("*No verifiable mathematical expressions found.*")
        lines.append("")
    else:
        lines.append("| Step | Expression | Status | Solver Evidence / Evaluation | Solutions | Domain Restrictions |")
        lines.append("|:---:|:---|:---:|:---|:---:|:---|")
        for expr in all_expressions:
            status_icon = {
                "verified": "✅ Verified",
                "incorrect": "❌ Incorrect",
                "invalid": "🚫 Invalid",
                "unsupported": "⚠️ Unsupported",
            }.get(expr.verification_status, expr.verification_status or "Unknown")

            expl = expr.solver_explanation.replace("|", "\\|") if expr.solver_explanation else "None"
            sols = ", ".join(expr.solutions) if expr.solutions else "N/A"
            conds = ", ".join(expr.domain_restrictions) if expr.domain_restrictions else "None"
            clean_expr = expr.raw_text.replace("|", "\\|")
            lines.append(f"| {expr.step_number} | `{clean_expr}` | {status_icon} | {expl} | {sols} | {conds} |")
        lines.append("")

    # Section D: Transition Verification
    lines.append("## D. Step Transition Verification")
    if not report.transitions:
        lines.append("*No multi-expression transitions evaluated.*")
        lines.append("")
    else:
        lines.append("| From | To | Valid? | Type | Solver Findings & Conditions |")
        lines.append("|:---|:---|:---:|:---|:---|")
        for trans in report.transitions:
            from_desc = f"Step {trans.prev_step_number}: `{trans.prev_expression}`" if trans.prev_expression else "Initial"
            to_desc = f"Step {trans.curr_step_number}: `{trans.curr_expression}`"
            valid_icon = "✅ Yes" if trans.is_valid else ("⚠️ Warning" if trans.transition_status == "warning" else "❌ No")
            findings = trans.explanation.replace("|", "\\|")
            if trans.conditions:
                findings += f" (Conditions: {', '.join(trans.conditions)})"
            lines.append(f"| {from_desc} | {to_desc} | {valid_icon} | `{trans.transformation_type}` | {findings} |")
        lines.append("")

    # Section E: Detected Errors and Warnings
    lines.append("## E. Detected Errors and Warnings")
    has_issues = False

    if report.detected_errors:
        has_issues = True
        lines.append(f"### ❌ Definite Mathematical Errors ({len(report.detected_errors)})")
        for err in report.detected_errors:
            lines.append(f"- **Step {err.step_number} (`{err.expression}`)**: `[{err.error_category}]` {err.explanation}")
            if err.conditions:
                lines.append(f"  * *Conditions:* {', '.join(err.conditions)}")
        lines.append("")

    if report.detected_warnings:
        has_issues = True
        lines.append(f"### ⚠️ Conditional Warnings ({len(report.detected_warnings)})")
        for warn in report.detected_warnings:
            lines.append(f"- **Step {warn.step_number} (`{warn.expression}`)**: `[{warn.error_category}]` {warn.explanation}")
            if warn.conditions:
                lines.append(f"  * *Conditions:* {', '.join(warn.conditions)}")
        lines.append("")

    if report.unsupported_issues:
        has_issues = True
        lines.append(f"### ℹ️ Unsupported Expressions ({len(report.unsupported_issues)})")
        for unsup in report.unsupported_issues:
            lines.append(
                f"- **Step {unsup.step_number} (`{unsup.expression}`)**: `[{unsup.error_category}]` {unsup.explanation} "
                "(Note: Not classified as incorrect; formal symbolic verification was skipped.)"
            )
        lines.append("")

    if not has_issues:
        lines.append("✅ **No mathematical errors, invalid transformations, or warnings were detected.**")
        lines.append("")

    # Section F: LLM Explanations
    lines.append("## F. LLM Explanations")
    if not report.llm_explanations:
        if not has_issues:
            lines.append("*Reasoning is mathematically sound. Zero LLM explanations required.*")
        else:
            lines.append("*LLM explanations were not requested or were unavailable for detected issues.*")
        lines.append("")
    else:
        for idx, exp in enumerate(report.llm_explanations, start=1):
            lines.append(f"### Explanation {idx}: Step {exp.step_number} (`{exp.expression}`)")
            lines.append(f"- **Issue Category:** `{exp.error_category}` (Severity: `{exp.severity}`)")
            lines.append(f"- **Deterministic Solver Evidence:**\n  > {exp.solver_evidence.strip()}")
            if exp.conditions:
                lines.append(f"- **Active Conditions:** {', '.join(exp.conditions)}")

            if exp.status == "failed":
                lines.append(f"- **LLM Audit Explanation:** ⚠️ *LLM explanation failed or returned empty:* {exp.llm_explanation}")
            else:
                lines.append("- **LLM Audit Explanation:**")
                for line in exp.llm_explanation.strip().split("\n"):
                    lines.append(f"  {line}")
            lines.append("")

    # Section G: Overall Verification Summary
    lines.append("## G. Overall Verification Summary")
    s = report.summary
    lines.append(f"- **Overall Status:** `{s.overall_status}`")
    lines.append(f"- **Total Reasoning Steps:** {s.total_steps}")
    lines.append(f"- **Total Mathematical Expressions:** {s.total_expressions}")
    lines.append(f"- **Verified Expressions:** {s.total_verified_expressions}")
    lines.append(f"- **Incorrect Expressions:** {s.total_incorrect_expressions}")
    lines.append(f"- **Invalid Expressions:** {s.total_invalid_expressions}")
    lines.append(f"- **Unsupported Expressions:** {s.total_unsupported_expressions}")
    lines.append(f"- **Total Definite Errors:** {s.total_detected_errors}")
    lines.append(f"- **Total Conditional Warnings:** {s.total_warnings}")
    lines.append(f"- **Summary:** {s.summary_text}")
    lines.append("")

    return "\n".join(lines)


# ==============================================================================
# Final Verification Report Generator
# ==============================================================================
def generate_verification_report(
    problem: str,
    raw_response: Optional[str] = None,
    extracted_reasoning: Optional[ExtractedReasoning] = None,
    structured_reasoning: Optional[Union[StructuredReasoning, str]] = None,
    verified_reasoning: Optional[VerifiedReasoning] = None,
    transition_verification: Optional[TransitionVerificationResult] = None,
    error_report: Optional[ReasoningErrorReport] = None,
    feedback_result: Optional[ReasoningFeedbackResult] = None,
) -> VerificationReport:
    """
    Combines the outputs of the hybrid LLM + SymPy reasoning error-detection pipeline
    into a structured, human-readable verification report.

    Parameters
    ----------
    problem : str
        The original mathematical word problem or question.
    raw_response : str, optional
        The complete raw textual reasoning generated by the LLM.
    extracted_reasoning : ExtractedReasoning, optional
        Extracted steps and final answer from step_extractor.py.
    structured_reasoning : Union[StructuredReasoning, str], optional
        Structured reasoning representation from step_structurer.py (or raw string).
    verified_reasoning : VerifiedReasoning, optional
        Individual SymPy verification results from step_verifier.py.
    transition_verification : TransitionVerificationResult, optional
        Consecutive step transition verification results from step_transition_verifier.py.
    error_report : ReasoningErrorReport, optional
        Consolidated error detection report from reasoning_error_detector.py.
    feedback_result : ReasoningFeedbackResult, optional
        LLM feedback and explanations from reasoning_feedback.py.

    Returns
    -------
    VerificationReport
        The consolidated verification report object.

    Raises
    ------
    ValueError
        If the problem string is empty, whitespace, or invalid.
    """
    # 1. Validate problem input
    if problem is None or not isinstance(problem, str) or not problem.strip():
        raise ValueError("Problem input cannot be empty, whitespace, or non-string.")

    cleaned_problem = problem.strip()

    # 2. Resolve intermediate pipeline stages if omitted
    # Extract raw_response if available from extracted_reasoning
    if raw_response is None and extracted_reasoning is not None:
        raw_response = getattr(extracted_reasoning, "raw_response", None)

    # Resolve structured reasoning
    sr: Optional[StructuredReasoning] = None
    if isinstance(structured_reasoning, StructuredReasoning):
        sr = structured_reasoning
    elif isinstance(structured_reasoning, str):
        sr = structure_reasoning_steps(structured_reasoning)
    elif extracted_reasoning is not None:
        sr = structure_reasoning_steps(extracted_reasoning)
    elif raw_response is not None and raw_response.strip():
        extracted = extract_reasoning_steps(raw_response)
        sr = structure_reasoning_steps(extracted)

    # Resolve verified reasoning
    vr: Optional[VerifiedReasoning] = None
    if isinstance(verified_reasoning, VerifiedReasoning):
        vr = verified_reasoning
    elif sr is not None and sr.steps:
        vr = verify_structured_reasoning(sr)

    # Resolve transition verification
    tr: Optional[TransitionVerificationResult] = None
    if isinstance(transition_verification, TransitionVerificationResult):
        tr = transition_verification
    elif sr is not None and sr.steps:
        tr = verify_step_transitions(sr, vr)

    # Resolve consolidated error report
    er: Optional[ReasoningErrorReport] = None
    if isinstance(error_report, ReasoningErrorReport):
        er = error_report
    elif sr is not None and sr.steps:
        er = detect_reasoning_errors(sr, vr, tr)

    # 3. Build Section A: Problem Information
    final_answer: Optional[str] = None
    if sr is not None and sr.final_answer:
        final_answer = sr.final_answer
    elif extracted_reasoning is not None and extracted_reasoning.final_answer:
        final_answer = extracted_reasoning.final_answer
    elif vr is not None and vr.final_answer:
        final_answer = vr.final_answer

    problem_info = ProblemInfo(
        problem=cleaned_problem,
        final_answer=final_answer,
    )

    # 4. Build Section B & C: Steps and Expression Verification Records
    step_records: List[StepRecord] = []
    # Index verified expressions by (step_number, raw_text) for easy lookup
    v_expr_map: Dict[Tuple[int, str], VerifiedExpression] = {}
    if vr is not None:
        for v_step in vr.steps:
            for v_expr in v_step.expressions:
                v_expr_map[(v_step.step_number, v_expr.raw_text)] = v_expr

    source_steps = sr.steps if (sr and sr.steps) else (vr.steps if (vr and vr.steps) else [])

    for s in source_steps:
        step_num = s.step_number
        step_hdr = getattr(s, "step_header", f"Step {step_num}")
        orig_txt = getattr(s, "original_text", "")

        expr_records: List[ExpressionRecord] = []
        for e in s.expressions:
            # Find matching verified expression if available
            v_match = v_expr_map.get((step_num, e.raw_text), None)

            # Extract solutions/calculated value from verified expression safely
            v_solutions: List[str] = []
            if v_match:
                if getattr(v_match, "calculated_value", None):
                    v_solutions = [str(v_match.calculated_value)]
                elif getattr(v_match, "solutions", None):
                    v_solutions = list(v_match.solutions)
            elif getattr(e, "solutions", None):
                v_solutions = list(e.solutions)

            v_restrictions = list(getattr(v_match, "domain_restrictions", [])) if v_match else list(getattr(e, "domain_restrictions", []))
            v_lhs_val = getattr(v_match, "lhs_value", None) if v_match else getattr(e, "lhs_value", None)
            v_rhs_val = getattr(v_match, "rhs_value", None) if v_match else getattr(e, "rhs_value", None)
            v_status = getattr(v_match, "verification_status", None) if v_match else getattr(e, "verification_status", None)
            v_expl = getattr(v_match, "explanation", None) if v_match else getattr(e, "explanation", None)

            expr_records.append(
                ExpressionRecord(
                    step_number=step_num,
                    raw_text=e.raw_text,
                    expression_type=getattr(e, "expression_type", "equation"),
                    lhs=getattr(e, "lhs", None),
                    rhs=getattr(e, "rhs", None),
                    normalized_lhs=getattr(e, "normalized_lhs", None),
                    normalized_rhs=getattr(e, "normalized_rhs", None),
                    label=getattr(e, "label", None),
                    verification_status=v_status,
                    solver_explanation=v_expl,
                    solutions=v_solutions,
                    domain_restrictions=v_restrictions,
                    lhs_value=v_lhs_val,
                    rhs_value=v_rhs_val,
                )
            )

        step_records.append(
            StepRecord(
                step_number=step_num,
                step_header=step_hdr,
                original_text=orig_txt,
                expressions=expr_records,
            )
        )

    # 5. Build Section D: Transition Verification Records
    transition_records: List[TransitionRecord] = []
    if tr is not None and tr.transitions:
        for t in tr.transitions:
            is_val = getattr(t, "is_valid", t.transition_status == "valid")
            transition_records.append(
                TransitionRecord(
                    prev_step_number=t.prev_step_number,
                    prev_expression=t.prev_expression,
                    curr_step_number=t.curr_step_number,
                    curr_expression=t.curr_expression,
                    is_valid=is_val,
                    transition_status=t.transition_status,
                    transformation_type=t.transformation_type,
                    explanation=t.explanation,
                    conditions=list(t.conditions),
                )
            )

    # 6. Build Section E: Detected Errors, Warnings, and Unsupported Issues
    detected_errors: List[DetectedIssueRecord] = []
    detected_warnings: List[DetectedIssueRecord] = []
    unsupported_issues: List[DetectedIssueRecord] = []

    if er is not None and er.step_reports:
        for s_rep in er.step_reports:
            for issue in s_rep.issues:
                rec = DetectedIssueRecord(
                    step_number=issue.step_number,
                    expression=issue.expression,
                    error_category=issue.error_category,
                    severity=issue.severity,
                    is_definite_error=issue.is_definite_error,
                    explanation=issue.explanation,
                    conditions=list(issue.conditions),
                )
                if issue.is_definite_error:
                    detected_errors.append(rec)
                elif issue.severity == "potential_issue":
                    detected_warnings.append(rec)
                elif issue.severity == "unsupported":
                    unsupported_issues.append(rec)

    # 7. Build Section F: LLM Explanations
    llm_explanations: List[LLMExplanationRecord] = []
    if feedback_result is not None and feedback_result.feedbacks:
        for fb in feedback_result.feedbacks:
            llm_explanations.append(
                LLMExplanationRecord(
                    step_number=fb.step_number,
                    expression=fb.expression,
                    error_category=fb.error_category,
                    severity=fb.severity,
                    solver_evidence=fb.solver_evidence,
                    llm_explanation=fb.llm_explanation,
                    status=fb.status,
                    conditions=list(fb.conditions),
                )
            )

    # 8. Build Section G: Overall Verification Summary & Metrics
    total_steps = len(step_records)
    total_expressions = sum(len(s.expressions) for s in step_records)

    total_verified_expressions = 0
    total_incorrect_expressions = 0
    total_invalid_expressions = 0
    total_unsupported_expressions = 0

    for s in step_records:
        for e in s.expressions:
            if e.verification_status == "verified":
                total_verified_expressions += 1
            elif e.verification_status == "incorrect":
                total_incorrect_expressions += 1
            elif e.verification_status == "invalid":
                total_invalid_expressions += 1
            elif e.verification_status == "unsupported":
                total_unsupported_expressions += 1

    total_detected_errors = er.total_definite_errors if er else len(detected_errors)
    total_warnings = er.total_potential_issues if er else len(detected_warnings)

    # Status Precedence Logic:
    # 1. verification_failed: if no steps or expressions could be parsed/extracted, or pipeline produced unusable results.
    # 2. errors_detected: if one or more definite mathematical errors or invalid transformations exist.
    # 3. warnings: if no definite errors, but one or more conditional warnings exist.
    # 4. partially_verified: if no definite errors and no warnings, but some expressions could not be verified (unsupported).
    # 5. verified: all relevant mathematical statements and transitions were successfully verified with zero errors, warnings, or unsupported expressions.
    if total_steps == 0 or total_expressions == 0:
        overall_status = "verification_failed"
        summary_text = "Verification failed: no reasoning steps or mathematical expressions could be parsed."
    elif total_detected_errors > 0:
        overall_status = "errors_detected"
        summary_text = (
            f"Verification detected {total_detected_errors} definite mathematical error(s) "
            f"across {total_steps} reasoning step(s)."
        )
    elif total_warnings > 0:
        overall_status = "warnings"
        summary_text = (
            f"Verification completed with {total_warnings} conditional warning(s) "
            f"(no definite errors detected)."
        )
    elif total_unsupported_expressions > 0:
        overall_status = "partially_verified"
        summary_text = (
            f"Verification partially completed: {total_unsupported_expressions} expression(s) "
            "contain natural language or non-symbolic syntax and were skipped without being marked incorrect."
        )
    else:
        overall_status = "verified"
        summary_text = (
            f"All {total_expressions} mathematical expression(s) across {total_steps} step(s) "
            "were successfully verified with zero errors or warnings."
        )

    summary = VerificationSummary(
        total_steps=total_steps,
        total_expressions=total_expressions,
        total_verified_expressions=total_verified_expressions,
        total_incorrect_expressions=total_incorrect_expressions,
        total_invalid_expressions=total_invalid_expressions,
        total_unsupported_expressions=total_unsupported_expressions,
        total_detected_errors=total_detected_errors,
        total_warnings=total_warnings,
        overall_status=overall_status,
        summary_text=summary_text,
    )

    return VerificationReport(
        problem_info=problem_info,
        steps=step_records,
        transitions=transition_records,
        detected_errors=detected_errors,
        detected_warnings=detected_warnings,
        unsupported_issues=unsupported_issues,
        llm_explanations=llm_explanations,
        summary=summary,
        raw_response=raw_response,
    )


# ==============================================================================
# Demonstrations for Objective 25
# ==============================================================================
def run_demonstrations():
    """
    Executes the 6 required demonstrations for Objective 25:
    1. Correct arithmetic reasoning.
    2. Incorrect arithmetic assertion.
    3. Invalid algebraic transformation.
    4. Lost solution through variable division.
    5. Conditional warning after squaring both sides.
    6. Reasoning containing unsupported expressions.
    """
    print("=" * 80)
    print("PHASE 3, OBJECTIVE 25: FINAL MATHEMATICAL REASONING VERIFICATION REPORT")
    print("=" * 80)

    demos = [
        (
            "Demo 1: Completely Correct Reasoning",
            "A shopkeeper buys 3 notebooks for ₹40 each and 2 pens for ₹15 each. What is the total cost?",
            (
                "**Step 1: Calculate notebook cost**\n"
                "Cost of notebooks = 3 * 40\n"
                "Cost of notebooks = 120\n\n"
                "**Step 2: Calculate pen cost**\n"
                "Cost of pens = 2 * 15\n"
                "Cost of pens = 30\n\n"
                "**Step 3: Calculate total cost**\n"
                "Total cost = 120 + 30\n"
                "Total cost = 150\n\n"
                "Final Answer: 150"
            ),
            None,
        ),
        (
            "Demo 2: Incorrect Arithmetic Assertion",
            "A shopkeeper buys 3 notebooks for ₹40 each. What is the total cost?",
            (
                "**Step 1: Calculate total cost**\n"
                "Cost of notebooks = 3 * 40\n"
                "3 * 40 = 130\n\n"
                "Final Answer: 130"
            ),
            ReasoningFeedbackResult(
                problem="A shopkeeper buys 3 notebooks for ₹40 each. What is the total cost?",
                feedbacks=[
                    StepFeedback(
                        step_number=1,
                        expression="3 * 40 = 130",
                        error_category="incorrect_assertion",
                        severity="definite_error",
                        solver_evidence="Mathematical assertion is incorrect: LHS (3 * 40 = 120) does not equal RHS (130 = 130).",
                        llm_explanation=(
                            "The step contains an arithmetic calculation error. The product of 3 and 40 is 120, "
                            "not 130. To correct it, replace 130 with 120."
                        ),
                        status="success",
                    )
                ],
                total_feedbacks=1,
                has_issues=True,
                status="success",
            ),
        ),
        (
            "Demo 3: Invalid Algebraic Transformation",
            "Solve the equation 2*x + 4 = 10 for x.",
            (
                "**Step 1: Subtract 4 from both sides**\n"
                "2*x + 4 = 10\n"
                "2*x = 14\n\n"
                "**Step 2: Divide by 2**\n"
                "x = 7\n\n"
                "Final Answer: 7"
            ),
            ReasoningFeedbackResult(
                problem="Solve the equation 2*x + 4 = 10 for x.",
                feedbacks=[
                    StepFeedback(
                        step_number=1,
                        expression="2*x = 14",
                        error_category="invalid_transformation",
                        severity="definite_error",
                        solver_evidence="Invalid algebraic transformation: original equation has solution(s) ['3'], but transformed equation has solution(s) ['7'].",
                        llm_explanation=(
                            "To isolate 2*x from 2*x + 4 = 10, 4 must be subtracted from both sides (10 - 4 = 6). "
                            "Instead, 4 was incorrectly added to the right side (10 + 4 = 14)."
                        ),
                        status="success",
                    )
                ],
                total_feedbacks=1,
                has_issues=True,
                status="success",
            ),
        ),
        (
            "Demo 4: Lost Solution through Variable Division",
            "Find all real solutions to x**2 = 2*x.",
            (
                "**Step 1: Divide both sides by x**\n"
                "x**2 = 2*x\n"
                "x = 2\n\n"
                "Final Answer: 2"
            ),
            ReasoningFeedbackResult(
                problem="Find all real solutions to x**2 = 2*x.",
                feedbacks=[
                    StepFeedback(
                        step_number=1,
                        expression="x = 2",
                        error_category="lost_solution",
                        severity="definite_error",
                        solver_evidence="Invalid algebraic transformation: divided both sides by variable expression 'x', losing solution(s) ['0'] and risking division by zero if x = 0.",
                        llm_explanation=(
                            "Dividing both sides by the variable x eliminates the valid root x = 0. "
                            "The equation should be factored as x*(x - 2) = 0 to yield both x = 0 and x = 2."
                        ),
                        status="success",
                    )
                ],
                total_feedbacks=1,
                has_issues=True,
                status="success",
            ),
        ),
        (
            "Demo 5: Conditional Warning after Squaring Both Sides",
            "Solve the equation x = 3.",
            (
                "**Step 1: Square both sides**\n"
                "x = 3\n"
                "x**2 = 9\n\n"
                "Final Answer: 3"
            ),
            ReasoningFeedbackResult(
                problem="Solve the equation x = 3.",
                feedbacks=[
                    StepFeedback(
                        step_number=1,
                        expression="x**2 = 9",
                        error_category="conditional_warning",
                        severity="potential_issue",
                        solver_evidence="Valid algebraic transformation: squared both sides of the equation.",
                        llm_explanation=(
                            "Squaring both sides is a one-way implication that introduces extraneous roots (x = -3). "
                            "All candidate solutions must be verified in the original equation."
                        ),
                        conditions=["Roots must be checked in original equation."],
                        status="success",
                    )
                ],
                total_feedbacks=1,
                has_issues=True,
                status="success",
            ),
        ),
        (
            "Demo 6: Reasoning Containing Unsupported Expressions",
            "A physics problem describes the relationship between distance, speed, and time.",
            (
                "**Step 1: State the physical relation**\n"
                "Distance = Speed * Time\n\n"
                "Final Answer: Distance = Speed * Time"
            ),
            None,
        ),
    ]

    for title, prob, reasoning_text, feedback_obj in demos:
        print("\n" + "=" * 80)
        print(f"SCENARIO: {title}")
        print("=" * 80)

        report = generate_verification_report(
            problem=prob,
            raw_response=reasoning_text,
            feedback_result=feedback_obj,
        )

        md_output = format_markdown_report(report)
        print(md_output)
        print("-" * 80)


if __name__ == "__main__":
    run_demonstrations()
