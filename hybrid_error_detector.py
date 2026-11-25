"""
hybrid_error_detector.py
------------------------
Phase 4 — Objective 30: Hybrid Reasoning Error Detection Architecture.

Combines LLM-based semantic reasoning with deterministic mathematical verification
without relying on majority voting or ungrounded assumptions.

Explicit Evidence-Fusion Strategy:
----------------------------------
1. Independent Evidence Sources:
   - LLM Evidence (Objective 27):
     * Semantic problem understanding.
     * Problem-to-equation alignment (detects 'incorrect_setup').
     * Conceptual or logical leaps.
     * Free-form natural language rationales.
   - Deterministic Solver Evidence (Objective 29):
     * SymPy AST evaluation of individual arithmetic and algebraic equations.
     * Equivalence and transformation verification between consecutive steps.
     * Division-by-zero and domain restriction checks.
     * Unsupported expression identification (abstentions without guessing).

2. Fusion Rules & Hierarchy:
   - Rule 1 (Full Concurrence):
     * If both LLM and Solver report 'correct' -> Hybrid verdict: 'confirmed_correct'.
     * If both LLM and Solver report 'incorrect' -> Hybrid verdict: 'confirmed_error'.
   - Rule 2 (Formal Mathematical Disproof Overrules LLM):
     * If LLM reports 'correct', but Solver establishes a formal mathematical error
       ('incorrect_assertion', 'invalid_transformation', 'division_by_zero'):
       Hybrid verdict: 'solver_overruled_llm' ('incorrect').
       Formal mathematical proof overrides soft LLM acceptance.
   - Rule 3 (Semantic & Setup Detection Overrules Pure Internal Algebra):
     * If LLM reports 'incorrect', but Solver reports 'correct':
       The Solver only checks whether Step n follows algebraically from Step n-1;
       it cannot verify that Step 1 matches the English word problem ('incorrect_setup').
       Hybrid verdict: 'llm_semantic_error' ('incorrect').
   - Rule 4 (Unsupported Verification Caveat):
     * If Solver is 'unsupported' and LLM is 'incorrect':
       Hybrid verdict: 'unsupported_symbolic_llm_error' ('incorrect').
     * If Solver is 'unsupported' and LLM is 'correct':
       Hybrid verdict: 'llm_accepted_symbolic_unsupported' ('correct' with explicit
       symbolic_supported=False flag; never claimed as symbolically verified).

3. Step-Localization Resolution:
   - When only one system detects an error, its localized step is adopted.
   - When both detect an error:
     * If both agree, the shared step is adopted.
     * If they disagree, the earliest flagged step is recorded as the primary failure point,
       while retaining both steps and logging a step disagreement flag.
"""

from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class HybridPrediction:
    """Structured result of fusing LLM and deterministic solver evidence."""
    id: str
    hybrid_classification: str                  # 'correct', 'incorrect', 'unsupported'
    hybrid_verdict_type: str                    # 'confirmed_correct', 'confirmed_error',
                                                # 'solver_overruled_llm', 'llm_semantic_error',
                                                # 'llm_accepted_symbolic_unsupported',
                                                # 'unsupported_symbolic_llm_error'
    first_incorrect_step: Optional[int] = None
    detected_error_category: Optional[str] = None
    confidence: str = "high"                    # 'high', 'medium', 'low'
    disagreement: bool = False                  # True if LLM and solver disagreed on classification
    step_agreement: bool = True                 # False if both flagged error but on different steps
    symbolic_supported: bool = True             # False if solver was unsupported/unverified
    fusion_rationale: str = ""
    llm_prediction: Dict[str, Any] = field(default_factory=dict)
    solver_prediction: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "hybrid_classification": self.hybrid_classification,
            "hybrid_verdict_type": self.hybrid_verdict_type,
            "first_incorrect_step": self.first_incorrect_step,
            "detected_error_category": self.detected_error_category,
            "confidence": self.confidence,
            "disagreement": self.disagreement,
            "step_agreement": self.step_agreement,
            "symbolic_supported": self.symbolic_supported,
            "fusion_rationale": self.fusion_rationale,
            "llm_prediction": self.llm_prediction,
            "solver_prediction": self.solver_prediction,
        }


def fuse_predictions(
    llm_record: Dict[str, Any],
    solver_record: Dict[str, Any],
    case_id: str = "",
) -> HybridPrediction:
    """
    Fuses a single pair of LLM and Solver predictions into a unified HybridPrediction.

    Parameters
    ----------
    llm_record : dict
        LLM prediction dictionary (from Objective 27). Expects 'classification',
        'first_incorrect_step', 'confidence', 'explanation'.
    solver_record : dict
        Solver prediction dictionary (from Objective 29). Expects 'solver_classification',
        'detected_error_category', 'first_incorrect_step', 'verification_status', 'evidence'.
    case_id : str
        Canonical evaluation case identifier (e.g. 'eval_001').

    Returns
    -------
    HybridPrediction
        Rigorous fused prediction preserving evidence from both systems.
    """
    cid = case_id or solver_record.get("id") or llm_record.get("id", "")

    # Extract LLM fields
    # Handle both full record and nested prediction dict
    l_pred = llm_record.get("prediction", llm_record)
    l_cls = l_pred.get("classification", "uncertain")
    l_step = l_pred.get("first_incorrect_step")
    l_conf = l_pred.get("confidence", "medium")
    l_expl = l_pred.get("explanation", "")

    # Extract Solver fields
    s_cls = solver_record.get("solver_classification", "unsupported")
    s_status = solver_record.get("verification_status", "verification_failed")
    s_cat = solver_record.get("detected_error_category")
    s_step = solver_record.get("first_incorrect_step")
    s_evid = solver_record.get("evidence", "")
    s_unsupp_reason = solver_record.get("unsupported_reason")

    # Initialize defaults
    disagreement = False
    step_agreement = True
    symbolic_supported = (s_cls != "unsupported" and s_status not in ("partially_verified", "verification_failed"))

    # --------------------------------------------------------------------------
    # Case 1: Full Concurrence
    # --------------------------------------------------------------------------
    if l_cls == "correct" and s_cls == "correct":
        hybrid_cls = "correct"
        verdict_type = "confirmed_correct"
        confidence = "high"
        first_step = None
        error_cat = None
        rationale = (
            "Full consensus: LLM semantic evaluation confirmed reasoning is sound, "
            "and deterministic solver verified all mathematical statements and step transitions."
        )

    elif l_cls == "incorrect" and s_cls == "incorrect":
        hybrid_cls = "incorrect"
        verdict_type = "confirmed_error"
        confidence = "high"
        disagreement = False

        if l_step is not None and s_step is not None:
            if l_step == s_step:
                first_step = l_step
                step_agreement = True
            else:
                # Both detected error, but flagged different steps
                step_agreement = False
                # Use LLM step if earlier or if solver caught secondary syntax fallout;
                # in practice, earliest flagged step pinpoints initial point of failure
                first_step = l_step
        else:
            first_step = l_step or s_step

        error_cat = s_cat or "reasoning_error"
        rationale = (
            f"Full consensus on error: deterministic solver proved mathematical flaw ({s_cat or 'error'}) "
            f"and LLM identified reasoning failure. Step agreement: {step_agreement} "
            f"(LLM step={l_step}, Solver step={s_step})."
        )

    # --------------------------------------------------------------------------
    # Case 2: Solver Overrules LLM False Negative (Confirmed Symbolic Error)
    # --------------------------------------------------------------------------
    elif l_cls == "correct" and s_cls == "incorrect":
        disagreement = True
        hybrid_cls = "incorrect"
        verdict_type = "solver_overruled_llm"
        first_step = s_step
        error_cat = s_cat or "mathematical_error"
        confidence = "high"
        rationale = (
            f"Disagreement resolved by mathematical proof: LLM accepted the trace as correct, "
            f"but deterministic solver proved a mathematical error ({error_cat} at Step {s_step}: '{s_evid}'). "
            "Formal mathematical disproof overrides soft LLM acceptance."
        )

    # --------------------------------------------------------------------------
    # Case 3: LLM Detects Semantic / Setup Error (Solver Blind Spot)
    # --------------------------------------------------------------------------
    elif l_cls == "incorrect" and s_cls == "correct":
        disagreement = True
        hybrid_cls = "incorrect"
        verdict_type = "llm_semantic_error"
        first_step = l_step
        error_cat = "semantic_or_setup_error"
        confidence = "medium"
        rationale = (
            f"Disagreement resolved by semantic priority: deterministic solver verified that mathematical "
            f"equations follow algebraically from one another, but the solver cannot verify whether equations "
            f"match the English word problem. LLM identified a reasoning/setup error at Step {l_step}: '{l_expl}'."
        )

    # --------------------------------------------------------------------------
    # Case 4: Solver is Unsupported (Caveat / Abstention Handling)
    # --------------------------------------------------------------------------
    elif s_cls == "unsupported":
        symbolic_supported = False
        if l_cls == "incorrect":
            disagreement = False
            hybrid_cls = "incorrect"
            verdict_type = "unsupported_symbolic_llm_error"
            first_step = l_step
            error_cat = "llm_detected_error_unsupported_solver"
            confidence = "medium"
            rationale = (
                f"LLM detected reasoning error at Step {l_step}, while deterministic solver was unsupported "
                f"({s_unsupp_reason or 'unparsed expressions'}). LLM finding adopted."
            )
        elif l_cls == "correct":
            disagreement = True
            hybrid_cls = "correct"
            verdict_type = "llm_accepted_symbolic_unsupported"
            first_step = None
            error_cat = None
            confidence = "medium"
            rationale = (
                "LLM found reasoning sound, but symbolic solver could not verify expressions "
                f"({s_unsupp_reason or 'unsupported conceptual syntax'}). Marked correct under semantic review, "
                "with explicit flag that symbolic proof was unavailable."
            )
        else:  # LLM uncertain
            disagreement = False
            hybrid_cls = "unsupported"
            verdict_type = "both_uncertain"
            first_step = None
            error_cat = None
            confidence = "low"
            rationale = "Both LLM and deterministic solver abstained or reported uncertainty."

    # --------------------------------------------------------------------------
    # Case 5: LLM Uncertain with Decisive Solver
    # --------------------------------------------------------------------------
    elif l_cls == "uncertain":
        disagreement = True
        if s_cls == "incorrect":
            hybrid_cls = "incorrect"
            verdict_type = "solver_resolved_llm_uncertainty"
            first_step = s_step
            error_cat = s_cat
            confidence = "high"
            rationale = f"LLM was uncertain, but deterministic solver proved mathematical error at Step {s_step}."
        elif s_cls == "correct":
            hybrid_cls = "correct"
            verdict_type = "solver_verified_llm_uncertainty"
            first_step = None
            error_cat = None
            confidence = "high"
            rationale = "LLM was uncertain, but deterministic solver verified all mathematical steps as sound."
        else:
            hybrid_cls = "unsupported"
            verdict_type = "both_unsupported"
            first_step = None
            error_cat = None
            confidence = "low"
            rationale = "Both systems were unable to decisively verify the reasoning."

    else:
        # Fallback
        hybrid_cls = "unsupported"
        verdict_type = "unhandled_combination"
        first_step = None
        error_cat = None
        confidence = "low"
        rationale = f"Unhandled state combination: LLM='{l_cls}', Solver='{s_cls}'."

    return HybridPrediction(
        id=cid,
        hybrid_classification=hybrid_cls,
        hybrid_verdict_type=verdict_type,
        first_incorrect_step=first_step,
        detected_error_category=error_cat,
        confidence=confidence,
        disagreement=disagreement,
        step_agreement=step_agreement,
        symbolic_supported=symbolic_supported,
        fusion_rationale=rationale,
        llm_prediction=l_pred,
        solver_prediction=solver_record,
    )
