"""
llm_improved_detector.py
------------------------
Phase 4 - Objective 28: Improved LLM-Based Mathematical Reasoning Error Detection.

Implements an enhanced LLM-based error detector utilizing a systematic, 9-stage
mathematical verification prompt. Evaluates proposed mathematical reasoning traces
from first principles using only the problem statement and the reasoning steps.

Strict Information Barrier:
- The LLM receives ONLY the problem statement and the proposed reasoning trace.
- Gold labels, reference answers, error categories, error steps, expected report statuses,
  and metadata are NEVER exposed to the prompt or the model.
- Zero reliance on SymPy, symbolic solvers, transition verifiers, or Phase 3 verification outputs.
"""

from dataclasses import dataclass, field
import json
import os
import re
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_llm_client, call_llm


IMPROVED_DETECTION_PROMPT_TEMPLATE = (
    "You are an expert mathematical reasoning auditor. Your task is to rigorously and systematically verify the proposed mathematical reasoning trace for a given problem.\n\n"
    "Problem:\n"
    "{problem}\n\n"
    "Proposed Reasoning Trace:\n"
    "{reasoning_text}\n\n"
    "Systematic Verification Procedure:\n"
    "Perform these checks internally before producing your final classification:\n"
    "1. Problem Understanding & Constraints: Understand the original problem and identify all explicit or implicit mathematical constraints (domains, integer/real restrictions, non-negativity, variable definitions, and units).\n"
    "2. Sequential Step-by-Step Audit: Read and audit the reasoning trace in its exact sequential order, step by step from first principles.\n"
    "3. Numerical & Arithmetic Verification: Independently compute and check every arithmetic calculation (addition, subtraction, multiplication, division, exponentiation, modular arithmetic).\n"
    "4. Equivalence-Preserving Transformations: Check whether each algebraic transformation strictly preserves mathematical equivalence (e.g. valid operations applied equally to both sides, no sign errors, no dropped terms, no illegal distribution or cancellation).\n"
    "5. Branch Analysis & Solution Preservation: Check division, multiplication, squaring, radical elimination, cancellation, and factoring for:\n"
    "   - Lost solutions (e.g., dividing both sides by an expression or variable that can be zero).\n"
    "   - Extraneous / invalid solutions introduced (e.g., squaring both sides of radical equations without checking validity).\n"
    "6. Domain Restrictions & Mathematical Boundaries: Check for division by zero, negative radicands under real square roots, base/argument domain violations in logarithms, and undefined mathematical operations.\n"
    "7. Logical & Conceptual Validity: Check the logical validity of the proof or solution flow:\n"
    "   - Do not assume that an equation being symbolically or algebraically valid proves that the reasoning is logically valid.\n"
    "   - Check for circular reasoning (assuming the conclusion, begging the question).\n"
    "   - Check for flawed setup (equations that do not match the problem conditions or assign inconsistent values).\n"
    "   - Check for unsupported assertions, missing branch cases, or invalid heuristic shortcuts (e.g. digit-cancellation fallacy).\n"
    "8. Localization of First Failure: Identify the exact 1-based integer number of the FIRST step where the reasoning fails, if an error exists. Subsequent steps after the first failure must not override identifying the initial break in validity.\n"
    "9. Classification Consistency: Ensure the final JSON classification strictly agrees with the explanation and identified error:\n"
    '   - "correct": Every mathematical assertion and transition is logically and algebraically sound, and the final answer is justified.\n'
    '   - "incorrect": The reasoning contains a definite mathematical, algebraic, arithmetic, or logical error.\n'
    '   - "uncertain": The reasoning contains ambiguous notation, insufficient information, or you cannot conclusively determine correctness.\n\n'
    "Output Format:\n"
    "Respond ONLY with a valid JSON object matching this exact schema (no markdown formatting, no surrounding commentary):\n"
    "{{\n"
    '  "classification": "correct" | "incorrect" | "uncertain",\n'
    '  "first_incorrect_step": <positive integer 1, 2, ... or null>,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "explanation": "<1-3 sentences concisely explaining your finding, explicitly identifying why the first incorrect step is invalid or confirming soundness>"\n'
    "}}"
)

ALLOWED_CLASSIFICATIONS = {"correct", "incorrect", "uncertain"}
ALLOWED_CONFIDENCES = {"high", "medium", "low"}


@dataclass
class LLMImprovedErrorDetection:
    """Structured result of an improved LLM mathematical reasoning error detection call."""
    classification: str                    # 'correct', 'incorrect', 'uncertain'
    first_incorrect_step: Optional[int]    # 1-based integer step number, or None
    confidence: str                        # 'high', 'medium', 'low'
    explanation: str                       # Brief natural-language explanation
    raw_response: str = ""                 # Unparsed LLM textual response
    latency_seconds: float = 0.0           # Inference time in seconds
    parsed_successfully: bool = True       # Whether JSON schema parsing succeeded
    provider: str = ""                     # 'gemini' or 'groq'
    model: str = ""                        # Model name used for inference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "first_incorrect_step": self.first_incorrect_step,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "raw_response": self.raw_response,
            "latency_seconds": round(self.latency_seconds, 3),
            "parsed_successfully": self.parsed_successfully,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMImprovedErrorDetection":
        return cls(
            classification=data.get("classification", "uncertain"),
            first_incorrect_step=data.get("first_incorrect_step"),
            confidence=data.get("confidence", "low"),
            explanation=data.get("explanation", ""),
            raw_response=data.get("raw_response", ""),
            latency_seconds=data.get("latency_seconds", 0.0),
            parsed_successfully=data.get("parsed_successfully", True),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
        )


def build_improved_detection_prompt(problem: str, reasoning_text: str) -> str:
    """
    Constructs the improved prompt sent to the LLM for systematic mathematical error detection.

    Strict Information Barrier:
    Only `problem` and `reasoning_text` are accepted. No gold labels, reference answers,
    error categories, error steps, expected statuses, or metadata may enter this function.

    Parameters
    ----------
    problem : str
        The mathematical problem text.
    reasoning_text : str
        The reasoning trace to verify.

    Returns
    -------
    str
        The formatted prompt text incorporating the 9-stage verification procedure.
    """
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Problem must be a non-empty string.")
    if not isinstance(reasoning_text, str) or not reasoning_text.strip():
        raise ValueError("Reasoning text must be a non-empty string.")

    return IMPROVED_DETECTION_PROMPT_TEMPLATE.format(
        problem=problem.strip(),
        reasoning_text=reasoning_text.strip(),
    )


def parse_llm_detection_response(raw_text: str) -> Dict[str, Any]:
    """
    Parses and sanitizes the raw JSON response from the LLM with strict validation.

    Handles:
    - Direct JSON string output.
    - Markdown code fences (e.g. ```json ... ```).
    - Loose whitespace and formatting variations.
    - Graceful fallback to classification="uncertain" and parsed_successfully=False
      if JSON is malformed, fields are missing, or types are invalid.

    Strict Validation Rules:
    - Classification must exactly match "correct", "incorrect", or "uncertain".
    - If classification is missing or invalid -> classification="uncertain", parsed_successfully=False.
    - If classification == "incorrect", first_incorrect_step must be a positive integer (>= 1).
      Booleans are rejected. If step is missing or invalid, parsed_successfully=False.
    - If classification is "correct" or "uncertain", first_incorrect_step is normalized to None.

    Returns
    -------
    dict
        Parsed dictionary with keys: classification, first_incorrect_step, confidence, explanation, parsed_successfully.
    """
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "classification": "uncertain",
            "first_incorrect_step": None,
            "confidence": "low",
            "explanation": "Empty or non-string response from LLM.",
            "parsed_successfully": False,
        }

    cleaned = raw_text.strip()

    # Strip markdown code blocks if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    data = None
    # 1. Attempt direct JSON parsing
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # 2. Attempt extracting innermost JSON block via regex
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        return {
            "classification": "uncertain",
            "first_incorrect_step": None,
            "confidence": "low",
            "explanation": f"Failed to parse valid JSON from response: {cleaned[:100]}...",
            "parsed_successfully": False,
        }

    # Strict Validation:
    parsed_successfully = True

    # 1. Classification validation (exact match only; no substring fallback)
    raw_class = str(data.get("classification", "")).strip().lower()
    if raw_class in ALLOWED_CLASSIFICATIONS:
        classification = raw_class
    else:
        classification = "uncertain"
        parsed_successfully = False

    # 2. First incorrect step validation
    raw_step = data.get("first_incorrect_step")
    first_incorrect_step = None

    if classification == "incorrect":
        if raw_step is None:
            first_incorrect_step = None
            parsed_successfully = False
        else:
            # Reject booleans (isinstance(True, int) is True in Python)
            if isinstance(raw_step, bool):
                first_incorrect_step = None
                parsed_successfully = False
            elif isinstance(raw_step, int):
                if raw_step >= 1:
                    first_incorrect_step = raw_step
                else:
                    first_incorrect_step = None
                    parsed_successfully = False
            elif isinstance(raw_step, str):
                step_str = raw_step.strip()
                step_match = re.search(r"^(?:step\s*)?(\d+)$", step_str, re.IGNORECASE)
                if step_match:
                    val = int(step_match.group(1))
                    if val >= 1:
                        first_incorrect_step = val
                    else:
                        first_incorrect_step = None
                        parsed_successfully = False
                else:
                    first_incorrect_step = None
                    parsed_successfully = False
            else:
                first_incorrect_step = None
                parsed_successfully = False
    else:
        # For correct or uncertain classifications, normalize the step to None
        first_incorrect_step = None

    # 3. Confidence validation
    raw_conf = str(data.get("confidence", "")).strip().lower()
    confidence = raw_conf if raw_conf in ALLOWED_CONFIDENCES else "medium"

    # 4. Explanation extraction
    explanation = str(data.get("explanation", "")).strip()
    if not explanation:
        explanation = f"Model classified reasoning as {classification}."

    return {
        "classification": classification,
        "first_incorrect_step": first_incorrect_step,
        "confidence": confidence,
        "explanation": explanation,
        "parsed_successfully": parsed_successfully,
    }


def detect_reasoning_error_improved(
    problem: str,
    reasoning_text: str,
    provider: str = "gemini",
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    max_retries: int = 4,
) -> LLMImprovedErrorDetection:
    """
    Evaluates a single mathematical problem and reasoning trace using the improved prompt.

    Strict Information Barrier:
    Only `problem` and `reasoning_text` are accepted.

    Parameters
    ----------
    problem : str
        The mathematical problem text.
    reasoning_text : str
        The reasoning trace to verify.
    provider : str
        'gemini' or 'groq'.
    client : Any, optional
        Pre-initialized LLM client (for dependency injection or testing).
    model_name : str, optional
        Model override (for testing).
    max_retries : int, default 4
        Maximum automatic retries for transient 503 / UNAVAILABLE errors.
        Set to 0 in interactive mode to avoid automatic retry loops.

    Returns
    -------
    LLMImprovedErrorDetection
        The parsed structured detection result.
    """
    prompt = build_improved_detection_prompt(problem, reasoning_text)

    # Initialize client if not injected
    if client is None or model_name is None:
        init_client, init_model = get_llm_client(provider)
        active_client = client if client is not None else init_client
        active_model = model_name if model_name is not None else init_model
    else:
        active_client = client
        active_model = model_name

    start_time = time.time()
    raw_response = ""
    for attempt in range(max_retries + 1):
        try:
            raw_response = call_llm(active_client, provider, active_model, prompt)
            break
        except Exception as e:
            err_str = str(e)
            is_transient_503 = (
                "503" in err_str
                or "UNAVAILABLE" in err_str
                or "high demand" in err_str.lower()
                or "overloaded" in err_str.lower()
            )
            if is_transient_503 and attempt < max_retries:
                wait_time = (attempt + 1) * 8.0
                print(f"    [Transient 503 UNAVAILABLE] Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue

            latency = time.time() - start_time
            return LLMImprovedErrorDetection(
                classification="uncertain",
                first_incorrect_step=None,
                confidence="low",
                explanation=f"LLM inference error: {e}",
                raw_response="",
                latency_seconds=latency,
                parsed_successfully=False,
                provider=provider,
                model=active_model,
            )

    latency = time.time() - start_time
    parsed = parse_llm_detection_response(raw_response)
    return LLMImprovedErrorDetection(
        classification=parsed["classification"],
        first_incorrect_step=parsed["first_incorrect_step"],
        confidence=parsed["confidence"],
        explanation=parsed["explanation"],
        raw_response=raw_response,
        latency_seconds=latency,
        parsed_successfully=parsed["parsed_successfully"],
        provider=provider,
        model=active_model,
    )
