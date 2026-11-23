"""
llm_error_detector.py
---------------------
Phase 4 - Objective 27: LLM-Only Error Detection Baseline.

Implements an independent LLM-only error detector that evaluates mathematical reasoning
traces using only the natural-language problem and the proposed reasoning steps.

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


DETECTION_PROMPT_TEMPLATE = (
    "You are an expert mathematical reasoning auditor. Your task is to rigorously verify the provided mathematical reasoning trace for a given problem.\n\n"
    "Problem:\n"
    "{problem}\n\n"
    "Proposed Reasoning Trace:\n"
    "{reasoning_text}\n\n"
    "Instructions:\n"
    "1. Verify each step in the proposed reasoning trace from first principles.\n"
    "2. Check for:\n"
    "   - Arithmetic errors (calculation mistakes in addition, subtraction, multiplication, division).\n"
    "   - Invalid algebraic transformations (illegal operations, dropped terms, false cancellations, sign errors).\n"
    "   - Lost solutions (dividing by variables without considering zero, incomplete branch analysis).\n"
    "   - Division by zero or domain violations (e.g. negative square root arguments in real numbers).\n"
    "   - Flawed problem setup (equations that do not match the problem prompt or assume the conclusion).\n"
    "3. Classify the reasoning trace into exactly one of three categories:\n"
    "   - \"correct\": Every mathematical assertion and transition is sound, and the final answer is justified.\n"
    "   - \"incorrect\": The reasoning contains a definite mathematical error or invalid transformation.\n"
    "   - \"uncertain\": The reasoning contains ambiguous notation, insufficient information, or you cannot conclusively determine correctness.\n"
    "4. If \"incorrect\", identify the 1-based integer number of the FIRST step where reasoning fails (e.g., 1, 2, 3). If \"correct\" or \"uncertain\", this must be null.\n"
    "5. Provide a concise explanation (1-3 sentences) detailing your finding.\n\n"
    "Output format:\n"
    "Respond ONLY with a valid JSON object matching this exact schema:\n"
    "{{\n"
    '  "classification": "correct" | "incorrect" | "uncertain",\n'
    '  "first_incorrect_step": <integer step number or null>,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "explanation": "<1-3 sentences explaining your assessment>"\n'
    "}}"
)

ALLOWED_CLASSIFICATIONS = {"correct", "incorrect", "uncertain"}
ALLOWED_CONFIDENCES = {"high", "medium", "low"}


@dataclass
class LLMErrorDetection:
    """Structured result of an LLM-only mathematical reasoning error detection call."""
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
    def from_dict(cls, data: Dict[str, Any]) -> "LLMErrorDetection":
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


def build_detection_prompt(problem: str, reasoning_text: str) -> str:
    """
    Constructs the prompt sent to the LLM for reasoning error detection.

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
        The formatted prompt text.
    """
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Problem must be a non-empty string.")
    if not isinstance(reasoning_text, str) or not reasoning_text.strip():
        raise ValueError("Reasoning text must be a non-empty string.")

    return DETECTION_PROMPT_TEMPLATE.format(
        problem=problem.strip(),
        reasoning_text=reasoning_text.strip(),
    )


def parse_llm_detection_response(raw_text: str) -> Dict[str, Any]:
    """
    Parses and sanitizes the raw JSON response from the LLM.

    Handles:
    - Direct JSON string output.
    - Markdown code fences (e.g. ```json ... ```).
    - Loose whitespace and formatting variations.
    - Graceful fallback to classification="uncertain" if JSON is malformed.

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

    # Extract JSON if wrapped in markdown code blocks
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Fallback regex to locate outer JSON object bounds
        brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
        if brace_match:
            cleaned = brace_match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except Exception as e:
        return {
            "classification": "uncertain",
            "first_incorrect_step": None,
            "confidence": "low",
            "explanation": f"Failed to parse JSON response ({e}).",
            "parsed_successfully": False,
        }

    if not isinstance(data, dict):
        return {
            "classification": "uncertain",
            "first_incorrect_step": None,
            "confidence": "low",
            "explanation": "Response JSON is not an object.",
            "parsed_successfully": False,
        }

    parsed_successfully = True

    # 1. Classification validation (exact match only, no substring heuristics)
    if "classification" not in data or data["classification"] is None:
        classification = "uncertain"
        parsed_successfully = False
    else:
        raw_class = str(data["classification"]).strip().lower()
        if raw_class in ALLOWED_CLASSIFICATIONS:
            classification = raw_class
        else:
            classification = "uncertain"
            parsed_successfully = False

    # 2. Validate first_incorrect_step
    raw_step = data.get("first_incorrect_step")
    first_incorrect_step = None

    if classification == "incorrect":
        if raw_step is not None:
            # Reject booleans explicitly (isinstance(True, int) is True in Python)
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
                # Accept exact digits or 'Step <digit>'
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


def detect_reasoning_error_llm(
    problem: str,
    reasoning_text: str,
    provider: str = "gemini",
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> LLMErrorDetection:
    """
    Evaluates a single mathematical problem and reasoning trace using an LLM.

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

    Returns
    -------
    LLMErrorDetection
        The parsed structured detection result.
    """
    prompt = build_detection_prompt(problem, reasoning_text)

    # Initialize client if not injected
    if client is None or model_name is None:
        init_client, init_model = get_llm_client(provider)
        active_client = client if client is not None else init_client
        active_model = model_name if model_name is not None else init_model
    else:
        active_client = client
        active_model = model_name

    start_time = time.time()
    try:
        raw_response = call_llm(active_client, provider, active_model, prompt)
        latency = time.time() - start_time
    except Exception as e:
        latency = time.time() - start_time
        return LLMErrorDetection(
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

    parsed = parse_llm_detection_response(raw_response)

    return LLMErrorDetection(
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
