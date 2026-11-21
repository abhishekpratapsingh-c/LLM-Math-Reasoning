"""
equation_extractor.py
---------------------
Phase 2: Automatic Error Detection in LLM Reasoning.
Module 2: Deterministic Equation Extractor.

Extracts explicit mathematical equations from LLM reasoning texts without
modifying, evaluating, or calling any LLM API.
"""

import re
from typing import List

# Regex patterns for stripping Markdown and LaTeX formatting artifacts
_LATEX_ENV_PATTERN = re.compile(r"\\+\[|\\+\]|\\+\(|\\+\)")
_LATEX_TEXT_PATTERN = re.compile(r"\\+text\{([^}]*)\}")
_LATEX_FRAC_PATTERN = re.compile(r"\\+frac\{([^{}]+)\}\{([^{}]+)\}")


def _preprocess_text(text: str) -> str:
    """Cleans LaTeX / Markdown wrappers while preserving mathematical structure.
    Converts LaTeX \frac{a}{b} -> ((a)/(b)), strips \text{...}, and unescapes $.
    """
    if not text:
        return ""

    # Replace escaped currency e.g. \$ -> $
    t = text.replace(r"\$", "$")

    # Replace LaTeX delimiters: \[ \] \( \)
    t = _LATEX_ENV_PATTERN.sub(" ", t)

    # Convert \frac{a}{b} -> ((a)/(b)) (run twice for simple nested fractions)
    t = _LATEX_FRAC_PATTERN.sub(r"((\1)/(\2))", t)
    t = _LATEX_FRAC_PATTERN.sub(r"((\1)/(\2))", t)

    # Convert \text{word} -> word
    t = _LATEX_TEXT_PATTERN.sub(r"\1", t)

    # Remove LaTeX symbol commands like \times -> * and \div -> /
    t = re.sub(r"\\+times\b", "*", t)
    t = re.sub(r"\\+div\b", "/", t)

    return t


def _clean_token(s: str) -> str:
    """Strips surrounding markdown formatting, bullet markers, and sentence-ending punctuation."""
    t = s.strip()
    while t and t[-1] in "*_`~":
        t = t[:-1].strip()
    while t and t[0] in "*_`~":
        t = t[1:].strip()

    # Strip bullet points and list dashes at the start: '- ', '* ', '• '
    t = re.sub(r"^[\-\*\•\–\—\s]+", "", t).strip()

    # Strip sentence-ending punctuation from edge
    while t and t[-1] in ".,;:!?'\"":
        t = t[:-1].strip()
    while t and t[0] in ".,;:!?'\"":
        t = t[1:].strip()

    # Repeat markdown strip in case punctuation was inside markdown
    while t and t[-1] in "*_`~":
        t = t[:-1].strip()
    while t and t[0] in "*_`~":
        t = t[1:].strip()

    t = t.replace("**", "").replace("__", "")

    # Clean unclosed brackets or trailing open parens
    if t.endswith("(") and t.count("(") > t.count(")"):
        t = t[:-1].strip()
    if t.startswith(")") and t.count(")") > t.count("("):
        t = t[1:].strip()

    return t.strip()


def _strip_units(expr: str) -> str:
    """Removes unit words (e.g. 'hours', 'eggs', 'mi/day', 'mi', 'days') while preserving operators."""
    cleaned = re.sub(r"\b[a-zA-Z]+(?:/[a-zA-Z]+)?\b", "", expr)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_numeric_part(expr: str) -> str:
    """Extract the longest contiguous numeric/operator expression, preserving parentheses."""
    pattern = r"[\d\.\s\+\-\*\/\^\(\)×÷]+"
    matches = re.findall(pattern, expr)
    candidates = []
    for m in matches:
        m_strip = m.strip()
        if any(c.isdigit() for c in m_strip):
            m_clean = re.sub(r"^[^\d\(]+|[^\d\)]+$", "", m_strip).strip()
            if m_clean and any(c.isdigit() for c in m_clean):
                candidates.append(m_clean)
    if not candidates:
        return ""
    return max(candidates, key=len).strip()


def _is_valid_equation(lhs: str, rhs: str) -> bool:
    """Validate whether an equality represents a mathematical equation."""
    if not lhs or not rhs:
        return False

    if re.match(r"^(?:step|case|option|problem)\s+\d+$", lhs, re.IGNORECASE):
        return False

    if "%" in rhs and not re.search(r"[+\-*/^]", rhs):
        return False

    has_digit_lhs = bool(re.search(r"\d", lhs))
    has_digit_rhs = bool(re.search(r"\d", rhs))

    if not (has_digit_lhs or has_digit_rhs):
        return False

    has_arithmetic_lhs = bool(re.search(r"[+\-*/^%×÷]", lhs))
    has_arithmetic_rhs = bool(re.search(r"[+\-*/^%×÷]", rhs))

    is_var_assign = bool(
        re.match(r"^[a-zA-Z_]\w*$", lhs) and has_digit_rhs
    )

    if not (
        has_arithmetic_lhs
        or has_arithmetic_rhs
        or is_var_assign
        or (has_digit_lhs and has_digit_rhs)
    ):
        return False

    words_lhs = re.findall(r"\b[a-zA-Z]+\b", lhs)
    if len(words_lhs) > 1 and not has_arithmetic_lhs:
        return False

    words_rhs = re.findall(r"\b[a-zA-Z]+\b", rhs)
    if len(words_rhs) > 1 and not has_arithmetic_rhs:
        return False

    return True


def extract_equations(reasoning: str, numeric_only: bool = True) -> List[str]:
    """Extracts explicit mathematical equations from LLM reasoning text.

    Parameters
    ----------
    reasoning : str
        The raw LLM reasoning response text.
    numeric_only : bool, default True
        When True, only equations whose both sides consist solely of numeric
        expressions (no identifiers) are returned. Variable assignments such as
        ``x = 20`` or ``Total = 100 + 50`` are omitted. When False, the original
        broader extraction behavior is preserved.

    Returns
    -------
    List[str]
        Deduplicated list of extracted equation strings in order of occurrence.
    """
    if not isinstance(reasoning, str) or not reasoning.strip():
        return []

    cleaned_text = _preprocess_text(reasoning)
    found_equations: List[str] = []
    seen = set()

    for line in cleaned_text.splitlines():
        line_str = line.strip()
        if not line_str or "=" not in line_str:
            continue

        # Split line into clauses on backtick boundaries and sentence boundaries
        clauses = []
        subparts = re.split(r"`([^`]+)`", line_str)
        for sub in subparts:
            if not sub:
                continue
            # Split on sentence boundaries: punctuation followed by space (avoids breaking decimals like 3.14)
            sentence_parts = re.split(r"(?<=[.!?])\s+(?=[A-Za-z`*_]|$)", sub)
            for sp in sentence_parts:
                sp_strip = sp.strip()
                if sp_strip and "=" in sp_strip:
                    clauses.append(sp_strip)

        for clause in clauses:
            # Handle chained equalities: A = B = C
            parts = [p.strip() for p in clause.split("=")]
            if len(parts) >= 2:
                for i in range(len(parts) - 1):
                    raw_lhs = parts[i]
                    raw_rhs = parts[i + 1]

                    lhs_cand = _clean_token(raw_lhs)
                    lhs_cand = re.sub(r"^(?:let|so|then|and|we\s+have)\s+", "", lhs_cand, flags=re.IGNORECASE).strip()
                    rhs_cand = _clean_token(raw_rhs)

                    # Ensure RHS does not consume subsequent sentences or trailing conjunctions
                    rhs_sentence_match = re.search(r"(?<=\d|\))\s*\.\s+[A-Za-z]", rhs_cand)
                    if rhs_sentence_match:
                        rhs_cand = rhs_cand[:rhs_sentence_match.start()].strip()
                    if " and " in rhs_cand and not re.search(r"\b(?:and)\b.*=", rhs_cand):
                        rhs_cand = re.split(r"\s+\band\b\s+", rhs_cand)[0].strip()

                    rhs_cand = _clean_token(rhs_cand)

                    if numeric_only:
                        # Strip units while preserving operators
                        lhs_no_units = _strip_units(lhs_cand)
                        rhs_no_units = _strip_units(rhs_cand)

                        num_lhs = _extract_numeric_part(lhs_no_units)
                        num_rhs = _extract_numeric_part(rhs_no_units)

                        if not num_lhs or not num_rhs:
                            continue
                        if re.search(r"[a-zA-Z]", num_lhs) or re.search(r"[a-zA-Z]", num_rhs):
                            continue
                        if not _is_valid_equation(num_lhs, num_rhs):
                            continue
                        eq_str = f"{num_lhs} = {num_rhs}"
                    else:
                        if not _is_valid_equation(lhs_cand, rhs_cand):
                            continue
                        eq_str = f"{lhs_cand} = {rhs_cand}"

                    if eq_str not in seen:
                        seen.add(eq_str)
                        found_equations.append(eq_str)

    return found_equations


def _run_tests():
    """Test suite covering original behavior, numeric‑only mode, and regressions."""
    test_cases = [
        # 1. Standard equations
        (
            "The calculation gives 5 + 3 = 8.",
            ["5 + 3 = 8"],
            True,
        ),
        (
            "We divide 100 / 4 = 25.",
            ["100 / 4 = 25"],
            True,
        ),
        # 2. Variable assignments (should be omitted in numeric_only mode)
        (
            "Total = 100 + 50. Therefore final answer is 150.",
            [],
            True,
        ),
        (
            "Let x = 20.",
            [],
            True,
        ),
        # 3. Variable assignments retained when numeric_only=False
        (
            "Total = 100 + 50. Therefore final answer is 150.",
            ["Total = 100 + 50"],
            False,
        ),
        (
            "Let x = 20.",
            ["x = 20"],
            False,
        ),
        # 4. Unit handling
        (
            "2 hours × 2 = 4 hours",
            ["2 × 2 = 4"],
            True,
        ),
        (
            "2 hours × 2 = 4 hours",
            ["2 hours × 2 = 4 hours"],
            False,
        ),
        # 5. Multiple equations in complex reasoning text
        (
            "Janet has 16 eggs. She uses 3 + 4 = 7 eggs. The remainder is 16 - 7 = 9 eggs. She earns 9 * 2 = 18.",
            ["3 + 4 = 7", "16 - 7 = 9", "9 * 2 = 18"],
            True,
        ),
        # 6. Markdown & LaTeX formatting
        (
            "Equation in code: `15 * 4 = 60` and bold **50 + 50 = 100**.",
            ["15 * 4 = 60", "50 + 50 = 100"],
            True,
        ),
        (
            r"Using LaTeX: \\[ \\text{May sales} = \\frac{48}{2} = 24 \\]",
            ["((48)/(2)) = 24"],
            True,
        ),
        # 7. Non‑equation text (must NOT extract)
        (
            "The wallet costs $100 and she needs 50 minutes.",
            [],
            True,
        ),
        # 8. Additional regression tests
        (
            "40 mi/day × 5 days = 200 mi",
            ["40 × 5 = 200"],
            True,
        ),
        (
            "The answer is **5 + 3 = 8**.",
            ["5 + 3 = 8"],
            True,
        ),
        (
            "The answer is `5 + 3 = 8`.",
            ["5 + 3 = 8"],
            True,
        ),
        (
            "0.1 + 0.2 = 0.3. And 0.5 + 0.5 = 1.0.",
            ["0.1 + 0.2 = 0.3", "0.5 + 0.5 = 1.0"],
            True,
        ),
    ]

    print("=" * 70)
    print("EQUATION EXTRACTOR TEST SUITE")
    print("=" * 70)
    passed = 0
    for text, expected, mode in test_cases:
        extracted = extract_equations(text, numeric_only=mode)
        ok = extracted == expected
        if ok:
            passed += 1
        tag = "[PASS]" if ok else "[FAIL]"
        preview = text if len(text) < 45 else text[:40] + "..."
        print(f"{tag} Text: {preview!r}, numeric_only={mode}")
        print(f"       Expected: {expected}")
        print(f"       Actual:   {extracted}\n")
    print("=" * 70)
    print(f"Test Summary: {passed}/{len(test_cases)} tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
