"""
problem_patterns.py
-------------------
Phase 2, Objective 12: Deterministic Pattern Matchers for Word Problems.

Implements strict, deterministic regex and linguistic pattern matchers for five
simple mathematical problem categories:
A. Repeated rate / multiplicative total
B. Multi-item shopping / weighted sum
C. Unknown quantity with known total
D. Additive partition / segmented difference
E. Percentage discount / original price

All matchers are offline, deterministic, and do not call any LLM or external service.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class PatternMatchResult:
    """Structured result of a problem pattern matching attempt."""
    category: str                       # e.g. 'repeated_rate', 'weighted_sum', etc.
    matched: bool
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


# Number word converter for simple textual numbers
_WORD_TO_NUMBER = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "dozen": 12, "half": 0.5
}


# Regex fragment matching an integer or decimal, with or without thousands-separating commas
_NUM_RE = r"[\d,]+(?:\.\d+)?"


def _parse_num(token: str) -> Optional[float]:
    """Converts a token (digits with optional commas or word) into a float."""
    if not token:
        return None
    token_clean = token.strip().lower().replace(",", "")
    if token_clean in _WORD_TO_NUMBER:
        return float(_WORD_TO_NUMBER[token_clean])
    try:
        return float(token_clean)
    except ValueError:
        return None


# ==============================================================================
# Category A: Repeated Rate / Multiplicative Total
# ==============================================================================
def match_repeated_rate(question: str) -> PatternMatchResult:
    """Matches problems where a quantity is repeated across multiple time/frequency dimensions.
    Example: 'James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint.
              How many total meters does he run a week?'
    Equation: x = 3 * 3 * 60 -> 540
    """
    # Requires an inquiry about total distance, volume, count, etc.
    if not re.search(r"how\s+many\s+total\b|how\s+many\b.*?\b(?:a|every|per)\s+(?:week|day|month|year)", question, re.I):
        return PatternMatchResult(category="repeated_rate", matched=False, explanation="No total rate question structure detected.")

    # Reject if it mentions discounts, shopping totals, or additive parts
    if re.search(r"\bdiscount\b|\bpaid\s+a\s+total\b|\bfirst\s+stopped\b", question, re.I):
        return PatternMatchResult(category="repeated_rate", matched=False, explanation="Contains conflicting problem cues.")

    # Pattern A1: N1 [items] N2 times a [period], N3 [unit] each [item]
    # Example: "3 sprints 3 times a week. He runs 60 meters each sprint."
    m1 = re.search(
        rf"({_NUM_RE}|[a-zA-Z]+)\s+([a-zA-Z]+)\s+({_NUM_RE}|[a-zA-Z]+)\s+times\s+a\s+([a-zA-Z]+)",
        question, re.I
    )
    m2 = re.search(
        rf"({_NUM_RE}|[a-zA-Z]+)\s+([a-zA-Z]+)\s+(?:each|per)\s+([a-zA-Z]+)",
        question, re.I
    )

    if m1 and m2:
        n1 = _parse_num(m1.group(1))
        n2 = _parse_num(m1.group(3))
        n3 = _parse_num(m2.group(1))
        if n1 is not None and n2 is not None and n3 is not None and n1 > 0 and n2 > 0 and n3 > 0:
            return PatternMatchResult(
                category="repeated_rate",
                matched=True,
                extracted_data={"factors": [n1, n2, n3]},
                explanation=f"Detected repeated rate: {n1} items, {n2} times, {n3} per item."
            )

    # Pattern A2: N1 times a [period], N2 [unit] each
    m_simple = re.search(
        rf"({_NUM_RE}|[a-zA-Z]+)\s+times\s+(?:a|per)\s+([a-zA-Z]+).*?({_NUM_RE}|[a-zA-Z]+)\s+([a-zA-Z]+)\s+(?:each|per)",
        question, re.I
    )
    if m_simple:
        n1 = _parse_num(m_simple.group(1))
        n2 = _parse_num(m_simple.group(3))
        if n1 is not None and n2 is not None and n1 > 0 and n2 > 0:
            return PatternMatchResult(
                category="repeated_rate",
                matched=True,
                extracted_data={"factors": [n1, n2]},
                explanation=f"Detected repeated rate: {n1} times, {n2} per occurrence."
            )

    return PatternMatchResult(category="repeated_rate", matched=False, explanation="Did not match repeated rate pattern.")


# ==============================================================================
# Category B: Multi-Item Shopping / Weighted Sum
# ==============================================================================
def match_weighted_sum(question: str) -> PatternMatchResult:
    """Matches problems calculating total cost from multiple itemized purchases.
    Example: 'She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes
              which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen.
              How much was the total cost?'
    Equation: x = 3 * 68 + 2 * 80 + 6 * 55 -> 694
    """
    # Must ask for total cost or total spent
    if not re.search(r"total\s+cost|total\s+price|how\s+much\s+(?:was|is)\s+the\s+total|spend\s+on\s+all|how\s+much\s+did\s+.*?\s+spend", question, re.I):
        return PatternMatchResult(category="weighted_sum", matched=False, explanation="No total cost inquiry detected.")

    # Reject if it mentions an unknown item (which belongs to Category C)
    if re.search(r"\bsome\b.*?(?:boxes|packs|items|pieces)|how\s+many\b.*?\b(?:buy|order|get)\b", question, re.I):
        return PatternMatchResult(category="weighted_sum", matched=False, explanation="Inquiry asks for count of unknown item (belongs to Category C).")

    items: List[Dict[str, float]] = []

    # Format B1: "N (dozen) [name] (which cost|for|at) $Price (per dozen|each)"
    # Example: "3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen"
    pattern1 = rf"({_NUM_RE}|[a-zA-Z]+)\s+(?:(?:a\s+)?dozen\s+)?([a-zA-Z\s]+?)\s+(?:which\s+cost|for|costing|at)\s+\$({_NUM_RE})"
    for match in re.finditer(pattern1, question, re.I):
        qty = _parse_num(match.group(1))
        price = _parse_num(match.group(3))
        if qty is not None and price is not None and qty > 0 and price > 0:
            items.append({"quantity": qty, "unit_price": price})

    # Format B2: "bought N1 pairs of shorts, N2 pairs of pants... One pair of shorts costs $Price1"
    # Example Q27 (Mishka)
    if len(items) < 2:
        items = []
        quantities = {}
        # Restrict quantity search to the purchase clause before price statements
        purchase_clause = re.split(r"(?:one|1)\s+(?:pair\s+of\s+)?[a-zA-Z]+\s+costs?", question, flags=re.I)[0]
        for match in re.finditer(rf"({_NUM_RE}|[a-zA-Z]+)\s+(?:pairs?\s+of\s+)?([a-zA-Z]+)", purchase_clause, re.I):
            q_val = _parse_num(match.group(1))
            name = match.group(2).lower()
            if q_val is not None and name not in {"dollars", "cost", "total", "pairs", "and"}:
                quantities[name] = q_val

        for match in re.finditer(rf"(?:one|1)\s+(?:pair\s+of\s+)?([a-zA-Z]+)\s+costs?\s+\$({_NUM_RE})", question, re.I):
            name = match.group(1).lower()
            price = _parse_num(match.group(2))
            if name in quantities and price is not None and price > 0:
                items.append({"quantity": quantities[name], "unit_price": price})

    if len(items) >= 2:
        return PatternMatchResult(
            category="weighted_sum",
            matched=True,
            extracted_data={"items": items},
            explanation=f"Detected multi-item shopping total with {len(items)} items."
        )

    return PatternMatchResult(category="weighted_sum", matched=False, explanation="Fewer than 2 itemized purchases found.")

    if len(items) >= 2:
        return PatternMatchResult(
            category="weighted_sum",
            matched=True,
            extracted_data={"items": items},
            explanation=f"Detected multi-item shopping total with {len(items)} items."
        )

    return PatternMatchResult(category="weighted_sum", matched=False, explanation="Fewer than 2 itemized purchases found.")


# ==============================================================================
# Category C: Unknown Quantity with Known Total
# ==============================================================================
def match_unknown_quantity(question: str) -> PatternMatchResult:
    """Matches problems with known purchases plus an unknown item count totaling a fixed amount.
    Example: 'Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each,
              4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50.
              How many boxes of pizza did Marie order if each box costs $8.50?'
    Equation: 12 + 5 * 3 + 4 * 1.50 + 8.50 * x = 50 -> 2
    """
    # 1. Total paid
    total_match = re.search(rf"paid\s+a\s+total\s+of\s+\$({_NUM_RE})|total\s+of\s+\$({_NUM_RE})", question, re.I)
    if not total_match:
        return PatternMatchResult(category="unknown_quantity", matched=False, explanation="No explicit total paid found.")

    total_val = _parse_num(total_match.group(1) or total_match.group(2))
    if total_val is None or total_val <= 0:
        return PatternMatchResult(category="unknown_quantity", matched=False, explanation="Invalid total value.")

    # 2. Unknown item rate
    rate_match = re.search(rf"(?:if\s+each|each)\s+([a-zA-Z\s]+?)\s+costs?\s+\$({_NUM_RE})", question, re.I)
    if not rate_match:
        return PatternMatchResult(category="unknown_quantity", matched=False, explanation="No rate found for unknown item.")

    unknown_name = rate_match.group(1).strip()
    unknown_rate = _parse_num(rate_match.group(2))
    if unknown_rate is None or unknown_rate <= 0:
        return PatternMatchResult(category="unknown_quantity", matched=False, explanation="Invalid unknown unit rate.")

    # 3. Known itemized purchases
    known_items: List[Dict[str, float]] = []

    # Single-item meal or fixed cost: e.g. "one chicken meal that costs $12"
    m_single = re.search(rf"(?:one|1)\s+([a-zA-Z\s]+?)\s+(?:that\s+)?costs?\s+\$({_NUM_RE})", question, re.I)
    if m_single and m_single.group(1).strip().lower() != unknown_name.lower():
        cost = _parse_num(m_single.group(2))
        if cost is not None and cost > 0:
            known_items.append({"quantity": 1.0, "unit_price": cost})

    # Multiple items with 'each': e.g. "5 packs of milk that costs $3 each"
    pattern_each = rf"({_NUM_RE}|[a-zA-Z]+)\s+([a-zA-Z\s]+?)\s+(?:that\s+)?costs?\s+\$({_NUM_RE})\s+(?:each|per)"
    for m in re.finditer(pattern_each, question, re.I):
        name = m.group(2).strip().lower()
        if unknown_name.lower() in name or name in unknown_name.lower():
            continue
        qty = _parse_num(m.group(1))
        unit_p = _parse_num(m.group(3))
        if qty is not None and unit_p is not None and qty > 0 and unit_p > 0:
            known_items.append({"quantity": qty, "unit_price": unit_p})

    if not known_items:
        return PatternMatchResult(category="unknown_quantity", matched=False, explanation="No known purchases detected.")

    return PatternMatchResult(
        category="unknown_quantity",
        matched=True,
        extracted_data={
            "known_items": known_items,
            "unknown_rate": unknown_rate,
            "total": total_val,
            "unknown_name": unknown_name,
        },
        explanation=f"Detected linear unknown quantity with total=${total_val}, rate=${unknown_rate}, and {len(known_items)} known purchases."
    )


# ==============================================================================
# Category D: Additive Partition / Segmented Difference
# ==============================================================================
def match_additive_partition(question: str) -> PatternMatchResult:
    """Matches problems involving a trip or partition with an overall total, initial stop, and end offset.
    Example: 'Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles.
              His second stop was 15 miles before the end of the trip. How many miles did he travel
              between his first and second stops?'
    Equation: 20 + x + 15 = 60 -> 25
    """
    # 1. Total span
    total_match = re.search(
        rf"({_NUM_RE})-(?:mile|kilometer|km|meter|hour|day|page)\s+(?:bike\s+trip|trip|journey|race|trail|drive)",
        question, re.I
    )
    if not total_match:
        return PatternMatchResult(category="additive_partition", matched=False, explanation="No overall partition/trip length detected.")

    total_val = _parse_num(total_match.group(1))
    if total_val is None or total_val <= 0:
        return PatternMatchResult(category="additive_partition", matched=False, explanation="Invalid total trip distance.")

    # 2. First stop offset
    start_match = re.search(
        rf"(?:first\s+stopped\s+after|stopped\s+after|first\s+stop\s+(?:was\s+after|was))\s+({_NUM_RE})",
        question, re.I
    )
    if not start_match:
        return PatternMatchResult(category="additive_partition", matched=False, explanation="No initial stop offset detected.")

    start_offset = _parse_num(start_match.group(1))
    if start_offset is None or start_offset <= 0:
        return PatternMatchResult(category="additive_partition", matched=False, explanation="Invalid start offset.")

    # 3. Second stop / end offset
    end_match = re.search(
        rf"({_NUM_RE})\s*(?:miles|km|kilometers|meters)?\s+before\s+the\s+end",
        question, re.I
    )
    if not end_match:
        return PatternMatchResult(category="additive_partition", matched=False, explanation="No end offset detected.")

    end_offset = _parse_num(end_match.group(1))
    if end_offset is None or end_offset <= 0:
        return PatternMatchResult(category="additive_partition", matched=False, explanation="Invalid end offset.")

    # Physical feasibility check: stops must not exceed or equal the entire trip
    if start_offset + end_offset >= total_val:
        return PatternMatchResult(
            category="additive_partition",
            matched=False,
            explanation=f"Invalid segment bounds: start ({start_offset}) + end ({end_offset}) >= total ({total_val})."
        )

    # Inquiry check
    if not re.search(r"between\s+(?:his|her|the)?\s*(?:first\s+and\s+second|stops)", question, re.I):
        return PatternMatchResult(category="additive_partition", matched=False, explanation="Inquiry does not ask for intermediate segment.")

    return PatternMatchResult(
        category="additive_partition",
        matched=True,
        extracted_data={
            "total": total_val,
            "start_offset": start_offset,
            "end_offset": end_offset,
        },
        explanation=f"Detected additive partition: total={total_val}, start_offset={start_offset}, end_offset={end_offset}."
    )


# ==============================================================================
# Category E: Percentage Discount / Original Price
# ==============================================================================
def match_percentage_discount(question: str) -> PatternMatchResult:
    """Matches problems finding the original price given the sale price and discount percentage.
    Example: 'Kyle bought last year's best-selling book for $19.50. This is with a 25% discount
              from the original price. What was the original price of the book?'
    Equation: x * (1 - 25/100) = 19.50 -> 26
    """
    # 1. Inquiry check
    if not re.search(r"what\s+was\s+the\s+original\s+(?:price|cost)|find\s+the\s+original\s+(?:price|cost)", question, re.I):
        return PatternMatchResult(category="percentage_discount", matched=False, explanation="Inquiry does not ask for original price.")

    # 2. Sale price
    price_match = re.search(rf"(?:bought|paid|for|purchased).*?\$({_NUM_RE})", question, re.I)
    if not price_match:
        return PatternMatchResult(category="percentage_discount", matched=False, explanation="No sale price found.")

    sale_price = _parse_num(price_match.group(1))
    if sale_price is None or sale_price <= 0:
        return PatternMatchResult(category="percentage_discount", matched=False, explanation="Invalid sale price.")

    # 3. Discount percentage
    disc_match = re.search(rf"({_NUM_RE})\%\s+discount", question, re.I)
    if not disc_match:
        return PatternMatchResult(category="percentage_discount", matched=False, explanation="No discount percentage found.")

    discount_percent = _parse_num(disc_match.group(1))
    if discount_percent is None or discount_percent <= 0 or discount_percent >= 100:
        return PatternMatchResult(category="percentage_discount", matched=False, explanation="Invalid discount percentage.")

    return PatternMatchResult(
        category="percentage_discount",
        matched=True,
        extracted_data={
            "sale_price": sale_price,
            "discount_percent": discount_percent,
        },
        explanation=f"Detected percentage discount: sale_price=${sale_price}, discount={discount_percent}%."
    )


# List of all active matchers in order of priority
ALL_MATCHERS = [
    match_percentage_discount,
    match_additive_partition,
    match_unknown_quantity,
    match_weighted_sum,
    match_repeated_rate,
]


def classify_problem(question: str) -> PatternMatchResult:
    """Attempts to match a question string against all supported deterministic categories."""
    if not isinstance(question, str) or not question.strip():
        return PatternMatchResult(category="none", matched=False, explanation="Empty or non-string question.")

    for matcher in ALL_MATCHERS:
        res = matcher(question)
        if res.matched:
            return res

    return PatternMatchResult(category="unsupported", matched=False, explanation="Question does not match any supported deterministic category.")
