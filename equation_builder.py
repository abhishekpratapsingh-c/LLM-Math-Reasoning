"""
equation_builder.py
-------------------
Phase 2, Objective 12: Deterministic Equation Builder for Structured Math Problems.

Transforms structured data extracted by PatternMatchResult (from problem_patterns.py)
into mathematically valid equation strings suitable for symbolic_solver.py.

Categories supported:
- repeated_rate: "x = N1 * N2 (* N3 ...)"
- weighted_sum: "x = Q1 * P1 + Q2 * P2 + ..."
- unknown_quantity: "Q1 * P1 + ... + R * x = Total"
- additive_partition: "Start + x + End = Total"
- percentage_discount: "x * (1 - Disc/100) = SalePrice"
"""

from typing import Any, Dict, List, Optional, Union
from problem_patterns import PatternMatchResult


def _fmt_num(val: Union[int, float]) -> str:
    """Formats numeric values cleanly: integers without trailing .0, floats as decimals."""
    if isinstance(val, (int, float)):
        try:
            f = float(val)
            if f.is_integer():
                return str(int(f))
            return f"{f:g}"
        except (ValueError, OverflowError):
            return str(val)
    return str(val)


def build_equation(match_result: PatternMatchResult) -> Optional[str]:
    """Constructs a deterministic mathematical equation string from a PatternMatchResult.

    Parameters
    ----------
    match_result : PatternMatchResult
        The result returned by classify_problem or individual pattern matchers.

    Returns
    -------
    Optional[str]
        A string equation (e.g. 'x = 3 * 3 * 60' or '20 + x + 15 = 60') ready for symbolic_solver,
        or None if match_result is invalid, unmatched, or contains missing fields.
    """
    if not isinstance(match_result, PatternMatchResult) or not match_result.matched:
        return None

    cat = match_result.category
    data = match_result.extracted_data

    # --------------------------------------------------------------------------
    # Category A: Repeated rate / multiplicative total
    # --------------------------------------------------------------------------
    if cat == "repeated_rate":
        factors = data.get("factors", [])
        if not factors or not isinstance(factors, list) or len(factors) < 2:
            return None
        formatted_factors = [_fmt_num(f) for f in factors]
        return f"x = {' * '.join(formatted_factors)}"

    # --------------------------------------------------------------------------
    # Category B: Multi-item shopping / weighted sum
    # --------------------------------------------------------------------------
    elif cat == "weighted_sum":
        items = data.get("items", [])
        if not items or not isinstance(items, list) or len(items) < 2:
            return None
        terms: List[str] = []
        for it in items:
            qty = it.get("quantity")
            price = it.get("unit_price")
            if qty is None or price is None:
                return None
            terms.append(f"{_fmt_num(qty)} * {_fmt_num(price)}")
        return f"x = {' + '.join(terms)}"

    # --------------------------------------------------------------------------
    # Category C: Unknown quantity with known total
    # --------------------------------------------------------------------------
    elif cat == "unknown_quantity":
        total = data.get("total")
        unknown_rate = data.get("unknown_rate")
        known_items = data.get("known_items", [])
        if total is None or unknown_rate is None or not known_items:
            return None

        known_terms: List[str] = []
        for it in known_items:
            qty = it.get("quantity", 1.0)
            price = it.get("unit_price")
            if price is None:
                return None
            if qty == 1.0 or qty == 1:
                known_terms.append(_fmt_num(price))
            else:
                known_terms.append(f"{_fmt_num(qty)} * {_fmt_num(price)}")

        lhs = " + ".join(known_terms) + f" + {_fmt_num(unknown_rate)} * x"
        return f"{lhs} = {_fmt_num(total)}"

    # --------------------------------------------------------------------------
    # Category D: Additive partition / segmented difference
    # --------------------------------------------------------------------------
    elif cat == "additive_partition":
        total = data.get("total")
        start_offset = data.get("start_offset")
        end_offset = data.get("end_offset")
        if total is None or start_offset is None or end_offset is None:
            return None
        return f"{_fmt_num(start_offset)} + x + {_fmt_num(end_offset)} = {_fmt_num(total)}"

    # --------------------------------------------------------------------------
    # Category E: Percentage discount / original price
    # --------------------------------------------------------------------------
    elif cat == "percentage_discount":
        sale_price = data.get("sale_price")
        discount_percent = data.get("discount_percent")
        if sale_price is None or discount_percent is None:
            return None
        return f"x * (1 - {_fmt_num(discount_percent)}/100) = {_fmt_num(sale_price)}"

    return None
