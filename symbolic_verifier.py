"""
symbolic_verifier.py
--------------------
Phase 2: Automatic Error Detection in LLM Reasoning.
Module 1: Symbolic Verification using SymPy and Restricted Python AST.

Verifies the mathematical correctness of individual equations using a strict,
restricted Python AST parser. Does not use eval(), exec(), or parse_expr().
Includes safety limits on input length, AST node count/depth, exponent magnitude,
division-by-zero checks, and exact decimal representation via SymPy Rational.
"""

import ast
from dataclasses import dataclass
import re
from typing import Optional
import sympy


# Safety Limits
MAX_EQUATION_LENGTH = 500       # Maximum allowed characters in the raw equation string
MAX_AST_NODES = 100             # Maximum allowed AST nodes in any expression
MAX_AST_DEPTH = 25              # Maximum allowed AST nesting depth
MAX_EXPONENT = 1000             # Maximum allowed exponent magnitude to prevent memory exhaustion


@dataclass
class VerificationResult:
    """
    Structured outcome of an equation verification.
    status: 'correct', 'incorrect', or 'invalid'
    """
    original_equation: str
    lhs_expression: Optional[str]
    rhs_expression: Optional[str]
    status: str
    explanation: str
    lhs_value: Optional[str] = None
    rhs_value: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "original_equation": self.original_equation,
            "lhs_expression": self.lhs_expression,
            "rhs_expression": self.rhs_expression,
            "status": self.status,
            "explanation": self.explanation,
            "lhs_value": self.lhs_value,
            "rhs_value": self.rhs_value,
        }


def _clean_side(side_str: str) -> str:
    """
    Normalizes a side of an equation before parsing.
    Cleans arithmetic unicode characters and currency/comma artifacts:
      - Currency symbols: $, ₹, €, £
      - Comma separators
      - Arithmetic symbols: × -> *, ÷ -> /
      - Unicode spaces (narrow non-breaking spaces, non-breaking spaces)
      - Multiplication 'x' between numbers: 5 x 4 -> 5 * 4
      - Caret (^) -> **
    """
    s = str(side_str).strip()
    # Normalize unicode whitespace variants to standard space
    s = re.sub(r"[\u202f\u00a0\u2000-\u200b]", " ", s)
    # Remove common currency symbols and comma separators
    s = re.sub(r"[\$,\u20b9\u20ac\u00a3]", "", s)
    # Normalize arithmetic multiplication and division symbols
    s = s.replace("\u00d7", "*").replace("\u00f7", "/")
    # If 'x' or 'X' is used as multiplication between numbers e.g. 5 x 4 -> 5 * 4
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", s)
    # Support caret ^ as exponentiation operator
    s = s.replace("^", "**")
    return s.strip()


def _check_ast_safety(tree: ast.AST):
    """
    Validates that the AST does not exceed reasonable node count and nesting depth limits.
    Raises ValueError if any limit is exceeded.
    """
    node_count = 0

    def _calc_depth(node: ast.AST) -> int:
        nonlocal node_count
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise ValueError(f"Expression exceeds maximum allowed AST node count ({MAX_AST_NODES}).")
        max_child_depth = 0
        for child in ast.iter_child_nodes(node):
            d = _calc_depth(child)
            if d > max_child_depth:
                max_child_depth = d
        return 1 + max_child_depth

    depth = _calc_depth(tree)
    if depth > MAX_AST_DEPTH:
        raise ValueError(f"Expression exceeds maximum allowed AST nesting depth ({MAX_AST_DEPTH}).")


def _ast_node_to_sympy(node: ast.AST, source_text: str):
    """
    Recursively converts an approved AST node into a SymPy expression.

    Allowed nodes:
    - ast.Expression: container for mode='eval'
    - ast.Constant: numeric int or float (converted to exact Rational)
    - ast.UnaryOp: UAdd (+x), USub (-x)
    - ast.BinOp: Add (+), Sub (-), Mult (*), Div (/), Pow (**)

    Explicitly rejects:
    - Function calls (ast.Call)
    - Attribute access (ast.Attribute)
    - Variable names / arbitrary identifiers (ast.Name)
    - Division by zero
    - Exponents exceeding MAX_EXPONENT
    - Imports, subscripts, comprehensions, and all other Python node types.
    """
    if isinstance(node, ast.Expression):
        return _ast_node_to_sympy(node.body, source_text)

    elif isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            raise ValueError(f"Boolean literals are not supported: {val}")
        elif isinstance(val, int):
            return sympy.Integer(val)
        elif isinstance(val, float):
            # Extract the exact textual representation of the float to avoid binary float artifacts
            # (e.g. 0.1 -> Rational(1, 10))
            if hasattr(node, "col_offset") and hasattr(node, "end_col_offset"):
                literal_str = source_text[node.col_offset:node.end_col_offset].strip()
                try:
                    return sympy.Rational(literal_str)
                except Exception:
                    pass
            return sympy.Rational(str(val))
        else:
            raise ValueError(f"Unsupported constant type: {type(val).__name__} ({val!r})")

    elif isinstance(node, ast.UnaryOp):
        operand = _ast_node_to_sympy(node.operand, source_text)
        if isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    elif isinstance(node, ast.BinOp):
        left = _ast_node_to_sympy(node.left, source_text)
        right = _ast_node_to_sympy(node.right, source_text)

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            # Division by zero check
            if right == 0 or (hasattr(right, "is_zero") and right.is_zero):
                raise ValueError("Division by zero is undefined.")
            if isinstance(left, sympy.Integer) and isinstance(right, sympy.Integer):
                return sympy.Rational(left, right)
            return left / right
        elif isinstance(node.op, ast.Pow):
            # Exponent magnitude check without relying on float(right) conversion
            if not (hasattr(right, "is_real") and right.is_real):
                raise ValueError(f"Invalid or non-real exponent expression: {right}")
            try:
                if right > MAX_EXPONENT or right < -MAX_EXPONENT:
                    raise ValueError(
                        f"Exponent magnitude exceeds maximum limit ({MAX_EXPONENT})."
                    )
            except TypeError:
                raise ValueError(f"Unsafe or non-comparable exponent expression: {right}")
            except Exception as err:
                if "exceeds maximum limit" in str(err):
                    raise
                raise ValueError(f"Invalid exponent expression: {right}")
            return left ** right
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    elif isinstance(node, ast.Call):
        raise ValueError(f"Function calls are strictly forbidden: '{ast.unparse(node) if hasattr(ast, 'unparse') else 'call'}'")

    elif isinstance(node, ast.Attribute):
        raise ValueError(f"Attribute access is strictly forbidden: '{ast.unparse(node) if hasattr(ast, 'unparse') else 'attribute'}'")

    elif isinstance(node, ast.Name):
        raise ValueError(f"Variables and identifiers are not allowed in numerical verifier: '{node.id}'")

    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def _parse_side(side_str: str):
    """
    Parses a single side string into a SymPy expression safely via restricted AST.
    Returns (sympy_expr, None) on success or (None, error_msg) on failure.
    """
    cleaned = _clean_side(side_str)
    if not cleaned:
        return None, "Empty expression"

    try:
        tree = ast.parse(cleaned, mode="eval")
        _check_ast_safety(tree)
        sympy_expr = _ast_node_to_sympy(tree, cleaned)

        # Reject expressions evaluating to infinity or undefined (e.g. complex division by zero)
        if hasattr(sympy_expr, "is_finite") and sympy_expr.is_finite is False:
            return None, "Expression evaluated to non-finite or undefined value (e.g. division by zero)."
        if sympy.zoo in sympy_expr.atoms() or sympy.nan in sympy_expr.atoms():
            return None, "Expression contains undefined or infinite values (zoo/nan)."

        return sympy_expr, None
    except SyntaxError as e:
        return None, f"Syntax error in expression '{side_str.strip()}': {e.msg}"
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Parsing failed for '{side_str.strip()}': {e}"


def verify_equation(equation: str) -> VerificationResult:
    """
    Verifies whether an equation is mathematically valid and correct.

    Parameters:
    -----------
    equation : str
        An equation string containing an equality sign (e.g. "60 + 20 = 80").

    Returns:
    --------
    VerificationResult
        Structured dataclass containing:
        - original_equation
        - lhs_expression
        - rhs_expression
        - status ('correct', 'incorrect', 'invalid')
        - explanation
        - lhs_value (optional string)
        - rhs_value (optional string)
    """
    if not isinstance(equation, str) or not equation.strip():
        return VerificationResult(
            original_equation=str(equation) if equation is not None else "",
            lhs_expression=None,
            rhs_expression=None,
            status="invalid",
            explanation="Invalid input: equation string is empty or not a string.",
        )

    eq_str = equation.strip()

    # Safety Limit 1: Maximum raw equation length
    if len(eq_str) > MAX_EQUATION_LENGTH:
        return VerificationResult(
            original_equation=eq_str[:50] + "...",
            lhs_expression=None,
            rhs_expression=None,
            status="invalid",
            explanation=f"Invalid equation: input length ({len(eq_str)}) exceeds maximum allowed limit ({MAX_EQUATION_LENGTH} characters).",
        )

    # 1. Validate '=' delimiter
    if "=" not in eq_str:
        return VerificationResult(
            original_equation=eq_str,
            lhs_expression=None,
            rhs_expression=None,
            status="invalid",
            explanation="Invalid equation: missing '=' operator.",
        )

    parts = eq_str.split("=")
    if len(parts) > 2:
        return VerificationResult(
            original_equation=eq_str,
            lhs_expression=None,
            rhs_expression=None,
            status="invalid",
            explanation=f"Invalid equation: multiple '=' signs found ({len(parts) - 1} '=' signs).",
        )

    raw_lhs, raw_rhs = parts[0].strip(), parts[1].strip()

    if not raw_lhs or not raw_rhs:
        return VerificationResult(
            original_equation=eq_str,
            lhs_expression=raw_lhs or None,
            rhs_expression=raw_rhs or None,
            status="invalid",
            explanation="Invalid equation: missing left-hand or right-hand expression.",
        )

    # 2. Parse LHS safely via restricted AST
    lhs_expr, lhs_err = _parse_side(raw_lhs)
    if lhs_err is not None:
        return VerificationResult(
            original_equation=eq_str,
            lhs_expression=raw_lhs,
            rhs_expression=raw_rhs,
            status="invalid",
            explanation=f"Left-hand side error: {lhs_err}",
        )

    # 3. Parse RHS safely via restricted AST
    rhs_expr, rhs_err = _parse_side(raw_rhs)
    if rhs_err is not None:
        return VerificationResult(
            original_equation=eq_str,
            lhs_expression=raw_lhs,
            rhs_expression=raw_rhs,
            status="invalid",
            explanation=f"Right-hand side error: {rhs_err}",
        )

    # 4. Symbolic equivalence evaluation
    try:
        diff = sympy.simplify(lhs_expr - rhs_expr)
        is_equivalent = bool(diff == 0)

        lhs_val_str = str(lhs_expr)
        rhs_val_str = str(rhs_expr)

        if is_equivalent:
            return VerificationResult(
                original_equation=eq_str,
                lhs_expression=raw_lhs,
                rhs_expression=raw_rhs,
                status="correct",
                explanation=f"Equation is mathematically correct: {raw_lhs} evaluates to {lhs_val_str}, matching RHS ({rhs_val_str}).",
                lhs_value=lhs_val_str,
                rhs_value=rhs_val_str,
            )
        else:
            return VerificationResult(
                original_equation=eq_str,
                lhs_expression=raw_lhs,
                rhs_expression=raw_rhs,
                status="incorrect",
                explanation=f"Equation is mathematically incorrect: LHS ({raw_lhs} = {lhs_val_str}) does not equal RHS ({raw_rhs} = {rhs_val_str}).",
                lhs_value=lhs_val_str,
                rhs_value=rhs_val_str,
            )
    except Exception as e:
        return VerificationResult(
            original_equation=eq_str,
            lhs_expression=raw_lhs,
            rhs_expression=raw_rhs,
            status="invalid",
            explanation=f"Error evaluating mathematical equivalence: {e}",
        )


def _run_tests():
    """
    Runs full verification test suite covering:
    - 19 baseline test cases
    - Decimal precision with exact rationals (0.1 + 0.2 = 0.3, etc.)
    - Division by zero
    - Extremely large exponent (> MAX_EXPONENT)
    - Excessively long expression (> MAX_EQUATION_LENGTH)
    - Excessively nested expression (> MAX_AST_DEPTH)
    - Multiplication formatted as '5 x 4 = 20'
    """
    test_cases = [
        # Baseline test cases
        ("60 + 20 = 80", "correct"),
        ("60 + 20 = 90", "incorrect"),
        ("100 / 4 = 25", "correct"),
        ("50 - 10 = 30", "incorrect"),
        ("60 + 20", "invalid"),                       # Missing '='
        ("60 + 20 = 80 = 80", "invalid"),             # Multiple '='
        (" = 80", "invalid"),                         # Missing LHS
        ("60 + 20 = ", "invalid"),                    # Missing RHS
        ("60 + os.system('echo') = 80", "invalid"),   # Malformed / unsafe code injection attempt
        ("(15 * 4) + (20 / 2) = 70", "correct"),      # Parentheses & combined ops
        ("5 * 4 = 25", "incorrect"),                  # Multiplication mismatch
        ("2 ** 3 = 8", "correct"),                    # Exponentiation
        ("2 ^ 3 = 8", "correct"),                     # Caret exponentiation
        ("(5 + 3) * 2 = 16", "correct"),              # Parentheses
        ("__import__('os') = 80", "invalid"),         # Function call rejection
        ("60 + __import__('sys').exit() = 80", "invalid"), # Nested function call rejection
        ("os.system = 80", "invalid"),                # Attribute access rejection
        ("x + 2 = 5", "invalid"),                     # Variable identifier rejection
        ("total = 100", "invalid"),                   # Arbitrary identifier rejection

        # New required test cases: Decimal precision
        ("0.1 + 0.2 = 0.3", "correct"),
        ("0.1 + 0.2 = 0.4", "incorrect"),

        # Multiplication written with 'x', unicode '×', and division unicode '÷'
        ("5 x 4 = 20", "correct"),
        ("5 × 4 = 20", "correct"),
        ("20 ÷ 4 = 5", "correct"),

        # Currency symbol handling
        ("₹100 + 50 = 150", "correct"),

        # Division by zero
        ("10 / 0 = 5", "invalid"),
        ("50 / (5 - 5) = 10", "invalid"),

        # Extremely large exponent & nested huge exponent
        ("2 ** 10000 = 8", "invalid"),
        ("2 ^ 5000 = 10", "invalid"),
        ("2 ** (10 ** 100) = 8", "invalid"),

        # Excessively long expression (> 500 chars)
        ("1 + " * 200 + "1 = 201", "invalid"),

        # Excessively nested expression (> 25 nesting depth)
        ("-(" * 30 + "1" + ")" * 30 + " = 1", "invalid"),
    ]

    print("=" * 70)
    print("SYMBOLIC VERIFIER HARDENED TEST SUITE")
    print("=" * 70)

    passed_count = 0
    for eq, expected_status in test_cases:
        result = verify_equation(eq)
        is_pass = result.status == expected_status
        if is_pass:
            passed_count += 1
        tag = "[PASS]" if is_pass else "[FAIL]"
        display_eq = eq if len(eq) < 45 else eq[:40] + "..."
        print(f"{tag} Equation: '{display_eq}'")
        print(f"       Expected: {expected_status:<10} | Actual: {result.status:<10}")
        print(f"       Explanation: {result.explanation}\n")

    print("=" * 70)
    print(f"Test Summary: {passed_count}/{len(test_cases)} tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
