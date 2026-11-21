"""
symbolic_solver.py
------------------
Phase 2, Objective 11: Equation solving and mathematical expression
simplification using SymPy and a restricted, secure Python AST parser.

Features:
- Restricted AST parsing (no eval(), exec(), or parse_expr()).
- Arithmetic, fractional, bracketed, and algebraic expression simplification.
- Linear and quadratic equation solving with deterministic root sorting.
- Proposed solution substitution and verification.
- Hardened safety limits: max length, AST node count, AST nesting depth,
  exponent magnitude, division-by-zero detection, and dunder/unsafe name blocking.
"""

import ast
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy


# ==============================================================================
# Security & Safety Limits (reused and aligned with symbolic_verifier.py)
# ==============================================================================
MAX_EXPRESSION_LENGTH = 500      # Maximum raw string length
MAX_AST_NODES = 100              # Maximum allowed AST nodes per expression
MAX_AST_DEPTH = 25               # Maximum allowed AST nesting depth
MAX_EXPONENT = 1000              # Maximum exponent magnitude to prevent memory exhaustion

FORBIDDEN_NAMES = frozenset({
    "__import__", "eval", "exec", "open", "compile", "globals", "locals",
    "builtins", "breakpoint", "input", "print", "help", "exit", "quit",
    "os", "sys", "subprocess", "posix", "nt", "shutil"
})


# ==============================================================================
# Result Dataclasses
# ==============================================================================
@dataclass
class SimplifyResult:
    """Outcome of an expression simplification."""
    status: str                         # 'simplified' or 'invalid'
    original_expression: str
    simplified_expression: Optional[str]
    explanation: str
    sympy_expr: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "original_expression": self.original_expression,
            "simplified_expression": self.simplified_expression,
            "explanation": self.explanation,
        }


@dataclass
class SolveResult:
    """Outcome of an equation solving attempt."""
    status: str                         # 'solved' or 'invalid'
    original_equation: str
    variable: Optional[str]
    solution_type: str                  # 'unique', 'multiple', 'no_solution', 'infinite_solutions', 'invalid'
    solutions: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "original_equation": self.original_equation,
            "variable": self.variable,
            "solution_type": self.solution_type,
            "solutions": self.solutions,
            "explanation": self.explanation,
        }


@dataclass
class SolutionVerificationResult:
    """Outcome of checking whether a proposed solution satisfies an equation."""
    status: str                         # 'correct', 'incorrect', or 'invalid'
    original_equation: str
    proposed_solution: Any
    explanation: str
    evaluated_lhs: Optional[str] = None
    evaluated_rhs: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "original_equation": self.original_equation,
            "proposed_solution": str(self.proposed_solution),
            "explanation": self.explanation,
            "evaluated_lhs": self.evaluated_lhs,
            "evaluated_rhs": self.evaluated_rhs,
        }


# ==============================================================================
# Secure AST Parsing and Transformation
# ==============================================================================
def _clean_math_text(text: str) -> str:
    """Normalizes input text for AST parsing:
    - Normalizes Unicode whitespace variants to standard space
    - Strips currency symbols ($, ₹, €, £) and commas
    - Normalizes arithmetic multiplication (×) and division (÷)
    - Replaces caret (^) with exponentiation (**)
    - Converts letter 'x' or 'X' between numbers to multiplication (5 x 4 -> 5 * 4)
    - Normalizes implicit multiplication between numbers and variables (2x -> 2 * x)
    """
    s = str(text).strip()
    # Normalize unicode whitespace
    s = re.sub(r"[\u202f\u00a0\u2000-\u200b]", " ", s)
    # Strip currency symbols and commas
    s = re.sub(r"[\$,\u20b9\u20ac\u00a3]", "", s)
    # Normalize arithmetic symbols
    s = s.replace("\u00d7", "*").replace("\u00f7", "/")
    # Multiplication 'x' between numbers
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", s)
    # Exponentiation caret
    s = s.replace("^", "**")
    # Implicit multiplication: digit followed by variable (e.g. 2x -> 2 * x, 3y -> 3 * y)
    s = re.sub(r"(?<=\d)\s*([a-zA-Z_][a-zA-Z0-9_]*)", r" * \1", s)
    return s.strip()


def _check_ast_safety(tree: ast.AST):
    """Validates that the AST does not exceed node count and nesting depth limits."""
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


def _ast_to_sympy(node: ast.AST, source_text: str, allow_variables: bool = True) -> Any:
    """Recursively converts an approved AST node to a SymPy expression.

    Permitted nodes:
    - ast.Expression: container for mode='eval'
    - ast.Constant: numeric int or float (converted to exact Rational)
    - ast.Name: single or standard variable identifier (mapped to sympy.Symbol)
    - ast.UnaryOp: UAdd (+x), USub (-x)
    - ast.BinOp: Add (+), Sub (-), Mult (*), Div (/), Pow (**)

    Explicitly rejected:
    - Function calls (ast.Call)
    - Attribute access (ast.Attribute)
    - Dunder and private identifiers (e.g. __import__, _x)
    - Exponents exceeding MAX_EXPONENT
    - Division by zero
    """
    if isinstance(node, ast.Expression):
        return _ast_to_sympy(node.body, source_text, allow_variables)

    elif isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            raise ValueError(f"Boolean literals are not supported: {val}")
        elif isinstance(val, int):
            return sympy.Integer(val)
        elif isinstance(val, float):
            if hasattr(node, "col_offset") and hasattr(node, "end_col_offset"):
                literal_str = source_text[node.col_offset:node.end_col_offset].strip()
                try:
                    return sympy.Rational(literal_str)
                except Exception:
                    pass
            return sympy.Rational(str(val))
        else:
            raise ValueError(f"Unsupported constant type: {type(val).__name__} ({val!r})")

    elif isinstance(node, ast.Name):
        var_name = node.id
        if var_name.startswith("_") or var_name in FORBIDDEN_NAMES:
            raise ValueError(f"Identifier '{var_name}' is forbidden for security.")
        if not allow_variables:
            raise ValueError(f"Variables are not allowed in this context: '{var_name}'")
        return sympy.Symbol(var_name)

    elif isinstance(node, ast.UnaryOp):
        operand = _ast_to_sympy(node.operand, source_text, allow_variables)
        if isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    elif isinstance(node, ast.BinOp):
        left = _ast_to_sympy(node.left, source_text, allow_variables)
        right = _ast_to_sympy(node.right, source_text, allow_variables)

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0 or (hasattr(right, "is_zero") and right.is_zero):
                raise ValueError("Division by zero is undefined.")
            if isinstance(left, sympy.Integer) and isinstance(right, sympy.Integer):
                return sympy.Rational(left, right)
            return left / right
        elif isinstance(node.op, ast.Pow):
            # Check exponent magnitude if exponent is numeric
            if hasattr(right, "is_real") and right.is_real and right.is_number:
                if right > MAX_EXPONENT or right < -MAX_EXPONENT:
                    raise ValueError(f"Exponent magnitude exceeds maximum limit ({MAX_EXPONENT}).")
            return left ** right
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    elif isinstance(node, ast.Call):
        raise ValueError("Function calls are strictly forbidden.")

    elif isinstance(node, ast.Attribute):
        raise ValueError("Attribute access is strictly forbidden.")

    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def _parse_expression_safely(expr_str: str, allow_variables: bool = True) -> Tuple[Optional[Any], Optional[str]]:
    """Parses a mathematical expression string using restricted AST into a SymPy expression."""
    if not isinstance(expr_str, str) or not expr_str.strip():
        return None, "Empty or non-string expression."

    raw_str = expr_str.strip()
    if len(raw_str) > MAX_EXPRESSION_LENGTH:
        return None, f"Expression length ({len(raw_str)}) exceeds maximum allowed limit ({MAX_EXPRESSION_LENGTH} characters)."

    cleaned = _clean_math_text(raw_str)
    if not cleaned:
        return None, "Empty expression after cleaning."

    try:
        tree = ast.parse(cleaned, mode="eval")
        _check_ast_safety(tree)
        sympy_expr = _ast_to_sympy(tree, cleaned, allow_variables=allow_variables)

        if hasattr(sympy_expr, "is_finite") and sympy_expr.is_finite is False:
            return None, "Expression evaluated to non-finite or undefined value (e.g. division by zero)."
        if sympy.zoo in sympy_expr.atoms() or sympy.nan in sympy_expr.atoms():
            return None, "Expression contains undefined or infinite values (zoo/nan)."

        return sympy_expr, None
    except SyntaxError as e:
        return None, f"Syntax error in expression '{expr_str.strip()}': {e.msg}"
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Parsing failed for '{expr_str.strip()}': {e}"


# ==============================================================================
# Public API 1: Expression Simplification
# ==============================================================================
def simplify_expression(expr_str: str) -> SimplifyResult:
    """Simplifies a mathematical expression string (arithmetic, fractions, brackets, powers, or algebraic).

    Parameters
    ----------
    expr_str : str
        The expression to simplify (e.g. '12 / 16', '(2 + 3)**2', '2*x + 3*x').

    Returns
    -------
    SimplifyResult
        Dataclass containing status, original_expression, simplified_expression, and explanation.
    """
    sympy_expr, err = _parse_expression_safely(expr_str, allow_variables=True)
    if err is not None:
        return SimplifyResult(
            status="invalid",
            original_expression=str(expr_str) if expr_str is not None else "",
            simplified_expression=None,
            explanation=err,
            sympy_expr=None,
        )

    try:
        simplified = sympy.simplify(sympy_expr)
        simplified_str = str(simplified)
        return SimplifyResult(
            status="simplified",
            original_expression=expr_str,
            simplified_expression=simplified_str,
            explanation=f"Expression successfully simplified to {simplified_str}.",
            sympy_expr=simplified,
        )
    except Exception as e:
        return SimplifyResult(
            status="invalid",
            original_expression=expr_str,
            simplified_expression=None,
            explanation=f"Simplification error: {e}",
            sympy_expr=None,
        )


# ==============================================================================
# Public API 2: Equation Solving
# ==============================================================================
def solve_equation(equation_str: str, variable: Optional[str] = None) -> SolveResult:
    """Solves a linear or quadratic equation for the specified or automatically detected variable.

    Parameters
    ----------
    equation_str : str
        Equation containing '=' (e.g. 'x + 5 = 12', 'x**2 = 16').
    variable : Optional[str], default None
        Target variable name to solve for. If None, auto-detected when unique.

    Returns
    -------
    SolveResult
        Dataclass containing status, variable, solution_type, solutions, and explanation.
        solution_type is one of: 'unique', 'multiple', 'no_solution', 'infinite_solutions', 'invalid'.
    """
    if not isinstance(equation_str, str) or not equation_str.strip():
        return SolveResult(
            status="invalid",
            original_equation=str(equation_str) if equation_str is not None else "",
            variable=variable,
            solution_type="invalid",
            solutions=[],
            explanation="Equation string is empty or not a string.",
        )

    raw_eq = equation_str.strip()
    if len(raw_eq) > MAX_EXPRESSION_LENGTH:
        return SolveResult(
            status="invalid",
            original_equation=raw_eq[:50] + "...",
            variable=variable,
            solution_type="invalid",
            solutions=[],
            explanation=f"Input length ({len(raw_eq)}) exceeds maximum limit ({MAX_EXPRESSION_LENGTH}).",
        )

    if "=" not in raw_eq:
        return SolveResult(
            status="invalid",
            original_equation=raw_eq,
            variable=variable,
            solution_type="invalid",
            solutions=[],
            explanation="Invalid equation: missing '=' operator.",
        )

    parts = raw_eq.split("=")
    if len(parts) > 2:
        return SolveResult(
            status="invalid",
            original_equation=raw_eq,
            variable=variable,
            solution_type="invalid",
            solutions=[],
            explanation=f"Invalid equation: multiple '=' operators found ({len(parts) - 1}).",
        )

    raw_lhs, raw_rhs = parts[0].strip(), parts[1].strip()
    if not raw_lhs or not raw_rhs:
        return SolveResult(
            status="invalid",
            original_equation=raw_eq,
            variable=variable,
            solution_type="invalid",
            solutions=[],
            explanation="Invalid equation: missing left-hand or right-hand expression.",
        )

    # Parse LHS and RHS
    lhs_expr, lhs_err = _parse_expression_safely(raw_lhs, allow_variables=True)
    if lhs_err:
        return SolveResult(
            status="invalid",
            original_equation=raw_eq,
            variable=variable,
            solution_type="invalid",
            solutions=[],
            explanation=f"Left-hand side error: {lhs_err}",
        )

    rhs_expr, rhs_err = _parse_expression_safely(raw_rhs, allow_variables=True)
    if rhs_err:
        return SolveResult(
            status="invalid",
            original_equation=raw_eq,
            variable=variable,
            solution_type="invalid",
            solutions=[],
            explanation=f"Right-hand side error: {rhs_err}",
        )

    # Detect free variables
    all_symbols = sorted(list(lhs_expr.free_symbols | rhs_expr.free_symbols), key=lambda s: s.name)

    # Variable target determination
    if variable is not None:
        target_sym = sympy.Symbol(str(variable).strip())
    else:
        if len(all_symbols) == 1:
            target_sym = all_symbols[0]
        elif len(all_symbols) == 0:
            # No variables present: check numeric identity vs inconsistency
            diff = sympy.simplify(lhs_expr - rhs_expr)
            if diff == 0:
                return SolveResult(
                    status="solved",
                    original_equation=raw_eq,
                    variable=None,
                    solution_type="infinite_solutions",
                    solutions=[],
                    explanation="Equation is an identity (satisfied for all values).",
                )
            else:
                return SolveResult(
                    status="solved",
                    original_equation=raw_eq,
                    variable=None,
                    solution_type="no_solution",
                    solutions=[],
                    explanation="Equation is inconsistent (no solution exists).",
                )
        else:
            var_names = [s.name for s in all_symbols]
            return SolveResult(
                status="invalid",
                original_equation=raw_eq,
                variable=None,
                solution_type="invalid",
                solutions=[],
                explanation=f"Ambiguous target variable: multiple variables found {var_names} and no target specified.",
            )

    var_name = target_sym.name

    # Solve lhs - rhs = 0 for target_sym
    try:
        diff = sympy.simplify(lhs_expr - rhs_expr)

        # Check for identity: diff == 0
        if diff == 0:
            return SolveResult(
                status="solved",
                original_equation=raw_eq,
                variable=var_name,
                solution_type="infinite_solutions",
                solutions=[],
                explanation=f"Equation is an identity (satisfied for all values of {var_name}).",
            )

        # If target variable canceled out completely and diff != 0
        if target_sym not in diff.free_symbols:
            return SolveResult(
                status="solved",
                original_equation=raw_eq,
                variable=var_name,
                solution_type="no_solution",
                solutions=[],
                explanation=f"Equation has no solution for {var_name} (inconsistent).",
            )

        raw_solutions = sympy.solve(diff, target_sym)

        if not raw_solutions:
            return SolveResult(
                status="solved",
                original_equation=raw_eq,
                variable=var_name,
                solution_type="no_solution",
                solutions=[],
                explanation=f"No solution found for {var_name}.",
            )

        # Deterministically sort solutions numerically if real, else by string
        def _sort_key(sol):
            try:
                if sol.is_real and sol.is_number:
                    return (0, float(sol))
            except Exception:
                pass
            return (1, str(sol))

        sorted_solutions = sorted(raw_solutions, key=_sort_key)
        sol_strings = [str(s) for s in sorted_solutions]

        if len(sol_strings) == 1:
            sol_type = "unique"
            expl = f"Unique solution found for {var_name}: {sol_strings[0]}."
        else:
            sol_type = "multiple"
            expl = f"Multiple solutions found for {var_name}: {', '.join(sol_strings)}."

        return SolveResult(
            status="solved",
            original_equation=raw_eq,
            variable=var_name,
            solution_type=sol_type,
            solutions=sol_strings,
            explanation=expl,
        )

    except Exception as e:
        return SolveResult(
            status="invalid",
            original_equation=raw_eq,
            variable=var_name,
            solution_type="invalid",
            solutions=[],
            explanation=f"Error solving equation: {e}",
        )


# ==============================================================================
# Public API 3: Proposed Solution Verification
# ==============================================================================
def verify_solution(
    equation_str: str,
    proposed_solution: Union[Dict[str, Any], str]
) -> SolutionVerificationResult:
    """Verifies whether a proposed variable assignment satisfies the original equation.

    Parameters
    ----------
    equation_str : str
        The original equation (e.g. 'x + 5 = 12', 'x**2 = 16').
    proposed_solution : dict or str
        The proposed values, e.g. {'x': 7}, {'x': '1/2'}, 'x = 7', or '7' (if unique variable).

    Returns
    -------
    SolutionVerificationResult
        Dataclass containing status ('correct', 'incorrect', 'invalid') and explanation.
    """
    if not isinstance(equation_str, str) or not equation_str.strip():
        return SolutionVerificationResult(
            status="invalid",
            original_equation=str(equation_str) if equation_str is not None else "",
            proposed_solution=proposed_solution,
            explanation="Equation string is empty or not a string.",
        )

    raw_eq = equation_str.strip()
    if "=" not in raw_eq:
        return SolutionVerificationResult(
            status="invalid",
            original_equation=raw_eq,
            proposed_solution=proposed_solution,
            explanation="Invalid equation: missing '=' operator.",
        )

    parts = raw_eq.split("=")
    if len(parts) != 2:
        return SolutionVerificationResult(
            status="invalid",
            original_equation=raw_eq,
            proposed_solution=proposed_solution,
            explanation=f"Invalid equation: must contain exactly one '=' operator (found {len(parts) - 1}).",
        )

    raw_lhs, raw_rhs = parts[0].strip(), parts[1].strip()
    lhs_expr, lhs_err = _parse_expression_safely(raw_lhs, allow_variables=True)
    if lhs_err:
        return SolutionVerificationResult(
            status="invalid",
            original_equation=raw_eq,
            proposed_solution=proposed_solution,
            explanation=f"Left-hand side error: {lhs_err}",
        )

    rhs_expr, rhs_err = _parse_expression_safely(raw_rhs, allow_variables=True)
    if rhs_err:
        return SolutionVerificationResult(
            status="invalid",
            original_equation=raw_eq,
            proposed_solution=proposed_solution,
            explanation=f"Right-hand side error: {rhs_err}",
        )

    all_symbols = sorted(list(lhs_expr.free_symbols | rhs_expr.free_symbols), key=lambda s: s.name)

    # Parse proposed_solution into a dictionary of {Symbol: SymPy value}
    sub_map: Dict[Any, Any] = {}

    if isinstance(proposed_solution, dict):
        for k, v in proposed_solution.items():
            sym = sympy.Symbol(str(k).strip())
            parsed_val, val_err = _parse_expression_safely(str(v), allow_variables=False)
            if val_err:
                return SolutionVerificationResult(
                    status="invalid",
                    original_equation=raw_eq,
                    proposed_solution=proposed_solution,
                    explanation=f"Invalid proposed solution value for '{k}': {val_err}",
                )
            sub_map[sym] = parsed_val

    elif isinstance(proposed_solution, str):
        prop_str = proposed_solution.strip()
        if "=" in prop_str:
            sol_parts = prop_str.split("=")
            if len(sol_parts) != 2:
                return SolutionVerificationResult(
                    status="invalid",
                    original_equation=raw_eq,
                    proposed_solution=proposed_solution,
                    explanation="Invalid proposed solution string format.",
                )
            var_name, val_part = sol_parts[0].strip(), sol_parts[1].strip()
            parsed_val, val_err = _parse_expression_safely(val_part, allow_variables=False)
            if val_err:
                return SolutionVerificationResult(
                    status="invalid",
                    original_equation=raw_eq,
                    proposed_solution=proposed_solution,
                    explanation=f"Invalid proposed value '{val_part}': {val_err}",
                )
            sub_map[sympy.Symbol(var_name)] = parsed_val
        else:
            # Standalone value: requires exactly one variable in the equation
            if len(all_symbols) != 1:
                return SolutionVerificationResult(
                    status="invalid",
                    original_equation=raw_eq,
                    proposed_solution=proposed_solution,
                    explanation=f"Cannot associate standalone value with equation containing {len(all_symbols)} variables.",
                )
            parsed_val, val_err = _parse_expression_safely(prop_str, allow_variables=False)
            if val_err:
                return SolutionVerificationResult(
                    status="invalid",
                    original_equation=raw_eq,
                    proposed_solution=proposed_solution,
                    explanation=f"Invalid proposed value '{prop_str}': {val_err}",
                )
            sub_map[all_symbols[0]] = parsed_val

    else:
        return SolutionVerificationResult(
            status="invalid",
            original_equation=raw_eq,
            proposed_solution=proposed_solution,
            explanation=f"Unsupported proposed_solution type: {type(proposed_solution).__name__}",
        )

    # Evaluate equation sides after substitution
    try:
        eval_lhs = sympy.simplify(lhs_expr.subs(sub_map))
        eval_rhs = sympy.simplify(rhs_expr.subs(sub_map))

        diff = sympy.simplify(eval_lhs - eval_rhs)
        is_correct = bool(diff == 0)

        lhs_str = str(eval_lhs)
        rhs_str = str(eval_rhs)

        if is_correct:
            return SolutionVerificationResult(
                status="correct",
                original_equation=raw_eq,
                proposed_solution=proposed_solution,
                explanation=f"Proposed solution satisfies the equation: LHS evaluates to {lhs_str}, matching RHS ({rhs_str}).",
                evaluated_lhs=lhs_str,
                evaluated_rhs=rhs_str,
            )
        else:
            return SolutionVerificationResult(
                status="incorrect",
                original_equation=raw_eq,
                proposed_solution=proposed_solution,
                explanation=f"Proposed solution does not satisfy the equation: LHS evaluates to {lhs_str} while RHS evaluates to {rhs_str}.",
                evaluated_lhs=lhs_str,
                evaluated_rhs=rhs_str,
            )
    except Exception as e:
        return SolutionVerificationResult(
            status="invalid",
            original_equation=raw_eq,
            proposed_solution=proposed_solution,
            explanation=f"Error evaluating solution substitution: {e}",
        )


# ==============================================================================
# Comprehensive Test Suite
# ==============================================================================
def _run_tests():
    """Runs tests for expression simplification, equation solving, solution verification, and security."""
    print("=" * 75)
    print("SYMBOLIC SOLVER COMPREHENSIVE TEST SUITE")
    print("=" * 75)

    passed_count = 0
    total_count = 0

    def _assert_test(name: str, actual: Any, expected: Any):
        nonlocal passed_count, total_count
        total_count += 1
        is_pass = actual == expected
        if is_pass:
            passed_count += 1
        tag = "[PASS]" if is_pass else "[FAIL]"
        print(f"{tag} {name}")
        if not is_pass:
            print(f"       Expected: {expected!r}")
            print(f"       Actual:   {actual!r}")

    # --------------------------------------------------------------------------
    # 1. Expression Simplification Tests
    # --------------------------------------------------------------------------
    print("\n--- 1. Expression Simplification ---")
    s1 = simplify_expression("(15 * 4) + (20 / 2)")
    _assert_test("Arithmetic: (15 * 4) + (20 / 2) -> 70", s1.simplified_expression, "70")

    s2 = simplify_expression("12 / 16")
    _assert_test("Fraction reduction: 12 / 16 -> 3/4", s2.simplified_expression, "3/4")

    s3 = simplify_expression("(2 + 3)**2")
    _assert_test("Powers & brackets: (2 + 3)**2 -> 25", s3.simplified_expression, "25")

    s4 = simplify_expression("2*x + 3*x")
    _assert_test("Algebraic combination: 2*x + 3*x -> 5*x", s4.simplified_expression, "5*x")

    s5 = simplify_expression("(x - 2)*(x + 2)")
    _assert_test("Algebraic expansion: (x - 2)*(x + 2) -> x**2 - 4", s5.simplified_expression, "x**2 - 4")

    # --------------------------------------------------------------------------
    # 2. Equation Solving Tests
    # --------------------------------------------------------------------------
    print("\n--- 2. Equation Solving ---")
    sol1 = solve_equation("x + 5 = 12")
    _assert_test("Linear: x + 5 = 12 -> [7] (unique)", (sol1.solution_type, sol1.solutions), ("unique", ["7"]))

    sol2 = solve_equation("2*x = 20")
    _assert_test("Linear multiplicative: 2*x = 20 -> [10] (unique)", (sol2.solution_type, sol2.solutions), ("unique", ["10"]))

    sol3 = solve_equation("x / 4 = 5")
    _assert_test("Linear division: x / 4 = 5 -> [20] (unique)", (sol3.solution_type, sol3.solutions), ("unique", ["20"]))

    sol4 = solve_equation("x**2 = 16")
    _assert_test("Quadratic: x**2 = 16 -> [-4, 4] (multiple)", (sol4.solution_type, sol4.solutions), ("multiple", ["-4", "4"]))

    sol5 = solve_equation("x + 1 = x + 2")
    _assert_test("Inconsistent: x + 1 = x + 2 -> [] (no_solution)", (sol5.solution_type, sol5.solutions), ("no_solution", []))

    sol6 = solve_equation("x + 1 = 1 + x")
    _assert_test("Identity: x + 1 = 1 + x -> [] (infinite_solutions)", (sol6.solution_type, sol6.solutions), ("infinite_solutions", []))

    sol7 = solve_equation("x + y = 10")
    _assert_test("Ambiguous multi-variable rejection", sol7.solution_type, "invalid")

    sol8 = solve_equation("x + y = 10", variable="x")
    _assert_test("Explicit variable target in multi-variable: x + y = 10", (sol8.solution_type, sol8.solutions), ("unique", ["10 - y"]))

    # --------------------------------------------------------------------------
    # 3. Proposed Solution Verification Tests
    # --------------------------------------------------------------------------
    print("\n--- 3. Solution Verification ---")
    v1 = verify_solution("x + 5 = 12", {"x": 7})
    _assert_test("Correct linear: x + 5 = 12 with x=7", v1.status, "correct")

    v2 = verify_solution("x + 5 = 12", "x = 8")
    _assert_test("Incorrect linear: x + 5 = 12 with x=8", v2.status, "incorrect")

    v3 = verify_solution("x**2 = 16", "4")
    _assert_test("Quadratic positive root: x**2 = 16 with 4", v3.status, "correct")

    v4 = verify_solution("x**2 = 16", "-4")
    _assert_test("Quadratic negative root: x**2 = 16 with -4", v4.status, "correct")

    v5 = verify_solution("2*x = 1", {"x": "1/2"})
    _assert_test("Fractional solution: 2*x = 1 with x=1/2", v5.status, "correct")

    # --------------------------------------------------------------------------
    # 4. Security & Safety Limits Tests
    # --------------------------------------------------------------------------
    print("\n--- 4. Security & Hardened Limits ---")
    sec1 = solve_equation("os.system('echo') + x = 5")
    _assert_test("Reject function calls: os.system", sec1.status, "invalid")

    sec2 = solve_equation("__import__('os') + x = 5")
    _assert_test("Reject imports: __import__", sec2.status, "invalid")

    sec3 = solve_equation("__dunder__ + 5 = 10")
    _assert_test("Reject dunder identifiers: __dunder__", sec3.status, "invalid")

    sec4 = simplify_expression("x**10000")
    _assert_test("Reject excessive exponent (>1000)", sec4.status, "invalid")

    sec5 = simplify_expression("10 / 0")
    _assert_test("Detect division by zero: 10 / 0", sec5.status, "invalid")

    sec6 = simplify_expression("1 + " * 300 + "1")
    _assert_test("Reject excessive length (>500 chars)", sec6.status, "invalid")

    sec7 = simplify_expression("-(" * 30 + "x" + ")" * 30)
    _assert_test("Reject excessive nesting depth (>25 depth)", sec7.status, "invalid")

    print("\n" + "=" * 75)
    print(f"Test Summary: {passed_count}/{total_count} tests passed.")
    print("=" * 75)


if __name__ == "__main__":
    _run_tests()
