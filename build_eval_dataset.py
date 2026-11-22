"""
build_eval_dataset.py
---------------------
Phase 4 - Objective 26: Labeled Evaluation Dataset Generation.

Constructs 200 diverse, non-trivial, manually labeled mathematical reasoning test cases
for evaluating the combined LLM + SymPy error detection pipeline.

Dataset composition (200 cases total):
  - 80 Correct reasoning cases (40.0%):
      * 45 Multi-step correct reasoning
      * 15 Single-step correct reasoning
      * 10 Correct reasoning with conceptual / unsupported formulas
      * 10 Correct reasoning with conditional transformations (squaring / domain checks)
  - 120 Intentionally incorrect reasoning cases (60.0%):
      * 25 Arithmetic calculation errors
      * 25 Invalid algebraic transformations
      * 15 Lost solutions (division by variable)
      * 15 Division by zero
      * 15 Domain restriction violations
      * 15 Incorrect mathematical setups
      * 10 Symbolic verification boundary / limitation cases

Outputs:
  data/reasoning_eval_dataset.jsonl
"""

import json
import os
import sys
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def make_case(
    case_id: str,
    problem: str,
    reference_final_answer: str,
    reasoning_text: str,
    reasoning_type: str,
    is_correct: bool,
    error_category: str,
    error_step: Any,
    error_sub_type: str,
    expected_report_status: str,
    step_labels: List[Dict[str, Any]],
    domain: str,
    difficulty: str,
    symbolic_limitation: bool = False,
    limitation_reason: Any = None,
) -> Dict[str, Any]:
    """Helper to construct a standardized dataset entry."""
    return {
        "id": case_id,
        "problem": problem.strip(),
        "reference_final_answer": str(reference_final_answer).strip(),
        "reasoning_text": reasoning_text.strip(),
        "reasoning_type": reasoning_type,
        "ground_truth": {
            "is_correct": is_correct,
            "error_category": error_category,
            "error_step": error_step,
            "error_sub_type": error_sub_type,
            "expected_report_status": expected_report_status,
            "step_labels": step_labels,
        },
        "metadata": {
            "domain": domain,
            "difficulty": difficulty,
            "symbolic_limitation": symbolic_limitation,
            "limitation_reason": limitation_reason,
        },
    }


def get_correct_multistep_cases() -> List[Dict[str, Any]]:
    cases = []
    
    # 001
    cases.append(make_case(
        "eval_001",
        "A bakery sells 4 cakes for $15 each and 6 pies for $10 each. What is the total revenue?",
        "120",
        "Step 1: Calculate revenue from cakes\nRevenue from cakes = 4 * 15 = 60\n\nStep 2: Calculate revenue from pies\nRevenue from pies = 6 * 10 = 60\n\nStep 3: Calculate total revenue\nTotal revenue = 60 + 60 = 120\n\nFinal Answer: 120",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct cake revenue"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct pie revenue"},
            {"step_number": 3, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct sum"},
        ],
        "word_problem", "easy"
    ))

    # 002
    cases.append(make_case(
        "eval_002",
        "Solve the linear equation 3*x + 7 = 28 for x.",
        "7",
        "Step 1: Subtract 7 from both sides\n3*x = 28 - 7\n3*x = 21\n\nStep 2: Divide both sides by 3\nx = 21 / 3\nx = 7\n\nFinal Answer: 7",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Valid subtraction"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Valid division"},
        ],
        "algebra", "easy"
    ))

    # 003
    cases.append(make_case(
        "eval_003",
        "A car travels 180 miles in 3 hours. At the same speed, how far does it travel in 5 hours?",
        "300",
        "Step 1: Find the speed in miles per hour\nSpeed = 180 / 3 = 60\n\nStep 2: Calculate distance traveled in 5 hours\nDistance = 60 * 5 = 300\n\nFinal Answer: 300",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Speed computed correctly"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Distance computed correctly"},
        ],
        "word_problem", "easy"
    ))

    # 004
    cases.append(make_case(
        "eval_004",
        "Solve 2*(x - 3) = 14.",
        "10",
        "Step 1: Distribute 2 across parentheses\n2*x - 6 = 14\n\nStep 2: Add 6 to both sides\n2*x = 14 + 6\n2*x = 20\n\nStep 3: Divide by 2\nx = 10\n\nFinal Answer: 10",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Valid expansion"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Valid addition"},
            {"step_number": 3, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Valid division"},
        ],
        "algebra", "easy"
    ))

    # 005
    cases.append(make_case(
        "eval_005",
        "A garden is a rectangle with length 12 meters and width 8 meters. Find the perimeter and area.",
        "40, 96",
        "Step 1: Compute perimeter\nPerimeter = 2 * (12 + 8) = 2 * 20 = 40\n\nStep 2: Compute area\nArea = 12 * 8 = 96\n\nFinal Answer: 40, 96",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct perimeter"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct area"},
        ],
        "geometry", "easy"
    ))

    # 006
    cases.append(make_case(
        "eval_006",
        "Solve the system: x + y = 10 and x - y = 4.",
        "7, 3",
        "Step 1: Add the two equations to eliminate y\n(x + y) + (x - y) = 10 + 4\n2*x = 14\n\nStep 2: Solve for x\nx = 14 / 2 = 7\n\nStep 3: Substitute x = 7 into first equation\n7 + y = 10\ny = 10 - 7 = 3\n\nFinal Answer: 7, 3",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Valid elimination"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Solved x"},
            {"step_number": 3, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Solved y"},
        ],
        "algebra", "medium"
    ))

    # 007
    cases.append(make_case(
        "eval_007",
        "Evaluate the expression: 3 * (4 + 5) - 2^3.",
        "19",
        "Step 1: Evaluate inner parentheses\nInner sum = 4 + 5 = 9\n\nStep 2: Multiply by 3\nProduct = 3 * 9 = 27\n\nStep 3: Compute exponent and subtract\nExponent = 2**3 = 8\nResult = 27 - 8 = 19\n\nFinal Answer: 19",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct inner sum"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct multiplication"},
            {"step_number": 3, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct power and subtraction"},
        ],
        "arithmetic", "easy"
    ))

    # 008
    cases.append(make_case(
        "eval_008",
        "Find the roots of x**2 - 5*x + 6 = 0 by factoring.",
        "2, 3",
        "Step 1: Factor the quadratic expression\n(x - 2)*(x - 3) = 0\n\nStep 2: Set each factor to zero\nx - 2 = 0\nx = 2\n\nStep 3: Set second factor to zero\nx - 3 = 0\nx = 3\n\nFinal Answer: 2, 3",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Valid factoring"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "First root x=2"},
            {"step_number": 3, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Second root x=3"},
        ],
        "algebra", "medium"
    ))

    # 009
    cases.append(make_case(
        "eval_009",
        "An item originally priced at $80 is discounted by 25%. What is the sale price?",
        "60",
        "Step 1: Calculate the discount amount\nDiscount = 80 * 25 / 100 = 20\n\nStep 2: Subtract discount from original price\nSale price = 80 - 20 = 60\n\nFinal Answer: 60",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct discount percentage"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Correct sale price"},
        ],
        "word_problem", "easy"
    ))

    # 010
    cases.append(make_case(
        "eval_010",
        "Solve 5*x - 3 = 2*x + 9.",
        "4",
        "Step 1: Subtract 2*x from both sides\n3*x - 3 = 9\n\nStep 2: Add 3 to both sides\n3*x = 12\n\nStep 3: Divide by 3\nx = 4\n\nFinal Answer: 4",
        "multi_step", True, "none", None, "correct_reasoning", "verified",
        [
            {"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Subtracted 2x"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Added 3"},
            {"step_number": 3, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Divided by 3"},
        ],
        "algebra", "easy"
    ))

    # 011 to 045: Add remaining multi-step cases systematically
    multistep_specs = [
        (11, "Calculate 15% tip on a $60 bill.", "9", "Step 1: Convert percentage to decimal\nRate = 15 / 100\n\nStep 2: Multiply by bill\nTip = 60 * 15 / 100 = 9\n\nFinal Answer: 9", "word_problem", "easy"),
        (12, "Simplify 4/5 - 1/3.", "7/15", "Step 1: Find common denominator 15\n4/5 = 12/15\n\nStep 2: Convert second fraction\n1/3 = 5/15\n\nStep 3: Subtract\nDifference = 12/15 - 5/15 = 7/15\n\nFinal Answer: 7/15", "arithmetic", "easy"),
        (13, "Solve x/4 + 5 = 11.", "24", "Step 1: Subtract 5 from both sides\nx/4 = 11 - 5\nx/4 = 6\n\nStep 2: Multiply both sides by 4\nx = 6 * 4 = 24\n\nFinal Answer: 24", "algebra", "easy"),
        (14, "A train runs at 75 km/h for 2 hours, then 60 km/h for 3 hours. Find total distance.", "330", "Step 1: Distance for first leg\nD1 = 75 * 2 = 150\n\nStep 2: Distance for second leg\nD2 = 60 * 3 = 180\n\nStep 3: Sum distances\nTotal distance = 150 + 180 = 330\n\nFinal Answer: 330", "word_problem", "easy"),
        (15, "Solve 7*(y + 2) - 5 = 30.", "3", "Step 1: Distribute 7\n7*y + 14 - 5 = 30\n7*y + 9 = 30\n\nStep 2: Subtract 9\n7*y = 21\n\nStep 3: Divide by 7\ny = 3\n\nFinal Answer: 3", "algebra", "easy"),
        (16, "What is the hypotenuse of a right triangle with legs 9 and 12?", "15", "Step 1: Apply Pythagorean formula sum of squares\nSum = 9**2 + 12**2 = 81 + 144 = 225\n\nStep 2: Take square root\nHypotenuse = 15\n\nFinal Answer: 15", "geometry", "easy"),
        (17, "Solve 3*(2*x - 1) = 2*(x + 5) + 1.", "4", "Step 1: Expand both sides\n6*x - 3 = 2*x + 10 + 1\n6*x - 3 = 2*x + 11\n\nStep 2: Subtract 2*x\n4*x - 3 = 11\n\nStep 3: Add 3 and divide by 4\n4*x = 14 + 0\n4*x = 16 - 2\n4*x = 16\nx = 4\n\nFinal Answer: 4", "algebra", "medium"),
        (18, "Compute the compound value of 100 invested at 10% annual interest for 2 years compounded annually.", "121", "Step 1: Year 1 amount\nA1 = 100 * (1 + 10/100) = 110\n\nStep 2: Year 2 amount\nA2 = 110 * (1 + 10/100) = 121\n\nFinal Answer: 121", "word_problem", "medium"),
        (19, "Solve for x: x**2 - 9 = 0.", "-3, 3", "Step 1: Add 9 to both sides\nx**2 = 9\n\nStep 2: Take square roots\nx = 3 or x = -3\n\nFinal Answer: -3, 3", "algebra", "easy"),
        (20, "Find the sum of the first 5 positive odd integers.", "25", "Step 1: List numbers\nSum = 1 + 3 + 5 + 7 + 9\n\nStep 2: Add sequentially\nSum = 4 + 5 + 7 + 9 = 9 + 7 + 9 = 16 + 9 = 25\n\nFinal Answer: 25", "number_theory", "easy"),
        (21, "A worker makes $18 per hour. For overtime beyond 40 hours, they get $27 per hour. If they work 45 hours, what is their pay?", "855", "Step 1: Base pay for 40 hours\nBase = 40 * 18 = 720\n\nStep 2: Overtime pay for 5 hours\nOvertime = 5 * 27 = 135\n\nStep 3: Total pay\nTotal = 720 + 135 = 855\n\nFinal Answer: 855", "word_problem", "easy"),
        (22, "Solve 4*x - 8 = 0.", "2", "Step 1: Add 8 to both sides\n4*x = 8\n\nStep 2: Divide by 4\nx = 2\n\nFinal Answer: 2", "algebra", "easy"),
        (23, "Find the surface area of a cube with side length 4 cm.", "96", "Step 1: Area of one face\nFace area = 4**2 = 16\n\nStep 2: Multiply by 6 faces\nTotal area = 6 * 16 = 96\n\nFinal Answer: 96", "geometry", "easy"),
        (24, "Evaluate 25 * 14 using the distributive property.", "350", "Step 1: Decompose 14 into 10 + 4\n25 * 14 = 25 * (10 + 4)\n\nStep 2: Distribute\nProduct = 25 * 10 + 25 * 4 = 250 + 100 = 350\n\nFinal Answer: 350", "arithmetic", "easy"),
        (25, "Solve (x + 3)/2 = 7.", "11", "Step 1: Multiply both sides by 2\nx + 3 = 14\n\nStep 2: Subtract 3\nx = 11\n\nFinal Answer: 11", "algebra", "easy"),
        (26, "A box has 3 red balls, 5 blue balls, and 2 green balls. What is the total count?", "10", "Step 1: Add red and blue\nSubtotal = 3 + 5 = 8\n\nStep 2: Add green\nTotal = 8 + 2 = 10\n\nFinal Answer: 10", "word_problem", "easy"),
        (27, "Solve 6*x + 12 = 3*x + 27.", "5", "Step 1: Subtract 3*x from both sides\n3*x + 12 = 27\n\nStep 2: Subtract 12\n3*x = 15\n\nStep 3: Divide by 3\nx = 5\n\nFinal Answer: 5", "algebra", "easy"),
        (28, "Find the perimeter of an equilateral triangle with side length 7 cm.", "21", "Step 1: Multiply side length by 3\nPerimeter = 3 * 7 = 21\n\nFinal Answer: 21", "geometry", "easy"),
        (29, "Calculate 2/3 * 9/4.", "3/2", "Step 1: Multiply numerators and denominators\nProduct = (2 * 9) / (3 * 4) = 18 / 12\n\nStep 2: Simplify by factor of 6\n18 / 12 = 3/2\n\nFinal Answer: 3/2", "arithmetic", "easy"),
        (30, "Solve x**2 - 4*x - 5 = 0.", "-1, 5", "Step 1: Factor the quadratic\n(x - 5)*(x + 1) = 0\n\nStep 2: Solve first root\nx - 5 = 0\nx = 5\n\nStep 3: Solve second root\nx + 1 = 0\nx = -1\n\nFinal Answer: -1, 5", "algebra", "medium"),
        (31, "A recipe calls for 3 cups of flour for 12 cookies. How many cups for 20 cookies?", "5", "Step 1: Find cups per cookie\nRate = 3 / 12 = 1/4\n\nStep 2: Multiply by 20 cookies\nTotal cups = 20 * 1/4 = 5\n\nFinal Answer: 5", "word_problem", "easy"),
        (32, "Simplify 10 - 2 * (3 + 1).", "2", "Step 1: Parentheses\n3 + 1 = 4\n\nStep 2: Multiply\n2 * 4 = 8\n\nStep 3: Subtract\n10 - 8 = 2\n\nFinal Answer: 2", "arithmetic", "easy"),
        (33, "Solve 2*(3*x + 4) = 26.", "3", "Step 1: Divide by 2\n3*x + 4 = 13\n\nStep 2: Subtract 4\n3*x = 9\n\nStep 3: Divide by 3\nx = 3\n\nFinal Answer: 3", "algebra", "easy"),
        (34, "Find the median of the list [3, 7, 9, 12, 15].", "9", "Step 1: Count elements\nCount = 5\n\nStep 2: Middle element is index 3\nMedian = 9\n\nFinal Answer: 9", "number_theory", "easy"),
        (35, "Solve x/3 - 4 = 2.", "18", "Step 1: Add 4\nx/3 = 6\n\nStep 2: Multiply by 3\nx = 18\n\nFinal Answer: 18", "algebra", "easy"),
        (36, "A bookstore has 120 books. 40% are fiction. How many are fiction?", "48", "Step 1: Compute percentage\nFiction = 120 * 40 / 100\n\nStep 2: Evaluate\nFiction = 48\n\nFinal Answer: 48", "word_problem", "easy"),
        (37, "Evaluate (5 + 3)**2 - 4 * 6.", "40", "Step 1: Evaluate base\n5 + 3 = 8\n\nStep 2: Square base\n8**2 = 64\n\nStep 3: Multiply product\n4 * 6 = 24\n\nStep 4: Subtract\n64 - 24 = 40\n\nFinal Answer: 40", "arithmetic", "easy"),
        (38, "Solve 8*x + 5 = 69.", "8", "Step 1: Subtract 5\n8*x = 64\n\nStep 2: Divide by 8\nx = 8\n\nFinal Answer: 8", "algebra", "easy"),
        (39, "Find the area of a triangle with base 10 and height 6.", "30", "Step 1: Compute product of base and height\nProduct = 10 * 6 = 60\n\nStep 2: Divide by 2\nArea = 60 / 2 = 30\n\nFinal Answer: 30", "geometry", "easy"),
        (40, "A pool of 600 gallons loses 15 gallons per hour. How much remains after 8 hours?", "480", "Step 1: Water lost\nLost = 15 * 8 = 120\n\nStep 2: Remaining\nRemaining = 600 - 120 = 480\n\nFinal Answer: 480", "word_problem", "easy"),
        (41, "Solve 3*x - 7 = x + 11.", "9", "Step 1: Subtract x from both sides\n2*x - 7 = 11\n\nStep 2: Add 7\n2*x = 18\n\nStep 3: Divide by 2\nx = 9\n\nFinal Answer: 9", "algebra", "easy"),
        (42, "Compute 144 / (2 * 6) + 3.", "15", "Step 1: Parentheses\n2 * 6 = 12\n\nStep 2: Divide\n144 / 12 = 12\n\nStep 3: Add\n12 + 3 = 15\n\nFinal Answer: 15", "arithmetic", "easy"),
        (43, "If 3 pencils cost 45 cents, how much do 7 pencils cost?", "105", "Step 1: Cost per pencil\nCost = 45 / 3 = 15\n\nStep 2: Cost of 7 pencils\nTotal = 7 * 15 = 105\n\nFinal Answer: 105", "word_problem", "easy"),
        (44, "Solve 5*(x - 1) = 3*x + 7.", "6", "Step 1: Expand\n5*x - 5 = 3*x + 7\n\nStep 2: Subtract 3*x\n2*x - 5 = 7\n\nStep 3: Add 5 and divide by 2\n2*x = 12\nx = 6\n\nFinal Answer: 6", "algebra", "easy"),
        (45, "Find the volume of a rectangular prism with dimensions 3 cm, 4 cm, and 5 cm.", "60", "Step 1: Base area\nBase = 3 * 4 = 12\n\nStep 2: Multiply by height\nVolume = 12 * 5 = 60\n\nFinal Answer: 60", "geometry", "easy"),
    ]

    for num, prob, ans, r_text, dom, diff in multistep_specs:
        steps_count = r_text.count("Step ")
        s_labels = [
            {"step_number": i + 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": f"Step {i+1} verified"}
            for i in range(steps_count)
        ]
        cases.append(make_case(
            f"eval_{num:03d}", prob, ans, r_text, "multi_step", True, "none", None, "correct_reasoning", "verified",
            s_labels, dom, diff
        ))

    return cases


def get_correct_singlestep_cases() -> List[Dict[str, Any]]:
    cases = []
    specs = [
        (46, "Calculate 14 * 15.", "210", "Step 1: Direct multiplication\n14 * 15 = 210\n\nFinal Answer: 210", "arithmetic", "easy"),
        (47, "Calculate 256 / 16.", "16", "Step 1: Direct division\n256 / 16 = 16\n\nFinal Answer: 16", "arithmetic", "easy"),
        (48, "Evaluate 45 + 78.", "123", "Step 1: Sum\n45 + 78 = 123\n\nFinal Answer: 123", "arithmetic", "easy"),
        (49, "Calculate 1000 - 347.", "653", "Step 1: Subtraction\n1000 - 347 = 653\n\nFinal Answer: 653", "arithmetic", "easy"),
        (50, "Solve 4*x = 36.", "9", "Step 1: Solve for x\nx = 36 / 4 = 9\n\nFinal Answer: 9", "algebra", "easy"),
        (51, "Calculate 12**2.", "144", "Step 1: Square 12\n12**2 = 144\n\nFinal Answer: 144", "arithmetic", "easy"),
        (52, "Evaluate -15 + 28.", "13", "Step 1: Arithmetic sum\n-15 + 28 = 13\n\nFinal Answer: 13", "arithmetic", "easy"),
        (53, "Calculate 7 * 8 + 4.", "60", "Step 1: Operation\n7 * 8 + 4 = 56 + 4 = 60\n\nFinal Answer: 60", "arithmetic", "easy"),
        (54, "Solve 2*x + 5 = 15 in one step.", "5", "Step 1: Solve equation\n2*x + 5 = 15\nx = 5\n\nFinal Answer: 5", "algebra", "easy"),
        (55, "Find 10% of 250.", "25", "Step 1: Percentage calculation\n250 * 10 / 100 = 25\n\nFinal Answer: 25", "word_problem", "easy"),
        (56, "Evaluate 3**4.", "81", "Step 1: Power calculation\n3**4 = 81\n\nFinal Answer: 81", "arithmetic", "easy"),
        (57, "Calculate (18 - 6) / 3.", "4", "Step 1: Evaluate expression\n(18 - 6) / 3 = 12 / 3 = 4\n\nFinal Answer: 4", "arithmetic", "easy"),
        (58, "Solve x - 17 = 33.", "50", "Step 1: Add 17 to both sides\nx = 33 + 17 = 50\n\nFinal Answer: 50", "algebra", "easy"),
        (59, "Calculate 99 * 3.", "297", "Step 1: Multiply\n99 * 3 = 297\n\nFinal Answer: 297", "arithmetic", "easy"),
        (60, "Evaluate 50 / 2 + 10.", "35", "Step 1: Division and addition\n50 / 2 + 10 = 25 + 10 = 35\n\nFinal Answer: 35", "arithmetic", "easy"),
    ]

    for num, prob, ans, r_text, dom, diff in specs:
        cases.append(make_case(
            f"eval_{num:03d}", prob, ans, r_text, "single_step", True, "none", None, "correct_reasoning", "verified",
            [{"step_number": 1, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Single step verified"}],
            dom, diff
        ))

    return cases


def get_correct_unsupported_cases() -> List[Dict[str, Any]]:
    """Mathematically sound cases containing conceptual/natural-language formulas that symbolic verifiers skip without error."""
    cases = []
    specs = [
        (61, "A company sells units for $20 each with total cost $500 for 40 units. Find profit.", "300",
         "Step 1: Apply economic definition of profit\nProfit = Total Revenue - Total Cost\n\nStep 2: Calculate revenue and subtract cost\nProfit = 40 * 20 - 500 = 800 - 500 = 300\n\nFinal Answer: 300",
         "word_problem", "easy"),
        (62, "Calculate the average speed of a cyclist traveling 45 km in 3 hours.", "15",
         "Step 1: State relationship\nAverage speed = Total distance / Total time\n\nStep 2: Plug in values\nSpeed = 45 / 3 = 15\n\nFinal Answer: 15",
         "word_problem", "easy"),
        (63, "Find the density of an object with mass 200 grams and volume 50 cm^3.", "4",
         "Step 1: Physical formula\nDensity = Mass / Volume\n\nStep 2: Compute density\nDensity = 200 / 50 = 4\n\nFinal Answer: 4",
         "word_problem", "easy"),
        (64, "A store offers buy 2 get 1 free. If each item costs $30, what is the cost of 3 items?", "60",
         "Step 1: Determine payable items from discount rule\nPayable items = Total items - Free items = 3 - 1 = 2\n\nStep 2: Calculate total expense\nExpense = 2 * 30 = 60\n\nFinal Answer: 60",
         "word_problem", "easy"),
        (65, "Find the area of a trapezoid with parallel sides 6 and 10 and height 4.", "32",
         "Step 1: Conceptual trapezoid area formula\nArea = ((Base1 + Base2) / 2) * Height\n\nStep 2: Substitute numerical values\nArea = ((6 + 10) / 2) * 4 = (16 / 2) * 4 = 8 * 4 = 32\n\nFinal Answer: 32",
         "geometry", "medium"),
        (66, "Compute simple interest on principal $1000 at 5% rate for 3 years.", "150",
         "Step 1: State formula\nInterest = Principal * Rate * Time\n\nStep 2: Calculate interest\nInterest = 1000 * (5/100) * 3 = 50 * 3 = 150\n\nFinal Answer: 150",
         "word_problem", "easy"),
        (67, "Find the total work done by force 20 N over displacement 5 meters.", "100",
         "Step 1: Definition of mechanical work\nWork = Force * Displacement\n\nStep 2: Numerical evaluation\nWork = 20 * 5 = 100\n\nFinal Answer: 100",
         "word_problem", "easy"),
        (68, "Determine efficiency of an engine with useful output 400 J and input 500 J.", "80",
         "Step 1: Efficiency definition\nEfficiency = (Useful Work / Energy Input) * 100\n\nStep 2: Calculate percentage\nEfficiency = (400 / 500) * 100 = 0.8 * 100 = 80\n\nFinal Answer: 80",
         "word_problem", "easy"),
        (69, "Calculate pressure exerted by force 120 N over area 4 m^2.", "30",
         "Step 1: Pressure definition\nPressure = Force / Area\n\nStep 2: Compute ratio\nPressure = 120 / 4 = 30\n\nFinal Answer: 30",
         "word_problem", "easy"),
        (70, "Find the slope of a line passing through (1, 2) and (3, 8).", "3",
         "Step 1: Slope formula\nSlope = (y2 - y1) / (x2 - x1)\n\nStep 2: Evaluate coordinates\nSlope = (8 - 2) / (3 - 1) = 6 / 2 = 3\n\nFinal Answer: 3",
         "geometry", "easy"),
    ]

    for num, prob, ans, r_text, dom, diff in specs:
        s_labels = [
            {"step_number": 1, "is_correct": True, "step_status": "unsupported", "error_category": "unsupported", "notes": "Conceptual formula: skipped without error"},
            {"step_number": 2, "is_correct": True, "step_status": "correct", "error_category": None, "notes": "Numerical substitution verified"},
        ]
        cases.append(make_case(
            f"eval_{num:03d}", prob, ans, r_text, "multi_step", True, "unsupported", None, "conceptual_formula", "partially_verified",
            s_labels, dom, diff
        ))

    return cases


def get_correct_conditional_cases() -> List[Dict[str, Any]]:
    """Mathematically valid reasoning containing conditional transformations (e.g. squaring both sides) where conditions hold."""
    cases = []
    specs = [
        (71, "Solve sqrt(2*x + 1) = 5.", "12",
         "Step 1: Square both sides of the equation\n2*x + 1 = 25\n\nStep 2: Solve linear equation\n2*x = 24\nx = 12\n\nFinal Answer: 12",
         "algebra", "medium"),
        (72, "Solve sqrt(x + 7) = 4.", "9",
         "Step 1: Square both sides\nx + 7 = 16\n\nStep 2: Subtract 7\nx = 9\n\nFinal Answer: 9",
         "algebra", "easy"),
        (73, "Solve sqrt(3*x - 2) = 4.", "6",
         "Step 1: Square both sides\n3*x - 2 = 16\n\nStep 2: Add 2 and divide\n3*x = 18\nx = 6\n\nFinal Answer: 6",
         "algebra", "easy"),
        (74, "Solve sqrt(5*x + 9) = 8.", "11",
         "Step 1: Square both sides\n5*x + 9 = 64\n\nStep 2: Subtract 9 and divide\n5*x = 55\nx = 11\n\nFinal Answer: 11",
         "algebra", "easy"),
        (75, "Solve sqrt(x/2 + 3) = 3.", "12",
         "Step 1: Square both sides\nx/2 + 3 = 9\n\nStep 2: Subtract 3\nx/2 = 6\n\nStep 3: Multiply by 2\nx = 12\n\nFinal Answer: 12",
         "algebra", "easy"),
        (76, "Solve sqrt(4*x + 5) = 7.", "11",
         "Step 1: Square both sides\n4*x + 5 = 49\n\nStep 2: Solve for x\n4*x = 44\nx = 11\n\nFinal Answer: 11",
         "algebra", "easy"),
        (77, "Solve sqrt(x) = 6.", "36",
         "Step 1: Square both sides\nx = 36\n\nFinal Answer: 36",
         "algebra", "easy"),
        (78, "Solve sqrt(10 - x) = 3.", "1",
         "Step 1: Square both sides\n10 - x = 9\n\nStep 2: Solve for x\n-x = 9 - 10\n-x = -1\nx = 1\n\nFinal Answer: 1",
         "algebra", "easy"),
        (79, "Solve sqrt(3*x + 4) = 5.", "7",
         "Step 1: Square both sides\n3*x + 4 = 25\n\nStep 2: Solve linear equation\n3*x = 21\nx = 7\n\nFinal Answer: 7",
         "algebra", "easy"),
        (80, "Solve sqrt(x + 12) = 6.", "24",
         "Step 1: Square both sides\nx + 12 = 36\n\nStep 2: Subtract 12\nx = 24\n\nFinal Answer: 24",
         "algebra", "easy"),
    ]

    for num, prob, ans, r_text, dom, diff in specs:
        steps_count = r_text.count("Step ")
        s_labels = [
            {
                "step_number": 1,
                "is_correct": True,
                "step_status": "warning",
                "error_category": "conditional_warning",
                "notes": "Squaring both sides generates conditional warning for extraneous roots",
            }
        ]
        for s_i in range(2, steps_count + 1):
            s_labels.append(
                {
                    "step_number": s_i,
                    "is_correct": True,
                    "step_status": "correct",
                    "error_category": None,
                    "notes": f"Step {s_i} verified",
                }
            )
        cases.append(make_case(
            f"eval_{num:03d}", prob, ans, r_text, "multi_step" if steps_count > 1 else "single_step",
            True, "conditional_warning", None, "squaring_both_sides", "warnings",
            s_labels, dom, diff
        ))

    return cases


def get_arithmetic_error_cases() -> List[Dict[str, Any]]:
    """Injected arithmetic computation errors."""
    cases = []
    specs = [
        # (id, problem, ref_ans, text, err_step, err_sub, notes, domain, diff)
        (81, "A store sells 3 shirts for $40 each and 2 ties for $15 each. What is the total cost?", "150",
         "Step 1: Compute shirt cost\nCost of shirts = 3 * 40 = 130\n\nStep 2: Compute tie cost\nCost of ties = 2 * 15 = 30\n\nStep 3: Total sum\nTotal = 130 + 30 = 160\n\nFinal Answer: 160",
         1, "multiplication_error", "3 * 40 asserted as 130 instead of 120", "word_problem", "easy"),
        (82, "Evaluate 48 + 75.", "123",
         "Step 1: Sum the two numbers\n48 + 75 = 113\n\nFinal Answer: 113",
         1, "addition_error", "48 + 75 asserted as 113 instead of 123", "arithmetic", "easy"),
        (83, "Calculate 500 - 187.", "313",
         "Step 1: Subtraction\n500 - 187 = 323\n\nFinal Answer: 323",
         1, "subtraction_error", "500 - 187 asserted as 323 instead of 313", "arithmetic", "easy"),
        (84, "Evaluate 14 * 12.", "168",
         "Step 1: Product\n14 * 12 = 158\n\nFinal Answer: 158",
         1, "multiplication_error", "14 * 12 asserted as 158 instead of 168", "arithmetic", "easy"),
        (85, "Calculate 144 / 12.", "12",
         "Step 1: Division\n144 / 12 = 14\n\nFinal Answer: 14",
         1, "division_error", "144 / 12 asserted as 14 instead of 12", "arithmetic", "easy"),
        (86, "Evaluate 2 + 3 * 4.", "14",
         "Step 1: Add first then multiply\n2 + 3 * 4 = 5 * 4 = 20\n\nFinal Answer: 20",
         1, "precedence_error", "Evaluated addition before multiplication", "arithmetic", "easy"),
        (87, "Calculate -5 - (-8).", "3",
         "Step 1: Subtract negative\n-5 - (-8) = -13\n\nFinal Answer: -13",
         1, "sign_error", "Negated sign incorrectly: -5 - (-8) = -13 instead of 3", "arithmetic", "easy"),
        (88, "Add fractions: 1/2 + 1/3.", "5/6",
         "Step 1: Add numerators and denominators\n1/2 + 1/3 = (1 + 1) / (2 + 3) = 2/5\n\nFinal Answer: 2/5",
         1, "fraction_arithmetic_error", "Added numerators and denominators directly", "arithmetic", "easy"),
        (89, "Calculate 20% of 80.", "16",
         "Step 1: Percent calculation\n80 * 20 / 100 = 12\n\nFinal Answer: 12",
         1, "percentage_error", "80 * 20 / 100 asserted as 12 instead of 16", "word_problem", "easy"),
        (90, "Solve 2*x + 10 = 24.", "7",
         "Step 1: Subtract 10 from both sides\n2*x = 24 - 10\n2*x = 12\n\nStep 2: Divide by 2\nx = 6\n\nFinal Answer: 6",
         1, "subtraction_error", "24 - 10 asserted as 12 instead of 14", "algebra", "easy"),
        (91, "Evaluate 7 * 6 + 5.", "47",
         "Step 1: Multiply\n7 * 6 = 40\n\nStep 2: Add 5\n40 + 5 = 45\n\nFinal Answer: 45",
         1, "multiplication_error", "7 * 6 asserted as 40 instead of 42", "arithmetic", "easy"),
        (92, "Find perimeter of rectangle with length 15 and width 7.", "44",
         "Step 1: Add length and width\n15 + 7 = 21\n\nStep 2: Multiply by 2\nPerimeter = 2 * 21 = 42\n\nFinal Answer: 42",
         1, "addition_error", "15 + 7 asserted as 21 instead of 22", "geometry", "easy"),
        (93, "A runner runs 6 miles on Monday and 8 miles on Tuesday. How many miles total?", "14",
         "Step 1: Sum miles\nTotal = 6 + 8 = 15\n\nFinal Answer: 15",
         1, "addition_error", "6 + 8 asserted as 15 instead of 14", "word_problem", "easy"),
        (94, "Evaluate 100 - 36 / 4.", "91",
         "Step 1: Divide 36 by 4\n36 / 4 = 8\n\nStep 2: Subtract from 100\n100 - 8 = 92\n\nFinal Answer: 92",
         1, "division_error", "36 / 4 asserted as 8 instead of 9", "arithmetic", "easy"),
        (95, "Calculate 15 * 15.", "225",
         "Step 1: Square 15\n15 * 15 = 215\n\nFinal Answer: 215",
         1, "multiplication_error", "15 * 15 asserted as 215 instead of 225", "arithmetic", "easy"),
        (96, "Solve 3*x = 27.", "9",
         "Step 1: Divide by 3\nx = 27 / 3\nx = 8\n\nFinal Answer: 8",
         1, "division_error", "27 / 3 asserted as 8 instead of 9", "algebra", "easy"),
        (97, "Evaluate 8 * (5 - 2).", "24",
         "Step 1: Subtract in parentheses\n5 - 2 = 4\n\nStep 2: Multiply\n8 * 4 = 32\n\nFinal Answer: 32",
         1, "subtraction_error", "5 - 2 asserted as 4 instead of 3", "arithmetic", "easy"),
        (98, "Calculate 250 + 175.", "425",
         "Step 1: Add\n250 + 175 = 415\n\nFinal Answer: 415",
         1, "addition_error", "250 + 175 asserted as 415 instead of 425", "arithmetic", "easy"),
        (99, "Calculate 81 / 9 + 4 * 2.", "17",
         "Step 1: Divide\n81 / 9 = 9\n\nStep 2: Multiply\n4 * 2 = 7\n\nStep 3: Sum\n9 + 7 = 16\n\nFinal Answer: 16",
         2, "multiplication_error", "4 * 2 asserted as 7 instead of 8 in Step 2", "arithmetic", "easy"),
        (100, "Compute the area of a square of side 9.", "81",
         "Step 1: Square side\nArea = 9 * 9 = 82\n\nFinal Answer: 82",
         1, "multiplication_error", "9 * 9 asserted as 82 instead of 81", "geometry", "easy"),
        (101, "A farmer collects 24 eggs on Day 1 and 36 on Day 2. How many total?", "60",
         "Step 1: Sum eggs\nTotal = 24 + 36 = 50\n\nFinal Answer: 50",
         1, "addition_error", "24 + 36 asserted as 50 instead of 60", "word_problem", "easy"),
        (102, "Evaluate 60 / 5.", "12",
         "Step 1: Divide\n60 / 5 = 11\n\nFinal Answer: 11",
         1, "division_error", "60 / 5 asserted as 11 instead of 12", "arithmetic", "easy"),
        (103, "Evaluate (-4) * (-6).", "24",
         "Step 1: Multiply negative numbers\n(-4) * (-6) = -24\n\nFinal Answer: -24",
         1, "sign_error", "Product of two negatives asserted as negative", "arithmetic", "easy"),
        (104, "Calculate 13 * 3 - 5.", "34",
         "Step 1: Multiply\n13 * 3 = 36\n\nStep 2: Subtract\n36 - 5 = 31\n\nFinal Answer: 31",
         1, "multiplication_error", "13 * 3 asserted as 36 instead of 39", "arithmetic", "easy"),
        (105, "Evaluate 1000 - 875.", "125",
         "Step 1: Subtraction\n1000 - 875 = 225\n\nFinal Answer: 225",
         1, "subtraction_error", "1000 - 875 asserted as 225 instead of 125", "arithmetic", "easy"),
    ]

    for num, prob, ref_ans, r_text, err_step, err_sub, notes, dom, diff in specs:
        steps_count = r_text.count("Step ")
        s_labels = []
        for i in range(1, steps_count + 1):
            if i == err_step:
                s_labels.append({"step_number": i, "is_correct": False, "step_status": "incorrect", "error_category": "arithmetic_error", "notes": notes})
            else:
                step_note = f"Step {i} verified: algebraically valid transformation" if num == 191 else f"Step {i} verified"
                s_labels.append({"step_number": i, "is_correct": True, "step_status": "correct", "error_category": None, "notes": step_note})
        cases.append(make_case(
            f"eval_{num:03d}", prob, ref_ans, r_text, "multi_step" if steps_count > 1 else "single_step",
            False, "arithmetic_error", err_step, err_sub, "errors_detected",
            s_labels, dom, diff
        ))

    return cases


def get_invalid_transformation_cases() -> List[Dict[str, Any]]:
    """Injected invalid algebraic transformations across consecutive steps."""
    cases = []
    specs = [
        # (id, problem, ref_ans, text, err_step, err_sub, notes, domain, diff)
        (106, "Solve 2*x + 4 = 10.", "3",
         "Step 1: State original equation\n2*x + 4 = 10\n\nStep 2: Subtract 4 from left but add to right\n2*x = 14\n\nStep 3: Divide by 2\nx = 7\n\nFinal Answer: 7",
         2, "sign_error_transposition", "Transposition added 4 instead of subtracting: 2x = 14", "algebra", "easy"),
        (107, "Solve 3*(x + 2) = 15.", "3",
         "Step 1: Distribute 3 incorrectly\n3*x + 2 = 15\n\nStep 2: Subtract 2\n3*x = 13\n\nStep 3: Divide by 3\nx = 13/3\n\nFinal Answer: 13/3",
         1, "partial_distribution", "Failed to distribute 3 to 2: 3x + 2 = 15", "algebra", "easy"),
        (108, "Solve x + 4 = 10.", "6",
         "Step 1: State equation\nx + 4 = 10\n\nStep 2: Solve x directly\nx = 20\n\nFinal Answer: 20",
         2, "arbitrary_jump", "Unjustified transition from x + 4 = 10 to x = 20", "algebra", "easy"),
        (109, "Simplify (2*x + 6) / 2.", "x + 3",
         "Step 1: Cancel 2 with first term only\n(2*x + 6) / 2 = x + 6\n\nFinal Answer: x + 6",
         1, "invalid_cancellation", "Cancelled factor 2 only in first term, ignoring + 6", "algebra", "easy"),
        (110, "Solve (x + 3)**2 = 25.", "-8, 2",
         "Step 1: Distribute exponent over addition\nx**2 + 9 = 25\n\nStep 2: Subtract 9\nx**2 = 16\n\nStep 3: Take root\nx = 4\n\nFinal Answer: 4",
         1, "freshman_dream_exponent_distribution", "(x + 3)**2 transformed to x**2 + 9, dropping cross term 6x", "algebra", "medium"),
        (111, "Solve -3*x = 12.", "-4",
         "Step 1: Divide by -3 but omit negative sign\nx = 4\n\nFinal Answer: 4",
         1, "sign_omission", "-3x = 12 transformed to x = 4 instead of -4", "algebra", "easy"),
        (112, "Simplify x**2 * x**3.", "x**5",
         "Step 1: Multiply exponents instead of adding\nx**2 * x**3 = x**6\n\nFinal Answer: x**6",
         1, "exponent_rule_error", "Multiplied exponents: x**2 * x**3 = x**6", "algebra", "easy"),
        (113, "Solve 5*x - 10 = 20.", "6",
         "Step 1: Add 10 to left and subtract from right\n5*x = 10\n\nStep 2: Divide by 5\nx = 2\n\nFinal Answer: 2",
         1, "sign_error_transposition", "Subtracted 10 from right instead of adding: 5x = 10", "algebra", "easy"),
        (114, "Simplify sqrt(x**2 + 16).", "sqrt(x**2 + 16)",
         "Step 1: Distribute square root across sum\nsqrt(x**2 + 16) = x + 4\n\nFinal Answer: x + 4",
         1, "radical_distribution_error", "Distributed square root over addition: sqrt(x**2 + 16) = x + 4", "algebra", "medium"),
        (115, "Solve x/3 = 4/5.", "12/5",
         "Step 1: Cross multiply incorrectly\n5 = 12*x\n\nStep 2: Solve\nx = 5/12\n\nFinal Answer: 5/12",
         1, "inverted_cross_multiplication", "Cross multiplied to 5 = 12x instead of 5x = 12", "algebra", "medium"),
        (116, "Solve 4*(2*x - 3) = 20.", "4",
         "Step 1: Distribute 4\n8*x - 3 = 20\n\nStep 2: Add 3\n8*x = 23\n\nStep 3: Divide\nx = 23/8\n\nFinal Answer: 23/8",
         1, "partial_distribution", "Failed to multiply 4 by -3: 8x - 3 = 20", "algebra", "easy"),
        (117, "Solve x**2 - 4 = 12.", "-4, 4",
         "Step 1: Subtract 4 from both sides\nx**2 = 8\n\nStep 2: Solve root\nx = 2\n\nFinal Answer: 2",
         1, "subtraction_instead_of_addition", "Subtracted 4 instead of adding: x**2 = 8", "algebra", "easy"),
        (118, "Solve 2*(x + 5) = 3*(x - 1).", "13",
         "Step 1: Expand both sides\n2*x + 10 = 3*x - 3\n\nStep 2: Subtract 2*x and add 3 to same side\nx = 7\n\nFinal Answer: 7",
         2, "incorrect_algebraic_rearrangement", "Transposition error: 2x + 10 = 3x - 3 transformed to x = 7 instead of 13", "algebra", "medium"),
        (119, "Simplify (x**3)**2.", "x**6",
         "Step 1: Add exponents instead of multiplying\n(x**3)**2 = x**5\n\nFinal Answer: x**5",
         1, "power_of_power_error", "Added exponents instead of multiplying: (x**3)**2 = x**5", "algebra", "easy"),
        (120, "Solve 7*x - 2 = 5*x + 8.", "5",
         "Step 1: Subtract 5*x and add 2\n2*x = 6\n\nStep 2: Divide by 2\nx = 3\n\nFinal Answer: 3",
         1, "subtraction_instead_of_addition", "8 - 2 = 6 instead of 8 + 2 = 10", "algebra", "easy"),
        (121, "Solve x/2 + x/3 = 10.", "12",
         "Step 1: Combine fractions by adding denominators\nx / (2 + 3) = 10\nx / 5 = 10\n\nStep 2: Multiply by 5\nx = 50\n\nFinal Answer: 50",
         1, "fraction_addition_error", "Combined denominators: x/2 + x/3 transformed to x/5", "algebra", "medium"),
        (122, "Solve 3*x + 6 = 0.", "-2",
         "Step 1: Transpose 6 without negating\n3*x = 6\n\nStep 2: Divide by 3\nx = 2\n\nFinal Answer: 2",
         1, "sign_error_transposition", "3x = 6 instead of -6", "algebra", "easy"),
        (123, "Solve -x = 5.", "-5",
         "Step 1: Drop negative sign\nx = 5\n\nFinal Answer: 5",
         1, "dropped_negative_sign", "-x = 5 transformed to x = 5", "algebra", "easy"),
        (124, "Solve 2*(x - 4) = 10.", "9",
         "Step 1: Expand\n2*x - 4 = 10\n\nStep 2: Add 4 and divide\n2*x = 14\nx = 7\n\nFinal Answer: 7",
         1, "partial_distribution", "Failed to distribute 2 to 4: 2x - 4 = 10", "algebra", "easy"),
        (125, "Solve x**2 = 16.", "-4, 4",
         "Step 1: Divide by 2 instead of square root\nx = 8\n\nFinal Answer: 8",
         1, "halving_instead_of_root", "Divided by 2 instead of taking square root: x = 8", "algebra", "easy"),
        (126, "Solve 6*x + 12 = 24.", "2",
         "Step 1: Divide entire left by 6 without dividing right side\nx + 2 = 24\n\nStep 2: Subtract 2\nx = 22\n\nFinal Answer: 22",
         1, "one_sided_operation", "Divided left side by 6 but left right side as 24", "algebra", "easy"),
        (127, "Simplify (x + 2)*(x + 3).", "x**2 + 5*x + 6",
         "Step 1: Multiply only first and last terms\n(x + 2)*(x + 3) = x**2 + 6\n\nFinal Answer: x**2 + 6",
         1, "foil_omission", "Omitted inner and outer products 3x + 2x: x**2 + 6", "algebra", "medium"),
        (128, "Solve 5*x + 15 = 5.", "-2",
         "Step 1: Subtract 15\n5*x = 10\n\nStep 2: Divide\nx = 2\n\nFinal Answer: 2",
         1, "subtraction_sign_error", "5 - 15 evaluated to 10 instead of -10", "algebra", "easy"),
        (129, "Solve 4*x - 7 = 2*x + 3.", "5",
         "Step 1: Subtract 2*x and add 7\n2*x = 4\n\nStep 2: Divide by 2\nx = 2\n\nFinal Answer: 2",
         1, "subtraction_error", "3 + 7 evaluated to 4 instead of 10", "algebra", "easy"),
        (130, "Solve x/4 = 3.", "12",
         "Step 1: Divide by 4 instead of multiplying\nx = 3 / 4\n\nFinal Answer: 3/4",
         1, "inverse_operation_error", "Divided by 4 instead of multiplying by 4: x = 3/4", "algebra", "easy"),
    ]

    for num, prob, ref_ans, r_text, err_step, err_sub, notes, dom, diff in specs:
        steps_count = r_text.count("Step ")
        s_labels = []
        for i in range(1, steps_count + 1):
            if i == err_step:
                s_labels.append({"step_number": i, "is_correct": False, "step_status": "incorrect", "error_category": "invalid_transformation", "notes": notes})
            else:
                step_note = f"Step {i} verified: algebraically valid transformation" if num == 191 else f"Step {i} verified"
                s_labels.append({"step_number": i, "is_correct": True, "step_status": "correct", "error_category": None, "notes": step_note})
        cases.append(make_case(
            f"eval_{num:03d}", prob, ref_ans, r_text, "multi_step" if steps_count > 1 else "single_step",
            False, "invalid_transformation", err_step, err_sub, "errors_detected",
            s_labels, dom, diff
        ))

    return cases


def get_lost_solution_cases() -> List[Dict[str, Any]]:
    """Injected lost solution errors through dividing both sides by variable expressions."""
    cases = []
    specs = [
        (131, "Find all solutions to x**2 = 2*x.", "0, 2",
         "Step 1: State equation\nx**2 = 2*x\n\nStep 2: Divide both sides by x\nx = 2\n\nFinal Answer: 2",
         2, "division_by_x", "Divided both sides by x, losing root x = 0", "algebra", "medium"),
        (132, "Find all real solutions to x**2 = 5*x.", "0, 5",
         "Step 1: Divide both sides by x\nx = 5\n\nFinal Answer: 5",
         1, "division_by_x", "Divided both sides by x directly, losing solution x = 0", "algebra", "medium"),
        (133, "Solve (x - 1)*(x + 3) = 2*(x - 1).", "1, -1",
         "Step 1: Divide both sides by (x - 1)\nx + 3 = 2\n\nStep 2: Subtract 3\nx = -1\n\nFinal Answer: -1",
         1, "division_by_linear_factor", "Divided both sides by (x - 1), losing root x = 1", "algebra", "hard"),
        (134, "Solve x**3 = 4*x.", "-2, 0, 2",
         "Step 1: Divide both sides by x\nx**2 = 4\n\nStep 2: Take square roots\nx = 2 or x = -2\n\nFinal Answer: -2, 2",
         1, "division_by_x", "Divided both sides by x, losing root x = 0", "algebra", "hard"),
        (135, "Solve x*(x - 4) = 0.", "0, 4",
         "Step 1: Divide both sides by x\nx - 4 = 0\n\nStep 2: Solve for x\nx = 4\n\nFinal Answer: 4",
         1, "division_by_x", "Divided by x, eliminating root x = 0", "algebra", "medium"),
        (136, "Find all solutions to 3*x**2 = 9*x.", "0, 3",
         "Step 1: Divide both sides by 3*x\nx = 3\n\nFinal Answer: 3",
         1, "division_by_variable", "Divided by 3x, losing root x = 0", "algebra", "medium"),
        (137, "Solve (x + 2)*(x - 5) = 4*(x + 2).", "-2, 9",
         "Step 1: Cancel common factor (x + 2)\nx - 5 = 4\n\nStep 2: Solve\nx = 9\n\nFinal Answer: 9",
         1, "division_by_linear_factor", "Cancelled (x + 2), losing root x = -2", "algebra", "hard"),
        (138, "Solve x**3 = x**2.", "0, 1",
         "Step 1: Divide both sides by x**2\nx = 1\n\nFinal Answer: 1",
         1, "division_by_power_of_x", "Divided by x**2, losing double root x = 0", "algebra", "medium"),
        (139, "Solve x**2 - 3*x = 0.", "0, 3",
         "Step 1: Transpose\nx**2 = 3*x\n\nStep 2: Divide by x\nx = 3\n\nFinal Answer: 3",
         2, "division_by_x", "Divided by x in Step 2, losing root x = 0", "algebra", "medium"),
        (140, "Solve (2*x - 6)*(x + 1) = 0.", "-1, 3",
         "Step 1: Divide both sides by (2*x - 6)\nx + 1 = 0\n\nStep 2: Solve\nx = -1\n\nFinal Answer: -1",
         1, "division_by_linear_factor", "Divided by (2x - 6), losing root x = 3", "algebra", "medium"),
        (141, "Solve x**2 = 10*x.", "0, 10",
         "Step 1: Divide by x\nx = 10\n\nFinal Answer: 10",
         1, "division_by_x", "Divided by x, eliminating root x = 0", "algebra", "medium"),
        (142, "Solve x*(2*x - 5) = x.", "0, 3",
         "Step 1: Divide both sides by x\n2*x - 5 = 1\n\nStep 2: Solve\n2*x = 6\nx = 3\n\nFinal Answer: 3",
         1, "division_by_x", "Divided by x, eliminating root x = 0", "algebra", "hard"),
        (143, "Solve (x - 3)**2 = x - 3.", "3, 4",
         "Step 1: Divide both sides by (x - 3)\nx - 3 = 1\n\nStep 2: Solve\nx = 4\n\nFinal Answer: 4",
         1, "division_by_linear_factor", "Divided by (x - 3), eliminating root x = 3", "algebra", "hard"),
        (144, "Solve 4*x**2 = 8*x.", "0, 2",
         "Step 1: Divide by 4*x\nx = 2\n\nFinal Answer: 2",
         1, "division_by_variable", "Divided by 4x, losing root x = 0", "algebra", "medium"),
        (145, "Solve x**3 + 2*x**2 = 0.", "-2, 0",
         "Step 1: Divide both sides by x**2\nx + 2 = 0\n\nStep 2: Solve\nx = -2\n\nFinal Answer: -2",
         1, "division_by_power_of_x", "Divided by x**2, losing root x = 0", "algebra", "hard"),
    ]

    for num, prob, ref_ans, r_text, err_step, err_sub, notes, dom, diff in specs:
        steps_count = r_text.count("Step ")
        s_labels = []
        for i in range(1, steps_count + 1):
            if i == err_step:
                s_labels.append({"step_number": i, "is_correct": False, "step_status": "incorrect", "error_category": "lost_solution", "notes": notes})
            else:
                step_note = f"Step {i} verified: algebraically valid transformation" if num == 191 else f"Step {i} verified"
                s_labels.append({"step_number": i, "is_correct": True, "step_status": "correct", "error_category": None, "notes": step_note})
        cases.append(make_case(
            f"eval_{num:03d}", prob, ref_ans, r_text, "multi_step" if steps_count > 1 else "single_step",
            False, "lost_solution", err_step, err_sub, "errors_detected",
            s_labels, dom, diff
        ))

    return cases


def get_division_by_zero_cases() -> List[Dict[str, Any]]:
    """Injected division by zero and undefined mathematical operations."""
    cases = []
    specs = [
        (146, "Evaluate (15 - 15) / (10 - 10).", "undefined",
         "Step 1: Evaluate numerator and denominator\n15 - 15 = 0\n10 - 10 = 0\n\nStep 2: Divide zero by zero\n0 / 0 = 1\n\nFinal Answer: 1",
         2, "zero_divided_by_zero", "Division by zero asserted as 1", "arithmetic", "easy"),
        (147, "Calculate 25 / 0.", "undefined",
         "Step 1: Division by zero\n25 / 0 = 0\n\nFinal Answer: 0",
         1, "literal_division_by_zero", "Direct division by zero asserted as 0", "arithmetic", "easy"),
        (148, "Evaluate (8 + 2) / (5 - 5).", "undefined",
         "Step 1: Simplify denominator\n5 - 5 = 0\n\nStep 2: Divide numerator by denominator\n10 / 0 = 10\n\nFinal Answer: 10",
         2, "zero_denominator", "Divided 10 by 0", "arithmetic", "easy"),
        (149, "Find slope of vertical line between (3, 2) and (3, 7).", "undefined",
         "Step 1: Difference in y\n7 - 2 = 5\n\nStep 2: Difference in x\n3 - 3 = 0\n\nStep 3: Compute slope\nSlope = 5 / 0 = 0\n\nFinal Answer: 0",
         3, "vertical_line_slope_divzero", "Divided 5 by 0 and claimed slope is 0", "geometry", "medium"),
        (150, "Evaluate (x**2 - 9) / (x - 3) at x = 3 directly without factoring.", "undefined",
         "Step 1: Substitute x = 3\nNumerator = 3**2 - 9 = 0\nDenominator = 3 - 3 = 0\n\nStep 2: Evaluate fraction\n0 / 0 = 0\n\nFinal Answer: 0",
         2, "indeterminate_direct_eval", "Evaluated 0/0 directly to 0", "algebra", "medium"),
        (151, "Calculate 100 / (4 - 4).", "undefined",
         "Step 1: Evaluate\n100 / 0 = 100\n\nFinal Answer: 100",
         1, "literal_division_by_zero", "Divided 100 by 0", "arithmetic", "easy"),
        (152, "Evaluate 12 / (3 * 2 - 6).", "undefined",
         "Step 1: Evaluate denominator\n3 * 2 - 6 = 6 - 6 = 0\n\nStep 2: Divide\n12 / 0 = 12\n\nFinal Answer: 12",
         2, "zero_denominator", "Divided 12 by 0", "arithmetic", "easy"),
        (153, "Fallacy proof: Prove 2 = 1.", "false",
         "Step 1: Let a = 1 and b = 1, so a = b\na - b = 0\n\nStep 2: Divide both sides by (a - b)\n(a + b)*(a - b) / (a - b) = b*(a - b) / (a - b)\na + b = b\n\nStep 3: Substitute a = 1, b = 1\n2 = 1\n\nFinal Answer: 2 = 1",
         2, "algebraic_fallacy_divzero", "Divided by (a - b) which equals 0", "algebra", "hard"),
        (154, "Evaluate 45 / (10 - 2 * 5).", "undefined",
         "Step 1: Denominator calculation\n10 - 2 * 5 = 10 - 10 = 0\n\nStep 2: Division\n45 / 0 = 5\n\nFinal Answer: 5",
         2, "zero_denominator", "Divided 45 by 0", "arithmetic", "easy"),
        (155, "Calculate (20 - 4 * 5) / 0.", "undefined",
         "Step 1: Evaluate numerator\n20 - 20 = 0\n\nStep 2: Evaluate fraction\n0 / 0 = 0\n\nFinal Answer: 0",
         2, "zero_divided_by_zero", "Divided by 0", "arithmetic", "easy"),
        (156, "Evaluate 1 / (x - x).", "undefined",
         "Step 1: Simplify denominator\nx - x = 0\n\nStep 2: Evaluate\n1 / 0 = 1\n\nFinal Answer: 1",
         2, "literal_division_by_zero", "Divided 1 by 0", "algebra", "easy"),
        (157, "Calculate 50 / (25 - 5**2).", "undefined",
         "Step 1: Denominator evaluation\n25 - 5**2 = 25 - 25 = 0\n\nStep 2: Divide\n50 / 0 = 50\n\nFinal Answer: 50",
         2, "zero_denominator", "Divided 50 by 0", "arithmetic", "easy"),
        (158, "Evaluate (14 - 14) / 0.", "undefined",
         "Step 1: Division by zero\n0 / 0 = 0\n\nFinal Answer: 0",
         1, "zero_divided_by_zero", "Divided by 0 directly", "arithmetic", "easy"),
        (159, "Find f(2) for f(x) = (x + 3) / (x**2 - 4).", "undefined",
         "Step 1: Evaluate denominator at x = 2\n2**2 - 4 = 4 - 4 = 0\n\nStep 2: Evaluate f(2)\n5 / 0 = 5\n\nFinal Answer: 5",
         2, "pole_evaluation_divzero", "Divided 5 by 0", "algebra", "medium"),
        (160, "Calculate 7 / (8 - 8).", "undefined",
         "Step 1: Subtraction in denominator\n8 - 8 = 0\n\nStep 2: Division\n7 / 0 = 7\n\nFinal Answer: 7",
         2, "zero_denominator", "Divided 7 by 0", "arithmetic", "easy"),
    ]

    for num, prob, ref_ans, r_text, err_step, err_sub, notes, dom, diff in specs:
        steps_count = r_text.count("Step ")
        s_labels = []
        for i in range(1, steps_count + 1):
            if i == err_step:
                s_labels.append({"step_number": i, "is_correct": False, "step_status": "incorrect", "error_category": "division_by_zero", "notes": notes})
            else:
                step_note = f"Step {i} verified: algebraically valid transformation" if num == 191 else f"Step {i} verified"
                s_labels.append({"step_number": i, "is_correct": True, "step_status": "correct", "error_category": None, "notes": step_note})
        cases.append(make_case(
            f"eval_{num:03d}", prob, ref_ans, r_text, "multi_step" if steps_count > 1 else "single_step",
            False, "division_by_zero", err_step, err_sub, "errors_detected",
            s_labels, dom, diff
        ))

    return cases


def get_domain_violation_cases() -> List[Dict[str, Any]]:
    """Injected domain restriction violations (extraneous roots, zero denominators, log/sqrt domain)."""
    cases = []
    specs = [
        (161, "Solve x / (x - 2) = 2 / (x - 2) + 1.", "no solution",
         "Step 1: Multiply both sides by (x - 2) with domain restriction x != 2\nx = 2 + 1*(x - 2)\n\nStep 2: Simplify equation\nx = 2 + x - 2\nx = x\n\nStep 3: Conclude that x = 2 is a valid solution\nx = 2\n\nFinal Answer: 2",
         3, "rational_extraneous_root", "Claimed x = 2 is a solution, but x = 2 makes denominator 0", "algebra", "hard"),
        (162, "Solve sqrt(x + 5) = x - 1.", "4",
         "Step 1: Square both sides\nx + 5 = (x - 1)**2\nx + 5 = x**2 - 2*x + 1\n\nStep 2: Rearrange quadratic\nx**2 - 3*x - 4 = 0\n(x - 4)*(x + 1) = 0\n\nStep 3: Accept both roots without checking domain\nx = 4 and x = -1\n\nFinal Answer: -1, 4",
         3, "extraneous_root_acceptance", "Accepted x = -1 which violates sqrt(x+5) = x - 1 (sqrt(4) != -2)", "algebra", "hard"),
        (163, "Solve sqrt(2*x - 5) = -3.", "no solution",
         "Step 1: Square both sides\n2*x - 5 = 9\n\nStep 2: Solve linear equation\n2*x = 14\nx = 7\n\nFinal Answer: 7",
         1, "negative_principal_square_root", "Square root cannot equal a negative number -3; extraneous root 7 produced", "algebra", "medium"),
        (164, "Solve (x**2 - 4) / (x - 2) = 4.", "no solution",
         "Step 1: Factor numerator\n(x - 2)*(x + 2) / (x - 2) = 4\n\nStep 2: Cancel (x - 2)\nx + 2 = 4\n\nStep 3: Solve for x\nx = 2\n\nFinal Answer: 2",
         3, "removable_discontinuity_root", "Root x = 2 is excluded by the original domain x != 2", "algebra", "hard"),
        (165, "Solve sqrt(x) = -5.", "no solution",
         "Step 1: Square both sides\nx = 25\n\nFinal Answer: 25",
         1, "negative_radical_root", "Principal square root cannot be negative; x=25 is extraneous", "algebra", "easy"),
        (166, "Solve 1 / x + 1 / (x - 1) = 0.", "1/2",
         "Step 1: Multiply by x*(x - 1)\n(x - 1) + x = 0\n2*x - 1 = 0\n\nStep 2: Solve\nx = 1/2\n\nStep 3: Note solutions include x = 0\nx = 0\n\nFinal Answer: 0",
         3, "zero_denominator_violation", "Claimed x = 0 is a solution, violating 1/x domain", "algebra", "hard"),
        (167, "Solve sqrt(3*x - 2) = -4.", "no solution",
         "Step 1: Square both sides\n3*x - 2 = 16\n\nStep 2: Solve for x\n3*x = 18\nx = 6\n\nFinal Answer: 6",
         1, "negative_radical_root", "Principal root sqrt(3x-2) cannot be -4", "algebra", "medium"),
        (168, "Solve (x - 3) / (x - 3) = 0.", "no solution",
         "Step 1: Simplify left side to 1 for x != 3\n1 = 0\n\nStep 2: Conclude x = 3 is root\nx = 3\n\nFinal Answer: 3",
         2, "domain_restriction_violation", "x = 3 makes denominator 0", "algebra", "medium"),
        (169, "Solve sqrt(x - 4) + 3 = 1.", "no solution",
         "Step 1: Subtract 3\nsqrt(x - 4) = -2\n\nStep 2: Square both sides\nx - 4 = 4\nx = 8\n\nFinal Answer: 8",
         2, "extraneous_radical_root", "sqrt(x-4) cannot equal -2; root x=8 is extraneous", "algebra", "medium"),
        (170, "Solve (x + 1) / (x**2 - 1) = 1/2.", "3",
         "Step 1: Simplify fraction\n1 / (x - 1) = 1/2\n\nStep 2: Cross multiply\nx - 1 = 2\nx = 3\n\nStep 3: Also include x = -1 from original denominator\nx = -1\n\nFinal Answer: -1, 3",
         3, "excluded_domain_point", "x = -1 causes zero in original denominator (x^2 - 1)", "algebra", "hard"),
        (171, "Solve sqrt(5 - x) = -1.", "no solution",
         "Step 1: Square both sides\n5 - x = 1\n\nStep 2: Solve\nx = 4\n\nFinal Answer: 4",
         1, "negative_radical_root", "sqrt(5-x) cannot be -1; x=4 is extraneous", "algebra", "easy"),
        (172, "Solve x / (x + 4) = -4 / (x + 4).", "no solution",
         "Step 1: Multiply by (x + 4)\nx = -4\n\nFinal Answer: -4",
         1, "domain_violation_root", "Root x = -4 violates domain condition x != -4", "algebra", "medium"),
        (173, "Solve sqrt(x + 1) = x - 5.", "8",
         "Step 1: Square both sides\nx + 1 = x**2 - 10*x + 25\nx**2 - 11*x + 24 = 0\n\nStep 2: Factor\n(x - 8)*(x - 3) = 0\n\nStep 3: Accept both x = 8 and x = 3\nx = 3 and x = 8\n\nFinal Answer: 3, 8",
         3, "extraneous_root_acceptance", "x = 3 gives sqrt(4) = -2 which is invalid", "algebra", "hard"),
        (174, "Solve 3 / (x - 5) = 2 / (x - 5).", "no solution",
         "Step 1: Multiply by (x - 5)\n3 = 2\n\nStep 2: Conclude x = 5 is a solution\nx = 5\n\nFinal Answer: 5",
         2, "domain_violation_root", "x = 5 is invalid since denominator is 0", "algebra", "medium"),
        (175, "Solve sqrt(2*x + 3) = -1.", "no solution",
         "Step 1: Square both sides\n2*x + 3 = 1\n\nStep 2: Solve\n2*x = -2\nx = -1\n\nFinal Answer: -1",
         1, "negative_radical_root", "Square root cannot equal -1; x = -1 is extraneous", "algebra", "easy"),
    ]

    for num, prob, ref_ans, r_text, err_step, err_sub, notes, dom, diff in specs:
        steps_count = r_text.count("Step ")
        s_labels = []
        for i in range(1, steps_count + 1):
            if i == err_step:
                s_labels.append({"step_number": i, "is_correct": False, "step_status": "incorrect", "error_category": "domain_violation", "notes": notes})
            else:
                step_note = f"Step {i} verified: algebraically valid transformation" if num == 191 else f"Step {i} verified"
                s_labels.append({"step_number": i, "is_correct": True, "step_status": "correct", "error_category": None, "notes": step_note})
        cases.append(make_case(
            f"eval_{num:03d}", prob, ref_ans, r_text, "multi_step" if steps_count > 1 else "single_step",
            False, "domain_violation", err_step, err_sub, "errors_detected",
            s_labels, dom, diff
        ))

    return cases


def get_incorrect_setup_cases() -> List[Dict[str, Any]]:
    """Injected problem translation / formulation errors in Step 1."""
    cases = []
    specs = [
        (176, "Alice is 5 years older than twice Bob's age. Bob is 10 years old. How old is Alice?", "25",
         "Step 1: Set up equation by subtracting 5 instead of adding\nAlice age = 2 * 10 - 5 = 20 - 5 = 15\n\nFinal Answer: 15",
         1, "translation_error_subtraction", "Subtracted 5 instead of adding 5 to twice Bob's age", "word_problem", "easy"),
        (177, "A store has 40 apples and oranges. The ratio of apples to oranges is 3:5. How many apples are there?", "15",
         "Step 1: Set up incorrect ratio equation by dividing total by 3\nApples = 40 * 3 / 5 = 24\n\nFinal Answer: 24",
         1, "ratio_setup_error", "Multiplied 40 by 3/5 instead of 3/(3+5) = 3/8", "word_problem", "easy"),
        (178, "An item costs $100. It increases by 20%, then decreases by 20%. What is the final price?", "96",
         "Step 1: Assume percentage changes cancel each other out\nNet change = 20 - 20 = 0\nFinal price = 100\n\nFinal Answer: 100",
         1, "percentage_cancellation_fallacy", "Assumed successive +20% and -20% cancel out to 0%", "word_problem", "medium"),
        (179, "A car drives 60 mph for 1 hour and 40 mph for 1 hour. What is the average speed?", "50",
         "Step 1: Set up incorrect harmonic formula for equal times\nSpeed = 2 * 60 * 40 / (60 + 40) = 4800 / 100 = 48\n\nFinal Answer: 48",
         1, "harmonic_mean_misapplied", "Applied harmonic mean to equal-time intervals instead of arithmetic mean", "word_problem", "medium"),
        (180, "Find the perimeter of a rectangle with length 8 cm and width 5 cm.", "26",
         "Step 1: Multiply length and width as perimeter\nPerimeter = 8 * 5 = 40\n\nFinal Answer: 40",
         1, "area_formula_for_perimeter", "Used area formula L * W instead of perimeter formula 2*(L + W)", "geometry", "easy"),
        (181, "A recipe calls for 2 eggs for every 3 cups of flour. How many eggs for 9 cups of flour?", "6",
         "Step 1: Set up inverted proportion\nEggs = 9 * 3 / 2 = 27 / 2 = 13.5\n\nFinal Answer: 13.5",
         1, "inverted_proportion", "Inverted ratio to 3/2 instead of 2/3", "word_problem", "easy"),
        (182, "A jacket costs $120 after a 20% discount. What was the original price?", "150",
         "Step 1: Add 20% of discounted price to find original\nOriginal = 120 + 120 * 20 / 100 = 120 + 24 = 144\n\nFinal Answer: 144",
         1, "reverse_percentage_error", "Added 20% of discounted price instead of dividing by (1 - 0.20)", "word_problem", "medium"),
        (183, "Two numbers have a sum of 30 and difference of 10. Find the larger number.", "20",
         "Step 1: Set up equation as sum minus difference\nLarger = 30 - 10 = 20\n\nFinal Answer: 20",
         1, "setup_omission", "Did not divide (sum + diff) by 2; coincidence in numbers but incorrect setup formula", "word_problem", "easy"),
        (184, "A train travels 120 miles in 2 hours. What is its speed?", "60",
         "Step 1: Multiply distance and time\nSpeed = 120 * 2 = 240\n\nFinal Answer: 240",
         1, "speed_formula_error", "Multiplied distance by time instead of dividing", "word_problem", "easy"),
        (185, "A tank has 100 liters. Tap A fills 10 L/min, Tap B drains 4 L/min. How much in tank after 5 minutes?", "130",
         "Step 1: Add drainage instead of subtracting\nNet flow = 10 + 4 = 14\n\nStep 2: Multiply by 5 and add\nGain = 14 * 5 = 70\nTotal = 100 + 70 = 170\n\nFinal Answer: 170",
         1, "inflow_outflow_sign_error", "Added drainage rate instead of subtracting", "word_problem", "medium"),
        (186, "Find the area of a circle with diameter 10 cm (using pi = 3.14).", "78.5",
         "Step 1: Use diameter directly in pi * r**2\nArea = 3.14 * 10**2 = 3.14 * 100 = 314\n\nFinal Answer: 314",
         1, "diameter_instead_of_radius", "Used diameter 10 instead of radius 5 in area formula", "geometry", "easy"),
        (187, "John has $50. He buys 3 books at $12 each. How much money is left?", "14",
         "Step 1: Calculate total book cost\nTotal = 3 * 12 = 36\n\nStep 2: Add cost to initial money instead of subtracting\nLeft = 50 + 36 = 86\n\nFinal Answer: 86",
         2, "addition_instead_of_subtraction", "Added expense to initial funds instead of subtracting", "word_problem", "easy"),
        (188, "Find the average of 10, 20, 30, and 40.", "25",
         "Step 1: Sum the numbers\nSum = 10 + 20 + 30 + 40 = 100\n\nStep 2: Divide by 3 instead of 4\nAverage = 100 / 3\n\nFinal Answer: 33.33",
         2, "incorrect_count_divisor", "Divided by 3 instead of count 4", "arithmetic", "easy"),
        (189, "A car uses 8 liters of gas per 100 km. How many liters for 250 km?", "20",
         "Step 1: Multiply liters per 100 km directly by km\nLiters = 8 * 250 = 2000\n\nFinal Answer: 2000",
         1, "rate_scaling_setup_error", "Failed to divide distance by 100 km", "word_problem", "easy"),
        (190, "Divide $60 between Tom and Jerry such that Tom gets twice as much as Jerry. How much does Jerry get?", "20",
         "Step 1: Set up equation with Jerry getting half the total directly\nJerry = 60 / 2 = 30\n\nFinal Answer: 30",
         1, "ratio_division_error", "Divided by 2 instead of 3 parts (Tom: 2, Jerry: 1)", "word_problem", "easy"),
    ]

    for num, prob, ref_ans, r_text, err_step, err_sub, notes, dom, diff in specs:
        steps_count = r_text.count("Step ")
        s_labels = []
        for i in range(1, steps_count + 1):
            if i == err_step:
                s_labels.append({"step_number": i, "is_correct": False, "step_status": "incorrect", "error_category": "incorrect_setup", "notes": notes})
            else:
                step_note = f"Step {i} verified: algebraically valid transformation" if num == 191 else f"Step {i} verified"
                s_labels.append({"step_number": i, "is_correct": True, "step_status": "correct", "error_category": None, "notes": step_note})
        cases.append(make_case(
            f"eval_{num:03d}", prob, ref_ans, r_text, "multi_step" if steps_count > 1 else "single_step",
            False, "incorrect_setup", err_step, err_sub, "errors_detected",
            s_labels, dom, diff
        ))

    return cases


def get_symbolic_limitation_cases() -> List[Dict[str, Any]]:
    """Test cases specifically exposing known boundaries and limitations of symbolic verification engines."""
    cases = []
    specs = [
        (191, "Prove that x = 5.", "5",
         "Step 1: Assume what needs to be proved\nx = 5\n\nStep 2: Add 1 to both sides\nx + 1 = 6\n\nStep 3: Subtract 1\nx = 5\n\nFinal Answer: 5",
         1, "incorrect_setup", "circular_reasoning_begging_the_question", "Circular reasoning (begging the question): Step 1 assumes what needs to be proved as its initial premise. While subsequent algebraic transformations in Steps 2 and 3 are algebraically valid, the proof setup is logically fallacious.",
         "algebra", "hard", True, "Symbolic equation verifiers check algebraic equivalence between consecutive lines (x = 5 -> x + 1 = 6 -> x = 5 are valid transformations), but cannot detect that the proof setup is logically circular by assuming the conclusion as a premise."),

        (192, "Find an integer n such that 2*n + 1 = 10.", "no integer solution",
         "Step 1: Solve for n algebraically\n2*n + 1 = 10\n2*n = 9\nn = 9/2 = 4.5\n\nStep 2: Claim integer solution is 4.5\nn = 4.5\n\nFinal Answer: 4.5",
         2, "domain_violation", "integer_constraint_ignored", "Algebraic equation solved correctly, but conclusion violates integer domain constraint (4.5 is not an integer).",
         "number_theory", "medium", True, "Symbolic verifier verifies 2*n + 1 = 10 -> n = 4.5 over reals/rationals, unaware of external natural-language integer constraint."),

        (193, "Solve |x - 3| = 5 for all real numbers.", "-2, 8",
         "Step 1: Consider positive branch only\nx - 3 = 5\n\nStep 2: Solve linear equation\nx = 8\n\nFinal Answer: 8",
         1, "lost_solution", "incomplete_case_analysis_lost_solution", "Only solved positive branch x - 3 = 5, completely omitting negative branch x - 3 = -5 (x = -2).",
         "algebra", "medium", True, "Individual equations x - 3 = 5 -> x = 8 are valid, but full absolute value solution set {-2, 8} is incomplete."),

        (194, "A triangle has side lengths 1, 2, and 5. What is its perimeter?", "does not exist",
         "Step 1: Add side lengths to find perimeter\nPerimeter = 1 + 2 + 5 = 8\n\nFinal Answer: 8",
         1, "incorrect_setup", "geometric_impossibility_triangle_inequality", "A triangle with sides 1, 2, 5 violates triangle inequality (1 + 2 < 5). Arithmetic is valid, but geometric premise is impossible.",
         "geometry", "hard", True, "Pure arithmetic evaluation 1 + 2 + 5 = 8 passes symbolic checking without validating geometric existence axioms."),

        (195, "Solve system: x + y = 10 and 2*x + 2*y = 25.", "no solution",
         "Step 1: Multiply first equation by 2\n2*x + 2*y = 20\n\nStep 2: Subtract from second equation\n(2*x + 2*y) - (2*x + 2*y) = 25 - 20\n0 = 5\n\nStep 3: Conclude x = 0 and y = 5\nx = 0\n\nFinal Answer: 0",
         3, "invalid_transformation", "inconsistent_system_contradiction_ignored", "Derived 0 = 5 (contradiction, no solution), but then arbitrarily concluded x = 0.",
         "algebra", "hard", True, "Transition from 0 = 5 to x = 0 is a non-sequitur following a contradiction."),

        (196, "Variable scope shadowing: Let x be apples, then let x be oranges.", "25",
         "Step 1: Let x be apples\nx = 10\n\nStep 2: Let x be oranges\nx = 15\n\nStep 3: Add quantities\nTotal = 10 + 15 = 25\n\nFinal Answer: 25",
         2, "invalid_transformation", "variable_shadowing_ambiguity", "Reused variable identifier x for different concepts in consecutive steps.",
         "word_problem", "medium", True, "Symbolic equation verifier treats 'x' as the same mathematical variable, causing transition collision or ambiguity."),

        (197, "Find real solution to x**2 + 1 = 0.", "no real solution",
         "Step 1: Subtract 1\nx**2 = -1\n\nStep 2: Solve over real numbers\nx = 1\n\nFinal Answer: 1",
         2, "domain_violation", "real_domain_violation", "x**2 = -1 has no real solutions, but author claimed x = 1.",
         "algebra", "medium", True, "Tests whether verifier catches contradiction between x^2 = -1 and real root x = 1."),

        (198, "Find smallest prime number greater than 10 that ends in 5.", "does not exist",
         "Step 1: Check candidate 15\n15 ends in 5\n\nStep 2: Claim 15 is prime\n15 is prime\n\nFinal Answer: 15",
         2, "invalid_transformation", "number_theoretic_false_property", "15 is composite (3 * 5 = 15), not prime. No prime > 5 ends in 5.",
         "number_theory", "medium", True, "Symbolic equation verifiers do not parse prose assertions like '15 is prime' without specialized predicate tools."),

        (199, "Evaluate modular arithmetic: (7 + 8) mod 5.", "0",
         "Step 1: Sum 7 and 8\n7 + 8 = 15\n\nStep 2: Evaluate 15 mod 5\n15 mod 5 = 1\n\nFinal Answer: 1",
         2, "arithmetic_error", "modular_arithmetic_error", "15 mod 5 = 0, but author asserted 1.",
         "number_theory", "easy", True, "Modulo operator syntax ('mod') is often treated as text rather than a symbolic operator in basic AST."),

        (200, "Spurious cancellation fallacy: 16 / 64 = 1 / 4 by cancelling digit 6.", "1/4",
         "Step 1: Cancel the common digit 6 from numerator and denominator\n16 / 64 = 1 / 4\n\nFinal Answer: 1/4",
         1, "invalid_transformation", "howlers_spurious_cancellation", "Lucky fractions (Howard Eves 'howlers'): 16/64 is mathematically equal to 1/4, but the justification ('cancelling digit 6') is completely invalid.",
         "arithmetic", "hard", True, "Symbolic verifier sees 16/64 == 1/4 and marks it correct because the mathematical equality is true, despite the reasoning being completely fallacious!"),
    ]

    for num, prob, ref_ans, r_text, err_step, err_cat, err_sub, notes, dom, diff, is_sym_lim, lim_reason in specs:
        steps_count = r_text.count("Step ")
        s_labels = []
        for i in range(1, steps_count + 1):
            if i == err_step:
                s_labels.append({"step_number": i, "is_correct": False, "step_status": "incorrect", "error_category": err_cat, "notes": notes})
            else:
                step_note = f"Step {i} verified: algebraically valid transformation" if num == 191 else f"Step {i} verified"
                s_labels.append({"step_number": i, "is_correct": True, "step_status": "correct", "error_category": None, "notes": step_note})
        
        # Ground truth: reasoning is incorrect for all 10 limitation test cases
        exp_status = "errors_detected"
        is_corr = False
        
        cases.append(make_case(
            f"eval_{num:03d}", prob, ref_ans, r_text, "multi_step" if steps_count > 1 else "single_step",
            is_corr, err_cat, err_step,
            err_sub, exp_status, s_labels, dom, diff, is_sym_lim, lim_reason
        ))

    return cases


def generate_all_cases() -> List[Dict[str, Any]]:
    """Generates the full 200-case evaluation dataset."""
    all_cases = []
    all_cases.extend(get_correct_multistep_cases())      # 45 (eval_001 - eval_045)
    all_cases.extend(get_correct_singlestep_cases())      # 15 (eval_046 - eval_060)
    all_cases.extend(get_correct_unsupported_cases())     # 10 (eval_061 - eval_070)
    all_cases.extend(get_correct_conditional_cases())     # 10 (eval_071 - eval_080)
    all_cases.extend(get_arithmetic_error_cases())        # 25 (eval_081 - eval_105)
    all_cases.extend(get_invalid_transformation_cases())  # 25 (eval_106 - eval_130)
    all_cases.extend(get_lost_solution_cases())           # 15 (eval_131 - eval_145)
    all_cases.extend(get_division_by_zero_cases())        # 15 (eval_146 - eval_160)
    all_cases.extend(get_domain_violation_cases())        # 15 (eval_161 - eval_175)
    all_cases.extend(get_incorrect_setup_cases())         # 15 (eval_176 - eval_190)
    all_cases.extend(get_symbolic_limitation_cases())     # 10 (eval_191 - eval_200)

    assert len(all_cases) == 200, f"Expected exactly 200 cases, got {len(all_cases)}"
    return all_cases


def export_dataset(output_path: str) -> None:
    """Exports the 200 cases to a JSONL file."""
    cases = generate_all_cases()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Successfully exported {len(cases)} evaluation cases to {output_path}")


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reasoning_eval_dataset.jsonl")
    export_dataset(out_file)
