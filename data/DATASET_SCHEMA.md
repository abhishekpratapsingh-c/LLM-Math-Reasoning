# Evaluation Dataset Schema & Documentation
**Phase 4 – Objective 26: Labeled Evaluation Dataset**

## 1. Overview

The evaluation dataset (`data/reasoning_eval_dataset.jsonl`) contains **200 manually curated and labeled mathematical reasoning test cases**. It is specifically designed to benchmark and evaluate the hybrid LLM + SymPy mathematical reasoning error-detection pipeline across varied problem types, reasoning styles, and error categories.

- **Total Test Cases**: 200
- **Format**: JSON Lines (`.jsonl`), UTF-8 encoded
- **Labeling Source**: Deterministic, expert-authored mathematical ground truth (independent of the error detection pipeline)
- **Zero Paid APIs**: Fully reproducible and offline

---

## 2. Dataset Schema Specification

Each line in `data/reasoning_eval_dataset.jsonl` is a self-contained JSON object conforming to the following structure:

```json
{
  "id": "eval_001",
  "problem": "A bakery sells 4 cakes for $15 each and 6 pies for $10 each. What is the total revenue?",
  "reference_final_answer": "120",
  "reasoning_text": "Step 1: Calculate revenue from cakes\nRevenue from cakes = 4 * 15 = 60\n\nStep 2: Calculate revenue from pies\nRevenue from pies = 6 * 10 = 60\n\nStep 3: Calculate total revenue\nTotal revenue = 60 + 60 = 120\n\nFinal Answer: 120",
  "reasoning_type": "multi_step",
  "ground_truth": {
    "is_correct": true,
    "error_category": "none",
    "error_step": null,
    "error_sub_type": "correct_reasoning",
    "expected_report_status": "verified",
    "step_labels": [
      {
        "step_number": 1,
        "is_correct": true,
        "step_status": "correct",
        "error_category": null,
        "notes": "Correct cake revenue"
      },
      {
        "step_number": 2,
        "is_correct": true,
        "step_status": "correct",
        "error_category": null,
        "notes": "Correct pie revenue"
      },
      {
        "step_number": 3,
        "is_correct": true,
        "step_status": "correct",
        "error_category": null,
        "notes": "Correct sum"
      }
    ]
  },
  "metadata": {
    "domain": "word_problem",
    "difficulty": "easy",
    "symbolic_limitation": false,
    "limitation_reason": null
  }
}
```

### Field Definitions

| Field Path | Type | Allowed Values / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Regex `^eval_\d{3}$` (`eval_001` .. `eval_200`) | Unique sequential identifier for the test case. |
| `problem` | `string` | Non-empty string | Natural-language problem prompt or mathematical question. |
| `reference_final_answer` | `string` | Non-empty string | Ground-truth mathematical solution to the problem. |
| `reasoning_text` | `string` | Contains `Step N:` and `Final Answer:` | Raw textual reasoning trace representing simulated LLM output. |
| `reasoning_type` | `string` | `"single_step"`, `"multi_step"` | Whether reasoning contains 1 step or $\ge 2$ steps. |
| `ground_truth.is_correct` | `boolean` | `true`, `false` | Overall mathematical correctness of the reasoning trace. |
| `ground_truth.error_category` | `string` | See Section 3 Taxonomy | High-level error classification category. |
| `ground_truth.error_step` | `integer` or `null` | $1 \le \text{step} \le \text{total\_steps}$ (or `null`) | 1-based index of the first step with an injected error. |
| `ground_truth.error_sub_type`| `string` | Descriptive string | Specific mathematical phenomenon / mistake type. |
| `ground_truth.expected_report_status` | `string` | `"verified"`, `"errors_detected"`, `"warnings"`, `"partially_verified"` | Expected top-level status in the Phase 3 verification report. |
| `ground_truth.step_labels` | `list[object]` | Length matches extracted step count | Per-step ground truth labels. |
| `ground_truth.step_labels[i].step_number` | `integer` | $1, 2, \dots$ | Step number. |
| `ground_truth.step_labels[i].is_correct` | `boolean` | `true`, `false` | Correctness of this specific step. |
| `ground_truth.step_labels[i].step_status` | `string` | `"correct"`, `"incorrect"`, `"unsupported"`, `"warning"` | Per-step status label. |
| `ground_truth.step_labels[i].error_category` | `string` or `null`| Category name or `null` | Error category if step is incorrect. |
| `ground_truth.step_labels[i].notes` | `string` | Descriptive string | Explanation of why the step is correct or incorrect. |
| `metadata.domain` | `string` | `"arithmetic"`, `"algebra"`, `"word_problem"`, `"geometry"`, `"number_theory"` | Mathematical subject domain. |
| `metadata.difficulty` | `string` | `"easy"`, `"medium"`, `"hard"` | Relative problem complexity. |
| `metadata.symbolic_limitation` | `boolean` | `true`, `false` | True if the case targets a known boundary of symbolic checkers. |
| `metadata.limitation_reason` | `string` or `null`| Non-empty string if limitation | Detailed explanation of why symbolic tools face challenges here. |

---

## 3. Taxonomy of Error Categories

1. **`none`** (60 cases, 30.0%): Completely sound mathematical reasoning with no assertion or transition errors.
2. **`invalid_transformation`** (29 cases, 14.5%): Consecutive algebraic step does not mathematically follow from the preceding step (sign error in transposition, partial distribution, false cancellation, illegal exponent/radical distribution, spurious cancellation howlers). Note: valid algebraic transitions in circular proofs are not classified as invalid transformations.
3. **`arithmetic_error`** (26 cases, 13.0%): Calculation mistake in numerical operations (addition, subtraction, multiplication, division, fractions, modular arithmetic, precedence).
4. **`domain_violation`** (17 cases, 8.5%): Producing or accepting solutions that violate mathematical domain constraints (extraneous roots from radical equations, roots that zero original denominators, asserting negative values for principal square roots, ignoring natural-language integer domain constraints, or asserting real roots for equations with no real roots).
5. **`lost_solution`** (16 cases, 8.0%): Dividing both sides of an equation by a variable factor (e.g. $x$, $(x-2)$, $x^2$) without accounting for zero, or performing incomplete branch/case analysis (e.g. absolute value equations), thereby dropping valid roots.
6. **`incorrect_setup`** (17 cases, 8.5%): First step mistranslates the natural-language word problem into an incorrect initial mathematical equation, sets up an impossible geometric premise (e.g. sides violating the Triangle Inequality), or sets up an assumption that assumes the conclusion to be proved (circular reasoning / begging the question).
7. **`division_by_zero`** (15 cases, 7.5%): Explicit or implicit division by zero (dividing by zero literal, evaluating a rational expression at its pole, or classic algebraic fallacies where $a - b = 0$).
8. **`unsupported`** (10 cases, 5.0%): Reasoning contains valid conceptual or natural-language definitions (e.g. `Profit = Revenue - Cost`) that symbolic solvers cannot verify directly as algebraic identities.
9. **`conditional_warning`** (10 cases, 5.0%): Valid one-way algebraic transformations (e.g. squaring both sides of $\sqrt{f(x)} = c$) that preserve true solutions but introduce the possibility of extraneous roots requiring verification.

---

## 4. Dataset Composition & Statistics

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Test Cases** | **200** | **100.0%** |
| **Correct Reasoning Cases** | **80** | **40.0%** |
| - *Multi-step verified (`none`)* | 45 | 22.5% |
| - *Single-step verified (`none`)* | 15 | 7.5% |
| - *Conceptual / Unsupported formulas (`unsupported`)* | 10 | 5.0% |
| - *Conditional transformations (`conditional_warning`)* | 10 | 5.0% |
| **Incorrect Reasoning Cases** | **120** | **60.0%** |
| - *Invalid algebraic transformations* | 29 | 14.5% |
| - *Arithmetic calculation errors* | 26 | 13.0% |
| - *Domain restriction violations* | 17 | 8.5% |
| - *Incorrect problem setups (geometric impossibility & circular proof setup)* | 17 | 8.5% |
| - *Lost solutions (division by variable / incomplete branches)* | 16 | 8.0% |
| - *Division by zero* | 15 | 7.5% |

### Expected Final Report Status Mapping

| Status | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **`errors_detected`** | 120 | 60.0% | All 120 incorrect cases where reasoning fails |
| **`verified`** | 60 | 30.0% | Sound reasoning cases with `error_category: "none"` |
| **`partially_verified`** | 10 | 5.0% | Sound cases containing unsupported semantic definitions |
| **`warnings`** | 10 | 5.0% | Sound cases with conditional / branching transformations |

### Domain Breakdown
- **Algebra**: 96 cases (48.0%)
- **Arithmetic**: 47 cases (23.5%)
- **Word Problems**: 38 cases (19.0%)
- **Geometry**: 14 cases (7.0%)
- **Number Theory**: 5 cases (2.5%)

### Reasoning Depth Breakdown
- **Multi-step Reasoning**: 134 cases (67.0%)
- **Single-step Reasoning**: 66 cases (33.0%)

---

## 5. Limitations of Symbolic Verification Covered in the Dataset

The dataset includes **10 designated boundary cases** (`eval_191` to `eval_200`, `symbolic_limitation: true`) exposing fundamental challenges for pure symbolic solvers:

1. **Circular Proof / Begging the Question (`eval_191`)**: A proof of $x = 5$ that begins by assuming $x = 5$ as a premise in Step 1. Both subsequent algebraic transitions ($x = 5 \implies x + 1 = 6$ in Step 2, and $x + 1 = 6 \implies x = 5$ in Step 3) are algebraically valid transformations. However, the initial proof setup assumes the conclusion to be proved (begging the question), making the logical proof fallacious. It is categorized as `incorrect_setup` (sub-type: `circular_reasoning_begging_the_question`), distinguishing valid algebraic transformations from flawed deductive proof structure.
2. **External Domain Constraints (`eval_192`)**: Solving $2n + 1 = 10 \implies n = 4.5$ over $\mathbb{R}$ is symbolically sound, but violates an external natural-language constraint ("find an integer $n$").
3. **Incomplete Case Analysis (`eval_193`)**: Solving $|x - 3| = 5$ by considering only the positive branch $x - 3 = 5 \implies x = 8$. The symbolic step is valid, but the solution set is incomplete (missing $x = -2$).
4. **Impossible Geometric Premises (`eval_194`)**: Calculating the perimeter of a triangle with sides $1, 2, 5$ as $1 + 2 + 5 = 8$. The arithmetic is symbolically correct, but the triangle cannot physically exist (violates the Triangle Inequality).
5. **Ignoring Contradictions (`eval_195`)**: Deriving $0 = 5$ from an inconsistent system, and subsequently declaring $x = 0$.
6. **Variable Scope Shadowing (`eval_196`)**: Reusing variable $x$ for "apples" in Step 1 and then "oranges" in Step 2.
7. **Real vs. Complex Domain (`eval_197`)**: Concluding that $x^2 + 1 = 0 \implies x = 1$ in the real domain.
8. **Semantic Predicates (`eval_198`)**: Reasoning about non-algebraic predicates such as "15 is prime".
9. **Modular Arithmetic Text Syntax (`eval_199`)**: Natural language modulo syntax like `15 mod 5 = 1`.
10. **Spurious Cancellation Howlers (`eval_200`)**: Cancelling the digit 6 in $16 / 64 = 1 / 4$. The mathematical equality $16/64 = 1/4$ is true ($0.25 = 0.25$), so symbolic equality checks pass, despite the justification being an arithmetic howler.
