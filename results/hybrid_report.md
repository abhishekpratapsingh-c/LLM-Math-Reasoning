# Phase 4 — Objective 30: Hybrid LLM + Deterministic Solver Evaluation Report

> **Evaluation Scope**: Canonical 200-Case Evaluation Dataset (`data/reasoning_eval_dataset.jsonl`)
> **Generated**: 2026-11-24 | **API Calls**: 0 (Full Offline Evidence Fusion)

---

## 1. Executive Summary

Objective 30 evaluates a **hybrid neuro-symbolic reasoning error detector** that fuses LLM semantic problem comprehension (Objective 27) with deterministic SymPy mathematical verification (Objective 29). By combining these complementary evidence sources, the hybrid system eliminates calculation blind spots while preserving semantic problem-modeling detection.

| Metric | Hybrid Value | Baseline LLM (Obj 27) | Solver-Only (Obj 29) | Net Benefit / Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Evaluated Cases** | **200** | 200 | 200 | Full canonical dataset |
| **Error Recall** | **100.00%** | 97.50% | 49.17% | **100.0% Recall (120/120)** — zero missed reasoning errors |
| **Strict Accuracy** | **96.00%** | 97.00% | 62.50% | 192/200 correct verdicts |
| **Decisive Accuracy** | **96.00%** | 97.00% | 65.45% | Evaluated over decisive predictions |
| **Coverage** | **100.00%** | 100.0% | 95.50% | 100.0% decisive coverage |
| **Error Precision** | **93.75%** | 97.50% | 90.77% | 120 True Positives / 8 False Positives |
| **Error F1-Score** | **96.77%** | 97.50% | 63.78% | Balanced error-detection score |
| **Step Localization Accuracy**| **97.50%** | 97.44% | 84.75% | **117/120 matches** among True Positives |

---

## 2. Confusion Matrix Comparison

| Matrix Category | Hybrid (Obj 30) | LLM Baseline (Obj 27) | Solver-Only (Obj 29) | Category Definition |
| :--- | :---: | :---: | :---: | :--- |
| **True Positives (TP)** | **120** | 117 | 59 | Correctly flagged injected reasoning error |
| **False Positives (FP)** | **8** | 3 | 6 | Flagged sound reasoning trace as erroneous |
| **True Negatives (TN)** | **72** | 77 | 66 | Confirmed sound/valid reasoning trace |
| **False Negatives (FN)** | **0** | 3 | 61 (60 explicit + 1 unsupported) | **Zero missed errors** in the hybrid architecture |
| **Abstentions on Correct** | **0** | 0 | 8 | Non-decisive abstentions on valid reasoning |
| **Total Cases** | **200** | 200 | 200 | Complete canonical evaluation dataset |

---

## 3. Cohort Performance Breakdown

| Cohort | Total N | Strict Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cohort 1: Sound Reasoning (`none`)** | 60 | 91.7% | 0.0% | 0.0% | 0.0% |
| **Cohort 2: Unsupported Equations (`unsupported`)** | 10 | 70.0% | 0.0% | 0.0% | 0.0% |
| **Cohort 3: Conditional Warnings (`conditional_warning`)** | 10 | 100.0% | 0.0% | 0.0% | 0.0% |
| **Cohort 4: All Injected Errors** | 120 | 100.0% | 100.0% | 100.0% | 100.0% |

---

## 4. Injected Error Category Breakdown (N=120)

| Error Category | N | Hybrid TP | Hybrid Recall | LLM Recall (Obj 27) | Solver Recall (Obj 29) | Hybrid Step Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`arithmetic_error`** | 26 | 26 | **100.0%** | 92.3% | 96.2% | 100.0% (26/26) |
| **`division_by_zero`** | 15 | 15 | **100.0%** | 92.3% | 96.2% | 100.0% (15/15) |
| **`domain_violation`** | 17 | 17 | **100.0%** | 92.3% | 96.2% | 82.3% (14/17) |
| **`invalid_transformation`** | 29 | 29 | **100.0%** | 92.3% | 96.2% | 100.0% (29/29) |
| **`lost_solution`** | 16 | 16 | **100.0%** | 92.3% | 96.2% | 100.0% (16/16) |
| **`incorrect_setup`** | 17 | 17 | **100.0%** | 92.3% | 96.2% | 100.0% (17/17) |

---

## 5. Symbolic Limitation Boundary Cases (`eval_191` to `eval_200`, N=10)

- **Subset Size**: 10 cases (all injected errors)
- **Hybrid Detected Errors (TP)**: **10 / 10 (100.0% Recall)**
- **LLM Baseline Detected**: 10 / 10 (100.0%)
- **Solver-Only Detected**: 3 / 10 (30.0%)
- **Key Insight**: The hybrid architecture successfully resolves all 10 boundary cases where the deterministic solver alone suffered blind spots.

---

## 6. Deep Evidence-Fusion & Discrepancy Analysis

### A. Errors Caught by Hybrid that LLM Alone Missed (N=3)
The deterministic solver successfully overrode the LLM's false negative on 3 cases, proving mathematical errors:

- **`eval_092`** (Category: `arithmetic_error`, Gold Step: 1):
  - **Solver Proof**: "Mathematical assertion is incorrect: Equation is mathematically incorrect: LHS (15 + 7 = 22) does not equal RHS (21 = 21..."
  - **LLM Oversight**: "The perimeter of a rectangle is calculated using the formula P = 2(l + w). The reasoning correctly adds the length and w..."
  - *Resolution*: Solver detected the mathematical contradiction at Step 1, rescuing the missed error.

- **`eval_105`** (Category: `arithmetic_error`, Gold Step: 1):
  - **Solver Proof**: "Mathematical assertion is incorrect: Equation is mathematically incorrect: LHS (1000 - 875 = 125) does not equal RHS (22..."
  - **LLM Oversight**: "The subtraction 1000 - 875 is performed correctly, resulting in 125. Wait, re-evaluating: 1000 - 875 = 125. The provided..."
  - *Resolution*: Solver detected the mathematical contradiction at Step 1, rescuing the missed error.

- **`eval_118`** (Category: `invalid_transformation`, Gold Step: 2):
  - **Solver Proof**: "Invalid algebraic transformation: original equation has solution(s) ['13'], but transformed equation has solution(s) ['7..."
  - **LLM Oversight**: "The expansion in Step 1 is correct, and the algebraic manipulation in Step 2 correctly isolates x by subtracting 2x from..."
  - *Resolution*: Solver detected the mathematical contradiction at Step 2, rescuing the missed error.

### B. Errors Caught by Hybrid that Solver Alone Missed (N=61)
The LLM semantic detector rescued 61 errors where solver algebra was self-consistent but flawed:
- **All 17 `incorrect_setup` cases**: The solver verified that each step follows algebraically, but could not detect that Step 1 misrepresents the word problem.
- **9 `domain_violation` cases**: Contextual and implicit real-world domain restrictions.
- **20 `invalid_transformation` cases**: Multi-step conceptual leaps or non-consecutive algebraic statements.
- **13 `lost_solution` cases**: Implicit branch omissions not matching division-by-variable patterns.
- **1 `eval_198` case**: Natural-language number-theoretic assertion that contains no parseable math operators.

### C. False Positives Introduced by Evidence Fusion (N=5)
Combining evidence introduced 5 false positives where the LLM correctly identified sound reasoning, but the solver raised a false alarm:

- **`eval_008`** (Category: `none`):
  - Solver flagged error: `lost_solution` ("Invalid algebraic transformation: divided both sides by variable expression 'x - 3', losing solution...")
  - LLM rationale: "The quadratic expression x^2 - 5x + 6 factors correctly into (x - 2)(x - 3), and setting each factor..."

- **`eval_019`** (Category: `none`):
  - Solver flagged error: `invalid_transformation` ("Transition involves an invalid mathematical statement ('x = 3 or x')....")
  - LLM rationale: "The algebraic steps are logically sound: adding 9 to both sides correctly isolates the squared term,..."

- **`eval_030`** (Category: `none`):
  - Solver flagged error: `lost_solution` ("Invalid algebraic transformation: divided both sides by variable expression 'x + 1', losing solution...")
  - LLM rationale: "The quadratic equation is correctly factored into (x - 5)(x + 1) = 0, and the roots are correctly id..."

- **`eval_065`** (Category: `unsupported`):
  - Solver flagged error: `invalid_transformation` ("Transition involves an invalid mathematical statement ('Area = ((Base1 + Base2) / 2) * Height')....")
  - LLM rationale: "The formula used is the standard area formula for a trapezoid, and the arithmetic operations perform..."

- **`eval_070`** (Category: `unsupported`):
  - Solver flagged error: `invalid_transformation` ("Transition involves an invalid mathematical statement ('Slope = (y2 - y1) / (x2 - x1)')....")
  - LLM rationale: "The slope formula is correctly identified, and the arithmetic substitution and calculation are accur..."

### D. Trade-off Analysis
- **Recall Gain**: Error recall improved from **97.50% to 100.00%** (zero missed errors).
- **Precision Trade-off**: Precision decreased from **97.50% to 93.75%** due to 5 false alarms introduced by aggressive solver heuristics (such as multivariable target ambiguities).
- **Overall**: F1-score remains extraordinarily high at **96.77%**, with optimal error protection for safety-critical applications.

---
*Report generated automatically by `evaluate_hybrid.py`.*