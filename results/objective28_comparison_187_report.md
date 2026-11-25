# Objective 28 vs Objective 27 Comparative Evaluation Report

> **Evaluation Scope**: Exact 187 Common Cases (Like-for-Like Benchmark)
> **Generated**: 2026-11-24 | **Dataset**: `data/reasoning_eval_dataset.jsonl`

---

## 1. Executive Summary

This evaluation benchmarks **Objective 28** (Improved LLM Error Detector with structured verification prompt, taxonomy, and systematic verification guidelines) directly against **Objective 27** (Baseline LLM Error Detector with direct prompt) across the exact **187 common cases** where both systems successfully completed API evaluation without transport errors.

| Metric | Baseline (Obj 27) | Improved (Obj 28) | Delta | Status / Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Evaluated Cases** | 187 | 187 | 0 | Identical common test set |
| **Strict Accuracy** | 96.79% | 96.79% | +0.00% | Exact parity across all 187 cases |
| **Decisive Accuracy** | 96.79% | 96.79% | +0.00% | Exact parity (zero abstentions in either) |
| **Decisive Coverage** | 100.00% | 100.00% | +0.00% | 100.0% coverage on both |
| **Uncertainty Rate** | 0.00% | 0.00% | +0.00% | 0.0% uncertainty on both |
| **Error Precision** | 97.20% | 97.20% | +0.00% | Identical TP (104) and FP (3) |
| **Conservative Recall** | 97.20% | 97.20% | +0.00% | Identical TP (104) and FN (3) |
| **Error F1-Score** | 97.20% | 97.20% | +0.00% | Exact parity |
| **Step Localization Accuracy** | 97.12% | 96.15% | -0.97% | Baseline 101/104 vs Improved 100/104 (-1 case) |

---

## 2. Evaluation Setup & Rigorous Controls

- **Canonical Dataset Size**: 200 cases (`data/reasoning_eval_dataset.jsonl`)
- **Common Evaluated Subset**: 187 cases
- **Excluded Cases Count**: 13 cases
- **Excluded Case IDs**: `eval_136, eval_140, eval_170, eval_171, eval_173, eval_175, eval_178, eval_179, eval_181, eval_182, eval_183, eval_184, eval_185`
- **Exclusion Rationale**: Transient Gemini API 503 UNAVAILABLE service overloads during Objective 28 evaluation. Strictly excluded from both systems to guarantee exact like-for-like comparison.
- **Model Configuration**:
  - **Objective 27**: `gemini` / `gemini-3.1-flash-lite` (Zero-shot direct reasoning error classification prompt)
  - **Objective 28**: `gemini` / `gemini-3.1-flash-lite` (Structured verification prompt with error typology, domain check rules, step-level localization, and systematic verification guidelines)
- **Information Integrity**: Verified that neither system's prediction logs contain leaked reference answers or gold labels.

---

## 3. Confusion Matrix Comparison (Positive Class = Incorrect)

| Matrix Category | Baseline (Obj 27) | Improved (Obj 28) | Delta | Category Definition |
| :--- | :---: | :---: | :---: | :--- |
| **True Positives (TP)** | 104 | 104 | +0 | Correctly flagged injected reasoning error |
| **False Positives (FP)** | 3 | 3 | +0 | Incorrectly flagged sound/valid reasoning trace |
| **True Negatives (TN)** | 77 | 77 | +0 | Correctly confirmed sound/valid reasoning trace |
| **False Negatives (Explicit)** | 3 | 3 | +0 | Predicted `correct` on an injected error |
| **False Negatives (Uncertain)** | 0 | 0 | +0 | Predicted `uncertain` on an injected error |
| **Total False Negatives** | 3 | 3 | +0 | Explicit FN + Uncertain FN |
| **Abstentions on Correct** | 0 | 0 | +0 | Predicted `uncertain` on valid reasoning |
| **Total Uncertain Predictions** | 0 | 0 | +0 | Overall abstention count across test set |

---

## 4. Cohort Performance Comparison

| Cohort | N | Baseline Acc | Improved Acc | Baseline Prec | Improved Prec | Baseline Recall | Improved Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort 1: Sound Reasoning (`none`)** | 60 | 96.7% | 96.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Cohort 2: Unsupported Equations (`unsupported`)** | 10 | 90.0% | 90.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Cohort 3: Conditional Warnings (`conditional_warning`)** | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Cohort 4: All Injected Errors** | 107 | 97.2% | 97.2% | 100.0% | 100.0% | 97.2% | 97.2% |

---

## 5. Injected Error Category Breakdown (N=107)

| Error Category | N | Baseline TP | Improved TP | Baseline FN | Improved FN | Baseline Recall | Improved Recall | Baseline Step Acc | Improved Step Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`invalid_transformation`** | 29 | 28 | 28 | 1 | 1 | 96.5% | 96.5% | 100.0% | 96.4% |
| **`arithmetic_error`** | 26 | 24 | 24 | 2 | 2 | 92.3% | 92.3% | 100.0% | 100.0% |
| **`domain_violation`** | 13 | 13 | 13 | 0 | 0 | 100.0% | 100.0% | 76.9% | 76.9% |
| **`incorrect_setup`** | 10 | 10 | 10 | 0 | 0 | 100.0% | 100.0% | 100.0% | 100.0% |
| **`lost_solution`** | 14 | 14 | 14 | 0 | 0 | 100.0% | 100.0% | 100.0% | 100.0% |
| **`division_by_zero`** | 15 | 15 | 15 | 0 | 0 | 100.0% | 100.0% | 100.0% | 100.0% |

---

## 6. Symbolic Limitation Boundary Cases (`eval_191` to `eval_200`, N=10)

This boundary subset tests complex reasoning scenarios where symbolic solver pipelines encounter domain limitations (variable shadowing, piecewise continuity, implicit branches, etc.):

- **Subset Size**: 10 cases (all present in common test set)
- **Baseline True Positives**: 10 / 10 (100.0%)
- **Improved True Positives**: 10 / 10 (100.0%)
- **Baseline Step Localization**: 10/10 (100.0%)
- **Improved Step Localization**: 9/10 (90.0%)
- **Discrepancy in Boundary Cases**: `eval_196` (variable shadowing) was localized to step 2 by Baseline and step 3 by Improved.

---

## 7. Deep Discrepancy & Failure Analysis

### A. Classification Agreement
- **Cases Agreed on Classification**: 187 / 187 (**100.0%**)
- **Cases Disagreed on Classification**: 0
- **Both Correct**: 181 cases (77 True Negatives, 104 True Positives)
- **Both Incorrect**: 6 cases (3 False Positives, 3 False Negatives)

### B. Step Localization Discrepancy (Common True Positives, N=104)
- **Step Matches Agreed**: 103 / 104 (99.0%)
- **Step Localization Disagreements**: 1 case

#### Case `eval_196` (invalid_transformation)
- **Gold Error Step**: Step `2`
- **Baseline Prediction**: Step `2` (Matches Gold: **True**)
- **Improved Prediction**: Step `3` (Matches Gold: **False**)
- **Baseline Rationale**: "The problem describes variable shadowing, where the second assignment of 'x' replaces the first. Step 2 incorrectly implies that the two assignments coexist as distinct quantities to be added, rather than the second assignment overwriting the first."
- **Improved Rationale**: "The problem describes variable shadowing, where the second assignment of 'x' replaces the first, meaning the original value of 'x' (apples) is no longer accessible. Step 3 incorrectly treats the two assignments as additive quantities rather than a sequential update of the same variable."
- **Analysis**: The problem involves variable shadowing where `x` is defined as apples and then re-assigned to oranges. Baseline flagged Step 2 where the shadowing first occurs. Improved identified Step 3 where the two incompatible quantities were algebraically combined. While both recognized the invalid reasoning, the prompt restructuring in Objective 28 shifted the attribution from the declaration step to the consumption step.

### C. Shared False Positives (N=3)
Both systems incorrectly flagged sound reasoning traces as erroneous on the following 3 cases:

- **`eval_017`** (Category: `none`):
  - Baseline flagged Step 3: "In Step 3, the reasoning incorrectly asserts that 4x = 14 is equivalent to 4x = 16. The arithmetic 11 + 3 = 14 is correc..."
  - Improved flagged Step 3: "In Step 3, the reasoning contains an arithmetic error: 11 + 3 equals 14, not 16. Consequently, the final answer x = 4 is..."
  - *Root Cause*: Hallucinated arithmetic error (`eval_017`), strict penalty on concise one-step algebra (`eval_054`), and penalty for omitting physical units (`eval_063`).

- **`eval_054`** (Category: `none`):
  - Baseline flagged Step 1: "The reasoning trace fails to show the necessary algebraic steps to isolate x, and the jump from 2x + 5 = 15 to x = 5 is ..."
  - Improved flagged Step 1: "The reasoning trace fails to show the necessary algebraic steps to isolate x, jumping directly from the original equatio..."
  - *Root Cause*: Hallucinated arithmetic error (`eval_017`), strict penalty on concise one-step algebra (`eval_054`), and penalty for omitting physical units (`eval_063`).

- **`eval_063`** (Category: `unsupported`):
  - Baseline flagged Step 2: "While the arithmetic is correct, the reasoning trace fails to include the necessary physical units (g/cm^3) in the final..."
  - Improved flagged Step 2: "While the arithmetic calculation 200 / 50 = 4 is correct, the reasoning trace fails to include the necessary physical un..."
  - *Root Cause*: Hallucinated arithmetic error (`eval_017`), strict penalty on concise one-step algebra (`eval_054`), and penalty for omitting physical units (`eval_063`).

### D. Shared False Negatives (N=3)
Both systems failed to detect injected errors on the following 3 cases:

- **`eval_092`** (Category: `arithmetic_error`, Gold Step: 1):
  - Both systems predicted `correct` (missed error).
  - *Root Cause*: Subtly plausible arithmetic slips (`eval_092`: 15 + 7 = 21, `eval_105`: 1000 - 875 = 225) and algebraic transposition sign errors (`eval_118`: 2x + 10 = 3x - 3 -> x = 7).

- **`eval_105`** (Category: `arithmetic_error`, Gold Step: 1):
  - Both systems predicted `correct` (missed error).
  - *Root Cause*: Subtly plausible arithmetic slips (`eval_092`: 15 + 7 = 21, `eval_105`: 1000 - 875 = 225) and algebraic transposition sign errors (`eval_118`: 2x + 10 = 3x - 3 -> x = 7).

- **`eval_118`** (Category: `invalid_transformation`, Gold Step: 2):
  - Both systems predicted `correct` (missed error).
  - *Root Cause*: Subtly plausible arithmetic slips (`eval_092`: 15 + 7 = 21, `eval_105`: 1000 - 875 = 225) and algebraic transposition sign errors (`eval_118`: 2x + 10 = 3x - 3 -> x = 7).

---

## 8. Limitations & Scope Constraints

1. **Excluded 13 Cases**: Thirteen cases (eval_136, eval_140, eval_170, eval_171, eval_173, eval_175, eval_178, eval_179, eval_181, eval_182, eval_183, eval_184, eval_185) were excluded because Objective 28 suffered transient HTTP 503 UNAVAILABLE service overloads. All 13 excluded cases were gold-standard errors (7 `incorrect_setup`, 4 `domain_violation`, 2 `lost_solution`). Consequently, the reported metrics reflect performance exclusively on the 187 successfully completed cases.
2. **Synthetic Benchmark Nature**: The 187 test cases represent a curated benchmark covering algebra, calculus, arithmetic, and domain constraints; performance here may not translate directly to uncurated mathematical text.
3. **Zero Abstention / High Decisiveness**: Both systems chose to make decisive predictions on 100% of cases, declining to utilize the `uncertain` classification.

---

## 9. Conclusion

Based strictly on the empirical data from the 187 common cases:

1. **Classification Equivalence**: Objective 28's expanded prompt, error taxonomy, domain verification rules, and systematic verification procedure achieved **exact parity** with Objective 27 on classification accuracy (96.79%), precision (97.20%), recall (97.20%), and F1-score (97.20%). Both systems produced the exact same classification on 100.0% of cases.
2. **Failure Mode Invariance**: Prompt engineering alone did not resolve the persistent failure modes of the underlying model. Both models failed on the exact same 3 false positives and the exact same 3 false negatives.
3. **Localization Trade-off**: In step localization, Objective 28 achieved 96.15% (100/104) compared to Objective 27's 97.12% (101/104), due to a single discrepancy on variable shadowing (`eval_196`) where Objective 28 penalized the usage step rather than the definition step.
4. **Takeaway for Hybrid Systems**: Pure prompt engineering offers diminishing returns on strong reasoning models like `gemini-3.1-flash-lite`. Achieving higher precision and eliminating subtle arithmetic/transposition oversights requires **deterministic symbolic verification** rather than prompt variations.

---
*Report generated automatically by `generate_objective28_comparison.py`.*