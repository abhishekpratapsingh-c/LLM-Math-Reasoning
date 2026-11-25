# Phase 4 — Objective 32: Step-Level Localization Evaluation Report

> **Evaluation Scope**: Canonical 200-Case Dataset & Like-for-Like 187 Common Cases  
> **Generated**: 2026-11-25 03:21:33 UTC | **External API Calls**: 0 (Fully Reproducible Artifact Audit)  

---

## 1. Executive Summary & Core Definitions

This evaluation investigates **first-error step localization accuracy**: when mathematical reasoning fails, does each detection paradigm accurately identify the exact step where reasoning first breaks down?

### Key Metric Distinctions
- **Conditional Step Localization Accuracy**: $\frac{\text{Exact Matches}}{\text{True Positives}}$. Evaluates localization precision strictly among detected errors.
- **Unconditional Step Localization Accuracy**: $\frac{\text{Exact Matches}}{\text{All Injected Errors}}$. Evaluates end-to-end detection and localization, penalizing false negatives (missed errors).
- **Mean Absolute Step Distance (MAE)**: Average $|k_{\text{pred}} - k_{\text{gold}}|$ computed over True Positives with numeric predictions.
- **First-Error Confusion Matrix**: Full contingency matrix accounting for sound reasoning, exact matches, mislocalizations, false negatives, and false positives.

> [!IMPORTANT]
> **First-Error vs. Per-Step Classification**: As established in the Objective 32 feasibility audit, none of the four systems output a per-step prediction vector $[y_1, \dots, y_N]$. Evaluating first-error localization evaluates what the models actually predicted without inventing missing labels for unreached steps.

---

## 2. Like-for-Like Comparison on the Common 187 Cases

All four systems evaluated on the exact **187 common cases** successfully completed by Objective 28 (80 sound cases, 107 injected error cases):

| Approach / Paradigm | Model / Methodology | Cond Loc Acc | Uncond Loc Acc | MAE Step Dist | Exact Matches | TP | FN | FP | Dist 0 | Dist 1 | Dist 2 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Obj 27: LLM Baseline** | Zero-Shot Direct Prompt | **97.12%** | **94.39%** | 0.0288 | 101 | 104 | 3 | 3 | 101 | 3 | 0 |
| **Obj 28: Improved LLM** | Improved LLM Verification Prompt | **96.15%** | **93.46%** | 0.0385 | 100 | 104 | 3 | 3 | 100 | 4 | 0 |
| **Obj 29: Solver-Only** | SymPy AST & Transition Verifier | **84.48%** | **45.79%** | 0.2069 | 49 | 58 | 49 | 6 | 49 | 6 | 3 |
| **Obj 30: Hybrid System** | Neuro-Symbolic Evidence Fusion | **97.20%** | **97.20%** | 0.0280 | 104 | 107 | 0 | 8 | 104 | 3 | 0 |

### Key Findings on Common Subset:
1. **Hybrid System Supremacy**: The Hybrid system achieved the highest unconditional step localization accuracy (**97.20%**, 104/107) because it suffered zero false negatives.
2. **Baseline vs. Improved Prompt**: The baseline LLM achieved slightly higher conditional localization (**97.12% vs 96.15%**) than the improved prompt because Objective 28 mislocalized `eval_196` (variable scope shadowing) to Step 3 instead of Step 2.
3. **Solver Deficit**: The deterministic solver achieved 84.48% conditional accuracy on its detected errors, but dropped to 45.79% unconditional accuracy due to 49 missed errors.

---

## 3. Full 200-Case Evaluation Results (Objectives 27, 29, 30)

Evaluated on all 200 canonical cases (80 sound cases, 120 injected errors):

| Approach | Strict Error Acc | Cond Loc Acc | Uncond Loc Acc | MAE Step Dist | Exact Matches | TP | FN | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Objective 27: LLM Baseline** | 97.00% | **97.44%** | **95.00%** | 0.0256 | 114 | 117 | 3 | 3 |
| **Objective 29: Solver-Only** | 62.50% | **84.75%** | **41.67%** | 0.2034 | 50 | 59 | 61 | 6 |
| **Objective 30: Hybrid System** | 96.00% | **97.50%** | **97.50%** | 0.0250 | 117 | 120 | 0 | 8 |

> [!NOTE]
> **Objective 28 Exclusion Notice**: Objective 28 completed 187 of 200 cases due to upstream Gemini API 503 UNAVAILABLE service overloads. To prevent ungrounded imputation, Objective 28 is evaluated exclusively on the 187 common cases.

---

## 4. Localization by Injected Error Category

Evaluating localization accuracy across all six benchmark mathematical error categories (Full 200 Cases for Obj 27, 29, 30; 187 Cases for Obj 28):

| Error Category | N (200 / 187) | Obj 27 Cond (Uncond) | Obj 28 Cond (Uncond) | Obj 29 Cond (Uncond) | Obj 30 Cond (Uncond) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`arithmetic_error`** | 26 / 26 | 100.0% (92.3%) | 100.0% (92.3%) | 100.0% (96.2%) | **100.0% (100.0%)** |
| **`division_by_zero`** | 15 / 15 | 100.0% (100.0%) | 100.0% (100.0%) | 93.3% (93.3%) | **100.0% (100.0%)** |
| **`domain_violation`** | 17 / 13 | 82.4% (82.4%) | 76.9% (76.9%) | 50.0% (23.5%) | **82.4% (82.4%)** |
| **`incorrect_setup`** | 17 / 10 | 100.0% (100.0%) | 100.0% (100.0%) | N/A (0.0%) | **100.0% (100.0%)** |
| **`invalid_transformation`** | 29 / 29 | 100.0% (96.6%) | 96.4% (93.1%) | 62.5% (17.2%) | **100.0% (100.0%)** |
| **`lost_solution`** | 16 / 14 | 100.0% (100.0%) | 100.0% (100.0%) | 66.7% (12.5%) | **100.0% (100.0%)** |

*Note on Solver-Only `incorrect_setup`*: Conditional accuracy is N/A because the solver detected 0 True Positives on setup errors (zero denominator handled explicitly without reporting 0%).*

---

## 5. Localization by Error-Step Depth

Evaluating localization accuracy by the position of the injected error in the reasoning chain (Full 200 Cases):

| Error Depth | Total Cases | Obj 27 Cond (Uncond) | Obj 29 Cond (Uncond) | Obj 30 Cond (Uncond) |
| :--- | :---: | :---: | :---: | :---: |
| **Step 1 Error** | 85 | 100.0% (97.6%) | 90.3% (32.9%) | **100.0% (100.0%)** |
| **Step 2 Error** | 27 | 92.3% (88.9%) | 86.4% (70.4%) | **92.6% (92.6%)** |
| **Step 3 Error** | 8 | 87.5% (87.5%) | 50.0% (37.5%) | **87.5% (87.5%)** |

---

## 6. First-Error Confusion Matrices

Full contingency matrices explicitly displaying exact matches, mislocalizations, false negatives, and false positives:

### A. Objective 30: Hybrid System (Full 200 Cases)
```
Ground Truth State  | No Error Pred | Pred Step 1 | Pred Step 2 | Pred Step 3 | Total Cases
--------------------+---------------+-------------+-------------+-------------+------------
Sound Reasoning     |      72       |      1      |      6      |      1      |     80     
Gold Error Step 1   |       0       |     85      |      0      |      0      |     85     
Gold Error Step 2   |       0       |      2      |     25      |      0      |     27     
Gold Error Step 3   |       0       |      0      |      1      |      7      |      8     
--------------------+---------------+-------------+-------------+-------------+------------
Total Predictions   |      72       |     88      |     32      |      8      |    200    
```

### B. Objective 27: LLM Baseline (Full 200 Cases)
```
Ground Truth State  | No Error Pred | Pred Step 1 | Pred Step 2 | Pred Step 3 | Total Cases
--------------------+---------------+-------------+-------------+-------------+------------
Sound Reasoning     |      77       |      1      |      1      |      1      |     80     
Gold Error Step 1   |       2       |     83      |      0      |      0      |     85     
Gold Error Step 2   |       1       |      2      |     24      |      0      |     27     
Gold Error Step 3   |       0       |      0      |      1      |      7      |      8     
--------------------+---------------+-------------+-------------+-------------+------------
Total Predictions   |      80       |     86      |     26      |      8      |    200    
```

### C. Objective 28: Improved LLM (Common 187 Cases)
```
Ground Truth State  | No Error Pred | Pred Step 1 | Pred Step 2 | Pred Step 3 | Total Cases
--------------------+---------------+-------------+-------------+-------------+------------
Sound Reasoning     |      77       |      1      |      1      |      1      |     80     
Gold Error Step 1   |       2       |     72      |      0      |      0      |     74     
Gold Error Step 2   |       1       |      2      |     23      |      1      |     27     
Gold Error Step 3   |       0       |      0      |      1      |      5      |      6     
--------------------+---------------+-------------+-------------+-------------+------------
Total Predictions   |      80       |     75      |     25      |      7      |    187    
```

### D. Objective 29: Solver-Only (Full 200 Cases)
```
Ground Truth State  | No Error Pred | Pred Step 1 | Pred Step 2 | Pred Step 3 | Total Cases
--------------------+---------------+-------------+-------------+-------------+------------
Sound Reasoning     |      74       |      0      |      5      |      1      |     80     
Gold Error Step 1   |      54       |     28      |      2      |      1      |     85     
Gold Error Step 2   |       5       |      3      |     19      |      0      |     27     
Gold Error Step 3   |       2       |      2      |      1      |      3      |      8     
--------------------+---------------+-------------+-------------+-------------+------------
Total Predictions   |     135       |     33      |     27      |      5      |    200    
```

---

## 7. Deep Discrepancy & Mislocalization Analysis

### A. The 3 LLM / Hybrid Mislocalizations
In all three cases, the LLM flagged the error earlier than the benchmark injection point due to subtle domain boundary interpretations:
1. **`eval_164`** (Category: `domain_violation`, Gold Step: 3, Predicted Step: 2):
   - The reasoning evaluated $\log(x - 2)$ at $x = 1$. Gold ground truth placed the violation at Step 3 (evaluating $\log(-1)$), but the LLM flagged Step 2 where $x = 1$ was substituted into the expression.
2. **`eval_168`** (Category: `domain_violation`, Gold Step: 2, Predicted Step: 1):
   - In Step 1, the student set up $\sqrt{x + 5} = -2$. The LLM immediately flagged Step 1 because the principal square root cannot equal a negative number, whereas gold ground truth tagged Step 2 where squaring produced an extraneous root.
3. **`eval_174`** (Category: `domain_violation`, Gold Step: 2, Predicted Step: 1):
   - Similar principal square root non-negativity boundary identified at initial statement (Step 1) rather than root test (Step 2).

### B. Objective 28 Boundary Discrepancy (`eval_196`)
- **`eval_196`** (Category: `invalid_transformation` / variable shadowing, Gold Step: 2, Predicted Step: 3):
   - Variable $x$ was repurposed across steps. Objective 27 localized the break at Step 2 (where shadowing occurred); Objective 28 localized the failure at Step 3 (where the shadowed value was evaluated). Distance = 1.

---

## 8. Verification & Validation Summary

All metrics and confusion matrix row/column totals strictly reconcile with the audited dataset ground truth and underlying prediction records:
- Full 200 cases: Exactly 80 Sound cases and 120 Error cases accounted for in every matrix.
- Common 187 cases: Exactly 80 Sound cases and 107 Error cases accounted for in every matrix.
- Zero API calls made; 100% reproducible offline.

---
*Report generated automatically by `evaluate_step_localization.py`.*