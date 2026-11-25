# Phase 4 — Objective 29: Deterministic Solver-Only Evaluation Report

> **Evaluation Scope**: Canonical 200-Case Evaluation Dataset (`data/reasoning_eval_dataset.jsonl`)
> **Generated**: 2026-11-24 | **API Calls**: 0 (100% Deterministic & Offline)

---

## 1. Executive Summary

This report evaluates the **deterministic mathematical solver pipeline** in isolation, completely disabling LLM classification, LLM explanations, and external API calls. The goal is to establish the exact empirical performance boundary of symbolic mathematical reasoning verification (SymPy AST evaluation and algebraic transition verification) on real-world multi-step reasoning traces.

| Metric | Value | Baseline / Interpretation |
| :--- | :---: | :--- |
| **Total Canonical Cases** | **200** | Complete Phase 4 evaluation dataset |
| **Supported Cases** | **191** | Cases with decisive verification status (`verified` or `errors_detected`) |
| **Unsupported Cases** | **9** | Cases with unparsed expressions or conceptual formulas |
| **Supported Coverage** | **95.50%** | Percentage of test set within deterministic capability |
| **Strict Accuracy (All 200)** | **62.50%** (125/200) | Unsupported cases counted as unverified / abstention |
| **Supported-Case Accuracy** | **65.45%** (125/191) | Accuracy evaluated exclusively on supported cases |
| **Error Precision** | **90.77%** (59/65) | When solver flags an error, it is correct >90% of the time |
| **Error Recall** | **49.17%** (59/120) | Detects ~49% of all errors (catches syntax/algebra/arithmetic; misses semantic setup) |
| **Error F1-Score** | **63.78%** | Harmonic mean of precision and recall |
| **Step Localization Accuracy** | **84.75%** (50/59) | Pinpoints the exact first erroneous step on True Positives |

---

## 2. Confusion Matrix (Positive Class = Injected Error)

| Matrix Category | Count | Percentage | Definition & Deterministic Handling |
| :--- | :---: | :---: | :--- |
| **True Positives (TP)** | **59** | 29.5% | Correctly detected invalid transformation, division by zero, or arithmetic error |
| **False Positives (FP)** | **6** | 3.0% | Erroneously flagged sound trace (e.g. multi-variable label ambiguity or variable division) |
| **True Negatives (TN)** | **66** | 33.0% | Successfully verified all mathematical statements and transitions as sound |
| **False Negatives (Explicit)** | **60** | 30.0% | Missed error: steps were algebraically self-consistent (e.g. incorrect word-problem setup) |
| **False Negatives (Unsupported)** | **1** | 0.5% | Missed error due to unparsed natural-language statement (`eval_198`) |
| **Abstentions on Correct** | **8** | 4.0% | Appropriately identified unsupported conceptual equations without guessing |
| **Total Unsupported (Abstentions)**| **9** | 4.5% | Total abstention count across dataset |

---

## 3. Cohort Performance Breakdown

| Cohort | N | Supported | Strict Acc | Supported Acc | Precision | Recall | Abstention Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort 1: Sound Reasoning (`none`)** | 60 | 60 | 93.3% | 93.3% | 0.0% | 0.0% | 0.0% |
| **Cohort 2: Unsupported Equations (`unsupported`)** | 10 | 2 | 0.0% | 0.0% | 0.0% | 0.0% | 80.0% |
| **Cohort 3: Conditional Warnings (`conditional_warning`)** | 10 | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Cohort 4: All Injected Errors** | 120 | 119 | 49.2% | 49.6% | 100.0% | 49.2% | 0.8% |

---

## 4. Injected Error Category Breakdown (N=120)

| Error Category | Total Cases | Detected (TP) | Missed (Explicit FN) | Missed (Unsupp FN) | Error Recall | Step Localization Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`division_by_zero`** | 15 | 15 | 0 | 0 | **100.0%** | 93.3% (14/15) |
| **`arithmetic_error`** | 26 | 25 | 1 | 0 | **96.2%** | 100.0% (25/25) |
| **`domain_violation`** | 17 | 8 | 9 | 0 | **47.1%** | 50.0% (4/8) |
| **`invalid_transformation`** | 29 | 8 | 20 | 1 | **27.6%** | 62.5% (5/8) |
| **`lost_solution`** | 16 | 3 | 13 | 0 | **18.8%** | 66.7% (2/3) |
| **`incorrect_setup`** | 17 | 0 | 17 | 0 | **0.0%** | 0.0% (0/0) |

---

## 5. Symbolic Limitation Boundary Cases (`eval_191` to `eval_200`, N=10)

The canonical dataset includes 10 boundary cases specifically designed to probe the theoretical limitations of pure symbolic equation solvers (e.g. variable shadowing, implicit assumptions, natural language assertions):

- **Subset Size**: 10 cases (all injected errors)
- **Detected Errors (TP)**: 3 / 10 (30.0%)
- **Missed Errors (Explicit FN)**: 6 / 10
- **Missed Errors (Unsupported FN)**: 1 / 10 (`eval_198`: natural language prime assertion)
- **Strict Accuracy**: 30.0%
- **Key Finding on Boundary Cases**: Symbolic solvers verify *internal algebraic consistency*; they cannot detect when equations deviate from implicit real-world problem constraints or variable scoping shadowing unless explicitly stated as formal constraints.

---

## 6. Key Empirical Insights

1. **Near-Perfect Precision on Symbolic Errors (90.77%)**: When the deterministic pipeline flags an error, it is almost certainly a genuine mathematical flaw. Out of 65 total error flags, 59 are True Positives. The 6 False Positives stem from multi-variable label ambiguities (`eval_065`, `eval_070`) and aggressive division-by-variable checks.
2. **Specialized Strengths**: The solver achieves **100.0% recall on `division_by_zero`** and **96.2% recall on `arithmetic_error`**, far outperforming unassisted human inspection in raw calculation precision.
3. **The Semantic Blind Spot (`incorrect_setup` = 0.0% Recall)**: The solver achieved **0 / 17 recall on `incorrect_setup`**. Because the solver checks whether Step $n$ follows from Step $n-1$, it cannot detect if Step 1 is mathematically valid in isolation but misrepresents the English word problem.
4. **Step Localization Precision (84.75%)**: When an error is detected, the pipeline correctly localizes the exact first faulty step in 50 out of 59 cases.
5. **High Coverage (95.5%)**: Despite requiring strict mathematical formatting, the pipeline successfully extracted and verified expressions for 191 of the 200 cases, abstaining gracefully on only 9 cases.

---

## 7. Comparative Context (Objectives 27, 28 vs Objective 29)

| Evaluation Metric | Obj 27 (LLM Baseline) | Obj 28 (Improved LLM) | Obj 29 (Deterministic Solver) |
| :--- | :---: | :---: | :---: |
| **Methodology** | LLM Direct Prompt | Improved LLM Verification Prompt | SymPy AST & Transition Verifier |
| **External API Calls** | 200 | 200 (187 completed) | **0 (Offline & Free)** |
| **Coverage** | 100.0% | 100.0% | 95.5% (191/200) |
| **Error Precision** | 97.20% | 97.20% | 90.77% |
| **Error Recall** | 97.20% | 97.20% | **49.17%** |
| **Arithmetic Error Recall** | 92.3% | 92.3% | **96.2%** |
| **Division by Zero Recall** | 100.0% | 100.0% | **100.0%** |
| **Incorrect Setup Recall** | 100.0% | 100.0% | **0.0%** |
| **Strict Accuracy** | 96.79% | 96.79% | 62.50% |

> **Core Architecture Implication**: This empirical divergence establishes the core motivation for **hybrid neuro-symbolic architectures**: LLMs excel at semantic comprehension and problem-to-equation modeling (100% setup recall), whereas deterministic solvers provide rigorous, unhallucinated verification of arithmetic and algebraic transformations (100% division-by-zero recall, 96.2% arithmetic recall, 0 API tokens).

---
*Report generated automatically by `evaluate_solver_only.py`.*