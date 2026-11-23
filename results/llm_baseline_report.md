# Phase 4 – Objective 27: LLM-Only Error Detection Baseline Report

## 1. Executive Summary

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Cases** | 200 | Complete Phase 4 evaluation dataset |
| **Strict Accuracy** | 97.0% | Uncertain predictions counted as incorrect |
| **Decisive Accuracy** | 97.0% | Accuracy evaluated only on decisive calls |
| **Decisive Coverage** | 100.0% | Percentage of cases model made a decisive call |
| **Uncertainty Rate** | 0.0% | Abstention frequency across dataset |
| **Error Precision** | 97.5% | True positives / total model incorrect flags |
| **Conservative Recall** | 97.5% | True positives / total gold errors (uncertain = missed) |
| **Error F1-Score** | 97.5% | Harmonic mean of precision & conservative recall |
| **Step Localization Accuracy** | 97.4% | Correct first error step among True Positives |

---

## 2. Confusion Matrix (Positive Class = Incorrect)

| Matrix Category | Count | Percentage | Definition |
| :--- | :--- | :--- | :--- |
| **True Positives (TP)** | 117 | 58.5% | Explicit incorrect on gold incorrect |
| **False Positives (FP)** | 3 | 1.5% | Explicit incorrect on gold correct |
| **True Negatives (TN)** | 77 | 38.5% | Explicit correct on gold correct |
| **False Negatives (Explicit)** | 3 | 1.5% | Explicit correct on gold incorrect |
| **False Negatives (Uncertain)** | 0 | 0.0% | Uncertain on gold incorrect (missed error) |
| **Abstentions on Correct** | 0 | 0.0% | Uncertain on gold correct (not counted as FP) |

---

## 3. Performance Across Dataset Cohorts

| Cohort | N | Strict Acc | Decisive Acc | Uncertainty | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sound Cases (`none`)** | 60 | 96.7% | 96.7% | 0.0% | 0.0% | 0.0% |
| **Unsupported Cases (`unsupported`)** | 10 | 90.0% | 90.0% | 0.0% | 0.0% | 0.0% |
| **Conditional Warnings (`conditional_warning`)** | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **All Incorrect Cases (Injected Errors)** | 120 | 97.5% | 97.5% | 0.0% | 100.0% | 97.5% |

---

## 4. Breakdown by Injected Error Category (N=120)

| Error Category | N | Detected (TP) | Missed (Explicit FN) | Missed (Uncertain FN) | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`invalid_transformation`** | 29 | 28 | 1 | 0 | 96.5% |
| **`arithmetic_error`** | 26 | 24 | 2 | 0 | 92.3% |
| **`domain_violation`** | 17 | 17 | 0 | 0 | 100.0% |
| **`incorrect_setup`** | 17 | 17 | 0 | 0 | 100.0% |
| **`lost_solution`** | 16 | 16 | 0 | 0 | 100.0% |
| **`division_by_zero`** | 15 | 15 | 0 | 0 | 100.0% |

---

## 5. Symbolic Limitation Boundary Cases (`eval_191` to `eval_200`, N=10)

- **Total Cases**: 10
- **Detected Errors (TP)**: 10 / 10
- **Conservative Recall**: 100.0%
- **Decisive Accuracy**: 100.0%
- **Uncertainty Rate**: 0.0%
