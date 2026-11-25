# Phase 4 – Objective 28: Improved LLM-Based Reasoning Error Detection Report

## 1. Executive Summary

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Cases** | 20 | Complete Phase 4 evaluation dataset |
| **Strict Accuracy** | 80.0% | Uncertain predictions counted as incorrect |
| **Decisive Accuracy** | 80.0% | Accuracy evaluated only on decisive calls |
| **Decisive Coverage** | 100.0% | Percentage of cases model made a decisive call |
| **Uncertainty Rate** | 0.0% | Abstention frequency across dataset |
| **Error Precision** | 87.5% | True positives / total model incorrect flags |
| **Conservative Recall** | 87.5% | True positives / total gold errors (uncertain = missed) |
| **Error F1-Score** | 87.5% | Harmonic mean of precision & conservative recall |
| **Step Localization Accuracy** | 92.9% | Correct first error step among True Positives |

---

## 2. Confusion Matrix (Positive Class = Incorrect)

| Matrix Category | Count | Percentage | Definition |
| :--- | :--- | :--- | :--- |
| **True Positives (TP)** | 14 | 70.0% | Explicit incorrect on gold incorrect |
| **False Positives (FP)** | 2 | 10.0% | Explicit incorrect on gold correct |
| **True Negatives (TN)** | 2 | 10.0% | Explicit correct on gold correct |
| **False Negatives (Explicit)** | 2 | 10.0% | Explicit correct on gold incorrect |
| **False Negatives (Uncertain)** | 0 | 0.0% | Uncertain on gold incorrect (missed error) |
| **Abstentions on Correct** | 0 | 0.0% | Uncertain on gold correct (not counted as FP) |

---

## 3. Performance Across Dataset Cohorts

| Cohort | N | Strict Acc | Decisive Acc | Uncertainty | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sound Cases (`none`)** | 2 | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% |
| **Unsupported Cases (`unsupported`)** | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Conditional Warnings (`conditional_warning`)** | 1 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **All Incorrect Cases (Injected Errors)** | 16 | 87.5% | 87.5% | 0.0% | 100.0% | 87.5% |

---

## 4. Breakdown by Injected Error Category (N=120)

| Error Category | N | Detected (TP) | Missed (Explicit FN) | Missed (Uncertain FN) | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`invalid_transformation`** | 5 | 4 | 1 | 0 | 80.0% |
| **`arithmetic_error`** | 2 | 1 | 1 | 0 | 50.0% |
| **`domain_violation`** | 3 | 3 | 0 | 0 | 100.0% |
| **`incorrect_setup`** | 3 | 3 | 0 | 0 | 100.0% |
| **`lost_solution`** | 2 | 2 | 0 | 0 | 100.0% |
| **`division_by_zero`** | 1 | 1 | 0 | 0 | 100.0% |

---

## 5. Symbolic Limitation Boundary Cases (`eval_191` to `eval_200`, N=10)

- **Total Cases**: 10
- **Detected Errors (TP)**: 10 / 10
- **Conservative Recall**: 100.0%
- **Decisive Accuracy**: 100.0%
- **Uncertainty Rate**: 0.0%
