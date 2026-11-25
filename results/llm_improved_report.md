# Phase 4 – Objective 28: Improved LLM-Based Reasoning Error Detection Report

## 1. Executive Summary

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Cases** | 200 | Complete Phase 4 evaluation dataset |
| **Strict Accuracy** | 8.0% | Uncertain predictions counted as incorrect |
| **Decisive Accuracy** | 80.0% | Accuracy evaluated only on decisive calls |
| **Decisive Coverage** | 10.0% | Percentage of cases model made a decisive call |
| **Uncertainty Rate** | 90.0% | Abstention frequency across dataset |
| **Error Precision** | 87.5% | True positives / total model incorrect flags |
| **Conservative Recall** | 11.7% | True positives / total gold errors (uncertain = missed) |
| **Error F1-Score** | 20.6% | Harmonic mean of precision & conservative recall |
| **Step Localization Accuracy** | 92.9% | Correct first error step among True Positives |

---

## 2. Confusion Matrix (Positive Class = Incorrect)

| Matrix Category | Count | Percentage | Definition |
| :--- | :--- | :--- | :--- |
| **True Positives (TP)** | 14 | 7.0% | Explicit incorrect on gold incorrect |
| **False Positives (FP)** | 2 | 1.0% | Explicit incorrect on gold correct |
| **True Negatives (TN)** | 2 | 1.0% | Explicit correct on gold correct |
| **False Negatives (Explicit)** | 2 | 1.0% | Explicit correct on gold incorrect |
| **False Negatives (Uncertain)** | 104 | 52.0% | Uncertain on gold incorrect (missed error) |
| **Abstentions on Correct** | 76 | 38.0% | Uncertain on gold correct (not counted as FP) |

---

## 3. Performance Across Dataset Cohorts

| Cohort | N | Strict Acc | Decisive Acc | Uncertainty | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sound Cases (`none`)** | 60 | 1.7% | 50.0% | 96.7% | 0.0% | 0.0% |
| **Unsupported Cases (`unsupported`)** | 10 | 0.0% | 0.0% | 90.0% | 0.0% | 0.0% |
| **Conditional Warnings (`conditional_warning`)** | 10 | 10.0% | 100.0% | 90.0% | 0.0% | 0.0% |
| **All Incorrect Cases (Injected Errors)** | 120 | 11.7% | 87.5% | 86.7% | 100.0% | 11.7% |

---

## 4. Breakdown by Injected Error Category (N=120)

| Error Category | N | Detected (TP) | Missed (Explicit FN) | Missed (Uncertain FN) | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`invalid_transformation`** | 29 | 4 | 1 | 24 | 13.8% |
| **`arithmetic_error`** | 26 | 1 | 1 | 24 | 3.9% |
| **`domain_violation`** | 17 | 3 | 0 | 14 | 17.6% |
| **`incorrect_setup`** | 17 | 3 | 0 | 14 | 17.6% |
| **`lost_solution`** | 16 | 2 | 0 | 14 | 12.5% |
| **`division_by_zero`** | 15 | 1 | 0 | 14 | 6.7% |

---

## 5. Symbolic Limitation Boundary Cases (`eval_191` to `eval_200`, N=10)

- **Total Cases**: 10
- **Detected Errors (TP)**: 10 / 10
- **Conservative Recall**: 100.0%
- **Decisive Accuracy**: 100.0%
- **Uncertainty Rate**: 0.0%
