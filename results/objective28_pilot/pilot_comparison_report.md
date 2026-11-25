# Phase 4 Objective 28: Improved vs Baseline Comparative Pilot Report

**Cases Evaluated**: 20 identical cases

## 1. Summary Metrics Comparison

| Metric | Baseline (Obj 27) | Improved (Obj 28) | Delta |
| :--- | :--- | :--- | :--- |
| **Strict Accuracy** | 80.0% | 80.0% | +0.0% |
| **Error Precision** | 87.5% | 87.5% | +0.0% |
| **Conservative Recall** | 87.5% | 87.5% | +0.0% |
| **Error F1-Score** | 87.5% | 87.5% | +0.0% |
| **Step Localization** | 100.0% | 92.9% | -7.1% |

## 2. Confusion Matrix Comparison

| Confusion Cell | Baseline | Improved | Delta |
| :--- | :--- | :--- | :--- |
| **True Positives (TP)** | 14 | 14 | +0 |
| **False Positives (FP)** | 2 | 2 | +0 |
| **True Negatives (TN)** | 2 | 2 | +0 |
| **False Negatives (FN_explicit)** | 2 | 2 | +0 |
| **False Negatives (FN_uncertain)** | 0 | 0 | +0 |

## 3. Discrepancies & Case-by-Case Corrections

- **Cases Agreed**: 19 / 20 (95.0%)
- **Cases Disagreed**: 1 / 20
- **Baseline Errors Corrected**: 0
- **New Regressions Introduced**: 1

### Regressions Introduced by Improved Detector
- **eval_196** (Step Match -> Step Mismatch): Baseline predicted `incorrect` (step 2), Improved predicted `incorrect` (step 3).
  - *Improved Explanation*: The problem describes variable shadowing, where the second assignment of 'x' replaces the first, meaning the original value of 'x' (apples) is no longer accessible. Step 3 incorrectly treats the two assignments as additive quantities rather than a sequential update of the same variable.
