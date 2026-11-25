# Common-Subset Comparative Evaluation Report (187 Cases)

> **Scope**: Exact 187 Common Cases evaluated across all four Phase 4 systems.
> **Generated**: 2026-11-24

---

## 1. Cross-System Comparison Table (187 Cases)

| System / Paradigm | Model / Architecture | Strict Acc | Precision | Recall | F1-Score | Step Loc Acc | TP | FP | TN | Total FN (exp/uns) | Abs (Corr) | Total Cases |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Obj 27: LLM Baseline** | Zero-Shot Direct Prompt | 96.79% | 97.20% | 97.20% | 97.20% | 97.12% | 104 | 3 | 77 | 3 (3/0) | 0 | 187 |
| **Obj 28: Improved LLM** | Improved LLM Verification Prompt | 96.79% | 97.20% | 97.20% | 97.20% | 96.15% | 104 | 3 | 77 | 3 (3/0) | 0 | 187 |
| **Obj 29: Solver-Only** | SymPy AST & Transition Verifier | 66.31% | 90.62% | 54.21% | 67.84% | 84.48% | 58 | 6 | 66 | 49 (48/1) | 8 | 187 |
| **Obj 30: Hybrid System** | Neuro-Symbolic Evidence Fusion | **95.72%** | 93.04% | **100.00%** | **96.40%** | **97.20%** | **107** | 8 | 72 | **0 (0/0)** | 0 | 187 |

*Note on Objective 29 Accounting: The 187 solver-only cases comprise 58 TP + 6 FP + 66 TN + 49 Total FN (48 explicit FN + 1 unsupported FN on `eval_198`) + 8 Abstentions on correct cases (Cohort 2 unsupported conceptual equations: `eval_061`, `eval_062`, `eval_063`, `eval_064`, `eval_066`, `eval_067`, `eval_068`, `eval_069`). The 8 abstentions on correct cases are non-decisive and not included in the classical TP/FP/TN/FN cells, ensuring all 187 cases are explicitly accounted for (58 + 6 + 66 + 48 + 1 + 8 = 187).*

---

## 2. Key Takeaways on the Common 187 Cases

1. **100% Error Recall**: The Hybrid system achieved **100.00% Recall (107/107)**, detecting all injected errors on the common test set.
2. **Elimination of Calculation Blind Spots**: The solver rescued the 3 calculation/transposition errors missed by the LLM (`eval_092`, `eval_105`, `eval_118`).
3. **Semantic Protection**: The LLM rescued all setup and multi-step leaps missed by the solver.
4. **Step Localization Superiority**: The Hybrid system attained **97.20% Step Localization Accuracy** (104/107 matches on True Positives).

---
*Report generated automatically by `evaluate_hybrid.py`.*