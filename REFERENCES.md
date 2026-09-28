# REFERENCES

Only sources I actually used are listed. AI assistance is recorded separately in [`AI_USE.md`](AI_USE.md).

## Data
- [D1] Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). *Human Activity Recognition Using Smartphones* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K (CC BY 4.0). Files used: `README.txt`, `features_info.txt`, `train/`, `test/`.

## Course material
- [C0] CO3117 HK261, *Individual Longitudinal Assignment – Two-Part Version* (assignment specification).
- [C1] CO3117 lecture slides HK261, *ML-Introduction* — Mitchell's definition, overfitting/underfitting, bias–variance decomposition, train/validation/test and leakage, multi-class metrics.
- [C2] CO3117 lecture slides HK261, *Decision_Tree* — entropy / information gain / Gini, Gain Ratio, continuous attributes, missing values and surrogate splits, pruning and cost-complexity.

## Textbooks
- [T1] T. M. Mitchell (1997), *Machine Learning*, McGraw-Hill, ch. 3 "Decision Tree Learning" — PlayTennis/Wind example used as a unit-test value in `tests/test_impurity.py`.

## Code
- [R1] E. Linder-Norén, *ML-From-Scratch*, https://github.com/eriklindernoren/ML-From-Scratch @ commit `a2806c6732eee8d27762edd6d864e0c179d8e9e8` — read after my first attempt (`2bdd47d`): `mlfromscratch/supervised_learning/decision_tree.py`, `mlfromscratch/utils/data_operation.py`, `mlfromscratch/utils/data_manipulation.py` (functions and line numbers in [`MODEL_LOG.md`](MODEL_LOG.md)).
- [S1] scikit-learn documentation, https://scikit-learn.org/stable/ — `DecisionTreeClassifier`, `cost_complexity_pruning_path`, `GroupKFold`, `LogisticRegression`, `DummyClassifier`, `f1_score`.