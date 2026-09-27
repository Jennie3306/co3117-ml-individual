# Release-day baseline diagnostic W01–W04 — corrections

- **First attempt:** [`exercises/release-baseline-w01-w04.pdf`](release-baseline-w01-w04.pdf) — handwritten, 3 pages, commit [`97fb616`](https://github.com/Jennie3306/co3117-ml-individual/commit/97fb616)
- **Conditions of the attempt:** 2026-09-27, 21:30–22:15, written after reviewing the lecture slides, no Internet, no AI.
- **Question sheet:** `Baseline_W01-W04_EN.docx` — **self-prepared, not provided by the instructor**: the specification requires a release-day diagnostic on W01–W04 foundations + Decision Trees but gives no question set, so the 11 questions were drafted by Claude (questions only, no answers — see `AI_USE.md`). I chose to attempt a **subset of 7 questions** and renumbered them; the numbering below follows my answer sheet. The 4 questions I left out of the diagnostic are answered after review in the appendix.
- **Checked against:** CO3117 slides *ML-Introduction* and *Decision_Tree*; Mitchell (1997) ch.3. Checking was AI-assisted (Claude compared my attempt with the slides; recorded in `AI_USE.md`).

## Summary

| My no. | Sheet no. | Topic | Result |
|---|---|---|---|
| 1a | 1a | Mitchell's definition; T, P, E | ✅ Correct |
| 1b | 1b | Underfitting vs overfitting | ✅ Correct |
| 1c | 1c | Bias–variance decomposition | ✅ Correct |
| 1d | 1f | Confusion-matrix metrics | ⚠️ Numbers correct; "why macro-F1 ≠ accuracy" not answered |
| 2a | 2a | Entropy, IG, Gini; C4.5 Gain Ratio | ✅ Correct (minor: log base not written) |
| 2b | 2c | Missing values | ✅ Correct |
| 2c | 2d | Pre/post-pruning; cost-complexity | ✅ Correct |

Not in my diagnostic (left out by choice; answered after review in the appendix): sheet 1d (three sets / leakage), 1e (learning curves), 2b (best threshold), 2e (split cost / overfitting).

Also missing from the header: date, start time, end time and the closed-book statement required by the rules on the question sheet.

---

## Question 1a — Mitchell's definition; T, P, E
- **My original answer:** quoted Mitchell's definition; T = classify a window of accelerometer + gyroscope signals into 6 activities (supervised classification); E = labelled sensor windows from volunteers; P = accuracy, precision, recall, F1 and macro-F1 on unseen data.
- **Correct / wrong / incomplete:** correct.
- **Source:** ML-Introduction, "Definition of Machine Learning (Mitchell, 1997)".

## Question 1b — Underfitting vs overfitting
- **My original answer:** table — underfitting: model too simple, high train error, validation error high and close to train, remedy = more complex model / more features; overfitting: model too complex, low train error, validation error much higher (large gap), remedy = regularisation, more data, pruning.
- **Correct / wrong / incomplete:** correct.
- **Source:** ML-Introduction, "Overfitting, Underfitting, and Good-fit".

## Question 1c — Bias–variance decomposition
- **My original answer:** with y = f(x) + ε, E[ε] = 0, Var(ε) = σ²: E[(y − f̂(x))²] = (E[f̂(x)] − f(x))² + E[(f̂(x) − E[f̂(x)])²] + σ² = Bias² + Variance + Noise; bias = model too simple (underfitting), variance = sensitivity to the training sample (overfitting), noise = irreducible.
- **Correct / wrong / incomplete:** correct.
- **Source:** ML-Introduction, "Bias-Variance Decomposition", "Bias-Variance: Derivation".

## Question 1d (sheet 1f) — Confusion-matrix metrics
- **My original answer:** Accuracy = 19/25 = 0.76; precision 8/9, 6/8, 5/8; recall 0.8, 0.6, 1; F1 0.842, 0.667, 0.769; macro-F1 = 0.759; weighted-F1 = 0.757; micro-F1 = accuracy = 0.76.
- **Correct / wrong / incomplete:** all numbers correct (precision as decimals: 0.889, 0.750, 0.625). Incomplete: the question also asked *why macro-F1 and accuracy can differ substantially*, which I did not answer.
- **Corrected answer (missing part):** accuracy (= micro-F1) aggregates over samples, so large classes dominate it; macro-F1 averages per-class F1, so every class counts equally. Under class imbalance a model can do well on the majority class and badly on a small class and still get high accuracy, while macro-F1 drops. Here the classes are fairly balanced, so the two are close (0.76 vs 0.759), but macro-F1 still exposes the weak class B (F1 = 0.667).
- **Source:** ML-Introduction, "Classification Metrics: Multi-Class".

## Question 2a — Entropy, IG, Gini; bias of IG; C4.5 correction
- **My original answer:** H(S) = −Σ pᵢ log pᵢ; IG(S, A) = H(S) − Σᵥ (|Sᵥ|/|S|) H(Sᵥ); Gini = 1 − Σ pᵢ²; many-valued attributes split S into tiny, nearly pure subsets, so IG ≈ H(S) although the split does not generalise (like an ID column); C4.5 uses Gain Ratio = IG / SplitInfo with SplitInfo(S, A) = −Σᵥ (|Sᵥ|/|S|) log₂(|Sᵥ|/|S|).
- **Correct / wrong / incomplete:** correct. Minor: the entropy formula should state log base 2 (H in bits; max = 1 for two balanced classes).
- **Source:** Decision_Tree, "Entropy", "Information Gain", "Gini Index", "C4.5 Improvements over ID3".

## Question 2b (sheet 2c) — Handling missing values
- **My original answer:** Naive: drop rows (wasteful) or impute mean/mode (may bias). C4.5 fractional weighting — build: compute IG/GR on samples where A is observed, send samples missing A to every child with weights proportional to branch frequencies; classify: send the sample down all branches with those weights, combine leaf distributions, predict the heaviest class. CART surrogate splits — build: pick the split using observed A, store the top-k other features whose splits best agree; classify: if A is missing route with the best available surrogate, then the next one.
- **Correct / wrong / incomplete:** correct.
- **Source:** Decision_Tree, "Missing Values: Challenges", "Surrogate Splits (CART)".

## Question 2c (sheet 2d) — Pre-pruning vs post-pruning; cost-complexity
- **My original answer:** Pre-pruning: stop when |S| < N_min, gain below threshold, or node nearly pure → majority leaf. Post-pruning (C4.5): grow fully, then bottom-up replace a subtree by a leaf if it is not significantly better (pessimistic error p̂ + z√(p̂(1 − p̂)/N)). R_α(T) = R(T) + α|T|: R(T) = training error, |T| = number of leaves, α = penalty per leaf; α = 0 gives the full tree, larger α gives a smaller tree; weakest-link pruning removes the node with the smallest g(t) = (R(t) − R(T_t)) / (|T_t| − 1), giving a nested sequence of subtrees; choose α by cross-validation (optionally the one-SE rule).
- **Correct / wrong / incomplete:** correct and complete.
- **Source:** Decision_Tree, "Pruning in C4.5", "Cost-Complexity Pruning (CART)", "Choosing α".

---

## Appendix — questions left out of the diagnostic, answered after review
These were not part of my first attempt, so they are not first-attempt evidence; I answer them here because they are exam-relevant (sketch, calculation, cost estimate).

### Sheet 1d — Why three sets? Two ways leakage can occur
- **Answer (written after review):** the training set fits the parameters; the validation set is used for hyper-parameter tuning and model selection (so it becomes optimistically biased); the test set is used once for an unbiased final estimate. When each person contributes many samples, the samples are not independent: (1) a random per-window split puts windows of the same subject in both train and test, so the model learns that person rather than the activity; (2) with 50 % overlapping windows, neighbouring windows share half their raw signal, so near-duplicates end up on both sides. Fix: split by subject (GroupKFold). A third path: fitting scalers / feature selection on the whole dataset before splitting.
- **Source:** ML-Introduction, "Training, Validation, and Test Sets"; my own `data/README.md` §5.
- **Note:** I already applied this in `data/README.md`; for the exam I should practise stating it in 3–4 sentences.

### Sheet 1e — Learning curves for high bias vs high variance
- **Answer (written after review):** *High bias:* training and validation error converge quickly and plateau at a high value with a small gap — more data does not help; use a more complex model or better features. *High variance:* training error is low, validation error much higher (large gap) that narrows as the training set grows — more data helps, as do regularisation and pruning.
- **Source:** ML-Introduction, "Bias-Variance Trade-off: Practical Implications".
- **Note:** key cue to remember — *gap* means variance, *high plateau* means bias. My own depth curve (`results/figures/r0_tree_depth_curve.png`) shows the same pattern along model complexity instead of data size.

### Sheet 2b — Best threshold for a continuous attribute
- **Answer (written after review):** (i) sort the distinct values and use midpoints between neighbours → 9 candidates {1.25, 1.75, …, 5.25}; only midpoints where the label changes (2.25, 2.75, 3.25) can be optimal. (ii) 4 × class 0, 6 × class 1 → H(S) = −0.4 log₂0.4 − 0.6 log₂0.6 = 0.971. (iii) t = 2.25: left [3, 0] H = 0, right [1, 6] H = 0.592 → weighted 0.414 → IG = 0.557. t = 2.75: left [3, 1] H = 0.811, right [1, 5] H = 0.650 → weighted 0.715 → IG = 0.256. t = 3.25: left [4, 1] H = 0.722, right [0, 5] H = 0 → weighted 0.361 → **IG = 0.610 (best)**.
- **Source:** Decision_Tree, "Handling Continuous Attributes"; verified with my own `best_threshold()` → (3.25, 0.610).
- **Note:** this is the only numerical tree question on the sheet and exactly the kind of calculation the written exam asks for — worth redoing by hand without notes.

### Sheet 2e — Cost of the best split at one node; why trees overfit
- **Answer (written after review):** for each of the d features: sort the n values (O(n log n)), then scan the n − 1 midpoints while updating class counts incrementally (O(n)) → O(d · n log n) per node, ≈ 561 × 7000 × 12.8 ≈ 5 × 10⁷ operations and ≈ 3.9 million candidate splits. Recomputing impurity from scratch for each threshold costs O(d · n²) — which is what my own `best_threshold` does (≈ 0.77 s per feature on HAR, ≈ 7 min for one root node). Trees overfit here because (1) millions of candidate splits mean some look good by chance; (2) greedy growth to pure leaves leaves very few samples per deep leaf; (3) 561 correlated features give many near-equivalent, unstable splits; (4) many windows per person let the tree learn subject-specific thresholds.
- **Source:** Decision_Tree, "CART: Handling Continuous Attributes"; my `MODEL_LOG.md` (Speed, Error analysis item 4).
- **Note:** this links the theory to the runtime I measured myself for `best_threshold`.

---

## What I learned from this diagnostic
- **Strong:** definitions and formulas (T/P/E, under/overfitting, bias–variance, entropy/IG/Gini/Gain Ratio, missing values, pruning) and the confusion-matrix arithmetic.
- **Weak:** I computed the metrics in 1d but did not explain *why* macro-F1 and accuracy differ; the questions I left out were exactly the sketch / calculation / cost-estimate ones, which the written exam also asks for — I need to practise them (see appendix).
- **Exam habit to fix:** write date and start/end time, and give at least a short answer to every part of a question.