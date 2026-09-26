# Data & use case (R0 draft – frozen at tag `release-baseline`)

## 1. Source and version
- **Dataset:** UCI Human Activity Recognition Using Smartphones (UCI id 240), DOI 10.24432/C54S4K, licence CC BY 4.0 — cited as [D1] in REFERENCES.md
- **Download:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones → the outer zip contains `UCI HAR Dataset.zip` → unzip into `data/` so that `data/UCI HAR Dataset/` exists
- **Download date:** 2026-09-25 · **Integrity check:** SHA-256 `C00B803081A5C797CD5E4B83700A9810B38D53D9D84E01917E090E1FDBC81031`
- Raw data is NOT committed to Git (see `.gitignore`).

## 2. Use case
- **Target variable:** The physical activity performed during each 2.56-second window (128 readings at 50 Hz), one of six classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
- **Input:** For each window, a 561-dimensional vector of time- and frequency-domain features computed by the dataset authors from the 3-axial accelerometer and gyroscope of a waist-mounted smartphone (Samsung Galaxy S II). The raw inertial signals are also provided but are not used at R0.
- **Decision context:** A daily-activity monitoring app for elderly people living alone summarises active versus sedentary time and flags unusually long lying periods during the day. Confusing SITTING with STANDING is relatively cheap because both count as sedentary time, whereas confusing LAYING with an upright posture could miss an alert, and confusing walking with static activities distorts the activity summary. Because every activity class matters to the user, per-class performance is more important than overall accuracy.
- **Sequential view (for HMM/CRF – Part II):** In the training file, each subject's windows form one contiguous block (21 blocks for 21 subjects). Within a subject, the label changes only 12–15 times across roughly 300–350 windows (checked for subjects 1, 3 and 5), which suggests that the rows are in temporal order and that each activity was performed about twice. Consecutive windows therefore usually share the same label, which an HMM could exploit to smooth independent predictions. To be verified for all 21 subjects before Part II.

## 3. Quick description
| Item | Value |
|---|---|
| Train / test windows | 7352 / 2947 |
| Train / test subjects | 21 / 9 (no overlap) |
| Number of features | 561 (values in [-1, 1]) |
| Class distribution (train) | WALKING 1226 (16.7%), WALKING_UPSTAIRS 1073 (14.6%), WALKING_DOWNSTAIRS 986 (13.4%), SITTING 1286 (17.5%), STANDING 1374 (18.7%), LAYING 1407 (19.1%) |
| Windows per training subject | 281–409 |

## 4. Experimental protocol (FROZEN)
| Item | Decision | Justification |
|---|---|---|
| Sealed test set | Original test split (9 subjects) | The official split is already subject-disjoint, so it simulates deployment on people the model has never seen. It is used only once, for the final cross-model comparison in Part II, so that no model or hyperparameter is chosen by looking at it. |
| Validation | GroupKFold k = 5 by subject on the 21 training subjects | Windows from the same person are highly similar and consecutive windows overlap by 50%, so a random per-window split would place near-duplicates in both training and validation and inflate the score. Grouping by subject measures generalisation to new people, which is the real use case. Five folds leave about 4 subjects per validation fold, and GroupKFold is deterministic, so the folds are reproducible. |
| Primary / secondary metrics | Macro-F1 / accuracy + confusion matrix | The classes are only mildly imbalanced (13.4% to 19.1%), but macro-F1 weights every activity equally, so a model cannot score well by favouring the larger classes. This matches the decision context, where each activity matters. Accuracy and the confusion matrix are reported to show which activities are confused (e.g. SITTING vs STANDING). |
| Seed | 42 | A fixed seed makes every stochastic step (model initialisation, sampling) reproducible across runs and weeks. The value itself is arbitrary; what matters is that it never changes. |
| Preprocessing | Every scaler/PCA/feature selection is fitted inside the training fold (Pipeline) | Placing each transformation inside a scikit-learn Pipeline means it is fitted only on the training part of each fold, so statistics of the validation (or test) subjects never leak into training. |
| Baselines kept all semester | Dummy (most_frequent) + Logistic Regression (scikit-learn) | The dummy classifier gives the floor: it always predicts LAYING, reaching accuracy 0.191 (the share of LAYING) but macro-F1 only 0.054. Logistic regression is a strong, simple linear reference: macro-F1 = 0.934 ± 0.042 over the 5 subject folds. The large fold-to-fold spread shows that some groups of subjects are much harder than others, so later model comparisons must consider this variability rather than small differences in the mean. |

## 5. Leakage checklist
- [x] No subject appears in both train and test, or in both a training fold and its validation fold (verified by `test_train_test_subjects_disjoint` and `test_folds_have_no_subject_leakage` in `tests/test_data.py`).
- [x] Preprocessing is fitted on training data only (`StandardScaler` is placed inside `make_pipeline`, so it is re-fitted on the training part of every fold).
- [x] The test set is never used for hyperparameter selection (`load_test()` raises an error unless `confirm_sealed_test=True`; all tuning uses subject-wise cross-validation on the training set).
- [x] Acknowledged: the dataset authors normalised all 561 features to [-1, 1] before releasing the train/test split, so the scaling bounds were computed with test subjects included. This is outside my control and its effect is expected to be small, because it is only a fixed linear rescaling of each feature and carries no label information. All preprocessing I add myself is fitted on training folds only.
- [x] Consecutive windows overlap by 50%, so neighbouring windows share half of their raw signal. A random per-window split would therefore put almost identical windows in both training and validation, letting the model "recognise" the signal instead of generalising, and would inflate the scores. Splitting by subject avoids this.
