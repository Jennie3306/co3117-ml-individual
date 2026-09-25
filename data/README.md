# Data & use case (R0 draft – frozen at tag `release-baseline`)

## 1. Source and version
- **Dataset:** UCI Human Activity Recognition Using Smartphones (UCI id 240), DOI 10.24432/C54S4K, licence CC BY 4.0 — cited as [D1] in REFERENCES.md
- **Download:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones → the outer zip contains `UCI HAR Dataset.zip` → unzip into `data/` so that `data/UCI HAR Dataset/` exists
- **Download date:** [YYYY-MM-DD] · **Integrity check:** [paste the SHA-256 of the zip file]
- Raw data is NOT committed to Git (see `.gitignore`).

## 2. Use case
- **Target variable:** [in your own words]
- **Input:** [..]
- **Decision context:** [who uses the prediction, for what, which errors are more costly]
- **Sequential view (for HMM/CRF – Part II):** [result of the temporal-order check in r0_explore]

## 3. Quick description (fill in from r0_explore.py output)
| Item | Value |
|---|---|
| Train / test windows | |
| Train / test subjects | |
| Number of features | |
| Class distribution (train) | |

## 4. Experimental protocol (FROZEN)
| Item | Decision | Justification |
|---|---|---|
| Sealed test set | Original test split (9 subjects) | [..] |
| Validation | GroupKFold k = 5 by subject on the 21 training subjects | [..] |
| Primary / secondary metrics | Macro-F1 / accuracy + confusion matrix | [..] |
| Seed | 42 | [..] |
| Preprocessing | Every scaler/PCA/feature selection is fitted inside the training fold (Pipeline) | [..] |
| Baselines kept all semester | Dummy (most_frequent) + Logistic Regression (scikit-learn) | [..] |

## 5. Leakage checklist
- [ ] No subject appears in both train and test, or in both a training fold and its validation fold
- [ ] Preprocessing is fitted on training data only
- [ ] The test set is never used for hyperparameter selection
- [ ] [Note: the 561 features were already normalised to [-1, 1] by the dataset authors before splitting — how do you interpret this?]
- [ ] [Windows overlap by 50%: why would a random per-window split leak information?]
