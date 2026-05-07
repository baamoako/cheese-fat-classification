# cheese-fat-classification
Binary classification of Canadian cheese fat levels using SVC. 81.3% accuracy
# Canadian Cheese Fat Classification

Binary classification of Canadian cheeses as "higher fat" or "lower fat" using supervised machine learning. Built in Python with scikit-learn on the Canadian Cheese Directory dataset (833 samples).

**Final model:** SVC with RBF kernel — **81.3% accuracy, 0.813 weighted F1-score**, beating the baseline classifier by 14 percentage points.

---

## Problem

Canadian cheese manufacturers label products by fat level, but the relationship between observable characteristics (moisture content, milk type, manufacturing method, organic status) and fat classification isn't obvious. This project asks: can a machine learning model learn that relationship reliably enough to be useful for production QA or regulatory compliance checks?

Short answer: yes, though not perfectly.

---

## Results

| Model | Test Accuracy | Weighted F1 | Notes |
|---|---|---|---|
| DummyClassifier (baseline) | ~66% | 0.6725 | Majority class only |
| K-Nearest Neighbours | — | — | Underperformed SVC |
| Decision Tree | — | — | Overfit |
| Random Forest | — | — | Overfit; text features added noise |
| **SVC (tuned)** | **81.3%** | **0.8131** | Best generalisation |

**Higher fat class (positive label):** Precision 0.856, Recall 0.862, F1 0.859  
**Lower fat class:** Precision 0.729, Recall 0.718, F1 0.723

The lower performance on "lower fat" reflects the class imbalance in the test set: 138 higher fat vs. 71 lower fat samples. The macro F1 of 0.791 shows the model handles both classes reasonably despite that gap.

---

## Dataset

**Source:** [Canadian Cheese Directory](https://www.kaggle.com/datasets/noahjanes/canadian-cheese-directory/data)  
**Size:** 833 cheeses, 13 features  
**Split:** 80% train / 20% test (stratified)

Key features used:
- `MoisturePercent` — continuous
- `MilkTypeEn` — nominal (cow, goat, sheep, buffalo, mixed)
- `ManufacturingTypeEn` — nominal
- `MilkTreatmentTypeEn` — nominal (raw, pasteurised, thermised)
- `CategoryTypeEn` — nominal (firm, soft, fresh, etc.)
- `FlavourEn`, `CharacteristicsEn` — text features (vectorised)
- `Organic` — binary
- `ManufacturerProvCode` — nominal (province of origin)

`CheeseId` dropped (unique identifier, no predictive value). `RindTypeEn` dropped per project scope.

---

## Approach

**Preprocessing:**  
- Numerical features scaled with `StandardScaler`  
- Nominal features encoded with `OneHotEncoder`  
- Text features (`FlavourEn`, `CharacteristicsEn`) vectorised with `CountVectorizer`  
- Missing values imputed with `SimpleImputer`  
- Class imbalance addressed with `SMOTE` in the training pipeline

**Model selection:**  
Five classifiers compared via cross-validation: DummyClassifier, KNN, Decision Tree, Random Forest, and SVC. Random Forest overfit heavily when text features were included (training F1 near 1.0, test F1 well below SVC). SVC with RBF kernel generalised best.

**Hyperparameter tuning:**  
`RandomizedSearchCV` over `C` and `gamma` for SVC. Final parameters selected by cross-validated weighted F1.

**Evaluation:**  
Confusion matrix, full classification report, and comparison against baseline reported on held-out test set.

---

## Setup

```bash
git clone https://github.com/your-username/cheese-fat-classification
cd cheese-fat-classification
pip install -r requirements.txt
```

Download `cheese_data.csv` from [Kaggle](https://www.kaggle.com/datasets/noahjanes/canadian-cheese-directory/data) and place it in the project root.

```bash
jupyter notebook cheese_classification.ipynb
```

**Requirements:**
```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
altair
scipy
```

---

## Repo Structure

```
cheese-fat-classification/
├── cheese_classification.ipynb   # Full analysis notebook
├── cheese_data.csv               # Source data (download from Kaggle)
├── requirements.txt
└── README.md
```

---

## What I'd Improve

The class imbalance (roughly 2:1 higher fat to lower fat) hurts recall on the minority class. SMOTE was applied in the pipeline but only partially compensated for this. A larger dataset or targeted oversampling with ADASYN would likely push the lower fat F1 above 0.75. I'd also explore feature importance from a tuned Random Forest on just the structured features (dropping text), since the text columns seem to be the main source of overfitting.

---

## Author

**Bismark Addo Amoako**  
