# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a **Random Forest Classifier**. Built to handle the real-world challenge of highly imbalanced financial data, with a comprehensive evaluation suite beyond simple accuracy.

---

## 📌 Repository Description

> Detects fraudulent credit card transactions using a Random Forest Classifier — featuring class imbalance analysis, correlation heatmap, and a full evaluation suite including Precision, Recall, F1-Score, MCC, and a Confusion Matrix.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── Credit_Card_Fraud_Detection.ipynb   # Main Jupyter Notebook
├── creditcard.csv                      # Input dataset (transaction records)
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
```

---

## 📊 Dataset

The dataset (`creditcard.csv`) contains anonymized credit card transactions labeled as fraudulent or legitimate.

| Feature | Description |
|---|---|
| `V1` – `V28` | PCA-transformed features (anonymized for confidentiality) |
| `Amount` | Transaction amount in currency units |
| `Class` | **Target variable** — `1` = Fraud, `0` = Normal |

> ⚠️ **Class Imbalance:** Fraudulent transactions make up a very small fraction of all records. The notebook computes the outlier fraction (`len(fraud) / len(valid)`) to quantify this imbalance.

---

## ⚙️ Workflow

### 1. Data Loading & Exploration
- Dataset loaded using `pandas`
- `.head()` and `.describe()` used for initial inspection

### 2. Class Distribution Analysis
- Fraud vs. valid transaction counts printed explicitly
- Outlier fraction (fraud ratio) computed
- Amount statistics compared separately for fraud and valid transactions

### 3. Correlation Analysis
- Full feature correlation matrix computed
- Visualized as a heatmap using `seaborn` (12×9 figure)

### 4. Feature & Target Separation
- Features `X`: all columns except `Class`
- Target `Y`: the `Class` column
- Data converted to NumPy arrays for model compatibility

### 5. Train-Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility

### 6. Model Training
- **Algorithm:** `RandomForestClassifier` (default hyperparameters)
- Trained directly on the raw split without scaling

### 7. Evaluation
A comprehensive set of metrics is reported:

| Metric | Why It Matters for Fraud Detection |
|---|---|
| **Accuracy** | Overall correctness (misleading on imbalanced data) |
| **Precision** | Of predicted frauds, how many are real? |
| **Recall** | Of actual frauds, how many were caught? |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **MCC** | Best single metric for imbalanced binary classification |
| **Confusion Matrix** | Visual breakdown of TP, FP, TN, FN |

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Array operations |
| `matplotlib` | Plotting and grid layouts |
| `seaborn` | Heatmaps (correlation + confusion matrix) |
| `scikit-learn` | Train-test split, model, and all evaluation metrics |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

> **Note:** Update the hardcoded dataset path in the notebook to a relative path:
> ```python
> # Change this:
> data = pd.read_csv('C:\\Study content\\ML\\DATASETS\\creditcard.csv')
>
> # To this:
> data = pd.read_csv('creditcard.csv')
> ```

---

## 📈 Results

The model outputs the following on the test set:
- Printed metrics: Accuracy, Precision, Recall, F1-Score, MCC
- A **Confusion Matrix heatmap** (Blues palette) with `Normal` vs `Fraud` labels

---

## 🔮 Possible Improvements

- Handle class imbalance using **SMOTE**, **undersampling**, or `class_weight='balanced'`
- Tune `RandomForestClassifier` hyperparameters with `GridSearchCV`
- Compare with other algorithms: XGBoost, Isolation Forest, Logistic Regression
- Add **ROC-AUC curve** for a fuller picture of classifier performance
- Use **cross-validation** for more robust evaluation
- Apply **feature scaling** (`StandardScaler`) to the `Amount` column

---

## 📄 License

This project is for learning proposes
