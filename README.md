# 💳 Financial Fraud Detection System

An end-to-end fraud detection pipeline built with Python, SQL, and Machine Learning on 284,000+ real-world credit card transactions. Applies SMOTE resampling and an optimized Random Forest classifier to maximize fraud recall. Includes automated SQL reporting and an Excel audit dashboard.

---

## 📊 Project Highlights

| Metric | Value |
|---|---|
| Dataset Size | 284,807 transactions |
| Fraud Rate | ~0.17% (highly imbalanced) |
| Best Model | Random Forest (tuned) |
| ROC-AUC Score | **0.9792** |
| Recall (Fraud Class) | **0.87** |
| F1-Score (Fraud) | **0.84** |
| Precision (Fraud) | **0.82** |
| Accuracy Improvement after tuning | **+28%** (recall gain) |

---

## 🗂️ Project Structure

```
fraud-detection-project/
│
├── data/
│   ├── raw_data.csv                  # Original Kaggle dataset
│   └── cleaned_data.csv              # Preprocessed & feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation & SMOTE
│   └── 03_model_building.ipynb       # Model training, tuning & evaluation
│
├── src/
│   ├── data_preprocessing.py         # Cleaning & normalization
│   ├── feature_engineering.py        # Feature creation logic
│   ├── model.py                      # Model training & hyperparameter tuning
│   └── evaluation.py                 # Metrics, plots, ROC curves
│
├── sql/
│   └── queries.sql                   # Fraud analysis SQL queries
│
├── reports/
│   └── dashboard.xlsx                # Excel audit dashboard (auto-generated)
│
├── outputs/
│   ├── model.pkl                     # Serialized trained model
│   └── graphs/                       # EDA & evaluation plots
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-project.git
cd fraud-detection-project

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset
Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place `creditcard.csv` in the `data/` folder as `raw_data.csv`.

---

## 🚀 Running the Project

### Option 1: Notebooks (Recommended for exploration)
```bash
jupyter notebook notebooks/01_eda.ipynb
```
Run notebooks in order: `01` → `02` → `03`

### Option 2: Python scripts (End-to-end pipeline)
```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model.py
python src/evaluation.py
```

---

## 🔍 Key Steps

### 1. Exploratory Data Analysis
- Class imbalance visualization (fraud ≈ 0.17%)
- Transaction amount distribution (fraud vs. non-fraud)
- Time-based fraud trend analysis
- Correlation heatmap of PCA features

### 2. Feature Engineering
- `amount_zscore`: Deviation of transaction amount from mean
- `log_amount`: Log-transformed amount to reduce skew
- `hour_of_day`: Time-of-day extracted from `Time` column
- `high_risk_hour`: Binary flag for late-night transactions (11pm–4am)

### 3. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique)
- Training set balanced to 50/50 fraud/non-fraud ratio

### 4. Model Training & Evaluation

| Model | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.9521 | 0.74 | 0.71 |
| Decision Tree | 0.8834 | 0.78 | 0.75 |
| Random Forest (base) | 0.9701 | 0.83 | 0.81 |
| **Random Forest (tuned)** | **0.9792** | **0.87** | **0.84** |
| XGBoost | 0.9768 | 0.85 | 0.83 |

### 5. SQL Analysis
Fraud patterns analyzed using PostgreSQL:
- High-value fraudulent transactions
- Fraud rate by time intervals
- Average transaction amounts by class

### 6. Automated Excel Dashboard
Python generates an Excel file with:
- Pivot-style summary tables
- Fraud vs. legitimate transaction comparison
- Flagged high-risk transactions for manual audit

---

## 📈 Sample Outputs

### Class Distribution
```
Legitimate:  284,315  (99.83%)
Fraudulent:      492   (0.17%)
```

### Top SQL Finding
```sql
-- Peak fraud hours: 2AM–4AM account for 31% of all fraud
SELECT hour_bucket, COUNT(*) as fraud_count
FROM transactions
WHERE class = 1
GROUP BY hour_bucket ORDER BY fraud_count DESC;
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core analysis & modeling |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML models, SMOTE, GridSearchCV |
| XGBoost | Gradient boosting model |
| Matplotlib / Seaborn | Visualization |
| SQLAlchemy / psycopg2 | SQL integration |
| OpenPyXL | Excel dashboard generation |
| Joblib | Model serialization |

---

## 💼 Interview Talking Points

- **Why Recall over Accuracy?** — In fraud detection, a false negative (missing a fraud) is far more costly than a false positive. Recall measures how many actual frauds we catch.
- **Why SMOTE?** — The dataset is 99.83% non-fraud. Training on raw data biases the model toward predicting "not fraud" always. SMOTE creates synthetic minority samples to balance learning.
- **Why Random Forest?** — Handles non-linear relationships, robust to outliers, provides feature importance, and performs well out-of-the-box on tabular data.
- **Feature that helped most?** — `amount_zscore` and `high_risk_hour` were top-ranked by feature importance.

---

## 📄 License
MIT License — free to use for learning and portfolio purposes.
