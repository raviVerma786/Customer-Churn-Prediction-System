# 📉 Customer Churn Prediction System

> **End-to-end ML pipeline** to predict telecom customer churn using Python, XGBoost, and Streamlit.

---

## 🗂️ Project Structure

```
customer_churn_prediction/
├── app.py                    # Streamlit web dashboard
├── requirements.txt
├── src/
│   ├── data_generator.py     # Synthetic dataset generation
│   ├── preprocess.py         # Feature engineering + preprocessing pipeline
│   ├── train.py              # Model training + evaluation + visualizations
│   └── predict.py            # Inference (single + batch)
├── models/                   # Saved models (generated after training)
├── data/                     # Train/test CSVs (generated after training)
├── outputs/                  # Metrics, plots, predictions
└── tests/
    └── test_churn_pipeline.py
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
cd src
python train.py
```
This will:
- Generate a 5,000-row synthetic telecom dataset
- Preprocess and engineer features
- Train 3 models (Logistic Regression, Random Forest, XGBoost)
- Save the best model + all evaluation plots

### 3. Run the dashboard
```bash
streamlit run app.py
```

### 4. Run a single prediction
```bash
cd src
python predict.py
```

### 5. Run tests
```bash
pytest tests/
```

---

## 🧠 Models Trained

| Model               | Strengths                         |
|---------------------|-----------------------------------|
| Logistic Regression | Interpretable baseline            |
| Random Forest       | Handles non-linearity, robust     |
| **XGBoost**         | Best performance, SHAP compatible |

---

## ⚙️ Features Engineered

| Feature            | Description                           |
|--------------------|---------------------------------------|
| `ChargesPerMonth`  | TotalCharges / (Tenure + 1)           |
| `ServiceCount`     | Total services subscribed             |
| `SupportCallRate`  | Support calls relative to tenure      |
| `LongTenure`       | Binary: tenure > 24 months            |
| `HighValueCustomer`| Binary: above-median monthly charges  |

---

## 📊 Evaluation Metrics

- **ROC-AUC**: Discriminative power across thresholds
- **F1 Score**: Balances precision and recall (key for imbalanced churn)
- **Recall**: Critical — minimize missed churners
- **Precision**: Avoid over-targeting stable customers

---

## 💡 Business Impact

This system enables proactive retention:
- 🔴 **High Risk (≥70%)** → Immediate outreach, offer contract upgrade
- 🟡 **Medium Risk (40–70%)** → Retention campaign, loyalty rewards
- 🟢 **Low Risk (<40%)** → Standard engagement

---

## 🛠️ Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `Pandas` · `SHAP` · `Streamlit` · `Matplotlib` · `pytest`

---

## 👤 Author

Built as a portfolio project for ML/AI engineering roles.
