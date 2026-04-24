"""
app.py
Streamlit dashboard for Customer Churn Prediction.
Run with: streamlit run app.py
"""

import sys
import streamlit as st

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import xgboost
    import shap
    import sklearn
except ImportError as e:
    st.error(f"Missing package: {e}")
    st.stop()

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Always resolve paths relative to this file, regardless of where streamlit is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Auto-train if models not found (for cloud deployment)
if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    from train import train_and_evaluate
    train_and_evaluate()

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div style="position: fixed; top: 0.7rem; z-index: 9999999;text-align: center;left: 50%;transform: translateX(-50%);">
    <span style="font-size: 1.5rem;">📉</span>
    <span style="font-size: 1.4rem; font-weight: 700; color: #4FC3F7;"> Churn Predictor</span>
</div>
""", unsafe_allow_html=True)

# --- Styling ---
st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: 700; color: #1565C0;}
    .metric-box {background: #f0f4ff; border-radius: 10px; padding: 1rem; text-align: center;}
    .high-risk {color: #d32f2f; font-weight: bold;}
    .medium-risk {color: #f57c00; font-weight: bold;}
    .low-risk {color: #388e3c; font-weight: bold;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    [data-testid="stMainMenu"] {display: none !important;}
    # [data-testid="stStatusWidget"] {display: none !important;}
    # last 3 are added to reduce clutter in frontend
    [data-testid="stStatusWidget"] {visibility: hidden !important;}
    .stStatusWidget {visibility: hidden !important;}
    iframe[title="streamlit_analytics"] {display: none !important;}
    [data-testid="stToolbar"] {background-color: #36454F !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    try:
        from preprocess import ChurnPreprocessor
        with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
            model = pickle.load(f)
        preprocessor = ChurnPreprocessor.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
        with open(os.path.join(OUTPUTS_DIR, "results.json")) as f:
            results = json.load(f)
        return model, preprocessor, results
    except FileNotFoundError:
        return None, None, None


def predict(model, preprocessor, customer_data: dict) -> dict:
    df = pd.DataFrame([customer_data])
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)
    risk = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")
    return {"pred": pred, "prob": round(float(prob), 4), "risk": risk}


# --- Sidebar ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <div style="font-size: 4rem;">📉</div>
    <div style="font-size: 0.75rem; color: #888; letter-spacing: 2px; font-weight:bold; text-transform: capitalize;">Churn Predictor</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown("""
<style>
    .stApp { background-color: #0e1117 !important; }
    .stApp * {color: #ffffff !important; }
    section[data-testid="stSidebar"] { background-color: #262730 !important; }
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["🏠 Dashboard", "🔍 Single Prediction", "📊 Model Metrics"])

model, preprocessor, results = load_artifacts()

# ========================
# PAGE: Dashboard
# ========================
if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">📉 Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Predict which telecom customers are likely to churn, and act before they leave.")

    if model is None:
        st.warning("⚠️ Models not found. Please run `python src/train.py` first.")
    else:
        best = results["best_model"]
        best_metrics = results["metrics"][best]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", best.split()[0])
        col2.metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
        col3.metric("F1 Score", f"{best_metrics['f1_score']:.3f}")
        col4.metric("Recall", f"{best_metrics['recall']:.3f}")

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            if os.path.exists(os.path.join(OUTPUTS_DIR, "model_comparison.png")):
                st.image(os.path.join(OUTPUTS_DIR, "model_comparison.png"), caption="Model Comparison", use_container_width=True)
        with col_b:
            best_label = best.replace(" ", "_")
            cm_path = os.path.join(OUTPUTS_DIR, f"confusion_matrix_{best_label}.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"Confusion Matrix — {best}", use_container_width=True)

        shap_path = os.path.join(OUTPUTS_DIR, "shap_summary.png")
        if os.path.exists(shap_path):
            st.image(shap_path, caption="SHAP Feature Importance", use_container_width=True)

# ========================
# PAGE: Single Prediction
# ========================
elif page == "🔍 Single Prediction":
    st.markdown('<p class="main-header">🔍 Predict Individual Customer Churn</p>', unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Models not found. Please run `python src/train.py` first.")
    else:
        with st.form("prediction_form"):
            st.subheader("Customer Details")
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.slider("Age", 18, 75, 35)
                senior = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Has Partner", [0, 1])
                dependents = st.selectbox("Has Dependents", [0, 1])
            with c2:
                tenure = st.slider("Tenure (months)", 1, 72, 6)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                monthly = st.number_input("Monthly Charges ($)", 20.0, 150.0, 95.0)
                total = st.number_input("Total Charges ($)", 20.0, 9000.0, float(monthly * tenure))
            with c3:
                online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                tech_supp = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                payment = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer", "Credit card"])
                support_calls = st.slider("Support Calls", 0, 10, 2)
                paperless = st.selectbox("Paperless Billing", [0, 1])

            submitted = st.form_submit_button("🔮 Predict Churn")

        if submitted:
            customer = {
                "CustomerID": "DEMO_001", "Gender": gender, "Age": age,
                "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "Tenure": tenure, "PhoneService": 1,
                "MultipleLines": "No", "InternetService": internet,
                "OnlineSecurity": online_sec, "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": tech_supp,
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly,
                "TotalCharges": total, "NumSupportCalls": support_calls,
                "AvgCallDuration": 12.0, "DataUsageGB": 15.0
            }
            result = predict(model, preprocessor, customer)

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("Prediction", "Will Churn ⚠️" if result["pred"] else "Will Stay ✅")
            r2.metric("Churn Probability", f"{result['prob']:.1%}")
            r3.metric("Risk Level", result["risk"])

            risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            st.markdown(f"### {risk_color[result['risk']]} Risk: **{result['risk']}**")

            if result["risk"] == "High":
                st.error("💡 **Action:** Offer discounted annual contract or assign retention specialist.")
            elif result["risk"] == "Medium":
                st.warning("💡 **Action:** Include in retention campaign with loyalty reward.")
            else:
                st.success("💡 **Action:** No immediate action needed.")

# ========================
# PAGE: Model Metrics
# ========================
elif page == "📊 Model Metrics":
    st.markdown('<p class="main-header">📊 Model Performance Metrics</p>', unsafe_allow_html=True)

    if results is None:
        st.warning("⚠️ No results found. Run training first.")
    else:
        metrics_data = []
        for name, m in results["metrics"].items():
            row = {"Model": name}
            row.update(m)
            metrics_data.append(row)
        df_metrics = pd.DataFrame(metrics_data).set_index("Model")
        st.dataframe(df_metrics.style.highlight_max(axis=0, color="#c8e6c9"), use_container_width=True)

        st.subheader("Feature Importance Plots")
        for model_name in results["metrics"].keys():
            path = os.path.join(OUTPUTS_DIR, f"feature_importance_{model_name.replace(' ', '_')}.png")
            if os.path.exists(path):
                st.image(path, caption=f"Feature Importance — {model_name}", use_container_width=True)
