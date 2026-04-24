"""
predict.py
Load trained model and run predictions on new customer data.
Can be used standalone or imported as a module.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from preprocess import ChurnPreprocessor


def load_model(model_path: str = "models/best_model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_preprocessor(path: str = "models/preprocessor.pkl"):
    return ChurnPreprocessor.load(path)


def predict_single(customer_data: dict,
                   model_path: str = "models/best_model.pkl",
                   preprocessor_path: str = "models/preprocessor.pkl") -> dict:
    """
    Predict churn probability for a single customer.
    
    Args:
        customer_data: dict of feature values (see sample below)
    Returns:
        dict with churn prediction and probability
    """
    model = load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)
    
    df = pd.DataFrame([customer_data])
    X = preprocessor.transform(df)
    
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)
    risk_level = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")
    
    return {
        "churn_prediction": pred,
        "churn_probability": round(float(prob), 4),
        "risk_level": risk_level,
        "recommendation": get_recommendation(risk_level, customer_data)
    }


def predict_batch(input_csv: str,
                  output_csv: str = "outputs/predictions.csv",
                  model_path: str = "models/best_model.pkl",
                  preprocessor_path: str = "models/preprocessor.pkl") -> pd.DataFrame:
    """
    Batch predictions from a CSV file.
    """
    model = load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)
    df = pd.read_csv(input_csv)
    
    customer_ids = df.get("CustomerID", pd.Series(range(len(df))))
    X = preprocessor.transform(df)
    
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    results = pd.DataFrame({
        "CustomerID": customer_ids,
        "ChurnPrediction": preds,
        "ChurnProbability": np.round(probs, 4),
        "RiskLevel": pd.cut(probs, bins=[0, 0.4, 0.7, 1.0],
                            labels=["Low", "Medium", "High"])
    })
    
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    results.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    print(f"Churn predicted: {preds.sum()} / {len(preds)} customers ({preds.mean():.1%})")
    return results


def get_recommendation(risk_level: str, customer_data: dict) -> str:
    """Generate a retention recommendation based on risk and customer profile."""
    contract = customer_data.get("Contract", "Month-to-month")
    tenure = customer_data.get("Tenure", 0)
    
    if risk_level == "High":
        if contract == "Month-to-month":
            return "Offer discounted annual contract upgrade immediately."
        elif tenure < 12:
            return "Assign dedicated onboarding support rep and loyalty discount."
        else:
            return "Proactive outreach: personalized retention offer within 48 hrs."
    elif risk_level == "Medium":
        return "Include in next retention campaign; offer loyalty reward."
    else:
        return "No immediate action needed; maintain standard engagement."


def print_results(result: dict):
    print("\n" + "="*45)
    print("   CHURN PREDICTION RESULT")
    print("="*45)
    print(f"  Prediction    : {'WILL CHURN ⚠️' if result['churn_prediction'] else 'WILL STAY ✅'}")
    print(f"  Probability   : {result['churn_probability']:.1%}")
    print(f"  Risk Level    : {result['risk_level']}")
    print(f"  Recommendation: {result['recommendation']}")
    print("="*45)


SAMPLE_CUSTOMER = {
    "CustomerID": "CUST_99999",
    "Gender": "Male",
    "Age": 35,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "Tenure": 6,
    "PhoneService": 1,
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": 1,
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 573.0,
    "NumSupportCalls": 4,
    "AvgCallDuration": 18.5,
    "DataUsageGB": 22.3
}


if __name__ == "__main__":
    print("Running single customer prediction...")
    result = predict_single(SAMPLE_CUSTOMER)
    print_results(result)

    print("\nRunning batch predictions on test set...")
    if os.path.exists("data/test.csv"):
        batch_results = predict_batch("data/test.csv", "outputs/batch_predictions.csv")
        print(batch_results.head(10).to_string(index=False))
