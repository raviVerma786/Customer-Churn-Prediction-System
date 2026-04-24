"""
data_generator.py
Generates a realistic synthetic telecom customer churn dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

def generate_churn_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """Generate a synthetic telecom churn dataset."""

    customer_id = [f"CUST_{str(i).zfill(5)}" for i in range(1, n_samples + 1)]
    tenure = np.random.randint(1, 73, size=n_samples)  # months

    monthly_charges = np.round(
        np.where(tenure < 12,
                 np.random.uniform(30, 60, n_samples),
                 np.random.uniform(50, 120, n_samples)), 2)

    total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.85, 1.05, n_samples), 2)

    age = np.random.randint(18, 75, size=n_samples)
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizen = np.where(age >= 60, 1, 0)
    partner = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    dependents = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    phone_service = np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9])
    multiple_lines = np.where(phone_service == 1,
                              np.random.choice(["No", "Yes"], size=n_samples, p=[0.55, 0.45]),
                              "No phone service")

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.45, 0.20])

    def internet_addon(prob_yes):
        return np.where(internet_service != "No",
                        np.random.choice(["No", "Yes"], size=n_samples, p=[1-prob_yes, prob_yes]),
                        "No internet service")

    online_security = internet_addon(0.35)
    online_backup = internet_addon(0.40)
    device_protection = internet_addon(0.35)
    tech_support = internet_addon(0.30)
    streaming_tv = internet_addon(0.45)
    streaming_movies = internet_addon(0.45)

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples, p=[0.55, 0.25, 0.20])

    paperless_billing = np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples, p=[0.34, 0.23, 0.22, 0.21])

    num_support_calls = np.random.poisson(lam=1.5, size=n_samples)
    avg_call_duration = np.round(np.random.uniform(2, 30, size=n_samples), 1)
    data_usage_gb = np.round(np.random.exponential(scale=15, size=n_samples), 2)

    # Churn logic: realistic correlations
    churn_prob = 0.10
    churn_prob += np.where(contract == "Month-to-month", 0.25, 0)
    churn_prob += np.where(internet_service == "Fiber optic", 0.08, 0)
    churn_prob += np.where(tenure < 12, 0.15, 0)
    churn_prob += np.where(tenure > 48, -0.10, 0)
    churn_prob += np.where(monthly_charges > 90, 0.10, 0)
    churn_prob += np.where(online_security == "No", 0.05, 0)
    churn_prob += np.where(tech_support == "No", 0.04, 0)
    churn_prob += np.where(num_support_calls > 3, 0.08, 0)
    churn_prob += np.where(payment_method == "Electronic check", 0.06, 0)
    churn_prob += np.where(senior_citizen == 1, 0.05, 0)
    churn_prob = np.clip(churn_prob, 0.02, 0.92)

    churn = np.random.binomial(1, churn_prob, size=n_samples)

    df = pd.DataFrame({
        "CustomerID": customer_id,
        "Gender": gender,
        "Age": age,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "NumSupportCalls": num_support_calls,
        "AvgCallDuration": avg_call_duration,
        "DataUsageGB": data_usage_gb,
        "Churn": churn
    })

    return df


def save_data(df: pd.DataFrame, output_dir: str = "data/") -> None:
    df = df.convert_dtypes(dtype_backend="numpy_nullable")
    os.makedirs(output_dir, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Churn"])
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "full_dataset.csv"), index=False)
    print(f"Dataset saved: {len(train_df)} train, {len(test_df)} test samples")
    print(f"Churn rate: {df['Churn'].mean():.2%}")


if __name__ == "__main__":
    df = generate_churn_dataset(5000)
    save_data(df)
    print(df.head())
    print(df.dtypes)
