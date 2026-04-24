"""
preprocess.py
Feature engineering and preprocessing pipeline for the churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import os


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing pipeline:
    - Encodes categorical features
    - Scales numerical features
    - Engineers new features
    """

    BINARY_MAP = {"Yes": 1, "No": 0}
    INTERNET_ADDON_MAP = {"Yes": 1, "No": 0, "No internet service": 0}
    PHONE_LINE_MAP = {"Yes": 1, "No": 0, "No phone service": 0}
    CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    GENDER_MAP = {"Male": 1, "Female": 0}

    CATEGORICAL_COLS = [
        "Gender", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]

    NUMERICAL_COLS = [
        "Age", "Tenure", "MonthlyCharges", "TotalCharges",
        "NumSupportCalls", "AvgCallDuration", "DataUsageGB"
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.le_internet = LabelEncoder()
        self.le_payment = LabelEncoder()
        self.fitted = False

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Gender"] = df["Gender"].map(self.GENDER_MAP)
        df["MultipleLines"] = df["MultipleLines"].map(self.PHONE_LINE_MAP)
        df["Contract"] = df["Contract"].map(self.CONTRACT_MAP)

        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]:
            df[col] = df[col].map(self.INTERNET_ADDON_MAP)

        internet_dummies = pd.get_dummies(df["InternetService"], prefix="Internet")
        df = pd.concat([df.drop("InternetService", axis=1), internet_dummies], axis=1)

        payment_dummies = pd.get_dummies(df["PaymentMethod"], prefix="Payment")
        df = pd.concat([df.drop("PaymentMethod", axis=1), payment_dummies], axis=1)

        return df

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Tenure"] = df["Tenure"].astype(float)
        df["TotalCharges"] = df["TotalCharges"].astype(float)
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
        df["NumSupportCalls"] = df["NumSupportCalls"].astype(float)
        df["ChargesPerMonth"] = df["TotalCharges"] / (df["Tenure"] + 1)
        df["HighValueCustomer"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
        df["LongTenure"] = (df["Tenure"] > 24).astype(int)
        df["ServiceCount"] = (
            df["PhoneService"].astype(int) +
            df["OnlineSecurity"].astype(int) +
            df["OnlineBackup"].astype(int) +
            df["DeviceProtection"].astype(int) +
            df["TechSupport"].astype(int) +
            df["StreamingTV"].astype(int) +
            df["StreamingMovies"].astype(int)
        )
        df["SupportCallRate"] = df["NumSupportCalls"] / (df["Tenure"] + 1)
        return df

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df = self._encode_categoricals(df)
        df = self._feature_engineer(df)
        num_cols = [c for c in self.NUMERICAL_COLS + ["ChargesPerMonth", "ServiceCount", "SupportCallRate"]
                    if c in df.columns]
        self.scaler.fit(df[num_cols])
        self.num_cols_ = num_cols
        self.feature_cols_ = [c for c in df.columns if c not in ["CustomerID", "Churn"]]
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = self._encode_categoricals(df)
        df = self._feature_engineer(df)
        df[self.num_cols_] = self.scaler.transform(df[self.num_cols_])
        missing = [c for c in self.feature_cols_ if c not in df.columns]
        for col in missing:
            df[col] = 0
        return df[self.feature_cols_]

    def save(self, path: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str = "models/preprocessor.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from data_generator import generate_churn_dataset
    df = generate_churn_dataset(500)
    prep = ChurnPreprocessor()
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    prep.fit(X)
    X_transformed = prep.transform(X)
    print("Transformed shape:", X_transformed.shape)
    print("Features:", list(X_transformed.columns))
