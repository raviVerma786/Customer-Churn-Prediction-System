"""
test_churn_pipeline.py
Basic unit tests for the churn prediction pipeline.
Run with: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_generator import generate_churn_dataset
from preprocess import ChurnPreprocessor


@pytest.fixture(scope="module")
def sample_df():
    return generate_churn_dataset(200)


@pytest.fixture(scope="module")
def preprocessor(sample_df):
    prep = ChurnPreprocessor()
    X = sample_df.drop("Churn", axis=1)
    prep.fit(X)
    return prep


class TestDataGenerator:
    def test_shape(self, sample_df):
        assert sample_df.shape[0] == 200
        assert "Churn" in sample_df.columns

    def test_churn_is_binary(self, sample_df):
        assert set(sample_df["Churn"].unique()).issubset({0, 1})

    def test_no_nulls(self, sample_df):
        assert sample_df.isnull().sum().sum() == 0

    def test_tenure_range(self, sample_df):
        assert sample_df["Tenure"].between(1, 72).all()

    def test_monthly_charges_positive(self, sample_df):
        assert (sample_df["MonthlyCharges"] > 0).all()


class TestPreprocessor:
    def test_fit_transform_shape(self, sample_df, preprocessor):
        X = sample_df.drop("Churn", axis=1)
        X_transformed = preprocessor.transform(X)
        assert X_transformed.shape[0] == len(sample_df)
        assert X_transformed.shape[1] > 10  # Feature engineered

    def test_no_nulls_after_transform(self, sample_df, preprocessor):
        X = sample_df.drop("Churn", axis=1)
        X_transformed = preprocessor.transform(X)
        assert X_transformed.isnull().sum().sum() == 0

    def test_engineered_features_exist(self, sample_df, preprocessor):
        X = sample_df.drop("Churn", axis=1)
        X_transformed = preprocessor.transform(X)
        engineered = ["ChargesPerMonth", "ServiceCount", "SupportCallRate"]
        for feat in engineered:
            assert feat in X_transformed.columns, f"Missing feature: {feat}"

    def test_transform_consistency(self, sample_df, preprocessor):
        X = sample_df.drop("Churn", axis=1)
        result1 = preprocessor.transform(X)
        result2 = preprocessor.transform(X)
        pd.testing.assert_frame_equal(result1, result2)


class TestPredictionOutput:
    def test_probabilities_range(self, sample_df, preprocessor):
        from sklearn.ensemble import RandomForestClassifier
        X = sample_df.drop("Churn", axis=1)
        y = sample_df["Churn"]
        X_proc = preprocessor.transform(X)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_proc, y)
        probs = model.predict_proba(X_proc)[:, 1]
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_predictions_binary(self, sample_df, preprocessor):
        from sklearn.ensemble import RandomForestClassifier
        X = sample_df.drop("Churn", axis=1)
        y = sample_df["Churn"]
        X_proc = preprocessor.transform(X)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_proc, y)
        preds = model.predict(X_proc)
        assert set(preds).issubset({0, 1})
