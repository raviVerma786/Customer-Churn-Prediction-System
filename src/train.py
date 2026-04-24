"""
train.py
Trains and compares multiple ML models for customer churn prediction.
Models: Logistic Regression, Random Forest, XGBoost
Outputs: best model, metrics, SHAP feature importances
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, precision_score, recall_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from preprocess import ChurnPreprocessor
from data_generator import generate_churn_dataset, save_data


def load_data(data_dir: str = "data/"):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train_df, test_df


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.1, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=3, random_state=42,
            eval_metric="logloss", verbosity=0)
    }


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def plot_confusion_matrix(y_test, y_pred, model_name: str, output_dir: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names: list, model_name: str, output_dir: str, top_n: int = 15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.nlargest(top_n, "Importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2196F3")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features — {model_name}", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(output_dir, f"feature_importance_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(results: dict, output_dir: str):
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    model_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Customer Churn Prediction", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def shap_analysis(model, X_test_df, output_dir: str):
    if not SHAP_AVAILABLE:
        print("  SHAP not installed, skipping SHAP analysis.")
        return
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, max_display=15)
        plt.title("SHAP Feature Importance (XGBoost)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP plot saved: {path}")
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")


def train_and_evaluate():
    print("=" * 60)
    print("  Customer Churn Prediction — Model Training")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Generate data if not present
    if not os.path.exists("data/train.csv"):
        print("\n[1/5] Generating dataset...")
        df = generate_churn_dataset(5000)
        save_data(df)
    else:
        print("\n[1/5] Loading existing dataset...")

    train_df, test_df = load_data()
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"  Churn rate (train): {train_df['Churn'].mean():.2%}")

    # Preprocess
    print("\n[2/5] Preprocessing...")
    preprocessor = ChurnPreprocessor()
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    preprocessor.save("models/preprocessor.pkl")
    print(f"  Feature matrix shape: {X_train_proc.shape}")

    # Train
    print("\n[3/5] Training models...")
    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring="roc_auc")
        print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        model.fit(X_train_proc, y_train)
        metrics = evaluate_model(model, X_test_proc, y_test)
        results[name] = metrics
        trained_models[name] = model
        print(f"  Test Metrics: {metrics}")

    # Save best model
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_model_name]
    print(f"\n[4/5] Best model: {best_model_name} (AUC={results[best_model_name]['roc_auc']:.4f})")

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/all_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)

    results_summary = {"best_model": best_model_name, "metrics": results}
    with open("outputs/results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Plots
    print("\n[5/5] Generating visualizations...")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test_proc)
        plot_confusion_matrix(y_test, y_pred, name, "outputs")
        plot_feature_importance(model, list(X_train_proc.columns), name, "outputs")

    plot_model_comparison(results, "outputs")

    if best_model_name == "XGBoost":
        shap_analysis(best_model, X_test_proc, "outputs")

    # Print classification report for best model
    y_pred_best = best_model.predict(X_test_proc)
    print(f"\n{'='*60}")
    print(f"  Classification Report — {best_model_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"]))

    print("\n✅ Training complete! Check outputs/ and models/ directories.")
    return trained_models, preprocessor, results


if __name__ == "__main__":
    train_and_evaluate()
