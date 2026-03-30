import joblib
import os
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

FEATURES = ["LOC", "Complexity", "CodeChurn", "Commits", "Developers"]
TARGET   = "Defect"
MODEL_PATH = "output/model.joblib"


def train_models(df: pd.DataFrame):
    """
    Train a Random Forest and a Logistic Regression on the dataset.
    Returns (rf_model, lr_pipeline, X_test, y_test).
    """
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    # Logistic Regression inside a scaling pipeline
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    lr.fit(X_train, y_train)

    # Cross-validation score
    cv_score = cross_val_score(rf, X, y, cv=5, scoring="f1").mean()
    logger.info(f"Random Forest 5-fold CV F1: {cv_score:.3f}")

    # Save best model (RF) to disk
    os.makedirs("output", exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    return rf, lr, X_test, y_test


def load_or_train_model(df: pd.DataFrame):
    """
    Load the saved model from disk if available, otherwise train a fresh one.
    Used by the Flask app so training doesn't happen on every restart.
    """
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading saved model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    logger.info("No saved model found. Training a new one...")
    df_clean = df.copy()
    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    os.makedirs("output", exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    return rf