import pandas as pd
import logging

logger = logging.getLogger(__name__)
FEATURES = ["LOC", "Complexity", "CodeChurn", "Commits", "Developers"]

def classify_risk(prob: float, high_thresh: float = 0.7, medium_thresh: float = 0.4) -> str:
    if prob > high_thresh:   return "High"
    elif prob > medium_thresh: return "Medium"
    return "Low"

def classify_severity(prob: float) -> str:
    if prob > 0.8:   return "Critical"
    elif prob > 0.6: return "High"
    elif prob > 0.4: return "Medium"
    return "Low"

def predict(df: pd.DataFrame, model) -> pd.DataFrame:
    X = df[FEATURES]
    df = df.copy()
    df["Prediction"]  = model.predict(X)
    df["Probability"] = model.predict_proba(X)[:, 1].round(4)
    df["Risk"]        = df["Probability"].apply(classify_risk)
    df["Severity"]    = df["Probability"].apply(classify_severity)
    return df

def predict_single(features: dict, model, high_thresh: float = 0.7, medium_thresh: float = 0.4) -> dict:
    X = [[float(features["LOC"]), float(features["Complexity"]),
          float(features["CodeChurn"]), float(features["Commits"]),
          float(features["Developers"])]]
    prob = round(float(model.predict_proba(X)[0][1]), 4)
    pred = int(model.predict(X)[0])
    return {
        "prediction":  pred,
        "probability": prob,
        "risk":        classify_risk(prob, high_thresh, medium_thresh),
        "severity":    classify_severity(prob),
    }