import logging
import os
import json
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

from modules.data_loader import load_data
from modules.preprocessing import preprocess_data
from modules.model import load_or_train_model, train_models
from modules.prediction import predict_single, predict
from modules.evaluation import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

os.makedirs("output", exist_ok=True)

HISTORY_FILE = "output/history.json"
DATA_PATH    = "data/data.csv"

# ── Startup ───────────────────────────────────────────────
df_raw = load_data(DATA_PATH)
df     = preprocess_data(df_raw)
rf, lr, X_test, y_test = train_models(df)
model  = load_or_train_model(df)
predictions = predict(df, rf)

metrics = {
    "rf": evaluate_model(rf, X_test, y_test, "Random Forest"),
    "lr": evaluate_model(lr, X_test, y_test, "Logistic Regression"),
}

# ── Helpers ───────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ── Routes ────────────────────────────────────────────────
@app.route("/")
def dashboard():
    results     = predictions.copy()
    risk_counts = results["Risk"].value_counts().to_dict()
    risk_counts = {
        "High":   risk_counts.get("High",   0),
        "Medium": risk_counts.get("Medium", 0),
        "Low":    risk_counts.get("Low",    0),
    }
    top5     = results.nlargest(5, "Probability")[["Module","Probability","Risk"]].to_dict(orient="records")
    all_mods = results[["Module","Probability","Risk"]].to_dict(orient="records")
    stats    = {
        "total":  len(results),
        "high":   risk_counts["High"],
        "medium": risk_counts["Medium"],
        "low":    risk_counts["Low"],
    }
    return render_template("dashboard.html", stats=stats, risk_counts=risk_counts,
                           top_modules=top5, all_modules=all_mods)

@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    required = ["LOC", "Complexity", "CodeChurn", "Commits", "Developers"]
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        result = predict_single(data, model)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"}), 422
    except Exception:
        logger.exception("Prediction error")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/history")
def history_page():
    history = list(reversed(load_history()))
    return render_template("history.html", history=history)

@app.route("/save_history", methods=["POST"])
def save_history_route():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No data"}), 400
    history = load_history()
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    history.append(data)
    save_history(history)
    return jsonify({"status": "saved"})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    save_history([])
    return jsonify({"status": "cleared"})

@app.route("/download_history")
def download_history():
    history = load_history()
    if not history:
        return jsonify({"error": "No history"}), 404
    path = "output/history_export.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    return send_file(path, as_attachment=True)

@app.route("/analytics")
def analytics_page():
    features     = ["LOC", "Complexity", "CodeChurn", "Commits", "Developers"]
    importances  = rf.feature_importances_.tolist()
    correlations = df[features + ["Defect"]].corr()["Defect"].drop("Defect").to_dict()
    return render_template("analytics.html", features=features,
                           importances=importances, metrics=metrics,
                           correlations=correlations)

@app.route("/download", methods=["POST"])
def download():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No data"}), 400
    pd.DataFrame([data]).to_csv("output/user_prediction.csv", index=False)
    return jsonify({"status": "saved"})

if __name__ == "__main__":
    app.run(debug=False)