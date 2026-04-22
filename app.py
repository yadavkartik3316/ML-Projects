"""
app.py
------
Flask API for Student Performance Prediction System.

Endpoints:
  GET  /           → serves the frontend UI
  POST /predict    → accepts JSON, returns predicted final score
  GET  /results    → returns model comparison metrics (JSON)
  GET  /images/<name> → serves saved visualization images
"""

import pickle
import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# ── Load trained artifacts ──────────────────────────────────────────────────

# Get absolute path of current file (VERY IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model.pkl is in SAME folder as app.py
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    FEATURES = artifacts["features"]
    RESULTS = artifacts["results"]
    MODELS = artifacts.get(
        "models",
        {
            "Linear Regression": model,
            "Decision Tree": model,
            "Random Forest": model,
        },
    )
    print("✅ Model loaded successfully from:", MODEL_PATH)

except FileNotFoundError:
    model = scaler = FEATURES = RESULTS = MODELS = None
    print("❌ ERROR: model.pkl not found at:", MODEL_PATH)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON:
    {
      "study_hours": 6,
      "attendance": 85,
      "prev_grade": 72,
      "assignment_rate": 90,
      "health": 4,
      "extra_activities": 1,
      "internet_access": 1,
      "parental_edu": 2,
      "family_support": 2
    }
    """

    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)
    algo = data.get("algorithm", "Linear Regression")
    selected_model = MODELS.get(algo, model)

    try:
        row = [float(data[f]) for f in FEATURES]
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400

    # Scale input
    row_scaled = scaler.transform([row])

    # Predict
    score = float(round(selected_model.predict(row_scaled)[0], 2))
    score = max(0.0, min(100.0, score))

    # Grade logic
    grade = (
        "A+" if score >= 90 else
        "A"  if score >= 80 else
        "B"  if score >= 70 else
        "C"  if score >= 60 else
        "D"  if score >= 50 else "F"
    )

    return jsonify({
        "predicted_score": score,
        "grade": grade
    })


@app.route("/results", methods=["GET"])
def results():
    if RESULTS is None:
        return jsonify({"error": "Model not loaded."}), 503
    return jsonify(RESULTS)


@app.route("/images/<filename>")
def images(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static", "img"), filename)


# ── Run App ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
