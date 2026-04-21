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

import pickle, os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# ── Load trained artifacts ──────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "model.pkl")

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()
    model    = artifacts["model"]
    scaler   = artifacts["scaler"]
    FEATURES = artifacts["features"]
    RESULTS  = artifacts["results"]
    print("Model loaded successfully.")
except FileNotFoundError:
    model = scaler = FEATURES = RESULTS = None
    print("WARNING: model.pkl not found. Run train_model.py first.")

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
      "parental_edu": 2,       # 0=none,1=hs,2=bachelor,3=master
      "family_support": 2      # 0=low,1=medium,2=high
    }
    Returns: { "predicted_score": 78.45, "grade": "B" }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)

    # Validate & extract features in correct order
    try:
        row = [float(data[f]) for f in FEATURES]
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400

    row_scaled = scaler.transform([row])
    score = float(round(model.predict(row_scaled)[0], 2))
    score = max(0.0, min(100.0, score))   # clamp to [0, 100]

    grade = (
        "A+" if score >= 90 else
        "A"  if score >= 80 else
        "B"  if score >= 70 else
        "C"  if score >= 60 else
        "D"  if score >= 50 else "F"
    )

    return jsonify({"predicted_score": score, "grade": grade})


@app.route("/results", methods=["GET"])
def results():
    """Returns model comparison metrics as JSON."""
    if RESULTS is None:
        return jsonify({"error": "Model not loaded."}), 503
    return jsonify(RESULTS)


@app.route("/images/<filename>")
def images(filename):
    """Serve visualization images."""
    return send_from_directory(os.path.join("static", "img"), filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
