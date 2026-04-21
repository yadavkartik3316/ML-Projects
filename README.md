# 🎓 Student Performance Prediction System

A complete end-to-end Data Science project that predicts a student's final exam score using machine learning.

---

## 📁 Project Structure

```
student_performance/
├── data/
│   └── students.csv          # Generated dataset (1000 rows)
├── models/
│   └── model.pkl             # Trained model + scaler + metadata
├── notebooks/
│   └── eda_report.py         # EDA script (convert to Jupyter)
├── static/
│   ├── css/style.css         # Frontend styles
│   ├── js/main.js            # Frontend logic
│   └── img/                  # Auto-generated visualizations
├── templates/
│   └── index.html            # Main UI
├── app.py                    # Flask API
├── generate_dataset.py       # Synthetic data generator
├── train_model.py            # Full ML pipeline
├── requirements.txt
├── Procfile                  # For Render/Heroku
└── README.md
```

---

## ⚙️ Setup & Run (Local)

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python generate_dataset.py

# 4. Train models (generates model.pkl + all charts)
python train_model.py

# 5. Start Flask server
python app.py
# → Open http://localhost:5000
```

---

## 🔌 API Reference

### POST `/predict`

Predicts the final score for a student.

**Request body (JSON):**
```json
{
  "study_hours":      6,
  "attendance":       85,
  "prev_grade":       72,
  "assignment_rate":  90,
  "health":           4,
  "extra_activities": 1,
  "internet_access":  1,
  "parental_edu":     2,
  "family_support":   2
}
```

| Field | Type | Range / Values |
|---|---|---|
| study_hours | float | 1–10 |
| attendance | float | 0–100 |
| prev_grade | float | 0–100 |
| assignment_rate | float | 0–100 |
| health | int | 1 (poor) – 5 (excellent) |
| extra_activities | int | 0=No, 1=Yes |
| internet_access | int | 0=No, 1=Yes |
| parental_edu | int | 0=none, 1=hs, 2=bachelor, 3=master |
| family_support | int | 0=low, 1=medium, 2=high |

**Response:**
```json
{
  "predicted_score": 78.45,
  "grade": "B"
}
```

### GET `/results`
Returns model comparison metrics (R², MAE, RMSE) as JSON.

---

## 🧪 Postman Testing

1. Open Postman → New Request
2. Method: `POST`
3. URL: `http://localhost:5000/predict`
4. Headers: `Content-Type: application/json`
5. Body → raw → JSON:
```json
{
  "study_hours": 7, "attendance": 90, "prev_grade": 80,
  "assignment_rate": 95, "health": 5, "extra_activities": 1,
  "internet_access": 1, "parental_edu": 3, "family_support": 2
}
```
6. Click **Send** → you should see `predicted_score` and `grade`.

---

## 🤖 ML Pipeline

### Algorithms Used

| Model | Why Chosen |
|---|---|
| Linear Regression | Baseline; interpretable; fast |
| Decision Tree | Captures non-linear patterns; explainable |
| Random Forest | Ensemble; reduces overfitting; best accuracy |

### Pipeline Steps
```
Raw CSV → Missing value imputation (median/mode)
       → Outlier capping (IQR method)
       → Label encoding (ordinal categories)
       → StandardScaler normalization
       → Train/Test split (80/20)
       → GridSearchCV hyperparameter tuning
       → Model evaluation (R², MAE, RMSE)
       → Best model saved as model.pkl
```

### Hyperparameter Tuning
- **Decision Tree**: `max_depth` ∈ {4,6,8,10}, `min_samples_split` ∈ {2,5,10}
- **Random Forest**: `n_estimators` ∈ {100,200}, `max_depth` ∈ {6,10,None}, `min_samples_split` ∈ {2,5}
- Both tuned with `GridSearchCV(cv=5, scoring='r2')`

---

## 🚀 Deployment (Render – Free Tier)

1. Push project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt && python generate_dataset.py && python train_model.py`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: Python 3
5. Click **Deploy** → get your public URL

> **Note:** `models/model.pkl` and `static/img/` are generated at build time via the build command.

---

## 📊 Visualizations Generated

| File | Description |
|---|---|
| `histogram.png` | Distribution of final scores |
| `boxplots.png` | Boxplots for key numeric features |
| `heatmap.png` | Correlation heatmap |
| `scatter.png` | Study hours vs final score |
| `model_comparison.png` | R², MAE, RMSE bar charts |

---

## 📐 Workflow Diagram

```
[Raw Data] → [Clean & Impute] → [Encode Categoricals]
    → [Cap Outliers] → [Normalize] → [EDA & Visualize]
        → [Train 3 Models] → [Tune Hyperparams]
            → [Compare R²/MAE/RMSE] → [Save Best Model]
                → [Flask API] → [Frontend UI]
```
