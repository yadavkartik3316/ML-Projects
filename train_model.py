"""
train_model.py
--------------
End-to-end ML pipeline:
  1. Load & inspect data
  2. Data cleaning / wrangling
  3. Outlier handling (IQR)
  4. Encoding categorical variables
  5. Normalization (StandardScaler)
  6. Descriptive statistics
  7. Data visualization (saved to static/img/)
  8. Train 3 models + hyperparameter tuning
  9. Model comparison (R², MAE, RMSE)
 10. Save best model + scaler + feature list as pickle
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("static/img", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data/students.csv")
print("Shape:", df.shape)
print(df.head())

# ─────────────────────────────────────────────
# 2. DATA WRANGLING & CLEANING
# ─────────────────────────────────────────────
print("\nMissing values:\n", df.isnull().sum())

# Fill numeric NaN with median
num_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical NaN with mode
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing after imputation:", df.isnull().sum().sum())

# ─────────────────────────────────────────────
# 3. OUTLIER HANDLING (IQR capping)
# ─────────────────────────────────────────────
def cap_outliers_iqr(data, col):
    Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    data[col] = data[col].clip(lower, upper)

for col in ["study_hours", "attendance", "prev_grade", "assignment_rate"]:
    cap_outliers_iqr(df, col)  # final_score excluded to preserve high-score range

# ─────────────────────────────────────────────
# 4. ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────
edu_order = {"none": 0, "high_school": 1, "bachelor": 2, "master": 3}
df["parental_edu"] = df["parental_edu"].map(edu_order)

support_order = {"low": 0, "medium": 1, "high": 2}
df["family_support"] = df["family_support"].map(support_order)

# ─────────────────────────────────────────────
# 5. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
print("\nDescriptive Statistics:\n", df.describe())
print("\nCorrelation with final_score:\n",
      df.corr()["final_score"].sort_values(ascending=False))

# ─────────────────────────────────────────────
# 6. DATA VISUALIZATION
# ─────────────────────────────────────────────
FEATURES = ["study_hours", "attendance", "prev_grade",
            "assignment_rate", "health", "extra_activities",
            "internet_access", "parental_edu", "family_support"]
TARGET = "final_score"

# 6a. Histogram of final score
plt.figure(figsize=(7, 4))
plt.hist(df[TARGET], bins=30, color="#4f8ef7", edgecolor="white")
plt.title("Distribution of Final Score")
plt.xlabel("Final Score"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/img/histogram.png", dpi=100); plt.close()

# 6b. Boxplots for numeric features
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
box_cols = ["study_hours", "attendance", "prev_grade",
            "assignment_rate", "health", "final_score"]
for ax, col in zip(axes.flat, box_cols):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor="#4f8ef7", color="#333"))
    ax.set_title(col); ax.set_xlabel("")
plt.suptitle("Boxplots of Key Features", fontsize=14)
plt.tight_layout()
plt.savefig("static/img/boxplots.png", dpi=100); plt.close()

# 6c. Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 8})
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/img/heatmap.png", dpi=100); plt.close()

# 6d. Scatter: study_hours vs final_score
plt.figure(figsize=(7, 4))
plt.scatter(df["study_hours"], df[TARGET], alpha=0.4, color="#4f8ef7")
plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours"); plt.ylabel("Final Score")
plt.tight_layout()
plt.savefig("static/img/scatter.png", dpi=100); plt.close()

print("Visualizations saved to static/img/")

# ─────────────────────────────────────────────
# 7. FEATURE / TARGET SPLIT + NORMALIZATION
# ─────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 8. TRAIN MODELS + HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

# --- Linear Regression (no tuning needed) ---
lr = LinearRegression()
lr.fit(X_train_sc, y_train)

# --- Decision Tree (GridSearchCV) ---
dt_params = {"max_depth": [4, 6, 8, 10], "min_samples_split": [2, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42),
                       dt_params, cv=5, scoring="r2", n_jobs=-1)
dt_grid.fit(X_train_sc, y_train)
dt = dt_grid.best_estimator_
print("Best DT params:", dt_grid.best_params_)

# --- Random Forest (GridSearchCV) ---
rf_params = {"n_estimators": [100, 200], "max_depth": [6, 10, None],
             "min_samples_split": [2, 5]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                       rf_params, cv=5, scoring="r2", n_jobs=-1)
rf_grid.fit(X_train_sc, y_train)
rf = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)

# ─────────────────────────────────────────────
# 9. MODEL COMPARISON
# ─────────────────────────────────────────────
def evaluate(name, model, X_t, y_t):
    pred = model.predict(X_t)
    return {
        "Model": name,
        "R2":    round(r2_score(y_t, pred), 4),
        "MAE":   round(mean_absolute_error(y_t, pred), 4),
        "RMSE":  round(np.sqrt(mean_squared_error(y_t, pred)), 4),
    }

results = pd.DataFrame([
    evaluate("Linear Regression",  lr, X_test_sc, y_test),
    evaluate("Decision Tree",      dt, X_test_sc, y_test),
    evaluate("Random Forest",      rf, X_test_sc, y_test),
])
print("\nModel Comparison:\n", results.to_string(index=False))

# Save comparison chart
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
metrics = ["R2", "MAE", "RMSE"]
colors  = ["#4f8ef7", "#f7874f", "#4ff7a0"]
for ax, metric, color in zip(axes, metrics, colors):
    ax.bar(results["Model"], results[metric], color=color, edgecolor="white")
    ax.set_title(metric); ax.set_xticklabels(results["Model"], rotation=15, ha="right")
plt.suptitle("Model Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("static/img/model_comparison.png", dpi=100); plt.close()

# ─────────────────────────────────────────────
# 10. SAVE BEST MODEL + ARTIFACTS
# ─────────────────────────────────────────────
best_row   = results.loc[results["R2"].idxmax()]
best_name  = best_row["Model"]
best_model = {"Linear Regression": lr, "Decision Tree": dt, "Random Forest": rf}[best_name]
print(f"\nBest model: {best_name}  (R²={best_row['R2']})")

artifacts = {
    "model":    best_model,
    "scaler":   scaler,
    "features": FEATURES,
    "results":  results.to_dict(orient="records"),
    "models":   {"Linear Regression": lr, "Decision Tree": dt, "Random Forest": rf},
}
with open("models/model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("Artifacts saved -> models/model.pkl")
