"""
generate_dataset.py
-------------------
Generates a realistic synthetic student performance dataset and saves it to data/students.csv
"""

import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.random.seed(42)
N = 1000

study_hours       = np.random.uniform(1, 10, N)
attendance        = np.random.uniform(40, 100, N)
prev_grade        = np.random.uniform(40, 100, N)
assignment_rate   = np.random.uniform(30, 100, N)
health            = np.random.choice([1, 2, 3, 4, 5], N)          # 1=poor, 5=excellent
extra_activities  = np.random.choice([0, 1], N)                    # 0=No, 1=Yes
internet_access   = np.random.choice([0, 1], N)                    # 0=No, 1=Yes
parental_edu      = np.random.choice(
    ["none", "high_school", "bachelor", "master"], N
)
family_support    = np.random.choice(["low", "medium", "high"], N)

# Encode support factors for score formula
parental_map = {"none": 0, "high_school": 1, "bachelor": 2, "master": 3}
support_map  = {"low": 0, "medium": 1, "high": 2}
parental_num = np.array([parental_map[p] for p in parental_edu])   # 0-3
support_num  = np.array([support_map[s]  for s in family_support]) # 0-2

# Score formula: each component normalized to 0-100 range
# Weights sum to 1.0 so max raw score = 100
score = (
    0.30 * (study_hours / 10 * 100)          # 0-100  (most impactful)
    + 0.25 * attendance                       # 0-100
    + 0.20 * prev_grade                       # 0-100
    + 0.12 * assignment_rate                  # 0-100
    + 0.05 * (health / 5 * 100)              # 0-100
    + 0.04 * (extra_activities * 100)         # 0 or 100
    + 0.02 * (internet_access * 100)          # 0 or 100
    + 0.01 * (parental_num / 3 * 100)        # 0-100
    + 0.01 * (support_num  / 2 * 100)        # 0-100
    + np.random.normal(0, 4, N)              # small realistic noise
)
score = np.clip(score, 0, 100).round(2)

df = pd.DataFrame({
    "study_hours":       study_hours.round(2),
    "attendance":        attendance.round(2),
    "prev_grade":        prev_grade.round(2),
    "assignment_rate":   assignment_rate.round(2),
    "health":            health,
    "extra_activities":  extra_activities,
    "internet_access":   internet_access,
    "parental_edu":      parental_edu,
    "family_support":    family_support,
    "final_score":       score,
})

# Inject ~5 % missing values for realism
for col in ["study_hours", "attendance", "assignment_rate", "parental_edu"]:
    mask = np.random.choice([True, False], N, p=[0.05, 0.95])
    df.loc[mask, col] = np.nan

df.to_csv("data/students.csv", index=False)
print(f"Dataset saved -> data/students.csv  ({len(df)} rows)")
