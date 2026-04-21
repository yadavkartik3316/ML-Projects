# notebooks/eda_report.py
# Run this file to reproduce all EDA outputs interactively.
# Convert to Jupyter: jupytext --to notebook eda_report.py

# %% [markdown]
# # Student Performance – Exploratory Data Analysis
# This notebook walks through every step of the ML pipeline.

# %% 1. Imports
import sys; sys.path.insert(0, "..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/students.csv")
print(df.shape)
df.head()

# %% 2. Missing values
df.isnull().sum()

# %% 3. Descriptive statistics
df.describe()

# %% 4. Correlation matrix
plt.figure(figsize=(10,7))
sns.heatmap(df.select_dtypes(include=np.number).corr(),
            annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# %% 5. Score distribution
df["final_score"].hist(bins=30, color="#4f8ef7", edgecolor="white")
plt.title("Final Score Distribution"); plt.show()

# %% 6. Study hours vs score
plt.scatter(df["study_hours"], df["final_score"], alpha=.4, color="#4f8ef7")
plt.xlabel("Study Hours"); plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score"); plt.show()
