#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# 1) Load data
games = pd.read_csv("Soccer_data_three_seasons.csv")

# 2) Select features/target (your original set)
FEATURES = ['B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA',
            'PSH', 'PSD', 'PSA', 'Avg>2.5', 'Avg<2.5', 'AHCh']
TARGET = 'FTR'  # H/D/A

X = games[FEATURES].apply(pd.to_numeric, errors="coerce")
y = games[TARGET]

# Drop rows with missing values in used columns
data = pd.concat([X, y], axis=1).dropna()
X, y = data[FEATURES], data[TARGET]

# 3) Train/test split (stratify for classification)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Pipeline: Standardize -> KNN
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=8))
])

pipe.fit(X_tr, y_tr)

# 5) Evaluate (on TEST set)
y_pred = pipe.predict(X_te)
print("Accuracy:", accuracy_score(y_te, y_pred))
print("Precision (macro):", precision_score(y_te, y_pred, average='macro', zero_division=0))
print("Recall (macro):", recall_score(y_te, y_pred, average='macro', zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))

# 6) Save portable artifact for the chat app
artifact = {
    "pipeline": pipe,
    "feature_cols": FEATURES,
    "task": "clf",
}
os.makedirs("models", exist_ok=True)
joblib.dump(artifact, "models/model.pkl")
print("Saved -> models/model.pkl")
print("Features:", FEATURES)
