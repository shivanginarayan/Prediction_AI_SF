# models/predict.py
import os
import pandas as pd
import joblib

_MODEL = None
_META = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def _lazy_load():
    global _MODEL, _META
    if _MODEL is not None:
        return _MODEL, _META
    if not os.path.exists(_MODEL_PATH):
        return None, None
    obj = joblib.load(_MODEL_PATH)          # expects {"pipeline": pipe, "feature_cols": [...], "task": "clf"}
    _MODEL = obj["pipeline"]
    _META = {k: obj[k] for k in obj if k != "pipeline"}
    return _MODEL, _META

def predict_from_features(features: dict) -> dict:
    model, meta = _lazy_load()
    if model is None:
        raise RuntimeError("No trained model found at models/model.pkl (run your KNN.py to create it).")

    cols = meta.get("feature_cols")
    df = pd.DataFrame([features])
    if cols:  # reorder & add any missing cols with zeros
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        df = df[cols]

    # Try probabilities if the classifier supports it
    try:
        proba = model.predict_proba(df)[0]
        labels = getattr(getattr(model, "named_steps", {}).get("knn", model), "classes_", None)
        if labels is None:
            labels = [f"class_{i}" for i in range(len(proba))]
        top = int(max(range(len(proba)), key=lambda i: proba[i]))
        return {
            "prediction": str(labels[top]),
            "proba": {str(labels[i]): float(proba[i]) for i in range(len(proba))},
            "used_features": {c: df.iloc[0][c] for c in df.columns},
        }
    except Exception:
        pred = model.predict(df)[0]
        return {"prediction": str(pred), "used_features": {c: df.iloc[0][c] for c in df.columns}}
