import os
import pickle
import tempfile
import numpy as np
from flask import Blueprint, request, jsonify
from feature_extraction import extract_features

# ---------------- Flask Blueprint ----------------
voice_bp = Blueprint("voice", __name__)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "parkinson_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# ---------------- Global variables ----------------
model, scaler, load_error = None, None, None

# ---------------- Feature order (match trained model) ----------------
FEATURE_ORDER = [
    "pitch_mean", "rms", "zcr", "spec_centroid", "spec_bw", "rolloff",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7",
    "mfcc8", "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13"
]

# ---------------- Load Model and Scaler ----------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded:", type(model))

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded:", type(scaler))

except Exception as e:
    load_error = f"Failed to load model/scaler: {e}"
    print(load_error)


# ---------------- Prediction Function ----------------
def predict_parkinsons(feature_values):
    """
    feature_values: list of features in the same order as FEATURE_ORDER
    Returns: predicted label (0/1) and probability
    """
    global model, scaler
    if model is None:
        raise ValueError("Model not loaded properly!")

    X = np.array([feature_values])

    # Apply scaler if exists
    if scaler is not None:
        X = scaler.transform(X)

    # Predict label
    pred = model.predict(X)[0]

    # Predict probability (label=1)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
    else:
        prob = 0.0

    return int(pred), float(prob)
