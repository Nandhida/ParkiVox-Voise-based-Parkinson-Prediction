import numpy as np
import pandas as pd
import librosa
import pickle

# -------------------------------
# 1️⃣ Fix librosa-numpy compatibility
# -------------------------------
if not hasattr(np, "complex"):
    np.complex = np.complex128

# -------------------------------
# 2️⃣ Load saved model & scaler
# -------------------------------
model = pickle.load(open("parkinson_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

print("✅ Model and scaler loaded successfully.")

# -------------------------------
# 3️⃣ Feature Extraction (same as main.py)
# -------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    features = {
        "pitch_mean": pitch_mean,
        "rms": rms,
        "zcr": zcr,
        "spec_centroid": spec_centroid,
        "spec_bw": spec_bw,
        "rolloff": rolloff,
    }
    
    for i, mfcc in enumerate(mfccs_mean, 1):
        features[f"mfcc{i}"] = mfcc
        
    return features


# -------------------------------
# 4️⃣ Predict a New Audio File
# -------------------------------
def predict_audio(file_path):
    print(f"\n🎵 Analyzing file: {file_path}")
    features = extract_features(file_path)

    # Ensure the order of features matches training columns
    X_columns = [
        "pitch_mean", "rms", "zcr", "spec_centroid", "spec_bw", "rolloff",
        *[f"mfcc{i}" for i in range(1, 14)]
    ]

    feature_values = [features[f] for f in X_columns]
    sample_df = pd.DataFrame([feature_values], columns=X_columns)
    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]

    print("Prediction:", "🧠 Parkinson's Detected" if prediction == 1 else "✅ No Parkinson's Detected")
    print("Probabitity:", round(probability * 100, 2), "%")

# -------------------------------
# 5️⃣ Run Example Prediction
# -------------------------------
if __name__ == "__main__":
    test_file = r"trial3 with 67\audio_dataset\HC_AH\HC_AH\AH_444B_E1586F09-1BF5-408D-A55E-96D9E8B76A43.wav"  # 🔄 Change to your test file path
    predict_audio(test_file)
