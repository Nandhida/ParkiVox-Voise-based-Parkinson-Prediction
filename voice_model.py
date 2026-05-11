import os
import zipfile
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -------------------------------
# 1️⃣ Unzip Dataset
# -------------------------------
MAIN_ZIP = "audio recording.zip"
EXTRACT_DIR = "audio_dataset"

# Extract main ZIP
if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(MAIN_ZIP, "r") as main_zip:
        main_zip.extractall(EXTRACT_DIR)
    print("✅ Extracted main dataset.")

# Extract HC and PD subfolders
HC_ZIP = os.path.join(EXTRACT_DIR, "HC_AH.zip")
PD_ZIP = os.path.join(EXTRACT_DIR, "PD_AH.zip")
HC_DIR = os.path.join(EXTRACT_DIR, "HC_AH")
PD_DIR = os.path.join(EXTRACT_DIR, "PD_AH")

if os.path.exists(HC_ZIP):
    with zipfile.ZipFile(HC_ZIP, "r") as z:
        z.extractall(HC_DIR)
if os.path.exists(PD_ZIP):
    with zipfile.ZipFile(PD_ZIP, "r") as z:
        z.extractall(PD_DIR)

# -------------------------------
# 2️⃣ Fix librosa-numpy compatibility
# -------------------------------
if not hasattr(np, "complex"):
    np.complex = np.complex128

# -------------------------------
# 3️⃣ Feature Extraction Function
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
# 4️⃣ Build Dataset
# -------------------------------
rows = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)
            label = 1 if "PD_AH" in file_path else 0  # 1 = Parkinson’s, 0 = Healthy
            try:
                features = extract_features(file_path)
                features["status"] = label
                rows.append(features)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

df = pd.DataFrame(rows)
print("\n✅ Dataset created!")
print("Shape:", df.shape)
print(df.head())

# -------------------------------
# 5️⃣ Train/Test Split & Scaling
# -------------------------------
X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6️⃣ Train Random Forest Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# -------------------------------
# 7️⃣ Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_scaled)
print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 8️⃣ Save Model & Scaler
# -------------------------------
pickle.dump(model, open("parkinson_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("\n💾 Model saved as 'parkinson_model.pkl' and scaler as 'scaler.pkl'")

# -------------------------------
# 9️⃣ Predict New Audio File
# -------------------------------
def predict_audio(file_path):
    features = extract_features(file_path)
    feature_values = [features[f] for f in X.columns]
    sample_df = pd.DataFrame([feature_values], columns=X.columns)
    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]

    print(f"\n🎵 File: {file_path}")
    print("Prediction:", "🧠 Parkinson's Detected" if prediction == 1 else "✅ No Parkinson's Detected")
    print("Confidence:", round(probability * 100, 2), "%")

# Example usage (update with a real file path)
# predict_audio("audio_dataset/PD_AH/sample1.wav")
