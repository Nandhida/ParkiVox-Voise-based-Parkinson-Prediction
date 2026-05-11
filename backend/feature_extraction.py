import numpy as np
import librosa

FEATURE_ORDER = [
    "pitch_mean", "rms", "zcr", "spec_centroid", "spec_bw", "rolloff",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7",
    "mfcc8", "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13"
]


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

"""
def extract_features(file_path):
    features = {f: 0.0 for f in FEATURE_ORDER}

    try:
        # Load and normalize audio
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        y = librosa.util.normalize(y)

        if len(y) < sr // 2:
            raise ValueError("Audio too short for reliable analysis")

        # Fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=1024)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            raise ValueError("No voiced segments detected")

        # MDVP:Fo(Hz) & FoStd
        MDVP_Fo = np.mean(f0)
        MDVP_FoStd = np.std(f0)

        # Jitter features
        diff1 = np.diff(f0)
        MDVP_Jitter = np.mean(np.abs(diff1)) / np.mean(f0) * 100
        rap_vals = np.abs(f0[1:-1] - (f0[:-2] + f0[1:-1] + f0[2:]) / 3)
        MDVP_RAP = np.mean(rap_vals) / np.mean(f0) * 100
        ppq_vals = np.abs(f0[2:-2] - (f0[:-4] + f0[1:-3] + f0[2:-2] + f0[3:-1] + f0[4:]) / 5)
        MDVP_PPQ = np.mean(ppq_vals) / np.mean(f0) * 100
        diff2 = np.diff(diff1)
        Jitter_DDP = np.mean(np.abs(diff2)) / np.mean(f0) * 100

        # Shimmer features
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        MDVP_Shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms) * 100
        MDVP_Shimmer_dB = 20 * np.log10(MDVP_Shimmer + 1e-10)
        apq3 = np.abs(rms[1:-1] - (rms[:-2] + rms[1:-1] + rms[2:]) / 3)
        Shimmer_APQ3 = np.mean(apq3) / np.mean(rms) * 100
        apq5 = np.abs(rms[2:-2] - (rms[:-4] + rms[1:-3] + rms[2:-2] + rms[3:-1] + rms[4:]) / 5)
        Shimmer_APQ5 = np.mean(apq5) / np.mean(rms) * 100

        # Harmonics-to-noise ratio
        harmonic = librosa.effects.harmonic(y)
        noise = y - harmonic
        HNR = 10 * np.log10(np.sum(harmonic ** 2) / (np.sum(noise ** 2) + 1e-10))

        # Update features dict
        features.update({
            "MDVP:Fo(Hz)": float(MDVP_Fo),
            "MDVP:Jitter(%)": float(MDVP_Jitter),
            "MDVP:RAP": float(MDVP_RAP),
            "MDVP:PPQ": float(MDVP_PPQ),
            "Jitter:DDP": float(Jitter_DDP),
            "MDVP:Shimmer": float(MDVP_Shimmer),
            "MDVP:Shimmer(dB)": float(MDVP_Shimmer_dB),
            "Shimmer:APQ3": float(Shimmer_APQ3),
            "Shimmer:APQ5": float(Shimmer_APQ5),
            "HNR": float(HNR)
        })

    except Exception as e:
        print(f"Feature extraction failed: {e}")
        features["error"] = str(e)

    return features
"""
