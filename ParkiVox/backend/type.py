import time
import numpy as np
from scipy.stats import skew, kurtosis

def run_typing_test(data):
    """
    data: { "text": "..." }
    Returns basic text statistics
    """
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}

    words = text.split()
    num_words = len(words)
    num_chars = len(text)

    # Example: fake timing logic (replace with real typing session logic)
    elapsed_time = np.random.uniform(5, 20)  # simulate typing time
    wpm = round((num_words / elapsed_time) * 60, 2) if elapsed_time > 0 else 0

    # Extra stats
    char_codes = [ord(c) for c in text if c.isalpha()]
    skewness = float(skew(char_codes)) if char_codes else 0
    kurt = float(kurtosis(char_codes)) if char_codes else 0

    return {
        "num_words": num_words,
        "num_chars": num_chars,
        "elapsed_time_sec": round(elapsed_time, 2),
        "wpm": wpm,
        "skewness": skewness,
        "kurtosis": kurt
    }
