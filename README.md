# ParkiVox

Voice-based Parkinson's Disease prediction using Random Forest machine learning model.

## Overview

ParkiVox is an intelligent voice-based diagnostic system designed to detect Parkinson's Disease (PD) in its early stages. The system analyzes acoustic characteristics of speech patterns to identify subtle voice changes associated with Parkinson's Disease.

### About Parkinson's Disease

Parkinson's Disease is a progressive neurological disorder that affects movement and speech. Speech impairment is one of the earliest detectable symptoms, often appearing before motor symptoms become apparent. ParkiVox leverages this early indicator to provide non-invasive screening.

### How It Works

1. **Audio Recording**: Users provide a voice sample (typically 3-30 seconds)
2. **Feature Extraction**: The system extracts 22+ acoustic features including:
   - Mel-Frequency Cepstral Coefficients (MFCC)
   - Shimmer (variation in amplitude)
   - Jitter (variation in pitch)
   - Fundamental frequency
   - Zero Crossing Rate
3. **Classification**: Random Forest model analyzes features to predict PD presence
4. **Results**: Generates prediction with confidence score and risk assessment

### Why Voice Analysis?

- **Non-invasive**: Simple voice recordings via smartphone/computer
- **Early Detection**: Speech changes appear in early-stage PD
- **Accessible**: No expensive equipment or lab tests needed
- **Repeatable**: Easy regular monitoring for disease progression

## Installation

```bash
# Clone the repository
git clone https://github.com/Nandhida/ParkiVox-Voise-based-Parkinson-Prediction.git
cd ParkiVox-Voise-based-Parkinson-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Web Application

```bash
cd ParkiVox/backend
python app.py
```

Access at `http://localhost:5000`

### Predict from Audio File

```bash
python predict_audio.py --audio path/to/audio.wav
```

## Project Structure

```
ParkiVox/
├── backend/
│   ├── app.py                    # Web application
│   ├── feature_extraction.py     # Extract acoustic features
│   ├── voice.py                  # Voice processing
│   ├── voice_model.py            # Random Forest model
│   ├── chatbot.py                # Chatbot interface
│   ├── generate_plots.py         # Result visualization
│   ├── check.py                  # Validation
│   ├── type.py                   # Type definitions
│   ├── model/                    # Trained models
│   ├── templates/                # HTML templates
│   ├── static/                   # CSS, JS, assets
│   └── uploads/                  # User uploaded audio
│
├── data/                         # Training data
├── uploads/                      # Additional uploads
├── predict_audio.py              # Audio prediction script
├── voice_model.py                # Voice model
└── .env                          # Configuration

```

## Configuration

Create a `.env` file in the `ParkiVox` directory:

```env
FLASK_ENV=development
FLASK_APP=backend/app.py
SECRET_KEY=your_secret_key
MAX_UPLOAD_SIZE=50MB
UPLOAD_FOLDER=backend/uploads
MODEL_PATH=backend/model/trained_model.pkl
```

## Features

- Extract 22+ acoustic features (MFCC, shimmer, jitter, fundamental frequency, etc.)
- Random Forest classification for high accuracy
- Web interface for easy audio upload
- Chatbot for interactive Q&A about Parkinson's
- Real-time predictions with confidence scores
- Risk assessment and analysis

## Model

- **Algorithm**: Random Forest Classifier
- **Input**: Acoustic features from voice recordings
- **Output**: Prediction (Healthy / Parkinson's Disease) with confidence score

## Disclaimer

⚠️ **Medical Use**: ParkiVox is a research tool. Results must be verified by healthcare professionals. Not a medical diagnostic device.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## License

MIT License

## Author

Nandhidakavi A L
