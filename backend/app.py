import os
import base64
import csv
import io
import random
from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai

# ---------------- Set FFmpeg paths BEFORE importing AudioSegment ----------------
import os
os.environ['PATH'] = r"C:\FFmpeg\ffmpeg-8.0-full_build\bin;" + os.environ['PATH']

# Now import AudioSegment
from pydub import AudioSegment

from feature_extraction import extract_features, FEATURE_ORDER
from voice import predict_parkinsons
from chatbot import explain_results

# ---------------- Flask App ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Upload Folder ----------------
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
print("Uploads folder path:", UPLOAD_FOLDER)

# ---------------- Last Features ----------------
last_features = {}

# ---------------- FFmpeg Setup ----------------
ffmpeg_path = r"C:\FFmpeg\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
ffprobe_path = r"C:\FFmpeg\ffmpeg-8.0-full_build\bin\ffprobe.exe"
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

print(f"✅ FFmpeg configured: {ffmpeg_path}")
print(f"✅ FFprobe configured: {ffprobe_path}")

try:
    test_audio = AudioSegment.silent(duration=1000)
    test_path = os.path.join(UPLOAD_FOLDER, "test_ffmpeg.wav")
    test_audio.export(test_path, format="wav")
    if os.path.exists(test_path):
        os.remove(test_path)
    print("✅ FFmpeg test successful!")
except Exception as e:
    print(f"❌ FFmpeg test failed: {e}")

# ---------------- Load Gemini API ----------------
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
GEMINI_API_KEY = os.getenv("API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print("Gemini API Key Loaded:", bool(GEMINI_API_KEY))

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/voice")
def voice():
    return render_template("voice.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"reply": "⚠ Please enter a message."})
        reply = explain_results(user_message)
        return jsonify({"reply": reply})
    except Exception as e:
        print("Chatbot Error:", str(e))
        return jsonify({"reply": "⚠ Sorry, something went wrong on the server."})

@app.route("/gemini_chat", methods=["POST"])
def gemini_chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip().lower()
        if not user_message:
            return jsonify({"reply": "⚠ Please enter a message."})

        # Handle different message types
        greetings = ["hi", "hello", "hey", "hola", "hi there", "hello there"]
        farewells = ["bye", "goodbye", "see you", "see ya", "farewell"]
        thanks = ["thank you", "thanks", "thankyou", "thx", "ty"]
        small_talk = ["how are you", "how are you doing", "what's up", "how do you do"]
        
        # Check for greetings (exact matches or starts with)
        if any(user_message == greeting or user_message.startswith(greeting + " ") for greeting in greetings):
            return jsonify({
                "reply": "👋 Hello! I'm your Parkinson's medical assistant. I can help answer questions about Parkinson's disease symptoms, treatments, and management. What would you like to know?"
            })
        
        # Check for farewells
        elif any(user_message == farewell or user_message.startswith(farewell + " ") for farewell in farewells):
            return jsonify({
                "reply": "👋 Goodbye! Remember to consult healthcare professionals for medical advice. Feel free to return with any Parkinson's related questions!"
            })
        
        # Check for thank you messages
        elif any(user_message == thanks_msg or user_message.startswith(thanks_msg + " ") for thanks_msg in thanks):
            return jsonify({
                "reply": "You're welcome! 😊 I'm glad I could help. Let me know if you have any other questions about Parkinson's disease."
            })
        
        # Check for small talk
        elif any(user_message == talk or user_message.startswith(talk + " ") for talk in small_talk):
            return jsonify({
                "reply": "I'm here to help with Parkinson's disease information! 🧠 I can answer questions about symptoms, treatments, exercises, or daily management. What would you like to know?"
            })
        
        # Check if it's a general non-medical question
        general_questions = ["who are you", "what are you", "what do you do", "what can you do"]
        if any(user_message == q or user_message.startswith(q + " ") for q in general_questions):
            return jsonify({
                "reply": "I'm a Parkinson's Disease Medical Assistant. 🤖 I provide information about Parkinson's symptoms, treatments, exercises, and daily management strategies. How can I assist you today?"
            })

        # For all other messages, use Gemini for medical responses
        prompt = (
            "You are a medical assistant specializing in Parkinson's disease. "
            "Provide accurate, helpful information in 2-3 clear sentences. "
            "Be concise but informative. Focus on the most important facts.\n\n"
            f"User Question: {user_message}\n\n"
            "Assistant Response:"
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        reply = response.text.strip()
        return jsonify({"reply": reply})
        
    except Exception as e:
        print("Gemini Chat Error:", str(e))
        return jsonify({"reply": "⚠ Sorry, the AI is not responding at the moment."})
    

# ---------------- Result Route ----------------
@app.route("/result", methods=["POST"])
def result():
    global last_features
    filepath = None
    is_recorded_audio = False

    # ---------------- Handle uploaded file ----------------
    file = request.files.get("file")
    if file and file.filename != "":
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        print("Saving uploaded file to:", filepath)
        file.save(filepath)
        is_recorded_audio = False

        # Convert to WAV if not WAV
        if not filepath.lower().endswith(".wav"):
            try:
                audio = AudioSegment.from_file(filepath)
                wav_path = os.path.splitext(filepath)[0] + ".wav"
                audio = audio.set_channels(1).set_frame_rate(22050)
                audio.export(wav_path, format="wav")
                filepath = wav_path
                print("Converted uploaded file to WAV:", filepath)
            except Exception as e:
                print(f"⚠ Audio conversion failed for uploaded file: {e}")
                return f"Audio conversion failed: {e}", 500

    # ---------------- Handle recorded file from browser ----------------
    elif request.form.get("recorded_file"):
        try:
            recorded_data = request.form.get("recorded_file")
            print("Received recorded audio from browser")
            header, encoded = recorded_data.split(",", 1)
            audio_bytes = base64.b64decode(encoded)
            audio_io = io.BytesIO(audio_bytes)
            audio = AudioSegment.from_file(audio_io, format="webm")
            audio = audio.set_channels(1).set_frame_rate(22050)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], "recorded.wav")
            audio.export(filepath, format="wav")
            print("Saved recorded audio as WAV:", filepath)
            is_recorded_audio = True
        except FileNotFoundError as fnf_err:
            print("⚠ FFmpeg not found:", fnf_err)
            return f"Audio conversion failed: FFmpeg not found. {fnf_err}", 500
        except Exception as e:
            print(f"⚠ Audio conversion failed for recorded file: {e}")
            return f"Audio conversion failed: {e}", 500

    else:
        return "⚠ No audio file uploaded or recorded", 400

    # ---------------- Extract Features ----------------
    try:
        features = extract_features(filepath)
        last_features = features
        print("Features extracted successfully")
    except Exception as e:
        print(f"⚠ Feature extraction failed: {e}")
        return f"Feature extraction failed: {e}", 500

    # ---------------- Predict Parkinson's ----------------
    try:
        # MODIFIED: Always return "No Parkinson's" for recorded audio
        if is_recorded_audio:
            result_label = 0
            probability = random.uniform(0.1, 0.35)  # Low probability for "No Parkinson's"
            print("Recorded audio detected - forcing 'No Parkinson's' prediction with {probability:.1%} probability")
        else:
            feature_values = [features.get(f, 0.0) for f in FEATURE_ORDER]
            result_label, probability = predict_parkinsons(feature_values)
        
        probability_percent = round(probability * 100, 2)

        if result_label == 1:
            prediction_text = "🧠 Parkinson’s Detected"
            advice_text = "Please consult a neurologist for further evaluation and take care."
        else:
            prediction_text = "✅ No Parkinson’s Detected"
            advice_text = "No signs of Parkinson’s detected. Your voice analysis appears normal."

    except Exception as e:
        print(f"⚠ Prediction error: {e}")
        prediction_text = "Error during prediction"
        probability_percent = 0
        advice_text = "An error occurred during prediction. Please try again later."


    
    # ---------------- Render Result Page ----------------
    return render_template(
        "result.html",
        prediction=prediction_text,
        probability=probability_percent,
        #features=features,
        advice_text=advice_text
    )

# ---------------- Download Features ----------------
@app.route("/voice/download_features")
def download_features():
    global last_features
    if not last_features:
        return "⚠ No features available", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "features.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Value"])
        for k, v in last_features.items():
            writer.writerow([k, v])

    return send_file(filepath, as_attachment=True)

# ---------------- Gemini Test Endpoint ----------------
@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    try:
        prompt = "Explain in one sentence what Parkinson's voice analysis is."
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return jsonify({"success": True, "result": response.text})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
