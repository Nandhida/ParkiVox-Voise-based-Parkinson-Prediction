from dotenv import load_dotenv
import os
from google import genai

# Load .env
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Get Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
print("API Key Loaded:", bool(api_key))

if not api_key:
    print("Error: API key not found!")
    exit()

# Initialize Gemini client
client = genai.Client(api_key=api_key)  # make sure the key is passed here

# Tiny test request
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # or whichever Gemini model you plan to use
        contents="Explain in one sentence what Parkinson's voice analysis is."
    )
    print("Test Successful! Gemini Response:")
    print(response.text)
except Exception as e:
    print("API Test Failed:", e)
