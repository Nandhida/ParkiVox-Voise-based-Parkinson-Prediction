import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def explain_results(user_message: str) -> str:
    if not user_message.strip():
        return "⚠ Please type something to start the chat."

    print("User message:", user_message)  # Debug

    if api_key:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"You are a helpful Parkinson’s disease medical assistant. "
                f"Answer concisely and clearly.\n\nUser: {user_message}"
            )
            if response and hasattr(response, "text") and response.text:
                print("Gemini response:", response.text)
                return response.text.strip()
            else:
                print("Gemini returned empty, using fallback")
                return get_response(user_message)
        except Exception as e:
            print("Gemini API Error:", e)
            return get_response(user_message)
    else:
        return get_response(user_message)


# Rule-based fallback
def get_response(user_message: str) -> str:
    message = user_message.lower()
    if "hello" in message or "hi" in message:
        return "Hello! How can I assist you with Parkinson’s related queries today?"
    elif "symptom" in message:
        return "Common symptoms of Parkinson’s include tremors, slow movement, stiffness, and balance problems."
    elif "treatment" in message:
        return "Treatment options include medications (like Levodopa), physiotherapy, and in some cases surgery."
    elif "diagnosis" in message:
        return "Parkinson’s is usually diagnosed based on symptoms, medical history, and neurological examination."
    elif "bye" in message:
        return "Goodbye! Take care and remember to follow up with your doctor regularly."
    else:
        return "I'm here to help! Could you please clarify your question about Parkinson’s?"
