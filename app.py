import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

app = Flask(__name__, template_folder="templates")
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "gemini-3.1-flash-lite"

try:
    SYSTEM_PROMPT = (BASE_DIR / "chatbot_config.txt").read_text(encoding="utf-8").strip()
except OSError:
    SYSTEM_PROMPT = "You are Medicine Information AI. Answer only educational medicine and pharmacology questions."

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a medicine-related study question."}), 400

    if client is None:
        return jsonify({"error": "Gemini API key is not configured."}), 500

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.2,
                "thinking_config": {"thinking_level": "low"},
            },
        )
        answer = (response.text or "").strip()
        return jsonify({"answer": answer or "I couldn't generate a response. Please try again."})
    except Exception:
        return jsonify({"error": "Something went wrong while contacting the AI service."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
