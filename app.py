from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, template_folder="templates", static_folder="static")

class ContextAwareChatbot:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        self.chat_history = []  # Stores past conversations for context

    def call_gemini_with_retry(self, prompt, retries=3, delay=2):
        """Calls Gemini AI with retry and delay to handle rate limits."""
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                time.sleep(delay)
                return response.text.strip() if response else "No relevant information found."
            except Exception as e:
                print(f"⚠️ API Error: {e} (Attempt {attempt+1}/{retries})")
                time.sleep(delay * (attempt + 1))

        return "⚠️ Error: Could not get a response. Please try again later."

    def intent_detection_agent(self, user_input):
        """Detects intent: Greeting, Question, Follow-up, or Exit."""
        prompt = f"""Classify the user's intent as one of the following:
        1. Greeting (hello, hi, hey)
        2. Question (asks for information)
        3. Follow-up (related to past conversation)
        4. Exit (goodbye, exit)
        Return only the category name.
        
        User Input: {user_input}
        """
        return self.call_gemini_with_retry(prompt)

    def research_agent(self, refined_query):
        """Finds relevant information using Gemini AI."""
        return self.call_gemini_with_retry(refined_query)

    def chat(self, user_input):
        """Handles chat requests."""
        intent = self.intent_detection_agent(user_input)

        if intent == "Greeting":
            return "Hello! How can I assist you today? 😊"

        response = self.research_agent(user_input)
        return response

chatbot = ContextAwareChatbot()

@app.route("/")
def index():
    return render_template("index.html")  # Ensure this template exists

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    response = chatbot.chat(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
