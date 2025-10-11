# backend.py
import google.genai as genai
import os 
import dotenv
dotenv.load_dotenv()

Model='models/gemini-2.5-pro'
# Replace with your Gemini API key
API_KEY = os.getenv("APIKEY")
client = genai.Client(api_key=API_KEY)

def ask_chatbot(prompt):
    response = client.models.generate_content(
        model=Model,
        contents=f"Answer as a helpful assistant: {prompt}"
    )
    return response.text

def summarize_text(text):
    response = client.models.generate_content(
        model=Model,
        contents=f"Summarize this text: {text}"
    )
    return response.text

def creative_writer(prompt):
    response = client.models.generate_content(
        model=Model,
        contents=f"Write a creative short story or poem about: {prompt}"
    )
    return response.text

def make_notes(text):
    response = client.models.generate_content(
        model=Model,
        contents=f"Make study notes from this text: {text}"
    )
    return response.text

def generate_ideas(prompt):
    response = client.models.generate_content(
        model=Model,
        contents=f"Generate creative ideas for: {prompt}"
    )
    return response.text
