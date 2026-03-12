from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client()

print("Asking Google Gemini A Question")

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What is RAG in AI?  Explain in 3-4 lines"
)

print("Response from Google Gemini...")

print(response.text)