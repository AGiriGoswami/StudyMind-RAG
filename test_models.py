import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=API_KEY)

for m in gemini_client.models.list():
    if "embed" in m.name.lower() or "embed" in str(m.supported_actions).lower():
        print(f"Model: {m.name}, Actions: {m.supported_actions}")
