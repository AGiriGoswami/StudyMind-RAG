import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

models_to_test = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-flash-latest']

for model_name in models_to_test:
    try:
        print(f"Testing {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, this is a test.")
        print(f"Success for {model_name}: {response.text[:20]}...")
    except Exception as e:
        print(f"Error for {model_name}: {type(e).__name__} - {e}")
