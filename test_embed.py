import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=API_KEY)

text_list = ["test 1", "test 2"]
try:
    result = gemini_client.models.embed_content(
        model="text-embedding-004",
        contents=text_list,
        config=types.EmbedContentConfig(task_type="retrieval_document")
    )
    print("Success")
except Exception as e:
    print("Error:", e)
    
try:
    result2 = gemini_client.models.embed_content(
        model="text-embedding-004",
        contents=text_list,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    print("Success 2")
except Exception as e:
    print("Error 2:", e)
