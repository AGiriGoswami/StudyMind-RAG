import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=API_KEY)

text_list = ["test " + str(i) for i in range(150)]

try:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text_list,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    print("Success 001, count:", len(result.embeddings))
except Exception as e:
    print("Error 001:", e)
    
try:
    contents = [text for text in text_list]
    result2 = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text_list,
    )
    print("Success 2 list of strings, count:", len(result2.embeddings))
except Exception as e:
    print("Error 2 string:", e)
    
