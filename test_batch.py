import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=API_KEY)

text_list = ["test " + str(i) for i in range(150)]
try:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text_list,
    )
    print("Success, dimension:", len(result.embeddings[0].values), "count:", len(result.embeddings))
except Exception as e:
    print("Error:", e)
