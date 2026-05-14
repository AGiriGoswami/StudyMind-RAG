import streamlit as st
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from groq import Groq

# Load environment variables for local development
load_dotenv()

# --- Configuration & Secrets ---
# Support both .env (local) and Streamlit secrets (cloud)
try:
    API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
except FileNotFoundError:
    API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("Google API Key is missing. Please set it in .env or Streamlit Secrets.")
    st.stop()

if not GROQ_API_KEY:
    st.error("Groq API Key is missing. Please set it in .env or Streamlit Secrets. You can get one for free at https://console.groq.com/")
    st.stop()

genai.configure(api_key=API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize session state for the vector store
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []

# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def get_gemini_embeddings(text_list, task_type="retrieval_document"):
    # Gemini returns a list of embeddings (one for each chunk)
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=text_list,
        task_type=task_type
    )
    return result['embedding']

def create_vector_store(chunks):
    # Get embeddings from Gemini API
    embeddings = get_gemini_embeddings(chunks, task_type="retrieval_document")
    
    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def query_rag(question, index, chunks, top_k=3):
    # Embed the question using Gemini API
    question_embedding = get_gemini_embeddings([question], task_type="retrieval_query")[0]
    
    # Search in FAISS
    distances, indices = index.search(np.array([question_embedding]).astype('float32'), top_k)
    
    # Retrieve relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join(relevant_chunks)
    
    # Generate answer using Groq (Llama 3.1)
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful academic assistant."
                },
                {
                    "role": "user",
                    "content": f"Answer the user's question based ONLY on the following context.\nIf the answer is not in the context, say 'I cannot answer this based on the provided document.'\n\nContext:\n{context}\n\nQuestion: {question}"
                }
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred with Groq: {str(e)}"

# --- UI ---
st.set_page_config(page_title="StudyMind", page_icon="📚", layout="centered")
st.title("📚 StudyMind - RAG Academic Assistant")

st.markdown("Upload a PDF document, process it, and ask questions about its content.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Extracting text and generating embeddings via Gemini..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text.strip():
                st.error("No extractable text found in this PDF.")
            else:
                chunks = chunk_text(text)
                index = create_vector_store(chunks)
                
                st.session_state.faiss_index = index
                st.session_state.text_chunks = chunks
                st.success("Document processed successfully! You can now ask questions.")

if st.session_state.faiss_index is not None:
    st.divider()
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question based on the document:")
    
    if st.button("Generate Answer"):
        if question.strip():
            with st.spinner("Analyzing document and generating answer..."):
                answer = query_rag(question, st.session_state.faiss_index, st.session_state.text_chunks)
                st.info("### Answer\n" + answer)
        else:
            st.warning("Please enter a question.")
