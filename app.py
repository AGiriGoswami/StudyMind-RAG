import streamlit as st
import os
import pypdf
import faiss
import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from groq import Groq
import io
import re
import time
import json
import streamlit.components.v1 as components
from gtts import gTTS

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
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp

def render_mermaid(code: str):
    components.html(
        f"""
        <div class="mermaid" id="mermaid-chart" style="background-color: white; padding: 20px;">
            {code}
        </div>
        <button id="download-png" style="margin-top: 15px; padding: 10px 15px; background-color: #6C63FF; color: white; border: none; border-radius: 8px; cursor: pointer; font-family: sans-serif;">📥 Download Flowchart as PNG</button>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
            
            document.getElementById('download-png').addEventListener('click', function() {{
                const svg = document.querySelector('.mermaid svg');
                if (!svg) return;
                
                let bBox;
                try {{
                    bBox = svg.getBBox();
                }} catch (e) {{
                    bBox = svg.getBoundingClientRect();
                }}
                const width = Math.max(bBox.width, svg.getBoundingClientRect().width, (svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal.width : 0));
                const height = Math.max(bBox.height, svg.getBoundingClientRect().height, (svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal.height : 0));
                
                const svgClone = svg.cloneNode(true);
                svgClone.setAttribute('width', width);
                svgClone.setAttribute('height', height);
                svgClone.style.maxWidth = 'none';
                
                const svgData = new XMLSerializer().serializeToString(svgClone);
                const canvas = document.createElement("canvas");
                
                canvas.width = width * 2;
                canvas.height = height * 2;
                
                const ctx = canvas.getContext("2d");
                ctx.scale(2, 2);
                
                const img = new Image();
                img.setAttribute("src", "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData))));
                
                img.onload = function() {{
                    ctx.fillStyle = "white";
                    ctx.fillRect(0, 0, width, height);
                    ctx.drawImage(img, 0, 0, width, height);
                    const canvasdata = canvas.toDataURL("image/png");
                    const a = document.createElement("a");
                    a.download = "flowchart.png";
                    a.href = canvasdata;
                    a.click();
                }};
            }});
        </script>
        """,
        height=450,
        scrolling=True,
    )

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

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

def query_rag(question, index, chunks, inc_short=True, inc_long=True, inc_flow=True, top_k=3):
    # Embed the question using Gemini API
    question_embedding = get_gemini_embeddings([question], task_type="retrieval_query")[0]
    
    # Search in FAISS
    distances, indices = index.search(np.array([question_embedding]).astype('float32'), top_k)
    
    # Retrieve relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join(relevant_chunks)
    
    sys_prompt = "You are a helpful assistant. Structure your answer strictly based on the following requirements:\n"
    if inc_short:
        sys_prompt += "1. Short Answer: A 1-2 sentence simple summary.\n"
    if inc_long:
        sys_prompt += "2. Long Answer: A detailed, extremely simple explanation as if to a beginner.\n"
    if inc_flow:
        sys_prompt += "3. Flowchart: You MUST output a complete Mermaid.js flowchart (graph TD) representing the ENTIRE topic. You MUST write the raw mermaid code inside a markdown block exactly like this: ```mermaid\n[your code here]\n```. CRITICAL: You MUST wrap all node text and edge labels in DOUBLE QUOTES to prevent syntax errors. Example: A[\"React.js (Frontend)\"] -->|\"Uses\"| B[\"Node.js\"]\n"
        
    # Generate answer using Groq (Llama 3.1)
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": f"First, try to answer the question using the following context. If the context doesn't contain the answer, use your general knowledge to answer it.\n\nContext:\n{context}\n\nQuestion: {question}"
                }
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content, chat_completion.usage
    except Exception as e:
        return f"An error occurred with Groq: {str(e)}", None

# --- UI ---
st.set_page_config(page_title="StudyMind", page_icon="📚", layout="centered")

st.markdown("""
<style>
    .stButton>button {
        background-color: #6C63FF;
        color: white;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #5A52D5;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

CHAT_FILE = "chat_history.json"

def load_chat():
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat(messages):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat()

# Sidebar for Setup and Configuration
with st.sidebar:
    st.title("📚 StudyMind Setup")
    st.markdown("Upload PDF documents or paste raw text below to build the knowledge base.")

    uploaded_files = st.file_uploader("Upload PDFs (optional)", type="pdf", accept_multiple_files=True)
    pasted_text = st.text_area("Or paste your text here (optional)", height=200)

    if pasted_text:
        word_count = len(pasted_text.split())
        char_count = len(pasted_text)
        st.caption(f"📝 **Text stats:** {word_count} words | {char_count} characters")

    if uploaded_files or pasted_text.strip():
        if st.button("Process Data"):
            with st.spinner("Extracting text and generating embeddings via Gemini..."):
                all_text = ""
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        all_text += extract_text_from_pdf(uploaded_file) + "\n"
                
                if pasted_text.strip():
                    all_text += pasted_text + "\n"
                
                if not all_text.strip():
                    st.error("No extractable text found in the inputs.")
                else:
                    chunks = chunk_text(all_text)
                    index = create_vector_store(chunks)
                    
                    st.session_state.faiss_index = index
                    st.session_state.text_chunks = chunks
                    st.success("Data processed successfully! You can now ask questions.")

    if st.session_state.faiss_index is not None:
        st.divider()
        st.subheader("⚙️ Output Preferences")
        st.session_state.inc_short = st.checkbox("Show Short Answer", value=st.session_state.get("inc_short", True))
        st.session_state.inc_long = st.checkbox("Show Long Answer", value=st.session_state.get("inc_long", True))
        st.session_state.inc_flow = st.checkbox("Show Flowchart", value=st.session_state.get("inc_flow", True))
        
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            save_chat([])
            st.rerun()

# Main Chat Interface
st.title("💬 StudyMind Chat")

if st.session_state.faiss_index is None:
    st.info("👈 Please process some documents in the sidebar first to begin chatting!")
else:
    # Display chat messages from history on app rerun
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])
                if "mermaid" in msg:
                    for code in msg["mermaid"]:
                        render_mermaid(code)
                if "usage" in msg and msg["usage"]:
                    u = msg["usage"]
                    p_tok = u.get("prompt_tokens", 0) if isinstance(u, dict) else u.prompt_tokens
                    c_tok = u.get("completion_tokens", 0) if isinstance(u, dict) else u.completion_tokens
                    t_tok = u.get("total_tokens", 0) if isinstance(u, dict) else u.total_tokens
                    st.caption(f"⚡ **Token Usage:** {p_tok} input + {c_tok} output = **{t_tok} total tokens**")
                
                # Only render audio for the last assistant message to save UI performance
                if i == len(st.session_state.messages) - 1:
                    try:
                        audio_fp = text_to_speech(msg["content"])
                        st.audio(audio_fp, format='audio/mp3')
                    except Exception as e:
                        st.error(f"Audio generation failed: {e}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat(st.session_state.messages)
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document and generating answer..."):
                inc_short = st.session_state.get("inc_short", True)
                inc_long = st.session_state.get("inc_long", True)
                inc_flow = st.session_state.get("inc_flow", True)
                
                raw_answer, usage = query_rag(
                    prompt, 
                    st.session_state.faiss_index, 
                    st.session_state.text_chunks, 
                    inc_short, 
                    inc_long, 
                    inc_flow
                )
                
                mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', raw_answer, re.DOTALL)
                text_part = re.sub(r'```mermaid\n.*?\n```', '', raw_answer, flags=re.DOTALL)
                
                # Stream the text portion
                st.write_stream(stream_text(text_part))
                
                # Render Flowcharts
                for code in mermaid_blocks:
                    render_mermaid(code)
                
                if usage:
                    st.caption(f"⚡ **Token Usage:** {usage.prompt_tokens} input + {usage.completion_tokens} output = **{usage.total_tokens} total tokens**")
                
                # Try TTS
                try:
                    audio_fp = text_to_speech(text_part)
                    st.audio(audio_fp, format='audio/mp3')
                except Exception as e:
                    st.error(f"Audio generation failed: {e}")
                
            # Add assistant response to chat history
            usage_dict = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens} if usage else None
            st.session_state.messages.append({
                "role": "assistant", 
                "content": text_part, 
                "mermaid": mermaid_blocks,
                "usage": usage_dict
            })
            save_chat(st.session_state.messages)
