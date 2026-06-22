import streamlit as st
import os
import pypdf
import faiss
import numpy as np
from google import genai
from google.genai import types
from streamlit_local_storage import LocalStorage
from dotenv import load_dotenv
from groq import Groq
import io
import re
import time
import json
import streamlit.components.v1 as components
from gtts import gTTS

# Load environment variables
load_dotenv()

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
    st.error("Groq API Key is missing. Please set it in .env or Streamlit Secrets.")
    st.stop()

gemini_client = genai.Client(api_key=API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Session State
if "faiss_index" not in st.session_state: st.session_state.faiss_index = None
if "text_chunks" not in st.session_state: st.session_state.text_chunks = []
if "sources" not in st.session_state: st.session_state.sources = []
if "flashcards" not in st.session_state: st.session_state.flashcards = []

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
        <div class="mermaid" id="mermaid-chart" style="background-color: #111827; color: #F8FAFC; padding: 20px; border-radius: 12px; border: 1px solid #263244;">
            {code}
        </div>
        <button id="download-png" style="margin-top: 15px; padding: 8px 12px; background-color: #3B82F6; color: white; border: none; border-radius: 6px; cursor: pointer; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 500;">📥 Download Flowchart as PNG</button>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    fontFamily: 'Inter',
                    primaryColor: '#1F2937',
                    primaryTextColor: '#F8FAFC',
                    primaryBorderColor: '#3B82F6',
                    lineColor: '#94A3B8',
                    secondaryColor: '#263244',
                    tertiaryColor: '#0B1220'
                }}
            }});
            
            document.getElementById('download-png').addEventListener('click', function() {{
                const svg = document.querySelector('.mermaid svg');
                if (!svg) return;
                
                let bBox;
                try {{ bBox = svg.getBBox(); }} catch (e) {{ bBox = svg.getBoundingClientRect(); }}
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
                    ctx.fillStyle = "#111827";
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

def get_gemini_embeddings(text_list, task_type="RETRIEVAL_DOCUMENT"):
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type.upper())
        )
        all_embeddings.extend([e.values for e in result.embeddings])
    return all_embeddings

def create_vector_store(chunks, current_index=None):
    embeddings = get_gemini_embeddings(chunks, task_type="retrieval_document")
    dimension = len(embeddings[0])
    
    if current_index is None:
        index = faiss.IndexFlatL2(dimension)
    else:
        index = current_index
        
    index.add(np.array(embeddings).astype('float32'))
    return index

def query_rag(question, index, chunks, inc_short=True, inc_long=True, inc_flow=True, top_k=3):
    question_embedding = get_gemini_embeddings([question], task_type="retrieval_query")[0]
    distances, indices = index.search(np.array([question_embedding]).astype('float32'), top_k)
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join(relevant_chunks)
    
    sys_prompt = "You are a helpful research assistant. Structure your answer strictly based on the following requirements:\n"
    if inc_short:
        sys_prompt += "1. Short Answer: A 1-2 sentence simple summary.\n"
    if inc_long:
        sys_prompt += "2. Long Answer: A detailed, extremely simple explanation as if to a beginner.\n"
    if inc_flow:
        sys_prompt += "3. Flowchart: You MUST output a complete Mermaid.js flowchart (graph TD) representing the ENTIRE topic. You MUST write the raw mermaid code inside a markdown block exactly like this: ```mermaid\n[your code here]\n```. CRITICAL MERMAID SYNTAX RULES: 1. You MUST NOT use parentheses (), brackets [], braces {}, or quotes inside node text. 2. DO NOT use special characters in node IDs, only simple letters and numbers (e.g. A1, B2). 3. Avoid long text in nodes. Example: A1[React JS Frontend] -->|Uses| B1[Node JS Backend]\n"
        
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content, chat_completion.usage
    except Exception as e:
        return f"An error occurred with Groq: {str(e)}", None

def generate_flashcards_from_chunks(chunks):
    if not chunks: return []
    # Use first few chunks to generate flashcards
    context = "\n\n".join(chunks[:5])
    
    sys_prompt = "You are a helpful educational assistant. Based on the provided context, generate exactly 5 flashcards for studying. You MUST output your response in valid JSON format only, returning a JSON object with a key 'flashcards' containing a list of objects with 'question' and 'answer' keys. Do not include any other text."
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nGenerate 5 flashcards in JSON format."}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        res = chat_completion.choices[0].message.content
        data = json.loads(res)
        return data.get("flashcards", [])
    except Exception as e:
        st.error(f"Error generating flashcards: {e}")
        return []

def render_flashcard(question, answer):
    html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    .flip-card {{
      background-color: transparent;
      width: 100%;
      height: 150px;
      perspective: 1000px;
      margin-bottom: 12px;
      font-family: 'Inter', sans-serif;
    }}
    .flip-card-inner {{
      position: relative;
      width: 100%;
      height: 100%;
      text-align: center;
      transition: transform 0.6s;
      transform-style: preserve-3d;
      cursor: pointer;
    }}
    .flip-card:hover .flip-card-inner {{
      transform: rotateY(180deg);
    }}
    .flip-card-front, .flip-card-back {{
      position: absolute;
      width: 100%;
      height: 100%;
      -webkit-backface-visibility: hidden;
      backface-visibility: hidden;
      border-radius: 12px;
      padding: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-sizing: border-box;
      border: 1px solid #263244;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    .flip-card-front {{
      background-color: #1F2937;
      color: #F8FAFC;
    }}
    .flip-card-back {{
      background-color: #3B82F6;
      color: white;
      transform: rotateY(180deg);
      border-color: #3B82F6;
    }}
    </style>
    <div class="flip-card">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <p style="margin: 0; font-size: 15px; font-weight: 500;">{question}</p>
        </div>
        <div class="flip-card-back">
          <p style="margin: 0; font-size: 14px; line-height: 1.5;">{answer}</p>
        </div>
      </div>
    </div>
    """
    components.html(html, height=165)

# --- UI Setup ---
st.set_page_config(page_title="StudyMind Workspace", page_icon="📓", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B1220 !important;
    color: #F8FAFC !important;
}

.stApp { background-color: #0B1220; }
.stAppHeader { display: none !important; }

/* Layout spacing */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; font-weight: 600 !important; }
p, span, div { color: #F8FAFC; }
.stMarkdown p { color: #94A3B8; }

/* Buttons */
.stButton > button {
    background-color: #1F2937 !important;
    color: #F8FAFC !important;
    border: 1px solid #263244 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background-color: #263244 !important;
    border-color: #3B82F6 !important;
    color: #3B82F6 !important;
}
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #3B82F6 !important;
    color: white !important;
    border-color: #3B82F6 !important;
}

/* Inputs */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: #111827 !important;
    border: 1px solid #263244 !important;
    border-radius: 8px !important;
    color: #F8FAFC !important;
}

/* Chat Input */
[data-testid="stChatInput"] {
    background-color: #111827 !important;
    border: 1px solid #263244 !important;
    border-radius: 12px !important;
}

/* Chat Messages */
.stChatMessage {
    background-color: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
}
[data-testid="chatAvatarIcon-user"] { background-color: #3B82F6 !important; }
[data-testid="chatAvatarIcon-assistant"] { background-color: #1F2937 !important; }
.stChatMessage div[data-testid="stMarkdownContainer"] p {
    color: #F8FAFC !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: #111827 !important;
    border: 1px dashed #3B82F6 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

/* Custom CSS Classes via Markdown */
.source-card {
    background-color: #1F2937;
    border: 1px solid #263244;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    transition: all 0.2s;
}
.source-card:hover {
    border-color: #3B82F6;
    background-color: #2563EB1A;
}
.source-title { font-weight: 500; font-size: 0.9rem; color: #F8FAFC; margin-bottom: 4px; }
.source-meta { font-size: 0.75rem; color: #94A3B8; }
.logo-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 24px;
}
.logo-icon { color: #3B82F6; }
hr { border-color: #263244 !important; margin: 1rem 0 !important; }

/* Tabs */
[data-testid="stTabs"] button {
    background-color: transparent !important;
    color: #94A3B8 !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

localS = LocalStorage()

if "messages" not in st.session_state:
    st.session_state.messages = []
    stored_chats = localS.getItem("chat_history")
    if stored_chats and isinstance(stored_chats, str):
        try: st.session_state.messages = json.loads(stored_chats)
        except Exception: pass

# --- Main Layout ---
col1, col2, col3 = st.columns([1, 2.5, 1], gap="large")

# LEFT PANEL: Sources
with col1:
    st.markdown('<div class="logo-title"><span class="logo-icon">📓</span> StudyMind</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("Add Source")
        uploaded_file = st.file_uploader("Upload PDF", accept_multiple_files=False, label_visibility="collapsed")
        pasted_text = st.text_area("Paste Text", placeholder="Or paste your text here...", height=100, label_visibility="collapsed")
        
        if st.button("Add to Knowledge Base", type="primary"):
            with st.spinner("Processing..."):
                all_text = ""
                source_name = ""
                if uploaded_file:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        all_text += extract_text_from_pdf(uploaded_file) + "\n"
                        source_name = uploaded_file.name
                
                if pasted_text.strip():
                    all_text += pasted_text + "\n"
                    if not source_name: source_name = "Pasted Text"
                
                if not all_text.strip():
                    st.error("No extractable text.")
                else:
                    chunks = chunk_text(all_text)
                    st.session_state.faiss_index = create_vector_store(chunks, st.session_state.faiss_index)
                    st.session_state.text_chunks.extend(chunks)
                    st.session_state.sources.append({"name": source_name, "words": len(all_text.split())})
                    st.success("Added!")

    st.markdown("### Your Sources")
    if not st.session_state.sources:
        st.markdown('<p style="color:#94A3B8; font-size:0.85rem;">No sources added yet. Add a PDF or text to begin.</p>', unsafe_allow_html=True)
    else:
        for s in st.session_state.sources:
            st.markdown(f"""
            <div class="source-card">
                <div class="source-title">📄 {s['name']}</div>
                <div class="source-meta">{s['words']} words</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        localS.setItem("chat_history", "[]", key="clear_chat_history")
        st.rerun()

# CENTER PANEL: Chat Workspace
with col2:
    st.markdown("### 💬 Workspace")
    
    if st.session_state.faiss_index is None:
        st.info("👈 Upload a document to start analyzing.")
    else:
        chat_container = st.container(height=650, border=False)
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "mermaid" in msg:
                        for code in msg["mermaid"]:
                            render_mermaid(code)
                    if "usage" in msg and msg["usage"]:
                        u = msg["usage"]
                        t_tok = u.get("total_tokens", 0) if isinstance(u, dict) else u.total_tokens
                        st.caption(f"⚡ Token Usage: **{t_tok}**")
                    
                    if i == len(st.session_state.messages) - 1 and msg["role"] == "assistant":
                        text_content = msg.get("content", "")
                        if text_content.strip():
                            try:
                                audio_fp = text_to_speech(text_content)
                                st.audio(audio_fp, format='audio/mp3')
                            except Exception:
                                pass

        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            localS.setItem("chat_history", json.dumps(st.session_state.messages), key="set_chat_user")
            st.rerun()
            
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        inc_short = st.session_state.get("inc_short", True)
                        inc_long = st.session_state.get("inc_long", True)
                        inc_flow = st.session_state.get("inc_flow", True)
                        
                        raw_answer, usage = query_rag(
                            st.session_state.messages[-1]["content"], 
                            st.session_state.faiss_index, 
                            st.session_state.text_chunks, 
                            inc_short, inc_long, inc_flow
                        )
                        
                        mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', raw_answer, re.DOTALL)
                        text_part = re.sub(r'```mermaid\n.*?\n```', '', raw_answer, flags=re.DOTALL)
                        
                        st.write_stream(stream_text(text_part))
                        for code in mermaid_blocks: render_mermaid(code)
                        
                        if usage:
                            st.caption(f"⚡ Token Usage: **{usage.total_tokens}**")
                        
                        if text_part.strip():
                            try:
                                audio_fp = text_to_speech(text_part)
                                st.audio(audio_fp, format='audio/mp3')
                            except Exception:
                                pass
                                
                    usage_dict = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens} if usage else None
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": text_part, 
                        "mermaid": mermaid_blocks,
                        "usage": usage_dict
                    })
                    localS.setItem("chat_history", json.dumps(st.session_state.messages), key="set_chat_assistant")
                    st.rerun()

# RIGHT PANEL: Notebook & Actions
with col3:
    st.markdown("### 📝 Notebook")
    with st.container(border=True):
        tab1, tab2, tab3 = st.tabs(["Notes", "Flashcards", "Mind Maps"])
        
        with tab1:
            st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">Save key insights here.</p>', unsafe_allow_html=True)
            st.text_area("Scratchpad", placeholder="Take notes while you research...", height=250, label_visibility="collapsed")
            st.button("Save Note")
            
        with tab2:
            st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">Generate flashcards from sources.</p>', unsafe_allow_html=True)
            if not st.session_state.text_chunks:
                st.info("Upload a document first to generate flashcards.")
            else:
                if st.button("Generate Flashcards"):
                    with st.spinner("Generating..."):
                        st.session_state.flashcards = generate_flashcards_from_chunks(st.session_state.text_chunks)
                
                if st.session_state.flashcards:
                    st.markdown("---")
                    for fc in st.session_state.flashcards:
                        render_flashcard(fc.get('question', 'Q'), fc.get('answer', 'A'))
                
        with tab3:
            st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">Visualize your topics.</p>', unsafe_allow_html=True)
            st.session_state.inc_flow = st.checkbox("Enable Auto-Mind Maps in Chat", value=st.session_state.get("inc_flow", True))
            st.caption("When enabled, the AI will automatically generate Mermaid diagrams for complex topics.")

    st.markdown("### ⚙️ Preferences")
    with st.container(border=True):
        st.session_state.inc_short = st.checkbox("Include Summary", value=st.session_state.get("inc_short", True))
        st.session_state.inc_long = st.checkbox("Include Deep Dive", value=st.session_state.get("inc_long", True))
