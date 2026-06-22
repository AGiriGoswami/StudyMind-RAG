import os

with open(r"c:\Users\ajay\Desktop\StudyMind-RAG\app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove LocalStorage import and init
content = content.replace("from streamlit_local_storage import LocalStorage\n", "")
content = content.replace("localS = LocalStorage()\n", "")

# 2. Add Persistence Setup and helper functions
setup_code = """
import os
import json
import faiss
import uuid

WORKSPACE_DIR = ".workspace"
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

def load_state():
    if "threads" not in st.session_state:
        threads_path = os.path.join(WORKSPACE_DIR, "chat_history.json")
        if os.path.exists(threads_path):
            try:
                with open(threads_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    st.session_state.threads = data.get("threads", {})
                    st.session_state.active_thread_id = data.get("active_thread_id")
            except Exception:
                st.session_state.threads = {}
                st.session_state.active_thread_id = None
        else:
            st.session_state.threads = {}
            st.session_state.active_thread_id = None
            
        if not st.session_state.threads:
            default_id = str(uuid.uuid4())
            st.session_state.threads[default_id] = {"name": "General Chat", "messages": []}
            st.session_state.active_thread_id = default_id
            
    if "saved_notes" not in st.session_state:
        notes_path = os.path.join(WORKSPACE_DIR, "saved_notes.json")
        if os.path.exists(notes_path):
            try:
                with open(notes_path, "r", encoding="utf-8") as f:
                    st.session_state.saved_notes = json.load(f)
            except Exception:
                st.session_state.saved_notes = []
        else:
            st.session_state.saved_notes = []
            
    if "sources" not in st.session_state:
        meta_path = os.path.join(WORKSPACE_DIR, "documents_metadata.json")
        faiss_path = os.path.join(WORKSPACE_DIR, "faiss_index.bin")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    st.session_state.sources = data.get("sources", [])
                    st.session_state.text_chunks = data.get("text_chunks", [])
            except Exception:
                st.session_state.sources = []
                st.session_state.text_chunks = []
        else:
            st.session_state.sources = []
            st.session_state.text_chunks = []
            
        if os.path.exists(faiss_path):
            try:
                st.session_state.faiss_index = faiss.read_index(faiss_path)
            except Exception:
                st.session_state.faiss_index = None
        else:
            st.session_state.faiss_index = None
            
    if "flashcards" not in st.session_state:
        st.session_state.flashcards = []

def save_threads():
    with open(os.path.join(WORKSPACE_DIR, "chat_history.json"), "w", encoding="utf-8") as f:
        json.dump({"active_thread_id": st.session_state.active_thread_id, "threads": st.session_state.threads}, f)

def save_notes():
    with open(os.path.join(WORKSPACE_DIR, "saved_notes.json"), "w", encoding="utf-8") as f:
        json.dump(st.session_state.saved_notes, f)

def save_sources():
    with open(os.path.join(WORKSPACE_DIR, "documents_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"sources": st.session_state.sources, "text_chunks": st.session_state.text_chunks}, f)
    if st.session_state.faiss_index is not None:
        faiss.write_index(st.session_state.faiss_index, os.path.join(WORKSPACE_DIR, "faiss_index.bin"))

# Call load state early
load_state()
"""

# Replace the old Session State lines (45-49)
old_state = '''# Session State
if "faiss_index" not in st.session_state: st.session_state.faiss_index = None
if "text_chunks" not in st.session_state: st.session_state.text_chunks = []
if "sources" not in st.session_state: st.session_state.sources = []
if "flashcards" not in st.session_state: st.session_state.flashcards = []'''

content = content.replace(old_state, setup_code)

# 3. Remove old LocalStorage initialization block
old_init_block = '''# Initialize Threading
if "threads" not in st.session_state:
    st.session_state.threads = {}
    st.session_state.active_thread_id = None
    
    stored_threads = localS.getItem("chat_threads")
    if stored_threads and isinstance(stored_threads, str):
        try: 
            data = json.loads(stored_threads)
            st.session_state.threads = data.get("threads", {})
            st.session_state.active_thread_id = data.get("active_thread_id")
        except Exception: pass

    if not st.session_state.threads:
        default_id = str(uuid.uuid4())
        st.session_state.threads[default_id] = {"name": "General Chat", "messages": []}
        st.session_state.active_thread_id = default_id

# Helper to save threads
def save_threads():
    data = {
        "active_thread_id": st.session_state.active_thread_id,
        "threads": st.session_state.threads
    }
    localS.setItem("chat_threads", json.dumps(data), key=f"save_threads_{time.time()}")

# Initialize Notes
if "saved_notes" not in st.session_state:
    st.session_state.saved_notes = []
    stored_notes = localS.getItem("saved_notes")
    if stored_notes and isinstance(stored_notes, str):
        try: st.session_state.saved_notes = json.loads(stored_notes)
        except Exception: pass'''

content = content.replace(old_init_block, "")

# 4. Replace localS.setItem logic with new save helpers
# Clear notes
content = content.replace('localS.setItem("saved_notes", "[]", key="clear_notes")', 'save_notes()')
# Save note
content = content.replace('localS.setItem("saved_notes", json.dumps(st.session_state.saved_notes), key=f"set_note_{len(st.session_state.saved_notes)}")', 'save_notes()')
# Threads are already using `save_threads()` helper we injected!

# 5. Inject save_sources() when sources are updated
add_source_block = '''                    st.session_state.sources.append({
                        "name": source_name, 
                        "words": words,
                        "pages": num_pages,
                        "reading_time": reading_time,
                        "raw_text": all_text
                    })
                    st.success("Added!")'''
new_add_source_block = '''                    st.session_state.sources.append({
                        "name": source_name, 
                        "words": words,
                        "pages": num_pages,
                        "reading_time": reading_time,
                        "raw_text": all_text
                    })
                    save_sources()
                    st.success("Added!")'''
content = content.replace(add_source_block, new_add_source_block)

# Remove the "Clear Chat History" localS if it exists (it was removed in threading)
# The localS object itself was removed, check if it's anywhere else.
content = content.replace("localS.", "# localS.")

with open(r"c:\Users\ajay\Desktop\StudyMind-RAG\app.py", "w", encoding="utf-8") as f:
    f.write(content)
