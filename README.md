. 9i90# 🧠 StudyMind - Modern AI Research Workspace

StudyMind is a premium, Retrieval-Augmented Generation (RAG) based academic workspace designed to help users ingest, analyze, and understand complex information. Built with a sleek, responsive UI similar to NotebookLM and Perplexity AI, it securely persists all of your data and provides powerful tools to interact with your knowledge base.

It uses **Google Gemini** for intelligent text embedding, **FAISS** for lightning-fast similarity search, and **Groq (Llama 3.1)** for blazing-fast conversational answers.

## ✨ Key Features of power 8i889>
- **Universal Knowledge Ingestion:** Upload PDF documents, paste raw text, scrape Web URLs, or fetch full transcripts directly from YouTube video links!
- **State-of-the-Art State Persistence:** Zero data loss on refresh. All chat history, source documents, FAISS embeddings, and user notes are securely saved to disk in a `.workspace/` directory.
- **Two-Speaker Podcast Generation:** Automatically converts your uploaded documents into a dynamic, two-speaker conversational podcast using Microsoft Edge TTS (`edge-tts`). Watch the real-time progress tracker as the AI writes the script, records the host and guest, and mixes the audio tracks.
- **AI Suggested Questions:** The moment you upload a document, the AI reads it and generates highly relevant, clickable suggested questions to help you start exploring immediately.
- **Mermaid.js Flowcharts:** Ask the AI to visualize a process or concept, and it will automatically generate and render a beautiful, downloadable flowchart.
- **Flashcard Generation:** Instantly turn your knowledge base into an interactive, digital flashcard deck for studying.
- **Rate-Limit Resiliency:** Built-in exponential backoff and dynamic pacing to gracefully handle Google's Free Tier API limit

## 🛠️ Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/) with custom HTML/CSS for a responsive, premium UI.
- **Embeddings:** [Google Gemini API](https://ai.google.dev/) (`gemini-embedding-001`)
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)
- **LLM:** [Groq API](https://groq.com/) (`llama-3.1-8b-instant`)
- **Document & Web Parsing:** `pypdf`, `beautifulsoup4`, `youtube-transcript-api`
- **Audio Generation:** `edge-tts` (Two-Speaker Podcast), `gTTS` (Chat Read-Aloud)

## 🚀 Setup & Run Locally

### 1. Clone the repository and navigate to the project directory
```bash
git clone <your-repo-url>
cd StudyMind-RAG
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# On Windows
.\.venv\Scripts\activate

# On Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory and add your API keys:
```ini
GOOGLE_API_KEY="your_google_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"
```
*Note: You can get a free Groq API key at [console.groq.com](https://console.groq.com/) and a Gemini API key at [Google AI Studio](https://aistudio.google.com/).*

### 5. Run the Application
```bash
streamlit run app.py
```