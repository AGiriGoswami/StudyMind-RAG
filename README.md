# 📚 StudyMind - RAG Based Academic Assistant

StudyMind is a Retrieval-Augmented Generation (RAG) based academic assistant that allows users to upload PDF documents and ask questions about their content. It uses Google's Gemini for embeddings, FAISS for efficient similarity search, and Groq's fast Llama 3.1 model to generate accurate answers based on the provided context.

## Features
- **PDF Upload:** Seamlessly extract text from uploaded PDF documents.
- **Intelligent Chunking & Embedding:** Text is chunked and embedded using Google's Gemini (`models/gemini-embedding-2`).
- **Fast Retrieval:** Uses FAISS vector database to retrieve the most relevant context for a user's question.
- **Accurate Answers:** Utilizes Groq's high-speed inference with the Llama 3.1 8B model to generate context-aware answers.

## Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Embeddings:** [Google Gemini API](https://ai.google.dev/)
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **LLM:** [Groq API](https://groq.com/) (running `llama-3.1-8b-instant`)
- **PDF Parsing:** [PyMuPDF](https://pymupdf.readthedocs.io/) (`fitz`)

## Setup & Run Locally

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