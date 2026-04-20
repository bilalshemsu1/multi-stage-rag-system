# AstuAI: Multi-Stage RAG System

AstuAI is a professional-grade Retrieval-Augmented Generation (RAG) system developed for Adama Science and Technology University (ASTU). It allows users to upload academic documents (PDF, DOCX, PPTX), index them into a high-performance vector database, and chat with them using state-of-the-art LLMs like Google Gemini, Cerebras, and Zydit.

## 🚀 System Overview

The system operates in a multi-stage pipeline:
1.  **Ingestion Stage**: Documents are parsed, split into semantic chunks with overlap, and converted into mathematical vectors using the **BGE-large** embedding model. These vectors are stored in **ChromaDB**.
2.  **Retrieval Stage**: When a user asks a question, the system embeds the query and performs a semantic search to find the most relevant document sections.
3.  **Augmentation Stage**: The retrieved context is formatted into a refined system prompt.
4.  **Generation Stage**: The LLM (Gemini/Cerebras/Zydit) generates a streaming response based strictly on the provided academic context.

---

## 🛠 Features

*   **Multi-Format Support**: Native parsing for PDF, DOCX, and PPTX.
*   **Intelligent Chunking**: Semantic splitting that preserves context across page boundaries.
*   **High-Speed Retrieval**: Powered by ChromaDB with HNSW indexing.
*   **Flexible LLM Backend**: Built-in support for:
    *   **Google Gemini** (via `google-genai` SDK)
    *   **Cerebras** (Ultra-fast inference)
    *   **Zydit** (OpenAI-compatible)
*   **Rich UI**: A clean, modern web interface with Markdown rendering and source citations.
*   **Streaming Responses**: Real-time answer generation using Server-Sent Events (SSE).

---

## ⚙️ Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   An API Key for Gemini, Cerebras, or Zydit.

### 2. Clone and Install
```bash
# Clone the repository
git clone https://github.com/bilalshemsu1/multi-stage-rag-system.git
cd multi-stage-rag-system

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory (use `.env.example` as a template):
```env
GEMINI_API_KEY="your_api_key_here"
LLM_PROVIDER=gemini
```

---

## 📖 Usage Guide

### Step 1: Prepare Your Documents
Place all your PDF, DOCX, or PPTX files into the `files/` directory.

### Step 2: Indexing (Ingestion)
Run the ingestion script to process the files and build the vector database:
```bash
python ingest.py
```
*The script will display a beautiful dashboard showing the progress and hardware acceleration (CPU/GPU) status.*

### Step 3: Start the API Server
Launch the FastAPI backend:
```bash
uvicorn main:app --reload
```
The server will start at `http://localhost:8000`.

### Step 4: Open the UI
Simply open `index.html` in your browser to start chatting with your documents!

---

## 🧬 Technical Architecture

*   **Embeddings**: Uses `BAAI/bge-large-en-v1.5` via `sentence-transformers`. It includes automatic BGE instruction prefixing for better retrieval accuracy.
*   **Vector Store**: `ChromaDB` (Persistent) using cosine similarity.
*   **API Framework**: `FastAPI` with asynchronous streaming handlers.
*   **UI**: Vanilla JS and CSS with `markdown-it` for high-performance rendering.

---

## 🤝 Project Credits
Developed for **Adama Science and Technology University (ASTU)**.

---
*Developed by Bilal Shemsu*
