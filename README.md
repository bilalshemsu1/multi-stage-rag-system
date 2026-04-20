# AstuAI: Multi-Stage RAG System

AstuAI is a production-grade **Retrieval-Augmented Generation (RAG)** system developed for **Adama Science and Technology University (ASTU)**. It bridges the gap between static academic documents and conversational AI by providing a high-performance pipeline for document ingestion, semantic retrieval, and context-aware generation.

---

## 🧬 Scientific & Logical Foundation

### What is RAG?
Standard LLMs like GPT or Gemini are trained on public data but lack knowledge of your private or university-specific documents. **Retrieval-Augmented Generation (RAG)** solves this by providing the AI with a "search engine" for your files. Instead of relying on its memory, the AI searches your documents for relevant facts and uses them to answer, virtually eliminating "hallucinations."

### Our Core Logic
AstuAI doesn't just "search" for keywords; it uses **Vector Math**. By converting text into high-dimensional vectors, the system understands that "Student Registration" and "Enrollment Process" are semantically similar, even if they share no common words.

---

## 🏗 Architectural Layers

### Layer 1: Document Parsing Engine (`ingest.py`)
This layer handles the extraction of raw text from varied formats including **PDF (PyMuPDF)**, **DOCX (python-docx)**, and **PPTX (python-pptx)**. It features:
*   **Metadata Extraction**: Captures document names, subjects, and page numbers.
*   **Clean Normalization**: Strips excessive whitespace and prepares text for embedding.

### Layer 2: Semantic Chunking
Text is split into semantic blocks (default ~2200 characters) with a **350-character overlap**. This ensures that if an important concept is discussed at the end of one chunk, it is also captured at the start of the next, preventing context loss during retrieval.

### Layer 3: Vector Embedding Engine (`core.py`)
We use the **BGE-large-en-v1.5** model, one of the highest-rated open-source embedding models.
*   **Instruction-Based Retrieval**: For queries, we automatically prefix the text with *"Represent this sentence for searching relevant passages:"*. This primes the model to output a vector optimized for search rather than general similarity.

### Layer 4: Distributed Retrieval Layer
Powered by **ChromaDB**, the system uses an **HNSW (Hierarchical Navigable Small World)** index.
*   **Metric**: Cosine Similarity. This ensures that the *direction* of the text's meaning is prioritized over the length or frequency of words.
*   **Filtering**: Supports boolean metadata filters (e.g., searching only within a specific document).

### Layer 5: RAG Orchestration & API (`main.py`)
A **FastAPI** backend that manages the conversation flow:
1.  Reduces user query to a vector.
2.  Retrieves the Top-K most relevant chunks.
3.  Injects those chunks into a "Ground Truth" system prompt.
4.  Ensures strictly Markdown-formatted responses.

### Layer 6: Multi-Provider Generation
Unlike single-model systems, AstuAI supports multiple high-performance backends:
*   **Google Gemini**: Deep reasoning and massive context windows.
*   **Cerebras**: The world's fastest inference for real-time applications.
*   **Zydit**: High-availability OpenAI-compatible endpoints.

---

## 🛠 Features & Capabilities

*   **Streaming SSE**: Server-Sent Events allow the UI to display words as they are generated, providing a fluid user experience.
*   **Hardware Acceleration**: Automatic detection of **NVIDIA (CUDA)**, **Apple (MPS)**, or **Intel (XPU)** hardware to speed up local embeddings.
*   **Citation Engine**: Every answer includes the specific source documents and page ranges used to generate the response.
*   **Modern UI**: Dark-mode optimized interface with full Markdown and LaTeX support.

---

## 📖 Setup & Deployment

### 1. Installation
```bash
git clone https://github.com/bilalshemsu1/multi-stage-rag-system.git
cd multi-stage-rag-system
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and add your API keys.

### 3. Execution
1.  **Ingest**: Place documents in `files/` and run `python ingest.py`.
2.  **Run Server**: `uvicorn main:app --reload`.
3.  **Chat**: Open `index.html` in your browser.

---

## 🤝 Project Credits
Developed for **Adama Science and Technology University (ASTU)** by **Bilal Shemsu**.
