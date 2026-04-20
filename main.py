from __future__ import annotations

import os
import time
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool
from google import genai
from google.genai import types

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger("astu_rag")

# Silence noisy libraries
for noisy in ["transformers", "sentence_transformers", "httpx", "watchfiles", "uvicorn"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

console = Console()

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR        = Path(os.environ.get("CHROMA_DIR", "chroma_store"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "pdf_chunks")
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

# Which LLM backend to use: "cerebras" (default) or "zydit"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "cerebras").lower()

# Cerebras
CEREBRAS_MODEL  = os.environ.get("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")
cerebras_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY", ""))

# Zydit  (OpenAI-compatible endpoint)
ZYDIT_MODEL  = os.environ.get("ZYDIT_MODEL", "z-ai/glm5")
zydit_client = OpenAI(
    api_key=os.environ.get("ZYDIT_API_KEY", ""),
    base_url="https://api.zydit.in/v1",
)

# Gemini
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
_gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=_gemini_api_key) if _gemini_api_key else None

# Convenience: active model name for logging / responses
if LLM_PROVIDER == "zydit":
    LLM_MODEL = ZYDIT_MODEL
elif LLM_PROVIDER == "gemini":
    LLM_MODEL = GEMINI_MODEL
else:
    LLM_MODEL = CEREBRAS_MODEL
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if CHROMA_DIR.is_dir():
        try:
            _collection()
        except Exception:
            pass
    yield


# ── Unified LLM helpers (Cerebras & Zydit share the same interface) ───────────
LLM_KWARGS = dict(max_tokens=20000, temperature=0.7, top_p=0.8)

def _llm_stream(messages: List[Dict[str, str]]):
    """Return a streaming iterator regardless of provider."""
    if LLM_PROVIDER == "zydit":
        return zydit_client.chat.completions.create(
            model=LLM_MODEL, messages=messages, stream=True, 
            timeout=LLM_TIMEOUT, **LLM_KWARGS
        )
    elif LLM_PROVIDER == "gemini":
        sys_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        history_msgs = messages[1:] if messages and messages[0]["role"] == "system" else messages
        
        genai_messages = []
        for m in history_msgs:
            role = "model" if m["role"] == "assistant" else m["role"] # "user" or "model" usually
            if role not in ["user", "model"]: role = "user"
            genai_messages.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
            
        config = types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=LLM_KWARGS["temperature"],
            top_p=LLM_KWARGS["top_p"],
            max_output_tokens=LLM_KWARGS["max_tokens"],
        )
        
        class MockDelta:
            def __init__(self, t): self.content = t
        class MockChoice:
            def __init__(self, t): self.delta = MockDelta(t)
        class MockChunk:
            def __init__(self, t): self.choices = [MockChoice(t)]

        for chunk in gemini_client.models.generate_content_stream(
            model=LLM_MODEL, contents=genai_messages, config=config
        ):
            yield MockChunk(chunk.text)
        return

    # default: cerebras
    return cerebras_client.chat.completions.create(
        model=LLM_MODEL, messages=messages, stream=True,
        timeout=LLM_TIMEOUT, # Cerebras SDK supports timeout
        max_completion_tokens=LLM_KWARGS["max_tokens"],
        temperature=LLM_KWARGS["temperature"], top_p=LLM_KWARGS["top_p"],
    )


def _llm_complete(messages: List[Dict[str, str]]) -> str:
    """Return the full answer string regardless of provider."""
    if LLM_PROVIDER == "zydit":
        resp = zydit_client.chat.completions.create(
            model=LLM_MODEL, messages=messages, stream=False,
            timeout=LLM_TIMEOUT, **LLM_KWARGS
        )
        return resp.choices[0].message.content or ""
    elif LLM_PROVIDER == "gemini":
        sys_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        history_msgs = messages[1:] if messages and messages[0]["role"] == "system" else messages
        
        genai_messages = []
        for m in history_msgs:
            role = "model" if m["role"] == "assistant" else m["role"]
            if role not in ["user", "model"]: role = "user"
            genai_messages.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
            
        config = types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=LLM_KWARGS["temperature"],
            top_p=LLM_KWARGS["top_p"],
            max_output_tokens=LLM_KWARGS["max_tokens"],
        )
        
        resp = gemini_client.models.generate_content(
            model=LLM_MODEL, contents=genai_messages, config=config
        )
        return resp.text or ""
    else:
        resp = cerebras_client.chat.completions.create(
            model=LLM_MODEL, messages=messages, stream=False,
            timeout=LLM_TIMEOUT,
            max_completion_tokens=LLM_KWARGS["max_tokens"],
            temperature=LLM_KWARGS["temperature"], top_p=LLM_KWARGS["top_p"],
        )
        return resp.choices[0].message.content or ""


# ── Cached singletons ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _collection():
    if not CHROMA_DIR.is_dir():
        raise RuntimeError(f"Chroma persist directory not found: {CHROMA_DIR.resolve()}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


def _encode_query(text: str) -> List[float]:
    """Embed a query string. Applies BGE instruction prefix automatically."""
    if "bge" in EMBEDDING_MODEL.lower() and "en" in EMBEDDING_MODEL.lower():
        text = "Represent this sentence for searching relevant passages: " + text
    model = _embedder()
    vec = model.encode(
        [text],
        convert_to_tensor=False,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="ASTU RAG API", version="2.0.0", lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log detailed validation errors (like mismatched history schema)."""
    logger.error(f"VALIDATION ERROR: {exc.errors()}")
    # Also print to console for quick look
    console.print(f"\n[bold red]VALIDATION ERROR:[/]\n{exc.errors()}\n")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log the full traceback for any unhandled 500 error."""
    logger.error(f"GLOBAL ERROR: {str(exc)}")
    traceback.print_exc()
    return HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    # Identify client
    ua = request.headers.get("user-agent", "unknown")
    client_label = "Laravel" if "Guzzle" in ua or "PHP" in ua else "Browser/Postman"
    
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    status_color = "green" if response.status_code < 400 else "red"
    if response.status_code >= 400:
        status_color = "bold red"
    
    # Create a small compact table for the request
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_row(
        f"[dim]{request_id}[/]",
        f"[bold blue]{client_label}[/]",
        f"[{status_color}]{response.status_code}[/]",
        f"[bold cyan]{request.method}[/]",
        f"[white]{request.url.path}[/]",
        f"[dim]{process_time:.1f}ms[/]"
    )
    console.print(table)
    return response

@app.on_event("startup")
async def startup_event():
    # Print a beautiful dashboard
    config_table = Table(box=box.SIMPLE, show_header=False)
    config_table.add_column("Key", style="bold cyan")
    config_table.add_column("Value")
    
    config_table.add_row("LLM Provider", f"[bold green]{LLM_PROVIDER.upper()}[/]")
    config_table.add_row("Model Name", LLM_MODEL)
    config_table.add_row("Vector Store", str(CHROMA_DIR.resolve()))
    config_table.add_row("Collection", CHROMA_COLLECTION)
    config_table.add_row("Embedding", EMBEDDING_MODEL)
    
    dashboard = Panel(
        config_table,
        title="[bold white]ASTU RAG API SERVER",
        subtitle="[dim]FastAPI V2.0.0",
        border_style="blue",
        expand=False
    )
    console.print("\n", dashboard, "\n")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatFilters(BaseModel):
    """
    Optional Chroma metadata filters sent from Laravel.
    Each set field is added as an $and clause on top of type=chunk.
    Example: {"source": "astu_docs"}
    """
    source: Optional[str] = Field(None, description="Filter by metadata.source")
    document_name: Optional[str] = Field(None, description="Filter by metadata.document_name")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's latest message")
    history: List[HistoryMessage] = Field(
        default_factory=list,
        description="Prior conversation turns (multi-turn context)",
    )
    top_k: int = Field(5, ge=1, le=50, description="Number of context chunks to retrieve")
    stream: bool = Field(True, description="Stream response as SSE (true) or return JSON (false)")
    @field_validator("history", mode="before")
    @classmethod
    def validate_history_list(cls, v):
        """Handle Laravel sending associative arrays (JSON objects) instead of lists."""
        if isinstance(v, dict):
            # Sort by keys to preserve order if they are string-numeric keys, 
            # then take the values.
            try:
                sorted_keys = sorted(v.keys(), key=lambda x: int(x))
                return [v[k] for k in sorted_keys]
            except (ValueError, TypeError):
                # If keys aren't numeric, just take the values in order
                return list(v.values())
        return v

    filters: Optional[ChatFilters] = Field(None, description="Optional metadata filters")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(10, ge=1, le=200)
    chunk_only: bool = Field(True, description="Exclude doc_marker chunks")
    where: Optional[Dict[str, Any]] = Field(None, description="Optional Chroma metadata filter")


class QueryHit(BaseModel):
    id: str
    text: Optional[str]
    distance: Optional[float]
    metadata: Optional[Dict[str, Any]]


class QueryResponse(BaseModel):
    results: List[QueryHit]
    collection: str
    persist_path: str
    embed_model: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_chroma_where(filters: Optional[ChatFilters]) -> Optional[Dict[str, Any]]:
    """
    Build a Chroma 'where' dict from optional ChatFilters.
    Returns None when no filters are provided so col.query runs without
    a where clause and never crashes on missing metadata fields.
    """
    if not filters:
        return None

    clauses: List[Dict[str, Any]] = []
    if filters.source:
        clauses.append({"source": {"$eq": filters.source}})
    if filters.document_name:
        clauses.append({"document_name": {"$eq": filters.document_name}})

    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _build_llm_messages(
    system_prompt: str,
    history: List[HistoryMessage],
    user_message: str,
) -> List[Dict[str, str]]:
    """Assemble system + history + current user message for the LLM."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_message})
    return messages


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        col = _collection()
        return {
            "ok": True,
            "collection": CHROMA_COLLECTION,
            "persist_path": str(CHROMA_DIR.resolve()),
            "count": col.count(),
            "embed_model": EMBEDDING_MODEL,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/v1/query", response_model=QueryResponse)
def query_vectors(body: QueryRequest):
    """Raw vector retrieval without LLM – useful for debugging context quality."""
    try:
        col = _collection()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")

    where: Optional[Dict[str, Any]] = None
    if body.chunk_only and body.where:
        where = {"$and": [{"type": "chunk"}, body.where]}
    elif body.chunk_only:
        where = {"type": "chunk"}
    elif body.where:
        where = body.where

    try:
        embedding = _encode_query(q)
        res = col.query(
            query_embeddings=[embedding],
            n_results=body.top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma query failed: {e}") from e

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: List[QueryHit] = [
        QueryHit(
            id=str(_id),
            text=doc,
            distance=float(dist) if dist is not None else None,
            metadata=dict(meta) if meta else None,
        )
        for _id, doc, meta, dist in zip(ids, docs, metas, dists)
    ]

    return QueryResponse(
        results=hits,
        collection=CHROMA_COLLECTION,
        persist_path=str(CHROMA_DIR.resolve()),
        embed_model=EMBEDDING_MODEL,
    )


@app.post("/v1/chat")
async def chat(body: ChatRequest):
    """
    Full 8-step RAG pipeline.

    Step 1  Validate request
    Step 2  Prepare query  (BGE prefix added inside _encode_query)
    Step 3  Embed query
    Step 4  Retrieve top_k chunks from Chroma (with optional filters)
    Step 5  (Optional) Rerank  ← placeholder, Chroma distance order used
    Step 6  Build system prompt with source-labelled context blocks
    Step 7  Call Cerebras LLM  (streaming or batch)
    Step 8  Yield SSE tokens / return JSON

    Laravel SSE consumer example:
        $response = Http::withOptions(['stream' => true])
                        ->post('http://localhost:8000/v1/chat', $payload);
        foreach ($response->throw()->toPsrResponse()->getBody() as $chunk) {
            // each chunk is "data: <token>\n\n"  or  "data: [DONE]\n\n"
        }
    """

    # ── Step 1: Validate ─────────────────────────────────────────────────────
    q = body.message.strip()
    if not q:
        raise HTTPException(status_code=422, detail="message must not be empty")

    try:
        col = _collection()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {e}") from e

    # ── Step 2 & 3: Prepare + Embed query ────────────────────────────────────
    try:
        # OFFLOAD TO THREADPOOL: _encode_query involves CPU-heavy model inference
        embedding = await run_in_threadpool(_encode_query, q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}") from e

    # ── Step 4: Retrieve top_k chunks ────────────────────────────────────────
    where = _build_chroma_where(body.filters)
    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": body.top_k,
        "include": ["documents", "metadatas"],
    }
    if where is not None:
        query_kwargs["where"] = where

    try:
        # OFFLOAD TO THREADPOOL: Chroma query is synchronous
        res = await run_in_threadpool(col.query, **query_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}") from e

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    # ── Step 5: Rerank (placeholder) ─────────────────────────────────────────
    # Chroma already returns chunks ordered by cosine distance.
    # Insert a cross-encoder reranker here when available.

    # ── Build deduplicated sources list ──────────────────────────────────────
    # One entry per unique (document_name, page-range) pair so the client can
    # display "Referenced from: X, pages Y-Z" under the chat bubble.
    seen_sources: set = set()
    sources: List[Dict[str, Any]] = []
    for meta in metas:
        if not meta:
            continue
        name = meta.get("document_name", "")
        start = meta.get("start_page")
        end   = meta.get("end_page")
        key   = (name, start, end)
        if name and key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "document": name,
                "start_page": start,
                "end_page": end,
            })

    # ── Step 6: Build prompt ──────────────────────────────────────────────────
    context_blocks: List[str] = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        if not doc:
            continue
        label = ""
        if meta:
            name = meta.get("document_name", "")
            pages = f"p{meta.get('start_page', '?')}-{meta.get('end_page', '?')}"
            label = f"[Source {i}: {name} {pages}]\n" if name else f"[Source {i}]\n"
        context_blocks.append(f"{label}{doc.strip()}")

    context_text = (
        "\n\n---\n\n".join(context_blocks)
        if context_blocks
        else "No relevant context found in the document repository."
    )

    system_prompt = (
        # ── STRICT OUTPUT ENFORCEMENT ─────────────────────────────────────────
        "STRICT REQUIREMENT: You MUST format EVERY response using valid Markdown syntax. "
        "Plain text responses are FORBIDDEN. "
        "Every response must contain at least one Markdown element "
        "(heading, bullet list, bold, table, or blockquote). "
        "Failure to use Markdown formatting is a violation of your output contract.\n\n"
        "---\n\n"
        # ── Core identity & answer rules ──────────────────────────────────────
        "You are a knowledgeable AI assistant for Adama Science and Technology University (ASTU). "
        "Answer questions accurately using ONLY the retrieved context provided below. "
        "If the answer is not present in the context, clearly say you don't have that information "
        "in the available documents. Keep answers concise, well-structured, and helpful.\n\n"
        "---\n\n"
        # ── Markdown formatting instructions ──────────────────────────────────
        "Your responses will be rendered using markdown-it. "
        "Use Markdown syntax intelligently to produce clean, structured, and readable academic content.\n\n"
        "AVAILABLE MARKDOWN TOOLS:\n\n"
        "1. Headings: # Main Title / ## Section / ### Subsection\n"
        "2. Lists: bullet points for concepts, numbered lists for steps\n"
        "3. Emphasis: **bold** for key terms, *italic* for definitions\n"
        "4. Tables: ONLY for comparisons, classifications, or structured data\n"
        "5. Blockquotes: > for important principles or key statements\n"
        "6. Code blocks: for code, formulas, or structured examples\n"
        "7. Horizontal rules: --- to separate major sections\n\n"
        "STRUCTURE RULES:\n\n"
        "- Start with a clear title (# or ##)\n"
        "- Break content into logical sections using headings\n"
        "- Use bullet points instead of long paragraphs\n"
        "- Avoid walls of text — keep paragraphs max 2-3 lines\n\n"
        "WHEN TO USE WHAT:\n\n"
        "- Definitions → short paragraph or bullet\n"
        "- Multiple concepts → bullet list\n"
        "- Categories/types → subsections or table\n"
        "- Comparisons → table\n"
        "- Processes → numbered list\n"
        "- Important principle → blockquote\n\n"
        "STYLE RULES:\n\n"
        "- Be academic and clear\n"
        "- Do NOT overuse emojis\n"
        "- Do NOT mix too many styles randomly\n"
        "- Prioritize clarity over decoration\n\n"
        "---\n\n"
        # ── Final reminder (reinforces compliance at end of prompt) ───────────
        "REMINDER: Your entire response MUST be formatted in Markdown. "
        "Do NOT write plain prose without any Markdown structure. "
        "Begin your response with a Markdown heading.\n\n"
        f"RETRIEVED CONTEXT:\n{context_text}"
    )

    llm_messages = _build_llm_messages(system_prompt, body.history, q)

    # ── Step 7 & 8: Call LLM → stream SSE tokens ────────────────────────────
    def generate():
        import json as _json
        try:
            yield f"event: sources\ndata: {_json.dumps(sources)}\n\n"

            for chunk in _llm_stream(llm_messages):
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield f"data: {content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"STREAM ERROR: {str(e)}")
            traceback.print_exc()
            yield f"data: [ERROR] {str(e)}\n\n"

    if body.stream:
        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── Non-streaming fallback (stream: false) ────────────────────────────────
    try:
        # OFFLOAD TO THREADPOOL: LLM calls are blocking I/O
        answer = await run_in_threadpool(_llm_complete, llm_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}") from e

    return {
        "answer": answer,
        "sources": sources,
        "context_chunks": len(docs),
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
    }
