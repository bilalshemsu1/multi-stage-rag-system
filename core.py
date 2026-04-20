import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_SOURCE_DIR = Path(os.environ.get("PDF_SOURCE_DIR", "files"))
DEFAULT_CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", "chroma_store"))
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "pdf_chunks")
DEFAULT_EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
DEFAULT_EMBED_BATCH = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

from tenacity import retry, stop_after_attempt, wait_exponential

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from sentence_transformers import SentenceTransformer
        import torch

        self.model_name = model_name
        self.batch_size = batch_size
        
        # Device selection: standard priority logic
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        # We retry initialization because it might fail if the model needs to be downloaded 
        # and there's a temporary network glitch.
        self.model = self._load_model_with_retry(model_name, device)
        
        self.dim = _safe_int(self.model.get_sentence_embedding_dimension(), 0)
        if self.dim <= 0:
            raise RuntimeError("Could not determine embedding dimension.")

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _load_model_with_retry(self, model_name: str, device: str) -> Any:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=device)

    @retry(
        wait=wait_exponential(min=1, max=5),
        stop=stop_after_attempt(2),
        reraise=True
    )
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self.model.encode(
                batch,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Ensure plain python lists (Chroma accepts list[list[float]])
            for v in vecs:
                if hasattr(v, "tolist"):
                    v = v.tolist()
                out.append(list(map(float, v)))
        return out


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedding_dim_hint: Optional[int] = None, embedder: Optional[Any] = None):
        import chromadb

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        # cosine is best with normalized embeddings
        metadata = {
            "hnsw:space": "cosine",
            "created_by": "process-file.py",
        }
        if embedding_dim_hint is not None:
            metadata["embedding_dim_hint"] = embedding_dim_hint
            
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
        )

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
            embeddings=[list(map(float, e)) for e in embeddings],
        )

    def get_ids_by_doc(self, document_name: str) -> List[str]:
        """
        Best-effort. Chroma doesn't support "return only ids with where" directly in all versions,
        but `get(where=...)` returns ids + metadatas.
        """
        try:
            res = self.collection.get(where={"document_name": document_name})
            return list(res.get("ids") or [])
        except Exception:
            return []

    def delete_ids(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=list(ids))

    def query(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        # Chroma 1.5+: ids are returned outside `include`; listing "ids" in include raises.
        kwargs = {
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if self.embedder:
            # Add instruction prefix for BGE models to improve retrieval
            prefix = ""
            if "bge" in self.embedder.model_name.lower() and "en" in self.embedder.model_name.lower():
                prefix = "Represent this sentence for searching relevant passages: "
            kwargs["query_embeddings"] = self.embedder.embed([prefix + query_text])
        else:
            kwargs["query_texts"] = [query_text]

        res = self.collection.query(**kwargs)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        out: List[Dict[str, Any]] = []
        for doc, meta, dist, _id in zip(docs, metas, dists, ids):
            out.append({"id": _id, "text": doc, "metadata": meta, "distance": dist})
        return out
