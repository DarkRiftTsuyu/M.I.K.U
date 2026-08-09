import logging
from . import helpers, sharedConst
import hashlib

_chroma_client     = None
_chroma_collection = None
_chroma_available  = False

VAULT_PATH = None
MEMORY_TYPE_FILES = None

log = logging.getLogger("miku")

def _init_chroma() -> bool:
    global _chroma_client, _chroma_collection, _chroma_available
    if _chroma_available:
        return True
    try:
        import chromadb
        db_path = str(VAULT_PATH / "chroma_db")
        log.info("[Chroma] Initialising at %s", db_path)
        _chroma_client = chromadb.PersistentClient(path=db_path)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="miku_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        _chroma_available = True
        log.info("[Chroma] Ready ✔")
        return True
    except Exception as exc:
        log.warning("[Chroma] Not available: %s", exc)
        return False


def rebuild_chroma_index() -> None:
    if not _chroma_available:
        log.warning("[Chroma] Cannot rebuild — Chroma not available.")
        return
    log.info("[Chroma] Rebuilding index…")
    for mem_type, file in MEMORY_TYPE_FILES.items():
        if not file.exists():
            continue
        content = file.read_text(encoding="utf-8")
        chunks = content.split("\n## ")
        for chunk in chunks:
            if not chunk.strip():
                continue
            entry_id = hashlib.sha1(chunk.encode()).hexdigest()[:12]
            _chroma_upsert(
                entry_id=entry_id,
                text=chunk[:2000],
                metadata={"memory_type": mem_type, "title": chunk.split("\n")[0][:80]},
            )
    log.info("[Chroma] Rebuild complete ✔")


def _chroma_upsert(entry_id: str, text: str, metadata: dict) -> None:
    if not _chroma_available:
        return
    try:
        _chroma_collection.upsert(ids=[entry_id], documents=[text], metadatas=[metadata])
    except Exception as exc:
        log.warning("[Chroma] Upsert failed: %s", exc)


def _chroma_search(query: str, n_results: int = 5) -> list[dict]:
    if not _chroma_available:
        return []
    try:
        results = _chroma_collection.query(query_texts=[query], n_results=n_results)
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            out.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
        return out
    except Exception as exc:
        log.warning("[Chroma] Search failed: %s", exc)
        return []


def _chroma_delete_by_keyword(keyword: str) -> int:
    if not _chroma_available:
        return 0
    try:
        results = _chroma_collection.query(query_texts=[keyword], n_results=20)
        ids_to_delete = []
        kw_lower = keyword.lower()
        for i, doc_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i].lower()
            if kw_lower in doc:
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            _chroma_collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception as exc:
        log.warning("[Chroma] Delete failed: %s", exc)
        return 0