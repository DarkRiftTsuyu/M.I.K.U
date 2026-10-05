from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import VAULT_PATH, MEMORY_TYPE_FILES, SEARCH_BACKEND, KNOWLEDGE_DIR

log = logging.getLogger("miku.search")

_chroma_client     = None
_chroma_collection = None
_chroma_available  = False

_STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from",
    "i","if","in","into","is","it","me","my","of","on","or",
    "our","so","that","the","their","then","they","this","to",
    "we","with","you","your",
}


def init_chroma() -> bool:
    global _chroma_client, _chroma_collection, _chroma_available
    if _chroma_available:
        return True
    try:
        import chromadb
        db_path = str(VAULT_PATH / "chroma_db")
        log.info("[Chroma] Initialising at %s", db_path)
        _chroma_client     = chromadb.PersistentClient(path=db_path)
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


def chroma_upsert(entry_id: str, text: str, metadata: dict) -> None:
    if not _chroma_available:
        return
    try:
        _chroma_collection.upsert(ids=[entry_id], documents=[text], metadatas=[metadata])
    except Exception as exc:
        log.warning("[Chroma] Upsert failed: %s", exc)


def chroma_search(query: str, n_results: int = 5) -> list[dict]:
    if not _chroma_available:
        return []
    try:
        results = _chroma_collection.query(query_texts=[query], n_results=n_results)
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            out.append({
                "id":       doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
        return out
    except Exception as exc:
        log.warning("[Chroma] Search failed: %s", exc)
        return []


def chroma_delete_by_keyword(keyword: str) -> int:
    if not _chroma_available:
        return 0
    try:
        results       = _chroma_collection.query(query_texts=[keyword], n_results=20)
        ids_to_delete = []
        kw_lower      = keyword.lower()
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


def rebuild_chroma_index() -> None:
    if not _chroma_available:
        log.warning("[Chroma] Cannot rebuild — Chroma not available.")
        return
    import hashlib
    log.info("[Chroma] Rebuilding index…")
    for mem_type, file in MEMORY_TYPE_FILES.items():
        if not file.exists():
            continue
        content = file.read_text(encoding="utf-8")
        chunks  = content.split("\n## ")
        for chunk in chunks:
            if not chunk.strip():
                continue
            entry_id = hashlib.sha1(chunk.encode()).hexdigest()[:12]
            chroma_upsert(
                entry_id=entry_id,
                text=chunk[:2000],
                metadata={"memory_type": mem_type, "title": chunk.split("\n")[0][:80]},
            )
    log.info("[Chroma] Rebuild complete ✔")


def is_chroma_available() -> bool:
    return _chroma_available


def _query_terms(query: str) -> list[str]:
    parts = re.findall(r"[A-Za-z0-9_\-]+", query.lower())
    return [p for p in parts if len(p) >= 3 and p not in _STOPWORDS][:12]


def _find_snippet(content: str, terms: list[str], snippet_len: int = 500) -> str | None:
    lowered = content.lower()
    hits    = [lowered.find(t) for t in terms if t and lowered.find(t) != -1]
    if not hits:
        return None
    idx     = min(hits)
    start   = max(0, idx - snippet_len // 2)
    end     = min(len(content), start + snippet_len)
    snippet = content[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(content):
        snippet = snippet + "…"
    return snippet


def search_vault(query: str, max_chars: int = 2000) -> str:
    results: list[str] = []
    user_facts_file = MEMORY_TYPE_FILES["user_facts"]
    if user_facts_file.exists():
        try:
            content = user_facts_file.read_text(encoding="utf-8")
            results.append(f"### user_identity\n{content[:1000]}")
        except Exception as exc:
            log.debug("Could not read user_facts: %s", exc)
    if SEARCH_BACKEND == "chroma" and _chroma_available:
        hits    = chroma_search(query, n_results=6)
        if hits:
            parts    = [h["document"] for h in hits]
            combined = "\n\n".join(parts)
            base     = "\n\n".join(results)
            return (base + "\n\n" + combined)[:max_chars]
    terms = _query_terms(query)
    if not terms:
        return "\n\n".join(results)[:max_chars]
    search_targets = list(MEMORY_TYPE_FILES.values()) + [KNOWLEDGE_DIR]
    seen_paths: set[Path] = set()
    for target in search_targets:
        if target.is_file() and target.exists() and target not in seen_paths:
            files = [target]
            seen_paths.add(target)
        elif target.is_dir():
            files = sorted(target.rglob("*.md"))
        else:
            continue
        for file in files:
            if file in seen_paths:
                continue
            seen_paths.add(file)
            try:
                content = file.read_text(encoding="utf-8")
            except Exception as exc:
                log.debug("Could not read %s: %s", file, exc)
                continue
            snippet = _find_snippet(content, terms)
            if not snippet:
                continue
            try:
                rel = file.relative_to(VAULT_PATH)
            except ValueError:
                rel = file
            results.append(f"### {rel.as_posix()}\n{snippet}")
            if sum(len(r) for r in results) >= max_chars:
                break
    return "\n\n".join(results)[:max_chars]
