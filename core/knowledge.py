from . import sharedConst, helpers, chroma
import logging
import hashlib
from datetime import datetime
import json
import re

log = logging.getLogger("miku")

MEMORY_TYPE_FILES = None
KNOWLEDGE_DIR = None
KNOWLEDGE_INDEX_FILE = None
KNOWLEDGE_DIR = None

def save_knowledge_entry(entry: dict) -> bool:
    try:
        title      = str(entry.get("title", "")).strip()
        tags       = entry.get("tags", []) or []
        summary_md = str(entry.get("summary_md", "")).strip()
        confidence = float(entry.get("confidence", 0.0))
    except Exception as exc:
        log.warning("save_knowledge_entry: bad entry: %s", exc)
        return False

    if not title or not summary_md:
        return False
    if confidence < sharedConst.KNOWLEDGE_CONFIDENCE_THRESHOLD:
        return False

    if chroma._chroma_available:
        existing = chroma._chroma_search(title + " " + summary_md[:200], n_results=1)
        if existing:
            doc = existing[0].get("document", "")
            title_words = set(title.lower().split())
            doc_words   = set(doc.lower().split())
            overlap = len(title_words & doc_words) / max(len(title_words), 1)
            if overlap > 0.7:
                log.debug("[knowledge] Skipping near-duplicate: %s", title)
                return False

    tags_clean = [str(t).strip() for t in tags if str(t).strip()][:12]
    mem_type   = _resolve_memory_type(tags_clean, title)
    target_file = MEMORY_TYPE_FILES.get(mem_type, MEMORY_TYPE_FILES["general"])
    target_file.parent.mkdir(parents=True, exist_ok=True)

    entry_id = hashlib.sha1((title + "\n" + summary_md).encode("utf-8")).hexdigest()[:12]

    block = (
        f"\n## {title}\n"
        f"<!-- id: {entry_id} -->\n"
        f"<!-- confidence: {confidence:.2f} -->\n"
        f"<!-- tags: {', '.join(tags_clean)} -->\n\n"
        f"{summary_md}\n\n---\n"
    )

    try:
        with open(target_file, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception as exc:
        log.warning("save_knowledge_entry write failed: %s", exc)
        return False

    index = _load_knowledge_index()
    ids   = index.setdefault("ids", [])
    if entry_id not in ids:
        ids.append(entry_id)
    _check_and_reinforce(index, tags_clean, title, summary_md)
    _save_knowledge_index(index)

    try:
        chroma._chroma_upsert(
            entry_id=entry_id,
            text=(title + "\n" + summary_md)[:2000],
            metadata={"memory_type": mem_type, "title": title},
        )
    except Exception as exc:
        log.warning("Chroma upsert failed: %s", exc)

    return True

def _load_knowledge_index() -> dict:
    try:
        if KNOWLEDGE_INDEX_FILE.exists():
            obj = json.loads(KNOWLEDGE_INDEX_FILE.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj.setdefault("tag_counts", {})
                obj.setdefault("title_counts", {})
                return obj
    except Exception as exc:
        log.warning("Could not load knowledge index: %s", exc)
    return {"ids": [], "tag_counts": {}, "title_counts": {}}


def _save_knowledge_index(index: dict) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    index["updated"] = datetime.now().isoformat(timespec="seconds")
    try:
        KNOWLEDGE_INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save knowledge index: %s", exc)

def _check_and_reinforce(index: dict, tags: list[str], title: str, summary_md: str) -> None:
    tag_counts   = index.setdefault("tag_counts", {})
    title_counts = index.setdefault("title_counts", {})
    title_slug   = re.sub(r"\W+", "_", title.lower())[:40]
    title_counts[title_slug] = title_counts.get(title_slug, 0) + 1

    reinforced_tags = []
    for tag in tags:
        tag = tag.lower().strip()
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if tag_counts[tag] == sharedConst.REINFORCE_THRESHOLD:
            reinforced_tags.append(tag)

    if title_counts.get(title_slug, 0) == sharedConst.REINFORCE_THRESHOLD:
        _write_reinforcement_summary(title, tags, summary_md, reason="title")
    for tag in reinforced_tags:
        _write_reinforcement_summary(title, [tag], summary_md, reason=f"tag:{tag}")


def _resolve_memory_type(tags: list[str], title: str) -> str:
    for tag in tags:
        t = tag.lower().strip()
        if t in sharedConst._TAG_TO_TYPE:
            return sharedConst._TAG_TO_TYPE[t]
    title_lower = title.lower()
    for keyword, mem_type in sharedConst._TAG_TO_TYPE.items():
        if keyword in title_lower:
            return mem_type
    return "general"


def _write_reinforcement_summary(title: str, tags: list[str], summary_md: str, reason: str) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    filename  = datetime.now().strftime("%Y-%m-%d") + ".md"
    file_path = KNOWLEDGE_DIR / filename
    tags_line = " ".join(f"#reinforced #{t}" for t in tags[:6])
    now = datetime.now()
    block = (
        f"\n## [REINFORCED] {title} ({now.strftime('%H:%M')})\n"
        f"<!-- reinforcement-reason: {reason} -->\n"
        f"{tags_line}\n\n"
        f"> This topic has appeared {sharedConst.REINFORCE_THRESHOLD}+ times — flagged as important.\n\n"
        f"{summary_md}\n\n---\n"
    )
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(block)
        log.info("Reinforcement summary written: %s", reason)
    except Exception as exc:
        log.warning("Could not write reinforcement summary: %s", exc)


_TRIVIAL_MSG_RE = re.compile(r"^[\w\s]{1,12}$")

def propose_knowledge(user_input: str, assistant_response: str) -> dict | None:
    if len(user_input.strip()) < 20 or _TRIVIAL_MSG_RE.match(user_input.strip()):
        return None

    url    = helpers._urljoin(sharedConst.OLLAMA_HOST, "/api/generate")
    prompt = (
        "You are a knowledge distiller for a local personal assistant.\n\n"
        "Task:\n"
        "- Extract ONLY durable, reusable knowledge worth storing for later.\n"
        "- Examples: user preferences, project conventions, decisions, definitions, stable facts.\n"
        "- Avoid: one-off chit chat, transient plans, anything sensitive.\n\n"
        "Return STRICT JSON with keys:\n"
        "- should_save: boolean\n"
        "- title: string (short)\n"
        "- tags: array of short strings\n"
        "- summary_md: string (markdown, 3-10 bullets max)\n"
        "- confidence: number from 0.0 to 1.0\n\n"
        "If nothing durable: should_save=false.\n\n"
        "Conversation:\n"
        f"<user>{user_input}</user>\n"
        f"<assistant>{assistant_response}</assistant>"
    )
    payload = {
        "model": sharedConst.KNOWLEDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 350},
    }
    try:
        resp = helpers._http_json("POST", url, payload=payload, timeout_s=sharedConst.OLLAMA_TIMEOUT_S)
    except Exception as exc:
        log.warning("[knowledge] propose_knowledge failed: %s", exc)
        return None

    text = (resp or {}).get("response", "")
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict) or not obj.get("should_save"):
        return None
    return obj