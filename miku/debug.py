"""
Run this from the same directory as miku_assistant.py to diagnose
why knowledge entries are not being written to disk.

Usage:
    python debug_memory.py
"""
from __future__ import annotations
import json
import os
import sys
import hashlib
import traceback
from pathlib import Path
from datetime import datetime

# ── Replicate config from miku_assistant.py ──────────────────────────────────
VAULT_PATH = Path("vault")
KNOWLEDGE_DIR = VAULT_PATH / "knowledge"
KNOWLEDGE_INDEX_FILE = KNOWLEDGE_DIR / "_index.json"
KNOWLEDGE_CONFIDENCE_THRESHOLD = float(os.environ.get("KNOWLEDGE_CONFIDENCE_THRESHOLD", "0.7"))

MEMORY_TYPE_FILES = {
    "user_facts":  KNOWLEDGE_DIR / "user_facts.md",
    "projects":    KNOWLEDGE_DIR / "projects.md",
    "preferences": KNOWLEDGE_DIR / "preferences.md",
    "general":     KNOWLEDGE_DIR / "general.md",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP = "─" * 60

def ok(msg):  print(f"  \033[32m✓\033[0m  {msg}")
def fail(msg): print(f"  \033[31m✗\033[0m  {msg}")
def warn(msg): print(f"  \033[33m⚠\033[0m  {msg}")
def info(msg): print(f"  \033[36m·\033[0m  {msg}")

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cwd():
    print(SEP)
    print("1. Working directory")
    cwd = Path.cwd()
    info(f"cwd = {cwd}")
    vault = cwd / "vault"
    if vault.exists():
        ok(f"vault/ exists at {vault}")
    else:
        fail(f"vault/ NOT found at {vault}")
        warn("Run this script from the same folder as miku_assistant.py")


def test_vault_structure():
    print(SEP)
    print("2. Vault directory structure")
    dirs = [
        VAULT_PATH,
        VAULT_PATH / "identity",
        VAULT_PATH / "memory" / "conversations",
        KNOWLEDGE_DIR,
    ]
    for d in dirs:
        if d.exists():
            ok(f"{d}  (exists)")
        else:
            fail(f"{d}  MISSING")
            try:
                d.mkdir(parents=True, exist_ok=True)
                warn(f"  → created {d}")
            except Exception as e:
                fail(f"  → could not create: {e}")


def test_write_permission():
    print(SEP)
    print("3. Write permissions")
    test_file = KNOWLEDGE_DIR / "_debug_write_test.tmp"
    try:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        ok(f"Can write to {KNOWLEDGE_DIR}")
    except Exception as e:
        fail(f"Cannot write to {KNOWLEDGE_DIR}: {e}")
        warn("Check file-system permissions or whether the path is read-only")


def test_index():
    print(SEP)
    print("4. Knowledge index")
    if not KNOWLEDGE_INDEX_FILE.exists():
        warn("_index.json does not exist yet (will be created on first save)")
        return
    try:
        raw = KNOWLEDGE_INDEX_FILE.read_text(encoding="utf-8")
        obj = json.loads(raw)
        ids  = obj.get("ids", [])
        tags = obj.get("tag_counts", {})
        info(f"Index has {len(ids)} entries, {len(tags)} tracked tags")
        ok("Index loaded successfully")
    except json.JSONDecodeError as e:
        fail(f"_index.json is corrupt: {e}")
        warn("Delete vault/knowledge/_index.json and restart")


def test_memory_type_files():
    print(SEP)
    print("5. Memory type files")
    for name, path in MEMORY_TYPE_FILES.items():
        if path.exists():
            size = path.stat().st_size
            ok(f"{name:<14} → {path}  ({size} bytes)")
        else:
            warn(f"{name:<14} → {path}  (not yet created)")


def test_propose_knowledge_mock():
    """Simulate what propose_knowledge returns and trace through save_knowledge_entry."""
    print(SEP)
    print("6. Simulated save pipeline (dry run)")

    # Fake entry that should definitely be saved
    entry = {
        "should_save": True,
        "title": "Debug test entry",
        "tags": ["test", "debug", "user"],
        "summary_md": "- This is a debug test entry\n- It should be saved to user_facts.md",
        "confidence": 0.95,
    }

    print()
    info(f"Input entry: {json.dumps(entry, indent=4)}")
    print()

    # Step 1: confidence gate
    confidence = float(entry.get("confidence", 0.0))
    if confidence < KNOWLEDGE_CONFIDENCE_THRESHOLD:
        fail(f"BLOCKED at confidence gate: {confidence:.2f} < {KNOWLEDGE_CONFIDENCE_THRESHOLD}")
        return
    ok(f"Confidence gate passed: {confidence:.2f} >= {KNOWLEDGE_CONFIDENCE_THRESHOLD}")

    # Step 2: should_save flag
    if not entry.get("should_save"):
        fail("BLOCKED: should_save is False")
        return
    ok("should_save = True")

    # Step 3: title / summary
    title = str(entry.get("title", "")).strip()
    summary_md = str(entry.get("summary_md", "")).strip()
    tags = [str(t).strip() for t in entry.get("tags", []) if str(t).strip()]
    tags_line = " ".join(f"#{t}" for t in tags[:12])
    ok(f"title = '{title}'")
    ok(f"tags  = {tags}")

    # Step 4: security filter
    material = (title + "\n" + tags_line + "\n" + summary_md).strip()
    blocked_terms = ("api_key", "apikey", "secret", "password", "private key", "-----begin", "token")
    hit = next((t for t in blocked_terms if t in material.lower()), None)
    if hit:
        fail(f"BLOCKED by security filter: found '{hit}' in material")
        return
    ok("Security filter passed")

    # Step 5: deduplication
    entry_id = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    info(f"entry_id = {entry_id}")
    index = _load_index_safe()
    known_ids = set(index.get("ids", []))
    if entry_id in known_ids:
        fail(f"BLOCKED: entry_id {entry_id} already in index (duplicate)")
        warn("The same text was saved before — this is expected behaviour, not a bug")
        return
    ok(f"Not a duplicate (id {entry_id} is new)")

    # Step 6: memory type routing
    mem_type = _resolve_type(tags, title)
    target_file = MEMORY_TYPE_FILES[mem_type]
    info(f"Routing to memory type: '{mem_type}' → {target_file}")

    # Step 7: actual write
    now = datetime.now()
    block = "\n".join([
        "",
        f"## {title} ({now.strftime('%H:%M')})",
        f"<!-- id: {entry_id} confidence: {confidence:.2f} -->",
        tags_line,
        "",
        summary_md,
        "",
        "---",
        "",
    ])

    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, "a", encoding="utf-8") as f:
            f.write(block)
        ok(f"Written to {target_file}")
    except Exception as e:
        fail(f"FILE WRITE FAILED: {e}")
        traceback.print_exc()
        return

    # Step 8: index update
    known_ids.add(entry_id)
    index["ids"] = sorted(known_ids)
    try:
        KNOWLEDGE_INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")
        ok(f"Index updated: {KNOWLEDGE_INDEX_FILE}")
    except Exception as e:
        fail(f"INDEX WRITE FAILED: {e}")
        traceback.print_exc()
        return

    # Verify the file actually contains the content
    content = target_file.read_text(encoding="utf-8")
    if entry_id in content:
        ok(f"Verified: entry_id {entry_id} found in {target_file}")
    else:
        fail(f"entry_id NOT found in file after write — something is very wrong")

    print()
    print(f"\033[32mAll steps passed. Check {target_file} for the debug entry.\033[0m")


def test_ollama_response_parsing():
    """Check whether propose_knowledge's JSON parsing would work on a real sample."""
    print(SEP)
    print("7. JSON extraction (simulated LLM output)")

    samples = [
        # Clean JSON
        '{"should_save": true, "title": "User likes Python", "tags": ["python", "preference"], "summary_md": "- Prefers Python", "confidence": 0.85}',
        # JSON wrapped in markdown fences (common LLM failure)
        '```json\n{"should_save": true, "title": "test", "tags": [], "summary_md": "- test", "confidence": 0.9}\n```',
        # JSON buried in prose (another common failure)
        'Sure! Here is what I found:\n{"should_save": true, "title": "test", "tags": [], "summary_md": "- ok", "confidence": 0.8}\nHope that helps!',
        # Low confidence (should be filtered)
        '{"should_save": true, "title": "test", "tags": [], "summary_md": "- uncertain", "confidence": 0.4}',
        # should_save = false
        '{"should_save": false, "title": "chit-chat", "tags": [], "summary_md": "- nothing durable", "confidence": 0.95}',
    ]

    for i, raw in enumerate(samples, 1):
        obj = _extract_json(raw)
        if obj is None:
            fail(f"Sample {i}: could not extract JSON")
            info(f"  raw = {raw[:80]!r}")
            continue
        confidence = float(obj.get("confidence", 0.0))
        should_save = bool(obj.get("should_save"))
        if confidence < KNOWLEDGE_CONFIDENCE_THRESHOLD:
            warn(f"Sample {i}: parsed OK but confidence {confidence:.2f} below threshold → will be discarded")
        elif not should_save:
            warn(f"Sample {i}: parsed OK but should_save=False → will be discarded")
        else:
            ok(f"Sample {i}: parsed OK, confidence={confidence:.2f}, title={obj.get('title')!r}")


# ── Helpers replicated from miku_assistant.py ─────────────────────────────────

def _load_index_safe() -> dict:
    try:
        if KNOWLEDGE_INDEX_FILE.exists():
            raw = KNOWLEDGE_INDEX_FILE.read_text(encoding="utf-8")
            obj = json.loads(raw)
            if isinstance(obj, dict):
                obj.setdefault("tag_counts", {})
                obj.setdefault("title_counts", {})
                return obj
    except Exception:
        pass
    return {"ids": [], "tag_counts": {}, "title_counts": {}}


_TAG_TO_TYPE = {
    "preference": "preferences", "preferences": "preferences",
    "style": "preferences", "likes": "preferences", "dislikes": "preferences",
    "project": "projects", "projects": "projects", "task": "projects",
    "goal": "projects", "deadline": "projects",
    "fact": "user_facts", "user": "user_facts", "name": "user_facts",
    "age": "user_facts", "location": "user_facts", "job": "user_facts",
    "background": "user_facts",
    "debug": "user_facts", "test": "user_facts",
}

def _resolve_type(tags: list[str], title: str) -> str:
    for tag in tags:
        t = tag.lower().strip()
        if t in _TAG_TO_TYPE:
            return _TAG_TO_TYPE[t]
    title_lower = title.lower()
    for keyword, mem_type in _TAG_TO_TYPE.items():
        if keyword in title_lower:
            return mem_type
    return "general"


def _extract_json(text: str) -> dict | None:
    import re
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n\033[1mMiku memory diagnostics\033[0m")
    print(f"Confidence threshold : {KNOWLEDGE_CONFIDENCE_THRESHOLD}")
    print(f"Vault path           : {Path('vault').resolve()}")
    print()
    test_cwd()
    test_vault_structure()
    test_write_permission()
    test_index()
    test_memory_type_files()
    test_propose_knowledge_mock()
    test_ollama_response_parsing()
    print(SEP)
    print()