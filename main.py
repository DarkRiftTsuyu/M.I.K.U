from __future__ import annotations

import json
import logging
import os
import re
import sys
import hashlib
import subprocess
import shutil
import tempfile
import threading
import time
import queue
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import importlib
import pkgutil

from core import speech_IO
from core import sharedConst
from core import pluginHandling
from core import knowledge
from core import helpers
from core import chroma
from core import applications

# dir / path based consts stay here to avoid conflicing being inside core/
BASE_DIR             = Path(__file__).resolve().parent
speech_IO.BASE_DIR = BASE_DIR
pluginHandling.BASE_DIR = BASE_DIR
VAULT_PATH           = BASE_DIR / "vault"
chroma.VAULT_PATH = VAULT_PATH
IDENTITY_FILE        = VAULT_PATH / "identity" / "miku.md"
MEMORY_DIR           = VAULT_PATH / "memory" / "conversations"
KNOWLEDGE_DIR        = VAULT_PATH / "knowledge"
knowledge.KNOWLEDGE_DIR = KNOWLEDGE_DIR
KNOWLEDGE_INDEX_FILE = KNOWLEDGE_DIR / "_index.json"
knowledge.KNOWLEDGE_INDEX_FILE = KNOWLEDGE_INDEX_FILE
STATE_FILE           = VAULT_PATH / "state.json"
REMINDERS_FILE       = VAULT_PATH / "reminders.json"

EVOLUTION_DIR          = VAULT_PATH / "evolution"
CAPABILITY_REQUESTS    = EVOLUTION_DIR / "capability_requests.json"
EVOLUTION_PENDING_DIR  = EVOLUTION_DIR / "pending"
pluginHandling.EVOLUTION_PENDING_DIR = EVOLUTION_PENDING_DIR 
EVOLUTION_SANDBOX_DIR  = EVOLUTION_DIR / "sandbox"
EVOLUTION_METRICS_FILE = EVOLUTION_DIR / "metrics.json"
pluginHandling.EVOLUTION_METRICS_FILE = EVOLUTION_METRICS_FILE

MEMORY_TYPE_FILES: dict[str, Path] = {
    "user_facts":  KNOWLEDGE_DIR / "user_facts.md",
    "projects":    KNOWLEDGE_DIR / "projects.md",
    "preferences": KNOWLEDGE_DIR / "preferences.md",
    "general":     KNOWLEDGE_DIR / "general.md",
}
chroma.MEMORY_TYPE_FILES = MEMORY_TYPE_FILES
knowledge.MEMORY_TYPE_FILES = MEMORY_TYPE_FILES

if TYPE_CHECKING:
    import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("miku")

speech_IO.log = log
pluginHandling.log = log
knowledge.log = log
applications.log = log
chroma.log = log

def plan_tool_call(user_input):
    prompt = f"""
Available tools:
{available_tools_prompt()}

If a tool should be used, return STRICT JSON:

{{
  "tool": "tool.name",
  "args": {{}}
}}

If none needed, return:
null

User:
<user>{user_input}</user>
"""
    return generate_non_stream(prompt)

def execute_tool_plan(plan):
    try:
        obj = helpers._extract_json_object(plan)
        if not obj:
            return None
        tool = obj["tool"]
        args = obj.get("args", {})
        if tool in sharedConst.PLUGINS:
            return sharedConst.PLUGINS[tool].execute(args)
    except Exception as e:
        log.warning(f"Tool execution failed: {e}")
        log.debug(f"{plan}")
    return None

_state_lock = threading.Lock()

_user_state: dict = {"mood": "neutral", "energy": "normal", "last_update": None}
_last_user_time:  float = time.time()
_last_user_message: str = ""
_last_miku_spoke: float = 0.0
_tts_enabled_flag: bool = sharedConst.TTS_ENABLED
speech_IO._tts_enabled_flag = _tts_enabled_flag

_voice_enabled_flag: bool = sharedConst.VOICE_ENABLED_DEFAULT
speech_IO._voice_enabled_flag = _voice_enabled_flag
_wake_word_stop_event: threading.Event | None = None
_wake_word_thread: threading.Thread | None = None

_initiative_counts: dict[str, int] = {}

def _get_user_state() -> dict:
    with _state_lock:
        return dict(_user_state)

def _set_user_state(state: dict) -> None:
    with _state_lock:
        _user_state.update(state)
    _persist_state()

def _touch_user_time(msg: str = "") -> None:
    global _last_user_time, _last_user_message
    with _state_lock:
        _last_user_time = time.time()
        _last_user_message = msg

def _touch_miku_spoke() -> None:
    global _last_miku_spoke
    with _state_lock:
        _last_miku_spoke = time.time()

def _initiative_allowed() -> bool:
    today = date.today().isoformat()
    with _state_lock:
        count = _initiative_counts.get(today, 0)
        if count >= sharedConst.MAX_UNPROMPTED_PER_DAY:
            return False
        _initiative_counts[today] = count + 1
    return True

def _load_state() -> None:
    global _user_state, _last_miku_spoke
    if not STATE_FILE.exists():
        return
    try:
        obj = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        with _state_lock:
            _user_state.update(obj.get("user_state", {}))
            _last_miku_spoke = float(obj.get("last_miku_spoke", 0.0))
    except Exception as exc:
        log.warning("Could not load state.json: %s", exc)

def _persist_state() -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        with _state_lock:
            obj = {
                "user_state":      dict(_user_state),
                "last_miku_spoke": _last_miku_spoke,
            }
        STATE_FILE.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save state.json: %s", exc)

def load_identity() -> str:
    if IDENTITY_FILE.exists():
        return IDENTITY_FILE.read_text(encoding="utf-8")
    return "You are Hatsune Miku, a friendly and warm personal assistant."

_OPEN_PATTERNS = re.compile(
    r"\b(?:open|launch|start|run|can\s+you\s+open|please\s+open)\s+(.+)",
    re.IGNORECASE,
)

def detect_open_intent(text: str) -> str | None:
    cleaned = re.sub(rf"\b{re.escape(sharedConst.WAKE_WORD)}\b[,]?\s*", "", text, flags=re.IGNORECASE).strip()
    m = _OPEN_PATTERNS.search(cleaned)
    if m:
        app = m.group(1).strip().rstrip("?. ")
        return app
    return None


_STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from",
    "i","if","in","into","is","it","me","my","of","on","or",
    "our","so","that","the","their","then","they","this","to",
    "we","with","you","your",
}

def _query_terms(query: str) -> list[str]:
    parts = re.findall(r"[A-Za-z0-9_\-]+", query.lower())
    return [p for p in parts if len(p) >= 3 and p not in _STOPWORDS][:12]

def _find_snippet(content: str, terms: list[str], snippet_len: int = 500) -> str | None:
    lowered = content.lower()
    hits = [lowered.find(t) for t in terms if t and lowered.find(t) != -1]
    if not hits:
        return None
    idx = min(hits)
    start = max(0, idx - snippet_len // 2)
    end   = min(len(content), start + snippet_len)
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

    if sharedConst.SEARCH_BACKEND == "chroma" and chroma._chroma_available:
        hits = chroma._chroma_search(query, n_results=6)
        if hits:
            parts = [h["document"] for h in hits]
            combined = "\n\n".join(parts)
            base = "\n\n".join(results)
            return (base + "\n\n" + combined)[:max_chars]

    terms = _query_terms(query)
    if not terms:
        return "\n\n".join(results)[:max_chars]

    search_targets = list(MEMORY_TYPE_FILES.values()) + [MEMORY_DIR]
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


def forget_keyword(keyword: str) -> str:
    kw = keyword.strip().lower()
    if not kw:
        return "Please provide a keyword to forget."

    chroma_deleted = chroma._chroma_delete_by_keyword(kw)

    md_edits = 0
    for mem_type, filepath in MEMORY_TYPE_FILES.items():
        if not filepath.exists():
            continue
        try:
            content = filepath.read_text(encoding="utf-8")
            sections = re.split(r"(?=\n## )", content)
            kept = [s for s in sections if kw not in s.lower()]
            if len(kept) != len(sections):
                filepath.write_text("".join(kept), encoding="utf-8")
                md_edits += len(sections) - len(kept)
        except Exception as exc:
            log.warning("forget_keyword: could not edit %s: %s", filepath, exc)

    if chroma_deleted == 0 and md_edits == 0:
        return f"No memories found containing '{keyword}'."
    return (f"Forgot {md_edits} markdown block(s) and {chroma_deleted} "
            f"vector entry/entries related to '{keyword}'.")


def save_conversation(user: str, assistant: str) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    filename  = datetime.now().strftime("%Y-%m-%d") + ".md"
    file_path = MEMORY_DIR / filename
    entry = (
        f"\n## {datetime.now().strftime('%H:%M:%S')}\n\n"
        f"**User:** {user}\n\n"
        f"**Miku:** {assistant}\n\n---\n"
    )
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as exc:
        log.warning("save_conversation failed: %s", exc)




def _load_reminders() -> list[dict]:
    if not sharedConst.REMINDERS_FILE.exists():
        return []
    try:
        return json.loads(sharedConst.REMINDERS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load reminders: %s", exc)
        return []


def _save_reminders(reminders: list[dict]) -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        sharedConst.REMINDERS_FILE.write_text(json.dumps(reminders, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save reminders: %s", exc)


def _check_reminders(speech_queue: queue.Queue) -> None:
    reminders = _load_reminders()
    now = datetime.now()
    remaining: list[dict] = []
    fired = False
    for r in reminders:
        try:
            due = datetime.fromisoformat(r["due"])
        except Exception:
            remaining.append(r)
            continue
        if now >= due:
            msg = r.get("message", "You have a reminder!")
            print(f"\nMiku: (Reminder) {msg}")
            speech_queue.put(f"Reminder: {msg}")
            fired = True
        else:
            remaining.append(r)
    if fired:
        _save_reminders(remaining)


def add_reminder(message: str, due_iso: str) -> str:
    reminders = _load_reminders()
    reminders.append({"message": message, "due": due_iso})
    _save_reminders(reminders)
    return f"Reminder set: '{message}' at {due_iso}."


def infer_user_state(user_input: str) -> dict:
    url    = helpers._urljoin(sharedConst.OLLAMA_HOST, "/api/generate")
    prompt = (
        "Classify the user's emotional state.\n\n"
        'Return STRICT JSON:\n{"mood":"happy|low|stressed|focused|bored|neutral|excited|tired",'
        '"energy":"low|normal|high"}\n\n'
        f"User message:\n<user>{user_input}</user>"
    )
    payload = {
        "model": sharedConst.MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 80},
    }
    try:
        resp   = helpers._http_json("POST", url, payload)
        parsed = helpers._extract_json_object((resp or {}).get("response", ""))
        if parsed:
            return {
                "mood":        parsed.get("mood", "neutral"),
                "energy":      parsed.get("energy", "normal"),
                "last_update": datetime.now().isoformat(),
            }
    except Exception as exc:
        log.debug("infer_user_state failed: %s", exc)
    return _get_user_state()


def build_prompt(user_input: str) -> str:
    identity = load_identity()
    memory   = search_vault(user_input)
    return (
        f"{identity}\n\n"
        "You are present in a real ongoing relationship.\n\n"
        "Never:\n"
        "- mention retrieved memory\n"
        "- mention processing data or context windows\n"
        "- describe internal systems\n"
        "- speak like customer support\n"
        "- narrate that you are an AI assistant\n\n"
        "How to behave:\n"
        "- respond naturally\n"
        "- if casual chat, be casual\n"
        "- if emotional support is needed, be gentle and brief\n"
        "- if technical help is needed, become sharp and precise\n"
        "- reference memories naturally, like a person remembering\n\n"
        f"Relevant memory:\n{memory}\n\n"
        f"Available tools:\n{available_tools_prompt()}\n"
        f"User just said:\n<user>{user_input}</user>\n\n"
        "Respond as Miku."
    )


def build_initiative_prompt() -> str:
    context = search_vault("recent user context")
    state   = _get_user_state()
    return (
        f"{load_identity()}\n\n"
        "Decide whether speaking right now would genuinely help.\n\n"
        "Rules:\n"
        "- Prefer silence\n"
        "- You may initiate naturally:\n"
        "  Good: noticing long silence and checking in, reminding about important recurring goals,\n"
        "        sharing a thought related to past conversations, gentle companionship if user seemed low.\n"
        "  Bad:  forced small talk, generic 'how are you', robotic reminders.\n"
        "- Never force conversation\n\n"
        f"Context:\n{context}\n\n"
        f"Current inferred user state:\nMood: {state['mood']}\nEnergy: {state['energy']}\n\n"
        'Return strict JSON:\n{"speak": true/false, "priority": 1-10, "message": ""}'
    )


def generate_non_stream(prompt: str) -> str:
    url = helpers._urljoin(sharedConst.OLLAMA_HOST, "/api/generate")
    payload = {
        "model": sharedConst.MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.95,
                    "repeat_penalty": 1.1, "num_predict": 300},
    }
    try:
        resp = helpers._http_json("POST", url, payload=payload, timeout_s=sharedConst.OLLAMA_TIMEOUT_S)
        return (resp or {}).get("response", "") if isinstance(resp, dict) else ""
    except Exception as exc:
        log.warning("generate_non_stream failed: %s", exc)
        return ""


def stream_generate(prompt: str, echo: bool = True) -> str:
    url = helpers._urljoin(sharedConst.OLLAMA_HOST, "/api/generate")
    payload = {
        "model": sharedConst.MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.7, "top_p": 0.95,
                    "repeat_penalty": 1.1, "num_predict": 300},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    collected    = ""
    sentence_buf = ""

    try:
        with urllib.request.urlopen(req, timeout=sharedConst.OLLAMA_TIMEOUT_S) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                chunk = obj.get("response", "")
                if chunk:
                    collected    += chunk
                    sentence_buf += chunk
                    if echo:
                        print(chunk, end="", flush=True)
                    parts = speech_IO._SENTENCE_END.split(sentence_buf)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            speech_IO.speak_sentence(sentence)
                        sentence_buf = parts[-1]
                if obj.get("done"):
                    break
    except Exception as exc:
        log.warning("stream_generate failed: %s", exc)

    if echo:
        print()

    if sentence_buf.strip():
        speech_IO.speak_sentence(sentence_buf)

    return collected


def speech_loop(speech_queue: queue.Queue) -> None:
    while True:
        try:
            msg = speech_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        print(f"\nMiku: {msg}")
        speech_IO.speak(msg)


def initiative_loop(speech_queue: queue.Queue) -> None:
    while True:
        time.sleep(300)

        with _state_lock:
            since_user = time.time() - _last_user_time
            since_miku = time.time() - _last_miku_spoke

        if since_user < 900:
            continue
        if since_miku < 1800:
            continue
        if not _initiative_allowed():
            continue

        prompt = build_initiative_prompt()
        try:
            response = generate_non_stream(prompt)
            decision = json.loads(response)
            if decision.get("speak") and decision.get("priority", 0) >= 6:
                speech_queue.put(decision["message"])
                _touch_miku_spoke()
        except Exception as exc:
            log.debug("initiative_loop: %s", exc)


def reminder_loop(speech_queue: queue.Queue) -> None:
    while True:
        time.sleep(60)
        try:
            _check_reminders(speech_queue)
        except Exception as exc:
            log.debug("reminder_loop: %s", exc)


def _handle_tts_command() -> None:
    global _tts_enabled_flag
    _tts_enabled_flag = not _tts_enabled_flag
    state = "ON" if _tts_enabled_flag else "OFF"
    if not _tts_enabled_flag and speech_IO._tts_worker_instance:
        speech_IO._tts_worker_instance.drain()
    print(f"[TTS] {state}")


def _print_memory_stats() -> None:
    index        = knowledge._load_knowledge_index()
    ids          = index.get("ids", [])
    tag_counts   = index.get("tag_counts", {})
    title_counts = index.get("title_counts", {})

    print("\n── Memory Stats ──────────────────")
    print(f"  Total knowledge entries : {len(ids)}")
    print(f"  Chroma available        : {chroma._chroma_available}")
    print("  Memory type files:")
    for mem_type, filepath in MEMORY_TYPE_FILES.items():
        if filepath.exists():
            lines = filepath.read_text(encoding="utf-8").count("\n")
            print(f"    {mem_type:<14} → {filepath.name}  ({lines} lines)")
        else:
            print(f"    {mem_type:<14} → (empty)")
    if tag_counts:
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"  Top tags: {', '.join(f'{t}×{c}' for t, c in top_tags)}")
    if title_counts:
        reinforced = [(t, c) for t, c in title_counts.items() if c >= sharedConst.REINFORCE_THRESHOLD]
        if reinforced:
            print(f"  Reinforced topics: {', '.join(t for t, _ in reinforced)}")
    print("──────────────────────────────────\n")


def _print_help() -> None:
    print(
        "\n── Commands ───────────────────────\n"
        "  /exit or /quit          — exit\n"
        "  /tts                    — toggle speech on/off\n"
        "  /voice [on|off]         — toggle wake-word voice input\n"
        "  /memory                 — show memory stats\n"
        "  /rebuild                — rebuild Chroma vector index\n"
        "  /forget <word>          — delete memories matching keyword\n"
        "  /models                 — list available Ollama models\n"
        "  /evolve                 — show evolution / plugin status\n"
        "  /evolve pending         — list plugins awaiting approval\n"
        "  /evolve approve <name>  — approve and install a plugin\n"
        "  /evolve reject <name>   — reject and discard a plugin\n"
        "  /evolve show <name>     — view generated plugin code\n"
        "  /help                   — show this list\n"
        "───────────────────────────────────\n"
    )


def _handle_input(
    user_input: str,
    is_voice: bool = False,
    voice_input_queue: queue.Queue | None = None,
) -> bool:
    global _tts_enabled_flag

    stripped = user_input.strip()
    if not stripped:
        return True

    _touch_user_time(stripped)
    _set_user_state(infer_user_state(stripped))

    if stripped in ("/exit", "/quit"):
        return False

    if stripped == "/voice" or stripped.startswith("/voice "):
        arg = stripped[6:].strip().lower() if stripped != "/voice" else ""
        if arg in ("", "toggle"):
            speech_IO._set_voice_enabled(not _voice_enabled_flag, voice_input_queue=voice_input_queue)
        elif arg in ("on", "enable", "start"):
            speech_IO._set_voice_enabled(True, voice_input_queue=voice_input_queue)
        elif arg in ("off", "disable", "stop"):
            speech_IO._set_voice_enabled(False, voice_input_queue=voice_input_queue)
        else:
            print("[Voice] Usage: /voice [on|off]")
        return True

    if stripped == "/tts":
        _handle_tts_command()
        return True

    if stripped == "/memory":
        _print_memory_stats()
        return True

    if stripped == "/rebuild":
        chroma.rebuild_chroma_index()
        print("[Chroma] Re-index complete.")
        return True

    if stripped == "/help":
        _print_help()
        return True

    if stripped.startswith("/forget "):
        keyword = stripped[8:].strip()
        result  = forget_keyword(keyword)
        print(result)
        return True

    if stripped == "/models":
        try:
            models = helpers._http_json("GET", helpers._urljoin(sharedConst.OLLAMA_HOST, "/api/tags"))
            for m in (models or {}).get("models", []):
                print(m["name"])
        except Exception as exc:
            print(f"Error: {exc}")
        return True

    if stripped == "/evolve" or stripped.startswith("/evolve "):
        arg = stripped[7:].strip() if stripped != "/evolve" else ""

        if arg == "" or arg == "status":
            pluginHandling._print_evolution_status()

        elif arg == "pending":
            pending = pluginHandling.list_pending_plugins()
            if not pending:
                print("No plugins awaiting approval.")
            else:
                for p in pending:
                    print(f"  - {p.get('plugin_name')}  ({p.get('capability')})")

        elif arg.startswith("approve "):
            name   = arg[8:].strip()
            result = pluginHandling.approve_plugin(name)
            print(result)

        elif arg.startswith("reject "):
            name   = arg[7:].strip()
            result = pluginHandling.reject_plugin(name)
            print(result)

        elif arg.startswith("show "):
            name = arg[5:].strip()
            print(pluginHandling.show_plugin_code(name))

        else:
            print("[MES] Usage: /evolve [pending|approve <name>|reject <name>|show <name>]")

        return True

    app_name = detect_open_intent(stripped)
    if app_name:
        result = applications._launch_app(app_name)
        print(f"Miku: {result}")
        speech_IO.speak(result)
        save_conversation(stripped, result)
        _touch_miku_spoke()
        return True

    plan        = plan_tool_call(stripped)
    tool_result = execute_tool_plan(plan)

    if tool_result:
        print(f"Miku: {tool_result}")
        speech_IO.speak(tool_result)
        save_conversation(stripped, tool_result)
        return True

    prompt   = build_prompt(stripped)
    print("Miku: ", end="", flush=True)
    response = stream_generate(prompt, echo=True)
    _touch_miku_spoke()

    if "I can't" in response or "I don't have" in response or "I'm unable" in response:
        if sharedConst.EVOLUTION_ENABLED:
            threading.Thread(
                target=pluginHandling.record_capability_gap,
                args=(_infer_capability_name(stripped), stripped),
                daemon=True,
            ).start()

    threading.Thread(
        target=save_conversation, args=(stripped, response), daemon=True
    ).start()

    if sharedConst.AUTO_SAVE_KNOWLEDGE:
        def _bg_knowledge():
            proposed = knowledge.propose_knowledge(stripped, response)
            if not proposed:
                return
            confidence = float(proposed.get("confidence", 0.0))
            try:
                saved = knowledge.save_knowledge_entry(proposed)
            except Exception as exc:
                log.warning("save_knowledge_entry: %s", exc)
                saved = False
            if saved:
                mem_type = knowledge._resolve_memory_type(
                    proposed.get("tags", []), proposed.get("title", "")
                )
                print(f"(saved to knowledge/{mem_type} — confidence: {confidence:.2f})")
        threading.Thread(target=_bg_knowledge, daemon=True).start()

    return True


def _infer_capability_name(user_input: str) -> str:
    words = re.findall(r"[a-zA-Z]+", user_input.lower())
    stopwords = {"can", "you", "please", "miku", "the", "a", "an", "i", "my", "me"}
    keywords  = [w for w in words if w not in stopwords and len(w) > 3]
    return "_".join(keywords[:3]) if keywords else "unknown_capability"


def discover_plugins():
    import plugins
    for _, name, _ in pkgutil.iter_modules(plugins.__path__):
        module = importlib.import_module(f"plugins.{name}")
        if hasattr(module, "PLUGIN") and hasattr(module, "execute"):
            sharedConst.PLUGINS[module.PLUGIN["name"]] = module


def available_tools_prompt():
    if not sharedConst.PLUGINS:
        return "No tools available."
    lines = []
    for name, module in sharedConst.PLUGINS.items():
        desc = module.PLUGIN["description"]
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def run() -> None:
    _load_state()

    if sharedConst.SEARCH_BACKEND == "chroma":
        ok = chroma._init_chroma()
        chroma_status = "ON" if ok else "OFF (chromadb not installed — using keyword fallback)"
    else:
        chroma_status = "OFF (SEARCH_BACKEND=keyword)"

    if _voice_enabled_flag:
        stt_status = "ON" if speech_IO._stt_init_if_needed() else "OFF (STT init failed)"
    else:
        stt_status = "OFF (disabled; use /voice to enable)"

    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    EVOLUTION_PENDING_DIR.mkdir(parents=True, exist_ok=True)
    EVOLUTION_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    discover_plugins()

    print(f"Connected to Ollama @ {sharedConst.OLLAMA_HOST}")
    print(f"Model             : {sharedConst.MODEL}")
    print(f"Vector search     : {chroma_status}")
    print(f"Voice / wake word : {stt_status}  (wake word: '{sharedConst.WAKE_WORD}')")
    tts_state = "ON" if sharedConst.TTS_ENABLED else "OFF"
    print(f"TTS               : {tts_state}")
    if sharedConst.AUTO_SAVE_KNOWLEDGE:
        print(f"Auto-knowledge    : ON  (model={sharedConst.KNOWLEDGE_MODEL}, "
              f"confidence≥{sharedConst.KNOWLEDGE_CONFIDENCE_THRESHOLD})")
    else:
        print("Auto-knowledge    : OFF")
    if sharedConst.EVOLUTION_ENABLED:
        metrics = pluginHandling._load_evolution_metrics()
        pending = pluginHandling.list_pending_plugins()
        print(f"Evolution (MES)   : ON  (coder={sharedConst.EVOLUTION_CODER_MODEL}, "
              f"threshold={sharedConst.EVOLUTION_THRESHOLD}, "
              f"gained={metrics.get('capabilities_gained', 0)})")
        if pending:
            print(f"  ⚠  {len(pending)} plugin(s) awaiting approval — use /evolve pending")
    else:
        print("Evolution (MES)   : OFF")
    print("Type /help for commands.  Use /voice to enable wake-word voice input.\n")

    _speech_queue: queue.Queue = queue.Queue()
    _voice_input_queue: queue.Queue = queue.Queue()

    threading.Thread(target=speech_loop,     args=(_speech_queue,), daemon=True).start()
    threading.Thread(target=initiative_loop,  args=(_speech_queue,), daemon=True).start()
    threading.Thread(target=reminder_loop,   args=(_speech_queue,), daemon=True).start()
    if _voice_enabled_flag:
        speech_IO._set_voice_enabled(True, voice_input_queue=_voice_input_queue)

    while True:
        try:
            voice_text = _voice_input_queue.get_nowait()
            if not _handle_input(voice_text, is_voice=True, voice_input_queue=_voice_input_queue):
                break
            continue
        except queue.Empty:
            pass

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not _handle_input(user_input, voice_input_queue=_voice_input_queue):
            break

    print("Goodbye!")
    _persist_state()


if __name__ == "__main__":
    run()