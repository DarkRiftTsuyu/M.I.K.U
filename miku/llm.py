from __future__ import annotations

import json
import logging
import urllib.request

from .config import (
    FAST_MODEL, MODEL, OLLAMA_HOST, OLLAMA_TIMEOUT_S,
    KNOWLEDGE_MODEL, KNOWLEDGE_CONFIDENCE_THRESHOLD,
)
from .utils import http_json, urljoin, extract_json_object

log = logging.getLogger("miku.llm")

_SENTENCE_END = __import__("re").compile(r"(?<=[.!?])\s+")


def generate_non_stream(prompt: str) -> str:
    url = urljoin(OLLAMA_HOST, "/api/generate")
    payload = {
        "model": FAST_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 160,
        },
    }
    try:
        resp = http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S)
        return (resp or {}).get("response", "") if isinstance(resp, dict) else ""
    except Exception as exc:
        log.warning("generate_non_stream failed: %s", exc)
        return ""


def generate_non_stream_custom(
    prompt: str, *, temperature: float = 0.5, num_predict: int = 220
) -> str:
    url = urljoin(OLLAMA_HOST, "/api/generate")
    payload = {
        "model": FAST_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "num_predict": int(num_predict),
        },
    }
    try:
        resp = http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S)
        return (resp or {}).get("response", "") if isinstance(resp, dict) else ""
    except Exception as exc:
        log.warning("generate_non_stream_custom failed: %s", exc)
        return ""


def stream_generate(prompt: str, echo: bool = True, speak_sentence_fn=None) -> str:
    url = urljoin(OLLAMA_HOST, "/api/generate")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "num_predict": 160,
        },
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
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
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
                    parts = _SENTENCE_END.split(sentence_buf)
                    if len(parts) > 1:
                        if speak_sentence_fn:
                            for sentence in parts[:-1]:
                                speak_sentence_fn(sentence)
                        sentence_buf = parts[-1]
                if obj.get("done"):
                    break
    except Exception as exc:
        log.warning("stream_generate failed: %s", exc)
    if echo:
        print()
    if sentence_buf.strip() and speak_sentence_fn:
        speak_sentence_fn(sentence_buf)
    return collected


def propose_knowledge(user_input: str, assistant_response: str) -> dict | None:
    import re
    _TRIVIAL = re.compile(r"^[\w\s]{1,12}$")
    if len(user_input.strip()) < 20 or _TRIVIAL.match(user_input.strip()):
        return None
    url = urljoin(OLLAMA_HOST, "/api/generate")
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
        "model": KNOWLEDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 350},
    }
    try:
        resp = http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S)
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


def infer_user_state(user_input: str) -> dict:
    url = urljoin(OLLAMA_HOST, "/api/generate")
    prompt = (
        "Classify the user's emotional state.\n\n"
        'Return STRICT JSON:\n{"mood":"happy|low|stressed|focused|bored|neutral|excited|tired",'
        '"energy":"low|normal|high"}\n\n'
        f"User message:\n<user>{user_input}</user>"
    )
    payload = {
        "model": FAST_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 80},
    }
    try:
        resp   = http_json("POST", url, payload)
        parsed = extract_json_object((resp or {}).get("response", ""))
        if parsed:
            return {
                "mood":        parsed.get("mood", "neutral"),
                "energy":      parsed.get("energy", "normal"),
                "last_update": __import__("datetime").datetime.now().isoformat(),
            }
    except Exception as exc:
        log.debug("infer_user_state failed: %s", exc)
    return {"mood": "neutral", "energy": "normal", "last_update": None}
