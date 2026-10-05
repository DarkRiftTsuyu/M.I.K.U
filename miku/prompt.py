from __future__ import annotations

import json
import logging
import re
import threading

from .config import (
    LAST_N_MESSAGES, MEMORY_MAX_CHARS, IDENTITY_FILE, LIFE_STORY_FILE,
    AUTO_SAVE_KNOWLEDGE, EVOLUTION_ENABLED, TOOL_FAILURE_TTL_S,
)
from .memory import (
    load_relationship, load_goals, load_active_task, save_active_task,
    get_conversation_summary, update_relationship,
    build_conversation_state_block, build_internal_notes_block,
    build_goals_block, build_task_block, save_knowledge_entry,
)
from .tools import (
    PLUGINS, available_tools_prompt, capability_constraints_for_input,
    build_capability_block, pop_tool_failure_notice,
)
from .search import search_vault
from .state import get_user_state, set_user_state
from .llm import generate_non_stream, infer_user_state, propose_knowledge
from .utils import extract_json_object, now_iso

log = logging.getLogger("miku.prompt")


def load_identity() -> str:
    if IDENTITY_FILE.exists():
        return IDENTITY_FILE.read_text(encoding="utf-8")
    return "You are Hatsune Miku, a friendly and warm personal assistant."


def build_prompt(user_input: str, recent_turns: list[dict]) -> str:
    identity     = load_identity()
    recent_conv  = _format_turns(recent_turns, LAST_N_MESSAGES)
    tool_notice  = pop_tool_failure_notice(TOOL_FAILURE_TTL_S)
    conv_state   = build_conversation_state_block()
    notes_block  = build_internal_notes_block()
    cap_guard    = capability_constraints_for_input(user_input)
    memory       = search_vault(user_input, max_chars=MEMORY_MAX_CHARS)
    rel_ctx      = _get_relationship_context()
    cap_block    = build_capability_block()
    task_block   = build_task_block()
    goals_block  = build_goals_block()
    conv_summary = get_conversation_summary()
    life_story   = ""
    if LIFE_STORY_FILE.exists():
        try:
            life_story = LIFE_STORY_FILE.read_text(encoding="utf-8")[:600]
        except Exception:
            pass
    sections = [
        identity,
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
        "- reference memories naturally, like a person remembering\n"
        "- if you cannot do something, say so clearly and offer an alternative\n"
        "- acknowledge active tasks and continue them naturally",
    ]
    if rel_ctx:
        sections.append(f"Relationship:\n{rel_ctx}")
    if life_story:
        sections.append(f"Shared history:\n{life_story}")
    if tool_notice:
        sections.append(f"System note:\n{tool_notice}")
    if recent_conv:
        sections.append(f"Recent conversation (last {LAST_N_MESSAGES} messages):\n{recent_conv}")
    if conv_state:
        sections.append(conv_state)
    if notes_block:
        sections.append(notes_block)
    if conv_summary:
        sections.append(f"Earlier in this conversation:\n{conv_summary}")
    if task_block:
        sections.append(task_block)
    if goals_block:
        sections.append(goals_block)
    if cap_guard:
        sections.append(cap_guard)
    sections.append(f"Relevant memory:\n{memory}")
    sections.append(f"{cap_block}")
    sections.append(f"Available tools:\n{available_tools_prompt()}")
    sections.append(f"User just said:\n<user>{user_input}</user>\n\nRespond as Miku.")
    return "\n\n".join(s for s in sections if s.strip())


def _format_turns(turns: list[dict], max_pairs: int) -> str:
    if max_pairs <= 0 or not turns:
        return ""
    items = turns[-max_pairs * 2:]
    lines: list[str] = []
    for it in items:
        role = "User" if it.get("role") == "user" else "Miku"
        text = str(it.get("content", "")).strip().replace("\n", " ")
        if len(text) > 260:
            text = text[:257] + "…"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _get_relationship_context() -> str:
    rel    = load_relationship()
    count  = rel.get("interaction_count", 0)
    first  = rel.get("first_seen", "recently")
    topics = rel.get("favorite_topics", [])[:5]
    last   = rel.get("last_seen", "today")
    lines  = [f"Relationship context: {count} interactions since {first}, last active {last}."]
    if topics:
        lines.append(f"Recurring interests: {', '.join(topics)}.")
    return "\n".join(lines)


def infer_and_set_user_state(user_input: str) -> None:
    state = infer_user_state(user_input)
    set_user_state(state)


def detect_and_update_task(user_input: str) -> None:
    task = load_active_task()
    if task:
        return
    lower = user_input.lower()
    task_keywords = ["help me", "i need to", "can you", "please", "i want to", "let's", "lets"]
    if not any(k in lower for k in task_keywords):
        return
    words = re.findall(r"[a-zA-Z]{3,}", lower)
    stop  = {"help","need","want","please","just","that","with","you","me","can","lets","let"}
    keywords = [w for w in words if w not in stop][:5]
    if not keywords:
        return
    goal = " ".join(keywords)
    save_active_task({
        "type": "conversational",
        "goal": goal,
        "status": "active",
        "last_update": now_iso(),
    })


def update_active_task_from_response(user_input: str, response: str) -> None:
    task = load_active_task()
    if not task:
        return
    prompt = (
        "A task is in progress. Based on the conversation, update the task status.\n\n"
        f"Task: {json.dumps(task)}\n"
        f"User: {user_input[:300]}\n"
        f"Assistant: {response[:300]}\n\n"
        "Return STRICT JSON:\n"
        '{"status": "active|completed|blocked", "progress_note": ""}\n'
        "If no update needed, return: null"
    )
    try:
        text = generate_non_stream(prompt)
        if not text.strip() or text.strip().lower() == "null":
            return
        obj = extract_json_object(text)
        if not obj:
            return
        status = str(obj.get("status", "")).strip().lower()
        if status in ("active", "completed", "blocked"):
            task["status"] = status
            task["last_update"] = now_iso()
            save_active_task(task if status != "completed" else None)
    except Exception as exc:
        log.debug("update_active_task_from_response: %s", exc)


def auto_update_goal_progress(user_input: str, response: str) -> None:
    goals = load_goals()
    active = [g for g in goals if g.get("status") == "active"]
    if not active:
        return
    from .memory import save_goals
    for g in active:
        prompt = (
            "Assess whether the following conversation suggests progress on a goal.\n\n"
            f"Goal: {g['goal']}\n"
            f"User: {user_input[:200]}\n"
            f"Assistant: {response[:200]}\n\n"
            "Return STRICT JSON:\n"
            '{"progress_delta": 0.0}\n'
            "Use a value 0.0 to 0.1 per turn. Return 0.0 if no progress."
        )
        try:
            text = generate_non_stream(prompt)
            obj  = extract_json_object(text)
            if obj:
                delta = float(obj.get("progress_delta", 0.0))
                if delta > 0:
                    g["progress"] = min(1.0, g.get("progress", 0.0) + delta)
                    g["last_updated"] = now_iso()
        except Exception:
            pass
    save_goals(goals)


def infer_capability_name(user_input: str) -> str:
    words     = re.findall(r"[a-zA-Z]+", user_input.lower())
    stopwords = {"can", "you", "please", "miku", "the", "a", "an", "i", "my", "me"}
    keywords  = [w for w in words if w not in stopwords and len(w) > 3]
    return "_".join(keywords[:3]) if keywords else "unknown_capability"


def run_bg_knowledge(user_input: str, response: str) -> None:
    if not AUTO_SAVE_KNOWLEDGE:
        return
    def _bg():
        from .memory import resolve_memory_type
        proposed = propose_knowledge(user_input, response)
        if not proposed:
            return
        confidence = float(proposed.get("confidence", 0.0))
        try:
            saved = save_knowledge_entry(proposed)
        except Exception as exc:
            log.warning("save_knowledge_entry: %s", exc)
            saved = False
        if saved:
            mem_type = resolve_memory_type(
                proposed.get("tags", []), proposed.get("title", "")
            )
            print(f"(saved to knowledge/{mem_type} — confidence: {confidence:.2f})")
    threading.Thread(target=_bg, daemon=True).start()
