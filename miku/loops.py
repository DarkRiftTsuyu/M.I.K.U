from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import date, datetime

from .config import (
    REFLECTION_HOUR, LAST_N_MESSAGES, DESKTOP_NOTIFY_REMINDERS,
    AUTO_SAVE_KNOWLEDGE,
)
from .memory import (
    load_reminders, save_reminders,
    write_journal_entry, run_reflection, update_life_story,
    load_relationship, load_goals,
)
from .llm import generate_non_stream
from .tts import speak
from .state import (
    get_last_user_time, get_last_miku_spoke, touch_miku_spoke,
    initiative_allowed, get_user_state,
)

log = logging.getLogger("miku.loops")


def speech_loop(speech_queue: queue.Queue) -> None:
    while True:
        try:
            msg = speech_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        print(f"\nMiku: {msg}")
        speak(msg)


def initiative_loop(speech_queue: queue.Queue) -> None:
    while True:
        time.sleep(300)
        since_user = time.time() - get_last_user_time()
        since_miku = time.time() - get_last_miku_spoke()
        if since_user < 900:
            continue
        if since_miku < 1800:
            continue
        if not initiative_allowed():
            continue
        prompt = _build_initiative_prompt()
        try:
            response = generate_non_stream(prompt)
            decision = json.loads(response)
            if decision.get("speak") and decision.get("priority", 0) >= 6:
                speech_queue.put(decision["message"])
                touch_miku_spoke()
        except Exception as exc:
            log.debug("initiative_loop: %s", exc)


def _build_initiative_prompt() -> str:
    from .search import search_vault
    from .memory import maybe_generate_curiosity_question
    context      = search_vault("recent user context")
    state        = get_user_state()
    rel          = load_relationship()
    topics       = rel.get("favorite_topics", [])[:5]
    curiosity_q  = maybe_generate_curiosity_question(generate_non_stream)
    curiosity_hint = f"\nCuriosity opportunity: consider asking — {curiosity_q}" if curiosity_q else ""
    from .config import IDENTITY_FILE
    identity = IDENTITY_FILE.read_text(encoding="utf-8") if IDENTITY_FILE.exists() else "You are Miku."
    return (
        f"{identity}\n\n"
        "Decide whether speaking right now would genuinely help.\n\n"
        "Rules:\n"
        "- Prefer silence\n"
        "- You may initiate naturally:\n"
        "  Good: noticing long silence and checking in, reminding about important recurring goals,\n"
        "        sharing a thought related to past conversations, gentle companionship if user seemed low,\n"
        "        asking a curiosity question from memory if enough time has passed.\n"
        "  Bad:  forced small talk, generic 'how are you', robotic reminders.\n"
        "- Never force conversation\n\n"
        f"Context:\n{context}\n\n"
        f"User's recurring interests: {', '.join(topics)}\n"
        f"Current inferred user state:\nMood: {state['mood']}\nEnergy: {state['energy']}\n"
        f"{curiosity_hint}\n\n"
        'Return strict JSON:\n{"speak": true/false, "priority": 1-10, "message": ""}'
    )


def reminder_loop(speech_queue: queue.Queue) -> None:
    while True:
        time.sleep(60)
        try:
            _check_reminders(speech_queue)
        except Exception as exc:
            log.debug("reminder_loop: %s", exc)


def _check_reminders(speech_queue: queue.Queue) -> None:
    from .tools import send_desktop_notification
    reminders = load_reminders()
    now       = datetime.now()
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
            if DESKTOP_NOTIFY_REMINDERS:
                send_desktop_notification(title="Miku Reminder", message=msg)
            fired = True
        else:
            remaining.append(r)
    if fired:
        save_reminders(remaining)


def reflection_loop() -> None:
    last_reflection_date: str = ""
    while True:
        time.sleep(600)
        today    = date.today().isoformat()
        now_hour = datetime.now().hour
        if today != last_reflection_date and now_hour >= REFLECTION_HOUR:
            last_reflection_date = today
            try:
                write_journal_entry(generate_non_stream)
                run_reflection(generate_non_stream)
                update_life_story(generate_non_stream)
            except Exception as exc:
                log.debug("reflection_loop: %s", exc)


def queue_continuity_updates(user_msg: str, assistant_msg: str) -> None:
    from .memory import (
        load_conversation_state, save_conversation_state,
        append_internal_note, load_internal_notes,
    )
    from .llm import generate_non_stream as gen
    from .utils import extract_json_object

    def _update_state():
        prev   = load_conversation_state()
        recent = _format_recent_for_update()
        prompt = (
            "You maintain a running conversation state for a long-lived assistant persona.\n\n"
            "Update the JSON state based on the latest interaction.\n\n"
            "Rules:\n"
            "- Keep it short and stable across turns\n"
            "- current_topic: 2-6 words\n"
            "- open_threads: 0-6 short items\n"
            "- last_goal: short phrase; may be empty\n"
            "- Do not include anything sensitive\n\n"
            f"Previous state JSON:\n{json.dumps(prev, indent=2)[:1200]}\n\n"
            f"Recent conversation (most recent last):\n{recent}\n\n"
            f"Latest user message: {user_msg[:400]}\n"
            f"Latest assistant message: {assistant_msg[:400]}\n\n"
            "Return STRICT JSON with keys: current_topic, open_threads, last_goal"
        )
        try:
            text = gen(prompt)
            obj  = extract_json_object(text)
            if not obj:
                return
            state = {
                "current_topic": str(obj.get("current_topic", "")).strip()[:80],
                "open_threads": [str(t).strip()[:80] for t in (obj.get("open_threads") or []) if str(t).strip()][:6],
                "last_goal": str(obj.get("last_goal", "")).strip()[:120],
            }
            save_conversation_state(state)
        except Exception as exc:
            log.debug("conversation_state update failed: %s", exc)

    def _bg_note():
        prompt = (
            "You write Miku's internal notes (private) to maintain continuity between sessions.\n\n"
            "From the latest exchange, decide if there is a worthwhile follow-up.\n\n"
            "Return STRICT JSON object:\n"
            '{"topic":"", "interest_level":"low|medium|high", "follow_up":""}\n'
            "If nothing to track, return: null\n\n"
            f"User: {user_msg[:500]}\n"
            f"Miku: {assistant_msg[:500]}"
        )
        try:
            text = gen(prompt)
            if not text.strip() or text.strip().lower() in ("null", "none"):
                return
            obj = extract_json_object(text)
            if not obj:
                return
            topic  = str(obj.get("topic", "")).strip()
            follow = str(obj.get("follow_up", "")).strip()
            if not topic and not follow:
                return
            append_internal_note({
                "topic": topic,
                "interest_level": str(obj.get("interest_level", "medium")),
                "follow_up": follow,
            })
        except Exception:
            pass

    threading.Thread(target=_update_state, daemon=True).start()
    threading.Thread(target=_bg_note,      daemon=True).start()


def _format_recent_for_update(n: int = 12) -> str:
    from .config import LAST_N_MESSAGES, RECENT_TURNS_FILE
    import json as _json
    turns: list[dict] = []
    try:
        if RECENT_TURNS_FILE.exists():
            lines = RECENT_TURNS_FILE.read_text(encoding="utf-8").splitlines()
            for line in lines[-n * 2:]:
                try:
                    obj = _json.loads(line)
                    if isinstance(obj, dict) and obj.get("role") in ("user", "assistant"):
                        turns.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    items = turns[-n * 2:]
    return "\n".join(
        f"{'User' if it['role'] == 'user' else 'Miku'}: {str(it.get('content', ''))[:200]}"
        for it in items
    )
