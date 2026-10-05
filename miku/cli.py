from __future__ import annotations

import json
import logging
import queue
import threading
import time

from .config import (
    OLLAMA_HOST, MODEL, TTS_ENABLED, AUTO_SAVE_KNOWLEDGE,
    KNOWLEDGE_MODEL, KNOWLEDGE_CONFIDENCE_THRESHOLD, SEARCH_BACKEND,
    WAKE_WORD, EVOLUTION_ENABLED, EVOLUTION_CODER_MODEL, EVOLUTION_THRESHOLD,
    JOURNAL_ENABLED, REFLECTION_HOUR, LAST_N_MESSAGES, EVOLUTION_DIR,
    EVOLUTION_PENDING_DIR, EVOLUTION_SANDBOX_DIR, JOURNAL_DIR,
    VOICE_ENABLED_DEFAULT, DESKTOP_NOTIFY_REMINDERS, IDENTITY_FILE,
    MEMORY_TYPE_FILES, KNOWLEDGE_DIR,
)
from .memory import (
    load_relationship, load_goals, load_active_task, save_active_task,
    add_goal, add_reminder, read_journal, run_reflection,
    load_knowledge_graph, load_knowledge_index, load_memory_scores,
    append_recent_turn, format_recent_conversation,
    load_recent_turns_from_disk, save_conversation,
    load_tool_context, save_tool_context,
)
from .search import init_chroma, rebuild_chroma_index, is_chroma_available, chroma_delete_by_keyword, search_vault
from .tools import (
    PLUGINS, discover_plugins, load_pending_confirmation, clear_pending_confirmation,
    save_pending_confirmation, detect_open_intent, launch_app, should_use_tools,
    plan_tool_call, execute_tool_plan, maybe_clear_tool_context,
    detect_missing_capability, send_desktop_notification,
    available_tools_prompt,
)
from .tts import speak, toggle_tts, is_tts_enabled
from .audio import set_voice_enabled, is_voice_enabled, stt_init_if_needed, get_silero_vad_status
from .state import (
    load_state, persist_state, touch_user_time, touch_miku_spoke,
    set_user_state, get_last_user_time, get_last_miku_spoke,
)
from .llm import stream_generate, generate_non_stream, infer_user_state
from .evolution import (
    load_evolution_metrics, list_pending_plugins, approve_plugin,
    reject_plugin, show_plugin_code, print_evolution_status,
    record_capability_gap,
)
from .memory import forget_keyword
from .search import chroma_delete_by_keyword
from .prompt import (
    build_prompt, infer_and_set_user_state, detect_and_update_task,
    update_active_task_from_response, auto_update_goal_progress,
    infer_capability_name, run_bg_knowledge, load_identity,
)
from .loops import (
    speech_loop, initiative_loop, reminder_loop, reflection_loop,
    queue_continuity_updates,
)
from .utils import urljoin, http_json

log = logging.getLogger("miku.cli")

_conversation_turn_count: int = 0
_conversation_turns_for_summary: list[dict] = []
_recent_turns: list[dict] = []


def _record_for_summary(user_msg: str, assistant_msg: str) -> None:
    global _conversation_turn_count, _conversation_turns_for_summary
    from .config import CONTEXT_COMPRESSION_INTERVAL
    from .memory import compress_conversation
    _conversation_turn_count += 1
    _conversation_turns_for_summary.append({"user": user_msg, "miku": assistant_msg[:500]})
    if _conversation_turn_count % CONTEXT_COMPRESSION_INTERVAL == 0:
        turns_copy = _conversation_turns_for_summary[:]
        threading.Thread(
            target=compress_conversation,
            args=(turns_copy, generate_non_stream),
            daemon=True,
        ).start()


def _append_turn(role: str, content: str) -> None:
    global _recent_turns
    _recent_turns = append_recent_turn(_recent_turns, role, content)


def _format_recent(max_pairs: int | None = None) -> str:
    n = max_pairs if max_pairs is not None else LAST_N_MESSAGES
    return format_recent_conversation(_recent_turns, n)


def _handle_input(
    user_input: str,
    voice_input_queue: queue.Queue | None = None,
) -> bool:
    global _recent_turns

    stripped = user_input.strip()
    if not stripped:
        return True

    touch_user_time(stripped)
    infer_and_set_user_state(stripped)
    from .memory import update_relationship
    update_relationship(stripped)

    if stripped in ("/exit", "/quit"):
        return False

    if stripped == "/voice" or stripped.startswith("/voice "):
        arg = stripped[6:].strip().lower() if stripped != "/voice" else ""
        if arg in ("", "toggle"):
            set_voice_enabled(not is_voice_enabled(), voice_input_queue=voice_input_queue)
        elif arg in ("on", "enable", "start"):
            set_voice_enabled(True, voice_input_queue=voice_input_queue)
        elif arg in ("off", "disable", "stop"):
            set_voice_enabled(False, voice_input_queue=voice_input_queue)
        else:
            print("[Voice] Usage: /voice [on|off]")
        return True

    if stripped.lower() in ("/confirm", "/yes", "/approve"):
        pending = load_pending_confirmation()
        if not pending:
            print("[Confirm] Nothing pending.")
            return True
        tool = pending.get("tool")
        args = pending.get("args", {})
        if tool not in PLUGINS:
            print(f"[Confirm] Tool no longer loaded: {tool}")
            clear_pending_confirmation()
            return True
        try:
            result = PLUGINS[tool].execute(args)
            clear_pending_confirmation()
            print(f"[Confirm] Executed.\n{result}")
        except Exception as exc:
            print(f"[Confirm] Failed: {exc}")
        return True

    if stripped.lower() in ("/cancel", "/deny"):
        if load_pending_confirmation():
            clear_pending_confirmation()
            print("[Confirm] Pending action cancelled.")
        else:
            print("[Confirm] Nothing pending.")
        return True

    if stripped == "/tts":
        state = toggle_tts()
        print(f"[TTS] {state}")
        return True

    if stripped == "/memory":
        _print_memory_stats()
        return True

    if stripped == "/relationship":
        _print_relationship_stats()
        return True

    if stripped == "/graph":
        _print_graph_stats()
        return True

    if stripped == "/goals":
        goals = load_goals()
        if not goals:
            print("No goals tracked yet. Use /goal <text> to add one.")
        else:
            print("\n── Goals ─────────────────────────")
            for g in goals:
                status = "✓" if g.get("status") == "complete" else "·"
                pct    = int(g.get("progress", 0) * 100)
                print(f"  {status} [{pct:3d}%] {g['goal']}")
            print("──────────────────────────────────\n")
        return True

    if stripped.startswith("/goal "):
        goal_text = stripped[6:].strip()
        if goal_text:
            print(add_goal(goal_text))
        else:
            print("Usage: /goal <goal description>")
        return True

    if stripped == "/task":
        task = load_active_task()
        if task:
            print(f"\nActive task: {task.get('goal')}\nStatus: {task.get('status')}")
            reqs = task.get("requirements", [])
            if reqs:
                print("Requirements: " + ", ".join(str(r) for r in reqs))
        else:
            print("No active task.")
        return True

    if stripped == "/task clear":
        save_active_task(None)
        print("Active task cleared.")
        return True

    if stripped == "/rebuild":
        rebuild_chroma_index()
        print("[Chroma] Re-index complete.")
        return True

    if stripped == "/reflect":
        print("Running reflection…")
        from .memory import run_reflection
        run_reflection(generate_non_stream)
        print("Done. Use /story or check vault/reflection.md.")
        return True

    if stripped == "/story":
        from .config import LIFE_STORY_FILE
        if LIFE_STORY_FILE.exists():
            print(LIFE_STORY_FILE.read_text(encoding="utf-8"))
        else:
            print("No life story written yet. It builds over time.")
        return True

    if stripped.startswith("/journal"):
        parts = stripped.split()
        n = 3
        if len(parts) > 1 and parts[1].isdigit():
            n = int(parts[1])
        journal = read_journal(days_back=max(1, min(n, 30)))
        print(journal if journal else "No journal entries found.")
        return True

    if stripped == "/help":
        _print_help()
        return True

    if stripped.startswith("/forget "):
        keyword = stripped[8:].strip()
        print(forget_keyword(keyword, chroma_delete_fn=chroma_delete_by_keyword))
        return True

    if stripped == "/habits" or stripped.startswith("/habits "):
        _handle_habits_command(stripped)
        return True

    if stripped.startswith("/habit add "):
        if "habit_tracker" in PLUGINS:
            name   = stripped[11:].strip()
            result = PLUGINS["habit_tracker"].execute({"action": "add_habit", "name": name})
        else:
            result = "Habit tracker plugin not loaded."
        print(result)
        return True

    if stripped.startswith("/habit log "):
        if "habit_tracker" in PLUGINS:
            name   = stripped[11:].strip()
            result = PLUGINS["habit_tracker"].execute({"action": "log_completion", "name": name})
        else:
            result = "Habit tracker plugin not loaded."
        print(result)
        return True

    if stripped == "/gmail" or stripped.startswith("/gmail "):
        _handle_gmail_command(stripped)
        return True

    if stripped == "/calendar" or stripped.startswith("/calendar "):
        _handle_calendar_command(stripped)
        return True

    if stripped == "/discord" or stripped.startswith("/discord "):
        _handle_discord_command(stripped)
        return True

    if stripped.startswith("/notify "):
        msg = stripped[8:].strip()
        if not msg:
            print("[Notify] Usage: /notify <message>")
            return True
        if "desktop_notify" in PLUGINS:
            try:
                result = PLUGINS["desktop_notify"].execute({"title": "Miku", "message": msg, "timeout": 5})
                print(f"[Notify] {result}")
            except Exception as exc:
                print(f"[Notify] Failed: {exc}")
        else:
            print("[Notify] desktop_notify plugin is not loaded.")
        return True

    if stripped == "/integrations":
        _print_integrations_status()
        return True

    if stripped == "/models":
        try:
            models = http_json("GET", urljoin(OLLAMA_HOST, "/api/tags"))
            for m in (models or {}).get("models", []):
                print(m["name"])
        except Exception as exc:
            print(f"Error: {exc}")
        return True

    if stripped == "/evolve" or stripped.startswith("/evolve "):
        _handle_evolve_command(stripped)
        return True

    _append_turn("user", stripped)

    app_name = detect_open_intent(stripped)
    if app_name:
        result = launch_app(app_name)
        print(f"Miku: {result}")
        speak(result)
        _append_turn("assistant", result)
        threading.Thread(target=save_conversation, args=(stripped, result), daemon=True).start()
        _record_for_summary(stripped, result)
        queue_continuity_updates(stripped, result)
        touch_miku_spoke()
        return True

    tool_context_closed = maybe_clear_tool_context(
        stripped,
        load_tool_context_fn=load_tool_context,
        save_tool_context_fn=save_tool_context,
        save_active_task_fn=save_active_task,
    )

    if not tool_context_closed and should_use_tools(stripped):
        plan        = plan_tool_call(stripped, load_tool_context, lambda: _format_recent(8))
        tool_result = execute_tool_plan(
            plan, stripped,
            load_tool_context_fn=load_tool_context,
            save_tool_context_fn=save_tool_context,
            save_active_task_fn=save_active_task,
            format_recent_fn=lambda: _format_recent(6),
        )
        if tool_result:
            print(f"Miku: {tool_result}")
            speak(tool_result)
            _append_turn("assistant", tool_result)
            threading.Thread(target=save_conversation, args=(stripped, tool_result), daemon=True).start()
            _record_for_summary(stripped, tool_result)
            queue_continuity_updates(stripped, tool_result)
            touch_miku_spoke()
            return True

    missing_cap = detect_missing_capability(stripped)
    if missing_cap:
        if EVOLUTION_ENABLED:
            threading.Thread(
                target=record_capability_gap,
                args=(missing_cap, stripped),
                daemon=True,
            ).start()
        pretty = missing_cap.replace("_", " ")
        result = (
            f"I can't currently do that because I don't have the {pretty} capability yet. "
            "I can record this as a plugin request if you'd like."
        )
        print(f"Miku: {result}")
        speak(result)
        _append_turn("assistant", result)
        threading.Thread(target=save_conversation, args=(stripped, result), daemon=True).start()
        _record_for_summary(stripped, result)
        queue_continuity_updates(stripped, result)
        touch_miku_spoke()
        return True

    detect_and_update_task(stripped)

    prompt   = build_prompt(stripped, _recent_turns)
    print("Miku: ", end="", flush=True)
    response = stream_generate(prompt, echo=True, speak_sentence_fn=__import__("miku.tts", fromlist=["speak_sentence"]).speak_sentence)
    touch_miku_spoke()

    _append_turn("assistant", response)
    _record_for_summary(stripped, response)

    if "I can't" in response or "I don't have" in response or "I'm unable" in response:
        if EVOLUTION_ENABLED:
            threading.Thread(
                target=record_capability_gap,
                args=(infer_capability_name(stripped), stripped),
                daemon=True,
            ).start()

    threading.Thread(target=save_conversation, args=(stripped, response), daemon=True).start()
    queue_continuity_updates(stripped, response)
    threading.Thread(target=update_active_task_from_response, args=(stripped, response), daemon=True).start()
    threading.Thread(target=auto_update_goal_progress, args=(stripped, response), daemon=True).start()
    run_bg_knowledge(stripped, response)

    return True


def _handle_habits_command(stripped: str) -> None:
    arg = stripped[7:].strip() if stripped != "/habits" else "view_today"
    if "habit_tracker" in PLUGINS:
        parts = arg.split(None, 1)
        sub   = parts[0].lower() if parts else "view_today"
        name  = parts[1].strip() if len(parts) > 1 else ""
        action_map = {
            "today":   "view_today",
            "streaks": "view_streaks",
            "list":    "list_habits",
            "add":     "add_habit",
            "log":     "log_completion",
            "done":    "log_completion",
            "remove":  "remove_habit",
            "delete":  "remove_habit",
        }
        action = action_map.get(sub, sub)
        a: dict = {"action": action}
        if name:
            a["name"] = name
        result = PLUGINS["habit_tracker"].execute(a)
    else:
        result = "Habit tracker plugin not loaded. Make sure plugins/habit_tracker.py is present."
    print(result)


def _handle_gmail_command(stripped: str) -> None:
    arg = stripped[6:].strip() if stripped != "/gmail" else "list_unread"
    if "gmail" not in PLUGINS:
        print(
            "Gmail plugin not loaded. Make sure plugins/gmail_integration.py is present\n"
            "and gmail_credentials.json is in the project folder.\n"
            "Run: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        )
        return
    parts = arg.split(None, 1)
    sub   = parts[0].lower() if parts else "list_unread"
    rest  = parts[1].strip() if len(parts) > 1 else ""
    action_map = {
        "inbox": "list_unread", "unread": "list_unread",
        "read":  "read_email",  "labels": "list_labels",
    }
    action = action_map.get(sub, sub)
    a: dict = {"action": action}
    if rest:
        if action == "read_email":
            a["message_id"] = rest
        elif action == "list_unread":
            try:
                a["max_results"] = int(rest)
            except ValueError:
                pass
    print(PLUGINS["gmail"].execute(a))


def _handle_calendar_command(stripped: str) -> None:
    arg = stripped[9:].strip() if stripped != "/calendar" else "list_events"
    if "calendar" not in PLUGINS:
        print(
            "Calendar plugin not loaded. Make sure plugins/calendar_integration.py is present\n"
            "and calendar_credentials.json (or gmail_credentials.json) is in the project folder.\n"
            "Run: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        )
        return
    parts = arg.split(None, 1)
    sub   = parts[0].lower() if parts else "list_events"
    rest  = parts[1].strip() if len(parts) > 1 else ""
    action_map = {
        "list": "list_events", "today": "list_events", "week": "list_events",
        "create": "create_event", "add": "create_event",
        "delete": "delete_event", "remove": "delete_event",
    }
    action = action_map.get(sub, sub)
    a: dict = {"action": action}
    if rest:
        if action == "list_events":
            try:
                a["days_ahead"] = int(rest)
            except ValueError:
                pass
        elif action == "create_event":
            a["summary"] = rest
    print(PLUGINS["calendar"].execute(a))


def _handle_discord_command(stripped: str) -> None:
    arg = stripped[8:].strip() if stripped != "/discord" else "list_messages"
    if "discord" not in PLUGINS:
        print(
            "Discord plugin not loaded. Make sure plugins/discord_bot.py is present\n"
            "and vault/discord_config.json contains your bot token."
        )
        return
    parts = arg.split(None, 1)
    sub   = parts[0].lower() if parts else "list_messages"
    rest  = parts[1].strip() if len(parts) > 1 else ""
    action_map = {
        "messages": "list_messages", "read": "list_messages",
        "mentions": "read_mentions", "dm":   "read_mentions",
        "send":     "draft_message", "draft":"draft_message",
        "debug":    "debug_me",      "channel": "debug_channel",
    }
    action = action_map.get(sub, sub)
    a: dict = {"action": action}
    if rest and action in ("draft_message", "send_message"):
        a["content"] = rest
    print(PLUGINS["discord"].execute(a))


def _handle_evolve_command(stripped: str) -> None:
    arg = stripped[7:].strip() if stripped != "/evolve" else ""
    if arg in ("", "status"):
        print_evolution_status()
    elif arg == "pending":
        pending = list_pending_plugins()
        if not pending:
            print("No plugins awaiting approval.")
        else:
            for p in pending:
                print(f"  - {p.get('plugin_name')}  ({p.get('capability')})")
    elif arg.startswith("approve "):
        print(approve_plugin(arg[8:].strip(), PLUGINS))
    elif arg.startswith("reject "):
        print(reject_plugin(arg[7:].strip()))
    elif arg.startswith("show "):
        print(show_plugin_code(arg[5:].strip()))
    else:
        print("[MES] Usage: /evolve [pending|approve <name>|reject <name>|show <name>]")


def _print_memory_stats() -> None:
    from .config import REINFORCE_THRESHOLD
    index      = load_knowledge_index()
    ids        = index.get("ids", [])
    tag_counts = index.get("tag_counts", {})
    title_counts = index.get("title_counts", {})
    scores     = load_memory_scores()
    high_conf  = [v for v in scores.values() if v.get("confidence", 0) >= 0.8]
    print("\n── Memory Stats ──────────────────")
    print(f"  Total knowledge entries : {len(ids)}")
    print(f"  High-confidence memories: {len(high_conf)}")
    print(f"  Chroma available        : {is_chroma_available()}")
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
        reinforced = [(t, c) for t, c in title_counts.items() if c >= REINFORCE_THRESHOLD]
        if reinforced:
            print(f"  Reinforced topics: {', '.join(t for t, _ in reinforced)}")
    print("──────────────────────────────────\n")


def _print_relationship_stats() -> None:
    rel    = load_relationship()
    count  = rel.get("interaction_count", 0)
    first  = rel.get("first_seen", "unknown")
    last   = rel.get("last_seen", "unknown")
    topics = rel.get("favorite_topics", [])
    goals  = load_goals()
    print("\n── Relationship ──────────────────")
    print(f"  First seen        : {first}")
    print(f"  Last seen         : {last}")
    print(f"  Total interactions: {count}")
    if topics:
        print(f"  Favourite topics  : {', '.join(topics[:8])}")
    if goals:
        print(f"  Goals tracked     : {len(goals)}")
        for g in goals[:5]:
            pct = int(g.get("progress", 0) * 100)
            print(f"    [{pct:3d}%] {g['goal']}")
    task = load_active_task()
    if task:
        print(f"  Active task       : {task.get('goal')} ({task.get('status')})")
    print("──────────────────────────────────\n")


def _print_graph_stats() -> None:
    graph = load_knowledge_graph()
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    top   = sorted(nodes.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n── Knowledge Graph ───────────────")
    print(f"  Nodes: {len(nodes)}   Edges: {len(edges)}")
    if top:
        print("  Most connected concepts:")
        for node, weight in top:
            print(f"    {node:<25} (weight: {weight})")
    print("──────────────────────────────────\n")


def _print_integrations_status() -> None:
    print("\n── Integrations ──────────────────────────────────────")
    integrations = {
        "gmail":          ("Gmail",           "gmail_credentials.json"),
        "calendar":       ("Google Calendar", "calendar_credentials.json / gmail_credentials.json"),
        "discord":        ("Discord Bot",     "vault/discord_config.json"),
        "habit_tracker":  ("Habit Tracker",   "built-in (vault/habit_tracker.json)"),
        "obsidian_notes": ("Obsidian Notes",  "OBSIDIAN_VAULT env or obsidian_vault_path.txt"),
        "desktop_notify": ("Desktop Notify",  "built-in (no setup needed)"),
    }
    for plugin_name, (label, setup) in integrations.items():
        loaded = plugin_name in PLUGINS
        status = "✓ loaded" if loaded else "○ not loaded"
        print(f"  {status:<12} {label:<22} ({setup})")
    vad = get_silero_vad_status()
    print(f"\n  Silero VAD : {vad}")
    print(f"  Desktop notifications: {'ON' if DESKTOP_NOTIFY_REMINDERS else 'OFF'}")
    print("──────────────────────────────────────────────────────\n")


def _print_help() -> None:
    print(
        "\n── Commands ────────────────────────────────────────────\n"
        "  /exit or /quit             — exit\n"
        "  /tts                       — toggle speech on/off\n"
        "  /voice [on|off]            — toggle wake-word voice input\n"
        "  /memory                    — show memory stats\n"
        "  /relationship              — show relationship & goal progress\n"
        "  /graph                     — show knowledge graph stats\n"
        "  /goals                     — list active goals\n"
        "  /goal <text>               — add a new goal\n"
        "  /task                      — show active task\n"
        "  /task clear                — clear active task\n"
        "  /journal [N]               — show last N journal entries (default 3)\n"
        "  /reflect                   — run reflection now\n"
        "  /story                     — show life story\n"
        "  /rebuild                   — rebuild Chroma vector index\n"
        "  /forget <word>             — delete memories matching keyword\n"
        "  /models                    — list available Ollama models\n"
        "  /evolve                    — show evolution / plugin status\n"
        "  /evolve pending            — list plugins awaiting approval\n"
        "  /evolve approve <name>     — approve and install a plugin\n"
        "  /evolve reject <name>      — reject and discard a plugin\n"
        "  /evolve show <name>        — view generated plugin code\n"
        "\n── Integrations ────────────────────────────────────────\n"
        "  /habits [today|streaks|list]\n"
        "  /habit add <name>          — add a habit to track\n"
        "  /habit log <name>          — mark a habit done today\n"
        "  /gmail [inbox|read <id>|labels]\n"
        "  /calendar [list|week|create <title>]\n"
        "  /discord [messages|mentions|send <text>]\n"
        "  /notify <message>          — send a desktop notification\n"
        "  /integrations              — show all integration statuses\n"
        "  /help                      — show this list\n"
        "────────────────────────────────────────────────────────\n"
    )


def run() -> None:
    global _recent_turns
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    load_state()

    if not _recent_turns:
        _recent_turns = load_recent_turns_from_disk(LAST_N_MESSAGES)

    if SEARCH_BACKEND == "chroma":
        ok = init_chroma()
        chroma_status = "ON" if ok else "OFF (chromadb not installed — using keyword fallback)"
    else:
        chroma_status = "OFF (SEARCH_BACKEND=keyword)"

    if VOICE_ENABLED_DEFAULT:
        stt_status = "ON" if stt_init_if_needed() else "OFF (STT init failed)"
    else:
        stt_status = "OFF (disabled; use /voice to enable)"

    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    EVOLUTION_PENDING_DIR.mkdir(parents=True, exist_ok=True)
    EVOLUTION_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

    discover_plugins()

    rel     = load_relationship()
    count   = rel.get("interaction_count", 0)
    pending = list_pending_plugins()

    print(f"Connected to Ollama @ {OLLAMA_HOST}")
    print(f"Model             : {MODEL}")
    print(f"Vector search     : {chroma_status}")
    print(f"Voice / wake word : {stt_status}  (wake word: '{WAKE_WORD}')")
    tts_state = "ON" if TTS_ENABLED else "OFF"
    print(f"TTS               : {tts_state}")
    if AUTO_SAVE_KNOWLEDGE:
        print(f"Auto-knowledge    : ON  (model={KNOWLEDGE_MODEL}, confidence≥{KNOWLEDGE_CONFIDENCE_THRESHOLD})")
    else:
        print("Auto-knowledge    : OFF")
    if EVOLUTION_ENABLED:
        metrics = load_evolution_metrics()
        print(
            f"Evolution (MES)   : ON  (coder={EVOLUTION_CODER_MODEL}, "
            f"threshold={EVOLUTION_THRESHOLD}, gained={metrics.get('capabilities_gained', 0)})"
        )
        if pending:
            print(f"  ⚠  {len(pending)} plugin(s) awaiting approval — use /evolve pending")
    else:
        print("Evolution (MES)   : OFF")
    active_goals = len([g for g in load_goals() if g.get("status") == "active"])
    print(f"Companion memory  : {count} interactions  |  Goals: {active_goals} active")
    print(f"Journal           : {'ON' if JOURNAL_ENABLED else 'OFF'}  |  Reflection hour: {REFLECTION_HOUR:02d}:00")
    print("Type /help for commands.  Use /voice to enable wake-word voice input.\n")

    _speech_queue:      queue.Queue = queue.Queue()
    _voice_input_queue: queue.Queue = queue.Queue()

    threading.Thread(target=speech_loop,    args=(_speech_queue,),  daemon=True).start()
    threading.Thread(target=initiative_loop, args=(_speech_queue,), daemon=True).start()
    threading.Thread(target=reminder_loop,  args=(_speech_queue,),  daemon=True).start()
    threading.Thread(target=reflection_loop,                         daemon=True).start()

    if VOICE_ENABLED_DEFAULT:
        set_voice_enabled(True, voice_input_queue=_voice_input_queue)

    while True:
        try:
            voice_text = _voice_input_queue.get_nowait()
            if not _handle_input(voice_text, voice_input_queue=_voice_input_queue):
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
    persist_state()
