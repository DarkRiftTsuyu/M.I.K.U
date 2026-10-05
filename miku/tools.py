from __future__ import annotations

import importlib
import logging
import pkgutil
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .config import (
    VAULT_PATH, PENDING_CONFIRMATION_FILE, EVOLUTION_ENABLED,
    WAKE_WORD, BASE_DIR,
)
from .utils import extract_json_object, save_json_file, load_json_file, now_iso

log = logging.getLogger("miku.tools")

PLUGINS: dict = {}

CONFIRM_REQUIRED_ACTIONS = {
    ("gmail",         "send_draft"),
    ("discord",       "send_message"),
    ("calendar",      "create_event"),
    ("calendar",      "reschedule_event"),
    ("calendar",      "delete_event"),
    ("habit_tracker", "remove_habit"),
    ("obsidian_notes","overwrite"),
}

_tool_failure_state: dict = {
    "recent_failure": False, "when": None, "where": None, "error": None,
}


def record_tool_failure(where: str, error: str) -> None:
    global _tool_failure_state
    _tool_failure_state = {
        "recent_failure": True,
        "when": now_iso(),
        "where": where,
        "error": (error or "")[:500],
    }


def pop_tool_failure_notice(ttl_s: int = 600) -> str:
    global _tool_failure_state
    from datetime import datetime
    state = dict(_tool_failure_state)
    if not state.get("recent_failure"):
        return ""
    when = state.get("when")
    try:
        if when:
            dt = datetime.fromisoformat(str(when))
            if (datetime.now() - dt).total_seconds() > ttl_s:
                _tool_failure_state = {"recent_failure": False, "when": None, "where": None, "error": None}
                return ""
    except Exception:
        pass
    _tool_failure_state = {"recent_failure": False, "when": None, "where": None, "error": None}
    loc = state.get("where") or "tool"
    err = state.get("error") or "unknown error"
    return (
        "A tool execution recently failed. Do not assume the task succeeded. "
        f"If relevant, acknowledge the failure briefly and ask to retry. (where={loc}; error={err})"
    )


def discover_plugins() -> None:
    try:
        import plugins as plugins_pkg
        for _, name, _ in pkgutil.iter_modules(plugins_pkg.__path__):
            try:
                module = importlib.import_module(f"plugins.{name}")
                if hasattr(module, "PLUGIN") and hasattr(module, "execute"):
                    PLUGINS[module.PLUGIN["name"]] = module
            except Exception as exc:
                log.warning("[Plugins] Could not load %s: %s", name, exc)
    except ImportError:
        pass


def _tool_schema_text(module) -> str:
    schema = getattr(module, "SCHEMA", None)
    if not schema:
        return ""
    required = schema.get("required") or {}
    optional = schema.get("optional") or {}
    lines: list[str] = []
    if isinstance(required, dict) and required:
        lines.append("Required args:")
        for k, v in list(required.items())[:12]:
            lines.append(f"- {k}: {v}")
    if isinstance(optional, dict) and optional:
        lines.append("Optional args:")
        for k, v in list(optional.items())[:12]:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def available_tools_prompt() -> str:
    if not PLUGINS:
        return "No tools available."
    lines = []
    for name, module in PLUGINS.items():
        desc = module.PLUGIN.get("description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def available_tools_schema_prompt() -> str:
    if not PLUGINS:
        return "No tools available."
    blocks: list[str] = []
    for name, module in PLUGINS.items():
        desc = getattr(module, "PLUGIN", {}).get("description", "")
        block = [f"Tool: {name}", f"Description: {desc}" if desc else "Description: (none)"]
        schema_text = _tool_schema_text(module)
        if schema_text:
            block.append(schema_text)
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def load_capabilities() -> dict:
    caps: dict = {}
    caps_file = VAULT_PATH / "capabilities.json"
    if caps_file.exists():
        try:
            obj = load_json_file(caps_file, default={})
            if isinstance(obj, dict):
                caps.update(obj)
        except Exception:
            pass
    for name, module in PLUGINS.items():
        caps.setdefault(name, {"enabled": True, "functions": list(getattr(module, "FUNCTIONS", [name]))})
    return caps


def build_capability_block() -> str:
    caps  = load_capabilities()
    lines = ["Enabled capabilities:"]
    for cap_name, info in caps.items():
        if (info or {}).get("enabled"):
            label = cap_name.replace("_", " ").title()
            lines.append(f"- {label}")
    return "\n".join(lines)


def capability_constraints_for_input(user_input: str) -> str:
    text = (user_input or "").lower()
    caps = load_capabilities()
    mentioned_disabled: list[str] = []
    for cap_name, info in caps.items():
        if bool((info or {}).get("enabled", False)):
            continue
        if cap_name in text or cap_name.replace("_", " ") in text:
            mentioned_disabled.append(cap_name)
    if not mentioned_disabled:
        return ""
    lines = ["Capability Check (authoritative):"]
    for cap in sorted(set(mentioned_disabled)):
        label = cap.replace("_", " ").title()
        lines.append(f"- {label} = unavailable (do not claim you can use it)")
    lines.append("If the user requests an unavailable capability, say you can't and offer an alternative.")
    return "\n".join(lines)


def _tool_action_name(args: dict) -> str:
    return str(args.get("action") or args.get("mode") or "").strip().lower()


def requires_confirmation(tool: str, args: dict) -> bool:
    return (tool, _tool_action_name(args)) in CONFIRM_REQUIRED_ACTIONS


def save_pending_confirmation(tool: str, args: dict) -> None:
    save_json_file(PENDING_CONFIRMATION_FILE, {
        "tool": tool, "args": args, "created": now_iso(),
    })


def load_pending_confirmation() -> dict | None:
    obj = load_json_file(PENDING_CONFIRMATION_FILE, default=None)
    return obj if isinstance(obj, dict) else None


def clear_pending_confirmation() -> None:
    try:
        if PENDING_CONFIRMATION_FILE.exists():
            PENDING_CONFIRMATION_FILE.unlink()
    except Exception:
        pass


def plan_tool_call(user_input: str, load_tool_context_fn, format_recent_fn) -> str:
    from .llm import generate_non_stream
    import json
    ctx    = load_tool_context_fn()
    recent = format_recent_fn()
    prompt = (
        f"Available tools (with schemas):\n{available_tools_schema_prompt()}\n\n"
        f"Tool context (authoritative):\n{json.dumps(ctx, indent=2)[:900]}\n\n"
        f"Recent conversation:\n{recent}\n\n"
        "If a tool should be used, return STRICT JSON:\n"
        '{"tool": "tool.name", "args": {}}\n\n'
        "Rules:\n"
        "- Return ONLY JSON or null (no extra text)\n"
        "- If a tool schema has required args, include them\n"
        "- If user says 'that file'/'that note' and tool context has last_note_title, use it\n"
        "- Choose mode=append for updates; mode=create for new notes; mode=overwrite only if user says overwrite\n\n"
        "If none needed, return: null\n\n"
        f"User:\n<user>{user_input}</user>"
    )
    return generate_non_stream(prompt)


def should_use_tools(text: str) -> bool:
    lower = (text or "").lower().strip()
    if not lower:
        return False
    chat_phrases = [
        "well done","good job","nice work","thank you","thanks",
        "what would you like","what do you want","how are you",
        "no need","never mind","it's okay","its okay","all good","no worries",
    ]
    if any(p in lower for p in chat_phrases):
        return False
    tool_verbs = [
        "create","update","append","write","open","launch","play",
        "send","delete","edit","overwrite","add","remove",
    ]
    return any(v in lower for v in tool_verbs)


def maybe_clear_tool_context(text: str, load_tool_context_fn, save_tool_context_fn, save_active_task_fn) -> bool:
    lower = (text or "").lower()
    if not lower.strip():
        return False
    close_phrases = [
        "no need","stop editing","done with that","forget that file",
        "that's enough","thats enough","no more appending","no more",
        "don't append","dont append","never mind",
    ]
    if not any(p in lower for p in close_phrases):
        return False
    try:
        ctx     = load_tool_context_fn()
        changed = False
        for k in ("last_note_title", "last_note_path"):
            if k in ctx:
                ctx.pop(k, None)
                changed = True
        if changed:
            save_tool_context_fn(ctx)
    except Exception:
        pass
    try:
        from .memory import load_active_task
        task = load_active_task()
        if isinstance(task, dict) and task.get("type") == "note_editing":
            save_active_task_fn(None)
    except Exception:
        pass
    return True


def looks_like_note_edit(text: str, load_tool_context_fn) -> bool:
    lower         = (text or "").lower()
    has_edit_verb = any(v in lower for v in ["update","append","write","add","edit","overwrite","create"])
    if not has_edit_verb:
        return False
    targets    = ["note","file","plugin ideas","that file","that note","obsidian"]
    has_target = any(t in lower for t in targets)
    if not has_target and "it" in lower:
        try:
            ctx = load_tool_context_fn()
            if str(ctx.get("last_note_title", "")).strip():
                has_target = True
        except Exception:
            pass
    return has_target


def _looks_like_real_markdown(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "\n" in t:
        if re.search(r"^\s{0,3}#{1,6}\s+\S", t, re.MULTILINE):
            return True
        if re.search(r"^\s*[-*+]\s+\S", t, re.MULTILINE):
            return True
        if re.search(r"^\s*\d+\.\s+\S", t, re.MULTILINE):
            return True
        if "```" in t:
            return True
    if t.startswith(("#", "-", "*", ">", "```")):
        return True
    return False


def _content_sounds_instructional(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _looks_like_real_markdown(t):
        return False
    lower = t.lower()
    if len(lower) <= 40 and not any(ch in lower for ch in (".", ":", "!", "?")):
        return True
    instructional_phrases = (
        "some ","you would like","please","generate","write","add ","append",
        "include","list","ideas","plugins","tasks","next steps","bullet",
    )
    if any(p in lower for p in instructional_phrases):
        return True
    if lower.startswith(("some ","a list","list of","ideas for","plugins for")):
        return True
    return False


def _expand_obsidian_content_if_needed(
    *, title: str, planned_content: str, user_input: str, mode: str, load_tool_context_fn, format_recent_fn
) -> str:
    import json
    from .llm import generate_non_stream_custom
    content = (planned_content or "").strip()
    if not content:
        return content
    if not _content_sounds_instructional(content):
        return content
    ctx    = load_tool_context_fn()
    recent = format_recent_fn()
    prompt = (
        "You are generating a markdown snippet to append to a note.\n\n"
        "Output ONLY the markdown content to append (no preface, no code fences).\n"
        "Be concrete and useful; avoid vague filler.\n\n"
        f"Note title: {title}\n"
        f"Mode: {mode}\n"
        f"Tool context (authoritative): {json.dumps(ctx, ensure_ascii=False)[:700]}\n\n"
        f"Recent conversation (most recent last):\n{recent}\n\n"
        f"User asked: {user_input.strip()[:600]}\n"
        f"Planned tool content (instruction-like): {content[:250]}\n\n"
        "Guidelines:\n"
        "- If the user asks for plugin ideas, include a section heading like '## Plugin Ideas'\n"
        "  and 4-8 bullet points with specific integrations.\n"
        "- Prefer concise bullets.\n"
        "- Do not mention these guidelines.\n"
    )
    generated = generate_non_stream_custom(prompt, temperature=0.45, num_predict=240).strip()
    if not generated:
        return content
    if (generated.startswith('"') and generated.endswith('"')) or (generated.startswith("'") and generated.endswith("'")):
        generated = generated[1:-1].strip()
    return generated or content


def execute_tool_plan(
    plan: str,
    user_input: str | None,
    load_tool_context_fn,
    save_tool_context_fn,
    save_active_task_fn,
    format_recent_fn,
) -> str | None:
    import json
    plan_text = (plan or "").strip()
    if not plan_text or plan_text.lower() in ("null", "none"):
        return None
    obj = extract_json_object(plan_text)
    if not obj:
        record_tool_failure("tool_planner", "invalid JSON from tool planner")
        log.warning("Tool execution failed: invalid planner JSON")
        return None
    tool = str(obj.get("tool", "")).strip()
    args = obj.get("args", {})
    if not tool:
        return None
    if not isinstance(args, dict):
        args = {}
    if tool not in PLUGINS:
        record_tool_failure("tool_router", f"unknown tool '{tool}'")
        return None
    try:
        if tool == "obsidian_notes":
            ctx = load_tool_context_fn()
            if "title" not in args or not str(args.get("title", "")).strip():
                last_title = str(ctx.get("last_note_title", "")).strip()
                if last_title:
                    args["title"] = last_title
    except Exception:
        pass
    if tool == "obsidian_notes":
        title = str(args.get("title", "")).strip()
        if not title:
            return "What should the note title be? (e.g., 'Plugin Ideas')"
    if tool == "obsidian_notes":
        try:
            if user_input and not looks_like_note_edit(str(user_input), load_tool_context_fn):
                return None
        except Exception:
            pass
    if tool == "obsidian_notes":
        try:
            title    = str(args.get("title", "")).strip()
            mode     = str(args.get("mode", "append")).strip().lower() or "append"
            planned  = str(args.get("content", "") or "")
            expanded = _expand_obsidian_content_if_needed(
                title=title, planned_content=planned,
                user_input=str(user_input or ""), mode=mode,
                load_tool_context_fn=load_tool_context_fn,
                format_recent_fn=format_recent_fn,
            )
            if expanded and expanded != planned:
                args["content"] = expanded
        except Exception as exc:
            log.debug("obsidian content expansion failed: %s", exc)
    try:
        if requires_confirmation(tool, args):
            save_pending_confirmation(tool, args)
            return (
                "I need confirmation before doing that.\n\n"
                f"Tool: {tool}\n"
                f"Action: {_tool_action_name(args)}\n"
                f"Args: {json.dumps(args, indent=2)}\n\n"
                "Type /confirm to run it, or /cancel to stop."
            )
        result = PLUGINS[tool].execute(args)
        if tool == "obsidian_notes":
            title = str(args.get("title", "")).strip()
            if title:
                mode     = str(args.get("mode", "append")).strip().lower() or "append"
                safe     = re.sub(r"[\\/:*?\"<>|]", "-", title).strip()[:120] or "Untitled"
                filename = safe + ".md"
                ctx      = load_tool_context_fn()
                ctx.update({
                    "last_note_title": title,
                    "last_note_path": filename,
                    "last_tool": tool,
                })
                save_tool_context_fn(ctx)
                status = "active" if mode == "create" else "completed"
                save_active_task_fn({
                    "type": "note_editing",
                    "goal": f"Edit Obsidian note: {title}",
                    "status": status,
                    "title": title,
                    "file": filename,
                    "last_update": now_iso(),
                })
        return result
    except Exception as e:
        record_tool_failure(tool, str(e))
        log.warning("Tool execution failed: %s", e)
        return None


def detect_missing_capability(text: str) -> str | None:
    lower = text.lower()
    missing_caps = {
        "spotify": ["play music", "play spotify", "skip track", "pause music", "next song"],
        "whatsapp": ["whatsapp", "send whatsapp"],
        "phone_call": ["call ", "phone call", "ring "],
        "twitter": ["tweet", "post on twitter", "twitter"],
        "youtube": ["play youtube", "youtube video"],
        "sms": ["send sms", "send text message", "text message to"],
    }
    for cap, phrases in missing_caps.items():
        if any(p in lower for p in phrases):
            caps = load_capabilities()
            if not caps.get(cap, {}).get("enabled", False):
                return cap
    return None


_APP_REGISTRY: list[tuple[list[str], str, str]] = [
    (["cider", "cider.exe"],          "Cider",              "Cider (Apple Music client)"),
    (["spotify", "spotify.exe"],      "Spotify",            "Spotify"),
    (["discord", "discord.exe"],      "Discord",            "Discord"),
    (["code", "code.exe", "vscode"],  "Visual Studio Code", "VS Code"),
    (["firefox", "firefox.exe"],      "Firefox",            "Firefox"),
    (["chrome", "google-chrome", "google-chrome-stable"], "Google Chrome", "Chrome"),
    (["steam", "steam.exe"],          "Steam",              "Steam"),
    (["obs", "obs-studio"],           "OBS",                "OBS Studio"),
    (["vlc"],                         "VLC",                "VLC"),
    (["slack", "slack.exe"],          "Slack",              "Slack"),
    (["notion", "notion.exe"],        "Notion",             "Notion"),
    (["obsidian", "obsidian.exe"],    "Obsidian",           "Obsidian"),
    (["terminal", "gnome-terminal", "konsole", "xterm"], "", "Terminal"),
    (["notepad", "gedit", "kate"],    "",                   "Text editor"),
    (["explorer", "nautilus", "thunar", "dolphin"], "",   "File manager"),
]

_OPEN_PATTERNS = re.compile(
    r"\b(?:open|launch|start|run|can\s+you\s+open|please\s+open)\s+(.+)",
    re.IGNORECASE,
)


def detect_open_intent(text: str) -> str | None:
    cleaned = re.sub(rf"\b{re.escape(WAKE_WORD)}\b[,]?\s*", "", text, flags=re.IGNORECASE).strip()
    m = _OPEN_PATTERNS.search(cleaned)
    if m:
        app = m.group(1).strip().rstrip("?. ")
        return app
    return None


def launch_app(app_name: str) -> str:
    target = app_name.strip().lower()
    matched_exes:   list[str] = []
    matched_bundle: str = ""
    matched_label:  str = target
    for exes, bundle, label in _APP_REGISTRY:
        if any(target in exe or exe in target for exe in exes):
            matched_exes   = exes
            matched_bundle = bundle
            matched_label  = label
            break
    if not matched_exes:
        matched_exes  = [target]
        matched_label = target
    if sys.platform == "darwin" and matched_bundle:
        result = subprocess.run(["open", "-a", matched_bundle], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Opened {matched_label}!"
    for exe in matched_exes:
        found = shutil.which(exe)
        if found:
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        [found],
                        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                else:
                    subprocess.Popen([found], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                return f"Opened {matched_label}!"
            except Exception as exc:
                log.warning("[Launcher] %s failed: %s", found, exc)
    if sys.platform == "win32":
        try:
            subprocess.Popen(
                ["explorer", f"shell:AppsFolder\\{target}"],
                creationflags=subprocess.DETACHED_PROCESS,
            )
            return f"Tried to open {matched_label} via Windows shell."
        except Exception:
            pass
    return f"I couldn't find {matched_label} on your system. Make sure it's installed and on your PATH."


def send_desktop_notification(title: str = "Miku", message: str = "") -> None:
    if "desktop_notify" in PLUGINS:
        try:
            PLUGINS["desktop_notify"].execute({"title": title, "message": message})
            return
        except Exception:
            pass
    _inline_notify(title, message)


def _inline_notify(title: str, message: str) -> None:
    try:
        if sys.platform == "win32":
            ps = shutil.which("powershell.exe") or shutil.which("powershell")
            if ps:
                t, m = title.replace("'", "\\'"), message.replace("'", "\\'")
                script = (
                    "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
                    "ContentType = WindowsRuntime] | Out-Null; "
                    "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
                    "ContentType = WindowsRuntime] | Out-Null; "
                    f"$xml=[Windows.UI.Notifications.ToastNotificationManager]"
                    f"::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
                    f"$xml.GetElementsByTagName('text')[0].AppendChild($xml.CreateTextNode('{t}'))|Out-Null;"
                    f"$xml.GetElementsByTagName('text')[1].AppendChild($xml.CreateTextNode('{m}'))|Out-Null;"
                    f"$toast=[Windows.UI.Notifications.ToastNotification]::new($xml);"
                    f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Miku').Show($toast)"
                )
                subprocess.Popen(
                    [ps, "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        elif sys.platform == "darwin":
            t, m = title.replace('"', '\\"'), message.replace('"', '\\"')
            subprocess.Popen(
                ["osascript", "-e", f'display notification "{m}" with title "{t}"'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            if shutil.which("notify-send"):
                subprocess.Popen(
                    ["notify-send", "-t", "5000", title, message],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
    except Exception as exc:
        log.debug("[Notify] desktop notification failed: %s", exc)
