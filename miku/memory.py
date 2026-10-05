from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from .config import (
    VAULT_PATH, MEMORY_DIR, KNOWLEDGE_DIR, KNOWLEDGE_INDEX_FILE,
    KNOWLEDGE_GRAPH_FILE, KNOWLEDGE_CONFIDENCE_THRESHOLD, REINFORCE_THRESHOLD,
    MEMORY_TYPE_FILES, TAG_TO_TYPE, RELATIONSHIP_FILE, GOALS_FILE,
    ACTIVE_TASK_FILE, CONVERSATION_STATE_FILE, INTERNAL_NOTES_FILE,
    RECENT_TURNS_FILE, TOOL_CONTEXT_FILE, REMINDERS_FILE,
    CONVERSATION_SUMMARY_FILE, CONTEXT_COMPRESSION_INTERVAL,
    CURIOSITY_FILE, CURIOSITY_MIN_CONFIDENCE, CURIOSITY_MIN_DAYS_BETWEEN,
    LIFE_STORY_FILE, REFLECTION_FILE, JOURNAL_DIR, JOURNAL_ENABLED,
    LAST_N_MESSAGES, STATE_FILE,
)
from .utils import load_json_file, save_json_file, now_iso, extract_json_object

log = logging.getLogger("miku.memory")


def load_identity(identity_file: Path) -> str:
    if identity_file.exists():
        return identity_file.read_text(encoding="utf-8")
    return "You are Hatsune Miku, a friendly and warm personal assistant."


def load_tool_context() -> dict:
    ctx = load_json_file(TOOL_CONTEXT_FILE, default={})
    return ctx if isinstance(ctx, dict) else {}


def save_tool_context(ctx: dict) -> None:
    ctx = dict(ctx or {})
    ctx["updated"] = now_iso()
    save_json_file(TOOL_CONTEXT_FILE, ctx)


def load_recent_turns_from_disk(n_pairs: int) -> list[dict]:
    if n_pairs <= 0:
        return []
    max_items = max(2, n_pairs * 2)
    turns: list[dict] = []
    try:
        if RECENT_TURNS_FILE.exists():
            lines = RECENT_TURNS_FILE.read_text(encoding="utf-8").splitlines()
            for line in lines[-max_items:]:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("role") in ("user", "assistant"):
                    turns.append({"role": obj["role"], "content": str(obj.get("content", ""))[:1200]})
            return turns[-max_items:]
    except Exception:
        pass
    try:
        if not MEMORY_DIR.exists():
            return []
        md_files = sorted(MEMORY_DIR.glob("*.md"))[-3:]
        user_pat = re.compile(r"^\*\*User:\*\*\s*(.*)$", re.MULTILINE)
        miku_pat = re.compile(r"^\*\*Miku:\*\*\s*(.*)$", re.MULTILINE)
        pairs: list[tuple[str, str]] = []
        for f in md_files:
            try:
                content = f.read_text(encoding="utf-8")
            except Exception:
                continue
            users = user_pat.findall(content)
            mikus = miku_pat.findall(content)
            for u, m in zip(users, mikus):
                pairs.append((u.strip(), m.strip()))
        for u, m in pairs[-n_pairs:]:
            turns.append({"role": "user", "content": u[:1200]})
            turns.append({"role": "assistant", "content": m[:1200]})
        return turns[-max_items:]
    except Exception:
        return []


def append_recent_turn(recent_turns: list[dict], role: str, content: str) -> list[dict]:
    if role not in ("user", "assistant"):
        return recent_turns
    content = (content or "").strip()
    if not content:
        return recent_turns
    recent_turns.append({"role": role, "content": content[:1200]})
    recent_turns = recent_turns[-max(4, LAST_N_MESSAGES * 2):]
    try:
        RECENT_TURNS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RECENT_TURNS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now_iso(), "role": role, "content": content[:1200]}) + "\n")
    except Exception:
        pass
    return recent_turns


def format_recent_conversation(recent_turns: list[dict], max_pairs: int) -> str:
    if max_pairs <= 0 or not recent_turns:
        return ""
    items = recent_turns[-max_pairs * 2:]
    lines: list[str] = []
    for it in items:
        role = "User" if it.get("role") == "user" else "Miku"
        text = str(it.get("content", "")).strip().replace("\n", " ")
        if len(text) > 260:
            text = text[:257] + "…"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


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


def load_relationship() -> dict:
    if RELATIONSHIP_FILE.exists():
        try:
            obj = json.loads(RELATIONSHIP_FILE.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {
        "interaction_count": 0,
        "first_seen": date.today().isoformat(),
        "favorite_topics": [],
        "topic_counts": {},
        "last_seen": date.today().isoformat(),
    }


def save_relationship(rel: dict) -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        RELATIONSHIP_FILE.write_text(json.dumps(rel, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save relationship.json: %s", exc)


def update_relationship(user_input: str) -> None:
    rel = load_relationship()
    rel["interaction_count"] = rel.get("interaction_count", 0) + 1
    rel["last_seen"] = date.today().isoformat()
    words = re.findall(r"[a-zA-Z]{4,}", user_input.lower())
    topic_counts = rel.get("topic_counts", {})
    stop = {"that","this","with","have","been","from","they","will","what","when","where","your","just","also","into","then","than","more","some","like","very","would","could","should"}
    for w in words:
        if w not in stop:
            topic_counts[w] = topic_counts.get(w, 0) + 1
    rel["topic_counts"] = topic_counts
    top = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    rel["favorite_topics"] = [t for t, _ in top]
    save_relationship(rel)


def get_relationship_context() -> str:
    rel    = load_relationship()
    count  = rel.get("interaction_count", 0)
    first  = rel.get("first_seen", "recently")
    topics = rel.get("favorite_topics", [])[:5]
    last   = rel.get("last_seen", "today")
    lines  = [f"Relationship context: {count} interactions since {first}, last active {last}."]
    if topics:
        lines.append(f"Recurring interests: {', '.join(topics)}.")
    return "\n".join(lines)


def load_goals() -> list[dict]:
    if GOALS_FILE.exists():
        try:
            obj = json.loads(GOALS_FILE.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return []


def save_goals(goals: list[dict]) -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        GOALS_FILE.write_text(json.dumps(goals, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save goals.json: %s", exc)


def add_goal(goal_text: str, milestones: list[str] | None = None) -> str:
    goals = load_goals()
    goals.append({
        "id": hashlib.sha1(goal_text.encode()).hexdigest()[:8],
        "goal": goal_text,
        "progress": 0.0,
        "status": "active",
        "milestones": milestones or [],
        "created": date.today().isoformat(),
        "last_updated": date.today().isoformat(),
    })
    save_goals(goals)
    return f"Goal set: '{goal_text}'."


def build_goals_block() -> str:
    goals = [g for g in load_goals() if g.get("status") == "active"]
    if not goals:
        return ""
    lines = ["Active goals:"]
    for g in goals[:3]:
        pct  = int(g.get("progress", 0) * 100)
        ms   = g.get("milestones", [])
        done = sum(1 for m in ms if isinstance(m, dict) and m.get("done"))
        milestone_note = f" ({done}/{len(ms)} milestones)" if ms else ""
        lines.append(f"  • {g['goal']}  [{pct}%]{milestone_note}")
    return "\n".join(lines)


def load_active_task() -> dict | None:
    if not ACTIVE_TASK_FILE.exists():
        return None
    try:
        obj = json.loads(ACTIVE_TASK_FILE.read_text(encoding="utf-8"))
        if not (isinstance(obj, dict) and obj.get("type")):
            return None
        status = str(obj.get("status", "")).strip().lower()
        if status in ("complete", "completed", "done"):
            return None
        return obj
    except Exception:
        return None


def save_active_task(task: dict | None) -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    if task is None:
        if ACTIVE_TASK_FILE.exists():
            ACTIVE_TASK_FILE.unlink()
        return
    try:
        ACTIVE_TASK_FILE.write_text(json.dumps(task, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save active_task.json: %s", exc)


def build_task_block() -> str:
    task = load_active_task()
    if not task:
        return ""
    goal   = task.get("goal", "unknown")
    status = task.get("status", "unknown")
    reqs   = task.get("requirements", [])
    lines  = [f"Current Active Task: {goal}", f"Status: {status}"]
    if reqs:
        lines.append("Requirements gathered: " + ", ".join(str(r) for r in reqs))
    return "\n".join(lines)


def load_conversation_state() -> dict:
    state = load_json_file(CONVERSATION_STATE_FILE, default={})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("current_topic", "")
    state.setdefault("open_threads", [])
    state.setdefault("last_goal", "")
    state.setdefault("updated", None)
    if not isinstance(state.get("open_threads"), list):
        state["open_threads"] = []
    state["open_threads"] = [str(t) for t in state["open_threads"] if str(t).strip()][:8]
    return state


def save_conversation_state(state: dict) -> None:
    state = dict(state or {})
    state["updated"] = now_iso()
    save_json_file(CONVERSATION_STATE_FILE, state)


def build_conversation_state_block() -> str:
    state   = load_conversation_state()
    topic   = str(state.get("current_topic", "")).strip()
    threads = state.get("open_threads", []) or []
    goal    = str(state.get("last_goal", "")).strip()
    lines   = ["Current conversation topic:"]
    lines.append(topic or "(not set yet)")
    lines.append("")
    lines.append("Open discussion threads:")
    if threads:
        for t in threads[:6]:
            lines.append(f"- {t}")
    else:
        lines.append("- (none)")
    if goal:
        lines.append("")
        lines.append(f"Last conversational goal: {goal}")
    lines.append("")
    lines.append("Avoid changing topic unless the user changes topic.")
    lines.append("If the user says they don't need anything right now, acknowledge and gently maintain continuity instead of resetting.")
    return "\n".join(lines)


def load_internal_notes() -> list[dict]:
    notes = load_json_file(INTERNAL_NOTES_FILE, default=[])
    if not isinstance(notes, list):
        return []
    cleaned: list[dict] = []
    for n in notes[-50:]:
        if not isinstance(n, dict):
            continue
        topic  = str(n.get("topic", "")).strip()
        follow = str(n.get("follow_up", "")).strip()
        level  = str(n.get("interest_level", "")).strip().lower()
        if not topic and not follow:
            continue
        if level not in ("low", "medium", "high"):
            level = "medium" if (topic or follow) else "low"
        cleaned.append({
            "topic": topic[:120],
            "interest_level": level,
            "follow_up": follow[:220],
            "created": n.get("created") or now_iso(),
        })
    return cleaned[-30:]


def save_internal_notes(notes: list[dict]) -> None:
    save_json_file(INTERNAL_NOTES_FILE, notes[-30:])


def append_internal_note(note: dict | None) -> None:
    if not note or not isinstance(note, dict):
        return
    topic  = str(note.get("topic", "")).strip()
    follow = str(note.get("follow_up", "")).strip()
    if not topic and not follow:
        return
    level = str(note.get("interest_level", "medium")).strip().lower()
    if level not in ("low", "medium", "high"):
        level = "medium"
    notes    = load_internal_notes()
    new_note = {
        "topic": topic[:120],
        "interest_level": level,
        "follow_up": follow[:220],
        "created": now_iso(),
    }
    if new_note["topic"]:
        notes = [n for n in notes if str(n.get("topic", "")).strip().lower() != new_note["topic"].lower()]
    notes.append(new_note)
    save_internal_notes(notes)


def build_internal_notes_block(max_notes: int = 5) -> str:
    notes = load_internal_notes()
    if not notes:
        return ""
    lines = ["Miku's internal notes (private, do not mention directly):"]
    for n in notes[-max_notes:]:
        topic  = str(n.get("topic", "")).strip()
        follow = str(n.get("follow_up", "")).strip()
        level  = str(n.get("interest_level", "medium")).strip().lower()
        if not topic and not follow:
            continue
        prefix = "high" if level == "high" else ("low" if level == "low" else "mid")
        if topic and follow:
            lines.append(f"- ({prefix}) {topic} — {follow}")
        elif topic:
            lines.append(f"- ({prefix}) {topic}")
        else:
            lines.append(f"- ({prefix}) {follow}")
    return "\n".join(lines)


def load_reminders() -> list[dict]:
    if not REMINDERS_FILE.exists():
        return []
    try:
        return json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load reminders: %s", exc)
        return []


def save_reminders(reminders: list[dict]) -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        REMINDERS_FILE.write_text(json.dumps(reminders, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save reminders: %s", exc)


def add_reminder(message: str, due_iso: str) -> str:
    reminders = load_reminders()
    reminders.append({"message": message, "due": due_iso})
    save_reminders(reminders)
    return f"Reminder set: '{message}' at {due_iso}."


def load_knowledge_graph() -> dict:
    if KNOWLEDGE_GRAPH_FILE.exists():
        try:
            return json.loads(KNOWLEDGE_GRAPH_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"nodes": {}, "edges": []}


def save_knowledge_graph(graph: dict) -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        KNOWLEDGE_GRAPH_FILE.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save knowledge_graph.json: %s", exc)


def update_knowledge_graph(tags: list[str], title: str, summary: str) -> None:
    graph = load_knowledge_graph()
    nodes = graph.setdefault("nodes", {})
    edges = graph.setdefault("edges", [])
    for tag in tags:
        t = tag.lower().strip()
        if t:
            nodes[t] = nodes.get(t, 0) + 1
    title_slug = re.sub(r"\W+", "_", title.lower())[:40]
    nodes[title_slug] = nodes.get(title_slug, 0) + 1
    for i, t1 in enumerate(tags):
        for t2 in tags[i+1:]:
            edge = sorted([t1.lower().strip(), t2.lower().strip()])
            key  = f"{edge[0]}__{edge[1]}"
            existing = next((e for e in edges if e.get("key") == key), None)
            if existing:
                existing["weight"] = existing.get("weight", 1) + 1
            else:
                edges.append({"key": key, "a": edge[0], "b": edge[1], "weight": 1})
    save_knowledge_graph(graph)


def graph_context_for(topic: str) -> str:
    graph   = load_knowledge_graph()
    edges   = graph.get("edges", [])
    t       = topic.lower().strip()
    related = []
    for edge in edges:
        if edge.get("a") == t:
            related.append((edge["b"], edge.get("weight", 1)))
        elif edge.get("b") == t:
            related.append((edge["a"], edge.get("weight", 1)))
    related.sort(key=lambda x: x[1], reverse=True)
    if not related:
        return ""
    top = [r[0] for r in related[:5]]
    return f"Graph context for '{topic}': related to {', '.join(top)}."


def load_memory_scores() -> dict:
    score_file = KNOWLEDGE_DIR / "_memory_scores.json"
    if score_file.exists():
        try:
            return json.loads(score_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_memory_scores(scores: dict) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    score_file = KNOWLEDGE_DIR / "_memory_scores.json"
    try:
        score_file.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save memory scores: %s", exc)


def score_memory_entry(entry_id: str, title: str, confirmed: bool = False) -> float:
    scores = load_memory_scores()
    rec    = scores.get(entry_id, {"mentions": 0, "confidence": 0.5, "fact": title})
    rec["mentions"] = rec.get("mentions", 0) + 1
    if confirmed:
        rec["confidence"] = min(1.0, rec.get("confidence", 0.5) + 0.1)
    else:
        n = rec["mentions"]
        rec["confidence"] = min(0.99, 0.4 + 0.06 * n)
    scores[entry_id] = rec
    save_memory_scores(scores)
    return rec["confidence"]


def high_confidence_memories(min_score: float = 0.7) -> list[dict]:
    scores = load_memory_scores()
    return [v for v in scores.values() if v.get("confidence", 0) >= min_score]


def load_knowledge_index() -> dict:
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


def save_knowledge_index(index: dict) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    index["updated"] = datetime.now().isoformat(timespec="seconds")
    try:
        KNOWLEDGE_INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save knowledge index: %s", exc)


def resolve_memory_type(tags: list[str], title: str) -> str:
    for tag in tags:
        t = tag.lower().strip()
        if t in TAG_TO_TYPE:
            return TAG_TO_TYPE[t]
    title_lower = title.lower()
    for keyword, mem_type in TAG_TO_TYPE.items():
        if keyword in title_lower:
            return mem_type
    return "general"


def check_and_reinforce(index: dict, tags: list[str], title: str, summary_md: str) -> None:
    tag_counts   = index.setdefault("tag_counts", {})
    title_counts = index.setdefault("title_counts", {})
    title_slug   = re.sub(r"\W+", "_", title.lower())[:40]
    title_counts[title_slug] = title_counts.get(title_slug, 0) + 1
    reinforced_tags = []
    for tag in tags:
        tag = tag.lower().strip()
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if tag_counts[tag] == REINFORCE_THRESHOLD:
            reinforced_tags.append(tag)
    if title_counts.get(title_slug, 0) == REINFORCE_THRESHOLD:
        _write_reinforcement_summary(title, tags, summary_md, reason="title")
    for tag in reinforced_tags:
        _write_reinforcement_summary(title, [tag], summary_md, reason=f"tag:{tag}")


def _write_reinforcement_summary(title: str, tags: list[str], summary_md: str, reason: str) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    filename  = datetime.now().strftime("%Y-%m-%d") + ".md"
    file_path = KNOWLEDGE_DIR / filename
    tags_line = " ".join(f"#reinforced #{t}" for t in tags[:6])
    now   = datetime.now()
    block = (
        f"\n## [REINFORCED] {title} ({now.strftime('%H:%M')})\n"
        f"{tags_line}\n\n"
        f"> This topic has appeared {REINFORCE_THRESHOLD}+ times — flagged as important.\n\n"
        f"{summary_md}\n\n---\n"
    )
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(block)
        log.info("Reinforcement summary written: %s", reason)
    except Exception as exc:
        log.warning("Could not write reinforcement summary: %s", exc)


def save_knowledge_entry(entry: dict, chroma_upsert_fn=None) -> bool:
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
    if confidence < KNOWLEDGE_CONFIDENCE_THRESHOLD:
        return False
    tags_clean  = [str(t).strip() for t in tags if str(t).strip()][:12]
    mem_type    = resolve_memory_type(tags_clean, title)
    target_file = MEMORY_TYPE_FILES.get(mem_type, MEMORY_TYPE_FILES["general"])
    target_file.parent.mkdir(parents=True, exist_ok=True)
    entry_id    = hashlib.sha1((title + "\n" + summary_md).encode("utf-8")).hexdigest()[:12]
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
    index = load_knowledge_index()
    ids   = index.setdefault("ids", [])
    if entry_id not in ids:
        ids.append(entry_id)
    check_and_reinforce(index, tags_clean, title, summary_md)
    save_knowledge_index(index)
    score_memory_entry(entry_id, title)
    update_knowledge_graph(tags_clean, title, summary_md)
    if chroma_upsert_fn:
        try:
            chroma_upsert_fn(
                entry_id=entry_id,
                text=(title + "\n" + summary_md)[:2000],
                metadata={"memory_type": mem_type, "title": title},
            )
        except Exception as exc:
            log.warning("Chroma upsert failed: %s", exc)
    return True


def forget_keyword(keyword: str, chroma_delete_fn=None) -> str:
    kw = keyword.strip().lower()
    if not kw:
        return "Please provide a keyword to forget."
    chroma_deleted = chroma_delete_fn(kw) if chroma_delete_fn else 0
    md_edits = 0
    for mem_type, filepath in MEMORY_TYPE_FILES.items():
        if not filepath.exists():
            continue
        try:
            content  = filepath.read_text(encoding="utf-8")
            sections = re.split(r"(?=\n## )", content)
            kept     = [s for s in sections if kw not in s.lower()]
            if len(kept) != len(sections):
                filepath.write_text("".join(kept), encoding="utf-8")
                md_edits += len(sections) - len(kept)
        except Exception as exc:
            log.warning("forget_keyword: could not edit %s: %s", filepath, exc)
    if chroma_deleted == 0 and md_edits == 0:
        return f"No memories found containing '{keyword}'."
    return (f"Forgot {md_edits} markdown block(s) and {chroma_deleted} "
            f"vector entry/entries related to '{keyword}'.")


def get_conversation_summary() -> str:
    if not CONVERSATION_SUMMARY_FILE.exists():
        return ""
    try:
        content = CONVERSATION_SUMMARY_FILE.read_text(encoding="utf-8")
        blocks  = content.strip().split("---")
        recent  = [b.strip() for b in blocks if b.strip()][-3:]
        return "\n\n".join(recent)
    except Exception:
        return ""


def compress_conversation(turns_for_summary: list[dict], generate_fn) -> None:
    if len(turns_for_summary) < CONTEXT_COMPRESSION_INTERVAL:
        return
    turns = turns_for_summary[:]
    conv_text = "\n".join(f"User: {t['user']}\nMiku: {t['miku']}" for t in turns)
    prompt = (
        "Compress the following conversation into a concise summary (max 200 words).\n"
        "Focus on: decisions made, facts established, emotional tone, tasks discussed.\n"
        "Write in third-person narrative style.\n\n"
        f"Conversation:\n{conv_text}"
    )
    try:
        summary = generate_fn(prompt)
        if not summary.strip():
            return
        VAULT_PATH.mkdir(parents=True, exist_ok=True)
        existing = ""
        if CONVERSATION_SUMMARY_FILE.exists():
            existing = CONVERSATION_SUMMARY_FILE.read_text(encoding="utf-8")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_block = f"\n## Summary at {ts}\n\n{summary.strip()}\n\n---\n"
        CONVERSATION_SUMMARY_FILE.write_text(existing + new_block, encoding="utf-8")
        log.info("[Context] Conversation compressed at %d turns.", len(turns))
    except Exception as exc:
        log.debug("compress_conversation: %s", exc)


def load_curiosity_state() -> dict:
    if CURIOSITY_FILE.exists():
        try:
            return json.loads(CURIOSITY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_question_date": {}, "asked_topics": []}


def save_curiosity_state(state: dict) -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        CURIOSITY_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save curiosity state: %s", exc)


def maybe_generate_curiosity_question(generate_fn) -> str | None:
    import random
    rel    = load_relationship()
    scores = high_confidence_memories(min_score=CURIOSITY_MIN_CONFIDENCE)
    topics = rel.get("favorite_topics", [])
    if not topics and not scores:
        return None
    curiosity  = load_curiosity_state()
    last_dates = curiosity.get("last_question_date", {})
    today_str  = date.today().isoformat()
    min_date   = (date.today() - timedelta(days=CURIOSITY_MIN_DAYS_BETWEEN)).isoformat()
    eligible   = []
    for topic in topics[:8]:
        last = last_dates.get(topic, "2000-01-01")
        if last <= min_date:
            eligible.append(topic)
    if not eligible:
        return None
    topic     = random.choice(eligible[:3])
    graph_ctx = graph_context_for(topic)
    prompt = (
        "You are Miku, a curious and caring AI companion.\n"
        "Generate a single natural follow-up question about a topic the user previously mentioned.\n"
        "Sound genuinely curious, not like a chatbot running a script.\n"
        "Keep it to one sentence.\n\n"
        f"Topic: {topic}\n"
        f"{graph_ctx}\n"
        f"User's interests: {', '.join(topics[:5])}\n"
    )
    try:
        question = generate_fn(prompt).strip()
        if question:
            curiosity["last_question_date"][topic] = today_str
            asked = curiosity.get("asked_topics", [])
            asked.append(topic)
            curiosity["asked_topics"] = asked[-50:]
            save_curiosity_state(curiosity)
            return question
    except Exception as exc:
        log.debug("maybe_generate_curiosity_question: %s", exc)
    return None


def read_journal(days_back: int = 7) -> str:
    if not JOURNAL_DIR.exists():
        return ""
    entries = []
    for i in range(days_back):
        d    = date.today() - timedelta(days=i)
        path = JOURNAL_DIR / f"{d.isoformat()}.md"
        if path.exists():
            try:
                entries.append(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return "\n\n---\n\n".join(entries[:3])


def write_journal_entry(generate_fn) -> None:
    if not JOURNAL_ENABLED:
        return
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    today_file = JOURNAL_DIR / f"{date.today().isoformat()}.md"
    if today_file.exists():
        return
    rel    = load_relationship()
    tasks  = load_active_task()
    goals  = [g["goal"] for g in load_goals() if g.get("status") == "active"]
    topics = rel.get("favorite_topics", [])[:5]
    count  = rel.get("interaction_count", 0)
    task_note = f"Active task: {tasks.get('goal', 'none')}." if tasks else "No active tasks today."
    goal_note = "Goals in progress: " + ", ".join(goals) + "." if goals else "No tracked goals yet."
    summary_ctx = get_conversation_summary()
    prompt = (
        "Write a brief journal entry for Miku, an AI companion, reflecting on today.\n"
        "Write in first person as Miku. Be warm, observational, personal.\n"
        "Keep it under 150 words. No headers.\n\n"
        f"Context:\n"
        f"- Total interactions with user: {count}\n"
        f"- User's recurring interests: {', '.join(topics) if topics else 'still learning'}\n"
        f"- {task_note}\n"
        f"- {goal_note}\n"
        f"- Recent conversation context:\n{summary_ctx[:400] if summary_ctx else 'nothing yet today'}\n"
    )
    try:
        entry = generate_fn(prompt)
        if not entry.strip():
            return
        content = f"# Journal — {date.today().isoformat()}\n\n{entry.strip()}\n"
        today_file.write_text(content, encoding="utf-8")
        log.info("[Journal] Entry written for %s.", date.today().isoformat())
    except Exception as exc:
        log.debug("write_journal_entry: %s", exc)


def run_reflection(generate_fn) -> None:
    rel     = load_relationship()
    topics  = rel.get("favorite_topics", [])[:8]
    count   = rel.get("interaction_count", 0)
    goals   = load_goals()
    summary = get_conversation_summary()
    journal = read_journal(days_back=3)
    prompt = (
        "You are Miku, a personal AI companion. Write a private reflection after reviewing your interactions.\n\n"
        "Notice:\n"
        "- Recurring patterns in the user's focus\n"
        "- Emotional trends\n"
        "- Ways you could be more helpful\n"
        "- What the user seems to be working toward\n\n"
        "Format as markdown with sections: Patterns, Emotional Tone, Support Opportunities.\n"
        "Be genuine and specific, not generic. Max 300 words.\n\n"
        f"Context:\n"
        f"- Interactions: {count}\n"
        f"- Favourite topics: {', '.join(topics)}\n"
        f"- Active goals: {json.dumps([g['goal'] for g in goals if g.get('status')=='active'])}\n"
        f"- Recent summaries:\n{summary[:600]}\n"
        f"- Recent journal:\n{journal[:400]}\n"
    )
    try:
        reflection = generate_fn(prompt)
        if not reflection.strip():
            return
        VAULT_PATH.mkdir(parents=True, exist_ok=True)
        existing = ""
        if REFLECTION_FILE.exists():
            existing = REFLECTION_FILE.read_text(encoding="utf-8")
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = f"\n## Reflection — {ts}\n\n{reflection.strip()}\n\n---\n"
        REFLECTION_FILE.write_text(existing + block, encoding="utf-8")
        log.info("[Reflection] Written at %s.", ts)
    except Exception as exc:
        log.debug("run_reflection: %s", exc)


def update_life_story(generate_fn) -> None:
    rel   = load_relationship()
    count = rel.get("interaction_count", 0)
    if count < 5 or count % 20 != 0:
        return
    existing = ""
    if LIFE_STORY_FILE.exists():
        existing = LIFE_STORY_FILE.read_text(encoding="utf-8")
    journal  = read_journal(days_back=14)
    summary  = get_conversation_summary()
    topics   = rel.get("favorite_topics", [])
    goals    = [g["goal"] for g in load_goals()]
    first    = rel.get("first_seen", "unknown date")
    prompt = (
        "Write or update a life story narrative for Miku's relationship with the user.\n"
        "This is a compressed third-person narrative of their shared history.\n"
        "Weave together topics, projects, goals, and emotional journey.\n"
        "Keep it under 400 words. Write in flowing prose, not bullet points.\n\n"
        f"First interaction: {first}\n"
        f"Total interactions: {count}\n"
        f"Recurring interests: {', '.join(topics)}\n"
        f"Goals pursued: {', '.join(goals[:5])}\n"
        f"Recent context:\n{summary[:500]}\n"
        f"Recent journals:\n{journal[:500]}\n"
        f"Existing story (update/extend it):\n{existing[:800]}\n"
    )
    try:
        story = generate_fn(prompt)
        if story.strip():
            VAULT_PATH.mkdir(parents=True, exist_ok=True)
            LIFE_STORY_FILE.write_text(
                f"# Life Story\n\n_Last updated: {date.today().isoformat()}_\n\n{story.strip()}\n",
                encoding="utf-8"
            )
            log.info("[LifeStory] Updated.")
    except Exception as exc:
        log.debug("update_life_story: %s", exc)
