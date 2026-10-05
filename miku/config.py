from __future__ import annotations

import os
from pathlib import Path

MAX_UNPROMPTED_PER_DAY  = int(os.environ.get("MAX_UNPROMPTED_PER_DAY", "6"))

TTS_ENABLED     = os.environ.get("TTS_ENABLED", "1") not in ("0", "false", "False")
TTS_QUEUE_MAX   = int(os.environ.get("TTS_QUEUE_MAX", "8"))
TTS_DEBUG       = os.environ.get("TTS_DEBUG", "0") in ("1", "true", "True")
TTS_MAX_CHARS   = int(os.environ.get("TTS_MAX_CHARS", "350"))

KITTEN_DIR         = os.environ.get("KITTEN_DIR", "./kitten-mini-en-v0_1-fp16")
KITTEN_MODEL       = os.environ.get("KITTEN_MODEL", "model.fp16.onnx")
KITTEN_VOICES      = os.environ.get("KITTEN_VOICES", "voices.bin")
KITTEN_TOKENS      = os.environ.get("KITTEN_TOKENS", "tokens.txt")
KITTEN_DATA_DIR    = os.environ.get("KITTEN_DATA_DIR", "espeak-ng-data")
KITTEN_SID         = int(os.environ.get("KITTEN_SID", "7"))
KITTEN_SPEED       = float(os.environ.get("KITTEN_SPEED", "1.2"))
KITTEN_NUM_THREADS = int(os.environ.get("KITTEN_NUM_THREADS", "2"))

OLLAMA_HOST      = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
FAST_MODEL       = os.environ.get("OLLAMA_FAST_MODEL", "llama3")
MODEL            = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "60"))

AUTO_SAVE_KNOWLEDGE            = os.environ.get("AUTO_SAVE_KNOWLEDGE", "1") not in ("0", "false", "False")
KNOWLEDGE_MODEL                = os.environ.get("KNOWLEDGE_MODEL", FAST_MODEL)
KNOWLEDGE_CONFIDENCE_THRESHOLD = float(os.environ.get("KNOWLEDGE_CONFIDENCE_THRESHOLD", "0.5"))

SEARCH_BACKEND      = os.environ.get("SEARCH_BACKEND", "chroma")
REINFORCE_THRESHOLD = int(os.environ.get("REINFORCE_THRESHOLD", "3"))

WAKE_WORD          = os.environ.get("WAKE_WORD", "miku").lower()
WHISPER_MODEL      = os.environ.get("WHISPER_MODEL", "tiny.en")
STT_SAMPLE_RATE    = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
STT_CHUNK_SECONDS  = float(os.environ.get("STT_CHUNK_SECONDS", "1.5"))
STT_POST_WAKE_SECS = float(os.environ.get("STT_POST_WAKE_SECS", "6.0"))
STT_INPUT_DEVICE   = os.environ.get("STT_INPUT_DEVICE", "").strip()

VOICE_ENABLED_DEFAULT = os.environ.get("VOICE_ENABLED", "0") in ("1", "true", "True")

VAD_ENABLED   = os.environ.get("VAD_ENABLED", "1") not in ("0", "false", "False")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "700"))

TRAY_ENABLED       = os.environ.get("TRAY_ENABLED", "0") in ("1", "true", "True")

GMAIL_ENABLED    = os.environ.get("GMAIL_ENABLED", "0") in ("1", "true", "True")
CALENDAR_ENABLED = os.environ.get("CALENDAR_ENABLED", "0") in ("1", "true", "True")
DISCORD_ENABLED  = os.environ.get("DISCORD_ENABLED", "0") in ("1", "true", "True")

DESKTOP_NOTIFY_REMINDERS = os.environ.get("DESKTOP_NOTIFY_REMINDERS", "1") not in ("0", "false", "False")

EVOLUTION_ENABLED     = os.environ.get("EVOLUTION_ENABLED", "1") not in ("0", "false", "False")
EVOLUTION_THRESHOLD   = int(os.environ.get("EVOLUTION_THRESHOLD", "5"))
EVOLUTION_CODER_MODEL = os.environ.get("EVOLUTION_CODER_MODEL", MODEL)
EVOLUTION_AUTO_TEST   = os.environ.get("EVOLUTION_AUTO_TEST", "1") not in ("0", "false", "False")

CONTEXT_COMPRESSION_INTERVAL = int(os.environ.get("CONTEXT_COMPRESSION_INTERVAL", "15"))
CURIOSITY_MIN_CONFIDENCE     = float(os.environ.get("CURIOSITY_MIN_CONFIDENCE", "0.7"))
CURIOSITY_MIN_DAYS_BETWEEN   = int(os.environ.get("CURIOSITY_MIN_DAYS_BETWEEN", "3"))
REFLECTION_HOUR              = int(os.environ.get("REFLECTION_HOUR", "19"))
JOURNAL_ENABLED              = os.environ.get("JOURNAL_ENABLED", "1") not in ("0", "false", "False")

LAST_N_MESSAGES      = int(os.environ.get("LAST_N_MESSAGES", "10"))
MEMORY_MAX_CHARS     = int(os.environ.get("MEMORY_MAX_CHARS", "900"))
TOOL_FAILURE_TTL_S   = int(os.environ.get("TOOL_FAILURE_TTL_S", "600"))

BASE_DIR  = Path(__file__).resolve().parent.parent
VAULT_PATH = BASE_DIR / "vault"

IDENTITY_FILE             = VAULT_PATH / "identity" / "miku.md"
MEMORY_DIR                = VAULT_PATH / "memory" / "conversations"
KNOWLEDGE_DIR             = VAULT_PATH / "knowledge"
KNOWLEDGE_INDEX_FILE      = KNOWLEDGE_DIR / "_index.json"
STATE_FILE                = VAULT_PATH / "state.json"
REMINDERS_FILE            = VAULT_PATH / "reminders.json"
ACTIVE_TASK_FILE          = VAULT_PATH / "active_task.json"
CAPABILITIES_FILE         = VAULT_PATH / "capabilities.json"
RELATIONSHIP_FILE         = VAULT_PATH / "relationship.json"
JOURNAL_DIR               = VAULT_PATH / "journal"
REFLECTION_FILE           = VAULT_PATH / "reflection.md"
LIFE_STORY_FILE           = VAULT_PATH / "life_story.md"
GOALS_FILE                = VAULT_PATH / "goals.json"
KNOWLEDGE_GRAPH_FILE      = VAULT_PATH / "knowledge_graph.json"
CURIOSITY_FILE            = VAULT_PATH / "curiosity_state.json"
CONVERSATION_SUMMARY_FILE = VAULT_PATH / "conversation_summary.md"
CONVERSATION_STATE_FILE   = VAULT_PATH / "conversation_state.json"
INTERNAL_NOTES_FILE       = VAULT_PATH / "internal_notes.json"
RECENT_TURNS_FILE         = VAULT_PATH / "recent_turns.jsonl"
TOOL_CONTEXT_FILE         = VAULT_PATH / "tool_context.json"
HABIT_TRACKER_FILE        = VAULT_PATH / "habit_tracker.json"
INTEGRATIONS_CONFIG_FILE  = VAULT_PATH / "integrations.json"
PENDING_CONFIRMATION_FILE = VAULT_PATH / "pending_confirmation.json"

EVOLUTION_DIR          = VAULT_PATH / "evolution"
CAPABILITY_REQUESTS    = EVOLUTION_DIR / "capability_requests.json"
EVOLUTION_PENDING_DIR  = EVOLUTION_DIR / "pending"
EVOLUTION_SANDBOX_DIR  = EVOLUTION_DIR / "sandbox"
EVOLUTION_METRICS_FILE = EVOLUTION_DIR / "metrics.json"

MEMORY_TYPE_FILES: dict[str, Path] = {
    "user_facts":  KNOWLEDGE_DIR / "user_facts.md",
    "projects":    KNOWLEDGE_DIR / "projects.md",
    "preferences": KNOWLEDGE_DIR / "preferences.md",
    "general":     KNOWLEDGE_DIR / "general.md",
}

TAG_TO_TYPE: dict[str, str] = {
    "preference": "preferences", "preferences": "preferences",
    "style": "preferences", "likes": "preferences", "dislikes": "preferences",
    "project": "projects", "projects": "projects", "task": "projects",
    "goal": "projects", "deadline": "projects",
    "fact": "user_facts", "user": "user_facts", "name": "user_facts",
    "age": "user_facts", "location": "user_facts", "job": "user_facts",
    "background": "user_facts",
}
