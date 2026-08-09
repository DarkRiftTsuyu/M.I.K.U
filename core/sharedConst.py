import os

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
MODEL            = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "60"))

AUTO_SAVE_KNOWLEDGE            = os.environ.get("AUTO_SAVE_KNOWLEDGE", "1") not in ("0", "false", "False")
KNOWLEDGE_MODEL                = os.environ.get("KNOWLEDGE_MODEL", MODEL)
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

EVOLUTION_ENABLED     = os.environ.get("EVOLUTION_ENABLED", "1") not in ("0", "false", "False")
EVOLUTION_THRESHOLD   = int(os.environ.get("EVOLUTION_THRESHOLD", "5"))
EVOLUTION_CODER_MODEL = os.environ.get("EVOLUTION_CODER_MODEL", MODEL)
EVOLUTION_AUTO_TEST   = os.environ.get("EVOLUTION_AUTO_TEST", "1") not in ("0", "false", "False")

_TAG_TO_TYPE: dict[str, str] = {
    "preference": "preferences", "preferences": "preferences",
    "style": "preferences", "likes": "preferences", "dislikes": "preferences",
    "project": "projects", "projects": "projects", "task": "projects",
    "goal": "projects", "deadline": "projects",
    "fact": "user_facts", "user": "user_facts", "name": "user_facts",
    "age": "user_facts", "location": "user_facts", "job": "user_facts",
    "background": "user_facts",
}

PLUGINS = {}