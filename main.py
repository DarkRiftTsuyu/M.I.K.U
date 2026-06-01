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

if TYPE_CHECKING:
    import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("miku")

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

BASE_DIR             = Path(__file__).resolve().parent
VAULT_PATH           = BASE_DIR / "vault"
IDENTITY_FILE        = VAULT_PATH / "identity" / "miku.md"
MEMORY_DIR           = VAULT_PATH / "memory" / "conversations"
KNOWLEDGE_DIR        = VAULT_PATH / "knowledge"
KNOWLEDGE_INDEX_FILE = KNOWLEDGE_DIR / "_index.json"
STATE_FILE           = VAULT_PATH / "state.json"
REMINDERS_FILE       = VAULT_PATH / "reminders.json"

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
        obj = json.loads(plan)
        if not obj:
            return None
        tool = obj["tool"]
        args = obj.get("args", {})
        if tool in PLUGINS:
            return PLUGINS[tool].execute(args)
    except Exception as e:
        log.warning(f"Tool execution failed: {e}")
    return None

_state_lock = threading.Lock()

_user_state: dict = {"mood": "neutral", "energy": "normal", "last_update": None}
_last_user_time:  float = time.time()
_last_user_message: str = ""
_last_miku_spoke: float = 0.0
_tts_enabled_flag: bool = TTS_ENABLED

_voice_enabled_flag: bool = VOICE_ENABLED_DEFAULT
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
        if count >= MAX_UNPROMPTED_PER_DAY:
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

def _urljoin(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")

def _http_json(method: str, url: str, payload=None, timeout_s: float | None = None):
    data = None
    headers = {"Accept": "application/json"}
    if payload:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    timeout = timeout_s if timeout_s is not None else OLLAMA_TIMEOUT_S
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {e.code} calling {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e

def load_identity() -> str:
    if IDENTITY_FILE.exists():
        return IDENTITY_FILE.read_text(encoding="utf-8")
    return "You are Hatsune Miku, a friendly and warm personal assistant."

_tts_worker_instance: Optional["_TTSWorker"] = None
_tts_available   = False
_tts_warned      = False

def _tts_log(msg: str) -> None:
    if TTS_DEBUG:
        log.debug("[TTS] %s", msg)

def _tts_init_if_needed() -> bool:
    global _tts_available, _tts_warned
    if not _tts_enabled_flag:
        return False
    if _tts_available:
        return True
    if _tts_warned:
        return False
    try:
        import sherpa_onnx
        import soundfile
        base = Path(KITTEN_DIR)
        if not base.is_absolute():
            base = (BASE_DIR / base).resolve()
        if not base.exists():
            _tts_warned = True
            log.warning("[TTS] Kitten directory not found: %s", base)
            return False
        missing = [p for p in (base/KITTEN_MODEL, base/KITTEN_VOICES,
                                base/KITTEN_TOKENS, base/KITTEN_DATA_DIR)
                   if not p.exists()]
        if missing:
            _tts_warned = True
            log.warning("[TTS] Missing files: %s", missing)
            return False
        _tts_available = True
        return True
    except ImportError:
        _tts_warned = True
        log.info("[TTS] sherpa-onnx not available — running without speech. "
                 "Install with: pip install sherpa-onnx soundfile")
        return False
    except Exception as exc:
        _tts_warned = True
        log.warning("[TTS] Init check failed: %s", exc)
        return False


def _find_powershell_exe() -> str | None:
    for name in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
        found = shutil.which(name)
        if found:
            return found
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    legacy = os.path.join(sysroot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return legacy if os.path.exists(legacy) else None


class _TTSWorker:
    def __init__(self) -> None:
        self.q: queue.Queue[str | None] = queue.Queue(maxsize=max(1, TTS_QUEUE_MAX))
        self.thread = threading.Thread(target=self._run, name="TTSWorker", daemon=True)
        self._tts = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.thread.start()

    def enqueue(self, text: str) -> None:
        if not text or not _tts_enabled_flag:
            return
        try:
            self.q.put_nowait(text)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(text)
            except Exception:
                pass

    def drain(self) -> None:
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def _init_tts(self) -> None:
        import sherpa_onnx
        base = Path(KITTEN_DIR)
        if not base.is_absolute():
            base = (BASE_DIR / base).resolve()
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
                    model=str((base / KITTEN_MODEL).resolve()),
                    voices=str((base / KITTEN_VOICES).resolve()),
                    tokens=str((base / KITTEN_TOKENS).resolve()),
                    data_dir=str((base / KITTEN_DATA_DIR).resolve()),
                ),
                num_threads=max(1, KITTEN_NUM_THREADS),
            ),
        )
        if not config.validate():
            raise ValueError("Invalid sherpa_onnx OfflineTtsConfig")
        self._tts = sherpa_onnx.OfflineTts(config)

    def _play_wav(self, path: str) -> None:
        if os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME)
                return
            except Exception as exc:
                _tts_log(f"winsound failed: {exc}")
            ps_exe = _find_powershell_exe()
            if not ps_exe:
                raise RuntimeError("No PowerShell found for WAV playback")
            escaped = path.replace("'", "''")
            ps = (f"$p='{escaped}'; $sp=New-Object System.Media.SoundPlayer($p);"
                  "$sp.Load(); $sp.PlaySync();")
            result = subprocess.run(
                [ps_exe, "-NoProfile", "-STA", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE if TTS_DEBUG else subprocess.DEVNULL,
                text=True, check=False,
            )
            if TTS_DEBUG and result.stderr:
                _tts_log(f"PowerShell stderr: {result.stderr.strip()}")
        else:
            for player, args in [
                ("aplay",  [path]),
                ("afplay", [path]),
                ("ffplay", ["-nodisp", "-autoexit", path]),
            ]:
                if shutil.which(player):
                    subprocess.run([player] + args,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   check=False)
                    return
            try:
                import sounddevice as sd
                import soundfile as sf
                data, sr = sf.read(path)
                sd.play(data, sr)
                sd.wait()
            except Exception as exc:
                _tts_log(f"sounddevice playback failed: {exc}")

    def _kitten_speak(self, clean_text: str) -> None:
        if not self._tts:
            return
        text = clean_text.strip()
        if not text:
            return
        chunks: list[str] = []
        while len(text) > TTS_MAX_CHARS:
            cut = text.rfind(" ", 0, TTS_MAX_CHARS)
            if cut < 80:
                cut = TTS_MAX_CHARS
            chunks.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            chunks.append(text)

        for part in chunks:
            if not _tts_enabled_flag:
                return
            import soundfile as sf
            audio = self._tts.generate(
                text=part,
                sid=int(KITTEN_SID),
                speed=float(max(0.1, min(3.0, KITTEN_SPEED))),
            )
            if audio is None or getattr(audio, "samples", None) is None:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            try:
                sf.write(wav_path, audio.samples, samplerate=int(audio.sample_rate))
                self._play_wav(wav_path)
            finally:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    def _run(self) -> None:
        global _tts_warned
        try:
            self._init_tts()
            _tts_log("Kitten ready")
        except Exception as exc:
            _tts_warned = True
            _tts_log(f"Kitten init failed: {exc}")
            return

        while True:
            text = self.q.get()
            if text is None:
                return
            if not _tts_enabled_flag:
                continue
            try:
                _tts_log(f"Speaking {min(len(text), 60)} chars…")
                clean = _tts_sanitize(text)
                if clean:
                    self._kitten_speak(clean)
                _tts_log("Done")
            except Exception as exc:
                log.warning("[TTS] Speak error: %s", exc)
                try:
                    self._init_tts()
                    _tts_log("Kitten reinitialised")
                except Exception as exc2:
                    log.error("[TTS] Kitten reinit failed: %s", exc2)
                    return


def _tts_sanitize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_tts_worker() -> _TTSWorker:
    global _tts_worker_instance
    if _tts_worker_instance is None or not _tts_worker_instance.thread.is_alive():
        _tts_worker_instance = _TTSWorker()
        _tts_worker_instance.start()
    return _tts_worker_instance


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def speak(text: str) -> None:
    if not _tts_init_if_needed():
        return
    worker = _get_tts_worker()
    worker.enqueue(text)


def speak_sentence(fragment: str) -> None:
    if not _tts_init_if_needed():
        return
    clean = _tts_sanitize(fragment)
    if clean:
        _get_tts_worker().enqueue(clean)


_stt_available = False
_stt_warned    = False
_whisper_model_instance = None
_stt_input_device_index: int | None = None
_stt_input_device_name: str = ""
_stt_device_default_sr: int | None = None
_stt_last_record_warn: float = 0.0
_stt_record_failures: int = 0


def _stt_warn_rate_limited(message: str) -> None:
    global _stt_last_record_warn, _stt_record_failures
    _stt_record_failures += 1
    now = time.time()
    if (now - _stt_last_record_warn) >= 5.0:
        _stt_last_record_warn = now
        log.warning("[STT] Record failed (%d): %s", _stt_record_failures, message)


def _ensure_sounddevice_input(sd) -> int | None:
    global _stt_input_device_index, _stt_input_device_name
    global _stt_device_default_sr

    try:
        devices = sd.query_devices()
    except Exception:
        return None

    if not devices:
        return None

    def is_input(idx: int) -> bool:
        try:
            d = sd.query_devices(idx)
            return int(d.get("max_input_channels", 0)) > 0
        except Exception:
            return False

    want = STT_INPUT_DEVICE
    selected: int | None = None

    if want:
        if want.isdigit():
            idx = int(want)
            if 0 <= idx < len(devices) and is_input(idx):
                selected = idx
        else:
            needle = want.lower()
            for idx in range(len(devices)):
                if not is_input(idx):
                    continue
                name = str(devices[idx].get("name", "")).lower()
                if needle in name:
                    selected = idx
                    break

    if selected is None:
        try:
            default_in = sd.default.device[0]
        except Exception:
            default_in = None
        if isinstance(default_in, int) and default_in >= 0 and is_input(default_in):
            selected = default_in

    if selected is None:
        for idx in range(len(devices)):
            if is_input(idx):
                selected = idx
                break

    if selected is None:
        return None

    try:
        cur = sd.default.device
        try:
            out_dev = cur[1]
        except Exception:
            out_dev = None
        sd.default.device = (selected, out_dev)
    except Exception:
        pass

    _stt_input_device_index = selected
    try:
        info = sd.query_devices(selected)
        _stt_input_device_name = str(info.get("name", ""))
        try:
            _stt_device_default_sr = int(float(info.get("default_samplerate", 0)) or 0) or None
        except Exception:
            _stt_device_default_sr = None
    except Exception:
        _stt_input_device_name = ""
    return selected


def _resample_audio(audio: "np.ndarray", orig_sr: int, target_sr: int) -> "np.ndarray":
    if orig_sr <= 0 or target_sr <= 0 or orig_sr == target_sr:
        return audio
    try:
        import numpy as np
        if audio.size == 0:
            return audio
        duration = audio.size / float(orig_sr)
        n_out = max(1, int(round(duration * target_sr)))
        x_old = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False, dtype=np.float64)
        y_new = np.interp(x_new, x_old, audio.astype(np.float64, copy=False)).astype(np.float32)
        return y_new
    except Exception:
        return audio


def _stt_init_if_needed() -> bool:
    global _stt_available, _stt_warned, _whisper_model_instance
    if _stt_available:
        return True
    if _stt_warned:
        return False
    try:
        from faster_whisper import WhisperModel
        import sounddevice as sd
        import numpy

        mic = _ensure_sounddevice_input(sd)
        if mic is None:
            _stt_warned = True
            log.info(
                "[STT] No usable input audio device found — wake word disabled. "
                "Configure a microphone in Windows or set STT_INPUT_DEVICE (index or name substring)."
            )
            return False

        _whisper_model_instance = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8"
        )
        _stt_available = True
        mic_label = _stt_input_device_name or str(mic)
        log.info("[STT] faster-whisper ready (model=%s, mic=%s)", WHISPER_MODEL, mic_label)
        return True
    except ImportError:
        _stt_warned = True
        log.info("[STT] faster-whisper or sounddevice not available. "
                 "Install with: pip install faster-whisper sounddevice numpy")
        return False
    except Exception as exc:
        _stt_warned = True
        log.warning("[STT] Init failed: %s", exc)
        return False


def _record_audio(duration_s: float) -> "np.ndarray | None":
    try:
        import sounddevice as sd
        import numpy as np

        if _stt_input_device_index is None or (isinstance(sd.default.device, (list, tuple)) and sd.default.device[0] in (None, -1)):
            _ensure_sounddevice_input(sd)

        record_sr = int(STT_SAMPLE_RATE)
        if _stt_device_default_sr:
            record_sr = int(_stt_device_default_sr)

        frames = int(max(1.0, duration_s) * record_sr)

        try:
            sd.check_input_settings(samplerate=record_sr, channels=1, dtype="float32")
        except Exception:
            if _stt_device_default_sr and record_sr != int(_stt_device_default_sr):
                record_sr = int(_stt_device_default_sr)

        audio = sd.rec(
            frames,
            samplerate=record_sr,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        mono = audio.flatten()
        if record_sr != int(STT_SAMPLE_RATE):
            mono = _resample_audio(mono, orig_sr=record_sr, target_sr=int(STT_SAMPLE_RATE))
        return mono
    except Exception as exc:
        _stt_warn_rate_limited(str(exc))
        return None


def _transcribe(audio: "np.ndarray") -> str:
    if _whisper_model_instance is None:
        return ""
    try:
        import numpy as np
        segments, _ = _whisper_model_instance.transcribe(
            audio, language="en", beam_size=1, vad_filter=True
        )
        return " ".join(s.text for s in segments).strip()
    except Exception as exc:
        log.warning("[STT] Transcribe failed: %s", exc)
        return ""


def wake_word_loop(input_queue: queue.Queue, stop_event: threading.Event | None = None) -> None:
    if stop_event is None:
        stop_event = threading.Event()

    if not _stt_init_if_needed():
        log.info("[Wake] STT unavailable — wake word disabled.")
        return

    log.info("[Wake] Listening for wake word '%s'…", WAKE_WORD)

    while not stop_event.is_set():
        chunk = _record_audio(STT_CHUNK_SECONDS)
        if stop_event.is_set():
            break
        if chunk is None:
            time.sleep(0.5)
            continue

        text = _transcribe(chunk).lower()
        if WAKE_WORD not in text:
            continue

        log.info("[Wake] Wake word detected!")
        print(f"\n[Miku heard her name — listening for {STT_POST_WAKE_SECS}s…]")

        speak("Yes?")

        command_audio = _record_audio(STT_POST_WAKE_SECS)
        if stop_event.is_set():
            break
        if command_audio is None:
            continue

        command_text = _transcribe(command_audio)
        if command_text:
            print(f"\n[Voice] {command_text}")
            input_queue.put(command_text)


def _set_voice_enabled(enabled: bool, voice_input_queue: queue.Queue | None = None) -> None:
    global _voice_enabled_flag, _wake_word_stop_event, _wake_word_thread

    enabled = bool(enabled)
    if enabled == _voice_enabled_flag and (not enabled or (_wake_word_thread and _wake_word_thread.is_alive())):
        state = "ON" if _voice_enabled_flag else "OFF"
        print(f"[Voice] {state}")
        return

    if not enabled:
        _voice_enabled_flag = False
        if _wake_word_stop_event is not None:
            _wake_word_stop_event.set()
        print("[Voice] OFF")
        return

    if voice_input_queue is None:
        print("[Voice] Error: voice input queue not ready yet.")
        return

    if not _stt_init_if_needed():
        print("[Voice] STT unavailable. Install faster-whisper + sounddevice + numpy, and ensure a microphone is configured.")
        _voice_enabled_flag = False
        return

    _voice_enabled_flag = True
    _wake_word_stop_event = threading.Event()
    _wake_word_thread = threading.Thread(
        target=wake_word_loop,
        args=(voice_input_queue, _wake_word_stop_event),
        daemon=True,
    )
    _wake_word_thread.start()
    print(f"[Voice] ON  (wake word: '{WAKE_WORD}')")


_APP_REGISTRY: list[tuple[list[str], str, str]] = [
    (["cider", "cider.exe"],          "Cider",              "Cider (Apple Music client)"),
    (["spotify", "spotify.exe"],      "Spotify",            "Spotify"),
    (["discord", "discord.exe"],      "Discord",            "Discord"),
    (["code", "code.exe", "vscode"],  "Visual Studio Code", "VS Code"),
    (["firefox", "firefox.exe"],      "Firefox",            "Firefox"),
    (["chrome", "google-chrome",
      "google-chrome-stable"],        "Google Chrome",      "Chrome"),
    (["steam", "steam.exe"],          "Steam",              "Steam"),
    (["obs", "obs-studio"],           "OBS",                "OBS Studio"),
    (["vlc"],                         "VLC",                "VLC"),
    (["slack", "slack.exe"],          "Slack",              "Slack"),
    (["notion", "notion.exe"],        "Notion",             "Notion"),
    (["obsidian", "obsidian.exe"],    "Obsidian",           "Obsidian"),
    (["terminal", "gnome-terminal",
      "konsole", "xterm"],            "Terminal",           "Terminal"),
    (["notepad", "gedit", "kate"],    "",                   "Text editor"),
    (["explorer", "nautilus",
      "thunar", "dolphin"],           "",                   "File manager"),
]

_OPEN_PATTERNS = re.compile(
    r"\b(?:open|launch|start|run|can\s+you\s+open|please\s+open)\s+(.+)",
    re.IGNORECASE,
)


def _launch_app(app_name: str) -> str:
    target = app_name.strip().lower()

    matched_exes: list[str] = []
    matched_bundle: str = ""
    matched_label: str = target

    for exes, bundle, label in _APP_REGISTRY:
        if any(target in exe or exe in target for exe in exes):
            matched_exes = exes
            matched_bundle = bundle
            matched_label = label
            break

    if not matched_exes:
        matched_exes = [target]
        matched_label = target

    if sys.platform == "darwin" and matched_bundle:
        result = subprocess.run(
            ["open", "-a", matched_bundle],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return f"Opened {matched_label}!"

    for exe in matched_exes:
        found = shutil.which(exe)
        if found:
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        [found],
                        creationflags=subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                else:
                    subprocess.Popen([found],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     start_new_session=True)
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

    return (f"I couldn't find {matched_label} on your system. "
            f"Make sure it's installed and on your PATH.")


def detect_open_intent(text: str) -> str | None:
    cleaned = re.sub(rf"\b{re.escape(WAKE_WORD)}\b[,]?\s*", "", text, flags=re.IGNORECASE).strip()
    m = _OPEN_PATTERNS.search(cleaned)
    if m:
        app = m.group(1).strip().rstrip("?. ")
        return app
    return None


_chroma_client     = None
_chroma_collection = None
_chroma_available  = False


def _init_chroma() -> bool:
    global _chroma_client, _chroma_collection, _chroma_available
    if _chroma_available:
        return True
    try:
        import chromadb
        db_path = str(VAULT_PATH / "chroma_db")
        log.info("[Chroma] Initialising at %s", db_path)
        _chroma_client = chromadb.PersistentClient(path=db_path)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="miku_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        _chroma_available = True
        log.info("[Chroma] Ready ✔")
        return True
    except Exception as exc:
        log.warning("[Chroma] Not available: %s", exc)
        return False


def rebuild_chroma_index() -> None:
    if not _chroma_available:
        log.warning("[Chroma] Cannot rebuild — Chroma not available.")
        return
    log.info("[Chroma] Rebuilding index…")
    for mem_type, file in MEMORY_TYPE_FILES.items():
        if not file.exists():
            continue
        content = file.read_text(encoding="utf-8")
        chunks = content.split("\n## ")
        for chunk in chunks:
            if not chunk.strip():
                continue
            entry_id = hashlib.sha1(chunk.encode()).hexdigest()[:12]
            _chroma_upsert(
                entry_id=entry_id,
                text=chunk[:2000],
                metadata={"memory_type": mem_type, "title": chunk.split("\n")[0][:80]},
            )
    log.info("[Chroma] Rebuild complete ✔")


def _chroma_upsert(entry_id: str, text: str, metadata: dict) -> None:
    if not _chroma_available:
        return
    try:
        _chroma_collection.upsert(ids=[entry_id], documents=[text], metadatas=[metadata])
    except Exception as exc:
        log.warning("[Chroma] Upsert failed: %s", exc)


def _chroma_search(query: str, n_results: int = 5) -> list[dict]:
    if not _chroma_available:
        return []
    try:
        results = _chroma_collection.query(query_texts=[query], n_results=n_results)
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            out.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
        return out
    except Exception as exc:
        log.warning("[Chroma] Search failed: %s", exc)
        return []


def _chroma_delete_by_keyword(keyword: str) -> int:
    if not _chroma_available:
        return 0
    try:
        results = _chroma_collection.query(query_texts=[keyword], n_results=20)
        ids_to_delete = []
        kw_lower = keyword.lower()
        for i, doc_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i].lower()
            if kw_lower in doc:
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            _chroma_collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception as exc:
        log.warning("[Chroma] Delete failed: %s", exc)
        return 0


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

    if SEARCH_BACKEND == "chroma" and _chroma_available:
        hits = _chroma_search(query, n_results=6)
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

    chroma_deleted = _chroma_delete_by_keyword(kw)

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


def _load_knowledge_index() -> dict:
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


def _save_knowledge_index(index: dict) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    index["updated"] = datetime.now().isoformat(timespec="seconds")
    try:
        KNOWLEDGE_INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save knowledge index: %s", exc)


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _resolve_memory_type(tags: list[str], title: str) -> str:
    for tag in tags:
        t = tag.lower().strip()
        if t in _TAG_TO_TYPE:
            return _TAG_TO_TYPE[t]
    title_lower = title.lower()
    for keyword, mem_type in _TAG_TO_TYPE.items():
        if keyword in title_lower:
            return mem_type
    return "general"


def _check_and_reinforce(index: dict, tags: list[str], title: str, summary_md: str) -> None:
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
    now = datetime.now()
    block = (
        f"\n## [REINFORCED] {title} ({now.strftime('%H:%M')})\n"
        f"<!-- reinforcement-reason: {reason} -->\n"
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


_TRIVIAL_MSG_RE = re.compile(r"^[\w\s]{1,12}$")

def propose_knowledge(user_input: str, assistant_response: str) -> dict | None:
    if len(user_input.strip()) < 20 or _TRIVIAL_MSG_RE.match(user_input.strip()):
        return None

    url    = _urljoin(OLLAMA_HOST, "/api/generate")
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
        resp = _http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S)
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


def save_knowledge_entry(entry: dict) -> bool:
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

    if _chroma_available:
        existing = _chroma_search(title + " " + summary_md[:200], n_results=1)
        if existing:
            doc = existing[0].get("document", "")
            title_words = set(title.lower().split())
            doc_words   = set(doc.lower().split())
            overlap = len(title_words & doc_words) / max(len(title_words), 1)
            if overlap > 0.7:
                log.debug("[knowledge] Skipping near-duplicate: %s", title)
                return False

    tags_clean = [str(t).strip() for t in tags if str(t).strip()][:12]
    mem_type   = _resolve_memory_type(tags_clean, title)
    target_file = MEMORY_TYPE_FILES.get(mem_type, MEMORY_TYPE_FILES["general"])
    target_file.parent.mkdir(parents=True, exist_ok=True)

    entry_id = hashlib.sha1((title + "\n" + summary_md).encode("utf-8")).hexdigest()[:12]

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

    index = _load_knowledge_index()
    ids   = index.setdefault("ids", [])
    if entry_id not in ids:
        ids.append(entry_id)
    _check_and_reinforce(index, tags_clean, title, summary_md)
    _save_knowledge_index(index)

    try:
        _chroma_upsert(
            entry_id=entry_id,
            text=(title + "\n" + summary_md)[:2000],
            metadata={"memory_type": mem_type, "title": title},
        )
    except Exception as exc:
        log.warning("Chroma upsert failed: %s", exc)

    return True


def _load_reminders() -> list[dict]:
    if not REMINDERS_FILE.exists():
        return []
    try:
        return json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load reminders: %s", exc)
        return []


def _save_reminders(reminders: list[dict]) -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        REMINDERS_FILE.write_text(json.dumps(reminders, indent=2), encoding="utf-8")
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
    url    = _urljoin(OLLAMA_HOST, "/api/generate")
    prompt = (
        "Classify the user's emotional state.\n\n"
        'Return STRICT JSON:\n{"mood":"happy|low|stressed|focused|bored|neutral|excited|tired",'
        '"energy":"low|normal|high"}\n\n'
        f"User message:\n<user>{user_input}</user>"
    )
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 80},
    }
    try:
        resp   = _http_json("POST", url, payload)
        parsed = _extract_json_object((resp or {}).get("response", ""))
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
    url = _urljoin(OLLAMA_HOST, "/api/generate")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.95,
                    "repeat_penalty": 1.1, "num_predict": 300},
    }
    try:
        resp = _http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S)
        return (resp or {}).get("response", "") if isinstance(resp, dict) else ""
    except Exception as exc:
        log.warning("generate_non_stream failed: %s", exc)
        return ""


def stream_generate(prompt: str, echo: bool = True) -> str:
    url = _urljoin(OLLAMA_HOST, "/api/generate")
    payload = {
        "model": MODEL,
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
                        for sentence in parts[:-1]:
                            speak_sentence(sentence)
                        sentence_buf = parts[-1]
                if obj.get("done"):
                    break
    except Exception as exc:
        log.warning("stream_generate failed: %s", exc)

    if echo:
        print()

    if sentence_buf.strip():
        speak_sentence(sentence_buf)

    return collected


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


def _load_evolution_metrics() -> dict:
    if EVOLUTION_METRICS_FILE.exists():
        try:
            return json.loads(EVOLUTION_METRICS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "plugins_created": 0,
        "plugins_approved": 0,
        "plugins_rejected": 0,
        "capabilities_gained": 0,
        "history": [],
    }


def _save_evolution_metrics(metrics: dict) -> None:
    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        EVOLUTION_METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("[MES] Could not save metrics: %s", exc)


def _load_capability_requests() -> dict:
    if CAPABILITY_REQUESTS.exists():
        try:
            return json.loads(CAPABILITY_REQUESTS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_capability_requests(requests: dict) -> None:
    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        CAPABILITY_REQUESTS.write_text(json.dumps(requests, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("[MES] Could not save capability requests: %s", exc)


def record_capability_gap(capability: str, request: str) -> None:
    if not EVOLUTION_ENABLED:
        return
    requests = _load_capability_requests()
    requests[capability] = requests.get(capability, 0) + 1
    _save_capability_requests(requests)
    log.info("[MES] Capability gap recorded: %s (count=%d)", capability, requests[capability])
    if requests[capability] >= EVOLUTION_THRESHOLD:
        log.info("[MES] Threshold reached for '%s' — queuing plugin generation.", capability)
        threading.Thread(
            target=_generate_plugin_async,
            args=(capability, request),
            daemon=True,
        ).start()


def _generate_plugin_async(capability: str, example_request: str) -> None:
    log.info("[MES] Generating plugin for capability: %s", capability)
    prompt = (
        "You are a Python plugin generator for a local AI assistant called Miku.\n\n"
        "Write a Python plugin file that fulfils the following capability.\n\n"
        "Rules:\n"
        "- No network access\n"
        "- No file deletion outside the plugin's own temp files\n"
        "- No shell=True subprocess calls\n"
        "- Single responsibility — one capability per plugin\n"
        "- Must define: PLUGIN = {'name': str, 'description': str}\n"
        "- Must define: execute(args: dict) -> str\n"
        "- Include a brief docstring\n\n"
        f"Capability needed: {capability}\n"
        f"Example user request that triggered this: {example_request}\n\n"
        "Return ONLY the Python source code. No markdown fences. No explanation."
    )
    url = _urljoin(OLLAMA_HOST, "/api/generate")
    payload = {
        "model": EVOLUTION_CODER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.95, "num_predict": 800},
    }
    try:
        resp = _http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S * 2)
        code = (resp or {}).get("response", "").strip()
    except Exception as exc:
        log.warning("[MES] Plugin generation failed: %s", exc)
        return

    if not code:
        log.warning("[MES] Empty code generated for %s.", capability)
        return

    safe_name = re.sub(r"[^a-z0-9_]", "_", capability.lower())[:40]
    EVOLUTION_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    sandbox_file = EVOLUTION_SANDBOX_DIR / f"{safe_name}.py"
    try:
        sandbox_file.write_text(code, encoding="utf-8")
    except Exception as exc:
        log.warning("[MES] Could not write sandbox file: %s", exc)
        return

    metrics = _load_evolution_metrics()
    metrics["plugins_created"] += 1
    _save_evolution_metrics(metrics)

    if EVOLUTION_AUTO_TEST:
        passed, reason = _test_plugin_sandbox(sandbox_file, safe_name)
    else:
        passed, reason = True, "auto-test disabled"

    if not passed:
        log.warning("[MES] Plugin '%s' failed testing: %s", safe_name, reason)
        metrics = _load_evolution_metrics()
        metrics["plugins_rejected"] += 1
        _save_evolution_metrics(metrics)
        return

    review_passed, review_reason = _llm_review_plugin(code)
    if not review_passed:
        log.warning("[MES] Plugin '%s' failed LLM review: %s", safe_name, review_reason)
        metrics = _load_evolution_metrics()
        metrics["plugins_rejected"] += 1
        _save_evolution_metrics(metrics)
        return

    EVOLUTION_PENDING_DIR.mkdir(parents=True, exist_ok=True)
    pending_file = EVOLUTION_PENDING_DIR / f"{safe_name}.py"
    try:
        shutil.copy2(sandbox_file, pending_file)
    except Exception as exc:
        log.warning("[MES] Could not move to pending: %s", exc)
        return

    pending_meta = EVOLUTION_PENDING_DIR / f"{safe_name}.json"
    meta = {
        "capability": capability,
        "plugin_name": safe_name,
        "status": "awaiting_approval",
        "created": datetime.now().isoformat(),
        "review_reason": review_reason,
    }
    try:
        pending_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("[MES] Could not write pending meta: %s", exc)

    _record_evolution_memory(capability, safe_name)
    print(f"\n[MES] New plugin ready for review: '{safe_name}'\n"
          f"      Use /evolve approve {safe_name} or /evolve show {safe_name}")


def _test_plugin_sandbox(plugin_file: Path, plugin_name: str) -> tuple[bool, str]:
    try:
        code = plugin_file.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"Could not read file: {exc}"

    if "import os" in code and ("os.remove" in code or "os.unlink" in code):
        return False, "Unsafe file deletion detected"
    if "subprocess" in code and "shell=True" in code:
        return False, "Unsafe shell=True subprocess detected"
    if "import socket" in code or "urllib" in code or "requests" in code:
        return False, "Network access detected"

    try:
        compile(code, str(plugin_file), "exec")
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    if "PLUGIN" not in code:
        return False, "Missing PLUGIN definition"
    if "def execute" not in code:
        return False, "Missing execute() function"

    return True, "Static checks passed"


def _llm_review_plugin(code: str) -> tuple[bool, str]:
    url = _urljoin(OLLAMA_HOST, "/api/generate")
    prompt = (
        "You are a security reviewer for a local AI assistant plugin system.\n\n"
        "Review the following Python plugin code.\n\n"
        "Look for:\n"
        "- Security vulnerabilities\n"
        "- Dangerous operations (file deletion, network calls, shell injection)\n"
        "- Infinite loops or resource exhaustion\n"
        "- Missing PLUGIN dict or execute() function\n"
        "- Any code that could harm the user's system\n\n"
        'Return STRICT JSON:\n{"result": "PASS" or "FAIL", "confidence": 0.0-1.0, "reason": "string"}\n\n'
        f"Plugin code:\n```python\n{code[:3000]}\n```"
    )
    payload = {
        "model": EVOLUTION_CODER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 200},
    }
    try:
        resp = _http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S)
        text = (resp or {}).get("response", "")
        obj  = _extract_json_object(text)
        if obj:
            result     = str(obj.get("result", "FAIL")).upper()
            reason     = str(obj.get("reason", ""))
            confidence = float(obj.get("confidence", 0.0))
            if result == "PASS" and confidence >= 0.7:
                return True, reason
            return False, reason or "LLM review failed"
    except Exception as exc:
        log.warning("[MES] LLM review failed: %s", exc)
    return False, "LLM review error"


def _record_evolution_memory(capability: str, plugin_name: str) -> None:
    entry = {
        "should_save": True,
        "title": f"Plugin created: {plugin_name}",
        "tags": ["plugin", "evolution", "capability"],
        "summary_md": (
            f"- Miku created a new plugin to handle the '{capability}' capability.\n"
            f"- Plugin name: `{plugin_name}`\n"
            f"- Status: awaiting user approval\n"
            f"- Created: {datetime.now().strftime('%Y-%m-%d')}"
        ),
        "confidence": 0.95,
    }
    try:
        save_knowledge_entry(entry)
    except Exception as exc:
        log.warning("[MES] Could not record evolution memory: %s", exc)


def list_pending_plugins() -> list[dict]:
    if not EVOLUTION_PENDING_DIR.exists():
        return []
    pending = []
    for meta_file in EVOLUTION_PENDING_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            pending.append(meta)
        except Exception:
            pass
    return pending


def approve_plugin(plugin_name: str) -> str:
    safe_name  = re.sub(r"[^a-z0-9_]", "_", plugin_name.lower())[:40]
    pending_py = EVOLUTION_PENDING_DIR / f"{safe_name}.py"
    pending_meta = EVOLUTION_PENDING_DIR / f"{safe_name}.json"

    if not pending_py.exists():
        return f"No pending plugin named '{safe_name}'."

    dest = BASE_DIR / "plugins" / f"{safe_name}.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(pending_py, dest)
    except Exception as exc:
        return f"Could not install plugin: {exc}"

    try:
        pending_py.unlink()
        if pending_meta.exists():
            pending_meta.unlink()
    except Exception:
        pass

    metrics = _load_evolution_metrics()
    metrics["plugins_approved"] += 1
    metrics["capabilities_gained"] += 1
    metrics.setdefault("history", []).append({
        "plugin": safe_name,
        "action": "approved",
        "date": datetime.now().isoformat(),
    })
    _save_evolution_metrics(metrics)

    try:
        module = importlib.import_module(f"plugins.{safe_name}")
        if hasattr(module, "PLUGIN") and hasattr(module, "execute"):
            PLUGINS[module.PLUGIN["name"]] = module
            log.info("[MES] Plugin '%s' loaded into runtime.", safe_name)
    except Exception as exc:
        log.warning("[MES] Plugin approved but failed to load at runtime: %s", exc)

    return f"Plugin '{safe_name}' approved and installed."


def reject_plugin(plugin_name: str) -> str:
    safe_name    = re.sub(r"[^a-z0-9_]", "_", plugin_name.lower())[:40]
    pending_py   = EVOLUTION_PENDING_DIR / f"{safe_name}.py"
    pending_meta = EVOLUTION_PENDING_DIR / f"{safe_name}.json"

    if not pending_py.exists():
        return f"No pending plugin named '{safe_name}'."

    try:
        pending_py.unlink()
        if pending_meta.exists():
            pending_meta.unlink()
    except Exception as exc:
        return f"Could not remove plugin: {exc}"

    metrics = _load_evolution_metrics()
    metrics["plugins_rejected"] += 1
    metrics.setdefault("history", []).append({
        "plugin": safe_name,
        "action": "rejected",
        "date": datetime.now().isoformat(),
    })
    _save_evolution_metrics(metrics)

    return f"Plugin '{safe_name}' rejected and removed."


def show_plugin_code(plugin_name: str) -> str:
    safe_name  = re.sub(r"[^a-z0-9_]", "_", plugin_name.lower())[:40]
    pending_py = EVOLUTION_PENDING_DIR / f"{safe_name}.py"
    if not pending_py.exists():
        return f"No pending plugin named '{safe_name}'."
    try:
        return pending_py.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Could not read plugin: {exc}"


def _print_evolution_status() -> None:
    metrics = _load_evolution_metrics()
    requests = _load_capability_requests()
    pending  = list_pending_plugins()

    print("\n── Evolution Status ───────────────")
    print(f"  Plugins created   : {metrics.get('plugins_created', 0)}")
    print(f"  Plugins approved  : {metrics.get('plugins_approved', 0)}")
    print(f"  Plugins rejected  : {metrics.get('plugins_rejected', 0)}")
    print(f"  Capabilities gained: {metrics.get('capabilities_gained', 0)}")
    if requests:
        print("  Capability gaps:")
        for cap, count in sorted(requests.items(), key=lambda x: x[1], reverse=True)[:8]:
            bar = "█" * min(count, 10)
            print(f"    {cap:<30} {bar} ({count})")
    if pending:
        print("  Pending approvals:")
        for p in pending:
            print(f"    - {p.get('plugin_name')} ({p.get('capability')})")
    else:
        print("  Pending approvals  : none")
    print("───────────────────────────────────\n")


def _handle_tts_command() -> None:
    global _tts_enabled_flag
    _tts_enabled_flag = not _tts_enabled_flag
    state = "ON" if _tts_enabled_flag else "OFF"
    if not _tts_enabled_flag and _tts_worker_instance:
        _tts_worker_instance.drain()
    print(f"[TTS] {state}")


def _print_memory_stats() -> None:
    index        = _load_knowledge_index()
    ids          = index.get("ids", [])
    tag_counts   = index.get("tag_counts", {})
    title_counts = index.get("title_counts", {})

    print("\n── Memory Stats ──────────────────")
    print(f"  Total knowledge entries : {len(ids)}")
    print(f"  Chroma available        : {_chroma_available}")
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
            _set_voice_enabled(not _voice_enabled_flag, voice_input_queue=voice_input_queue)
        elif arg in ("on", "enable", "start"):
            _set_voice_enabled(True, voice_input_queue=voice_input_queue)
        elif arg in ("off", "disable", "stop"):
            _set_voice_enabled(False, voice_input_queue=voice_input_queue)
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
        rebuild_chroma_index()
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
            models = _http_json("GET", _urljoin(OLLAMA_HOST, "/api/tags"))
            for m in (models or {}).get("models", []):
                print(m["name"])
        except Exception as exc:
            print(f"Error: {exc}")
        return True

    if stripped == "/evolve" or stripped.startswith("/evolve "):
        arg = stripped[7:].strip() if stripped != "/evolve" else ""

        if arg == "" or arg == "status":
            _print_evolution_status()

        elif arg == "pending":
            pending = list_pending_plugins()
            if not pending:
                print("No plugins awaiting approval.")
            else:
                for p in pending:
                    print(f"  - {p.get('plugin_name')}  ({p.get('capability')})")

        elif arg.startswith("approve "):
            name   = arg[8:].strip()
            result = approve_plugin(name)
            print(result)

        elif arg.startswith("reject "):
            name   = arg[7:].strip()
            result = reject_plugin(name)
            print(result)

        elif arg.startswith("show "):
            name = arg[5:].strip()
            print(show_plugin_code(name))

        else:
            print("[MES] Usage: /evolve [pending|approve <name>|reject <name>|show <name>]")

        return True

    app_name = detect_open_intent(stripped)
    if app_name:
        result = _launch_app(app_name)
        print(f"Miku: {result}")
        speak(result)
        save_conversation(stripped, result)
        _touch_miku_spoke()
        return True

    plan        = plan_tool_call(stripped)
    tool_result = execute_tool_plan(plan)

    if tool_result:
        print(f"Miku: {tool_result}")
        speak(tool_result)
        save_conversation(stripped, tool_result)
        return True

    prompt   = build_prompt(stripped)
    print("Miku: ", end="", flush=True)
    response = stream_generate(prompt, echo=True)
    _touch_miku_spoke()

    if "I can't" in response or "I don't have" in response or "I'm unable" in response:
        if EVOLUTION_ENABLED:
            threading.Thread(
                target=record_capability_gap,
                args=(_infer_capability_name(stripped), stripped),
                daemon=True,
            ).start()

    threading.Thread(
        target=save_conversation, args=(stripped, response), daemon=True
    ).start()

    if AUTO_SAVE_KNOWLEDGE:
        def _bg_knowledge():
            proposed = propose_knowledge(stripped, response)
            if not proposed:
                return
            confidence = float(proposed.get("confidence", 0.0))
            try:
                saved = save_knowledge_entry(proposed)
            except Exception as exc:
                log.warning("save_knowledge_entry: %s", exc)
                saved = False
            if saved:
                mem_type = _resolve_memory_type(
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
            PLUGINS[module.PLUGIN["name"]] = module


def available_tools_prompt():
    if not PLUGINS:
        return "No tools available."
    lines = []
    for name, module in PLUGINS.items():
        desc = module.PLUGIN["description"]
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def run() -> None:
    _load_state()

    if SEARCH_BACKEND == "chroma":
        ok = _init_chroma()
        chroma_status = "ON" if ok else "OFF (chromadb not installed — using keyword fallback)"
    else:
        chroma_status = "OFF (SEARCH_BACKEND=keyword)"

    if _voice_enabled_flag:
        stt_status = "ON" if _stt_init_if_needed() else "OFF (STT init failed)"
    else:
        stt_status = "OFF (disabled; use /voice to enable)"

    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    EVOLUTION_PENDING_DIR.mkdir(parents=True, exist_ok=True)
    EVOLUTION_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    discover_plugins()

    print(f"Connected to Ollama @ {OLLAMA_HOST}")
    print(f"Model             : {MODEL}")
    print(f"Vector search     : {chroma_status}")
    print(f"Voice / wake word : {stt_status}  (wake word: '{WAKE_WORD}')")
    tts_state = "ON" if TTS_ENABLED else "OFF"
    print(f"TTS               : {tts_state}")
    if AUTO_SAVE_KNOWLEDGE:
        print(f"Auto-knowledge    : ON  (model={KNOWLEDGE_MODEL}, "
              f"confidence≥{KNOWLEDGE_CONFIDENCE_THRESHOLD})")
    else:
        print("Auto-knowledge    : OFF")
    if EVOLUTION_ENABLED:
        metrics = _load_evolution_metrics()
        pending = list_pending_plugins()
        print(f"Evolution (MES)   : ON  (coder={EVOLUTION_CODER_MODEL}, "
              f"threshold={EVOLUTION_THRESHOLD}, "
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
        _set_voice_enabled(True, voice_input_queue=_voice_input_queue)

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