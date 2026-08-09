from .sharedConst import *
from typing import Optional
from pathlib import Path
import importlib
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
import numpy as np

from .sharedConst import *

log = logging.getLogger("miku")


_tts_worker_instance: Optional["_TTSWorker"] = None
_tts_available   = False
_tts_warned      = False

#region Constant var (dir based) helpers
def _runtime_module():
    for name in ("__main__", "main"):
        module = sys.modules.get(name)
        if module is not None:
            return module
    try:
        return importlib.import_module("main")
    except Exception:
        return None


def _runtime_value(name, default=None):
    module = _runtime_module()
    if module is None:
        return default
    return getattr(module, name, default)


def _get_base_dir() -> Path:
    base = _runtime_value("BASE_DIR", None)
    if base is None:
        return Path(__file__).resolve().parents[1]
    return Path(base).resolve()


def _get_tts_enabled_flag() -> bool:
    return bool(_runtime_value("_get_tts_enabled_flag()", TTS_ENABLED))


def _get_voice_enabled_flag() -> bool:
    return bool(_runtime_value("_voice_enabled_flag", False))
#endregion

def _tts_log(msg: str) -> None:
    if TTS_DEBUG:
        log.debug("[TTS] %s", msg)

def _tts_init_if_needed() -> bool:
    global _tts_available, _tts_warned
    if not _get_tts_enabled_flag():
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
            base = (_get_base_dir() / base).resolve()
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
        if not text or not _get_tts_enabled_flag():
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
            base = (_get_base_dir() / base).resolve()
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
            if not _get_tts_enabled_flag():
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
            if not _get_tts_enabled_flag():
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