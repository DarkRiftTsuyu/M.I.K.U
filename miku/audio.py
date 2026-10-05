from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

from .config import (
    WAKE_WORD, WHISPER_MODEL, STT_SAMPLE_RATE, STT_CHUNK_SECONDS,
    STT_POST_WAKE_SECS, STT_INPUT_DEVICE, VAD_ENABLED, VAD_THRESHOLD,
    VAD_SILENCE_MS,
)
from .tts import speak

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("miku.audio")

_stt_available: bool           = False
_stt_warned: bool              = False
_whisper_model_instance        = None
_stt_input_device_index: int | None = None
_stt_input_device_name: str    = ""
_stt_device_default_sr: int | None = None
_stt_last_record_warn: float   = 0.0
_stt_record_failures: int      = 0

_silero_vad_model  = None
_silero_vad_warned = False

_voice_enabled_flag: bool                  = False
_wake_word_stop_event: threading.Event | None = None
_wake_word_thread: threading.Thread | None = None


def _stt_warn_rate_limited(message: str) -> None:
    global _stt_last_record_warn, _stt_record_failures
    _stt_record_failures += 1
    now = time.time()
    if (now - _stt_last_record_warn) >= 5.0:
        _stt_last_record_warn = now
        log.warning("[STT] Record failed (%d): %s", _stt_record_failures, message)


def _ensure_sounddevice_input(sd) -> int | None:
    global _stt_input_device_index, _stt_input_device_name, _stt_device_default_sr
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

    want     = STT_INPUT_DEVICE
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
        n_out    = max(1, int(round(duration * target_sr)))
        x_old    = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float64)
        x_new    = np.linspace(0.0, duration, num=n_out,       endpoint=False, dtype=np.float64)
        return np.interp(x_new, x_old, audio.astype(np.float64, copy=False)).astype(np.float32)
    except Exception:
        return audio


def stt_init_if_needed() -> bool:
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
            log.info("[STT] No usable input audio device found — wake word disabled.")
            return False
        _whisper_model_instance = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        _stt_available = True
        mic_label = _stt_input_device_name or str(mic)
        log.info("[STT] faster-whisper ready (model=%s, mic=%s)", WHISPER_MODEL, mic_label)
        return True
    except ImportError:
        _stt_warned = True
        log.info("[STT] faster-whisper or sounddevice not available.")
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
        audio = sd.rec(frames, samplerate=record_sr, channels=1, dtype="float32")
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
        segments, _ = _whisper_model_instance.transcribe(audio, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segments).strip()
    except Exception as exc:
        log.warning("[STT] Transcribe failed: %s", exc)
        return ""


def _init_silero_vad():
    global _silero_vad_model, _silero_vad_warned
    if _silero_vad_model is not None:
        return _silero_vad_model
    if _silero_vad_warned:
        return None
    try:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            verbose=False,
        )
        model.eval()
        _silero_vad_model = model
        log.info("[VAD] Silero VAD loaded ✔")
        return model
    except Exception as exc:
        _silero_vad_warned = True
        log.info("[VAD] Silero VAD not available (%s) — using fixed-window recording.", exc)
        return None


def _vad_is_speech(chunk: "np.ndarray", model, sr: int = 16000) -> float:
    try:
        import torch
        import numpy as np
        tensor = torch.tensor(chunk, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return float(model(tensor, sr).item())
    except Exception:
        return 0.0


def _record_with_vad(max_seconds: float) -> "np.ndarray | None":
    model = _init_silero_vad() if VAD_ENABLED else None
    if model is None:
        return _record_audio(max_seconds)
    try:
        import sounddevice as sd
        import numpy as np
        sr             = int(STT_SAMPLE_RATE)
        frame_size     = 512
        silence_frames = max(1, int((VAD_SILENCE_MS / 1000) * sr / frame_size))
        max_frames     = int(max_seconds * sr / frame_size)
        frames_collected: list[np.ndarray] = []
        silent_frame_count = 0
        speech_started     = False
        for _ in range(max_frames):
            chunk_raw = sd.rec(frame_size, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            chunk = chunk_raw.flatten()
            frames_collected.append(chunk)
            prob = _vad_is_speech(chunk, model, sr)
            if prob > VAD_THRESHOLD:
                speech_started     = True
                silent_frame_count = 0
            else:
                if speech_started:
                    silent_frame_count += 1
                    if silent_frame_count >= silence_frames:
                        break
        if not frames_collected:
            return None
        return np.concatenate(frames_collected, axis=0)
    except Exception as exc:
        log.warning("[VAD] Record-with-VAD failed: %s — falling back to fixed window.", exc)
        return _record_audio(max_seconds)


def wake_word_loop(input_queue: queue.Queue, stop_event: threading.Event | None = None) -> None:
    if stop_event is None:
        stop_event = threading.Event()
    if not stt_init_if_needed():
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
        command_audio = _record_with_vad(STT_POST_WAKE_SECS) if VAD_ENABLED else _record_audio(STT_POST_WAKE_SECS)
        if stop_event.is_set():
            break
        if command_audio is None:
            continue
        command_text = _transcribe(command_audio)
        if command_text:
            print(f"\n[Voice] {command_text}")
            input_queue.put(command_text)


def set_voice_enabled(enabled: bool, voice_input_queue: queue.Queue | None = None) -> None:
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
    if not stt_init_if_needed():
        print("[Voice] STT unavailable.")
        _voice_enabled_flag = False
        return
    _voice_enabled_flag   = True
    _wake_word_stop_event = threading.Event()
    _wake_word_thread     = threading.Thread(
        target=wake_word_loop,
        args=(voice_input_queue, _wake_word_stop_event),
        daemon=True,
    )
    _wake_word_thread.start()
    print(f"[Voice] ON  (wake word: '{WAKE_WORD}')")


def is_voice_enabled() -> bool:
    return _voice_enabled_flag


def get_stt_status() -> str:
    return _stt_input_device_name


def get_silero_vad_status() -> str:
    if _silero_vad_model is not None:
        return "available"
    if _silero_vad_warned:
        return "unavailable"
    return "unavailable (pip install torch)"
