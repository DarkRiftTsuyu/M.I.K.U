from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from .config import (
    TTS_ENABLED, TTS_QUEUE_MAX, TTS_DEBUG, TTS_MAX_CHARS,
    KITTEN_DIR, KITTEN_MODEL, KITTEN_VOICES, KITTEN_TOKENS,
    KITTEN_DATA_DIR, KITTEN_SID, KITTEN_SPEED, KITTEN_NUM_THREADS,
    BASE_DIR,
)

log = logging.getLogger("miku.tts")

_tts_enabled_flag: bool = TTS_ENABLED
_tts_available: bool    = False
_tts_warned: bool       = False
_tts_worker_instance: Optional["_TTSWorker"] = None

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _tts_log(msg: str) -> None:
    if TTS_DEBUG:
        log.debug("[TTS] %s", msg)


def _tts_sanitize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_powershell_exe() -> str | None:
    for name in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
        found = shutil.which(name)
        if found:
            return found
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    legacy  = os.path.join(sysroot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return legacy if os.path.exists(legacy) else None


def tts_init_if_needed() -> bool:
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
        log.info("[TTS] sherpa-onnx not available — running without speech.")
        return False
    except Exception as exc:
        _tts_warned = True
        log.warning("[TTS] Init check failed: %s", exc)
        return False


class _TTSWorker:
    def __init__(self) -> None:
        self.q: queue.Queue[str | None] = queue.Queue(maxsize=max(1, TTS_QUEUE_MAX))
        self.thread = threading.Thread(target=self._run, name="TTSWorker", daemon=True)
        self._tts     = None
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


def _get_tts_worker() -> _TTSWorker:
    global _tts_worker_instance
    if _tts_worker_instance is None or not _tts_worker_instance.thread.is_alive():
        _tts_worker_instance = _TTSWorker()
        _tts_worker_instance.start()
    return _tts_worker_instance


def speak(text: str) -> None:
    if not tts_init_if_needed():
        return
    _get_tts_worker().enqueue(text)


def speak_sentence(fragment: str) -> None:
    if not tts_init_if_needed():
        return
    clean = _tts_sanitize(fragment)
    if clean:
        _get_tts_worker().enqueue(clean)


def toggle_tts() -> str:
    global _tts_enabled_flag
    _tts_enabled_flag = not _tts_enabled_flag
    if not _tts_enabled_flag and _tts_worker_instance:
        _tts_worker_instance.drain()
    return "ON" if _tts_enabled_flag else "OFF"


def is_tts_enabled() -> bool:
    return _tts_enabled_flag
