from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, datetime

from .config import (
    STATE_FILE, VAULT_PATH, MAX_UNPROMPTED_PER_DAY, TTS_ENABLED,
    VOICE_ENABLED_DEFAULT,
)

log = logging.getLogger("miku.state")

_state_lock = threading.Lock()

_user_state: dict         = {"mood": "neutral", "energy": "normal", "last_update": None}
_last_user_time: float    = time.time()
_last_user_message: str   = ""
_last_miku_spoke: float   = 0.0
_initiative_counts: dict  = {}

_tool_failure_state: dict = {
    "recent_failure": False, "when": None, "where": None, "error": None,
}


def get_user_state() -> dict:
    with _state_lock:
        return dict(_user_state)


def set_user_state(state: dict) -> None:
    with _state_lock:
        _user_state.update(state)
    persist_state()


def touch_user_time(msg: str = "") -> None:
    global _last_user_time, _last_user_message
    with _state_lock:
        _last_user_time    = time.time()
        _last_user_message = msg


def touch_miku_spoke() -> None:
    global _last_miku_spoke
    with _state_lock:
        _last_miku_spoke = time.time()


def get_last_user_time() -> float:
    with _state_lock:
        return _last_user_time


def get_last_miku_spoke() -> float:
    with _state_lock:
        return _last_miku_spoke


def initiative_allowed() -> bool:
    today = date.today().isoformat()
    with _state_lock:
        count = _initiative_counts.get(today, 0)
        if count >= MAX_UNPROMPTED_PER_DAY:
            return False
        _initiative_counts[today] = count + 1
    return True


def get_tool_failure_state() -> dict:
    with _state_lock:
        return dict(_tool_failure_state)


def set_tool_failure_state(state: dict) -> None:
    global _tool_failure_state
    with _state_lock:
        _tool_failure_state = dict(state)
    persist_state()


def load_state() -> None:
    global _user_state, _last_miku_spoke, _tool_failure_state
    if not STATE_FILE.exists():
        return
    try:
        obj = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        with _state_lock:
            _user_state.update(obj.get("user_state", {}))
            _last_miku_spoke = float(obj.get("last_miku_spoke", 0.0))
            tfs = obj.get("tool_failure_state")
            if isinstance(tfs, dict):
                _tool_failure_state.update({
                    "recent_failure": bool(tfs.get("recent_failure", False)),
                    "when":  tfs.get("when"),
                    "where": tfs.get("where"),
                    "error": tfs.get("error"),
                })
    except Exception as exc:
        log.warning("Could not load state.json: %s", exc)


def persist_state() -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        with _state_lock:
            obj = {
                "user_state":         dict(_user_state),
                "last_miku_spoke":    _last_miku_spoke,
                "tool_failure_state": dict(_tool_failure_state),
            }
        STATE_FILE.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not save state.json: %s", exc)
