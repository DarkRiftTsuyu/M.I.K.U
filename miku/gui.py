from __future__ import annotations

# ── Dependency check (no auto-install) ───────────────────────────────────────
try:
    from PyQt6.QtCore import (
        Qt, QThread, pyqtSignal, QTimer, QSize, QPoint,
    )
    from PyQt6.QtGui import (
        QColor, QFont, QPainter, QPen, QBrush, QRadialGradient,
        QPalette,
    )
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QTextEdit, QLineEdit, QPushButton, QScrollArea,
        QFrame, QSplitter, QProgressBar, QTabWidget,
        QGraphicsDropShadowEffect, QTableWidget, QTableWidgetItem,
        QHeaderView, QMessageBox,
    )
except ImportError:
    import sys
    print(
        "\n[Miku] PyQt6 is not installed.\n"
        "Please install it manually:\n\n"
        "    pip install PyQt6\n\n"
        "Then re-run:  python gui.py\n"
    )
    sys.exit(1)

import importlib.util
import json
import os
import queue
import re
import sys
import threading
import types
import time
from datetime import date, datetime
from pathlib import Path

# ── Colour Palette ────────────────────────────────────────────────────────────
NIGHT       = "#0a0e1a"
NIGHT_MID   = "#0f1628"
NIGHT_LIGHT = "#161e36"
BORDER      = "#1e2d50"
CYAN        = "#00e5ff"
CYAN_DIM    = "#007a8a"
TEAL        = "#26c6da"
PINK        = "#f48fb1"
TEXT_MAIN   = "#dde8f0"
TEXT_DIM    = "#6a84a0"
TEXT_WHITE  = "#ffffff"
GREEN       = "#69f0ae"
AMBER       = "#ffd740"
RED         = "#ff5252"
PURPLE      = "#b39ddb"

FONT_DISPLAY = "Segoe UI"

# ── Vault paths (mirrors main.py) ─────────────────────────────────────────────
BASE_DIR               = Path(__file__).resolve().parent
VAULT_PATH             = BASE_DIR / "vault"
RELATIONSHIP_FILE      = VAULT_PATH / "relationship.json"
GOALS_FILE             = VAULT_PATH / "goals.json"
ACTIVE_TASK_FILE       = VAULT_PATH / "active_task.json"
CONVERSATION_STATE_FILE= VAULT_PATH / "conversation_state.json"
TOOL_CONTEXT_FILE      = VAULT_PATH / "tool_context.json"
STATE_FILE             = VAULT_PATH / "state.json"
JOURNAL_DIR            = VAULT_PATH / "journal"
REFLECTION_FILE        = VAULT_PATH / "reflection.md"
LIFE_STORY_FILE        = VAULT_PATH / "life_story.md"
CURIOSITY_FILE         = VAULT_PATH / "curiosity_state.json"
RECENT_TURNS_FILE      = VAULT_PATH / "recent_turns.jsonl"
KNOWLEDGE_INDEX_FILE   = VAULT_PATH / "knowledge" / "_index.json"
KNOWLEDGE_GRAPH_FILE   = VAULT_PATH / "knowledge_graph.json"
EVOLUTION_DIR          = VAULT_PATH / "evolution"
EVOLUTION_METRICS_FILE = EVOLUTION_DIR / "metrics.json"
CAPABILITY_REQUESTS    = EVOLUTION_DIR / "capability_requests.json"
EVOLUTION_PENDING_DIR  = EVOLUTION_DIR / "pending"

# ─────────────────────────────────────────────────────────────────────────────
#  Event bridge
#  Instead of monkey-patching builtins.print, we inject lightweight callback
#  hooks directly into the loaded module's namespace.  stream_generate calls
#  print(chunk, end="", flush=True) — we replace that with a module-level
#  _gui_print that dispatches to registered callbacks without touching the
#  global builtins at all.
# ─────────────────────────────────────────────────────────────────────────────
class EventBridge:
    """
    Holds per-module print/log/error callbacks.
    main.py's functions call  _gui_print / _gui_log / _gui_error
    if those names exist in the module's own namespace.
    """
    def __init__(self):
        self._on_chunk:    list = []
        self._on_log:      list = []
        self._on_error:    list = []
        self._on_state:    list = []   # dict snapshots of internal state

    def on_chunk(self,  fn): self._on_chunk.append(fn)
    def on_log(self,    fn): self._on_log.append(fn)
    def on_error(self,  fn): self._on_error.append(fn)
    def on_state(self,  fn): self._on_state.append(fn)

    def emit_chunk(self, text: str):
        for fn in self._on_chunk:
            try: fn(text)
            except Exception: pass

    def emit_log(self, text: str):
        for fn in self._on_log:
            try: fn(text)
            except Exception: pass

    def emit_error(self, text: str):
        for fn in self._on_error:
            try: fn(text)
            except Exception: pass

    def emit_state(self, snap: dict):
        for fn in self._on_state:
            try: fn(snap)
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
#  Service registry — prevents duplicate background threads
# ─────────────────────────────────────────────────────────────────────────────
class ServiceRegistry:
    _lock = threading.Lock()
    _services: dict[str, threading.Thread] = {}

    @classmethod
    def register(cls, name: str, thread: threading.Thread) -> bool:
        """Register a service thread.  Returns False if already running."""
        with cls._lock:
            existing = cls._services.get(name)
            if existing and existing.is_alive():
                return False
            cls._services[name] = thread
            return True

    @classmethod
    def is_running(cls, name: str) -> bool:
        with cls._lock:
            t = cls._services.get(name)
            return t is not None and t.is_alive()

    @classmethod
    def start_once(cls, name: str, target, args=()) -> bool:
        """Start a daemon thread only if not already running."""
        with cls._lock:
            existing = cls._services.get(name)
            if existing and existing.is_alive():
                return False
            t = threading.Thread(target=target, args=args, daemon=True, name=name)
            cls._services[name] = t
            t.start()
            return True

    @classmethod
    def snapshot(cls) -> dict[str, bool]:
        with cls._lock:
            return {k: v.is_alive() for k, v in cls._services.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Worker thread
# ─────────────────────────────────────────────────────────────────────────────
class MikuWorker(QThread):
    # Event-based signals — no more builtins.print patching
    sig_chunk   = pyqtSignal(str)   # streaming response token
    sig_done    = pyqtSignal()      # turn complete
    sig_log     = pyqtSignal(str)   # system log line
    sig_error   = pyqtSignal(str)   # error message
    sig_state   = pyqtSignal(dict)  # internal state snapshot

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._voice_queue: queue.Queue = queue.Queue()
        self.bridge = EventBridge()
        self.miku   = None   # loaded main module

    def _load_main(self) -> bool:
        if self.miku is not None:
            return True

        try:
            # Import the refactored Miku package directly.
            import miku.config as cfg
            import miku.cli as cli
            import miku.state as state
            import miku.memory as memory
            import miku.search as search
            import miku.tools as tools
            import miku.loops as loops
            import miku.tts as tts
            import miku.audio as audio
            import miku.evolution as evolution
            import miku.llm as llm

            # ── Inject event callbacks into relevant module namespaces ─────
            bridge = self.bridge

            def _gui_print(*args, end="\n", flush=False, **kwargs):
                text = " ".join(str(a) for a in args)
                bridge.emit_chunk(text + ("" if end == "\n" else end))

            def _gui_log(msg: str):
                bridge.emit_log(str(msg))

            def _gui_error(msg: str):
                bridge.emit_error(str(msg))

            # Patch only Miku modules, not global builtins.
            for mod in (cli, loops, llm, tools, memory, evolution):
                try:
                    mod.print = _gui_print
                    mod._gui_print = _gui_print
                    mod._gui_log = _gui_log
                    mod._gui_error = _gui_error
                except Exception:
                    pass

            # ── Wire bridge to Qt signals ─────────────────────────────────
            bridge.on_chunk(lambda t: self.sig_chunk.emit(t))
            bridge.on_log(lambda t: self.sig_log.emit(t))
            bridge.on_error(lambda t: self.sig_error.emit(t))
            bridge.on_state(lambda s: self.sig_state.emit(s))

            # ── Initialise backend state ──────────────────────────────────
            state.load_state()

            if not getattr(cli, "_recent_turns", []):
                cli._recent_turns = memory.load_recent_turns_from_disk(cfg.LAST_N_MESSAGES)

            if cfg.SEARCH_BACKEND == "chroma":
                search.init_chroma()

            tools.discover_plugins()
            state.load_state()

            # ── Store a tiny backend adapter for the rest of gui.py ─────────
            self.miku = types.SimpleNamespace(
                cfg=cfg,
                cli=cli,
                state=state,
                memory=memory,
                search=search,
                tools=tools,
                loops=loops,
                tts=tts,
                audio=audio,
                evolution=evolution,
                llm=llm,

                _handle_input=cli._handle_input,
                _set_voice_enabled=audio.set_voice_enabled,

                approve_plugin=lambda name: evolution.approve_plugin(name, tools.PLUGINS),
                reject_plugin=evolution.reject_plugin,
            )

            # ── Start background services ─────────────────────────────────
            self._speech_queue = queue.Queue()
            self._voice_input_q = queue.Queue()

            ServiceRegistry.start_once(
                "speech_loop",
                loops.speech_loop,
                args=(self._speech_queue,),
            )

            ServiceRegistry.start_once(
                "initiative_loop",
                loops.initiative_loop,
                args=(self._speech_queue,),
            )

            ServiceRegistry.start_once(
                "reminder_loop",
                loops.reminder_loop,
                args=(self._speech_queue,),
            )

            ServiceRegistry.start_once(
                "reflection_loop",
                loops.reflection_loop,
            )

            if cfg.VOICE_ENABLED_DEFAULT:
                audio.set_voice_enabled(True, voice_input_queue=self._voice_input_q)

            self.sig_log.emit("✓ Miku package backend online")
            self.sig_log.emit(
                "✓ Services: "
                + ", ".join(k for k, v in ServiceRegistry.snapshot().items() if v)
            )

            return True

        except Exception as exc:
            self.sig_error.emit(f"Failed to load Miku package backend: {exc}")
            return False
    # ── Module loading ───────────────────────────────────────────────────────
   

    # ── Public interface ─────────────────────────────────────────────────────
    def send_message(self, text: str, is_voice: bool = False):
        self._queue.put(("msg", text, is_voice))

    def set_voice_enabled(self, enabled: bool):
        self._queue.put(("voice", enabled))

    def approve_plugin(self, name: str):
        self._queue.put(("approve_plugin", name))

    def reject_plugin(self, name: str):
        self._queue.put(("reject_plugin", name))

    # ── Main loop ────────────────────────────────────────────────────────────
    def run(self):
        if not self._load_main():
            return
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            kind = item[0]
            if kind == "msg":
                _, text, is_voice = item
                try:
                    self.miku._handle_input(
                        text,
                        voice_input_queue=self._voice_input_q,
                    )
                except Exception as exc:
                    self.sig_error.emit(f"Error: {exc}")
                self.sig_done.emit()
                self._emit_state_snapshot()

            elif kind == "voice":
                try:
                    self.miku._set_voice_enabled(
                        item[1], voice_input_queue=self._voice_input_q
                    )
                except Exception as exc:
                    self.sig_error.emit(f"Voice error: {exc}")

            elif kind == "approve_plugin":
                try:
                    result = self.miku.approve_plugin(item[1])
                    self.sig_log.emit(f"[MES] {result}")
                except Exception as exc:
                    self.sig_error.emit(f"Approve plugin error: {exc}")
                self._emit_state_snapshot()

            elif kind == "reject_plugin":
                try:
                    result = self.miku.reject_plugin(item[1])
                    self.sig_log.emit(f"[MES] {result}")
                except Exception as exc:
                    self.sig_error.emit(f"Reject plugin error: {exc}")
                self._emit_state_snapshot()
                
    def _emit_state_snapshot(self):
        """Gather a snapshot of Miku's internal state and emit it."""
        if not self.miku:
            return

        try:
            b = self.miku

            task = b.memory.load_active_task()
            cs = b.memory.load_conversation_state()
            tc = b.memory.load_tool_context()

            try:
                tf = b.tools._tool_failure_state
            except Exception:
                tf = {}

            us = b.state.get_user_state()

            pending = (
                b.evolution.list_pending_plugins()
                if hasattr(b.evolution, "list_pending_plugins")
                else []
            )

            metrics = (
                b.evolution.load_evolution_metrics()
                if hasattr(b.evolution, "load_evolution_metrics")
                else {}
            )

            snap = {
                "active_task": task,
                "conversation": cs,
                "tool_context": tc,
                "tool_failure": dict(tf or {}),
                "user_state": us,
                "pending_plugins": pending,
                "evolution": metrics,
                "services": ServiceRegistry.snapshot(),
                "tts_enabled": b.tts.is_tts_enabled(),
                "voice_enabled": b.audio.is_voice_enabled(),
            }

            self.sig_state.emit(snap)

        except Exception as exc:
            self.sig_error.emit(f"State snapshot failed: {exc}")
    def get_state_snapshot(self) -> dict:
        if not self.miku:
            return {}
        try:
            self._emit_state_snapshot()
        except Exception:
            pass
        return {}

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
#  Avatar widget
# ─────────────────────────────────────────────────────────────────────────────
class AvatarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(90, 90)
        self._phase    = 0.0
        self._speaking = False
        self._thinking = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def set_speaking(self, v: bool): self._speaking = v
    def set_thinking(self, v: bool): self._thinking = v

    def _tick(self):
        speed = 0.10 if self._speaking else (0.05 if self._thinking else 0.025)
        self._phase = (self._phase + speed) % (2 * 3.14159)
        self.update()

    def paintEvent(self, _ev):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, r = 45, 45, 32
        pulse = (math.sin(self._phase) + 1) / 2

        for rad, base_alpha in [(44, 0.07), (40, 0.14), (36, 0.22)]:
            a = (pulse if (self._speaking or self._thinking) else 0.35) * base_alpha * 3.5
            col = QColor(CYAN if not self._thinking else AMBER)
            col.setAlphaF(min(a, 1.0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawEllipse(QPoint(cx, cy), rad, rad)

        grad = QRadialGradient(cx - 8, cy - 8, 38)
        grad.setColorAt(0, QColor("#26c6da"))
        grad.setColorAt(0.5, QColor("#00838f"))
        grad.setColorAt(1, QColor("#006064"))
        p.setBrush(QBrush(grad))
        pen = QPen(QColor(CYAN)); pen.setWidthF(1.5)
        p.setPen(pen)
        p.drawEllipse(QPoint(cx, cy), r, r)

        # Hair tails
        p.setPen(Qt.PenStyle.NoPen)
        hc = QColor("#004d5c"); hc.setAlphaF(0.65)
        p.setBrush(QBrush(hc))
        p.drawEllipse(22, 8, 9, 20)
        p.drawEllipse(59, 8, 9, 20)

        # Face
        p.setBrush(QBrush(QColor("#e0f7fa")))
        p.drawEllipse(QPoint(cx, cy + 2), 18, 18)

        # Eyes
        ec = QColor(CYAN); ec.setAlphaF(0.85)
        p.setBrush(QBrush(ec))
        p.drawEllipse(37, 40, 5, 6)
        p.drawEllipse(50, 40, 5, 6)

        # Mouth
        if self._speaking:
            p.setPen(QPen(QColor("#006064"), 1.5))
            p.drawArc(40, 50, 10, 5, 0, -180 * 16)


# ─────────────────────────────────────────────────────────────────────────────
#  Chat widgets
# ─────────────────────────────────────────────────────────────────────────────
class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool, ts: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)

        bubble = QFrame()
        inner  = QVBoxLayout(bubble)
        inner.setContentsMargins(12, 8, 12, 8)
        inner.setSpacing(2)

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setFont(QFont(FONT_DISPLAY, 10))

        ts_lbl = QLabel(ts)
        ts_lbl.setFont(QFont(FONT_DISPLAY, 7))

        if is_user:
            lbl.setStyleSheet(f"color:{TEXT_MAIN};")
            ts_lbl.setStyleSheet(f"color:{TEXT_DIM};")
            bubble.setStyleSheet(f"""
                QFrame{{background:{NIGHT_LIGHT};border:1px solid {BORDER};
                        border-radius:14px 14px 4px 14px;}}""")
            outer.addStretch()
            inner.addWidget(lbl)
            inner.addWidget(ts_lbl, alignment=Qt.AlignmentFlag.AlignRight)
            outer.addWidget(bubble)
        else:
            lbl.setStyleSheet(f"color:{TEXT_WHITE};")
            ts_lbl.setStyleSheet(f"color:{CYAN_DIM};")
            bubble.setStyleSheet(f"""
                QFrame{{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #0d2a3a,stop:1 #0a1e2e);
                        border:1px solid {CYAN_DIM};
                        border-radius:14px 14px 14px 4px;}}""")
            inner.addWidget(lbl)
            inner.addWidget(ts_lbl, alignment=Qt.AlignmentFlag.AlignLeft)
            outer.addWidget(bubble)
            outer.addStretch()

        sh = QGraphicsDropShadowEffect()
        sh.setBlurRadius(10)
        sh.setColor(QColor(0, 0, 0, 50))
        sh.setOffset(0, 2)
        bubble.setGraphicsEffect(sh)


class StreamingBubble(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)
        self._bubble = QFrame()
        self._bubble.setStyleSheet(f"""
            QFrame{{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #0d2a3a,stop:1 #0a1e2e);
                    border:1px solid {CYAN_DIM};
                    border-radius:14px 14px 14px 4px;}}""")
        inner = QVBoxLayout(self._bubble)
        inner.setContentsMargins(12, 8, 12, 8)
        self._lbl = QLabel("")
        self._lbl.setWordWrap(True)
        self._lbl.setFont(QFont(FONT_DISPLAY, 10))
        self._lbl.setStyleSheet(f"color:{TEXT_WHITE};")
        self._lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        inner.addWidget(self._lbl)
        outer.addWidget(self._bubble)
        outer.addStretch()
        self._text = ""

    def append(self, chunk: str):
        self._text += chunk
        self._lbl.setText(re.sub(r"^Miku:\s*", "", self._text.strip()))

    def final_text(self) -> str:
        return re.sub(r"^Miku:\s*", "", self._text.strip())


# ─────────────────────────────────────────────────────────────────────────────
#  Chat panel
# ─────────────────────────────────────────────────────────────────────────────
class ChatPanel(QWidget):
    message_submitted = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._streaming: StreamingBubble | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(f"""
            QScrollArea{{border:none;background:{NIGHT};}}
            QScrollBar:vertical{{background:{NIGHT_MID};width:5px;border-radius:2px;}}
            QScrollBar::handle:vertical{{background:{CYAN_DIM};border-radius:2px;min-height:20px;}}
        """)
        self._msg_w  = QWidget()
        self._msg_l  = QVBoxLayout(self._msg_w)
        self._msg_l.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._msg_l.setSpacing(4)
        self._msg_l.setContentsMargins(4, 8, 4, 8)
        self._scroll.setWidget(self._msg_w)
        root.addWidget(self._scroll)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"border:none;border-top:1px solid {BORDER};")
        div.setFixedHeight(1)
        root.addWidget(div)

        bar = QWidget()
        bar.setFixedHeight(60)
        bar.setStyleSheet(f"background:{NIGHT_MID};")
        bl  = QHBoxLayout(bar)
        bl.setContentsMargins(10, 8, 10, 8)
        bl.setSpacing(8)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Say something to Miku…")
        self._input.setFont(QFont(FONT_DISPLAY, 10))
        self._input.setStyleSheet(f"""
            QLineEdit{{background:{NIGHT_LIGHT};color:{TEXT_MAIN};
                       border:1px solid {BORDER};border-radius:20px;padding:6px 16px;}}
            QLineEdit:focus{{border:1px solid {CYAN_DIM};}}""")
        self._input.returnPressed.connect(self._on_send)

        self._send = QPushButton("Send")
        self._send.setFixedSize(64, 36)
        self._send.setFont(QFont(FONT_DISPLAY, 9, QFont.Weight.Medium))
        self._send.clicked.connect(self._on_send)
        self._send.setStyleSheet(f"""
            QPushButton{{background:{CYAN_DIM};color:{NIGHT};border-radius:18px;font-weight:600;}}
            QPushButton:hover{{background:{CYAN};}}
            QPushButton:pressed{{background:{TEAL};}}
            QPushButton:disabled{{background:{BORDER};color:{TEXT_DIM};}}""")

        bl.addWidget(self._input)
        bl.addWidget(self._send)
        root.addWidget(bar)

    def _on_send(self):
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self.add_user_msg(text)
        self._send.setEnabled(False)
        self.message_submitted.emit(text, False)

    def add_user_msg(self, text: str):
        ts = datetime.now().strftime("%H:%M")
        self._msg_l.addWidget(ChatBubble(text, is_user=True, ts=ts))
        self._scroll_bottom()

    def start_stream(self):
        self._streaming = StreamingBubble()
        self._msg_l.addWidget(self._streaming)
        self._scroll_bottom()

    def feed_stream(self, chunk: str):
        if self._streaming:
            self._streaming.append(chunk)
            self._scroll_bottom()

    def end_stream(self):
        if self._streaming:
            txt = self._streaming.final_text()
            self._msg_l.removeWidget(self._streaming)
            self._streaming.deleteLater()
            self._streaming = None
            if txt:
                ts = datetime.now().strftime("%H:%M")
                self._msg_l.addWidget(ChatBubble(txt, is_user=False, ts=ts))
        self._send.setEnabled(True)
        self._scroll_bottom()

    def add_sys_msg(self, text: str):
        lbl = QLabel(text)
        lbl.setFont(QFont(FONT_DISPLAY, 8))
        lbl.setStyleSheet(f"color:{TEXT_DIM};padding:2px 16px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._msg_l.addWidget(lbl)
        self._scroll_bottom()

    def add_error_msg(self, text: str):
        lbl = QLabel(f"⚠  {text}")
        lbl.setFont(QFont(FONT_DISPLAY, 8))
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{RED};padding:4px 16px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._msg_l.addWidget(lbl)
        self._scroll_bottom()

    def load_history(self):
        if not RECENT_TURNS_FILE.exists():
            return
        try:
            lines = RECENT_TURNS_FILE.read_text(encoding="utf-8").splitlines()[-40:]
            for line in lines:
                try:
                    obj  = json.loads(line)
                    role = obj.get("role", "")
                    text = str(obj.get("content", "")).strip()
                    if not text:
                        continue
                    ts_r = obj.get("ts", "")
                    try:
                        ts = datetime.fromisoformat(ts_r).strftime("%H:%M") if ts_r else ""
                    except Exception:
                        ts = ""
                    self._msg_l.addWidget(
                        ChatBubble(text, is_user=(role == "user"), ts=ts)
                    )
                except Exception:
                    continue
        except Exception:
            pass
        self._scroll_bottom()

    def _scroll_bottom(self):
        QTimer.singleShot(60, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))


# ─────────────────────────────────────────────────────────────────────────────
#  Reusable card widget
# ─────────────────────────────────────────────────────────────────────────────
class Card(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame{{background:{NIGHT_LIGHT};border:1px solid {BORDER};border-radius:8px;}}""")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(12, 10, 12, 10)
        self._lay.setSpacing(6)

        hdr = QLabel(title.upper())
        hdr.setFont(QFont(FONT_DISPLAY, 7, QFont.Weight.Bold))
        hdr.setStyleSheet(f"color:{CYAN_DIM};letter-spacing:1px;background:transparent;border:none;")
        self._lay.addWidget(hdr)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"border:none;border-top:1px solid {BORDER};background:transparent;")
        div.setFixedHeight(1)
        self._lay.addWidget(div)

    def body(self) -> QVBoxLayout:
        return self._lay

    def _stat(self, txt: str) -> QLabel:
        l = QLabel(txt)
        l.setFont(QFont(FONT_DISPLAY, 8))
        l.setStyleSheet(f"color:{TEXT_DIM};background:transparent;border:none;")
        return l


def _stat_label(txt: str, color: str = TEXT_DIM) -> QLabel:
    l = QLabel(txt)
    l.setFont(QFont(FONT_DISPLAY, 8))
    l.setStyleSheet(f"color:{color};background:transparent;border:none;")
    return l


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard tab – live internal state
# ─────────────────────────────────────────────────────────────────────────────
class DashboardTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{NIGHT_MID};")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea{{border:none;background:{NIGHT_MID};}}
            QScrollBar:vertical{{background:{NIGHT_MID};width:4px;border-radius:2px;}}
            QScrollBar::handle:vertical{{background:{CYAN_DIM};border-radius:2px;min-height:16px;}}
        """)
        inner = QWidget()
        inner.setStyleSheet(f"background:{NIGHT_MID};")
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll)
        self._build_cards()

    def _build_cards(self):
        L = self._lay

        # ── Conversation state ────────────────────────────────────────────
        c = Card("Conversation")
        L.addWidget(c)
        self._topic_lbl    = _stat_label("Topic: —")
        self._threads_lbl  = _stat_label("Threads: —")
        self._threads_lbl.setWordWrap(True)
        c.body().addWidget(self._topic_lbl)
        c.body().addWidget(self._threads_lbl)

        # ── Active task ───────────────────────────────────────────────────
        t = Card("Active Task")
        L.addWidget(t)
        self._task_goal_lbl   = _stat_label("Goal: —")
        self._task_status_lbl = _stat_label("Status: —")
        self._task_reqs_lbl   = _stat_label("Requirements: —")
        self._task_reqs_lbl.setWordWrap(True)
        t.body().addWidget(self._task_goal_lbl)
        t.body().addWidget(self._task_status_lbl)
        t.body().addWidget(self._task_reqs_lbl)

        # ── Tool context ──────────────────────────────────────────────────
        tc = Card("Tool Context")
        L.addWidget(tc)
        self._tc_lbl = _stat_label("(none)")
        self._tc_lbl.setWordWrap(True)
        self._tc_failure_lbl = _stat_label("Last failure: —")
        self._tc_failure_lbl.setWordWrap(True)
        tc.body().addWidget(self._tc_lbl)
        tc.body().addWidget(self._tc_failure_lbl)

        # ── User state ────────────────────────────────────────────────────
        us = Card("Companion Sense")
        L.addWidget(us)
        self._mood_sense_lbl   = _stat_label("Mood: —")
        self._energy_sense_lbl = _stat_label("Energy: —")
        us.body().addWidget(self._mood_sense_lbl)
        us.body().addWidget(self._energy_sense_lbl)

        # ── Services ──────────────────────────────────────────────────────
        sv = Card("Background Services")
        L.addWidget(sv)
        self._svc_rows: dict[str, QLabel] = {}
        for name in ("speech_loop", "initiative_loop", "reminder_loop", "reflection_loop"):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            dot = _stat_label("○", TEXT_DIM)
            lbl = _stat_label(name)
            self._svc_rows[name] = dot
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            sv.body().addLayout(row)

        # ── System ────────────────────────────────────────────────────────
        sys_c = Card("System")
        L.addWidget(sys_c)
        self._ollama_lbl   = _stat_label("Ollama: checking…")
        self._model_lbl    = _stat_label("Model: —")
        self._tts_lbl      = _stat_label("TTS: —")
        self._voice_lbl    = _stat_label("Voice: —")
        self._mem_lbl      = _stat_label("Memory entries: —")
        for w in (self._ollama_lbl, self._model_lbl, self._tts_lbl,
                  self._voice_lbl, self._mem_lbl):
            sys_c.body().addWidget(w)

        L.addStretch()

    def update_from_snap(self, snap: dict):
        # Conversation state
        cs = snap.get("conversation", {})
        topic   = cs.get("current_topic", "") or "—"
        threads = cs.get("open_threads", [])
        self._topic_lbl.setText(f"Topic: {topic[:60]}")
        self._threads_lbl.setText(
            "Threads: " + (", ".join(str(t)[:40] for t in threads[:4]) or "—")
        )

        # Active task
        task = snap.get("active_task")
        if task:
            self._task_goal_lbl.setText(f"Goal: {task.get('goal','—')[:60]}")
            self._task_status_lbl.setText(f"Status: {task.get('status','—')}")
            reqs = task.get("requirements", [])
            self._task_reqs_lbl.setText(
                "Reqs: " + (", ".join(str(r) for r in reqs[:4]) if reqs else "—")
            )
        else:
            self._task_goal_lbl.setText("Goal: (none)")
            self._task_status_lbl.setText("Status: —")
            self._task_reqs_lbl.setText("Reqs: —")

        # Tool context
        tc = snap.get("tool_context", {})
        tc_parts = []
        if tc.get("last_note_title"):
            tc_parts.append(f"Note: {tc['last_note_title']}")
        if tc.get("last_note_path"):
            tc_parts.append(f"Path: {tc['last_note_path']}")
        self._tc_lbl.setText(", ".join(tc_parts) or "(none)")

        tf = snap.get("tool_failure", {})
        if tf.get("recent_failure"):
            self._tc_failure_lbl.setText(
                f"⚠ {tf.get('where','?')}: {str(tf.get('error',''))[:60]}"
            )
            self._tc_failure_lbl.setStyleSheet(
                f"color:{RED};background:transparent;border:none;"
            )
        else:
            self._tc_failure_lbl.setText("No recent failures")
            self._tc_failure_lbl.setStyleSheet(
                f"color:{TEXT_DIM};background:transparent;border:none;"
            )

        # User state
        us = snap.get("user_state", {})
        self._mood_sense_lbl.setText(f"Mood: {us.get('mood','—')}")
        self._energy_sense_lbl.setText(f"Energy: {us.get('energy','—')}")

        # Services
        svcs = snap.get("services", {})
        for name, dot in self._svc_rows.items():
            alive = svcs.get(name, False)
            dot.setText("●" if alive else "○")
            dot.setStyleSheet(
                f"color:{'#69f0ae' if alive else TEXT_DIM};"
                "background:transparent;border:none;"
            )

        # System
        tts_on   = snap.get("tts_enabled", False)
        voice_on = snap.get("voice_enabled", False)
        self._tts_lbl.setText(f"TTS: {'ON' if tts_on else 'OFF'}")
        self._tts_lbl.setStyleSheet(
            f"color:{GREEN if tts_on else TEXT_DIM};background:transparent;border:none;"
        )
        self._voice_lbl.setText(f"Voice: {'ON' if voice_on else 'OFF'}")
        self._voice_lbl.setStyleSheet(
            f"color:{GREEN if voice_on else TEXT_DIM};background:transparent;border:none;"
        )

    def refresh_system(self):
        model = os.environ.get("OLLAMA_MODEL", "llama3")
        self._model_lbl.setText(f"Model: {model}")
        if KNOWLEDGE_INDEX_FILE.exists():
            try:
                idx = json.loads(KNOWLEDGE_INDEX_FILE.read_text())
                self._mem_lbl.setText(f"Memory entries: {len(idx.get('ids', []))}")
            except Exception:
                pass
        try:
            import urllib.request as _ur
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            _ur.urlopen(f"{host}/api/tags", timeout=1)
            self._ollama_lbl.setText("Ollama: ● online")
            self._ollama_lbl.setStyleSheet(
                f"color:{GREEN};background:transparent;border:none;"
            )
        except Exception:
            self._ollama_lbl.setText("Ollama: ○ offline")
            self._ollama_lbl.setStyleSheet(
                f"color:{RED};background:transparent;border:none;"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Goals / Relationship tab
# ─────────────────────────────────────────────────────────────────────────────
class GoalsTab(QWidget):
    command_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{NIGHT_MID};")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea{{border:none;background:{NIGHT_MID};}}
            QScrollBar:vertical{{background:{NIGHT_MID};width:4px;border-radius:2px;}}
            QScrollBar::handle:vertical{{background:{CYAN_DIM};border-radius:2px;min-height:16px;}}
        """)
        inner = QWidget()
        inner.setStyleSheet(f"background:{NIGHT_MID};")
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll)
        self._build()

    def _build(self):
        L = self._lay

        # Relationship
        rel_c = Card("Relationship")
        L.addWidget(rel_c)
        self._interactions_lbl = _stat_label("Interactions: —")
        self._since_lbl        = _stat_label("Friends since: —")
        self._last_seen_lbl    = _stat_label("Last seen: —")
        self._topics_lbl       = _stat_label("Interests: —")
        self._topics_lbl.setWordWrap(True)
        for w in (self._interactions_lbl, self._since_lbl,
                  self._last_seen_lbl, self._topics_lbl):
            rel_c.body().addWidget(w)

        # Goals
        goals_c = Card("Goals")
        L.addWidget(goals_c)
        self._goals_lay = goals_c.body()
        self._no_goals_lbl = _stat_label("No active goals yet")
        self._goals_lay.addWidget(self._no_goals_lbl)

        add_btn = QPushButton("+ Add goal…")
        add_btn.setFont(QFont(FONT_DISPLAY, 8))
        add_btn.setFixedHeight(26)
        add_btn.setStyleSheet(f"""
            QPushButton{{background:{NIGHT};color:{CYAN};border:1px solid {CYAN_DIM};
                        border-radius:4px;}}
            QPushButton:hover{{background:{NIGHT_LIGHT};}}""")
        add_btn.clicked.connect(self._on_add_goal)
        goals_c.body().addWidget(add_btn)

        # Curiosity
        cur_c = Card("Curiosity")
        L.addWidget(cur_c)
        self._curiosity_lbl = _stat_label("Loading…")
        self._curiosity_lbl.setWordWrap(True)
        cur_c.body().addWidget(self._curiosity_lbl)

        # Journal
        jrn_c = Card("Recent Journal")
        L.addWidget(jrn_c)
        self._journal_txt = QTextEdit()
        self._journal_txt.setReadOnly(True)
        self._journal_txt.setFixedHeight(130)
        self._journal_txt.setFont(QFont(FONT_DISPLAY, 8))
        self._journal_txt.setStyleSheet(f"""
            QTextEdit{{background:{NIGHT};color:{TEXT_DIM};border:none;border-radius:4px;}}""")
        jrn_c.body().addWidget(self._journal_txt)

        L.addStretch()

    def _on_add_goal(self):
        self.command_requested.emit("/goal ")

    def refresh(self):
        self._refresh_rel()
        self._refresh_goals()
        self._refresh_curiosity()
        self._refresh_journal()

    def _refresh_rel(self):
        if not RELATIONSHIP_FILE.exists():
            return
        try:
            rel = json.loads(RELATIONSHIP_FILE.read_text(encoding="utf-8"))
            self._interactions_lbl.setText(f"Interactions: {rel.get('interaction_count',0)}")
            self._since_lbl.setText(f"Friends since: {rel.get('first_seen','—')}")
            self._last_seen_lbl.setText(f"Last seen: {rel.get('last_seen','—')}")
            topics = rel.get("favorite_topics", [])[:5]
            self._topics_lbl.setText(
                "Interests: " + (", ".join(topics) if topics else "—")
            )
        except Exception:
            pass

    def _refresh_goals(self):
        # Remove old dynamic goal rows (keep static items: title, div, no-goals, add btn)
        remove = []
        for i in range(self._goals_lay.count()):
            item = self._goals_lay.itemAt(i)
            w = item.widget() if item else None
            if w and isinstance(w, QWidget) and hasattr(w, "_goal_row"):
                remove.append(w)
        for w in remove:
            self._goals_lay.removeWidget(w)
            w.deleteLater()

        if not GOALS_FILE.exists():
            self._no_goals_lbl.setVisible(True)
            return
        try:
            goals  = json.loads(GOALS_FILE.read_text(encoding="utf-8"))
            active = [g for g in goals if g.get("status") == "active"]
        except Exception:
            return

        self._no_goals_lbl.setVisible(not active)

        for g in active[:6]:
            pct = int(g.get("progress", 0) * 100)
            row = QWidget()
            row._goal_row = True
            row.setStyleSheet("background:transparent;border:none;")
            rl = QVBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 2)
            rl.setSpacing(2)
            lbl = QLabel(g.get("goal", "")[:50])
            lbl.setFont(QFont(FONT_DISPLAY, 8))
            lbl.setStyleSheet(f"color:{TEXT_MAIN};background:transparent;border:none;")
            pct_lbl = QLabel(f"{pct}%")
            pct_lbl.setFont(QFont(FONT_DISPLAY, 7))
            pct_lbl.setStyleSheet(f"color:{CYAN_DIM};background:transparent;border:none;")
            bar = QProgressBar()
            bar.setValue(pct)
            bar.setFixedHeight(4)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar{{background:{NIGHT};border-radius:2px;border:none;}}
                QProgressBar::chunk{{background:{CYAN};border-radius:2px;}}""")
            header_row = QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)
            header_row.addWidget(lbl)
            header_row.addStretch()
            header_row.addWidget(pct_lbl)
            rl.addLayout(header_row)
            rl.addWidget(bar)
            # Insert before the add-button (second-to-last item)
            insert_pos = max(0, self._goals_lay.count() - 1)
            self._goals_lay.insertWidget(insert_pos, row)

    def _refresh_curiosity(self):
        if not CURIOSITY_FILE.exists():
            self._curiosity_lbl.setText("No curiosity data yet")
            return
        try:
            cs   = json.loads(CURIOSITY_FILE.read_text(encoding="utf-8"))
            asked = cs.get("asked_topics", [])[-5:]
            dates = cs.get("last_question_date", {})
            if asked:
                parts = [f"{t} ({dates.get(t, '?')})" for t in reversed(asked)]
                self._curiosity_lbl.setText("Recent topics: " + ", ".join(parts))
            else:
                self._curiosity_lbl.setText("No curiosity questions asked yet")
        except Exception:
            pass

    def _refresh_journal(self):
        today = date.today()
        entries = []
        for i in range(4):
            d = today - __import__("datetime").timedelta(days=i)
            path = JOURNAL_DIR / f"{d.isoformat()}.md"
            if path.exists():
                try:
                    entries.append(path.read_text(encoding="utf-8")[:400])
                except Exception:
                    pass
        self._journal_txt.setPlainText(
            "\n\n─────\n\n".join(entries) if entries else "No journal entries yet."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MES (plugin evolution) tab
# ─────────────────────────────────────────────────────────────────────────────
class MESTab(QWidget):
    approve_requested = pyqtSignal(str)
    reject_requested  = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{NIGHT_MID};")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea{{border:none;background:{NIGHT_MID};}}
            QScrollBar:vertical{{background:{NIGHT_MID};width:4px;border-radius:2px;}}
            QScrollBar::handle:vertical{{background:{CYAN_DIM};border-radius:2px;min-height:16px;}}
        """)
        inner = QWidget()
        inner.setStyleSheet(f"background:{NIGHT_MID};")
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll)
        self._build()

    def _build(self):
        L = self._lay

        # Metrics
        met_c = Card("Plugin Metrics")
        L.addWidget(met_c)
        self._created_lbl  = _stat_label("Created: —")
        self._approved_lbl = _stat_label("Approved: —")
        self._rejected_lbl = _stat_label("Rejected: —")
        self._gained_lbl   = _stat_label("Capabilities gained: —")
        for w in (self._created_lbl, self._approved_lbl,
                  self._rejected_lbl, self._gained_lbl):
            met_c.body().addWidget(w)

        # Capability gaps
        gap_c = Card("Capability Gaps")
        L.addWidget(gap_c)
        self._gaps_table = QTableWidget(0, 2)
        self._gaps_table.setHorizontalHeaderLabels(["Capability", "Requests"])
        self._gaps_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._gaps_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._gaps_table.setFixedHeight(120)
        self._gaps_table.setStyleSheet(f"""
            QTableWidget{{background:{NIGHT};color:{TEXT_MAIN};
                          border:none;gridline-color:{BORDER};}}
            QHeaderView::section{{background:{NIGHT_LIGHT};color:{CYAN_DIM};
                                  border:none;padding:4px;font-size:8pt;}}
            QTableWidget::item{{padding:4px;}}""")
        self._gaps_table.verticalHeader().setVisible(False)
        gap_c.body().addWidget(self._gaps_table)

        # Pending approvals
        pend_c = Card("Pending Approvals")
        L.addWidget(pend_c)
        self._pending_lay = pend_c.body()
        self._no_pending_lbl = _stat_label("No plugins awaiting approval")
        self._pending_lay.addWidget(self._no_pending_lbl)

        # Approval history
        hist_c = Card("Approval History")
        L.addWidget(hist_c)
        self._history_table = QTableWidget(0, 3)
        self._history_table.setHorizontalHeaderLabels(["Plugin", "Action", "Date"])
        self._history_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._history_table.setFixedHeight(130)
        self._history_table.setStyleSheet(self._gaps_table.styleSheet())
        self._history_table.verticalHeader().setVisible(False)
        hist_c.body().addWidget(self._history_table)

        L.addStretch()

    def refresh(self, snap: dict | None = None):
        self._refresh_metrics()
        self._refresh_gaps()
        self._refresh_pending(snap)
        self._refresh_history()

    def _refresh_metrics(self):
        if not EVOLUTION_METRICS_FILE.exists():
            return
        try:
            m = json.loads(EVOLUTION_METRICS_FILE.read_text(encoding="utf-8"))
            self._created_lbl.setText(f"Created: {m.get('plugins_created',0)}")
            self._approved_lbl.setText(f"Approved: {m.get('plugins_approved',0)}")
            self._approved_lbl.setStyleSheet(
                f"color:{GREEN};background:transparent;border:none;"
            )
            self._rejected_lbl.setText(f"Rejected: {m.get('plugins_rejected',0)}")
            self._rejected_lbl.setStyleSheet(
                f"color:{RED};background:transparent;border:none;"
            )
            self._gained_lbl.setText(f"Capabilities gained: {m.get('capabilities_gained',0)}")
            self._gained_lbl.setStyleSheet(
                f"color:{CYAN};background:transparent;border:none;"
            )
        except Exception:
            pass

    def _refresh_gaps(self):
        if not CAPABILITY_REQUESTS.exists():
            return
        try:
            reqs = json.loads(CAPABILITY_REQUESTS.read_text(encoding="utf-8"))
            sorted_reqs = sorted(reqs.items(), key=lambda x: x[1], reverse=True)[:10]
            self._gaps_table.setRowCount(len(sorted_reqs))
            for row, (cap, cnt) in enumerate(sorted_reqs):
                self._gaps_table.setItem(row, 0, QTableWidgetItem(cap))
                cnt_item = QTableWidgetItem(str(cnt))
                cnt_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._gaps_table.setItem(row, 1, cnt_item)
        except Exception:
            pass

    def _refresh_pending(self, snap: dict | None):
        # Remove old pending rows
        remove = []
        for i in range(self._pending_lay.count()):
            item = self._pending_lay.itemAt(i)
            w = item.widget() if item else None
            if w and isinstance(w, QWidget) and hasattr(w, "_pending_row"):
                remove.append(w)
        for w in remove:
            self._pending_lay.removeWidget(w)
            w.deleteLater()

        pending = []
        if snap:
            pending = snap.get("pending_plugins", [])
        else:
            if EVOLUTION_PENDING_DIR.exists():
                for mf in EVOLUTION_PENDING_DIR.glob("*.json"):
                    try:
                        pending.append(json.loads(mf.read_text(encoding="utf-8")))
                    except Exception:
                        pass

        self._no_pending_lbl.setVisible(not pending)
        for p in pending:
            name = p.get("plugin_name", "?")
            cap  = p.get("capability", "?")
            rev  = p.get("review_reason", "")
            row  = QWidget()
            row._pending_row = True
            row.setStyleSheet(f"background:{NIGHT};border:1px solid {BORDER};"
                              "border-radius:6px;")
            rl = QVBoxLayout(row)
            rl.setContentsMargins(8, 6, 8, 6)
            rl.setSpacing(4)

            name_lbl = QLabel(f"<b>{name}</b>  <span style='color:{CYAN_DIM}'>{cap}</span>")
            name_lbl.setFont(QFont(FONT_DISPLAY, 8))
            name_lbl.setStyleSheet(f"color:{TEXT_MAIN};background:transparent;border:none;")
            rl.addWidget(name_lbl)

            if rev:
                rev_lbl = QLabel(rev[:80])
                rev_lbl.setFont(QFont(FONT_DISPLAY, 7))
                rev_lbl.setStyleSheet(f"color:{TEXT_DIM};background:transparent;border:none;")
                rev_lbl.setWordWrap(True)
                rl.addWidget(rev_lbl)

            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 0, 0, 0)
            btn_row.setSpacing(6)

            approve_btn = QPushButton("✓ Approve")
            approve_btn.setFixedHeight(24)
            approve_btn.setFont(QFont(FONT_DISPLAY, 8))
            approve_btn.setStyleSheet(f"""
                QPushButton{{background:{NIGHT_LIGHT};color:{GREEN};
                             border:1px solid {GREEN};border-radius:4px;}}
                QPushButton:hover{{background:#1a3020;}}""")
            _n = name
            approve_btn.clicked.connect(lambda _, n=_n: self.approve_requested.emit(n))

            reject_btn = QPushButton("✗ Reject")
            reject_btn.setFixedHeight(24)
            reject_btn.setFont(QFont(FONT_DISPLAY, 8))
            reject_btn.setStyleSheet(f"""
                QPushButton{{background:{NIGHT_LIGHT};color:{RED};
                             border:1px solid {RED};border-radius:4px;}}
                QPushButton:hover{{background:#2a1010;}}""")
            reject_btn.clicked.connect(lambda _, n=_n: self.reject_requested.emit(n))

            btn_row.addWidget(approve_btn)
            btn_row.addWidget(reject_btn)
            btn_row.addStretch()
            rl.addLayout(btn_row)
            self._pending_lay.addWidget(row)

    def _refresh_history(self):
        if not EVOLUTION_METRICS_FILE.exists():
            return
        try:
            m       = json.loads(EVOLUTION_METRICS_FILE.read_text(encoding="utf-8"))
            history = m.get("history", [])[-15:]
            self._history_table.setRowCount(len(history))
            for row, entry in enumerate(reversed(history)):
                action = entry.get("action", "?")
                color  = GREEN if action == "approved" else RED
                plugin_item = QTableWidgetItem(entry.get("plugin", "?"))
                action_item = QTableWidgetItem(action)
                action_item.setForeground(QColor(color))
                date_raw = entry.get("date", "")
                try:
                    date_str = datetime.fromisoformat(date_raw).strftime("%m-%d %H:%M")
                except Exception:
                    date_str = date_raw[:16]
                date_item = QTableWidgetItem(date_str)
                date_item.setForeground(QColor(TEXT_DIM))
                self._history_table.setItem(row, 0, plugin_item)
                self._history_table.setItem(row, 1, action_item)
                self._history_table.setItem(row, 2, date_item)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Knowledge tab
# ─────────────────────────────────────────────────────────────────────────────
class KnowledgeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{NIGHT_MID};")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea{{border:none;background:{NIGHT_MID};}}
            QScrollBar:vertical{{background:{NIGHT_MID};width:4px;border-radius:2px;}}
            QScrollBar::handle:vertical{{background:{CYAN_DIM};border-radius:2px;min-height:16px;}}
        """)
        inner = QWidget()
        inner.setStyleSheet(f"background:{NIGHT_MID};")
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll)
        self._build()

    def _build(self):
        L = self._lay

        # Memory overview
        mem_c = Card("Memory Index")
        L.addWidget(mem_c)
        self._total_lbl    = _stat_label("Total entries: —")
        self._high_conf_lbl= _stat_label("High-confidence: —")
        self._top_tags_lbl = _stat_label("Top tags: —")
        self._top_tags_lbl.setWordWrap(True)
        for w in (self._total_lbl, self._high_conf_lbl, self._top_tags_lbl):
            mem_c.body().addWidget(w)

        # Knowledge graph
        graph_c = Card("Knowledge Graph")
        L.addWidget(graph_c)
        self._nodes_lbl    = _stat_label("Nodes: —")
        self._edges_lbl    = _stat_label("Edges: —")
        self._top_nodes_lbl= _stat_label("Most connected: —")
        self._top_nodes_lbl.setWordWrap(True)
        for w in (self._nodes_lbl, self._edges_lbl, self._top_nodes_lbl):
            graph_c.body().addWidget(w)

        # Reflection
        refl_c = Card("Latest Reflection")
        L.addWidget(refl_c)
        self._reflection_txt = QTextEdit()
        self._reflection_txt.setReadOnly(True)
        self._reflection_txt.setFixedHeight(150)
        self._reflection_txt.setFont(QFont(FONT_DISPLAY, 8))
        self._reflection_txt.setStyleSheet(f"""
            QTextEdit{{background:{NIGHT};color:{TEXT_DIM};border:none;border-radius:4px;}}""")
        refl_c.body().addWidget(self._reflection_txt)

        L.addStretch()

    def refresh(self):
        self._refresh_memory()
        self._refresh_graph()
        self._refresh_reflection()

    def _refresh_memory(self):
        if not KNOWLEDGE_INDEX_FILE.exists():
            return
        try:
            idx = json.loads(KNOWLEDGE_INDEX_FILE.read_text(encoding="utf-8"))
            ids = idx.get("ids", [])
            self._total_lbl.setText(f"Total entries: {len(ids)}")

            scores_file = VAULT_PATH / "knowledge" / "_memory_scores.json"
            if scores_file.exists():
                scores = json.loads(scores_file.read_text(encoding="utf-8"))
                high = sum(1 for v in scores.values() if v.get("confidence", 0) >= 0.8)
                self._high_conf_lbl.setText(f"High-confidence: {high}")

            tag_counts = idx.get("tag_counts", {})
            top = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            self._top_tags_lbl.setText(
                "Top tags: " + (", ".join(f"{t}×{c}" for t, c in top) if top else "—")
            )
        except Exception:
            pass

    def _refresh_graph(self):
        if not KNOWLEDGE_GRAPH_FILE.exists():
            return
        try:
            graph = json.loads(KNOWLEDGE_GRAPH_FILE.read_text(encoding="utf-8"))
            nodes = graph.get("nodes", {})
            edges = graph.get("edges", [])
            self._nodes_lbl.setText(f"Nodes: {len(nodes)}")
            self._edges_lbl.setText(f"Edges: {len(edges)}")
            top = sorted(nodes.items(), key=lambda x: x[1], reverse=True)[:5]
            self._top_nodes_lbl.setText(
                "Most connected: " + (", ".join(n for n, _ in top) if top else "—")
            )
        except Exception:
            pass

    def _refresh_reflection(self):
        if not REFLECTION_FILE.exists():
            self._reflection_txt.setPlainText("No reflections yet.")
            return
        try:
            content = REFLECTION_FILE.read_text(encoding="utf-8")
            blocks  = [b.strip() for b in content.strip().split("---") if b.strip()]
            self._reflection_txt.setPlainText(blocks[-1] if blocks else "No reflections yet.")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Right panel (tabbed dashboard)
# ─────────────────────────────────────────────────────────────────────────────
class CompanionPanel(QWidget):
    voice_toggled     = pyqtSignal(bool)
    tts_toggled       = pyqtSignal(bool)
    command_requested = pyqtSignal(str)
    approve_requested = pyqtSignal(str)
    reject_requested  = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self._voice_on = False
        self._tts_on   = True
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._periodic_refresh)
        self._timer.start(5000)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background:{NIGHT_MID};")

        # ── Header ────────────────────────────────────────────────────────
        hdr = QFrame()
        hdr.setStyleSheet(f"""
            QFrame{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0d1e36,stop:1 {NIGHT_MID});
                    border-bottom:1px solid {BORDER};}}""")
        hdr.setFixedHeight(90)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 8, 10, 8)

        self.avatar = AvatarWidget()
        hl.addWidget(self.avatar)

        nc = QVBoxLayout()
        nc.setSpacing(2)
        name_lbl = QLabel("Miku")
        name_lbl.setFont(QFont(FONT_DISPLAY, 15, QFont.Weight.Bold))
        name_lbl.setStyleSheet(f"color:{CYAN};background:transparent;border:none;")
        sub_lbl  = QLabel("your companion")
        sub_lbl.setFont(QFont(FONT_DISPLAY, 8))
        sub_lbl.setStyleSheet(f"color:{TEXT_DIM};background:transparent;border:none;")
        self._mood_lbl = QLabel("● ready")
        self._mood_lbl.setFont(QFont(FONT_DISPLAY, 8))
        self._mood_lbl.setStyleSheet(f"color:{GREEN};background:transparent;border:none;")
        nc.addWidget(name_lbl)
        nc.addWidget(sub_lbl)
        nc.addWidget(self._mood_lbl)
        hl.addLayout(nc)
        hl.addStretch()
        root.addWidget(hdr)

        # ── Voice / TTS bar ───────────────────────────────────────────────
        btn_bar = QWidget()
        btn_bar.setFixedHeight(42)
        btn_bar.setStyleSheet(f"background:{NIGHT_MID};border-bottom:1px solid {BORDER};")
        bl = QHBoxLayout(btn_bar)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(6)
        self._voice_btn = self._toggle_btn("🎙 Voice")
        self._tts_btn   = self._toggle_btn("🔊 TTS")
        self._tts_btn.setChecked(True)
        self._tts_btn.setText("🔊 TTS ON")
        self._voice_btn.clicked.connect(self._toggle_voice)
        self._tts_btn.clicked.connect(self._toggle_tts)
        bl.addWidget(self._voice_btn)
        bl.addWidget(self._tts_btn)
        bl.addStretch()
        root.addWidget(btn_bar)

        # ── Tabs ──────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane{{border:none;background:{NIGHT_MID};}}
            QTabBar::tab{{background:{NIGHT_LIGHT};color:{TEXT_DIM};
                          padding:5px 10px;font-size:8pt;
                          border:none;border-bottom:2px solid transparent;}}
            QTabBar::tab:selected{{color:{CYAN};border-bottom:2px solid {CYAN};
                                   background:{NIGHT_MID};}}
            QTabBar::tab:hover{{color:{TEXT_MAIN};}}""")

        self.dash_tab  = DashboardTab()
        self.goals_tab = GoalsTab()
        self.mes_tab   = MESTab()
        self.know_tab  = KnowledgeTab()

        self._tabs.addTab(self.dash_tab,  "Live")
        self._tabs.addTab(self.goals_tab, "Goals")
        self._tabs.addTab(self.mes_tab,   "MES")
        self._tabs.addTab(self.know_tab,  "Memory")

        self.goals_tab.command_requested.connect(self.command_requested)
        self.mes_tab.approve_requested.connect(self.approve_requested)
        self.mes_tab.reject_requested.connect(self.reject_requested)

        root.addWidget(self._tabs)

        # ── Quick actions (bottom bar) ────────────────────────────────────
        qa_bar = QWidget()
        qa_bar.setFixedHeight(34)
        qa_bar.setStyleSheet(f"background:{NIGHT};border-top:1px solid {BORDER};")
        ql = QHBoxLayout(qa_bar)
        ql.setContentsMargins(6, 4, 6, 4)
        ql.setSpacing(4)
        for icon, cmd in [("📔", "/journal 3"), ("🪞", "/reflect"),
                           ("🧠", "/memory"),  ("🎯", "/goals"),
                           ("⚗️", "/evolve")]:
            b = QPushButton(icon)
            b.setFixedSize(28, 26)
            b.setFont(QFont(FONT_DISPLAY, 10))
            b.setToolTip(cmd)
            b.setStyleSheet(f"""
                QPushButton{{background:transparent;border:none;color:{TEXT_DIM};}}
                QPushButton:hover{{color:{CYAN};background:{NIGHT_LIGHT};border-radius:4px;}}""")
            _c = cmd
            b.clicked.connect(lambda _, c=_c: self.command_requested.emit(c))
            ql.addWidget(b)
        ql.addStretch()
        root.addWidget(qa_bar)

    def _toggle_btn(self, label: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setFont(QFont(FONT_DISPLAY, 8, QFont.Weight.Medium))
        btn.setFixedHeight(28)
        btn.setStyleSheet(f"""
            QPushButton{{background:{NIGHT};color:{TEXT_DIM};
                         border:1px solid {BORDER};border-radius:5px;}}
            QPushButton:checked{{background:{NIGHT_LIGHT};color:{CYAN};
                                 border:1px solid {CYAN_DIM};}}
            QPushButton:hover{{border-color:{CYAN_DIM};}}""")
        return btn

    def _toggle_voice(self):
        self._voice_on = not self._voice_on
        self._voice_btn.setChecked(self._voice_on)
        self._voice_btn.setText("🎙 Voice ON" if self._voice_on else "🎙 Voice")
        self.voice_toggled.emit(self._voice_on)

    def _toggle_tts(self):
        self._tts_on = not self._tts_on
        self._tts_btn.setChecked(self._tts_on)
        self._tts_btn.setText("🔊 TTS ON" if self._tts_on else "🔊 TTS")
        self.tts_toggled.emit(self._tts_on)

    # ── State update from worker ──────────────────────────────────────────
    def update_state(self, snap: dict):
        self.dash_tab.update_from_snap(snap)
        self.mes_tab.refresh(snap)

    # ── Mood display ──────────────────────────────────────────────────────
    def set_ready(self):
        self.avatar.set_speaking(False)
        self.avatar.set_thinking(False)
        self._mood_lbl.setText("● ready")
        self._mood_lbl.setStyleSheet(
            f"color:{GREEN};background:transparent;border:none;"
        )

    def set_thinking(self):
        self.avatar.set_thinking(True)
        self.avatar.set_speaking(False)
        self._mood_lbl.setText("● thinking…")
        self._mood_lbl.setStyleSheet(
            f"color:{AMBER};background:transparent;border:none;"
        )

    def set_speaking(self):
        self.avatar.set_speaking(True)
        self.avatar.set_thinking(False)
        self._mood_lbl.setText("● speaking")
        self._mood_lbl.setStyleSheet(
            f"color:{CYAN};background:transparent;border:none;"
        )

    def add_log(self, text: str):
        # Forward to dashboard activity log (which lives in the DashboardTab's
        # service section — we just emit to the main window's activity log)
        pass  # handled by the activity log in the main window

    # ── Periodic file-based refresh ───────────────────────────────────────
    def _periodic_refresh(self):
        current = self._tabs.currentIndex()
        try:
            if current == 0:
                self.dash_tab.refresh_system()
            elif current == 1:
                self.goals_tab.refresh()
            elif current == 2:
                self.mes_tab.refresh()
            elif current == 3:
                self.know_tab.refresh()
        except Exception:
            pass

    def full_refresh(self):
        self.dash_tab.refresh_system()
        self.goals_tab.refresh()
        self.mes_tab.refresh()
        self.know_tab.refresh()


# ─────────────────────────────────────────────────────────────────────────────
#  Activity log (bottom-of-left-pane)
# ─────────────────────────────────────────────────────────────────────────────
class ActivityLog(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(f"background:{NIGHT_MID};border-top:1px solid {BORDER};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 4, 12, 4)
        self._lbl = QLabel("Initialising…")
        self._lbl.setFont(QFont("Consolas", 7))
        self._lbl.setStyleSheet(f"color:{TEXT_DIM};background:transparent;border:none;")
        self._lbl.setWordWrap(True)
        lay.addWidget(self._lbl)
        self._lines: list[str] = []

    def add(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._lines.append(f"{ts}  {text}")
        self._lines = self._lines[-3:]
        self._lbl.setText("  ·  ".join(self._lines[-2:]))


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────
class MikuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Miku")
        self.resize(1160, 740)
        self.setMinimumSize(720, 520)
        self._apply_palette()
        self._build_ui()
        self._start_worker()
        QTimer.singleShot(400, self._on_startup)

    def _apply_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,     QColor(NIGHT))
        pal.setColor(QPalette.ColorRole.Base,       QColor(NIGHT_MID))
        pal.setColor(QPalette.ColorRole.Text,       QColor(TEXT_MAIN))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_MAIN))
        self.setPalette(pal)
        self.setStyleSheet(f"QMainWindow{{background:{NIGHT};}}")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        main_h.setContentsMargins(0, 0, 0, 0)
        main_h.setSpacing(0)

        # ── Left pane ─────────────────────────────────────────────────────
        left = QWidget()
        left.setStyleSheet(f"background:{NIGHT};")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        # Topbar
        topbar = QFrame()
        topbar.setFixedHeight(44)
        topbar.setStyleSheet(f"background:{NIGHT_MID};border-bottom:1px solid {BORDER};")
        tl = QHBoxLayout(topbar)
        tl.setContentsMargins(14, 0, 14, 0)
        t1 = QLabel("Miku")
        t1.setFont(QFont(FONT_DISPLAY, 12, QFont.Weight.Bold))
        t1.setStyleSheet(f"color:{CYAN};background:transparent;border:none;")
        t2 = QLabel("companion")
        t2.setFont(QFont(FONT_DISPLAY, 9))
        t2.setStyleSheet(f"color:{TEXT_DIM};background:transparent;border:none;")
        self._status_dot  = QLabel("●")
        self._status_dot.setFont(QFont(FONT_DISPLAY, 10))
        self._status_dot.setStyleSheet(f"color:{GREEN};background:transparent;border:none;")
        self._status_text = QLabel("starting…")
        self._status_text.setFont(QFont(FONT_DISPLAY, 8))
        self._status_text.setStyleSheet(f"color:{TEXT_DIM};background:transparent;border:none;")
        tl.addWidget(t1)
        tl.addWidget(t2)
        tl.addStretch()
        tl.addWidget(self._status_dot)
        tl.addWidget(self._status_text)
        left_lay.addWidget(topbar)

        self._chat    = ChatPanel()
        left_lay.addWidget(self._chat)

        self._log_bar = ActivityLog()
        left_lay.addWidget(self._log_bar)

        # ── Right pane ────────────────────────────────────────────────────
        self._sidebar = CompanionPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(self._sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setStyleSheet(
            f"QSplitter::handle{{background:{BORDER};width:1px;}}"
        )
        main_h.addWidget(splitter)

        # ── Wire sidebar signals ──────────────────────────────────────────
        self._chat.message_submitted.connect(self._on_user_msg)
        self._sidebar.voice_toggled.connect(self._on_voice)
        self._sidebar.tts_toggled.connect(self._on_tts)
        self._sidebar.command_requested.connect(self._on_command)
        self._sidebar.approve_requested.connect(self._on_approve)
        self._sidebar.reject_requested.connect(self._on_reject)

    def _start_worker(self):
        self._worker = MikuWorker()
        self._worker.sig_chunk.connect(self._on_chunk)
        self._worker.sig_done.connect(self._on_done)
        self._worker.sig_log.connect(self._on_log)
        self._worker.sig_error.connect(self._on_error)
        self._worker.sig_state.connect(self._on_state)
        self._worker.start()

    def _on_startup(self):
        self._chat.load_history()
        self._chat.add_sys_msg("↑ previous conversation  ───  type below to continue")
        self._sidebar.full_refresh()
        self._set_status("ready", GREEN)

    # ── Worker signal handlers ────────────────────────────────────────────
    def _on_chunk(self, text: str):
        self._chat.feed_stream(text)
        self._sidebar.set_speaking()

    def _on_done(self):
        self._chat.end_stream()
        self._sidebar.set_ready()
        self._set_status("ready", GREEN)

    def _on_log(self, text: str):
        self._log_bar.add(text)

    def _on_error(self, text: str):
        self._chat.add_error_msg(text)
        self._log_bar.add(f"⚠ {text}")
        self._set_status("error", RED)
        QTimer.singleShot(4000, lambda: self._set_status("ready", GREEN))

    def _on_state(self, snap: dict):
        self._sidebar.update_state(snap)
        # If there are pending plugins, highlight the MES tab
        if snap.get("pending_plugins"):
            self._sidebar._tabs.setTabText(
                2, f"MES ⚠{len(snap['pending_plugins'])}"
            )
        else:
            self._sidebar._tabs.setTabText(2, "MES")

    # ── Input handlers ────────────────────────────────────────────────────
    def _on_user_msg(self, text: str, is_voice: bool):
        self._set_status("thinking", AMBER)
        self._sidebar.set_thinking()
        self._chat.start_stream()
        self._worker.send_message(text, is_voice)

    def _on_command(self, cmd: str):
        # /goal  with no text → focus input and prepend the command
        if cmd == "/goal ":
            self._chat._input.setText("/goal ")
            self._chat._input.setFocus()
            return
        self._chat.add_user_msg(cmd)
        self._set_status("thinking", AMBER)
        self._sidebar.set_thinking()
        self._chat.start_stream()
        self._worker.send_message(cmd, False)

    def _on_voice(self, enabled: bool):
        self._worker.set_voice_enabled(enabled)
        self._log_bar.add(f"Voice {'ON' if enabled else 'OFF'}")

    def _on_tts(self, enabled: bool):
        if self._worker.miku:
            self._worker.miku._tts_enabled_flag = enabled
        self._log_bar.add(f"TTS {'ON' if enabled else 'OFF'}")

    def _on_approve(self, name: str):
        self._worker.approve_plugin(name)

    def _on_reject(self, name: str):
        self._worker.reject_plugin(name)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str):
        self._status_text.setText(text)
        self._status_dot.setStyleSheet(
            f"color:{color};background:transparent;border:none;"
        )

    def closeEvent(self, event):
        if self._worker.miku:
            try:
                self._worker.miku._persist_state()
            except Exception:
                pass
        self._worker.stop()
        self._worker.quit()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Miku")
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(NIGHT))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_MAIN))
    pal.setColor(QPalette.ColorRole.Base,            QColor(NIGHT_MID))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(NIGHT_LIGHT))
    pal.setColor(QPalette.ColorRole.Text,            QColor(TEXT_MAIN))
    pal.setColor(QPalette.ColorRole.Button,          QColor(NIGHT_LIGHT))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_MAIN))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(CYAN_DIM))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(NIGHT))
    app.setPalette(pal)

    win = MikuWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════════════════════════
#  GUI MVP EXTENSIONS
#  New panels: IntegrationsTab, HabitsTab
#  Enhanced: CompanionPanel gets two new tabs, MikuWindow gets system-tray hook
# ═══════════════════════════════════════════════════════════════════════════════

HABIT_TRACKER_FILE  = VAULT_PATH / "habit_tracker.json"
REMINDERS_FILE      = VAULT_PATH / "reminders.json"
DISCORD_CONFIG_FILE = VAULT_PATH / "discord_config.json"


# ─────────────────────────────────────────────────────────────────────────────
#  Integrations status tab
# ─────────────────────────────────────────────────────────────────────────────
class IntegrationsTab(QWidget):
    """Shows which integrations are loaded and offers quick-configure shortcuts."""

    command_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{NIGHT_MID};")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea{{border:none;background:{NIGHT_MID};}}
            QScrollBar:vertical{{background:{NIGHT_MID};width:4px;border-radius:2px;}}
            QScrollBar::handle:vertical{{background:{CYAN_DIM};border-radius:2px;min-height:16px;}}
        """)
        inner = QWidget()
        inner.setStyleSheet(f"background:{NIGHT_MID};")
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll)
        self._build()

    def _build(self):
        from PyQt6.QtWidgets import Card as _Card  # noqa — use the top-level Card
        L = self._lay

        # ── Status card ───────────────────────────────────────────────────
        status_c = Card("Plugin Status")
        L.addWidget(status_c)
        self._status_rows: dict[str, QLabel] = {}
        for key, label in [
            ("gmail",          "Gmail"),
            ("calendar",       "Google Calendar"),
            ("discord",        "Discord Bot"),
            ("habit_tracker",  "Habit Tracker"),
            ("obsidian_notes", "Obsidian Notes"),
            ("desktop_notify", "Desktop Notify"),
        ]:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            dot = _stat_label("○", TEXT_DIM)
            lbl = _stat_label(label)
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            status_c.body().addLayout(row)
            self._status_rows[key] = dot

        # ── Quick commands ────────────────────────────────────────────────
        cmd_c = Card("Quick Commands")
        L.addWidget(cmd_c)

        buttons = [
            ("📬 Gmail inbox",      "/gmail inbox"),
            ("📅 Calendar today",   "/calendar today"),
            ("💬 Discord messages", "/discord messages"),
            ("📋 Habit status",     "/habits today"),
            ("🔔 Test notification","/notify Miku is running!"),
            ("🔌 Show status",      "/integrations"),
        ]
        for label, cmd in buttons:
            btn = QPushButton(label)
            btn.setFont(QFont(FONT_DISPLAY, 8))
            btn.setFixedHeight(28)
            btn.setStyleSheet(f"""
                QPushButton{{background:{NIGHT};color:{CYAN};
                             border:1px solid {BORDER};border-radius:4px;text-align:left;padding:0 8px;}}
                QPushButton:hover{{background:{NIGHT_LIGHT};border-color:{CYAN_DIM};}}
            """)
            btn.clicked.connect(lambda _=False, c=cmd: self.command_requested.emit(c))
            cmd_c.body().addWidget(btn)

        # ── Setup hints ───────────────────────────────────────────────────
        hint_c = Card("Setup Guide")
        L.addWidget(hint_c)
        hints = QLabel(
            "<b>Gmail / Calendar</b><br>"
            "1. Enable APIs at console.cloud.google.com<br>"
            "2. Download OAuth credentials → <code>gmail_credentials.json</code><br>"
            "3. <code>pip install google-auth-oauthlib google-api-python-client</code><br><br>"
            "<b>Discord Bot</b><br>"
            "1. discord.com/developers → New App → Bot → copy token<br>"
            "2. Create <code>vault/discord_config.json</code> with bot_token<br><br>"
            "<b>Obsidian Notes</b><br>"
            "Set <code>OBSIDIAN_VAULT</code> env var to your vault path<br>"
            "or create <code>obsidian_vault_path.txt</code><br><br>"
            "<b>System Tray</b><br>"
            "<code>pip install pystray pillow</code> + set <code>TRAY_ENABLED=1</code>"
        )
        hints.setFont(QFont(FONT_DISPLAY, 8))
        hints.setWordWrap(True)
        hints.setStyleSheet(f"color:{TEXT_DIM};background:transparent;border:none;")
        hints.setOpenExternalLinks(False)
        hint_c.body().addWidget(hints)

        L.addStretch()

    def refresh(self, snap: dict | None = None):
        """Update dot colours based on which plugins are loaded in main."""
        if snap is None:
            return
        # snap doesn't expose plugin list directly; poll via worker if available
        # For now, check vault files as proxy
        loaded = {
            "gmail":          (BASE_DIR / "gmail_credentials.json").exists()
                              or (VAULT_PATH / "gmail_token.json").exists(),
            "calendar":       (VAULT_PATH / "calendar_token.json").exists()
                              or (BASE_DIR / "calendar_credentials.json").exists(),
            "discord":        DISCORD_CONFIG_FILE.exists(),
            "habit_tracker":  True,   # always available (no auth needed)
            "obsidian_notes": bool(os.environ.get("OBSIDIAN_VAULT"))
                              or (BASE_DIR / "obsidian_vault_path.txt").exists(),
            "desktop_notify": True,   # always available
        }
        for key, dot in self._status_rows.items():
            ok = loaded.get(key, False)
            dot.setText("●" if ok else "○")
            dot.setStyleSheet(
                f"color:{GREEN if ok else TEXT_DIM};background:transparent;border:none;"
            )

    def update_from_snap(self, snap: dict):
        self.refresh(snap)


# ─────────────────────────────────────────────────────────────────────────────
#  Habits panel tab
# ─────────────────────────────────────────────────────────────────────────────
class HabitsTab(QWidget):
    command_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{NIGHT_MID};")
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea{{border:none;background:{NIGHT_MID};}}
            QScrollBar:vertical{{background:{NIGHT_MID};width:4px;border-radius:2px;}}
            QScrollBar::handle:vertical{{background:{CYAN_DIM};border-radius:2px;min-height:16px;}}
        """)
        inner = QWidget()
        inner.setStyleSheet(f"background:{NIGHT_MID};")
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll)
        self._build()

    def _build(self):
        L = self._lay

        # Today's habits card
        today_c = Card("Today's Habits")
        L.addWidget(today_c)
        self._today_lay  = today_c.body()
        self._today_lbl  = _stat_label("No habits yet — add one below")
        self._today_lay.addWidget(self._today_lbl)

        # Streak card
        streak_c = Card("Streaks")
        L.addWidget(streak_c)
        self._streak_lay = streak_c.body()
        self._streak_lbl = _stat_label("No streaks yet")
        self._streak_lay.addWidget(self._streak_lbl)

        # Quick-log card
        quick_c = Card("Quick Actions")
        L.addWidget(quick_c)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        self._new_habit_input = QLineEdit()
        self._new_habit_input.setPlaceholderText("new habit name…")
        self._new_habit_input.setFont(QFont(FONT_DISPLAY, 8))
        self._new_habit_input.setStyleSheet(f"""
            QLineEdit{{background:{NIGHT};color:{TEXT_MAIN};
                       border:1px solid {BORDER};border-radius:4px;padding:4px 8px;}}
        """)
        add_btn = QPushButton("Add")
        add_btn.setFixedSize(48, 26)
        add_btn.setFont(QFont(FONT_DISPLAY, 8))
        add_btn.setStyleSheet(f"""
            QPushButton{{background:{CYAN_DIM};color:{NIGHT};border-radius:4px;}}
            QPushButton:hover{{background:{CYAN};}}
        """)
        add_btn.clicked.connect(self._on_add_habit)
        add_row.addWidget(self._new_habit_input)
        add_row.addWidget(add_btn)
        quick_c.body().addLayout(add_row)

        view_btn = QPushButton("🔄 Refresh habits")
        view_btn.setFont(QFont(FONT_DISPLAY, 8))
        view_btn.setFixedHeight(26)
        view_btn.setStyleSheet(f"""
            QPushButton{{background:{NIGHT};color:{CYAN};
                         border:1px solid {BORDER};border-radius:4px;}}
            QPushButton:hover{{background:{NIGHT_LIGHT};}}
        """)
        view_btn.clicked.connect(self.refresh)
        quick_c.body().addWidget(view_btn)

        L.addStretch()

    def _on_add_habit(self):
        name = self._new_habit_input.text().strip()
        if name:
            self.command_requested.emit(f"/habit add {name}")
            self._new_habit_input.clear()

    def refresh(self):
        self._refresh_today()
        self._refresh_streaks()

    def _refresh_today(self):
        # Clear dynamic rows
        for i in reversed(range(self._today_lay.count())):
            item = self._today_lay.itemAt(i)
            w = item.widget() if item else None
            if w and w is not self._today_lbl:
                self._today_lay.removeWidget(w)
                w.deleteLater()

        if not HABIT_TRACKER_FILE.exists():
            self._today_lbl.setVisible(True)
            return

        try:
            data   = json.loads(HABIT_TRACKER_FILE.read_text(encoding="utf-8"))
            habits = data.get("habits", {})
        except Exception:
            return

        today = date.today().isoformat()
        self._today_lbl.setVisible(not habits)

        for name, h in habits.items():
            comps = h.get("completions", [])
            done  = today in comps
            label = h.get("description", name)
            row   = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            tick  = _stat_label("✓" if done else "○", GREEN if done else TEXT_DIM)
            lbl   = _stat_label(label)
            row.addWidget(tick)
            row.addWidget(lbl)
            row.addStretch()
            if not done:
                log_btn = QPushButton("Log")
                log_btn.setFixedSize(38, 20)
                log_btn.setFont(QFont(FONT_DISPLAY, 7))
                log_btn.setStyleSheet(f"""
                    QPushButton{{background:{NIGHT};color:{CYAN};
                                 border:1px solid {BORDER};border-radius:3px;}}
                    QPushButton:hover{{background:{NIGHT_LIGHT};}}
                """)
                log_btn.clicked.connect(
                    lambda _=False, n=name: self.command_requested.emit(f"/habit log {n}")
                )
                row.addWidget(log_btn)
            w = QWidget()
            w.setStyleSheet("background:transparent;border:none;")
            w.setLayout(row)
            self._today_lay.addWidget(w)

    def _refresh_streaks(self):
        for i in reversed(range(self._streak_lay.count())):
            item = self._streak_lay.itemAt(i)
            w    = item.widget() if item else None
            if w and w is not self._streak_lbl:
                self._streak_lay.removeWidget(w)
                w.deleteLater()

        if not HABIT_TRACKER_FILE.exists():
            self._streak_lbl.setVisible(True)
            return

        try:
            data   = json.loads(HABIT_TRACKER_FILE.read_text(encoding="utf-8"))
            habits = data.get("habits", {})
        except Exception:
            return

        self._streak_lbl.setVisible(not habits)

        def streak(comps):
            import datetime as _dt
            dates = sorted(set(comps), reverse=True)
            today = _dt.date.today()
            s = 0
            exp = today
            for ds in dates:
                try:
                    d = _dt.date.fromisoformat(ds)
                except ValueError:
                    continue
                if d == exp:
                    s += 1; exp = d - _dt.timedelta(days=1)
                elif s == 0 and d == today - _dt.timedelta(days=1):
                    s += 1; exp = d - _dt.timedelta(days=1)
                else:
                    break
            return s

        rows = []
        for name, h in habits.items():
            comps  = h.get("completions", [])
            s      = streak(comps)
            label  = h.get("description", name)
            rows.append((s, f"  {label}: {s}d streak"))
        rows.sort(key=lambda x: x[0], reverse=True)
        for _, text in rows:
            self._streak_lay.addWidget(_stat_label(text))

    def update_from_snap(self, snap: dict):
        pass  # refresh called manually or on timer


# ─────────────────────────────────────────────────────────────────────────────
#  Patch CompanionPanel to add the two new tabs
# ─────────────────────────────────────────────────────────────────────────────
_OrigCompanionPanel = CompanionPanel


class CompanionPanel(_OrigCompanionPanel):  # type: ignore[misc]
    """Extends the original CompanionPanel with Integrations and Habits tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Inject two new tabs into _tabs
        self._integrations_tab = IntegrationsTab()
        self._habits_tab       = HabitsTab()

        self._integrations_tab.command_requested.connect(
            lambda c: self.command_requested.emit(c)
        )
        self._habits_tab.command_requested.connect(
            lambda c: self.command_requested.emit(c)
        )

        self._tabs.addTab(self._integrations_tab, "Integrations")
        self._tabs.addTab(self._habits_tab,        "Habits")

    def full_refresh(self):
        super().full_refresh()
        self._habits_tab.refresh()

    def update_state(self, snap: dict):
        super().update_state(snap)
        self._integrations_tab.update_from_snap(snap)


# ─────────────────────────────────────────────────────────────────────────────
#  Patch MikuWindow._on_state so new tab index stays correct
# ─────────────────────────────────────────────────────────────────────────────
_OrigMikuWindow = MikuWindow


class MikuWindow(_OrigMikuWindow):  # type: ignore[misc]
    def _on_state(self, snap: dict):
        self._sidebar.update_state(snap)
        # Tab 2 = MES, tabs 3/4 = Integrations/Habits
        n_pending = len(snap.get("pending_plugins", []))
        if n_pending:
            self._sidebar._tabs.setTabText(2, f"MES ⚠{n_pending}")
        else:
            self._sidebar._tabs.setTabText(2, "MES")