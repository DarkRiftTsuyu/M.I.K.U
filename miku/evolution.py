from __future__ import annotations

import importlib
import json
import logging
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path

from .config import (
    EVOLUTION_ENABLED, EVOLUTION_THRESHOLD, EVOLUTION_CODER_MODEL,
    EVOLUTION_AUTO_TEST, EVOLUTION_DIR, EVOLUTION_PENDING_DIR,
    EVOLUTION_SANDBOX_DIR, EVOLUTION_METRICS_FILE, CAPABILITY_REQUESTS,
    BASE_DIR, OLLAMA_HOST, OLLAMA_TIMEOUT_S,
)
from .utils import http_json, urljoin, extract_json_object
from .memory import save_knowledge_entry

log = logging.getLogger("miku.evolution")


def load_evolution_metrics() -> dict:
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


def save_evolution_metrics(metrics: dict) -> None:
    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        EVOLUTION_METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("[MES] Could not save metrics: %s", exc)


def load_capability_requests() -> dict:
    if CAPABILITY_REQUESTS.exists():
        try:
            return json.loads(CAPABILITY_REQUESTS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_capability_requests(requests: dict) -> None:
    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        CAPABILITY_REQUESTS.write_text(json.dumps(requests, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("[MES] Could not save capability requests: %s", exc)


def record_capability_gap(capability: str, request: str) -> None:
    if not EVOLUTION_ENABLED:
        return
    requests = load_capability_requests()
    requests[capability] = requests.get(capability, 0) + 1
    save_capability_requests(requests)
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
    url     = urljoin(OLLAMA_HOST, "/api/generate")
    payload = {
        "model": EVOLUTION_CODER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.95, "num_predict": 800},
    }
    try:
        resp = http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S * 2)
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
    metrics = load_evolution_metrics()
    metrics["plugins_created"] += 1
    save_evolution_metrics(metrics)
    if EVOLUTION_AUTO_TEST:
        passed, reason = _test_plugin_sandbox(sandbox_file, safe_name)
    else:
        passed, reason = True, "auto-test disabled"
    if not passed:
        log.warning("[MES] Plugin '%s' failed testing: %s", safe_name, reason)
        metrics = load_evolution_metrics()
        metrics["plugins_rejected"] += 1
        save_evolution_metrics(metrics)
        return
    review_passed, review_reason = _llm_review_plugin(code)
    if not review_passed:
        log.warning("[MES] Plugin '%s' failed LLM review: %s", safe_name, review_reason)
        metrics = load_evolution_metrics()
        metrics["plugins_rejected"] += 1
        save_evolution_metrics(metrics)
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
        "capability":    capability,
        "plugin_name":   safe_name,
        "status":        "awaiting_approval",
        "created":       datetime.now().isoformat(),
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
    url = urljoin(OLLAMA_HOST, "/api/generate")
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
        resp = http_json("POST", url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S)
        text = (resp or {}).get("response", "")
        obj  = extract_json_object(text)
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


def approve_plugin(plugin_name: str, plugins_registry: dict) -> str:
    safe_name    = re.sub(r"[^a-z0-9_]", "_", plugin_name.lower())[:40]
    pending_py   = EVOLUTION_PENDING_DIR / f"{safe_name}.py"
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
    metrics = load_evolution_metrics()
    metrics["plugins_approved"] += 1
    metrics["capabilities_gained"] += 1
    metrics.setdefault("history", []).append({
        "plugin": safe_name,
        "action": "approved",
        "date":   datetime.now().isoformat(),
    })
    save_evolution_metrics(metrics)
    try:
        module = importlib.import_module(f"plugins.{safe_name}")
        if hasattr(module, "PLUGIN") and hasattr(module, "execute"):
            plugins_registry[module.PLUGIN["name"]] = module
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
    metrics = load_evolution_metrics()
    metrics["plugins_rejected"] += 1
    metrics.setdefault("history", []).append({
        "plugin": safe_name,
        "action": "rejected",
        "date":   datetime.now().isoformat(),
    })
    save_evolution_metrics(metrics)
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


def print_evolution_status() -> None:
    metrics  = load_evolution_metrics()
    requests = load_capability_requests()
    pending  = list_pending_plugins()
    print("\n── Evolution Status ───────────────")
    print(f"  Plugins created    : {metrics.get('plugins_created', 0)}")
    print(f"  Plugins approved   : {metrics.get('plugins_approved', 0)}")
    print(f"  Plugins rejected   : {metrics.get('plugins_rejected', 0)}")
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
