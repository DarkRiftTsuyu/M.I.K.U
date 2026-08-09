import urllib
import json
from . import sharedConst

def _urljoin(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")

def _http_json(method: str, url: str, payload=None, timeout_s: float | None = None):
    data = None
    headers = {"Accept": "application/json"}
    if payload:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    timeout = timeout_s if timeout_s is not None else sharedConst.OLLAMA_TIMEOUT_S
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