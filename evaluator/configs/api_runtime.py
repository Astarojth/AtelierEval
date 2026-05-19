from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CHAT_ENDPOINT = "/v1/chat/completions"


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or (text.startswith("<") and text.endswith(">"))


def _find_api_modellist(start: Path) -> Optional[Path]:
    current = start.resolve()
    search_roots = [current, *current.parents]
    for root in search_roots:
        candidate = root / "api_modellist.txt"
        if candidate.exists():
            return candidate
    return None


def load_api_spec(start: Path) -> Optional[Dict[str, Any]]:
    path = _find_api_modellist(start)
    if path is None:
        return None

    provider = ""
    api_base = ""
    api_endpoint = DEFAULT_CHAT_ENDPOINT
    api_keys: List[str] = []
    models: List[str] = []
    reading_keys = False
    reading_models = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("provider:"):
            provider = line.split(":", 1)[1].strip()
            reading_keys = False
            reading_models = False
            continue
        if lower.startswith("api_base:"):
            api_base = line.split(":", 1)[1].strip()
            reading_keys = False
            reading_models = False
            continue
        if lower.startswith("api_endpoint:"):
            api_endpoint = line.split(":", 1)[1].strip() or DEFAULT_CHAT_ENDPOINT
            reading_keys = False
            reading_models = False
            continue
        if lower.startswith("api_key:"):
            maybe_key = line.split(":", 1)[1].strip()
            if maybe_key:
                api_keys.append(maybe_key)
            reading_keys = True
            reading_models = False
            continue
        if lower.startswith("model list"):
            reading_keys = False
            reading_models = True
            continue
        if reading_keys:
            api_keys.append(line)
            continue
        if reading_models:
            models.append(line)

    if not api_base and not api_keys and not models:
        return None

    return {
        "path": str(path),
        "provider": provider,
        "api_base": api_base.rstrip("/"),
        "api_endpoint": api_endpoint.strip() or DEFAULT_CHAT_ENDPOINT,
        "api_keys": api_keys,
        "models": models,
    }


def apply_api_spec_to_config(cfg: Dict[str, Any], *, config_path: Path) -> Dict[str, Any]:
    spec = load_api_spec(config_path)
    if spec is None:
        return cfg

    api_cfg = cfg.setdefault("api", {})
    api_key_override = os.getenv("ATELIEREVAL_API_KEY_OVERRIDE", "").strip()
    if _is_placeholder(api_cfg.get("base_url")):
        api_cfg["base_url"] = spec["api_base"]
    if _is_placeholder(api_cfg.get("endpoint")):
        api_cfg["endpoint"] = spec["api_endpoint"]
    if api_key_override:
        api_cfg["api_key"] = api_key_override
    elif _is_placeholder(api_cfg.get("api_key")) and spec["api_keys"]:
        api_cfg["api_key"] = spec["api_keys"][0]
    if spec["models"]:
        api_cfg["model"] = spec["models"][0]
        api_cfg["model_fallbacks"] = spec["models"][1:]
    cfg["_api_runtime_source"] = spec["path"]
    return cfg


def get_model_candidates(api_cfg: Dict[str, Any]) -> List[str]:
    ordered = [str(api_cfg.get("model", "")).strip()]
    ordered.extend(str(item).strip() for item in api_cfg.get("model_fallbacks", []) if str(item).strip())

    unique: List[str] = []
    for name in ordered:
        if name and name not in unique:
            unique.append(name)
    return unique


def build_openai_base_url(api_cfg: Dict[str, Any]) -> str:
    base_url = str(api_cfg.get("base_url", "")).strip().rstrip("/")
    endpoint = str(api_cfg.get("endpoint", DEFAULT_CHAT_ENDPOINT)).strip()
    if not base_url:
        return base_url
    if not endpoint:
        return base_url
    suffix = "/chat/completions"
    if endpoint.endswith(suffix):
        prefix = endpoint[: -len(suffix)].rstrip("/")
        if prefix and not base_url.endswith(prefix):
            return f"{base_url}{prefix}"
    return base_url
