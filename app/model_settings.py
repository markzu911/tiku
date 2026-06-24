import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


_SETTINGS_PATH = Path(os.getenv("MODEL_SETTINGS_FILE", ".model-provider.json"))
_SETTINGS_LOCK = Lock()
_DEFAULT_PROVIDER = "current"


def _provider_config(provider: str) -> dict[str, str]:
    if provider == "current":
        return {
            "provider": "current",
            "label": f"当前模型（{os.getenv('OPENAI_MODEL', 'gpt-5.5')}）",
            "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            "model": os.getenv("OPENAI_MODEL", "gpt-5.5").strip(),
        }
    if provider == "qwen":
        return {
            "provider": "qwen",
            "label": "Qwen 3.7 Plus",
            "api_key": (os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")).strip(),
            "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/"),
            "model": os.getenv("DASHSCOPE_MODEL", "qwen3.7-plus").strip(),
        }
    raise ValueError("unsupported model provider")


def _read_active_provider() -> str:
    try:
        stored = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        provider = stored.get("provider") if isinstance(stored, dict) else ""
        if provider in {"current", "qwen"}:
            return provider
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return os.getenv("ACTIVE_MODEL_PROVIDER", _DEFAULT_PROVIDER).strip() or _DEFAULT_PROVIDER


def get_active_model_config() -> dict[str, str]:
    return _provider_config(_read_active_provider())


def get_model_provider_status() -> dict[str, Any]:
    active_provider = _read_active_provider()
    options = []
    for provider in ("current", "qwen"):
        config = _provider_config(provider)
        options.append(
            {
                "provider": provider,
                "label": config["label"],
                "model": config["model"],
                "configured": bool(config["api_key"] and config["base_url"] and config["model"]),
            }
        )
    active = _provider_config(active_provider)
    return {
        "provider": active_provider,
        "label": active["label"],
        "model": active["model"],
        "options": options,
    }


def set_active_model_provider(provider: str) -> dict[str, Any]:
    config = _provider_config(provider)
    if not config["api_key"] or not config["base_url"] or not config["model"]:
        raise ValueError(f"{config['label']} is not configured")

    with _SETTINGS_LOCK:
        temp_path = _SETTINGS_PATH.with_suffix(".tmp")
        temp_path.write_text(json.dumps({"provider": provider}), encoding="utf-8")
        temp_path.replace(_SETTINGS_PATH)
    return get_model_provider_status()
