import base64
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()


@dataclass(frozen=True)
class AIProviderConfig:
    provider: str
    api_key: str
    base_url: str
    model: str


def get_ai_provider() -> str:
    provider = os.getenv("AI_PROVIDER", "").strip().lower()
    if not provider:
        openai_base_url = os.getenv("OPENAI_BASE_URL", "").lower()
        provider = "zhipu" if os.getenv("ZHIPU_API_KEY") or "bigmodel.cn" in openai_base_url else "openai"
    if provider in {"bigmodel", "glm", "zhipuai"}:
        return "zhipu"
    return provider


def get_chat_config(task: str) -> AIProviderConfig:
    provider = get_ai_provider()
    if provider == "zhipu":
        api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="请先在 .env 中配置 ZHIPU_API_KEY")

        model = (
            os.getenv("ZHIPU_VISION_MODEL")
            if task == "vision"
            else os.getenv("ZHIPU_TEXT_MODEL")
        ) or os.getenv("ZHIPU_MODEL") or ("glm-4v-plus" if task == "vision" else "glm-4-plus")
        return AIProviderConfig(
            provider=provider,
            api_key=api_key,
            base_url=os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/"),
            model=model,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="please set OPENAI_API_KEY before calling model")

    model = os.getenv("OPENAI_VISION_MODEL" if task == "vision" else "OPENAI_TEXT_MODEL") or os.getenv(
        "OPENAI_MODEL", "gpt-4o-mini"
    )
    return AIProviderConfig(
        provider="openai",
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/"),
        model=model,
    )


def get_image_config() -> AIProviderConfig:
    provider = get_ai_provider()
    if provider == "zhipu":
        api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="请先在 .env 中配置 ZHIPU_API_KEY")
        return AIProviderConfig(
            provider=provider,
            api_key=api_key,
            base_url=os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/"),
            model=os.getenv("ZHIPU_IMAGE_MODEL", "cogview-3-flash"),
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="please set OPENAI_API_KEY before generating images")
    return AIProviderConfig(
        provider="openai",
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/"),
        model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
    )


def chat_completions_url(config: AIProviderConfig) -> str:
    if config.provider == "zhipu":
        return f"{config.base_url}/chat/completions"
    if config.base_url.endswith("/v1"):
        return f"{config.base_url}/chat/completions"
    return f"{config.base_url}/v1/chat/completions"


def image_generations_url(config: AIProviderConfig) -> str:
    if config.provider == "zhipu":
        return f"{config.base_url}/images/generations"
    if config.base_url.endswith("/v1"):
        return f"{config.base_url}/images/generations"
    return f"{config.base_url}/v1/images/generations"


def should_send_response_format() -> bool:
    value = os.getenv("AI_RESPONSE_FORMAT", "json_object").strip().lower()
    return value not in {"", "none", "off", "false", "0"}


async def request_chat_completion(
    messages: list[dict[str, Any]],
    task: str,
    *,
    json_response: bool = True,
    timeout: int = 120,
) -> str:
    config = get_chat_config(task)
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
    }
    if json_response and should_send_response_format():
        payload["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                chat_completions_url(config),
                headers={"Authorization": f"Bearer {config.api_key}"},
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"{config.provider} model API error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"{config.provider} model request failed: {exc}") from exc

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail=f"{config.provider} model response missing content") from exc
    return normalize_chat_content(content)


async def request_image_generation(prompt: str, *, size: str | None = None, timeout: int = 120) -> str:
    config = get_image_config()
    image_size = size or os.getenv("ZHIPU_IMAGE_SIZE" if config.provider == "zhipu" else "OPENAI_IMAGE_SIZE", "1024x1024")
    payload: dict[str, Any] = {
        "model": config.model,
        "prompt": prompt,
        "size": image_size,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                image_generations_url(config),
                headers={"Authorization": f"Bearer {config.api_key}"},
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"{config.provider} image API error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"{config.provider} image request failed: {exc}") from exc

    data = response.json()
    image_items = data.get("data") if isinstance(data, dict) else None
    if not image_items:
        raise HTTPException(status_code=502, detail=f"{config.provider} image response missing data")

    first_image = image_items[0] or {}
    if first_image.get("url"):
        return str(first_image["url"])
    if first_image.get("b64_json"):
        return f"data:image/png;base64,{first_image['b64_json']}"

    image_base64 = first_image.get("base64") or first_image.get("image_base64")
    if image_base64:
        return f"data:image/png;base64,{image_base64}"
    raise HTTPException(status_code=502, detail=f"{config.provider} image response missing url/base64")


def normalize_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(parts)
    return str(content or "")


def is_image_generation_enabled() -> bool:
    return os.getenv("ENABLE_IMAGE_GENERATION", "").strip().lower() in {"1", "true", "yes", "on"}


def to_data_image_url(mime_type: str, image_bytes: bytes) -> str:
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{image_base64}"
