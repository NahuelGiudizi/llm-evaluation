"""
Model Discovery

Detects available models from various providers.
"""

import logging
import os
import subprocess

from .models import ModelInfo, ModelsResponse

logger = logging.getLogger(__name__)


def discover_ollama_models() -> list[ModelInfo]:
    """Detect available Ollama models"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 1:
                models.append(
                    ModelInfo(
                        id=parts[0],
                        name=parts[0],
                        provider="ollama",
                        size=parts[2] if len(parts) > 2 else None,
                        modified=parts[3] if len(parts) > 3 else None,
                    )
                )
        return models
    except Exception as e:
        logger.warning(f"Failed to detect Ollama models: {e}")
        return []


def discover_api_models() -> dict[str, list[ModelInfo]]:
    """Detect available API models from environment variables"""
    models: dict[str, list[ModelInfo]] = {
        "openai": [],
        "anthropic": [],
        "deepseek": [],
        "huggingface": [],
    }

    if os.environ.get("OPENAI_API_KEY"):
        models["openai"] = [
            ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai"),
            ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini", provider="openai"),
            ModelInfo(id="gpt-4-turbo", name="GPT-4 Turbo", provider="openai"),
            ModelInfo(id="gpt-3.5-turbo", name="GPT-3.5 Turbo", provider="openai"),
        ]

    if os.environ.get("ANTHROPIC_API_KEY"):
        models["anthropic"] = [
            ModelInfo(
                id="claude-3-5-sonnet-20241022", name="Claude 3.5 Sonnet", provider="anthropic"
            ),
            ModelInfo(id="claude-3-opus-20240229", name="Claude 3 Opus", provider="anthropic"),
            ModelInfo(id="claude-3-haiku-20240307", name="Claude 3 Haiku", provider="anthropic"),
        ]

    if os.environ.get("DEEPSEEK_API_KEY"):
        models["deepseek"] = [
            ModelInfo(id="deepseek-chat", name="DeepSeek Chat", provider="deepseek"),
            ModelInfo(id="deepseek-coder", name="DeepSeek Coder", provider="deepseek"),
        ]

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"):
        models["huggingface"] = [
            ModelInfo(
                id="meta-llama/Llama-2-7b-chat-hf",
                name="Llama 2 7B Chat",
                provider="huggingface",
            ),
            ModelInfo(
                id="mistralai/Mistral-7B-Instruct-v0.2",
                name="Mistral 7B Instruct",
                provider="huggingface",
            ),
        ]

    return models


def discover_models() -> ModelsResponse:
    """Discover all available models from all providers"""
    ollama_models = discover_ollama_models()
    api_models = discover_api_models()

    return ModelsResponse(
        ollama=ollama_models,
        openai=api_models["openai"],
        anthropic=api_models["anthropic"],
        deepseek=api_models["deepseek"],
        huggingface=api_models["huggingface"],
    )
