"""
External configuration management with Pydantic

Enterprise-grade configuration with validation, type safety, and environment support
"""

from pathlib import Path
from typing import Literal, Optional

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings

    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings, Field  # type: ignore[no-redef]
        from pydantic import validator as field_validator  # type: ignore[no-redef]

        PYDANTIC_V2 = False
    except ImportError:
        raise ImportError(
            "Config requires pydantic-settings package. "
            "Install with: pip install pydantic-settings"
        )


class EvaluatorConfig(BaseSettings):
    """
    Main evaluator configuration settings

    Loads from environment variables or .env file
    """

    # Model Configuration
    default_model: str = Field(default="llama3.2:1b", description="Default LLM model to evaluate")

    default_provider: Literal["ollama", "openai", "anthropic"] = Field(
        default="ollama", description="Default LLM provider"
    )

    # Generation Settings
    default_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Default temperature for generation"
    )

    default_max_tokens: int = Field(
        default=500, gt=0, description="Default max tokens per response"
    )

    default_timeout: int = Field(default=30, gt=0, description="Default timeout in seconds")

    default_retry_attempts: int = Field(
        default=3, ge=1, le=10, description="Default number of retry attempts"
    )

    # Evaluation Settings
    performance_samples: int = Field(
        default=10, gt=0, description="Number of samples for performance evaluation"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    log_file: Optional[Path] = Field(
        default=None, description="Log file path (None for console only)"
    )

    # Output Settings
    output_dir: Path = Field(
        default=Path.home() / ".llm-benchmark" / "outputs",
        description="Default output directory for reports and results",
    )

    # Environment
    environment: Literal["dev", "test", "prod"] = Field(
        default="dev", description="Runtime environment"
    )

    if PYDANTIC_V2:

        @field_validator("output_dir")
        @classmethod
        def create_output_dir(cls, v: Path) -> Path:
            """Ensure output directory exists"""
            v.mkdir(parents=True, exist_ok=True)
            return v

        model_config = {
            "env_prefix": "LLM_EVAL_",
            "env_file": ".env",
            "env_file_encoding": "utf-8",
        }
    else:

        @field_validator("output_dir")
        def create_output_dir(cls, v: Path) -> Path:  # type: ignore[misc]
            """Ensure output directory exists"""
            v.mkdir(parents=True, exist_ok=True)
            return v

        class Config:
            env_prefix = "LLM_EVAL_"
            env_file = ".env"
            env_file_encoding = "utf-8"


class OllamaConfig(BaseSettings):
    """
    Ollama-specific configuration
    """

    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")

    pull_missing_models: bool = Field(
        default=False, description="Automatically pull missing models"
    )

    keep_alive: Optional[str] = Field(
        default=None, description="Keep model in memory duration (e.g., '5m', '1h')"
    )

    if PYDANTIC_V2:
        model_config = {
            "env_prefix": "OLLAMA_",
            "env_file": ".env",
            "env_file_encoding": "utf-8",
        }
    else:

        class Config:
            env_prefix = "OLLAMA_"
            env_file = ".env"
            env_file_encoding = "utf-8"


class BenchmarkConfig(BaseSettings):
    """
    Benchmark configuration
    """

    use_demo_benchmarks: bool = Field(
        default=True,
        description="Use demo benchmarks (fast) or real datasets (slow)",
    )

    mmlu_subset: Optional[str] = Field(
        default=None, description="MMLU subset to use (e.g., 'abstract_algebra')"
    )

    cache_dir: Optional[Path] = Field(
        default=Path("~/.cache/llm_evaluator").expanduser(),
        description="Cache directory for downloaded datasets",
    )

    if PYDANTIC_V2:

        @field_validator("cache_dir")
        @classmethod
        def create_cache_dir(cls, v: Optional[Path]) -> Optional[Path]:
            """Ensure cache directory exists"""
            if v:
                v.mkdir(parents=True, exist_ok=True)
            return v

        model_config = {
            "env_prefix": "BENCHMARK_",
            "env_file": ".env",
            "env_file_encoding": "utf-8",
        }
    else:

        @field_validator("cache_dir")
        def create_cache_dir(cls, v: Optional[Path]) -> Optional[Path]:  # type: ignore[misc]
            """Ensure cache directory exists"""
            if v:
                v.mkdir(parents=True, exist_ok=True)
            return v

        class Config:
            env_prefix = "BENCHMARK_"
            env_file = ".env"
            env_file_encoding = "utf-8"


# Singleton instances
_evaluator_config: Optional[EvaluatorConfig] = None
_ollama_config: Optional[OllamaConfig] = None
_benchmark_config: Optional[BenchmarkConfig] = None


def get_evaluator_config() -> EvaluatorConfig:
    """Get evaluator configuration singleton"""
    global _evaluator_config
    if _evaluator_config is None:
        _evaluator_config = EvaluatorConfig()
    return _evaluator_config


def get_ollama_config() -> OllamaConfig:
    """Get Ollama configuration singleton"""
    global _ollama_config
    if _ollama_config is None:
        _ollama_config = OllamaConfig()
    return _ollama_config


def get_benchmark_config() -> BenchmarkConfig:
    """Get benchmark configuration singleton"""
    global _benchmark_config
    if _benchmark_config is None:
        _benchmark_config = BenchmarkConfig()
    return _benchmark_config


def reset_config() -> None:
    """Reset all configuration singletons (for testing)"""
    global _evaluator_config, _ollama_config, _benchmark_config
    _evaluator_config = None
    _ollama_config = None
    _benchmark_config = None
