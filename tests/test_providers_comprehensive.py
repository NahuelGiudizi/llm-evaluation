"""
Extended tests for API providers to improve coverage
"""

import importlib.util
from unittest.mock import Mock

import pytest


# Check if optional dependencies are available
def has_anthropic():
    return importlib.util.find_spec("anthropic") is not None


def has_openai():
    return importlib.util.find_spec("openai") is not None


class TestAnthropicProviderImports:
    """Test Anthropic provider import paths"""

    @pytest.mark.skipif(not has_anthropic(), reason="anthropic package not installed")
    def test_anthropic_provider_module_exists(self):
        """Test that anthropic_provider module can be imported"""
        from llm_evaluator.providers import anthropic_provider

        assert anthropic_provider is not None

    @pytest.mark.skipif(not has_anthropic(), reason="anthropic package not installed")
    def test_anthropic_provider_class_exists(self):
        """Test that AnthropicProvider class exists"""
        from llm_evaluator.providers.anthropic_provider import AnthropicProvider

        assert AnthropicProvider is not None


class TestOpenAIProviderImports:
    """Test OpenAI provider import paths"""

    @pytest.mark.skipif(not has_openai(), reason="openai package not installed")
    def test_openai_provider_module_exists(self):
        """Test that openai_provider module can be imported"""
        from llm_evaluator.providers import openai_provider

        assert openai_provider is not None

    @pytest.mark.skipif(not has_openai(), reason="openai package not installed")
    def test_openai_provider_class_exists(self):
        """Test that OpenAIProvider class exists"""
        from llm_evaluator.providers.openai_provider import OpenAIProvider

        assert OpenAIProvider is not None


class TestHuggingFaceProviderImports:
    """Test HuggingFace provider import paths"""

    def test_huggingface_provider_module_exists(self):
        """Test that huggingface_provider module can be imported"""
        from llm_evaluator.providers import huggingface_provider

        assert huggingface_provider is not None

    def test_huggingface_provider_class_exists(self):
        """Test that HuggingFaceProvider class exists"""
        from llm_evaluator.providers.huggingface_provider import HuggingFaceProvider

        assert HuggingFaceProvider is not None


class TestDeepSeekProviderImports:
    """Test DeepSeek provider import paths"""

    @pytest.mark.skipif(
        not has_openai(), reason="openai package not installed (DeepSeek uses OpenAI client)"
    )
    def test_deepseek_provider_module_exists(self):
        """Test that deepseek_provider module can be imported"""
        from llm_evaluator.providers import deepseek_provider

        assert deepseek_provider is not None

    @pytest.mark.skipif(
        not has_openai(), reason="openai package not installed (DeepSeek uses OpenAI client)"
    )
    def test_deepseek_provider_class_exists(self):
        """Test that DeepSeekProvider class exists"""
        from llm_evaluator.providers.deepseek_provider import DeepSeekProvider

        assert DeepSeekProvider is not None


class TestProviderBaseClasses:
    """Test provider base class functionality"""

    def test_generation_config_defaults(self):
        """Test GenerationConfig defaults"""
        from llm_evaluator.providers.base import GenerationConfig

        config = GenerationConfig()

        assert config.temperature == 0.7
        assert config.max_tokens == 512  # Default is 512
        assert config.top_p == 0.9  # Default is 0.9

    def test_generation_result_creation(self):
        """Test GenerationResult creation"""
        from llm_evaluator.providers.base import GenerationResult

        result = GenerationResult(
            text="Test text", response_time=0.5, tokens_used=10, model="test-model", metadata={}
        )

        assert result.text == "Test text"
        assert result.response_time == 0.5
        assert result.tokens_used == 10

    def test_provider_error_creation(self):
        """Test ProviderError creation"""
        from llm_evaluator.providers.base import ProviderError

        error = ProviderError("Test error")
        assert str(error) == "Test error"

    def test_provider_error_with_cause(self):
        """Test ProviderError with cause"""
        from llm_evaluator.providers.base import ProviderError

        cause = ValueError("Original error")
        error = ProviderError("Wrapped error")
        error.__cause__ = cause

        assert error.__cause__ is cause


class TestOllamaProviderAdditional:
    """Additional tests for Ollama provider"""

    def test_ollama_provider_init(self):
        """Test OllamaProvider initialization"""
        from llm_evaluator.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(model="llama3.2:1b")

        assert provider.model == "llama3.2:1b"

    def test_ollama_provider_with_custom_base_url(self):
        """Test OllamaProvider with custom base URL"""
        from llm_evaluator.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(model="llama3.2:1b", base_url="http://custom:11434")

        assert provider.model == "llama3.2:1b"
        assert "custom" in provider.base_url

    def test_ollama_provider_with_config(self):
        """Test OllamaProvider with config"""
        from llm_evaluator.providers.base import GenerationConfig
        from llm_evaluator.providers.ollama_provider import OllamaProvider

        config = GenerationConfig(temperature=0.5, max_tokens=1000)
        provider = OllamaProvider(model="llama3.2:1b", config=config)

        assert provider.config.temperature == 0.5
        assert provider.config.max_tokens == 1000


class TestCachedProviderAdditional:
    """Additional tests for CachedProvider"""

    def test_cached_provider_wraps_provider(self):
        """Test CachedProvider wraps base provider correctly"""
        import tempfile

        from llm_evaluator.providers.base import GenerationConfig
        from llm_evaluator.providers.cached_provider import CachedProvider

        mock_provider = Mock()
        mock_provider.model = "test-model"
        mock_provider.config = GenerationConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            cached = CachedProvider(mock_provider, cache_dir=tmpdir)

            assert cached.provider is mock_provider
            assert cached.model == "test-model"

    def test_cached_provider_max_cache_size(self):
        """Test CachedProvider respects max cache size"""
        import tempfile

        from llm_evaluator.providers.base import GenerationConfig
        from llm_evaluator.providers.cached_provider import CachedProvider

        mock_provider = Mock()
        mock_provider.model = "test"
        mock_provider.config = GenerationConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            cached = CachedProvider(mock_provider, cache_dir=tmpdir, max_cache_size=100)

            assert cached.max_cache_size == 100

    def test_cached_provider_ttl(self):
        """Test CachedProvider TTL setting"""
        import tempfile

        from llm_evaluator.providers.base import GenerationConfig
        from llm_evaluator.providers.cached_provider import CachedProvider

        mock_provider = Mock()
        mock_provider.model = "test"
        mock_provider.config = GenerationConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            cached = CachedProvider(mock_provider, cache_dir=tmpdir, ttl_seconds=3600)

            assert cached.ttl_seconds == 3600

    def test_cached_provider_stats_initial(self):
        """Test CachedProvider initial stats"""
        import tempfile

        from llm_evaluator.providers.base import GenerationConfig
        from llm_evaluator.providers.cached_provider import CachedProvider

        mock_provider = Mock()
        mock_provider.model = "test"
        mock_provider.config = GenerationConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            cached = CachedProvider(mock_provider, cache_dir=tmpdir)

            assert cached.stats["hits"] == 0
            assert cached.stats["misses"] == 0


class TestGenerationConfigValidation:
    """Test GenerationConfig with various values"""

    def test_config_temperature_range(self):
        """Test various temperature values"""
        from llm_evaluator.providers.base import GenerationConfig

        config = GenerationConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = GenerationConfig(temperature=1.0)
        assert config.temperature == 1.0

        config = GenerationConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_config_max_tokens_values(self):
        """Test various max_tokens values"""
        from llm_evaluator.providers.base import GenerationConfig

        config = GenerationConfig(max_tokens=1)
        assert config.max_tokens == 1

        config = GenerationConfig(max_tokens=4096)
        assert config.max_tokens == 4096

    def test_config_top_p_values(self):
        """Test various top_p values"""
        from llm_evaluator.providers.base import GenerationConfig

        config = GenerationConfig(top_p=0.0)
        assert config.top_p == 0.0

        config = GenerationConfig(top_p=0.5)
        assert config.top_p == 0.5

        config = GenerationConfig(top_p=1.0)
        assert config.top_p == 1.0

    def test_config_top_k_values(self):
        """Test various top_k values"""
        from llm_evaluator.providers.base import GenerationConfig

        config = GenerationConfig(top_k=1)
        assert config.top_k == 1

        config = GenerationConfig(top_k=50)
        assert config.top_k == 50

        config = GenerationConfig(top_k=None)
        assert config.top_k is None


class TestGenerationResultMetadata:
    """Test GenerationResult metadata handling"""

    def test_result_with_empty_metadata(self):
        """Test result with empty metadata"""
        from llm_evaluator.providers.base import GenerationResult

        result = GenerationResult(
            text="Test", response_time=0.5, tokens_used=10, model="test", metadata={}
        )

        assert result.metadata == {}

    def test_result_with_rich_metadata(self):
        """Test result with rich metadata"""
        from llm_evaluator.providers.base import GenerationResult

        result = GenerationResult(
            text="Test",
            response_time=0.5,
            tokens_used=10,
            model="test",
            metadata={"stop_reason": "eos", "finish_reason": "stop", "model_version": "1.0"},
        )

        assert result.metadata["stop_reason"] == "eos"
        assert result.metadata["finish_reason"] == "stop"

    def test_result_cached_flag(self):
        """Test result cached flag"""
        from llm_evaluator.providers.base import GenerationResult

        result = GenerationResult(
            text="Test", response_time=0.5, tokens_used=10, model="test", metadata={}, cached=True
        )

        assert result.cached is True
