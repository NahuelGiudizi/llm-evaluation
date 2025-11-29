"""
Abstract LLM Provider interface for dependency injection

This module defines the contract that all LLM providers must implement,
enabling clean architecture and easy swapping of LLM backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
from enum import Enum


class ProviderType(Enum):
    """Supported LLM provider types"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 40
    timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class GenerationResult:
    """Result from LLM generation with strict typing (no Any)"""
    text: str
    response_time: float
    token_count: int
    model_name: str
    metadata: Dict[str, Union[str, int, float, bool]]  # Strict typing


@dataclass
class ProviderError(Exception):
    """Base exception for provider errors"""
    message: str
    original_error: Optional[Exception] = None
    retry_after: Optional[int] = None


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded"""
    pass


class TimeoutError(ProviderError):
    """Raised when request times out"""
    pass


class ModelNotFoundError(ProviderError):
    """Raised when model is not available"""
    pass


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers
    
    All concrete providers (Ollama, OpenAI, etc.) must implement this interface.
    This enables dependency injection and easy provider swapping.
    
    Example:
        >>> provider = OllamaProvider(model="llama3.2:1b")
        >>> result = provider.generate("What is Python?")
        >>> print(f"Response: {result.text}")
        >>> print(f"Time: {result.response_time:.2f}s")
    """
    
    def __init__(self, model: str, config: Optional[GenerationConfig] = None):
        """
        Initialize provider
        
        Args:
            model: Model identifier (provider-specific)
            config: Generation configuration
        """
        self.model = model
        self.config = config or GenerationConfig()
        self._validate_config()
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            config: Optional override of default config
            
        Returns:
            GenerationResult with response text, timing, and metadata
            
        Raises:
            RateLimitError: If rate limit exceeded
            TimeoutError: If request times out
            ModelNotFoundError: If model not available
            ProviderError: For other provider-specific errors
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts (batch processing)
        
        Args:
            prompts: List of input prompts
            config: Optional override of default config
            
        Returns:
            List of GenerationResult objects
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider/model is available
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Get information about the model
        
        Returns:
            Dictionary with model metadata (size, context, etc.) - strictly typed
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not 0 <= self.config.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.config.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
    
    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type"""
        return self._get_provider_type()
    
    @abstractmethod
    def _get_provider_type(self) -> ProviderType:
        """Return the provider type enum"""
        pass
