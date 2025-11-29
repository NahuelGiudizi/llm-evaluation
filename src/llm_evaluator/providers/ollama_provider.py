"""
Ollama provider implementation

Concrete implementation of LLMProvider for Ollama local models.
"""

import time
import logging
from typing import Dict, List, Optional, Union

import ollama

from . import (
    LLMProvider,
    GenerationConfig,
    GenerationResult,
    ProviderType,
    ProviderError,
    TimeoutError,
    ModelNotFoundError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama local LLM provider
    
    Implements LLMProvider interface for Ollama-hosted models.
    Supports local models like llama3.2, mistral, phi-3, etc.
    
    Example:
        >>> provider = OllamaProvider(model="llama3.2:1b")
        >>> result = provider.generate("Explain Python")
        >>> print(result.text)
    """
    
    def __init__(
        self, 
        model: str = "llama3.2:1b",
        config: Optional[GenerationConfig] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Ollama provider
        
        Args:
            model: Ollama model name (e.g., "llama3.2:1b")
            config: Generation configuration
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        super().__init__(model, config)
        self.base_url = base_url
        self._client = None
        
    def _get_client(self):
        """Lazy initialization of Ollama client"""
        if self._client is None:
            try:
                if self.base_url:
                    self._client = ollama.Client(host=self.base_url)
                else:
                    self._client = ollama
            except Exception as e:
                raise ProviderError(
                    message="Failed to initialize Ollama client",
                    original_error=e
                )
        return self._client
    
    def generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate text using Ollama
        
        Args:
            prompt: Input prompt
            config: Optional config override
            
        Returns:
            GenerationResult with response
            
        Raises:
            TimeoutError: If request exceeds timeout
            ModelNotFoundError: If model not found
            ProviderError: For other errors
        """
        cfg = config or self.config
        client = self._get_client()
        
        for attempt in range(cfg.retry_attempts):
            try:
                start_time = time.time()
                
                response = client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": cfg.temperature,
                        "num_predict": cfg.max_tokens,
                        "top_p": cfg.top_p,
                        "top_k": cfg.top_k,
                    }
                )
                
                response_time = time.time() - start_time
                
                # Extract text and metadata
                text = response.get("message", {}).get("content", "")
                eval_count = response.get("eval_count", 0)
                
                return GenerationResult(
                    text=text,
                    response_time=response_time,
                    token_count=eval_count,
                    model_name=self.model,
                    metadata={
                        "total_duration": response.get("total_duration", 0),
                        "load_duration": response.get("load_duration", 0),
                        "prompt_eval_count": response.get("prompt_eval_count", 0),
                        "eval_count": eval_count,
                        "eval_duration": response.get("eval_duration", 0),
                    }
                )
                
            except ollama.ResponseError as e:
                if "not found" in str(e).lower():
                    raise ModelNotFoundError(
                        message=f"Model '{self.model}' not found. Run: ollama pull {self.model}",
                        original_error=e
                    )
                if attempt < cfg.retry_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise ProviderError(
                    message=f"Ollama request failed: {e}",
                    original_error=e
                )
                
            except Exception as e:
                if attempt < cfg.retry_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(2 ** attempt)
                    continue
                raise ProviderError(
                    message=f"Unexpected error: {e}",
                    original_error=e
                )
        
        raise ProviderError(message=f"All {cfg.retry_attempts} attempts failed")
    
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple prompts
        
        Note: Ollama doesn't support true batch processing,
        so this sequentially processes each prompt
        
        Args:
            prompts: List of prompts
            config: Optional config override
            
        Returns:
            List of GenerationResult objects
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, config)
                results.append(result)
            except ProviderError as e:
                logger.error(f"Failed to generate for prompt: {e}")
                # Append error result
                results.append(GenerationResult(
                    text=f"ERROR: {e.message}",
                    response_time=0.0,
                    token_count=0,
                    model_name=self.model,
                    metadata={"error": str(e)}
                ))
        return results
    
    def is_available(self) -> bool:
        """
        Check if Ollama service and model are available
        
        Returns:
            True if service is running and model exists
        """
        try:
            client = self._get_client()
            # Try to list models to check if service is up
            models_response = client.list()
            
            # ollama.list() returns a ListResponse object with .models attribute
            models = models_response.models if hasattr(models_response, 'models') else []
            
            # Check if our model is in the list
            # Model objects have .model attribute containing the name
            model_names = [m.model if hasattr(m, 'model') else str(m) for m in models]
            
            # Match exact name or base name (without tag)
            model_base = self.model.split(":")[0]
            return any(
                self.model == name or 
                model_base == name.split(":")[0]
                for name in model_names
            )
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Get Ollama model information
        
        Returns:
            Dictionary with model metadata (strictly typed)
        """
        try:
            client = self._get_client()
            response = client.show(self.model)
            return {
                "model": self.model,
                "format": response.get("format", "unknown"),
                "family": response.get("details", {}).get("family", "unknown"),
                "parameter_size": response.get("details", {}).get("parameter_size", "unknown"),
                "quantization": response.get("details", {}).get("quantization_level", "unknown"),
            }
        except Exception as e:
            logger.warning(f"Could not fetch model info: {e}")
            return {
                "model": self.model,
                "error": str(e)
            }
    
    def _get_provider_type(self) -> ProviderType:
        """Return Ollama provider type"""
        return ProviderType.OLLAMA
