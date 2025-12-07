"""
LLM service factory for creating configured LLM clients.
"""
import asyncio
import httpx
import logging
from typing import Dict, Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModelSettings
from pydantic_ai.models.fallback import FallbackModel
from openai import OpenAI, AsyncOpenAI

from codewiki.src.config import Config

# Import will be done lazily to avoid circular imports
def get_performance_tracker():
    from codewiki.src.be.performance_metrics import performance_tracker
    return performance_tracker

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def create_main_model(config: Config) -> OpenAIModel:
    """Create the main LLM model from configuration."""
    return OpenAIModel(
        model_name=config.main_model,
        provider=OpenAIProvider(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key
        ),
        settings=OpenAIModelSettings(
            temperature=0.0,
            max_tokens=32768
        )
    )


def create_fallback_model(config: Config) -> OpenAIModel:
    """Create the fallback LLM model from configuration."""
    return OpenAIModel(
        model_name=config.fallback_model,
        provider=OpenAIProvider(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key
        ),
        settings=OpenAIModelSettings(
            temperature=0.0,
            max_tokens=32768
        )
    )


def create_fallback_models(config: Config) -> FallbackModel:
    """Create fallback models chain from configuration."""
    main = create_main_model(config)
    fallback = create_fallback_model(config)
    return FallbackModel(main, fallback)


def create_openai_client(config: Config) -> OpenAI:
    """Create OpenAI client from configuration."""
    return OpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key
    )


class ClientManager:
    """Manages async OpenAI clients with connection pooling."""
    
    def __init__(self):
        self._clients: Dict[str, AsyncOpenAI] = {}
        self._lock = asyncio.Lock()
    
    async def get_client(self, config: Config) -> AsyncOpenAI:
        """Get or create async OpenAI client with connection pooling."""
        client_key = f"{config.llm_base_url}_{config.llm_api_key[:8]}"
        
        async with self._lock:
            if client_key not in self._clients:
                self._clients[client_key] = AsyncOpenAI(
                    base_url=config.llm_base_url,
                    api_key=config.llm_api_key,
                    http_client=httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_keepalive_connections=20,
                            max_connections=100
                        )
                    )
                )
            return self._clients[client_key]

# Global client manager
client_manager = ClientManager()


def call_llm(
    prompt: str,
    config: Config,
    model: str = None,
    temperature: float = 0.0
) -> str:
    """
    Call LLM with the given prompt (synchronous version for backward compatibility).
    
    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting
        
    Returns:
        LLM response text
    """
    if model is None:
        model = config.main_model
    
    client = create_openai_client(config)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=32768
    )
    return response.choices[0].message.content


async def call_llm_async_with_retry(
    prompt: str,
    config: Config,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> str:
    """
    Call LLM asynchronously with connection pooling and retry logic.
    
    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting
        max_tokens: Maximum tokens (defaults to config value)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (seconds)
        
    Returns:
        LLM response text
        
    Raises:
        Exception: If all retries are exhausted
    """
    if model is None:
        model = config.main_model
    
    if max_tokens is None:
        max_tokens = getattr(config, 'max_tokens_per_module', 32768)
    
    client = await client_manager.get_client(config)
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            start_time = asyncio.get_event_loop().time()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            api_time = asyncio.get_event_loop().time() - start_time
            
            # Record successful API call
            get_performance_tracker().record_api_call(api_time)
            
            return response.choices[0].message.content
            
        except Exception as e:
            last_exception = e
            
            # Check if this is a retryable error
            if not _is_retryable_error(e):
                logger.error(f"Non-retryable error in LLM call: {e}")
                raise
            
            if attempt < max_retries:
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}), "
                             f"retrying in {delay:.2f}s: {e}")
                
                await asyncio.sleep(delay)
            else:
                logger.error(f"LLM call failed after {max_retries + 1} attempts: {e}")
    
    # All retries exhausted
    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError("All retries exhausted but no exception was captured")


def _is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception should be retried
    """
    error_str = str(exception).lower()
    
    # Network-related errors
    retryable_patterns = [
        'connection',
        'timeout',
        'network',
        'rate limit',
        'too many requests',
        'temporary',
        'service unavailable',
        'internal server error',
        '503',
        '502',
        '504',
        '429'
    ]
    
    return any(pattern in error_str for pattern in retryable_patterns)


async def call_llm_async(
    prompt: str,
    config: Config,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> str:
    """
    Call LLM asynchronously with connection pooling.
    
    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting
        max_tokens: Maximum tokens (defaults to config value)
        
    Returns:
        LLM response text
    """
    return await call_llm_async_with_retry(
        prompt, config, model, temperature, max_tokens
    )