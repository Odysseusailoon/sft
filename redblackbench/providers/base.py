"""Base LLM provider interface for RedBlackBench."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider.
    
    Attributes:
        model: Model identifier (e.g., 'gpt-4', 'claude-3-opus-20240229')
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response
        api_key: API key (if not using environment variable)
    """
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    Defines the interface that all LLM providers must implement.
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider configuration
        """
        self.config = config
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'openai', 'anthropic')."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            system_prompt: The system message for the conversation
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response text
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"

