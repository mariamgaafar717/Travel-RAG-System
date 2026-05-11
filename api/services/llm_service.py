"""LLM service for answer generation using various models."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based answer generation."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        use_gpu: bool = True,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """Initialize LLM service.
        
        Args:
            model_name: Name of the Hugging Face model to use
            use_gpu: Whether to use GPU if available
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._generator = None
        self._is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize the LLM model.
        
        Returns:
            True if successfully initialized, False otherwise
        """
        try:
            import torch
            from transformers import pipeline
            
            logger.info(f"Loading LLM model: {self.model_name}")
            
            gpu_available = self.use_gpu and torch.cuda.is_available()
            device = 0 if gpu_available else -1
            torch_dtype = torch.float16 if gpu_available else torch.float32

            if self.use_gpu and not gpu_available:
                logger.warning("GPU requested for LLM, but CUDA is not available. Falling back to CPU.")
            
            self._generator = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"torch_dtype": torch_dtype},
                device=device,
            )
            
            self._is_initialized = True
            logger.info("LLM model initialized successfully")
            return True
        
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.info("Install with: pip install torch transformers")
            return False
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return False
    
    def generate(self, prompt: str) -> str:
        """Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if not self._is_initialized or self._generator is None:
            logger.warning("LLM not initialized, returning empty response")
            return ""
        
        try:
            output = self._generator(
                prompt,
                max_new_tokens=self.max_tokens,
                max_length=None,
                return_full_text=False,
                do_sample=False,
                temperature=self.temperature,
            )
            
            return output[0]["generated_text"].strip()
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    def is_initialized(self) -> bool:
        """Check if LLM is initialized."""
        return self._is_initialized
