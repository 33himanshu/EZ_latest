from typing import List, Dict, Any, Optional
import requests
import json
import logging
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 500
    repetition_penalty: float = 1.1


class Chatbot:
    """Handles chat interactions with the LLM."""
    
    def __init__(
        self, 
        model_name: str = "llama3:instruct",
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the chatbot.
        
        Args:
            model_name: Name of the model to use
            base_url: Base URL of the Ollama API server
            system_prompt: Optional system prompt to set the assistant's behavior
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.conversation_history: List[Dict[str, str]] = []
        
        if self.system_prompt:
            self.conversation_history.append({
                'role': 'system',
                'content': self.system_prompt
            })
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are a helpful AI assistant that provides accurate and concise answers based on the provided context. 
        If the context is provided, use it to answer the question. If you don't know the answer, say that you don't know, 
        don't try to make up an answer. Be helpful, harmless, and honest in your responses."""
    
    def format_prompt(
        self,
        question: str,
        context: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format the prompt with context and chat history.
        
        Args:
            question: The user's question
            context: List of context strings (retrieved chunks)
            chat_history: List of previous messages in the conversation
            
        Returns:
            Formatted prompt string
        """
        # Start with system prompt if available
        prompt_parts = []
        
        # Add chat history if available
        if chat_history:
            for msg in chat_history:
                role = "User" if msg['role'] == 'user' else "Assistant"
                prompt_parts.append(f"{role}: {msg['content']}")
        
        # Add context if available
        if context:
            context_str = "\n\n".join([f"[Context {i+1}] {c}" for i, c in enumerate(context)])
            prompt_parts.append(f"\nContext:\n{context_str}")
        
        # Add the current question
        prompt_parts.append(f"\nUser: {question}")
        
        # Add instruction for the assistant
        prompt_parts.append("\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate_response(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            
        Returns:
            Generated response text
        """
        if config is None:
            config = GenerationConfig()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "num_predict": config.max_tokens,
                        "repeat_penalty": config.repetition_penalty
                    }
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Handle both streaming and non-streaming responses
            if 'response' in result:
                return result['response'].strip()
            else:
                return json.dumps(result).strip()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def chat(
        self,
        message: str,
        context: Optional[List[str]] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate a response to a user message with optional context.
        
        Args:
            message: User's message
            context: List of context strings (retrieved chunks)
            config: Generation configuration
            
        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
        
        # Format the prompt
        prompt = self.format_prompt(
            question=message,
            context=context,
            chat_history=self.conversation_history[:-1]  # Exclude current message
        )
        
        # Generate response
        response = self.generate_response(prompt, config)
        
        # Add assistant's response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        if self.system_prompt:
            self.conversation_history.append({
                'role': 'system',
                'content': self.system_prompt
            })
