import requests
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class QAService:
    """Handles question answering and challenge generation using Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3:instruct"):
        """Initialize the QA service.
        
        Args:
            base_url: Base URL of the Ollama API server
            model: Name of the language model to use
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.generate_url = f"{self.base_url}/api/generate"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate_text(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate text using the language model.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt to guide the model
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "max_tokens": kwargs.get("max_tokens", 1000)
                    }
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text
            if isinstance(result, dict) and 'message' in result:
                return result['message'].get('content', '').strip()
            elif isinstance(result, dict) and 'response' in result:
                return result['response'].strip()
            else:
                logger.warning(f"Unexpected response format: {result}")
                return str(result).strip()
                
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def answer_question(
        self, 
        question: str, 
        context: List[str],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Generate an answer to a question based on the provided context.
        
        Args:
            question: The question to answer
            context: List of context strings
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Format the prompt
        context_str = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(context)])
        
        prompt = f"""Answer the question based on the following context. If the answer cannot be found in the context, say "I don't know."

Context:
{context_str}

Question: {question}

Answer:"""
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Be concise and accurate in your responses. If the answer is not in the context, say "I don't know."
Always include the relevant part of the context that supports your answer."""
        
        # Generate the answer
        answer = self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            **generation_kwargs
        )
        
        return {
            'answer': answer,
            'context_used': context
        }
    
    def generate_challenge_questions(
        self,
        context: List[str],
        num_questions: int = 3,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """Generate challenge questions based on the provided context.
        
        Args:
            context: List of context strings
            num_questions: Number of questions to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of question-answer pairs with context
        """
        context_str = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(context)])
        
        prompt = f"""Generate {num_questions} challenging questions based on the following context. 
For each question, provide a detailed answer that can be found directly in the context.
Format the response as a JSON array of objects with 'question' and 'answer' keys.

Context:
{context_str}

Questions:"""
        
        system_prompt = """You are a helpful assistant that generates challenging questions based on the provided context. 
The questions should test deep understanding and require reasoning about the content.
Return a JSON array of objects with 'question' and 'answer' keys."""
        
        # Generate the questions
        response = self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            **generation_kwargs
        )
        
        # Parse the response
        try:
            # Try to extract JSON from the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                questions = json.loads(json_str)
                
                # Ensure we have the correct number of questions
                if isinstance(questions, list) and questions:
                    return questions[:num_questions]
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse questions as JSON: {e}")
        
        # Fallback: Return a default question if parsing fails
        return [
            {
                'question': 'What is the main topic of the document?',
                'answer': 'The document discusses ' + context[0][:100] + '...'
            }
        ]
    
    def evaluate_answer(
        self,
        question: str,
        user_answer: str,
        reference_answer: str,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Evaluate a user's answer against a reference answer.
        
        Args:
            question: The original question
            user_answer: The user's answer
            reference_answer: The reference/correct answer
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Evaluation result with score and feedback
        """
        prompt = f"""Evaluate the user's answer to the question based on the reference answer.

Question: {question}
Reference Answer: {reference_answer}
User's Answer: {user_answer}

Provide a score from 0.0 to 1.0 and brief feedback."""
        
        system_prompt = """You are an evaluator that assesses answers based on their accuracy and completeness compared to the reference answer.
Return a JSON object with 'score' (float 0.0-1.0) and 'feedback' (string) keys."""
        
        # Generate the evaluation
        response = self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            **generation_kwargs
        )
        
        # Parse the response
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                evaluation = json.loads(json_str)
                if 'score' in evaluation and 'feedback' in evaluation:
                    return evaluation
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.warning(f"Failed to parse evaluation as JSON: {e}")
        
        # Fallback: Return a default evaluation
        return {
            'score': 0.5,
            'feedback': 'The answer is partially correct but could be more detailed.'
        }
