"""
Enhanced Challenge Mode with improved question generation and evaluation.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple, Optional
import re
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EnhancedQuestionGenerator:
    def __init__(self):
        """Initialize the enhanced question generator."""
        try:
            # Use a more capable model for question generation
            self.question_model_name = "valhalla/t5-small-qg-hl"
            self.tokenizer = AutoTokenizer.from_pretrained(self.question_model_name)
            self.question_model = AutoModelForSeq2SeqLM.from_pretrained(self.question_model_name)
            self.question_pipeline = pipeline(
                "text2text-generation",
                model=self.question_model,
                tokenizer=self.tokenizer,
                device=-1  # Use CPU
            )
        except Exception:
            # Fallback to a simpler model
            self.question_pipeline = pipeline(
                "text2text-generation",
                model="t5-small",
                device=-1
            )
        
        # Initialize embedding model for semantic evaluation
        try:
            self.embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
        except Exception:
            try:
                self.embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
            except Exception:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Enhanced question type templates with difficulty levels and scoring
        self.question_templates = {
            'factual': {
                'templates': [
                    ("What is {}?", "🔍"),
                    ("Who is {}?", "👤"), 
                    ("When did {} occur?", "📅"),
                    ("Where is {} located?", "📍"),
                    ("How many {} are mentioned?", "🔢")
                ],
                'difficulty': '🟢 Easy',
                'max_score': 5,
                'description': 'Tests recall of specific facts and details.'
            },
            'inferential': {
                'templates': [
                    ("Why might {} be important?", "💡"),
                    ("What can be inferred about {}?", "🧠"),
                    ("What does {} suggest about {}?", "🔍"),
                    ("How does {} relate to {}?", "🔄"),
                    ("What are the implications of {}?", "⚡")
                ],
                'difficulty': '🟡 Medium',
                'max_score': 8,
                'description': 'Tests ability to draw conclusions from the text.'
            },
            'analytical': {
                'templates': [
                    ("Compare and contrast {} and {}.", "⚖️"),
                    ("Analyze the relationship between {} and {}.", "🔗"),
                    ("Evaluate the effectiveness of {}.", "📊"),
                    ("What are the strengths and weaknesses of {}?", "⚖️"),
                    ("How does {} support the main argument?", "🏗️")
                ],
                'difficulty': '🟠 Challenging',
                'max_score': 10,
                'description': 'Tests critical thinking and analysis skills.'
            },
            'comprehension': {
                'templates': [
                    ("Summarize the main points about {}.", "📝"),
                    ("Explain the significance of {}.", "💡"),
                    ("Describe the process of {}.", "🔄"),
                    ("What is the purpose of {}?", "🎯"),
                    ("How does {} work?", "⚙️")
                ],
                'difficulty': '🟡 Medium',
                'max_score': 7,
                'description': 'Tests understanding of main ideas and concepts.'
            }
        }
    
    def generate_questions(self, document_text: str, num_questions: int = 3) -> List[Dict]:
        """
        Generate diverse, high-quality questions from document text.
        
        Args:
            document_text: The source document text
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries with metadata
        """
        questions = []
        
        # Extract key concepts and entities
        key_concepts = self._extract_key_concepts(document_text)
        
        # Split document into meaningful chunks
        chunks = self._split_into_chunks(document_text)
        
        # Generate different types of questions
        question_types = ['factual', 'inferential', 'analytical', 'comprehension']
        
        for i in range(num_questions):
            question_type = question_types[i % len(question_types)]
            
            # Select a relevant chunk
            chunk = random.choice(chunks) if chunks else document_text[:1000]
            
            # Generate question based on type
            question_data = self._generate_typed_question(
                chunk, question_type, key_concepts, document_text
            )
            
            if question_data:
                questions.append(question_data)
        
        # Ensure we have enough questions
        while len(questions) < num_questions:
            chunk = random.choice(chunks) if chunks else document_text[:1000]
            question_data = self._generate_fallback_question(chunk, document_text)
            if question_data:
                questions.append(question_data)
        
        return questions[:num_questions]
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts and entities from text."""
        # Simple keyword extraction - can be enhanced with NER
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out common words
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}
        concepts = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get most frequent concepts
        concept_counts = {}
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Return top concepts
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, count in sorted_concepts[:10]]
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into meaningful chunks."""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_typed_question(self, chunk: str, question_type: str, 
                                key_concepts: List[str], full_text: str) -> Optional[Dict]:
        """Generate a question of a specific type with enhanced metadata."""
        try:
            # Get question type details
            qtype_info = self.question_templates[question_type]
            template, emoji = random.choice(qtype_info['templates'])
            
            # Select concepts for the question
            if not key_concepts:
                return None
                
            # Try to use multiple concepts for analytical questions
            if question_type == 'analytical' and len(key_concepts) >= 2:
                concept1 = random.choice(key_concepts)
                concept2 = random.choice([c for c in key_concepts if c != concept1])
                try:
                    question = template.format(concept1, concept2)
                    concepts_used = [concept1, concept2]
                except (IndexError, KeyError):
                    question = template.format(concept1, concept1)
                    concepts_used = [concept1]
            else:
                concept = random.choice(key_concepts)
                try:
                    question = template.format(concept)
                    concepts_used = [concept]
                except (IndexError, KeyError):
                    return None
            
            # Clean up the question
            question = self._clean_question(question)
            
            if question and len(question) > 10:
                # Get context boundaries
                context_start = max(0, full_text.find(chunk) - 100)
                context_end = min(len(full_text), full_text.find(chunk) + len(chunk) + 100)
                
                # Get surrounding context
                surrounding_context = full_text[context_start:context_end]
                
                return {
                    'question': f"{emoji} {question}",
                    'type': question_type,
                    'type_emoji': emoji,
                    'type_description': qtype_info['description'],
                    'difficulty': qtype_info['difficulty'],
                    'max_score': qtype_info['max_score'],
                    'context': chunk,
                    'context_start': full_text.find(chunk),
                    'context_end': full_text.find(chunk) + len(chunk),
                    'surrounding_context': surrounding_context,
                    'key_concepts': concepts_used,
                    'concept_count': len(concepts_used),
                    'hints': self._generate_hints(question_type, concepts_used)
                }
        
        except Exception as e:
            print(f"Error generating {question_type} question: {e}")
        
        return None
    
    def _generate_fallback_question(self, chunk: str, full_text: str) -> Optional[Dict]:
        """Generate a fallback question using templates."""
        try:
            # Extract a key phrase from the chunk
            sentences = chunk.split('.')
            if sentences:
                sentence = sentences[0].strip()
                
                # Simple question generation
                if 'is' in sentence.lower():
                    question = f"What {sentence.lower()}?"
                elif 'are' in sentence.lower():
                    question = f"What {sentence.lower()}?"
                else:
                    question = f"What can you tell me about the content in this section?"
                
                return {
                    'question': question,
                    'type': 'comprehension',
                    'context': chunk,
                    'context_start': full_text.find(chunk),
                    'context_end': full_text.find(chunk) + len(chunk),
                    'difficulty': 'medium',
                    'key_concepts': []
                }
        except Exception as e:
            print(f"Error generating fallback question: {e}")
        
        return None
    
    def _clean_question(self, question: str) -> str:
        """Clean and format the generated question."""
        # Remove common prefixes
        prefixes = ['Question:', 'Q:', 'Generate a question:', 'Ask:']
        for prefix in prefixes:
            if question.startswith(prefix):
                question = question[len(prefix):].strip()
        
        # Ensure question ends with question mark
        if not question.endswith('?'):
            question += '?'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        return question
    
    def _generate_hints(self, question_type: str, concepts: List[str]) -> List[str]:
        """Generate hints based on question type and concepts."""
        hints = []
        
        # General hint about question type
        if question_type == 'factual':
            hints.append("Look for specific details and facts in the text.")
        elif question_type == 'inferential':
            hints.append("Consider what the text implies rather than just what it states directly.")
        elif question_type == 'analytical':
            hints.append("Examine the relationships between different parts of the text.")
        else:  # comprehension
            hints.append("Focus on understanding the main ideas and their significance.")
        
        # Add concept-specific hints
        if concepts:
            hints.append(f"Pay attention to information about: {', '.join(concepts[:2])}")
        
        # Add general test-taking strategy
        hints.append("Make sure your answer is clear, well-structured, and supported by the text.")
        
        return hints
        
    def _assess_difficulty(self, question: str, context: str) -> str:
        """Assess question difficulty based on question type and content."""
        # Simple heuristic for difficulty
        q_lower = question.lower()
        if any(word in q_lower for word in ['what', 'who', 'when', 'where']):
            return '🟢 Easy (1-2 points)'
        elif any(word in q_lower for word in ['why', 'how', 'explain', 'describe']):
            return '🟡 Medium (3-5 points)'
        else:
            return '🔴 Hard (6-10 points)'


class EnhancedAnswerEvaluator:
    def __init__(self):
        """Initialize the enhanced answer evaluator."""
        try:
            self.embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
        except Exception:
            try:
                self.embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
            except Exception:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_answer(self, question: str, user_answer: str, context: str, 
                       expected_concepts: List[str] = None) -> Dict:
        """
        Evaluate user answer using semantic similarity and concept matching.
        
        Args:
            question: The question asked
            user_answer: User's response
            context: Relevant document context
            expected_concepts: Key concepts that should be mentioned
            
        Returns:
            Evaluation results with score and feedback
        """
        if not user_answer.strip():
            return {
                'score': 0,
                'feedback': 'Please provide an answer.',
                'reference': context[:200] + "...",
                'is_correct': False
            }
        
        # Calculate semantic similarity
        semantic_score = self._calculate_semantic_similarity(user_answer, context)
        
        # Check concept coverage
        concept_score = self._check_concept_coverage(user_answer, expected_concepts or [])
        
        # Check answer completeness
        completeness_score = self._assess_completeness(user_answer, question)
        
        # Combine scores
        final_score = (semantic_score * 0.5 + concept_score * 0.3 + completeness_score * 0.2) * 100
        
        # Generate feedback
        feedback = self._generate_feedback(final_score, semantic_score, concept_score, completeness_score)
        
        return {
            'score': min(100, max(0, int(final_score))),
            'feedback': feedback,
            'reference': context[:300] + "...",
            'is_correct': final_score >= 60,
            'semantic_score': semantic_score,
            'concept_score': concept_score,
            'completeness_score': completeness_score
        }
    
    def _calculate_semantic_similarity(self, answer: str, context: str) -> float:
        """Calculate semantic similarity between answer and context."""
        try:
            answer_embedding = self.embedding_model.encode([answer])
            context_embedding = self.embedding_model.encode([context])
            
            similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
            return max(0, similarity)
        except Exception:
            return 0.0
    
    def _check_concept_coverage(self, answer: str, expected_concepts: List[str]) -> float:
        """Check how many expected concepts are covered in the answer."""
        if not expected_concepts:
            return 0.8  # Default score if no specific concepts expected
        
        answer_lower = answer.lower()
        covered_concepts = sum(1 for concept in expected_concepts 
                             if concept.lower() in answer_lower)
        
        return covered_concepts / len(expected_concepts) if expected_concepts else 0.8
    
    def _assess_completeness(self, answer: str, question: str) -> float:
        """Assess the completeness of the answer based on question type."""
        answer_length = len(answer.split())
        
        # Minimum expected lengths based on question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'who is']):
            min_length = 5
        elif any(word in question_lower for word in ['explain', 'describe', 'analyze']):
            min_length = 15
        elif any(word in question_lower for word in ['compare', 'evaluate']):
            min_length = 20
        else:
            min_length = 10
        
        return min(1.0, answer_length / min_length)
    
    def _generate_feedback(self, final_score: float, semantic_score: float, 
                          concept_score: float, completeness_score: float) -> str:
        """Generate detailed feedback based on scores."""
        feedback_parts = []
        
        if final_score >= 80:
            feedback_parts.append("Excellent answer!")
        elif final_score >= 60:
            feedback_parts.append("Good answer with room for improvement.")
        else:
            feedback_parts.append("Your answer needs significant improvement.")
        
        if semantic_score < 0.3:
            feedback_parts.append("Your answer doesn't seem closely related to the document content.")
        elif semantic_score < 0.5:
            feedback_parts.append("Try to align your answer more closely with the document content.")
        
        if concept_score < 0.5:
            feedback_parts.append("Consider including more key concepts from the text.")
        
        if completeness_score < 0.5:
            feedback_parts.append("Your answer could be more detailed and comprehensive.")
        
        return " ".join(feedback_parts)
