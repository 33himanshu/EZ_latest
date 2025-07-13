from transformers import pipeline
from typing import List, Dict, Tuple
import re
from question_answering import extract_context, highlight_text

# Load models
generator = pipeline("text-generation", model="gpt2", device=-1)  # Use CPU

# Initialize QA pipeline for finding relevant context
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=-1  # Use CPU
)

def find_relevant_context(document_text: str, question: str) -> Dict:
    """Find the most relevant context in the document for a given question"""
    chunks = extract_context(document_text)
    best_score = 0
    best_chunk = {
        'text': document_text[:500],  # Default to first 500 chars
        'start': 0,
        'end': min(500, len(document_text))
    }
    
    for chunk in chunks:
        try:
            # Use QA model to find relevance of chunk to question
            result = qa_pipeline(
                question=question,
                context=chunk['text'],
                max_answer_len=150,
                max_question_len=100,
                max_seq_len=512
            )
            
            if result['score'] > best_score:
                best_score = result['score']
                best_chunk = {
                    'text': chunk['text'],
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'score': result['score']
                }
        except Exception as e:
            print(f"Error finding context: {e}")
            continue
    
    return best_chunk

def generate_questions(document_text: str) -> List[Dict]:
    """Generate questions with relevant document context"""
    # First, extract key sections from the document
    chunks = extract_context(document_text)
    
    # Select up to 3 key chunks to base questions on
    key_chunks = chunks[:3]
    questions = []
    
    for chunk in key_chunks:
        try:
            # Generate a question specific to this chunk
            prompt = f"""Generate one specific, detailed question that can be answered from the following text.
            The question should test comprehension and require understanding of the content.
            
            Text: {chunk['text'][:1000]}
            
            Question:"""
            
            generated = generator(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                truncation=True
            )
            
            # Extract the question
            output = generated[0]['generated_text']
            question = output.split('Question:')[-1].split('?')[0].strip() + '?'
            
            # Find the most relevant context for this question
            context = find_relevant_context(document_text, question)
            
            questions.append({
                'question': question,
                'context': context['text'],
                'context_start': context['start'],
                'context_end': context['end']
            })
            
            if len(questions) >= 3:  # Limit to 3 questions
                break
                
        except Exception as e:
            print(f"Error generating question: {e}")
            continue
    
    # Fallback questions if generation fails
    if not questions:
        questions = [
            {
                'question': "What is the main topic of the document?",
                'context': document_text[:1000],
                'context_start': 0,
                'context_end': min(1000, len(document_text))
            },
            {
                'question': "What are the key points mentioned in the document?",
                'context': document_text[1000:2000] if len(document_text) > 1000 else document_text,
                'context_start': 1000 if len(document_text) > 1000 else 0,
                'context_end': min(2000, len(document_text))
            },
            {
                'question': "What conclusions or recommendations does the document present?",
                'context': document_text[-1000:] if len(document_text) > 1000 else document_text,
                'context_start': max(0, len(document_text) - 1000),
                'context_end': len(document_text)
            }
        ]
    
    return questions

def evaluate_answer(question_data: Dict, user_answer: str) -> Dict:
    """
    Evaluate the user's answer against the document context
    Returns a dictionary with evaluation results
    """
    # In a real implementation, this would use NLP to compare the answer with the context
    # For this demo, we'll provide a basic evaluation
    
    # Check if the answer is empty
    if not user_answer.strip():
        return {
            'is_correct': False,
            'feedback': 'Please provide an answer.',
            'reference': question_data['context']
        }
    
    # Simple keyword matching as a basic check
    answer_keywords = set(word.lower() for word in user_answer.split() if len(word) > 3)
    context_keywords = set(word.lower() for word in question_data['context'].split() if len(word) > 3)
    
    # Count matching keywords
    matching_keywords = answer_keywords.intersection(context_keywords)
    match_ratio = len(matching_keywords) / max(1, len(answer_keywords))
    
    if match_ratio > 0.5:
        return {
            'is_correct': True,
            'feedback': 'Your answer appears to be relevant to the document content.',
            'reference': highlight_text(
                question_data['context'],
                question_data.get('answer_start', 0),
                question_data.get('answer_end', 100)
            )
        }
    else:
        return {
            'is_correct': False,
            'feedback': 'Your answer may not fully address the question based on the document content.',
            'reference': highlight_text(
                question_data['context'],
                question_data.get('context_start', 0),
                question_data.get('context_end', 500)
            )
        }
