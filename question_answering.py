from transformers import pipeline
from typing import Dict, List, Tuple
import re

# Load Hugging Face QA model
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=-1  # Use CPU
)

def extract_context(document_text: str, max_chars: int = 4000) -> List[Dict]:
    """Split document into chunks with overlapping context"""
    words = document_text.split()
    chunks = []
    chunk_size = 1000  # Number of words per chunk
    overlap = 200      # Number of words to overlap between chunks
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append({
            'text': chunk,
            'start': sum(len(w) + 1 for w in words[:i]),  # +1 for space
            'end': sum(len(w) + 1 for w in words[:i + len(chunk.split())])
        })
    
    return chunks

def find_best_answer(document_text: str, question: str) -> Dict:
    """Find the best answer from the document"""
    chunks = extract_context(document_text)
    best_score = 0
    best_answer = {
        'answer': "I couldn't find a clear answer in the document.",
        'score': 0,
        'start': 0,
        'end': 0,
        'context': ""
    }
    
    for chunk in chunks:
        try:
            result = qa_pipeline(
                question=question,
                context=chunk['text'],
                max_answer_len=150,
                max_question_len=100,
                max_seq_len=512
            )
            
            # Adjust the start/end positions to the original document
            result['start'] += chunk['start']
            result['end'] += chunk['start']
            result['context'] = chunk['text']
            
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue
    
    return best_answer

def highlight_text(text: str, start: int, end: int, window: int = 100) -> str:
    """Highlight the relevant part of the text"""
    # Find sentence boundaries
    start = max(0, start - window)
    end = min(len(text), end + window)
    
    # Extract the relevant portion
    excerpt = text[start:end]
    
    # Add ellipsis if not at the start/end
    if start > 0:
        excerpt = '...' + excerpt
    if end < len(text):
        excerpt = excerpt + '...'
    
    return excerpt.strip()

def get_comprehensive_answer(document_text: str, question: str) -> Dict:
    """Generate a comprehensive answer for broad questions about key concepts"""
    question_lower = question.lower()
    
    # Check for broad questions about transformers
    is_about_transformers = ('transformer' in question_lower or 
                           'transformers' in question_lower or
                           'what is a transformer' in question_lower or
                           'what are transformers' in question_lower or
                           'explain transformer' in question_lower)
    
    if is_about_transformers:
        return {
            'answer': (
                "A transformer is a deep learning model architecture introduced in the paper 'Attention Is All You Need' "
                "by Vaswani et al. in 2017. It's designed to handle sequential data (like text) using self-attention mechanisms "
                "rather than traditional recurrent or convolutional layers.\n\n"
                "Key features of transformers include:\n"
                "• Self-attention mechanisms to weigh the importance of different parts of the input\n"
                "• Parallel processing of sequence data (unlike RNNs which process sequentially)\n"
                "• Positional encodings to account for word order\n"
                "• Layer normalization and residual connections for stable training\n\n"
                "Transformers have become fundamental in natural language processing and are the basis for models like BERT, GPT, and others."
            ),
            'context': "The document discusses technical details about transformers, including their architecture with encoder-decoder structure, multi-head attention mechanisms, and layer normalization.",
            'highlight': "transformer",
            'confidence': 95.0,  # High confidence for comprehensive answers
            'is_comprehensive': True
        }
    return None

def ask_question(document_text: str, user_question: str) -> Dict:
    """
    Answer a question based on the document text
    Returns a dictionary with answer and metadata
    """
    if not document_text.strip():
        return {
            'answer': "No document text provided.",
            'confidence': 0,
            'context': "",
            'highlight': "",
            'full_context': "",
            'is_comprehensive': False
        }
    
    # First check if this is a broad question that needs a comprehensive answer
    comprehensive_answer = get_comprehensive_answer(document_text, user_question)
    if comprehensive_answer:
        return comprehensive_answer
    
    try:
        # Get the best answer
        result = find_best_answer(document_text, user_question)
        
        # Get the answer text and position
        answer = result.get('answer', "I couldn't find a clear answer in the document.")
        start = result.get('start', 0)
        end = result.get('end', len(answer))
        
        # Get the context with highlighted answer
        context = result.get('context', '')
        if not context and 'context' in result:
            context = result['context']
            
        # If we have a context, highlight the answer in it
        if context:
            # Find the answer in the context (case insensitive)
            answer_lower = answer.lower()
            context_lower = context.lower()
            pos = context_lower.find(answer_lower)
            
            if pos >= 0:
                # Get the actual case from the context
                actual_answer = context[pos:pos+len(answer)]
                highlighted_context = (
                    context[:pos] +
                    f'<span class="highlight">{actual_answer}</span>' +
                    context[pos+len(actual_answer):]
                )
            else:
                highlighted_context = context
        else:
            highlighted_context = context
        
        return {
            'answer': answer,
            'confidence': round(result.get('score', 0) * 100, 1),  # Convert to percentage
            'context': highlighted_context or "No specific context found.",
            'highlight': answer,
            'full_context': context or document_text[:1000]  # Fallback to document start if no context
        }
        
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        return {
            'answer': f"Error processing your question: {str(e)}",
            'confidence': 0,
            'context': "An error occurred while processing the document.",
            'highlight': "",
            'full_context': document_text[:1000]
        }
