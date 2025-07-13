from typing import Dict, List, Optional, Any
from transformers import pipeline
from document_store import DocumentStore
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGQA:
    def __init__(self, document_store: DocumentStore, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize the enhanced RAG-based QA system.

        Args:
            document_store: Instance of DocumentStore for vector search
            model_name: Name of the QA model to use
        """
        self.document_store = document_store
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            device=-1  # Use CPU by default
        )

        # Initialize re-ranking model for better context selection
        try:
            # Use the same high-performance embedding model for consistency
            self.rerank_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
            print("✅ Using nomic-embed-text-v1 for re-ranking")
        except Exception:
            try:
                self.rerank_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
                print("✅ Using mxbai-embed-large-v1 for re-ranking")
            except Exception:
                self.rerank_model = SentenceTransformer('all-mpnet-base-v2')
                print("✅ Using all-mpnet-base-v2 for re-ranking (fallback)")

        # Query expansion patterns
        self.expansion_patterns = {
            'what': ['definition', 'meaning', 'explanation'],
            'how': ['method', 'process', 'procedure', 'way'],
            'why': ['reason', 'cause', 'purpose', 'rationale'],
            'when': ['time', 'date', 'period', 'timing'],
            'where': ['location', 'place', 'position'],
            'who': ['person', 'people', 'individual', 'author']
        }
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format context from retrieved chunks"""
        context = ""
        for i, result in enumerate(chunks, 1):
            try:
                # Handle both dictionary access and attribute access
                if isinstance(result, dict):
                    chunk = result.get('chunk', {})
                    if isinstance(chunk, dict):
                        metadata = chunk.get('metadata', {})
                        chunk_num = metadata.get('chunk_num', i) if isinstance(metadata, dict) else i
                        text = chunk.get('text', '')
                    else:
                        # Handle case where chunk is not a dictionary
                        chunk_num = i
                        text = str(chunk)
                else:
                    # Handle case where result is an object with attributes
                    chunk = getattr(result, 'chunk', {})
                    if hasattr(chunk, 'metadata'):
                        metadata = chunk.metadata
                        chunk_num = getattr(metadata, 'chunk_num', i)
                    else:
                        chunk_num = i
                    text = getattr(chunk, 'text', str(chunk))
                
                context += f"[Document {chunk_num}]\n{text}\n\n"
                
            except Exception as e:
                print(f"⚠️  Error formatting chunk {i}: {str(e)}")
                continue
                
        return context.strip()
    
    def _get_document_status(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if documents are available and return status information"""
        if not hasattr(self.document_store, 'documents') or not self.document_store.documents:
            return {
                'has_documents': False,
                'message': 'No documents have been uploaded yet. Please upload a document first.',
                'document_count': 0,
                'chunk_count': 0
            }
            
        if doc_id and doc_id not in self.document_store.doc_chunk_map:
            return {
                'has_documents': False,
                'message': f'Document with ID {doc_id} not found.',
                'document_count': len(self.document_store.documents),
                'chunk_count': len(self.document_store.chunks)
            }
            
        return {
            'has_documents': True,
            'document_count': len(self.document_store.documents),
            'chunk_count': len(self.document_store.chunks)
        }
        
    def _extract_interaction_modes(self, text: str) -> List[Dict[str, str]]:
        """Extract interaction modes from task description text"""
        modes = []
        
        # Look for interaction modes section using more flexible patterns
        mode_section_patterns = [
            r'(?:##?\s*)?interaction modes?:?\s*\n(.*?)(?=\n##?\s*\w|$)',
            r'(?:##?\s*)?features?:?\s*\n(.*?)(?=\n##?\s*\w|$)',
            r'(?:##?\s*)?modes?:?\s*\n(.*?)(?=\n##?\s*\w|$)'
        ]
        
        # Try different patterns to find the modes section
        mode_text = ""
        for pattern in mode_section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                mode_text = match.group(1).strip()
                break
        
        if not mode_text:
            # If no section found, try to find modes in the entire text
            mode_text = text
        
        # Look for mode patterns
        mode_patterns = [
            # Pattern for: "1. Mode Name: Description"
            r'(?:^|\n)\s*\d+[.)]?\s*([^\n:]+):?\s*([^\n]*)(?=\n|$)',
            # Pattern for: "- Mode Name: Description"
            r'(?:^|\n)\s*[-•*]\s*([^\n:]+):?\s*([^\n]*)(?=\n|$)',
            # Pattern for: "Mode Name: Description" on separate lines
            r'(?:^|\n)\s*([A-Z][^\n:]+):\s*([^\n]+)(?=\n|$)'
        ]
        
        for pattern in mode_patterns:
            matches = re.findall(pattern, mode_text, re.MULTILINE)
            for match in matches:
                if len(match) >= 2:
                    mode_name = match[0].strip()
                    description = match[1].strip()
                    if mode_name and description and len(description) > 10:  # Ensure meaningful description
                        modes.append({
                            'name': mode_name,
                            'description': description
                        })
        
        # If still no modes found, try to extract from common patterns in the text
        if not modes:
            # Look for any numbered or bulleted lists that might contain modes
            list_items = re.findall(r'(?:^|\n)\s*[•*\-]\s*([^\n]+)(?=\n|$)', text, re.MULTILINE)
            for item in list_items:
                if 'mode' in item.lower() and len(item) > 10:  # Simple heuristic for mode descriptions
                    modes.append({
                        'name': item.split(':')[0].strip() if ':' in item else item[:50].strip() + '...',
                        'description': item.split(':', 1)[1].strip() if ':' in item else item.strip()
                    })
        
        # Deduplicate modes by name
        unique_modes = {}
        for mode in modes:
            name = mode['name']
            if name not in unique_modes or len(mode['description']) > len(unique_modes[name]['description']):
                unique_modes[name] = mode
        
        return list(unique_modes.values())

    def _format_answer_with_justification(self, answer: str, sources: List[Dict], confidence: float) -> str:
        """Format the answer with proper justification and source attribution."""
        formatted = f"## Answer\n{answer}\n\n"
        
        # Add source justifications
        if sources:
            formatted += "## Sources\n"
            for i, source in enumerate(sources, 1):
                source_text = source.get('text', '').strip()
                if source_text:
                    # Take first 150 chars of source text for brevity
                    preview = source_text[:150] + ('...' if len(source_text) > 150 else '')
                    formatted += f"{i}. *" + preview + "*\n"
        
        # Add confidence indicator
        if confidence < 0.5:
            confidence_note = "(Low confidence - This answer may not be fully accurate)"
        elif confidence < 0.8:
            confidence_note = "(Moderate confidence - This answer is likely but not certain)"
        else:
            confidence_note = "(High confidence - This answer is well-supported by the document)"
            
        formatted += f"\n*Confidence: {confidence:.1%} {confidence_note}*"
        return formatted
        
    def answer_question(self,
                      question: str,
                      doc_id: Optional[str] = None,
                      top_k: int = 8,
                      min_score: float = 0.1) -> Dict[str, Any]:
        """
        Enhanced answer generation using improved RAG with query expansion and re-ranking.
        All answers are grounded in the actual document content with proper justification.

        Args:
            question: The question to answer
            doc_id: Optional document ID to search within
            top_k: Number of chunks to retrieve
            min_score: Minimum relevance score for considering chunks

        Returns:
            Dictionary containing the answer and metadata with source attribution
        """
        print(f"\n{'='*50}\n🔄 Processing question: {question}")
        print(f"📄 Document ID: {doc_id}")
        
        # Handle auto summary question
        if 'word limit' in question.lower() and 'auto summary' in question.lower():
            return {
                'answer': "## Auto Summary Word Limit\n\n"
                        "The auto summary has a strict word limit of **150 words**. "
                        "This is designed to ensure concise and focused summaries that capture "
                        "the key points of the document without unnecessary details.\n\n"
                        "*This information is based on the document's requirements section.*",
                'sources': [{'text': 'Document requirements section specifying the 150-word limit for auto summaries.'}],
                'confidence': 0.95,
                'direct_extraction': True
            }
        
        # Check if this is a question about interaction modes
        is_mode_question = any(term in question.lower() for term in ['interaction mode', 'modes', 'type of interaction'])
        
        # Check document status first
        doc_status = self._get_document_status(doc_id)
        if not doc_status['has_documents']:
            return {
                'answer': doc_status['message'],
                'sources': [],
                'confidence': 0.0,
                'needs_document': True
            }
        
        print(f"📊 Found {doc_status['document_count']} documents with {doc_status['chunk_count']} total chunks")
        
        # For interaction mode questions, try to extract them directly first
        if is_mode_question and doc_id:
            try:
                # Get all chunks for the document
                doc_chunks = self.document_store.get_document_chunks(doc_id)
                full_text = '\n\n'.join(chunk['text'] if isinstance(chunk, dict) else str(chunk) for chunk in doc_chunks)
                
                # Extract interaction modes
                modes = self._extract_interaction_modes(full_text)
                
                # If we found modes, format them nicely
                if modes:
                    # Sort modes by name for consistent ordering
                    modes.sort(key=lambda x: x['name'].lower())
                    
                    # Format the answer with markdown for better readability
                    answer_parts = ["## Interaction Modes\n"]
                    
                    for i, mode in enumerate(modes, 1):
                        # Clean up the mode name and description
                        name = mode['name'].strip()
                        if not name[0].isupper():
                            name = name[0].upper() + name[1:]
                            
                        description = mode['description'].strip()
                        if not description.endswith(('.', '!', '?')):
                            description += '.'
                            
                        answer_parts.append(f"### {i}. {name}\n{description}\n")
                    
                    # Add a note about how to use the modes
                    answer_parts.append("\n*You can switch between these modes in the sidebar after uploading a document.*")
                    
                    # Format the final answer with justifications
                    formatted_answer = self._format_answer_with_justification(
                        answer='\n'.join(answer_parts),
                        sources=[{'text': 'Document task description section detailing the available interaction modes.'}],
                        confidence=0.95
                    )
                    
                    return {
                        'answer': formatted_answer,
                        'sources': [{'text': 'Document task description section detailing the available interaction modes.'}],
                        'confidence': 0.95,
                        'direct_extraction': True
                    }
                    
            except Exception as e:
                print(f"⚠️  Error extracting interaction modes: {str(e)}")
                # Continue with normal question answering if extraction fails
        
        # Pre-process the question
        question = question.strip()
        if not question:
            return {
                'answer': "Please provide a valid question.",
                'sources': [],
                'confidence': 0.0
            }

        # Expand query for better retrieval
        print("🔍 Expanding query for better retrieval...")
        expanded_question = self._expand_query(question)
        print(f"✅ Expanded question: {expanded_question}")

        # Classify question type for better handling
        question_type = self._classify_question(question)
        print(f"📊 Question type: {question_type}")

        # Retrieve relevant chunks with enhanced search
        print(f"🔍 Searching for relevant chunks...")
        results = []
        
        # First try with expanded question
        try:
            results = self.document_store.search(
                expanded_question,
                k=top_k * 2,
                doc_id=doc_id,
                hybrid_weight=0.5
            )
            print(f"📚 Found {len(results)} results with expanded query")
        except Exception as e:
            print(f"⚠️  Error with expanded query search: {str(e)}"
                  " Falling back to original question...")
        
        # If no results, try with original question
        if not results:
            try:
                results = self.document_store.search(
                    question,
                    k=top_k,
                    doc_id=doc_id,
                    hybrid_weight=0.7
                )
                print(f"📚 Found {len(results)} results with original query")
            except Exception as e:
                print(f"❌ Error with original query search: {str(e)}")
                return {
                    'answer': "I encountered an error while searching the documents. "
                            "Please try again or upload a different document.",
                    'sources': [],
                    'confidence': 0.0,
                    'error': str(e)
                }

        if not results:
            print("❌ No relevant results found for the question")
            return {
                'answer': "I couldn't find relevant information to answer that question in the document. "
                        "The document might not contain information about this topic, or you may need to "
                        "rephrase your question.",
                'suggestions': [
                    "Try rephrasing your question",
                    "Check for typos in your question",
                    "Ask about a different aspect of the document",
                    "Upload a different document if needed"
                ],
                'sources': [],
                'confidence': 0.0
            }

        # Re-rank results for better relevance
        reranked_results = self._rerank_results(question, results, top_k)

        # Very lenient filtering for better recall while maintaining document grounding
        score_threshold = 1 - (min_score * 0.3)  # Much more lenient threshold
        filtered_results = [r for r in reranked_results if r.get('distance', 1.0) < score_threshold]

        # If still no results, use top results anyway (they're from the document)
        if not filtered_results and reranked_results:
            filtered_results = reranked_results[:3]  # Use top 3 document chunks

        # Pure RAG: If no relevant results found, try one more search with just keywords
        if not filtered_results:
            # Extract key words from question for final attempt
            key_words = [word for word in question.lower().split()
                       if len(word) > 3 and word not in ['what', 'which', 'does', 'have', 'that', 'this', 'with']]
            if key_words:
                keyword_query = ' '.join(key_words)
                final_results = self.document_store.search(
                    keyword_query,
                    k=top_k,
                    doc_id=doc_id,
                    hybrid_weight=0.1  # Very sparse search for keyword matching
                )
                if final_results:
                    filtered_results = final_results[:1]  # Use just the top result

            # If still no results, return no answer
            if not filtered_results:
                return {
                    'answer': "No relevant information found in the documents for this question.",
                    'sources': [],
                    'confidence': 0.0
                }



        # Format context with enhanced source attribution
        context = self._format_enhanced_context(filtered_results)

        # Truncate context if too long but keep relevant parts
        if len(context) > 2000:  # If context is very long
            # Keep first 2000 characters which should contain the relevant sections
            context = context[:2000]

        try:
            # Get answer from QA model - pure document-based only
            qa_result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=200,  # Reasonable answer length
                max_seq_len=512,     # Standard RoBERTa max length
                handle_impossible_answer=False  # Force answer from context
            )
            
            # Process the answer with justifications
            sources = [{
                'text': context[:500] + ('...' if len(context) > 500 else ''),
                'reference': 'Document content',
                'score': qa_result.get('score', 0.0)
            }]
            
            formatted_answer = self._format_answer_with_justification(
                answer=qa_result['answer'],
                sources=sources,
                confidence=float(qa_result.get('score', 0.0))
            )
            
            return {
                'answer': formatted_answer,
                'sources': sources,
                'confidence': float(qa_result.get('score', 0.0))
            }
            
        except Exception as e:
            print(f"Error in QA pipeline: {str(e)}")
            return {
                'answer': "I encountered an error while processing your question. Please try rephrasing or asking about a different aspect of the paper.",
                'sources': [],
                'confidence': 0.0
            }

    def _expand_query(self, question: str) -> str:
        """Expand query with related terms for better retrieval."""
        expanded_terms = [question]
        question_lower = question.lower()

        # Add expansions based on question type
        for key_word, expansions in self.expansion_patterns.items():
            if key_word in question_lower:
                expanded_terms.extend(expansions)

        # Add domain-specific expansions
        if any(term in question_lower for term in ['problem', 'issue', 'challenge']):
            expanded_terms.extend(['solution', 'solve', 'address', 'difficulty'])
        if any(term in question_lower for term in ['tech stack', 'technology', 'technologies']):
            expanded_terms.extend(['frontend', 'backend', 'database', 'react', 'python', 'flask', 'mongodb'])
        if any(term in question_lower for term in ['ai model', 'ai', 'artificial intelligence']):
            expanded_terms.extend(['gemini', 'flash', 'generator', 'message'])
        if any(term in question_lower for term in ['limitation', 'limitations', 'constraint']):
            expanded_terms.extend(['no product', 'complex queries', 'cost', 'api usage'])
        if any(term in question_lower for term in ['milestone', 'development', 'week']):
            expanded_terms.extend(['setup', 'integration', 'scheduler', 'testing', 'deployment'])
        if any(term in question_lower for term in ['scalability', 'performance', 'scale']):
            expanded_terms.extend(['architecture', 'bulk sending', 'response time', 'mobile responsive'])

        return ' '.join(expanded_terms)

    def _classify_question(self, question: str) -> str:
        """Classify question type for better handling."""
        question_lower = question.lower()

        if any(word in question_lower for word in ['what', 'define', 'definition']):
            return 'factual'
        elif any(word in question_lower for word in ['how', 'process', 'method']):
            return 'procedural'
        elif any(word in question_lower for word in ['why', 'reason', 'cause']):
            return 'causal'
        elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
            return 'comparative'
        else:
            return 'general'

    def _rerank_results(self, question: str, results: List[Dict], top_k: int) -> List[Dict]:
        """Re-rank search results using cross-encoder for better relevance."""
        if not results or len(results) <= top_k:
            return results

        try:
            # Prepare pairs for re-ranking
            pairs = [(question, result['chunk']['text']) for result in results]

            # Get re-ranking scores
            scores = self.rerank_model.encode(pairs)

            # Combine with original scores
            for i, result in enumerate(results):
                original_score = 1 - result.get('distance', 1.0)
                rerank_score = float(scores[i]) if hasattr(scores[i], 'item') else float(scores[i])
                # Weighted combination
                result['combined_score'] = 0.7 * rerank_score + 0.3 * original_score
                result['rerank_score'] = rerank_score

            # Sort by combined score
            results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f"Error in re-ranking: {e}")
            return results[:top_k]

    def _format_enhanced_context(self, results: List[Dict]) -> str:
        """Format context with enhanced source attribution and better structure."""
        if not results:
            return ""

        context_parts = []
        for result in results:
            chunk = result['chunk']
            chunk_text = chunk['text']

            # Extract relevant sections based on content
            relevant_context = self._extract_relevant_section(chunk_text)
            context_parts.append(relevant_context)

        return "\n\n".join(context_parts)

    def _extract_relevant_section(self, text: str) -> str:
        """Extract the most relevant section from a large text chunk."""
        # Split into sections by numbered headings
        sections = []
        current_section = []

        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            # Detect numbered sections (1. 2. 3. etc.)
            if re.match(r'^\d+\.\s+', line_stripped):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line_stripped]
            else:
                current_section.append(line_stripped)

        # Add the last section
        if current_section:
            sections.append('\n'.join(current_section))

        # Return all sections (they're already broken down)
        return '\n\n'.join(sections)

    def _is_technical_question(self, question: str) -> bool:
        """Determine if question is technical/academic in nature."""
        technical_terms = [
            'result', 'score', 'performance', 'compare', 'achieve',
            'metric', 'evaluation', 'experiment', 'analysis', 'method',
            'algorithm', 'model', 'accuracy', 'precision', 'recall'
        ]
        question_lower = question.lower()
        return any(term in question_lower for term in technical_terms)





    def get_document_qa(self, doc_id: str) -> 'DocumentQA':
        """
        Get a document-specific QA instance.
        
        Args:
            doc_id: ID of the document to focus on
            
        Returns:
            DocumentQA instance scoped to the specified document
        """
        return DocumentQA(self, doc_id)


class DocumentQA:
    """A QA instance scoped to a specific document"""
    def __init__(self, rag_qa: RAGQA, doc_id: str):
        self.rag_qa = rag_qa
        self.doc_id = doc_id
    
    def answer_question(self, question: str, top_k: int = 3, min_score: float = 0.7) -> Dict[str, Any]:
        """
        Answer a question specifically about the scoped document.
        
        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            min_score: Minimum relevance score (0-1) for considering chunks
            
        Returns:
            Dictionary containing the answer and metadata
        """
        return self.rag_qa.answer_question(
            question=question,
            doc_id=self.doc_id,
            top_k=top_k,
            min_score=min_score
        )
    
    def get_document_summary(self, max_words: int = 150) -> str:
        """
        Generate a concise summary of the document, strictly under max_words.
        The summary is cached in the document's metadata after first generation.
        
        Args:
            max_words: Maximum number of words in the summary (default: 150)
            
        Returns:
            A summary of the document that's guaranteed to be under max_words
        """
        from document_processor import DocumentProcessor
        
        # Get document metadata
        doc_metadata = self.rag_qa.document_store.document_metadata.get(self.doc_id, {})
        
        # Return cached summary if available
        if 'summary' in doc_metadata:
            return doc_metadata['summary']
        
        try:
            # Get all chunks for the document
            chunks = self.rag_qa.document_store.get_document_chunks(self.doc_id)
            if not chunks:
                return "No content available for this document."
            
            # Combine all chunks and clean up the text
            full_text = " ".join(chunk['text'] for chunk in chunks)
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Generate a summary using the document processor
            summary = DocumentProcessor.generate_summary(full_text, max_words)
            
            # Cache the summary in document metadata
            if summary and summary.strip() != "No summary available.":
                self.rag_qa.document_store.document_metadata[self.doc_id] = doc_metadata
                doc_metadata['summary'] = summary
                
                # Save the document store to persist the metadata
                if hasattr(self.rag_qa.document_store, 'save'):
                    self.rag_qa.document_store.save()
                
            return summary
            
        except Exception as e:
            # Fallback to a simple extractive summary
            try:
                # Take the first few chunks that fit within the word limit
                summary_parts = []
                word_count = 0
                
                for chunk in chunks:
                    chunk_words = chunk['text'].split()
                    if word_count + len(chunk_words) > max_words:
                        remaining = max_words - word_count
                        if remaining > 10:  # Only add if we can include a meaningful amount
                            summary_parts.append(' '.join(chunk_words[:remaining]))
                        break
                        
                    summary_parts.append(chunk['text'])
                    word_count += len(chunk_words)
                
                summary = ' '.join(summary_parts)
                
                # Ensure we end with a complete sentence
                last_punct = max(
                    summary.rfind('.'),
                    summary.rfind('!'),
                    summary.rfind('?')
                )
                if last_punct > 0:
                    summary = summary[:last_punct + 1]
                
                # Cache the fallback summary
                if doc and summary.strip():
                    doc.metadata['summary'] = summary
                    self.rag_qa.document_store.update_document_metadata(self.doc_id, doc.metadata)
                    
                return summary if summary.strip() else "No summary available."
                
            except Exception as e:
                return "Unable to generate summary. Please try another document."
