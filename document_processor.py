import os
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from pathlib import Path
import re

class DocumentProcessor:
    """Handles document parsing and chunking for RAG system."""
    
    @staticmethod
    def read_document(file_path: str) -> str:
        """Read document content based on file extension"""
        _, ext = os.path.splitext(file_path.lower())
        print(f"\n{'='*50}\n📄 Reading document: {file_path}")
        print(f"📝 File extension: {ext}")
        
        try:
            content = ""
            if ext == '.pdf':
                print("🔍 Processing PDF document...")
                content = DocumentProcessor._read_pdf(file_path)
            elif ext == '.txt':
                print("📝 Processing text document...")
                content = DocumentProcessor._read_txt(file_path)
            else:
                error_msg = f"Unsupported file type: {ext}"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
                
            print(f"✅ Successfully read document. Content length: {len(content)} characters")
            if len(content) > 0:
                print(f"📄 Sample content (first 200 chars):\n{content[:200]}...")
            else:
                print("⚠️  Warning: Document content is empty")
                
            return content
            
        except Exception as e:
            error_msg = f"Error reading document: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise Exception(error_msg)
    
    @staticmethod
    def _read_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
    
    @staticmethod
    def _read_txt(file_path: str) -> str:
        """Read text from a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def chunk_document(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        print(f"\n✂️  Chunking document (chunk_size={chunk_size}, overlap={overlap})...")
        
        if not text or len(text.strip()) == 0:
            print("⚠️  Warning: Empty text provided for chunking")
            return []
            
        words = text.split()
        total_words = len(words)
        print(f"📊 Total words: {total_words}")
        
        if total_words <= chunk_size:
            print(f"📦 Document fits in single chunk")
            return [text]
            
        chunks = []
        i = 0
        chunk_num = 1
        
        while i < total_words:
            end = min(i + chunk_size, total_words)
            chunk_text = ' '.join(words[i:end])
            chunks.append(chunk_text)
            
            print(f"📦 Chunk {chunk_num}: Words {i}-{end-1} (size: {end-i} words)")
            if chunk_num <= 3 or chunk_num % 10 == 0:  # Log first 3 chunks and then every 10th
                print(f"   Preview: {chunk_text[:100]}...")
                
            if end == total_words:
                break
                
            i += (chunk_size - overlap)
            chunk_num += 1
            
        print(f"✅ Created {len(chunks)} chunks with {overlap} words overlap")
        return chunks
    
    @staticmethod
    def chunk_text(
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata."""
        if not text.strip():
            return []
            
        # Split into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if current_length + len(para) > chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'length': len(chunk_text),
                    'num_paragraphs': len(current_chunk)
                })
                
                # Keep overlap if needed
                if chunk_overlap > 0 and len(current_chunk) > 1:
                    overlap = max(1, len(current_chunk) // 2)
                    current_chunk = current_chunk[-overlap:]
                    current_length = sum(len(p) for p in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(para)
            current_length += len(para)
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'length': len(chunk_text),
                'num_paragraphs': len(current_chunk)
            })
        
        return chunks
    
    @staticmethod
    def generate_summary(text: str, max_words: int = 150) -> str:
        """Generate a concise summary of the text, strictly under max_words.
        
        Args:
            text: The input text to summarize
            max_words: Maximum number of words in the summary (default: 150)
            
        Returns:
            A summary that's guaranteed to be under max_words
        """
        from summarizer import generate_summary as generate_ai_summary
        
        try:
            # First try with the AI summarizer
            summary = generate_ai_summary(text, max_words)
            
            # Fallback to extractive summarization if AI summary is too long
            words = summary.split()
            if len(words) > max_words:
                summary = ' '.join(words[:max_words])
                
                # Ensure we end with a complete sentence
                last_punct = max(
                    summary.rfind('.'),
                    summary.rfind('!'),
                    summary.rfind('?')
                )
                if last_punct > 0:
                    summary = summary[:last_punct + 1]
                    
            return summary
            
        except Exception as e:
            # Fallback to simple extractive summarization
            sentences = re.split(r'(?<=[.!?])\s+', text)
            summary = ' '.join(sentences[:3])  # First few sentences as summary
            
            # Ensure word limit is strictly enforced
            words = summary.split()
            if len(words) > max_words:
                summary = ' '.join(words[:max_words])
                
                # Ensure we end with a complete sentence
                last_punct = max(
                    summary.rfind('.'),
                    summary.rfind('!'),
                    summary.rfind('?')
                )
                if last_punct > 0:
                    summary = summary[:last_punct + 1]
                    
            return summary
