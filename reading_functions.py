from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import docx2txt
import pandas as pd
import re
from pathlib import Path

class DocumentReader:
    """Handles document parsing and chunking."""
    
    @staticmethod
    def read_document(file_path: str) -> Tuple[str, str]:
        """Read a document and return its content and metadata."""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return DocumentReader._read_pdf(file_path), 'pdf'
            elif file_extension in ['.docx', '.doc']:
                return DocumentReader._read_docx(file_path), 'docx'
            elif file_extension == '.txt':
                return DocumentReader._read_txt(file_path), 'txt'
            elif file_extension in ['.csv', '.xlsx']:
                return DocumentReader._read_table(file_path), file_extension[1:]
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise Exception(f"Error reading {file_path}: {str(e)}")
    
    @staticmethod
    def _read_pdf(file_path: Path) -> str:
        """Extract text from PDF file."""
        text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
    
    @staticmethod
    def _read_docx(file_path: Path) -> str:
        """Extract text from DOCX file."""
        return docx2txt.process(file_path)
    
    @staticmethod
    def _read_txt(file_path: Path) -> str:
        """Read text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def _read_table(file_path: Path) -> str:
        """Convert table data to text."""
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:  # .xlsx
            df = pd.read_excel(file_path)
        return df.to_string()


class TextChunker:
    """Handles text chunking for RAG systems."""
    
    @staticmethod
    def chunk_text(
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
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
                
            # If paragraph is very long, split it into sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if current_length + len(sentence) > chunk_size and current_chunk:
                        chunks.append({
                            'text': ' '.join(current_chunk).strip(),
                            'start_char': len(' '.join(chunks[-1]['text'])) if chunks else 0,
                            'end_char': len(' '.join(chunks[-1]['text'])) + len(' '.join(current_chunk).strip()) if chunks else len(' '.join(current_chunk).strip())
                        })
                        current_chunk = current_chunk[-(chunk_overlap // 50):]  # Keep some overlap
                        current_length = sum(len(s) for s in current_chunk)
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            else:
                if current_length + len(para) > chunk_size and current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk).strip(),
                        'start_char': chunks[-1]['end_char'] if chunks else 0,
                        'end_char': (chunks[-1]['end_char'] if chunks else 0) + len(' '.join(current_chunk).strip())
                    })
                    current_chunk = current_chunk[-(chunk_overlap // 50):]  # Keep some overlap
                    current_length = sum(len(s) for s in current_chunk)
                current_chunk.append(para)
                current_length += len(para)
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk).strip(),
                'start_char': chunks[-1]['end_char'] if chunks else 0,
                'end_char': (chunks[-1]['end_char'] if chunks else 0) + len(' '.join(current_chunk).strip())
            })
        
        return chunks
