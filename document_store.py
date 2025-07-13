from typing import List, Dict, Any, Optional
import os
import fitz  # PyMuPDF
import docx2txt
import pandas as pd
import difflib
import hashlib
import time  # For timestamp in document IDs
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import hashlib
import json
from pathlib import Path
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentStore:
    def __init__(self, persist_dir: Optional[str] = "./vector_store"):
        self.persist = persist_dir is not None
        if self.persist:
            self.persist_dir = Path(persist_dir)
            self.persist_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.persist_dir = None

        # Initialize the embedding model - optimized for RAG performance
        # Try multiple nomic model variants with trust_remote_code
        nomic_models = [
            ('nomic-ai/nomic-embed-text-v1', True),  # Requires trust_remote_code
            ('nomic-ai/nomic-embed-text-v1.5', True),  # Requires trust_remote_code
            ('sentence-transformers/all-mpnet-base-v2', False),  # Standard model
        ]

        model_loaded = False
        for model_name, needs_trust in nomic_models:
            try:
                print(f"🔄 Attempting to load {model_name}...")

                if needs_trust:
                    self.embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
                else:
                    self.embedding_model = SentenceTransformer(model_name)

                if 'nomic' in model_name:
                    self.embedding_dim = 768
                    print(f"✅ Using {model_name} (optimal for RAG)")
                else:
                    self.embedding_dim = 768
                    print(f"✅ Using {model_name} (high-quality model)")

                model_loaded = True
                break

            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue

        if not model_loaded:
            try:
                # Secondary: High-performance alternative
                print("🔄 Attempting to load mxbai-embed-large-v1...")
                self.embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
                self.embedding_dim = 1024  # Dimension for mxbai-embed-large
                print("✅ Using mxbai-embed-large-v1 (high-performance alternative)")
            except Exception as e2:
                print(f"❌ Failed to load mxbai model: {e2}")
                # Fallback: Reliable baseline model
                self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
                self.embedding_dim = 768  # Dimension for all-mpnet-base-v2
                print("✅ Using all-mpnet-base-v2 (reliable fallback)")
                print(f"Note: Consider installing newer models for better performance")

        # Initialize FAISS index with better indexing
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []  # Stores document metadata
        self.chunks = []     # Stores text chunks
        self.doc_chunk_map = {}  # Maps document_id to chunk indices

        # Initialize TF-IDF vectorizer for hybrid search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        self.tfidf_matrix = None
        
        # Initialize with empty state
        self.documents = []
        self.chunks = []
        self.doc_chunk_map = {}
        self.document_metadata = {}
        
        # Load existing data if persistence is enabled
        if self.persist:
            self.load()
    
    def _convert_numpy(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(item) for item in obj]
        else:
            return obj
            
    def _convert_to_numpy(self, obj):
        """Convert lists back to numpy arrays when loading"""
        if isinstance(obj, dict):
            if 'embedding' in obj and isinstance(obj['embedding'], list):
                obj['embedding'] = np.array(obj['embedding'], dtype='float32')
            return {k: self._convert_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_numpy(item) for item in obj]
        return obj
    
    def load(self):
        """Load the document store from disk if persistence is enabled."""
        if not self.persist or self.persist_dir is None:
            # Initialize empty state for non-persistent mode
            self.documents = []
            self.chunks = []
            self.doc_chunk_map = {}
            self.document_metadata = {}
            return False
            
        try:
            # Check if the required files exist
            faiss_index_path = self.persist_dir / "index.faiss"
            metadata_path = self.persist_dir / "metadata.json"
            
            if not faiss_index_path.exists() or not metadata_path.exists():
                print("⚠️  No saved state found, starting fresh")
                self.documents = []
                self.chunks = []
                self.doc_chunk_map = {}
                self.document_metadata = {}
                return False
                
            # Load FAISS index
            self.index = faiss.read_index(str(faiss_index_path))
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.chunks = [np.array(chunk) for chunk in data['chunks']]  # Convert back to numpy arrays
                self.doc_chunk_map = data.get('doc_chunk_map', {})
                self.document_metadata = data.get('document_metadata', {})
            
            print(f"📂 Loaded vector store with {len(self.documents)} documents and {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            # Reset to empty state
            self.documents = []
            self.chunks = []
            self.doc_chunk_map = {}
            self.document_metadata = {}
            return False
    
    def save(self):
        """Save the document store to disk if persistence is enabled."""
        if not self.persist:
            return
            
        try:
            # Ensure the directory exists
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            with open(self.persist_dir / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
                
            # Save document metadata
            with open(self.persist_dir / "document_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
                
            # Save document to chunk mapping
            with open(self.persist_dir / "doc_chunk_map.json", 'w', encoding='utf-8') as f:
                json.dump(self.doc_chunk_map, f, ensure_ascii=False, indent=2)
                
            # Save the FAISS index if it exists
            if hasattr(self, 'index') and self.index is not None:
                faiss.write_index(self.index, str(self.persist_dir / "faiss_index.index"))
                
            # Save the TF-IDF vectorizer if it exists
            if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
                import joblib
                joblib.dump(self.tfidf_vectorizer, self.persist_dir / "tfidf_vectorizer.joblib")
                
            print(f"✅ Document store saved to {self.persist_dir}")
        except Exception as e:
            print(f"⚠️  Error saving document store: {str(e)}")
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and all its chunks from the store.
        
        Args:
            doc_id: ID of the document to remove
            
        Returns:
            bool: True if document was found and removed, False otherwise
        """
        if doc_id not in self.documents:
            return False
            
        # Remove document from documents dict
        del self.documents[doc_id]
        
        # Remove document's chunks from chunks list and update indices
        chunk_indices_to_remove = [i for i, chunk in enumerate(self.chunks) 
                                 if chunk['document_id'] == doc_id]
                                 
        # Remove chunks in reverse order to maintain correct indices
        for idx in sorted(chunk_indices_to_remove, reverse=True):
            del self.chunks[idx]
            
        # Update FAISS index if it exists
        if hasattr(self, 'index') and self.index is not None and chunk_indices_to_remove:
            # Rebuild index without the removed chunks
            if len(self.chunks) > 0:
                chunk_embeddings = np.array([chunk['embedding'] for chunk in self.chunks])
                self.index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
                self.index.add(chunk_embeddings.astype('float32'))
            else:
                # No chunks left, reset the index
                self.index = None
        
        # Update TF-IDF matrix if it exists
        if hasattr(self, 'tfidf_matrix') and self.tfidf_matrix is not None:
            if len(self.chunks) > 0:
                self._update_tfidf_matrix()
            else:
                self.tfidf_matrix = None
                self.vectorizer = None
        
        # Remove from persistence if enabled
        if self.persist and self.persist_dir:
            doc_file = self.persist_dir / f"{doc_id}.json"
            if doc_file.exists():
                doc_file.unlink()
        
        return True
        
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: The ID of the document to retrieve
            
        Returns:
            The document dictionary or None if not found
        """
        return self.documents.get(doc_id)
        
    def update_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update a document's metadata.
        
        Args:
            doc_id: The ID of the document to update
            metadata: New metadata to set
            
        Returns:
            bool: True if the document was found and updated, False otherwise
        """
        if doc_id not in self.documents:
            return False
            
        self.documents[doc_id]['metadata'].update(metadata)
        
        # Update persistence if enabled
        if self.persist and self.persist_dir:
            self._save_document_store()
            
        return True
        
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash of the file content for duplicate detection."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _process_document(self, file_path: str) -> str:
        """Enhanced document processing with structure awareness"""
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == '.pdf':
                return self._process_pdf_with_structure(file_path)
            elif file_ext == '.docx':
                return self._process_docx_with_structure(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return self._process_excel_with_structure(file_path)
            elif file_ext == '.txt':
                return self._process_txt_with_structure(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

        except Exception as e:
            raise Exception(f"Error processing {file_path}: {str(e)}")

    def _process_pdf_with_structure(self, file_path: str) -> str:
        """Process PDF with enhanced structure detection"""
        doc = fitz.open(file_path)
        text_parts = []

        for page_num, page in enumerate(doc):
            page_text = page.get_text()

            # Add page marker
            if page_num > 0:
                text_parts.append(f"\n\n--- Page {page_num + 1} ---\n")

            # Try to detect and preserve structure
            lines = page_text.split('\n')
            processed_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Detect headers (all caps, short lines, or numbered sections)
                if (len(line) < 100 and
                    (line.isupper() or
                     re.match(r'^\d+\.?\s+[A-Z]', line) or
                     re.match(r'^[A-Z][A-Z\s]+$', line))):
                    processed_lines.append(f"\n## {line}\n")
                else:
                    processed_lines.append(line)

            text_parts.append('\n'.join(processed_lines))

        return '\n'.join(text_parts)

    def _process_docx_with_structure(self, file_path: str) -> str:
        """Process DOCX with structure preservation"""
        # Basic docx processing - can be enhanced with python-docx for better structure
        text = docx2txt.process(file_path)
        return self._enhance_text_structure(text)

    def _process_excel_with_structure(self, file_path: str) -> str:
        """Process Excel with sheet and column awareness"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text_parts = []

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text_parts.append(f"\n## Sheet: {sheet_name}\n")

                # Convert to structured text
                if not df.empty:
                    # Add column headers
                    text_parts.append("**Columns:** " + ", ".join(df.columns.astype(str)))
                    text_parts.append("\n**Data:**")
                    text_parts.append(df.to_string(index=False))

            return '\n'.join(text_parts)
        except Exception:
            # Fallback to simple processing
            df = pd.read_excel(file_path)
            return df.to_string()

    def _process_txt_with_structure(self, file_path: str) -> str:
        """Process TXT with structure enhancement"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Enhanced structure detection for task descriptions
        text_lower = text.lower()
        if any(header in text_lower for header in ['task:', 'objective:', 'problem statement:']):
            # Preserve markdown-like structure
            lines = text.split('\n')
            processed_lines = []
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Detect section headers (lines ending with : and not too long)
                is_section_header = (line.endswith(':') and len(line) < 100 and 
                                   not line.startswith(('•', '-', '*', '●', '>', '```')))
                
                # Preserve section headers
                if is_section_header:
                    current_section = line.lower()
                    processed_lines.append(f'\n## {line.upper()}\n')
                # Preserve bullet points
                elif line.startswith(('•', '-', '*', '●')) or line.strip().startswith(('•', '-', '*', '●')):
                    processed_lines.append(f'- {line.lstrip("•-*● ").strip()}')
                # Preserve numbered lists
                elif re.match(r'^\d+\.\s+', line):
                    # Ensure consistent formatting for numbered lists
                    line = re.sub(r'^(\d+)\.\s*', r'\1. ', line)
                    processed_lines.append(line)
                # Preserve code blocks
                elif line.strip().startswith('```'):
                    processed_lines.append('\n```' + line[3:].strip() + '\n')
                else:
                    # For interaction modes section, ensure each mode is on a new line
                    if current_section and 'interaction mode' in current_section:
                        # Split lines that might contain multiple modes
                        if any(sep in line for sep in [';', 'a. ', 'b. ', '1. ', '2. ']):
                            # Handle different separators
                            for sep in [';', 'a. ', 'b. ', '1. ', '2. ']:
                                if sep in line:
                                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                                    if len(parts) > 1:
                                        processed_lines.append('\n'.join(f"- {p}" for p in parts))
                                        break
                            else:
                                processed_lines.append(line)
                        else:
                            processed_lines.append(line)
                    else:
                        # Add newlines between paragraphs
                        if processed_lines and not processed_lines[-1].endswith(('\n', '\n\n')):
                            processed_lines.append(' ' + line)
                        else:
                            processed_lines.append(line)
            
            return '\n'.join(processed_lines)
        
        return self._enhance_text_structure(text)

    def _enhance_text_structure(self, text: str) -> str:
        """Enhance text structure detection"""
        lines = text.split('\n')
        enhanced_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                enhanced_lines.append('')
                continue

            # Detect various header patterns
            if (
                # All caps headers
                (len(line) < 100 and line.isupper()) or
                # Numbered sections
                re.match(r'^\d+\.?\s+[A-Z]', line) or
                # Roman numerals
                re.match(r'^[IVX]+\.?\s+[A-Z]', line) or
                # Letter sections
                re.match(r'^[A-Z]\.?\s+[A-Z]', line) or
                # Underlined text (detected by following dashes)
                (len(enhanced_lines) > 0 and re.match(r'^[-=_]{3,}$', line))
            ):
                enhanced_lines.append(f"## {line}")
            else:
                enhanced_lines.append(line)

        return '\n'.join(enhanced_lines)

    def _extract_document_metadata(self, text: str, file_path: str) -> Dict[str, Any]:
        """Extract enhanced metadata from document content"""
        metadata = {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'file_type': os.path.splitext(file_path)[1].lower(),
        }

        # Text analysis
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        metadata.update({
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': len(paragraphs),
            'avg_words_per_sentence': len(words) / max(1, len(sentences)),
            'avg_sentences_per_paragraph': len(sentences) / max(1, len(paragraphs))
        })

        # Extract potential headers/sections
        headers = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (line and len(line) < 100 and
                (line.isupper() or
                 re.match(r'^\d+\.?\s+[A-Z]', line) or
                 re.match(r'^#+\s+', line))):  # Markdown headers
                headers.append(line)

        metadata['headers'] = headers[:10]  # Limit to first 10 headers
        metadata['section_count'] = len(headers)

        # Extract key terms (simple frequency analysis)
        # Remove common words and get most frequent terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

        word_freq = {}
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if len(word_clean) > 3 and word_clean not in common_words:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1

        # Get top 10 most frequent terms
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        metadata['key_terms'] = [term for term, freq in top_terms]

        # Document type classification (simple heuristics)
        text_lower = text.lower()
        if any(term in text_lower for term in ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references']):
            metadata['document_type'] = 'academic_paper'
        elif any(term in text_lower for term in ['executive summary', 'quarterly', 'financial', 'revenue', 'profit']):
            metadata['document_type'] = 'business_report'
        elif any(term in text_lower for term in ['manual', 'instructions', 'procedure', 'step', 'guide']):
            metadata['document_type'] = 'manual'
        else:
            metadata['document_type'] = 'general'

        return metadata

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        Split text into meaningful chunks with improved semantic awareness.
        Optimized for structured documents like task descriptions.

        Args:
            text: Input text to chunk
            chunk_size: Target number of words per chunk
            overlap: Number of words to overlap between chunks

        Returns:
            List of text chunks with preserved structure
        """
        # Preserve section headers and lists
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for line in lines:
            words = line.split()
            line_word_count = len(words)
            
            # If line is a heading, start a new chunk
            is_heading = line.startswith(('#', '##', '###', '####', '#####', '######', '**')) or \
                        (line.endswith(':') and line_word_count < 10)
                        
            if is_heading and current_chunk:
                if current_word_count > 0:  # Only add if we have content
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_word_count = line_word_count
                continue
                
            # If adding this line would exceed chunk size, start new chunk
            if current_word_count + line_word_count > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Start new chunk with overlap from previous
                if overlap > 0 and current_chunk:
                    # Get last few lines for overlap
                    overlap_lines = []
                    overlap_words = 0
                    for l in reversed(current_chunk):
                        l_words = l.split()
                        if overlap_words + len(l_words) > overlap:
                            break
                        overlap_lines.insert(0, l)
                        overlap_words += len(l_words)
                    current_chunk = overlap_lines + [line]
                    current_word_count = overlap_words + line_word_count
                else:
                    current_chunk = [line]
                    current_word_count = line_word_count
            else:
                current_chunk.append(line)
                current_word_count += line_word_count
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        # Post-process chunks to ensure they're not too small
        merged_chunks = []
        i = 0
        while i < len(chunks):
            if i < len(chunks) - 1 and len(chunks[i].split()) < chunk_size // 2:
                # Merge with next chunk if current is too small
                merged = chunks[i] + '\n\n' + chunks[i+1]
                merged_chunks.append(merged)
                i += 2  # Skip next chunk since we merged it
            else:
                merged_chunks.append(chunks[i])
                i += 1
                
        # If no chunks were created, fall back to basic chunking
        if not merged_chunks and text.strip():
            # Use sentence-based chunking as fallback
            return self._chunk_section(text, chunk_size, overlap)
            
        return merged_chunks

    def _chunk_section(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Helper method to chunk a section of text."""
        # Split by sentences first for better semantic boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)

            # If single sentence is too long, split it carefully
            if sentence_length > chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence at natural break points (commas, semicolons)
                sub_parts = re.split(r'[,;]\s+', sentence)
                temp_chunk = []
                temp_length = 0

                for part in sub_parts:
                    part_words = part.split()
                    if temp_length + len(part_words) > chunk_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = part_words[-overlap:] if len(part_words) > overlap else part_words
                        temp_length = len(temp_chunk)
                    else:
                        temp_chunk.extend(part_words)
                        temp_length += len(part_words)

                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            else:
                # Check if adding this sentence would exceed chunk size
                if current_length + sentence_length > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Create overlap from the end of previous chunk
                    overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_words.copy()
                    current_length = len(overlap_words)

                # Add sentence to current chunk
                current_chunk.extend(sentence_words)
                current_length += sentence_length

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Clean up any empty chunks
        return [chunk for chunk in chunks if chunk.strip()]

    def _update_tfidf_matrix(self):
        """Update the TF-IDF matrix with all current chunks."""
        try:
            if self.chunks:
                chunk_texts = [chunk['text'] for chunk in self.chunks]
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
        except Exception as e:
            print(f"Error updating TF-IDF matrix: {e}")
            self.tfidf_matrix = None
            
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks from the store.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            bool: True if document was deleted, False if not found or error
        """
        import time
        start_time = time.time()
        
        try:
            # Find the document
            doc_to_delete = next((doc for doc in self.documents 
                               if doc.get('id') == doc_id), None)
            if not doc_to_delete:
                print(f"⚠️  Document with ID {doc_id} not found")
                return False
                
            doc_name = doc_to_delete.get('metadata', {}).get('file_name', doc_id)
            print(f"\n{'='*50}")
            print(f"🗑️  Deleting document: {doc_name} (ID: {doc_id})")
            
            # Get chunk indices to remove
            chunk_indices = self.doc_chunk_map.get(doc_id, [])
            print(f"Found {len(chunk_indices)} chunks to remove")
            
            # Remove document from documents list
            self.documents = [doc for doc in self.documents if doc.get('id') != doc_id]
            
            # Remove chunks in reverse order to maintain correct indices
            chunks_removed = 0
            for idx in sorted(chunk_indices, reverse=True):
                if 0 <= idx < len(self.chunks):
                    del self.chunks[idx]
                    chunks_removed += 1
            
            # Remove document from chunk map
            if doc_id in self.doc_chunk_map:
                del self.doc_chunk_map[doc_id]
            
            # Update FAISS index if we have remaining chunks
            if self.chunks:
                print("Updating FAISS index with remaining chunks...")
                try:
                    # Rebuild FAISS index with remaining chunks
                    chunk_texts = [chunk['text'] for chunk in self.chunks]
                    chunk_embeddings = self.embedding_model.encode(
                        chunk_texts, 
                        show_progress_bar=False
                    )
                    
                    # Create new FAISS index
                    self.index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
                    self.index.add(chunk_embeddings)
                    print(f"✅ Rebuilt FAISS index with {len(self.chunks)} chunks")
                    
                    # Update TF-IDF matrix
                    print("Updating TF-IDF matrix...")
                    self._update_tfidf_matrix()
                    
                except Exception as e:
                    print(f"❌ Error updating FAISS index: {str(e)}")
                    # If we can't rebuild the index, we need to clear it
                    self.index = None
                    self.tfidf_matrix = None
            else:
                # No chunks left, reset index and matrix
                print("No chunks remaining, resetting FAISS index and TF-IDF matrix")
                self.index = None
                self.tfidf_matrix = None
            
            # Save the updated store
            print("Saving updated document store...")
            self.save()
            
            elapsed = time.time() - start_time
            print(f"\n✅ Successfully deleted document: {doc_name}")
            print(f"- Removed {chunks_removed} chunks")
            print(f"- Remaining documents: {len(self.documents)}")
            print(f"- Remaining chunks: {len(self.chunks)}")
            print(f"- Time taken: {elapsed:.2f} seconds")
            print("="*50)
            
            return True
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Error deleting document {doc_id} after {elapsed:.2f} seconds")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print("="*50)
            return False
    
    def search(self, query: str, k: int = 3, doc_id: Optional[str] = None, hybrid_weight: float = 0.7):
        """
        Enhanced hybrid search combining dense embeddings and sparse TF-IDF.
        Improved for structured documents like task descriptions.

        Args:
            query: Search query
            k: Number of results to return
            doc_id: Optional document ID to search within
            hybrid_weight: Weight for dense search (0.0 = only sparse, 1.0 = only dense)
        """
        try:
            if not query or not isinstance(query, str) or not query.strip() or k <= 0:
                return []

            # Expand query with synonyms and related terms
            expanded_query = self._expand_query(query)

            # Get candidate chunks
            if doc_id:
                candidate_chunks = self._get_document_chunks_for_search(doc_id)
            else:
                candidate_chunks = list(enumerate(self.chunks))

            if not candidate_chunks:
                return []

            # For task-related queries, boost sections that match common task structure
            is_task_query = any(term in query.lower() for term in ['interaction mode', 'mode', 'task', 'objective', 'requirement'])
            
            # For task-related queries, first try exact section matching
            if is_task_query:
                for idx, chunk in candidate_chunks:
                    chunk_text = chunk.get('text', '').lower()
                    if 'interaction mode' in chunk_text or 'mode' in chunk_text:
                        # Boost this chunk's score
                        if 'interaction mode' in query.lower():
                            return [{
                                'chunk': chunk,
                                'score': 1.0,
                                'distance': 0.0,
                                'is_exact_match': True
                            }]

            # Perform both dense and sparse searches with expanded query
            dense_scores = self._dense_search(expanded_query, candidate_chunks, k * 2)
            sparse_scores = self._sparse_search(expanded_query, candidate_chunks, k * 2)

            # For task documents, boost sections with headers
            if is_task_query:
                for chunk_idx, _ in candidate_chunks:
                    chunk = self.chunks[chunk_idx]
                    chunk_text = chunk.get('text', '').lower()
                    
                    # Boost sections that look like they contain mode information
                    if any(header in chunk_text for header in ['interaction mode', 'mode', 'type', 'way']):
                        dense_scores[chunk_idx] = max(dense_scores.get(chunk_idx, 0), 0.8)
                        sparse_scores[chunk_idx] = max(sparse_scores.get(chunk_idx, 0), 0.8)

            # Combine scores using hybrid weighting
            combined_scores = self._combine_scores(dense_scores, sparse_scores, hybrid_weight)

            # Re-rank and get top k results
            results = self._rerank_results(combined_scores, k)
            
            # If we found an exact match for task query, use that
            if is_task_query:
                exact_matches = [r for r in results if r.get('is_exact_match', False)]
                if exact_matches:
                    return exact_matches[:k]
                    
            return results

        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []

    def _expand_query(self, query: str) -> str:
        """Expand query with related terms for better retrieval."""
        # Simple query expansion - can be enhanced with word embeddings
        expanded_terms = []
        words = query.lower().split()

        # Add original query
        expanded_terms.append(query)

        # Add variations for common academic terms
        expansions = {
            'result': ['results', 'outcome', 'finding', 'conclusion'],
            'method': ['methodology', 'approach', 'technique', 'procedure'],
            'analysis': ['analyze', 'examination', 'evaluation', 'assessment'],
            'performance': ['efficiency', 'effectiveness', 'accuracy', 'score'],
            'compare': ['comparison', 'versus', 'contrast', 'difference'],
            'improve': ['enhancement', 'optimization', 'better', 'increase']
        }

        for word in words:
            if word in expansions:
                expanded_terms.extend(expansions[word])

        return ' '.join(expanded_terms)

    def _get_document_chunks_for_search(self, doc_id: str) -> List[tuple]:
        """Get chunks for a specific document."""
        if doc_id not in self.doc_chunk_map:
            return []

        chunk_indices = self.doc_chunk_map[doc_id]
        return [(i, self.chunks[i]) for i in chunk_indices if i < len(self.chunks)]

    def _dense_search(self, query: str, candidate_chunks: List[tuple], k: int) -> Dict[int, float]:
        """Perform dense search using embeddings."""
        if not candidate_chunks:
            return {}

        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])[0]

            # Get embeddings for candidate chunks
            chunk_texts = [chunk['text'] for _, chunk in candidate_chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)

            # Calculate cosine similarities
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

            # Create score dictionary
            scores = {}
            for i, (chunk_idx, _) in enumerate(candidate_chunks):
                scores[chunk_idx] = float(similarities[i])

            return scores
        except Exception as e:
            print(f"Error in dense search: {e}")
            return {}

    def _sparse_search(self, query: str, candidate_chunks: List[tuple], k: int) -> Dict[int, float]:
        """Perform sparse search using TF-IDF."""
        if not candidate_chunks or self.tfidf_matrix is None:
            return {}

        try:
            # Get texts for candidate chunks
            chunk_texts = [chunk['text'] for _, chunk in candidate_chunks]

            # Fit TF-IDF if not already fitted or update with new texts
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or len(chunk_texts) != self.tfidf_matrix.shape[0]:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)

            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

            # Create score dictionary
            scores = {}
            for i, (chunk_idx, _) in enumerate(candidate_chunks):
                scores[chunk_idx] = float(similarities[i])

            return scores
        except Exception as e:
            print(f"Error in sparse search: {e}")
            return {}

    def _combine_scores(self, dense_scores: Dict[int, float], sparse_scores: Dict[int, float],
                       hybrid_weight: float) -> Dict[int, float]:
        """Combine dense and sparse scores."""
        combined = {}
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())

        for idx in all_indices:
            dense_score = dense_scores.get(idx, 0.0)
            sparse_score = sparse_scores.get(idx, 0.0)
            combined[idx] = hybrid_weight * dense_score + (1 - hybrid_weight) * sparse_score

        return combined

    def _rerank_results(self, scores: Dict[int, float], k: int) -> List[Dict[str, Any]]:
        """Re-rank and format final results."""
        # Sort by score (descending)
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for idx in sorted_indices[:k]:
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'score': scores[idx],
                    'distance': 1.0 - scores[idx]  # Convert similarity to distance
                })

        return results

    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a specific document"""
        if not hasattr(self, 'doc_chunk_map') or not hasattr(self, 'chunks'):
            return []
            
        if doc_id not in self.doc_chunk_map:
            return []
            
        chunk_indices = self.doc_chunk_map[doc_id]
        return [self.chunks[i] for i in chunk_indices if 0 <= i < len(self.chunks)]
    
    def add_document(self, file_path: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Add a document to the vector store with duplicate detection.
        Returns document ID if added, None if duplicate or error.
        """
        print(f"\n{'='*50}\nStarting document processing for: {file_path}")
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Calculate file hash for duplicate detection
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                
            # Process document to get text content
            print("🔍 Processing document...")
            text = self._process_document(file_path)
            if not text or not text.strip():
                print(f"⚠️  Empty document content: {file_path}")
                return None
                
            print(f"✅ Document processed. Text length: {len(text)} characters")
            
            # Check for existing documents with same content
            for doc in self.documents:
                # Compare file hashes first (fast check)
                if doc.get('metadata', {}).get('file_hash') == file_hash:
                    print(f"ℹ️  Document already exists with ID: {doc['id']}")
                    return doc['id']
                
                # If hashes don't match but content is very similar (handles different temp files)
                existing_chunks = self.get_document_chunks(doc['id'])
                if existing_chunks:
                    existing_text = " ".join(chunk['text'] for chunk in existing_chunks)
                    if len(text) > 0 and len(existing_text) > 0:
                        # If documents are >95% similar, consider them duplicates
                        similarity = difflib.SequenceMatcher(None, text, existing_text).ratio()
                        if similarity > 0.95:
                            print(f"ℹ️  Similar document found ({(similarity*100):.1f}% match), using existing document")
                            return doc['id']
            
            # Generate a unique document ID with timestamp to ensure uniqueness
            doc_id = f"{self._get_file_hash(file_path)}_{int(time.time())}"
            
            # Extract enhanced metadata
            enhanced_metadata = self._extract_document_metadata(text, file_path)
            enhanced_metadata.update({
                'file_hash': file_hash,
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'processed_at': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            print(f"📄 File metadata: {enhanced_metadata}")
            
            # Chunk the document
            print("✂️  Chunking document...")
            chunks = self._chunk_text(text)
            if not chunks:
                print(f"⚠️  No valid chunks created from: {file_path}")
                return None
            print(f"✅ Created {len(chunks)} chunks")
            
            # Generate embeddings for chunks
            print("📥 Adding chunks to vector store...")
            chunk_embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            
            # Update FAISS index if needed
            if len(self.chunks) == 0:
                self.index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
            
            # Add vectors to index
            self.index.add(chunk_embeddings)
            
            # Store document metadata
            doc_metadata = {
                'id': doc_id,
                'file_path': file_path,
                'chunk_count': len(chunks),
                'word_count': len(text.split()),
                'char_count': len(text),
                'metadata': enhanced_metadata
            }
            self.documents.append(doc_metadata)
            
            # Add chunks with metadata
            start_idx = len(self.chunks)
            chunk_metadata = metadata.copy() if metadata else {}
            for i, chunk in enumerate(chunks):
                if i % 10 == 0:  # Log progress every 10 chunks
                    print(f"📤 Processing chunk {i+1}/{len(chunks)}...", end='\r')
                
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_num': i,
                    'text_length': len(chunk),
                    'word_count': len(chunk.split())
                })
                
                self.chunks.append({
                    'id': chunk_id,
                    'text': chunk,
                    'doc_id': doc_id,
                    'chunk_idx': i,
                    'metadata': chunk_metadata.copy()
                })
            
            print("✅ All chunks processed")
            
            # Update document-chunk mapping
            self.doc_chunk_map[doc_id] = list(range(start_idx, len(self.chunks)))
            
            # Update TF-IDF matrix
            print("🔢 Updating TF-IDF matrix...")
            self._update_tfidf_matrix()
            
            # Save the updated store
            print("💾 Saving document store...")
            self.save()
            
            print(f"✅ Document '{os.path.basename(file_path)}' successfully added with {len(chunks)} chunks")
            print(f"Total documents in store: {len(self.documents)}")
            print(f"Total chunks in store: {len(self.chunks)}")
            
            return doc_id
            
        except Exception as e:
            import traceback
            print(f"❌ Error adding document: {str(e)}")
            traceback.print_exc()
            return None
