#!/usr/bin/env python3
"""
Script to re-index existing documents with improved chunking strategy.
This will help fix the single-chunk issue and improve retrieval accuracy.
"""

import os
import shutil
from pathlib import Path
from document_store import DocumentStore
from rag_qa import RAGQA

def backup_existing_store(store_path: str) -> str:
    """Create a backup of the existing vector store."""
    backup_path = f"{store_path}_backup"
    if os.path.exists(store_path):
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(store_path, backup_path)
        print(f"✅ Backup created at: {backup_path}")
        return backup_path
    return ""

def reindex_documents():
    """Re-index all documents with improved chunking."""
    store_path = "./vector_store"
    
    # Create backup
    backup_path = backup_existing_store(store_path)
    
    try:
        # Load existing document store to get document info
        old_store = DocumentStore(persist_dir=store_path)
        
        if not old_store.documents:
            print("❌ No documents found to re-index.")
            return
        
        print(f"📄 Found {len(old_store.documents)} document(s) to re-index")
        
        # Get document information
        documents_info = []
        for doc in old_store.documents:
            documents_info.append({
                'id': doc['id'],
                'file_path': doc['file_path'],
                'metadata': doc.get('metadata', {})
            })
            print(f"  - {doc.get('metadata', {}).get('filename', 'Unknown')}")
        
        # Clear the vector store directory
        if os.path.exists(store_path):
            shutil.rmtree(store_path)
        
        # Create new document store with improved chunking
        print("\n🔄 Creating new document store with improved chunking...")
        new_store = DocumentStore(persist_dir=store_path)
        
        # Re-add documents
        success_count = 0
        for doc_info in documents_info:
            try:
                file_path = doc_info['file_path']
                metadata = doc_info['metadata']
                
                # Check if original file still exists
                if not os.path.exists(file_path):
                    print(f"⚠️  Original file not found: {file_path}")
                    # Try to find the file in common locations
                    filename = metadata.get('filename', '')
                    if filename:
                        # Check if it's in the current directory
                        local_path = os.path.join('.', filename)
                        if os.path.exists(local_path):
                            file_path = local_path
                            print(f"✅ Found file locally: {local_path}")
                        else:
                            print(f"❌ Could not locate file: {filename}")
                            continue
                    else:
                        continue
                
                print(f"📝 Re-indexing: {metadata.get('filename', file_path)}")
                
                # Add document with improved chunking
                doc_id = new_store.add_document(file_path, metadata)
                
                # Get chunk information
                chunks = new_store.get_document_chunks(doc_id)
                print(f"   ✅ Created {len(chunks)} chunks (was 1 chunk before)")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error re-indexing {doc_info.get('metadata', {}).get('filename', 'unknown')}: {e}")
        
        print(f"\n🎉 Re-indexing complete! Successfully processed {success_count}/{len(documents_info)} documents")
        
        # Test the new system
        print("\n🧪 Testing improved RAG system...")
        rag_qa = RAGQA(new_store)
        
        # Test with a simple question
        test_questions = [
            "What is PromoMate?",
            "What are the key features?",
            "What is the tech stack?",
            "What are the limitations?"
        ]
        
        for question in test_questions:
            try:
                result = rag_qa.answer_question(question, top_k=5, min_score=0.1)
                if result['answer'] and "No relevant information" not in result['answer']:
                    print(f"✅ '{question}' - Answer found (confidence: {result['confidence']:.2f})")
                else:
                    print(f"❌ '{question}' - No answer found")
            except Exception as e:
                print(f"❌ '{question}' - Error: {e}")
        
        print(f"\n📊 Final Statistics:")
        print(f"   - Total documents: {len(new_store.documents)}")
        print(f"   - Total chunks: {len(new_store.chunks)}")
        print(f"   - Average chunks per document: {len(new_store.chunks) / max(1, len(new_store.documents)):.1f}")
        
    except Exception as e:
        print(f"❌ Error during re-indexing: {e}")
        
        # Restore backup if something went wrong
        if backup_path and os.path.exists(backup_path):
            print("🔄 Restoring backup...")
            if os.path.exists(store_path):
                shutil.rmtree(store_path)
            shutil.copytree(backup_path, store_path)
            print("✅ Backup restored")

if __name__ == "__main__":
    print("🚀 Starting document re-indexing with improved chunking...")
    reindex_documents()
    print("✨ Done!")
