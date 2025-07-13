#!/usr/bin/env python3
"""
Debug script to check document content and search results
"""

from document_store import DocumentStore
from rag_qa import RAGQA

def debug_document_content():
    """Debug what's actually in the document."""
    
    # Initialize components
    doc_store = DocumentStore(persist_dir="./vector_store")
    
    print("=== DOCUMENT CONTENT DEBUG ===")
    print(f"Documents: {len(doc_store.documents)}")
    print(f"Chunks: {len(doc_store.chunks)}")
    
    if doc_store.chunks:
        chunk = doc_store.chunks[0]
        text = chunk['text']
        
        print(f"\nChunk length: {len(text)} characters")
        print(f"First 500 chars:\n{text[:500]}...")
        
        # Look for specific content
        print("\n=== CONTENT ANALYSIS ===")
        
        # Check for AI model mentions
        if 'gemini' in text.lower():
            print("✅ Found 'Gemini' in document")
            # Find the context around Gemini
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'gemini' in line.lower():
                    print(f"Gemini context: {line.strip()}")
        else:
            print("❌ 'Gemini' not found in document")
        
        # Check for tech stack
        if 'tech stack' in text.lower():
            print("✅ Found 'Tech Stack' in document")
            lines = text.split('\n')
            in_tech_section = False
            for line in lines:
                if 'tech stack' in line.lower():
                    in_tech_section = True
                    print(f"Tech stack section start: {line.strip()}")
                elif in_tech_section and line.strip():
                    if line.strip()[0].isdigit() and '. ' in line:
                        break
                    print(f"Tech item: {line.strip()}")
        else:
            print("❌ 'Tech Stack' not found in document")
        
        # Check for limitations
        if 'limitation' in text.lower():
            print("✅ Found 'Limitations' in document")
            lines = text.split('\n')
            in_limit_section = False
            for line in lines:
                if 'limitation' in line.lower():
                    in_limit_section = True
                    print(f"Limitations section: {line.strip()}")
                elif in_limit_section and line.strip():
                    if line.strip()[0].isdigit() and '. ' in line:
                        break
                    print(f"Limitation: {line.strip()}")
        else:
            print("❌ 'Limitations' not found in document")

def test_specific_searches():
    """Test specific search queries."""
    
    doc_store = DocumentStore(persist_dir="./vector_store")
    rag_qa = RAGQA(doc_store)
    
    print("\n=== SEARCH TESTS ===")
    
    test_queries = [
        "AI model",
        "Gemini Flash",
        "tech stack",
        "React.js",
        "limitations",
        "frontend backend"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = doc_store.search(query, k=3)
        print(f"Results found: {len(results)}")
        
        if results:
            for i, result in enumerate(results):
                score = result.get('score', result.get('distance', 'N/A'))
                text_preview = result['chunk']['text'][:200] + "..."
                print(f"  [{i+1}] Score: {score} | Preview: {text_preview}")

if __name__ == "__main__":
    debug_document_content()
    test_specific_searches()
