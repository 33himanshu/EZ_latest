#!/usr/bin/env python3
"""
Debug script to test search and QA functionality
"""

from document_store import DocumentStore
from rag_qa import RAGQA

def test_search_and_qa():
    """Test search and QA functionality"""
    
    # Initialize components
    print("🔍 Initializing components...")
    doc_store = DocumentStore(persist_dir="./vector_store")
    rag_qa = RAGQA(doc_store)
    
    # Check if documents are loaded
    print(f"📄 Documents loaded: {len(doc_store.documents)}")
    print(f"📝 Chunks available: {len(doc_store.chunks)}")
    
    if not doc_store.documents:
        print("❌ No documents found!")
        return
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "tech stack",
        "PromoMate tech stack", 
        "frontend backend",
        "React.js",
        "Python Flask",
        "MongoDB",
        "Gemini Flash"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = doc_store.search(query, k=3)
        print(f"   Results found: {len(results)}")
        
        if results:
            for i, result in enumerate(results):
                score = result.get('score', result.get('distance', 'N/A'))
                text_preview = result['chunk']['text'][:100] + "..."
                print(f"   [{i+1}] Score: {score:.3f} | Text: {text_preview}")
        else:
            print("   ❌ No results found")
    
    # Test QA functionality
    print("\n🤖 Testing QA functionality...")
    qa_queries = [
        "What is PromoMate's tech stack?",
        "What frontend technology does PromoMate use?",
        "What is the backend technology?",
        "What database does PromoMate use?",
        "What AI model does PromoMate use?"
    ]
    
    for query in qa_queries:
        print(f"\n❓ Question: '{query}'")
        try:
            result = rag_qa.answer_question(query)
            print(f"   Answer: {result.get('answer', 'No answer')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   Sources: {len(result.get('sources', []))}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_search_and_qa()
