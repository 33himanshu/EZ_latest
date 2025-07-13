#!/usr/bin/env python3
"""
Test script to verify document QA accuracy with the PromoMate document.
"""

from document_store import DocumentStore
from rag_qa import RAGQA
import json

def test_document_qa():
    """Test the document QA system with various questions."""
    
    print("🧪 Testing Document QA System")
    print("=" * 50)
    
    # Initialize the system
    doc_store = DocumentStore(persist_dir="./vector_store")
    rag_qa = RAGQA(doc_store)
    
    # Check if we have documents
    if not doc_store.documents:
        print("❌ No documents found. Please upload a document first.")
        return
    
    print(f"📄 Found {len(doc_store.documents)} document(s)")
    print(f"📊 Total chunks: {len(doc_store.chunks)}")
    
    # Get the first document for testing
    doc_id = doc_store.documents[0]['id']
    doc_name = doc_store.documents[0].get('metadata', {}).get('filename', 'Unknown')
    print(f"🎯 Testing with document: {doc_name}")
    print()
    
    # Test questions based on the PromoMate document content
    test_questions = [
        # Basic factual questions
        "What is PromoMate?",
        "What problem does PromoMate solve?",
        "What are the key features of PromoMate?",
        
        # Technical questions
        "What is the tech stack used in PromoMate?",
        "What AI model is used for message generation?",
        "What database is used?",
        "What frontend framework is used?",
        
        # Specific feature questions
        "Does PromoMate support bulk messaging?",
        "What analytics features are available?",
        "Does it support multiple languages?",
        "What chatbot features are included?",
        
        # Requirements questions
        "What are the functional requirements?",
        "What are the non-functional requirements?",
        "What is the user flow?",
        
        # Development questions
        "What are the development milestones?",
        "How long is the development timeline?",
        "What happens in week 4 of development?",
        
        # Limitations
        "What are the limitations of PromoMate?",
        "Does it have a product database?",
        "What about WhatsApp API costs?",
    ]
    
    # Test each question
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"❓ Question {i}: {question}")
        
        try:
            # Test with document-specific QA
            result = rag_qa.answer_question(
                question=question,
                doc_id=doc_id,
                top_k=5,
                min_score=0.1  # Low threshold for better recall
            )
            
            answer = result.get('answer', 'No answer')
            confidence = result.get('confidence', 0.0)
            sources = result.get('sources', [])
            
            # Evaluate the result
            if answer and "No relevant information" not in answer and "Unable to process" not in answer:
                status = "✅ ANSWERED"
                print(f"   {status} (Confidence: {confidence:.2f})")
                print(f"   Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                if sources:
                    scores_list = [f"{s.get('score', 0):.2f}" for s in sources[:3]]
                print(f"   Sources: {len(sources)} chunk(s) with scores: {scores_list}")
            else:
                status = "❌ NO ANSWER"
                print(f"   {status}")
                print(f"   Response: {answer}")
            
            results.append({
                'question': question,
                'status': status,
                'confidence': confidence,
                'answer_length': len(answer) if answer else 0,
                'source_count': len(sources)
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                'question': question,
                'status': '❌ ERROR',
                'confidence': 0.0,
                'answer_length': 0,
                'source_count': 0
            })
        
        print()
    
    # Summary statistics
    print("📊 SUMMARY STATISTICS")
    print("=" * 50)
    
    answered = sum(1 for r in results if r['status'] == '✅ ANSWERED')
    no_answer = sum(1 for r in results if r['status'] == '❌ NO ANSWER')
    errors = sum(1 for r in results if r['status'] == '❌ ERROR')
    
    total = len(results)
    accuracy = (answered / total) * 100 if total > 0 else 0
    
    print(f"Total Questions: {total}")
    print(f"Answered: {answered} ({answered/total*100:.1f}%)")
    print(f"No Answer: {no_answer} ({no_answer/total*100:.1f}%)")
    print(f"Errors: {errors} ({errors/total*100:.1f}%)")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    
    # Average confidence for answered questions
    answered_results = [r for r in results if r['status'] == '✅ ANSWERED']
    if answered_results:
        avg_confidence = sum(r['confidence'] for r in answered_results) / len(answered_results)
        print(f"Average Confidence: {avg_confidence:.2f}")
    
    print()
    
    # Recommendations
    if accuracy < 70:
        print("🔧 RECOMMENDATIONS FOR IMPROVEMENT:")
        print("- Consider re-indexing documents with better chunking")
        print("- Lower the min_score threshold further")
        print("- Check if document content is properly processed")
        print("- Verify embedding model is working correctly")
    elif accuracy < 90:
        print("✅ Good performance! Consider fine-tuning:")
        print("- Adjust hybrid search weights")
        print("- Optimize chunk size and overlap")
        print("- Improve query expansion patterns")
    else:
        print("🎉 Excellent performance! The system is working well.")
    
    return results

if __name__ == "__main__":
    test_document_qa()
