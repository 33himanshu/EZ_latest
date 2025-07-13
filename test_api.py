#!/usr/bin/env python3
"""
Test script for the Research Assistant API.
This script verifies that all API endpoints are working correctly.
"""
import os
import sys
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configuration
BASE_URL = "http://localhost:8000"
TEST_PDF = "sample.pdf"  # Path to a sample PDF for testing

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")

def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")

def test_health() -> bool:
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok":
            print_success("Health check passed")
            return True
        else:
            print_error("Health check failed: Unexpected response")
            return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_upload() -> Optional[str]:
    """Test the document upload endpoint."""
    if not os.path.exists(TEST_PDF):
        print(f"Skipping upload test: {TEST_PDF} not found")
        return None
    
    try:
        with open(TEST_PDF, 'rb') as f:
            files = {'file': (TEST_PDF, f, 'application/pdf')}
            response = requests.post(
                f"{BASE_URL}/documents/upload",
                files=files,
                timeout=30
            )
        
        response.raise_for_status()
        data = response.json()
        
        if 'document_id' in data and 'summary' in data:
            print_success(f"Document uploaded successfully (ID: {data['document_id']})")
            print(f"   Summary: {data['summary'][:100]}...")
            return data['document_id']
        else:
            print_error("Upload test failed: Invalid response format")
            return None
    except Exception as e:
        print_error(f"Upload test failed: {e}")
        return None

def test_ask(document_id: str) -> bool:
    """Test the question answering endpoint."""
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={
                "document_id": document_id,
                "question": "What is this document about?"
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if 'answer' in data and 'sources' in data:
            print_success("Question answered successfully")
            print(f"   Answer: {data['answer'][:100]}...")
            print(f"   Sources: {len(data['sources'])} context items")
            return True
        else:
            print_error("Ask test failed: Invalid response format")
            return False
    except Exception as e:
        print_error(f"Ask test failed: {e}")
        return False

def test_challenge(document_id: str) -> bool:
    """Test the challenge generation endpoint."""
    try:
        response = requests.post(
            f"{BASE_URL}/challenge",
            json={
                "document_id": document_id,
                "num_questions": 2
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if 'questions' in data and isinstance(data['questions'], list):
            print_success(f"Generated {len(data['questions'])} challenge questions")
            for i, q in enumerate(data['questions'][:2], 1):
                print(f"   {i}. {q['question']}")
            return True
        else:
            print_error("Challenge test failed: Invalid response format")
            return False
    except Exception as e:
        print_error(f"Challenge test failed: {e}")
        return False

def test_evaluate(document_id: str) -> bool:
    """Test the answer evaluation endpoint."""
    try:
        question = "What is the main topic of this document?"
        user_answer = "This document is about artificial intelligence."
        reference_answer = "The document discusses machine learning and neural networks."
        
        response = requests.post(
            f"{BASE_URL}/evaluate",
            json={
                "document_id": document_id,
                "question": question,
                "user_answer": user_answer,
                "reference_answer": reference_answer
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if 'score' in data and 'feedback' in data:
            print_success("Evaluation test passed")
            print(f"   Score: {data['score']:.1%}")
            print(f"   Feedback: {data['feedback']}")
            return True
        else:
            print_error("Evaluation test failed: Invalid response format")
            return False
    except Exception as e:
        print_error(f"Evaluation test failed: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Starting Research Assistant API Tests\n")
    
    # Check if API is running
    if not test_health():
        print("\n❌ API is not running. Please start the API first.")
        print("   Run: uvicorn api:app --reload --port 8000")
        sys.exit(1)
    
    # Run tests
    print("\n🔍 Running tests...")
    document_id = test_upload()
    
    if not document_id:
        print("\n❌ Some tests failed. Cannot continue with document-specific tests.")
        sys.exit(1)
    
    # Give the document some time to be processed
    print("\n⏳ Waiting for document processing...")
    time.sleep(5)
    
    # Run document-specific tests
    tests = [
        ("Question Answering", test_ask, document_id),
        ("Challenge Generation", test_challenge, document_id),
        ("Answer Evaluation", test_evaluate, document_id)
    ]
    
    results = []
    for name, test_func, *args in tests:
        print(f"\n🧪 Testing: {name}")
        print("-" * 50)
        result = test_func(*args)
        results.append((name, result))
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {'✅' if passed else '❌'} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
