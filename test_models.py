#!/usr/bin/env python3
"""
Test script to check which embedding models are available and working.
"""

from sentence_transformers import SentenceTransformer
import time

def test_model(model_name):
    """Test if a model can be loaded and used."""
    try:
        print(f"🔄 Testing {model_name}...")
        start_time = time.time()
        
        # Try to load the model
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        
        # Test encoding
        test_text = "This is a test sentence for embedding."
        embedding = model.encode([test_text])
        
        print(f"✅ {model_name} - SUCCESS")
        print(f"   Load time: {load_time:.2f}s")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Dimensions: {embedding.shape[1]}")
        return True, embedding.shape[1]
        
    except Exception as e:
        print(f"❌ {model_name} - FAILED: {e}")
        return False, 0

def main():
    """Test multiple embedding models."""
    print("🧪 Testing Embedding Models for RAG System")
    print("=" * 60)
    
    # Models to test in priority order
    models_to_test = [
        # Nomic models (best for RAG)
        'nomic-ai/nomic-embed-text-v1',
        'nomic-ai/nomic-embed-text-v1.5',
        'nomic-ai/nomic-embed-text',
        
        # MixedBread models (high performance)
        'mixedbread-ai/mxbai-embed-large-v1',
        'mixedbread-ai/mxbai-embed-large',
        
        # Reliable alternatives
        'sentence-transformers/all-mpnet-base-v2',
        'all-mpnet-base-v2',
        'all-MiniLM-L6-v2',
        
        # BGE models (good performance)
        'BAAI/bge-large-en-v1.5',
        'BAAI/bge-base-en-v1.5',
    ]
    
    working_models = []
    
    for model_name in models_to_test:
        success, dimensions = test_model(model_name)
        if success:
            working_models.append((model_name, dimensions))
        print("-" * 60)
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 60)
    
    if working_models:
        print("✅ Working Models (in order of preference):")
        for i, (model, dims) in enumerate(working_models, 1):
            print(f"   {i}. {model} ({dims} dimensions)")
        
        print(f"\n🎯 RECOMMENDATION: Use '{working_models[0][0]}'")
        
        # Generate code snippet
        best_model = working_models[0][0]
        print(f"\n💻 CODE TO USE:")
        print(f"self.embedding_model = SentenceTransformer('{best_model}')")
        print(f"self.embedding_dim = {working_models[0][1]}")
        
    else:
        print("❌ No models working! Check your internet connection and sentence-transformers installation.")
        print("\nTry: pip install --upgrade sentence-transformers")

if __name__ == "__main__":
    main()
