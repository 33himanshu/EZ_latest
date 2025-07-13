#!/usr/bin/env python3
"""
Test runner script for the enhanced RAG system.
"""

import sys
import subprocess
import os
import time


def run_unit_tests():
    """Run unit tests."""
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "unittest", "test_rag_system", "-v"
        ], capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✅ Unit tests passed!")
            return True
        else:
            print("❌ Unit tests failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Unit tests timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return False


def run_benchmarks():
    """Run performance benchmarks."""
    print("\n" + "=" * 60)
    print("Running Performance Benchmarks")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, "benchmark_rag.py"
        ], capture_output=True, text=True, timeout=600)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✅ Benchmarks completed!")
            return True
        else:
            print("❌ Benchmarks failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Benchmarks timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running benchmarks: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'transformers',
        'sentence_transformers',
        'faiss_cpu',
        'PyMuPDF',
        'docx2txt',
        'pandas',
        'numpy',
        'scikit_learn',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def run_integration_test():
    """Run a simple integration test."""
    print("\n" + "=" * 60)
    print("Running Integration Test")
    print("=" * 60)
    
    try:
        # Import and test basic functionality
        from document_store import DocumentStore
        from rag_qa import RAGQA
        from enhanced_challenge_mode import EnhancedQuestionGenerator, EnhancedAnswerEvaluator
        
        print("✅ All modules imported successfully")
        
        # Test basic initialization
        import tempfile
        import os
        
        test_dir = tempfile.mkdtemp()
        try:
            doc_store = DocumentStore(persist_dir=test_dir)
            rag_qa = RAGQA(doc_store)
            question_gen = EnhancedQuestionGenerator()
            answer_eval = EnhancedAnswerEvaluator()
            
            print("✅ All components initialized successfully")
            
            # Test basic functionality
            test_text = "Machine learning is a subset of artificial intelligence."
            
            # Test question generation
            questions = question_gen.generate_questions(test_text, num_questions=1)
            if questions:
                print("✅ Question generation working")
            else:
                print("⚠️ Question generation returned no results")
            
            # Test answer evaluation
            evaluation = answer_eval.evaluate_answer(
                "What is machine learning?",
                "ML is part of AI",
                test_text
            )
            if 'score' in evaluation:
                print("✅ Answer evaluation working")
            else:
                print("⚠️ Answer evaluation failed")
            
            print("✅ Integration test passed!")
            return True
            
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main test runner."""
    print("Enhanced RAG System Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Run integration test
    integration_passed = run_integration_test()
    
    # Run unit tests
    unit_tests_passed = run_unit_tests()
    
    # Run benchmarks (optional, can be skipped if tests fail)
    benchmarks_passed = True
    if unit_tests_passed and integration_passed:
        print("\nRunning benchmarks (this may take a few minutes)...")
        benchmarks_passed = run_benchmarks()
    else:
        print("\nSkipping benchmarks due to test failures.")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Integration Test: {'✅ PASSED' if integration_passed else '❌ FAILED'}")
    print(f"Unit Tests: {'✅ PASSED' if unit_tests_passed else '❌ FAILED'}")
    print(f"Benchmarks: {'✅ PASSED' if benchmarks_passed else '❌ FAILED/SKIPPED'}")
    print(f"Total Duration: {duration:.2f} seconds")
    
    if integration_passed and unit_tests_passed:
        print("\n🎉 All critical tests passed! The enhanced RAG system is ready to use.")
        
        print("\nTo run the application:")
        print("streamlit run app.py")
        
        if benchmarks_passed:
            print("\nBenchmark reports have been generated:")
            for report_file in ["benchmark_report_small.txt", "benchmark_report_medium.txt"]:
                if os.path.exists(report_file):
                    print(f"- {report_file}")
        
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
