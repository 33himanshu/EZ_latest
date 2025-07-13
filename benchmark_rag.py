"""
Performance benchmarking script for the enhanced RAG system.
"""

import time
import tempfile
import os
import shutil
import statistics
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

from document_store import DocumentStore
from rag_qa import RAGQA
from enhanced_challenge_mode import EnhancedQuestionGenerator, EnhancedAnswerEvaluator


class RAGBenchmark:
    """Benchmark suite for RAG system performance."""
    
    def __init__(self):
        """Initialize benchmark environment."""
        self.test_dir = tempfile.mkdtemp()
        self.results = {}
        
    def cleanup(self):
        """Clean up benchmark environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_documents(self, num_docs: int = 5, doc_size: str = "medium") -> List[str]:
        """Create test documents of varying sizes."""
        documents = []
        
        # Base content templates
        small_content = "This is a small test document. " * 50
        medium_content = "This is a medium test document with more content. " * 200
        large_content = "This is a large test document with extensive content. " * 1000
        
        content_map = {
            "small": small_content,
            "medium": medium_content,
            "large": large_content
        }
        
        base_content = content_map.get(doc_size, medium_content)
        
        for i in range(num_docs):
            doc_path = os.path.join(self.test_dir, f"test_doc_{i}.txt")
            
            # Add some variation to each document
            content = f"# Document {i+1}\n\n{base_content}\n\nSpecific content for document {i+1}."
            
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            documents.append(doc_path)
        
        return documents
    
    def benchmark_document_processing(self, doc_paths: List[str]) -> Dict[str, Any]:
        """Benchmark document processing performance."""
        print("Benchmarking document processing...")
        
        doc_store = DocumentStore(persist_dir=os.path.join(self.test_dir, "benchmark_store"))
        
        processing_times = []
        indexing_times = []
        
        for doc_path in doc_paths:
            # Time document processing
            start_time = time.time()
            text = doc_store._process_document(doc_path)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Time document indexing
            start_time = time.time()
            doc_id = doc_store.add_document(doc_path)
            indexing_time = time.time() - start_time
            indexing_times.append(indexing_time)
        
        return {
            'processing_times': processing_times,
            'indexing_times': indexing_times,
            'avg_processing_time': statistics.mean(processing_times),
            'avg_indexing_time': statistics.mean(indexing_times),
            'total_documents': len(doc_paths),
            'total_chunks': len(doc_store.chunks)
        }
    
    def benchmark_search_performance(self, doc_store: DocumentStore, queries: List[str]) -> Dict[str, Any]:
        """Benchmark search performance."""
        print("Benchmarking search performance...")
        
        search_times = []
        result_counts = []
        
        for query in queries:
            start_time = time.time()
            results = doc_store.search(query, k=5)
            search_time = time.time() - start_time
            
            search_times.append(search_time)
            result_counts.append(len(results))
        
        return {
            'search_times': search_times,
            'result_counts': result_counts,
            'avg_search_time': statistics.mean(search_times),
            'avg_results_per_query': statistics.mean(result_counts),
            'total_queries': len(queries)
        }
    
    def benchmark_qa_performance(self, rag_qa: RAGQA, questions: List[str]) -> Dict[str, Any]:
        """Benchmark question answering performance."""
        print("Benchmarking QA performance...")
        
        qa_times = []
        confidence_scores = []
        
        for question in questions:
            start_time = time.time()
            try:
                result = rag_qa.answer_question(question)
                qa_time = time.time() - start_time
                
                qa_times.append(qa_time)
                confidence_scores.append(result.get('confidence', 0))
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                qa_times.append(float('inf'))
                confidence_scores.append(0)
        
        return {
            'qa_times': qa_times,
            'confidence_scores': confidence_scores,
            'avg_qa_time': statistics.mean([t for t in qa_times if t != float('inf')]),
            'avg_confidence': statistics.mean(confidence_scores),
            'total_questions': len(questions),
            'successful_answers': len([t for t in qa_times if t != float('inf')])
        }
    
    def benchmark_question_generation(self, generator: EnhancedQuestionGenerator, texts: List[str]) -> Dict[str, Any]:
        """Benchmark question generation performance."""
        print("Benchmarking question generation...")
        
        generation_times = []
        question_counts = []
        
        for text in texts:
            start_time = time.time()
            try:
                questions = generator.generate_questions(text, num_questions=3)
                generation_time = time.time() - start_time
                
                generation_times.append(generation_time)
                question_counts.append(len(questions))
            except Exception as e:
                print(f"Error generating questions: {e}")
                generation_times.append(float('inf'))
                question_counts.append(0)
        
        return {
            'generation_times': generation_times,
            'question_counts': question_counts,
            'avg_generation_time': statistics.mean([t for t in generation_times if t != float('inf')]),
            'avg_questions_per_text': statistics.mean(question_counts),
            'total_texts': len(texts)
        }
    
    def benchmark_answer_evaluation(self, evaluator: EnhancedAnswerEvaluator, 
                                  qa_pairs: List[tuple]) -> Dict[str, Any]:
        """Benchmark answer evaluation performance."""
        print("Benchmarking answer evaluation...")
        
        evaluation_times = []
        scores = []
        
        for question, answer, context in qa_pairs:
            start_time = time.time()
            try:
                result = evaluator.evaluate_answer(question, answer, context)
                evaluation_time = time.time() - start_time
                
                evaluation_times.append(evaluation_time)
                scores.append(result.get('score', 0))
            except Exception as e:
                print(f"Error evaluating answer: {e}")
                evaluation_times.append(float('inf'))
                scores.append(0)
        
        return {
            'evaluation_times': evaluation_times,
            'scores': scores,
            'avg_evaluation_time': statistics.mean([t for t in evaluation_times if t != float('inf')]),
            'avg_score': statistics.mean(scores),
            'total_evaluations': len(qa_pairs)
        }
    
    def run_comprehensive_benchmark(self, doc_size: str = "medium", num_docs: int = 5) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        print(f"Running comprehensive benchmark with {num_docs} {doc_size} documents...")
        
        # Create test documents
        doc_paths = self.create_test_documents(num_docs, doc_size)
        
        # Initialize components
        doc_store = DocumentStore(persist_dir=os.path.join(self.test_dir, "benchmark_store"))
        
        # Benchmark document processing
        processing_results = self.benchmark_document_processing(doc_paths)
        
        # Initialize RAG components
        rag_qa = RAGQA(doc_store)
        question_generator = EnhancedQuestionGenerator()
        answer_evaluator = EnhancedAnswerEvaluator()
        
        # Test queries
        test_queries = [
            "machine learning",
            "artificial intelligence",
            "data analysis",
            "neural networks",
            "deep learning"
        ]
        
        # Benchmark search
        search_results = self.benchmark_search_performance(doc_store, test_queries)
        
        # Test questions
        test_questions = [
            "What is machine learning?",
            "How does AI work?",
            "What are the applications of data science?",
            "Explain neural networks",
            "What is deep learning?"
        ]
        
        # Benchmark QA
        qa_results = self.benchmark_qa_performance(rag_qa, test_questions)
        
        # Get document texts for question generation
        doc_texts = []
        for doc_path in doc_paths[:2]:  # Use first 2 docs to save time
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_texts.append(f.read())
        
        # Benchmark question generation
        generation_results = self.benchmark_question_generation(question_generator, doc_texts)
        
        # Create QA pairs for evaluation benchmark
        qa_pairs = [
            ("What is AI?", "Artificial intelligence is machine intelligence", "AI context"),
            ("How does ML work?", "Machine learning uses algorithms", "ML context"),
            ("What is data science?", "Data science analyzes data", "Data context")
        ]
        
        # Benchmark answer evaluation
        evaluation_results = self.benchmark_answer_evaluation(answer_evaluator, qa_pairs)
        
        # Compile comprehensive results
        comprehensive_results = {
            'document_processing': processing_results,
            'search_performance': search_results,
            'qa_performance': qa_results,
            'question_generation': generation_results,
            'answer_evaluation': evaluation_results,
            'system_info': {
                'document_size': doc_size,
                'num_documents': num_docs,
                'total_chunks': processing_results['total_chunks']
            }
        }
        
        return comprehensive_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("# RAG System Performance Benchmark Report")
        report.append("=" * 50)
        
        # System info
        system_info = results['system_info']
        report.append(f"\n## System Configuration")
        report.append(f"- Document Size: {system_info['document_size']}")
        report.append(f"- Number of Documents: {system_info['num_documents']}")
        report.append(f"- Total Chunks: {system_info['total_chunks']}")
        
        # Document processing
        doc_proc = results['document_processing']
        report.append(f"\n## Document Processing Performance")
        report.append(f"- Average Processing Time: {doc_proc['avg_processing_time']:.3f}s")
        report.append(f"- Average Indexing Time: {doc_proc['avg_indexing_time']:.3f}s")
        
        # Search performance
        search = results['search_performance']
        report.append(f"\n## Search Performance")
        report.append(f"- Average Search Time: {search['avg_search_time']:.3f}s")
        report.append(f"- Average Results per Query: {search['avg_results_per_query']:.1f}")
        
        # QA performance
        qa = results['qa_performance']
        report.append(f"\n## Question Answering Performance")
        report.append(f"- Average QA Time: {qa['avg_qa_time']:.3f}s")
        report.append(f"- Average Confidence: {qa['avg_confidence']:.3f}")
        report.append(f"- Success Rate: {qa['successful_answers']}/{qa['total_questions']}")
        
        # Question generation
        gen = results['question_generation']
        report.append(f"\n## Question Generation Performance")
        report.append(f"- Average Generation Time: {gen['avg_generation_time']:.3f}s")
        report.append(f"- Average Questions per Text: {gen['avg_questions_per_text']:.1f}")
        
        # Answer evaluation
        eval_res = results['answer_evaluation']
        report.append(f"\n## Answer Evaluation Performance")
        report.append(f"- Average Evaluation Time: {eval_res['avg_evaluation_time']:.3f}s")
        report.append(f"- Average Score: {eval_res['avg_score']:.1f}")
        
        return "\n".join(report)


def main():
    """Run benchmark suite."""
    benchmark = RAGBenchmark()
    
    try:
        # Run benchmarks for different document sizes
        sizes = ["small", "medium"]
        all_results = {}
        
        for size in sizes:
            print(f"\n{'='*60}")
            print(f"Running benchmark for {size} documents...")
            print(f"{'='*60}")
            
            results = benchmark.run_comprehensive_benchmark(doc_size=size, num_docs=3)
            all_results[size] = results
            
            # Generate and print report
            report = benchmark.generate_report(results)
            print(report)
            
            # Save report to file
            report_file = f"benchmark_report_{size}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to {report_file}")
        
        print(f"\n{'='*60}")
        print("Benchmark completed successfully!")
        print(f"{'='*60}")
        
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
