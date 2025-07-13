"""
Comprehensive test suite for the enhanced RAG system.
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
import numpy as np

# Import the modules to test
from document_store import DocumentStore
from rag_qa import RAGQA
from enhanced_challenge_mode import EnhancedQuestionGenerator, EnhancedAnswerEvaluator


class TestDocumentStore(unittest.TestCase):
    """Test cases for DocumentStore functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.doc_store = DocumentStore(persist_dir=self.test_dir)
        
        # Create a test document
        self.test_file = os.path.join(self.test_dir, "test_doc.txt")
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("""
# Introduction
This is a test document for the RAG system.

## Section 1: Background
The document contains multiple sections with headers.
It discusses various topics including machine learning and AI.

## Section 2: Methods
We use advanced techniques for document processing.
The system implements hybrid search capabilities.

## Conclusion
This concludes our test document.
            """.strip())
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_document_processing(self):
        """Test document processing functionality."""
        # Test text processing
        text = self.doc_store._process_document(self.test_file)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertIn("Introduction", text)
        self.assertIn("machine learning", text)
    
    def test_metadata_extraction(self):
        """Test metadata extraction."""
        with open(self.test_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        metadata = self.doc_store._extract_document_metadata(text, self.test_file)
        
        # Check basic metadata
        self.assertIn('file_name', metadata)
        self.assertIn('word_count', metadata)
        self.assertIn('document_type', metadata)
        self.assertIn('headers', metadata)
        self.assertIn('key_terms', metadata)
        
        # Check values
        self.assertGreater(metadata['word_count'], 0)
        self.assertGreater(len(metadata['headers']), 0)
        self.assertGreater(len(metadata['key_terms']), 0)
    
    def test_text_chunking(self):
        """Test text chunking functionality."""
        text = "This is a test sentence. This is another sentence. " * 50
        chunks = self.doc_store._chunk_text(text, chunk_size=100, overlap=20)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks have reasonable sizes
        for chunk in chunks:
            word_count = len(chunk.split())
            self.assertLessEqual(word_count, 150)  # Allow some flexibility
    
    def test_document_addition(self):
        """Test adding documents to the store."""
        doc_id = self.doc_store.add_document(self.test_file)
        
        self.assertIsInstance(doc_id, str)
        self.assertGreater(len(doc_id), 0)
        
        # Check that document was added
        self.assertEqual(len(self.doc_store.documents), 1)
        self.assertGreater(len(self.doc_store.chunks), 0)
        
        # Check document metadata
        doc_metadata = self.doc_store.documents[0]
        self.assertEqual(doc_metadata['id'], doc_id)
        self.assertIn('word_count', doc_metadata)
    
    def test_search_functionality(self):
        """Test search functionality."""
        # Add document first
        doc_id = self.doc_store.add_document(self.test_file)
        
        # Test search
        results = self.doc_store.search("machine learning", k=3)
        
        self.assertIsInstance(results, list)
        if results:  # If results found
            self.assertLessEqual(len(results), 3)
            
            # Check result structure
            for result in results:
                self.assertIn('chunk', result)
                self.assertIn('score', result)
                self.assertIn('text', result['chunk'])


class TestRAGQA(unittest.TestCase):
    """Test cases for RAGQA functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.doc_store = DocumentStore(persist_dir=self.test_dir)
        self.rag_qa = RAGQA(self.doc_store)
        
        # Create and add a test document
        self.test_file = os.path.join(self.test_dir, "test_doc.txt")
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("""
Machine learning is a subset of artificial intelligence.
It involves training algorithms on data to make predictions.
Deep learning uses neural networks with multiple layers.
Natural language processing helps computers understand text.
            """.strip())
        
        self.doc_id = self.doc_store.add_document(self.test_file)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_query_expansion(self):
        """Test query expansion functionality."""
        expanded = self.rag_qa._expand_query("what is machine learning")
        
        self.assertIsInstance(expanded, str)
        self.assertIn("what is machine learning", expanded)
        # Should contain additional terms
        self.assertGreater(len(expanded.split()), len("what is machine learning".split()))
    
    def test_question_classification(self):
        """Test question classification."""
        test_cases = [
            ("What is machine learning?", "factual"),
            ("How does deep learning work?", "procedural"),
            ("Why is AI important?", "causal"),
            ("Compare ML and DL", "comparative")
        ]
        
        for question, expected_type in test_cases:
            result = self.rag_qa._classify_question(question)
            self.assertEqual(result, expected_type)
    
    def test_technical_question_detection(self):
        """Test technical question detection."""
        technical_questions = [
            "What are the performance metrics?",
            "How does the algorithm achieve accuracy?",
            "Compare the experimental results"
        ]
        
        non_technical_questions = [
            "What is the main topic?",
            "Who wrote this document?",
            "When was this published?"
        ]
        
        for question in technical_questions:
            self.assertTrue(self.rag_qa._is_technical_question(question))
        
        for question in non_technical_questions:
            self.assertFalse(self.rag_qa._is_technical_question(question))
    
    @patch('transformers.pipeline')
    def test_answer_question(self, mock_pipeline):
        """Test answer generation functionality."""
        # Mock the QA pipeline
        mock_qa = Mock()
        mock_qa.return_value = {
            'answer': 'Machine learning is a subset of AI',
            'score': 0.9
        }
        mock_pipeline.return_value = mock_qa
        
        # Create new RAGQA instance with mocked pipeline
        rag_qa = RAGQA(self.doc_store)
        rag_qa.qa_pipeline = mock_qa
        
        result = rag_qa.answer_question("What is machine learning?", doc_id=self.doc_id)
        
        self.assertIsInstance(result, dict)
        self.assertIn('answer', result)
        self.assertIn('confidence', result)
        self.assertIn('sources', result)


class TestEnhancedQuestionGenerator(unittest.TestCase):
    """Test cases for EnhancedQuestionGenerator."""
    
    def setUp(self):
        """Set up test environment."""
        self.generator = EnhancedQuestionGenerator()
        self.test_text = """
        Machine learning is a method of data analysis that automates analytical model building.
        It is a branch of artificial intelligence based on the idea that systems can learn from data,
        identify patterns and make decisions with minimal human intervention.
        """
    
    def test_key_concept_extraction(self):
        """Test key concept extraction."""
        concepts = self.generator._extract_key_concepts(self.test_text)
        
        self.assertIsInstance(concepts, list)
        # Should extract some concepts
        self.assertGreater(len(concepts), 0)
    
    def test_text_chunking(self):
        """Test text chunking for question generation."""
        chunks = self.generator._split_into_chunks(self.test_text, chunk_size=50)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # Check chunk sizes
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 70)  # Allow some flexibility
    
    def test_question_cleaning(self):
        """Test question cleaning functionality."""
        test_cases = [
            ("Question: What is ML", "What is ML?"),
            ("what is machine learning", "What is machine learning?"),
            ("Q: How does it work", "How does it work?"),
        ]
        
        for input_q, expected in test_cases:
            result = self.generator._clean_question(input_q)
            self.assertEqual(result, expected)
    
    def test_difficulty_assessment(self):
        """Test difficulty assessment."""
        test_cases = [
            ("What is machine learning?", "easy"),
            ("Analyze the performance of the algorithm", "hard"),
            ("How does the system work?", "medium")
        ]
        
        for question, expected_difficulty in test_cases:
            result = self.generator._assess_difficulty(question, self.test_text)
            self.assertEqual(result, expected_difficulty)


class TestEnhancedAnswerEvaluator(unittest.TestCase):
    """Test cases for EnhancedAnswerEvaluator."""
    
    def setUp(self):
        """Set up test environment."""
        self.evaluator = EnhancedAnswerEvaluator()
        self.context = "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
    
    def test_semantic_similarity(self):
        """Test semantic similarity calculation."""
        answer1 = "ML is part of AI that helps computers learn automatically"
        answer2 = "The weather is nice today"
        
        sim1 = self.evaluator._calculate_semantic_similarity(answer1, self.context)
        sim2 = self.evaluator._calculate_semantic_similarity(answer2, self.context)
        
        self.assertIsInstance(sim1, float)
        self.assertIsInstance(sim2, float)
        self.assertGreater(sim1, sim2)  # First answer should be more similar
    
    def test_concept_coverage(self):
        """Test concept coverage assessment."""
        expected_concepts = ["machine learning", "artificial intelligence", "computers"]
        
        good_answer = "Machine learning is part of artificial intelligence for computers"
        poor_answer = "This is about something else entirely"
        
        score1 = self.evaluator._check_concept_coverage(good_answer, expected_concepts)
        score2 = self.evaluator._check_concept_coverage(poor_answer, expected_concepts)
        
        self.assertGreater(score1, score2)
    
    def test_completeness_assessment(self):
        """Test answer completeness assessment."""
        questions = [
            ("What is ML?", "short answer", 0.5),
            ("Explain machine learning in detail", "short", 0.3),
            ("What is ML?", "Machine learning is a comprehensive field that involves...", 1.0)
        ]
        
        for question, answer, min_expected in questions:
            score = self.evaluator._assess_completeness(answer, question)
            if min_expected == 1.0:
                self.assertGreaterEqual(score, min_expected)
            else:
                self.assertLessEqual(score, 1.0)
    
    def test_evaluate_answer(self):
        """Test complete answer evaluation."""
        question = "What is machine learning?"
        good_answer = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        poor_answer = "I don't know"
        
        result1 = self.evaluator.evaluate_answer(question, good_answer, self.context)
        result2 = self.evaluator.evaluate_answer(question, poor_answer, self.context)
        
        # Check result structure
        for result in [result1, result2]:
            self.assertIn('score', result)
            self.assertIn('feedback', result)
            self.assertIn('is_correct', result)
        
        # Good answer should score higher
        self.assertGreater(result1['score'], result2['score'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete RAG system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.doc_store = DocumentStore(persist_dir=self.test_dir)
        self.rag_qa = RAGQA(self.doc_store)
        self.question_generator = EnhancedQuestionGenerator()
        self.answer_evaluator = EnhancedAnswerEvaluator()
        
        # Create a comprehensive test document
        self.test_file = os.path.join(self.test_dir, "comprehensive_doc.txt")
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("""
# Machine Learning Overview

## Introduction
Machine learning (ML) is a subset of artificial intelligence (AI) that provides systems 
the ability to automatically learn and improve from experience without being explicitly programmed.

## Types of Machine Learning
1. Supervised Learning: Uses labeled training data
2. Unsupervised Learning: Finds patterns in unlabeled data  
3. Reinforcement Learning: Learns through interaction with environment

## Applications
Machine learning has applications in:
- Image recognition
- Natural language processing
- Recommendation systems
- Autonomous vehicles

## Conclusion
Machine learning continues to evolve and transform various industries.
            """.strip())
        
        self.doc_id = self.doc_store.add_document(self.test_file)
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Document should be processed and indexed
        self.assertEqual(len(self.doc_store.documents), 1)
        self.assertGreater(len(self.doc_store.chunks), 0)
        
        # 2. Search should work
        results = self.doc_store.search("machine learning types", k=3)
        self.assertGreater(len(results), 0)
        
        # 3. Question generation should work
        with open(self.test_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        questions = self.question_generator.generate_questions(text, num_questions=2)
        self.assertEqual(len(questions), 2)
        
        for question in questions:
            self.assertIn('question', question)
            self.assertIn('type', question)
            self.assertIn('context', question)
        
        # 4. Answer evaluation should work
        test_question = "What is machine learning?"
        test_answer = "Machine learning is a subset of AI that enables automatic learning"
        
        evaluation = self.answer_evaluator.evaluate_answer(
            test_question, test_answer, text[:500]
        )
        
        self.assertIn('score', evaluation)
        self.assertIn('feedback', evaluation)
        self.assertGreater(evaluation['score'], 0)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
