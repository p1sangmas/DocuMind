#!/usr/bin/env python3
"""
Comprehensive test suite for DocuMind AI-Powered Knowledge Base Assistant
Tests all major components and functionality
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

def setup_test_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported successfully"""
    logger.info("🔍 Testing module imports...")
    
    try:
        from src.document_processor import DocumentProcessor, MetadataEnhancer
        from src.chunking import SemanticChunker
        from src.retriever import HybridRetriever
        from src.llm_handler import AdaptiveQAChain, OllamaHandler, PromptTemplateManager
        from src.evaluator import RAGEvaluator, FeedbackProcessor
        from src.utils import (
            setup_logging, timing_decorator, generate_session_id,
            StreamlitUtils, ValidationUtils, safe_execute
        )
        from config.settings import *
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_document_processing():
    """Test document processing pipeline"""
    logger.info("📄 Testing document processing...")
    
    try:
        from src.document_processor import DocumentProcessor, MetadataEnhancer
        
        processor = DocumentProcessor()
        metadata_enhancer = MetadataEnhancer()
        
        # Test with a sample PDF
        test_pdf = "data/documents/test_policy.pdf"
        if not os.path.exists(test_pdf):
            logger.warning(f"Test PDF not found: {test_pdf}")
            return False
        
        # Extract text
        text, method, metadata = processor.extract_with_fallback(test_pdf)
        
        if text and len(text) > 0:
            logger.info(f"✅ Text extraction successful using {method}")
            logger.info(f"   - Extracted {len(text)} characters")
            logger.info(f"   - Metadata: {metadata}")
            return True
        else:
            logger.error("❌ Text extraction failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Document processing test failed: {e}")
        return False

def test_semantic_chunking():
    """Test semantic chunking functionality"""
    logger.info("✂️ Testing semantic chunking...")
    
    try:
        from src.chunking import SemanticChunker
        
        chunker = SemanticChunker()
        
        # Test text
        test_text = """
        This is the first paragraph about remote work policies. 
        It contains information about eligibility and requirements.
        
        This is the second paragraph about health benefits.
        It discusses insurance coverage and employee contributions.
        
        This is the third paragraph about vacation policies.
        It outlines the different types of leave available to employees.
        """
        
        chunks = chunker.chunk_by_semantic_similarity(test_text)
        
        if chunks and len(chunks) > 0:
            logger.info(f"✅ Semantic chunking successful")
            logger.info(f"   - Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                logger.info(f"   - Chunk {i+1}: {len(chunk)} characters")
            return True
        else:
            logger.error("❌ Semantic chunking failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Semantic chunking test failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama LLM connection"""
    logger.info("🤖 Testing Ollama connection...")
    
    try:
        from src.llm_handler import OllamaHandler
        
        ollama = OllamaHandler()
        
        if ollama.is_available():
            logger.info("✅ Ollama service is available")
            
            # Test simple generation
            response = ollama.generate("Hello, can you respond with 'Hello from Ollama'?", max_tokens=50)
            
            if response and len(response) > 0:
                logger.info(f"✅ Ollama generation successful: {response[:100]}...")
                return True
            else:
                logger.error("❌ Ollama generation failed")
                return False
        else:
            logger.error("❌ Ollama service not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ollama test failed: {e}")
        return False

def test_vector_database():
    """Test ChromaDB vector database functionality"""
    logger.info("🗂️ Testing vector database...")
    
    try:
        from src.retriever import HybridRetriever
        
        retriever = HybridRetriever()
        
        # Test document addition
        test_documents = [
            {
                'content': 'Remote work policy allows employees to work from home up to 3 days per week.',
                'metadata': {
                    'source_document': 'test_policy.pdf',
                    'page_number': 1,
                    'chunk_id': 'test_chunk_1'
                }
            },
            {
                'content': 'Health insurance covers medical, dental, and vision with 80% company contribution.',
                'metadata': {
                    'source_document': 'test_benefits.pdf',
                    'page_number': 1,
                    'chunk_id': 'test_chunk_2'
                }
            }
        ]
        
        retriever.add_documents(test_documents)
        logger.info("✅ Documents added to vector database")
        
        # Test retrieval
        results = retriever.retrieve("What is the remote work policy?", top_k=2)
        
        if results and len(results) > 0:
            logger.info(f"✅ Vector search successful")
            logger.info(f"   - Retrieved {len(results)} documents")
            for i, doc in enumerate(results):
                logger.info(f"   - Result {i+1}: {doc['content'][:50]}...")
            return True
        else:
            logger.error("❌ Vector search failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Vector database test failed: {e}")
        return False

def test_end_to_end_qa():
    """Test end-to-end question answering"""
    logger.info("💬 Testing end-to-end Q&A...")
    
    try:
        from src.retriever import HybridRetriever
        from src.llm_handler import AdaptiveQAChain
        
        retriever = HybridRetriever()
        qa_chain = AdaptiveQAChain()
        
        # Ensure we have test documents
        test_documents = [
            {
                'content': 'Our company remote work policy allows full-time employees to work from home up to 3 days per week. Employees must maintain core hours between 10 AM and 3 PM EST.',
                'metadata': {
                    'source_document': 'remote_work_policy.pdf',
                    'page_number': 1,
                    'chunk_id': 'policy_chunk_1'
                }
            }
        ]
        
        retriever.add_documents(test_documents)
        
        # Test query
        query = "How many days per week can employees work from home?"
        context_documents = retriever.retrieve(query)
        
        if context_documents:
            response = qa_chain.process_query(query, context_documents)
            
            if response and response.get('answer'):
                logger.info("✅ End-to-end Q&A successful")
                logger.info(f"   - Query: {query}")
                logger.info(f"   - Answer: {response['answer'][:100]}...")
                logger.info(f"   - Confidence: {response.get('confidence', 'unknown')}")
                logger.info(f"   - Sources: {len(response.get('sources', []))}")
                return True
            else:
                logger.error("❌ Q&A response generation failed")
                return False
        else:
            logger.error("❌ Document retrieval failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ End-to-end Q&A test failed: {e}")
        return False

def test_evaluation_framework():
    """Test evaluation framework"""
    logger.info("📊 Testing evaluation framework...")
    
    try:
        from src.evaluator import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # Mock data for testing
        query = "What is the remote work policy?"
        response = {
            'answer': 'Employees can work from home up to 3 days per week with core hours 10 AM - 3 PM EST.',
            'confidence': 'high',
            'sources': [{'document': 'remote_work_policy.pdf'}]
        }
        context_documents = [
            {
                'content': 'Remote work policy allows employees to work from home up to 3 days per week.',
                'metadata': {'source_document': 'remote_work_policy.pdf'}
            }
        ]
        
        # Test evaluation
        evaluation_results = evaluator.evaluate_response(query, response, context_documents, response_time=2.5)
        
        if evaluation_results and len(evaluation_results) > 0:
            logger.info("✅ Evaluation framework successful")
            for metric, score in evaluation_results.items():
                logger.info(f"   - {metric}: {score:.3f}")
            return True
        else:
            logger.error("❌ Evaluation framework failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Evaluation framework test failed: {e}")
        return False

def test_feedback_system():
    """Test feedback collection system"""
    logger.info("📝 Testing feedback system...")
    
    try:
        from src.evaluator import FeedbackProcessor
        
        feedback_processor = FeedbackProcessor()
        
        # Test feedback storage
        feedback_processor.process_user_feedback(
            query="Test question",
            response="Test answer",
            rating=4,
            corrections="Good response",
            session_id="test_session"
        )
        
        # Test feedback analysis
        analysis = feedback_processor.get_feedback_analysis(days=30)
        
        if analysis and 'total_feedback' in analysis:
            logger.info("✅ Feedback system successful")
            logger.info(f"   - Analysis: {analysis}")
            return True
        else:
            logger.error("❌ Feedback system failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Feedback system test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    logger.info("🔧 Testing utility functions...")
    
    try:
        from src.utils import (
            generate_session_id, sanitize_filename, format_file_size,
            ValidationUtils, safe_execute
        )
        
        # Test session ID generation
        session_id = generate_session_id()
        assert len(session_id) == 8, "Session ID should be 8 characters"
        
        # Test filename sanitization
        safe_name = sanitize_filename("test<>file.pdf")
        assert "<" not in safe_name and ">" not in safe_name, "Unsafe characters should be removed"
        
        # Test file size formatting
        formatted_size = format_file_size(1024)
        assert "KB" in formatted_size, "Should format as KB"
        
        # Test validation
        is_valid, message = ValidationUtils.validate_query("Test query")
        assert is_valid, "Valid query should pass validation"
        
        logger.info("✅ Utility functions successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility functions test failed: {e}")
        return False

def run_performance_test():
    """Run basic performance tests"""
    logger.info("⚡ Running performance tests...")
    
    try:
        from src.retriever import HybridRetriever
        
        retriever = HybridRetriever()
        
        # Add multiple documents for performance testing
        test_documents = []
        for i in range(10):
            test_documents.append({
                'content': f'This is test document {i} with content about various policies and procedures. ' * 10,
                'metadata': {
                    'source_document': f'test_doc_{i}.pdf',
                    'chunk_id': f'chunk_{i}'
                }
            })
        
        start_time = time.time()
        retriever.add_documents(test_documents)
        add_time = time.time() - start_time
        
        start_time = time.time()
        results = retriever.retrieve("test document policies", top_k=5)
        search_time = time.time() - start_time
        
        logger.info("✅ Performance test successful")
        logger.info(f"   - Document addition: {add_time:.3f}s for {len(test_documents)} docs")
        logger.info(f"   - Search time: {search_time:.3f}s")
        logger.info(f"   - Retrieved {len(results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger = setup_test_logging()
    
    logger.info("🚀 Starting DocuMind comprehensive test suite...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Document Processing", test_document_processing),
        ("Semantic Chunking", test_semantic_chunking),
        ("Ollama Connection", test_ollama_connection),
        ("Vector Database", test_vector_database),
        ("End-to-End Q&A", test_end_to_end_qa),
        ("Evaluation Framework", test_evaluation_framework),
        ("Feedback System", test_feedback_system),
        ("Utility Functions", test_utility_functions),
        ("Performance", run_performance_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("🎯 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! DocuMind is ready for use.")
    else:
        logger.warning(f"⚠️ {total-passed} tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
