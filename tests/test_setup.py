"""
Test script to verify DocuMind setup and basic functionality
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.chunking import SemanticChunker
from src.retriever import HybridRetriever
from src.llm_handler import OllamaHandler
from src.utils import setup_logging

def test_imports():
    """Test if all modules can be imported successfully"""
    print("✅ Testing imports...")
    try:
        from src.document_processor import DocumentProcessor, MetadataEnhancer
        from src.chunking import SemanticChunker
        from src.retriever import HybridRetriever
        from src.llm_handler import AdaptiveQAChain, OllamaHandler
        from src.evaluator import RAGEvaluator, FeedbackProcessor
        from src.utils import StreamlitUtils, ValidationUtils
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_document_processor():
    """Test document processing functionality"""
    print("\n✅ Testing document processor...")
    try:
        processor = DocumentProcessor()
        print("✅ Document processor initialized!")
        return True
    except Exception as e:
        print(f"❌ Document processor error: {e}")
        return False

def test_chunker():
    """Test semantic chunking"""
    print("\n✅ Testing semantic chunker...")
    try:
        chunker = SemanticChunker()
        
        # Test with sample text
        sample_text = """
        This is the first paragraph about artificial intelligence. AI has revolutionized many industries.
        Machine learning is a subset of AI that focuses on algorithms. Deep learning uses neural networks.
        Natural language processing deals with text analysis. Computer vision processes images and videos.
        """
        
        chunks = chunker.chunk_by_semantic_similarity(sample_text)
        print(f"✅ Chunker created {len(chunks)} chunks from sample text!")
        return True
    except Exception as e:
        print(f"❌ Chunker error: {e}")
        return False

def test_retriever():
    """Test retrieval system initialization"""
    print("\n✅ Testing retriever...")
    try:
        retriever = HybridRetriever()
        info = retriever.get_collection_info()
        print(f"✅ Retriever initialized! Collection: {info}")
        return True
    except Exception as e:
        print(f"❌ Retriever error: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("\n✅ Testing Ollama connection...")
    try:
        ollama = OllamaHandler()
        is_available = ollama.is_available()
        
        if is_available:
            print("✅ Ollama is available!")
            
            # Test simple generation
            response = ollama.generate("Hello! Please respond with just 'Hi there!'", max_tokens=50)
            if response:
                print(f"✅ Ollama response: {response[:100]}...")
            else:
                print("⚠️ Ollama available but no response generated")
        else:
            print("⚠️ Ollama is not available. Please ensure it's running with llama3.1:8b model.")
            print("   Install: https://ollama.ai")
            print("   Run: ollama pull llama3.1:8b")
        
        return is_available
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return False

def test_validation_utils():
    """Test validation utilities"""
    print("\n✅ Testing validation utils...")
    try:
        from src.utils import ValidationUtils
        
        # Test query validation
        is_valid, msg = ValidationUtils.validate_query("What is artificial intelligence?")
        assert is_valid, f"Query validation failed: {msg}"
        
        # Test rating validation
        is_valid, msg = ValidationUtils.validate_feedback_rating(5)
        assert is_valid, f"Rating validation failed: {msg}"
        
        print("✅ Validation utils working correctly!")
        return True
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    """Run all tests"""
    setup_logging("INFO")
    
    print("🧠 DocuMind System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_document_processor,
        test_chunker,
        test_retriever,
        test_validation_utils,
        test_ollama_connection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DocuMind is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Start the application: streamlit run app.py")
        print("2. Upload some PDF documents")
        print("3. Start asking questions!")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\n🔧 Common issues:")
        print("- Missing dependencies: pip install -r requirements.txt")
        print("- Ollama not running: ollama pull llama3.1:8b && ollama serve")
        print("- ChromaDB issues: Try deleting data/vectorstore folder")

if __name__ == "__main__":
    main()
