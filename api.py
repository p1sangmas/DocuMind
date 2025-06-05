"""
DocuMind - AI-Powered Knowledge Base Assistant
Flask API Server for Web Interface
"""

import os
import logging
import time
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import our modules
from src.document_processor import DocumentProcessor, MetadataEnhancer
from src.chunking import SemanticChunker
from src.retriever import HybridRetriever
from src.llm_handler import AdaptiveQAChain
from src.evaluator import RAGEvaluator, FeedbackProcessor
from src.utils import (
    setup_logging, timing_decorator, generate_session_id,
    ValidationUtils, safe_execute
)
from config.settings import PAGE_TITLE, AUTO_LOAD_DOCUMENTS, DOCUMENTS_DIR

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

class DocuMindAPI:
    """API wrapper for DocuMind Knowledge Base Assistant"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.metadata_enhancer = MetadataEnhancer()
        self.chunker = SemanticChunker()
        self.retriever = HybridRetriever()
        self.qa_chain = AdaptiveQAChain()
        self.evaluator = RAGEvaluator()
        self.feedback_processor = FeedbackProcessor()
        
        # Track processed files
        self.processed_files = []
        
        # Load documents on startup if auto-load is enabled
        if AUTO_LOAD_DOCUMENTS:
            self._auto_load_documents()
    
    @timing_decorator
    def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process user query and generate response"""
        try:
            # Retrieve relevant documents
            start_time = time.time()
            context_documents = self.retriever.retrieve(query)
            retrieval_time = time.time() - start_time
            
            if not context_documents:
                return {"error": "No relevant documents found for your query."}
            
            # Get conversation history
            conversation_history = []
            # Note: In a real implementation, we would store and retrieve conversation history by session_id
            
            # Generate response
            start_time = time.time()
            response = self.qa_chain.process_query(
                query, 
                context_documents, 
                conversation_history[-10:] if conversation_history else []  # Last 10 messages
            )
            generation_time = time.time() - start_time
            
            if response.get('error'):
                return {"error": response.get('error')}
            
            # Add the response to the result
            result = {
                "answer": response.get('answer', 'No answer generated.'),
                "confidence": response.get('confidence', 'unknown'),
                "sources": response.get('sources', []),
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + generation_time
            }
            
            # Add evaluation metrics
            total_time = retrieval_time + generation_time
            evaluation_results = self.evaluator.evaluate_response(
                query, response, context_documents, response_time=total_time
            )
            
            result["evaluation"] = evaluation_results
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": f"An error occurred while processing your query: {str(e)}"}
    
    def process_feedback(self, query: str, answer: str, rating: int, feedback_text: str, session_id: str) -> Dict[str, Any]:
        """Process user feedback on response"""
        try:
            is_valid, validation_message = ValidationUtils.validate_feedback_rating(rating)
            if not is_valid:
                return {"error": validation_message}
            
            self.feedback_processor.process_user_feedback(
                query, answer, rating, feedback_text, session_id
            )
            
            return {"success": True, "message": "Feedback submitted successfully."}
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {"error": f"An error occurred while processing your feedback: {str(e)}"}
    
    def upload_document(self, file) -> Dict[str, Any]:
        """Upload and process a PDF document"""
        try:
            # Validate file
            is_valid, validation_message = ValidationUtils.validate_pdf_file(file)
            if not is_valid:
                logger.error(f"Document validation failed: {validation_message}")
                return {"error": validation_message}
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # Log before saving
                logger.info(f"Saving temporary file for {file.filename}")
                
                # Read and write to make sure content is transferred correctly
                file_data = file.read()
                if not file_data or len(file_data) == 0:
                    logger.error(f"Empty file data for {file.filename}")
                    return {"error": "File data is empty"}
                    
                tmp_file.write(file_data)
                tmp_file.flush()
                tmp_file_path = tmp_file.name
                
                # Verify the temp file was created with content
                if os.path.getsize(tmp_file_path) == 0:
                    logger.error(f"Temp file is empty: {tmp_file_path}")
                    return {"error": "Failed to save file content"}
                    
                logger.info(f"Temporary file created: {tmp_file_path}, size: {os.path.getsize(tmp_file_path)} bytes")
            
            try:
                # Extract text
                text, method, metadata = self.document_processor.extract_with_fallback(tmp_file_path)
                
                # Update metadata with file info
                filename = file.filename
                display_name = filename
                if display_name.lower().endswith('.pdf'):
                    display_name = display_name[:-4]
                
                metadata.update({
                    'original_filename': filename,
                    'display_name': display_name,
                    'file_size': os.path.getsize(tmp_file_path),
                    'extraction_method': method,
                    'title': display_name
                })
                
                # Chunk text
                chunks = self.chunker.chunk_by_semantic_similarity(text)
                
                # Enhance with metadata
                enhanced_chunks = self.metadata_enhancer.enhance_chunks(chunks, metadata)
                
                # Add to retrieval system
                self.retriever.add_documents(enhanced_chunks)
                
                # Create file info
                file_info = {
                    'filename': filename,
                    'size': os.path.getsize(tmp_file_path),
                    'chunks': len(chunks),
                    'extraction_method': method,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'auto_loaded': False
                }
                
                # Add to processed files list
                self.processed_files.append(file_info)
                
                return {"success": True, "file_info": file_info}
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"error": f"An error occurred while processing your document: {str(e)}"}
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            return self.retriever.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": f"An error occurred while getting collection info: {str(e)}"}
    
    def get_processed_files(self) -> Dict[str, Any]:
        """Get list of processed files"""
        try:
            logger.info(f"Retrieving processed files list, count: {len(self.processed_files)}")
            return {"success": True, "files": self.processed_files}
        except Exception as e:
            logger.error(f"Error getting processed files: {e}")
            return {"error": f"An error occurred while getting processed files: {str(e)}"}
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear all documents from the knowledge base"""
        try:
            self.retriever.clear_collection()
            
            # Clear processed files list
            self.processed_files = []
            
            # Check if collection is actually empty
            collection_info = self.retriever.get_collection_info()
            doc_count = collection_info.get('document_count', 0)
            
            if doc_count > 0:
                logger.warning(f"After clearing, collection still has {doc_count} documents. Trying hard reset.")
                # If documents still exist, try a more forceful approach
                from reset_chroma import reset_chromadb
                reset_chromadb()
                
                # Reinitialize the retriever
                self.retriever = HybridRetriever()
            
            return {"success": True, "message": "Knowledge base cleared successfully."}
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return {"error": f"An error occurred while clearing the knowledge base: {str(e)}"}
    
    def _auto_load_documents(self) -> bool:
        """Automatically load documents from the data/documents directory"""
        from config.settings import DOCUMENTS_DIR, AUTO_LOAD_SKIP_EXISTING
        
        try:
            # Check if directory exists
            documents_dir = Path(DOCUMENTS_DIR)
            if not documents_dir.exists():
                logger.warning(f"Documents directory {documents_dir} does not exist. Skipping auto-load.")
                return False
                
            # Get all PDF files
            pdf_files = list(documents_dir.glob("*.pdf"))
            if not pdf_files:
                logger.info("No PDF files found in documents directory. Skipping auto-load.")
                return False
                
            logger.info(f"Auto-loading {len(pdf_files)} documents from {documents_dir}")
                
            # Process each PDF file
            processed_documents = []
            
            for pdf_path in pdf_files:
                try:
                    filename = pdf_path.name
                    logger.info(f"Auto-loading document: {filename}")
                    
                    # Extract text
                    text, method, metadata = self.document_processor.extract_with_fallback(str(pdf_path))
                    
                    # Get file size
                    file_size = pdf_path.stat().st_size
                    
                    # Update metadata
                    display_name = filename
                    if display_name.lower().endswith('.pdf'):
                        display_name = display_name[:-4]
                    
                    metadata.update({
                        'original_filename': filename,
                        'display_name': display_name,
                        'file_size': file_size,
                        'extraction_method': method,
                        'title': display_name
                    })
                    
                    # Chunk text
                    chunks = self.chunker.chunk_by_semantic_similarity(text)
                    
                    # Enhance with metadata
                    enhanced_chunks = self.metadata_enhancer.enhance_chunks(chunks, metadata)
                    
                    # Add to processed documents
                    processed_documents.extend(enhanced_chunks)
                    
                    # Add to processed files list (instance variable)
                    self.processed_files.append({
                        'filename': filename,
                        'size': file_size,
                        'chunks': len(chunks),
                        'extraction_method': method,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'auto_loaded': True
                    })
                    
                    logger.info(f"Successfully processed {filename} ({len(chunks)} chunks)")
                    
                except Exception as e:
                    logger.error(f"Error auto-loading {pdf_path.name}: {e}")
                    
            # Add documents to retrieval system
            if processed_documents:
                try:
                    self.retriever.add_documents(processed_documents)
                    logger.info(f"Successfully auto-loaded {len(processed_documents)} document chunks into knowledge base")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error adding auto-loaded documents to knowledge base: {e}")
                    
            return len(processed_documents) > 0
            
        except Exception as e:
            logger.error(f"Error in auto-loading documents: {e}")
            return False


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize API
api = DocuMindAPI()

# Serve web frontend
@app.route('/')
def index():
    try:
        return send_from_directory(os.path.join(os.getcwd(), 'web'), 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return jsonify({'error': 'Page not found'}), 404

@app.route('/<path:path>')
def serve_static(path):
    # Handle API routes first
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    
    # Handle static files (CSS, JS, images)
    try:
        return send_from_directory(os.path.join(os.getcwd(), 'web'), path)
    except Exception as e:
        logger.error(f"Error serving static file {path}: {e}")
        # For SPA routing, fallback to index.html for unknown routes
        if not path.startswith(('css/', 'js/', 'assets/')):
            try:
                return send_from_directory(os.path.join(os.getcwd(), 'web'), 'index.html')
            except:
                pass
        return jsonify({'error': 'File not found'}), 404

# API endpoint to get system status
@app.route('/api/status', methods=['GET'])
def status():
    collection_info = api.get_collection_info()
    
    from src.document_processor import OCR_AVAILABLE
    ollama_status = api.qa_chain.ollama.is_available()
    
    status_data = {
        'status': 'ok',
        'system_name': PAGE_TITLE,
        'ocr_available': OCR_AVAILABLE,
        'ollama_available': ollama_status,
        'collection': collection_info
    }
    
    return jsonify(convert_numpy_types(status_data))

# API endpoint to process queries
@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query']
    session_id = data.get('session_id', generate_session_id())
    
    # Validate query
    is_valid, validation_message = ValidationUtils.validate_query(query)
    if not is_valid:
        return jsonify({'error': validation_message}), 400
    
    result = api.process_query(query, session_id)
    
    if 'error' in result:
        return jsonify(convert_numpy_types(result)), 500
    
    # Convert numpy types to JSON-serializable types
    result = convert_numpy_types(result)
    return jsonify(result)

# API endpoint to submit feedback
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    
    if not data:
        return jsonify({'error': 'No feedback data provided'}), 400
    
    query = data.get('query', '')
    answer = data.get('answer', '')
    rating = data.get('rating', 0)
    feedback_text = data.get('feedback_text', '')
    session_id = data.get('session_id', generate_session_id())
    
    result = api.process_feedback(query, answer, rating, feedback_text, session_id)
    
    if 'error' in result:
        return jsonify(convert_numpy_types(result)), 400
    
    return jsonify(convert_numpy_types(result))

# API endpoint to upload documents
@app.route('/api/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        logger.error("Upload failed: No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.error("Upload failed: Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    # Log file details
    logger.info(f"Uploading file: {file.filename}, Content-Type: {file.content_type}, Size: {file.content_length or 'unknown'}")
    
    # Handle file content
    try:
        # Read the file data to check if it's empty
        file_data = file.read()
        file_size = len(file_data)
        logger.info(f"File data size: {file_size} bytes")
        
        if file_size == 0:
            logger.error(f"Upload failed: File {file.filename} is empty (0 bytes)")
            return jsonify({'error': 'File is empty'}), 400
        
        # Rewind the file
        file.seek(0)
        
        # Proceed with processing
        result = api.upload_document(file)
        
        if 'error' in result:
            logger.error(f"Upload failed: {result['error']}")
            return jsonify(convert_numpy_types(result)), 400
        
        logger.info(f"File uploaded successfully: {file.filename}")
        return jsonify(convert_numpy_types(result))
    
    except Exception as e:
        logger.error(f"Upload exception: {str(e)}")
        return jsonify({'error': f"File upload error: {str(e)}"}), 500

# API endpoint to get collection info
@app.route('/api/collection', methods=['GET'])
def get_collection():
    result = api.get_collection_info()
    
    if 'error' in result:
        return jsonify(convert_numpy_types(result)), 500
    
    return jsonify(convert_numpy_types(result))

# API endpoint to clear knowledge base
@app.route('/api/collection', methods=['DELETE'])
def clear_collection():
    result = api.clear_knowledge_base()
    
    if 'error' in result:
        return jsonify(convert_numpy_types(result)), 500
    
    return jsonify(convert_numpy_types(result))

# API endpoint to get processed files
@app.route('/api/files', methods=['GET'])
def get_files():
    try:
        result = api.get_processed_files()
        
        if 'error' in result:
            logger.error(f"Error in /api/files endpoint: {result['error']}")
            return jsonify(convert_numpy_types(result)), 500
        
        # Log success
        logger.info(f"Successfully retrieved file list. Count: {len(result.get('files', []))}")
        return jsonify(convert_numpy_types(result))
    except Exception as e:
        logger.error(f"Unexpected error in /api/files endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
