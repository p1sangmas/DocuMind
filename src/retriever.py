"""Hybrid retrieval system combining semantic and keyword-based search"""

import logging
from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from config.settings import (
    EMBEDDING_MODEL, CROSS_ENCODER_MODEL, TOP_K_DOCUMENTS,
    SEMANTIC_WEIGHT, KEYWORD_WEIGHT, VECTORSTORE_PERSIST_DIRECTORY,
    VECTORSTORE_COLLECTION_NAME
)

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines semantic similarity with keyword matching for optimal results"""
    
    def __init__(self):
        # Initialize embedding model with CPU device to avoid MPS issues
        import torch
        device = 'cpu'  # Force CPU for now to avoid MPS device issues
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        self.cross_encoder = None  # Lazy loaded
        self.semantic_weight = SEMANTIC_WEIGHT
        self.keyword_weight = KEYWORD_WEIGHT
        
        # Initialize keyword search components first
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents = []
        self.document_ids = []
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=VECTORSTORE_PERSIST_DIRECTORY,
            settings=Settings(allow_reset=True)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(VECTORSTORE_COLLECTION_NAME)
            logger.info(f"Loaded existing collection: {VECTORSTORE_COLLECTION_NAME}")
            # Load existing documents for keyword search
            self._load_existing_documents()
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=VECTORSTORE_COLLECTION_NAME,
                metadata={"description": "Document chunks with semantic embeddings"}
            )
            logger.info(f"Created new collection: {VECTORSTORE_COLLECTION_NAME}")
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both semantic and keyword indexes"""
        logger.info(f"Adding {len(documents)} documents to retrieval system")
        
        # Prepare data for ChromaDB
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            # Create a more user-friendly ID that includes original filename if available
            filename = doc['metadata'].get('original_filename', f'document_{i}')
            
            # Clean filename for ID usage - keep alphanumeric and underscores only
            clean_filename = ''.join(c if c.isalnum() else '_' for c in filename)
            
            # Ensure we don't have too many underscores by replacing multiple with single
            while '__' in clean_filename:
                clean_filename = clean_filename.replace('__', '_')
                
            # Remove .pdf extension if present
            if clean_filename.lower().endswith('_pdf'):
                clean_filename = clean_filename[:-4]
                
            # Create a unique identifier with the filename
            doc_id = f"{clean_filename}_{i}"
            
            texts.append(doc['content'])
            metadatas.append(doc['metadata'])
            ids.append(doc_id)
            
            # Store for keyword search
            self.documents.append(doc['content'])
            self.document_ids.append(doc_id)
        
        # Add to ChromaDB (embeddings will be generated automatically)
        try:
            # Generate embeddings manually for better control
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            logger.info("Successfully added documents to semantic index")
        except Exception as e:
            logger.error(f"Failed to add documents to semantic index: {e}")
            raise
        
        # Build keyword search index
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
            logger.info("Successfully built keyword search index")
        except Exception as e:
            logger.error(f"Failed to build keyword search index: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = TOP_K_DOCUMENTS) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid approach"""
        # Check if collection has documents
        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("No documents in retrieval system")
                return []
        except Exception as e:
            logger.error(f"Failed to check collection count: {e}")
            return []
        
        # If documents list is empty but collection has documents, reload them
        if not self.documents and collection_count > 0:
            logger.info("Documents list empty but collection has documents, reloading...")
            self._load_existing_documents()
        
        try:
            # Semantic retrieval
            semantic_results = self._semantic_search(query, top_k * 2)
            
            # Keyword retrieval (only if we have documents and tfidf matrix)
            keyword_results = []
            if self.documents and self.tfidf_matrix is not None:
                keyword_results = self._keyword_search(query, top_k * 2)
            
            # Combine and rerank results
            combined_results = self._combine_and_rerank(
                semantic_results, keyword_results, query, top_k
            )
            
            logger.info(f"Retrieved {len(combined_results)} documents for query: {query[:50]}...")
            return combined_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform semantic similarity search using ChromaDB"""
        try:
            # Get collection count for proper limit
            collection_count = self.collection.count()
            n_results = min(top_k, collection_count) if collection_count > 0 else top_k
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            semantic_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i],
                    'semantic_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'source': 'semantic'
                }
                semantic_results.append(result)
            
            return semantic_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search using TF-IDF"""
        if self.tfidf_matrix is None:
            return []
        
        try:
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            keyword_results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include relevant results
                    # Try to get real metadata from collection by document ID
                    doc_id = self.document_ids[idx]
                    
                    # Use the document ID to look up actual metadata if possible
                    try:
                        # Look up the document in the collection
                        lookup_results = self.collection.get(ids=[doc_id])
                        if lookup_results and lookup_results['metadatas'] and lookup_results['metadatas'][0]:
                            doc_metadata = lookup_results['metadatas'][0]
                        else:
                            # Fallback with enriched metadata
                            doc_metadata = {
                                'source_document': doc_id,
                                'title': f"Document {doc_id}",
                                'author': "Unknown",
                                'creation_date': "Unknown"
                            }
                    except:
                        # Fallback if lookup fails
                        doc_metadata = {
                            'source_document': doc_id,
                            'title': f"Document {doc_id}",
                            'author': "Unknown",
                            'creation_date': "Unknown"
                        }
                    
                    result = {
                        'content': self.documents[idx],
                        'metadata': doc_metadata,
                        'id': doc_id,
                        'keyword_score': similarities[idx],
                        'source': 'keyword'
                    }
                    keyword_results.append(result)
            
            return keyword_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_and_rerank(self, semantic_results: List[Dict], 
                          keyword_results: List[Dict], query: str, top_k: int) -> List[Dict[str, Any]]:
        """Combine and rerank results using hybrid scoring"""
        
        # Create a unified result set
        all_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result['id']
            all_results[doc_id] = {
                'content': result['content'],
                'metadata': result['metadata'],
                'semantic_score': result.get('semantic_score', 0),
                'keyword_score': 0
            }
        
        # Merge keyword results
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in all_results:
                all_results[doc_id]['keyword_score'] = result.get('keyword_score', 0)
            else:
                all_results[doc_id] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'semantic_score': 0,
                    'keyword_score': result.get('keyword_score', 0)
                }
        
        # Calculate hybrid scores
        final_results = []
        for doc_id, data in all_results.items():
            # Normalize scores (simple min-max normalization)
            semantic_score = data['semantic_score']
            keyword_score = data['keyword_score']
            
            # Hybrid score calculation
            hybrid_score = (
                semantic_score * self.semantic_weight +
                keyword_score * self.keyword_weight
            )
            
            final_results.append({
                'content': data['content'],
                'metadata': data['metadata'],
                'score': hybrid_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score
            })
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply cross-encoder reranking if available
        if len(final_results) > 1:
            final_results = self._cross_encoder_rerank(query, final_results[:top_k*2])
        
        return final_results[:top_k]
    
    def _cross_encoder_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Apply cross-encoder reranking for final precision"""
        try:
            if self.cross_encoder is None:
                self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            
            # Prepare pairs for cross-encoder
            pairs = [(query, result['content']) for result in results]
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Update results with cross-encoder scores
            for i, result in enumerate(results):
                result['cross_score'] = cross_scores[i]
                # Boost final score with cross-encoder
                result['score'] = result['score'] * 0.6 + cross_scores[i] * 0.4
            
            # Re-sort by updated scores
            results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info("Applied cross-encoder reranking")
            
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                'collection_name': VECTORSTORE_COLLECTION_NAME,
                'document_count': count,
                'embedding_model': EMBEDDING_MODEL,
                'has_keyword_index': self.tfidf_matrix is not None
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            # First attempt to delete the collection
            try:
                self.chroma_client.delete_collection(VECTORSTORE_COLLECTION_NAME)
                logger.info(f"Deleted collection: {VECTORSTORE_COLLECTION_NAME}")
            except Exception as e:
                logger.warning(f"Error deleting collection: {e}, will try to reset it")
                
            # Re-initialize the collection - create new or reset existing
            try:
                self.collection = self.chroma_client.get_collection(VECTORSTORE_COLLECTION_NAME)
                # If we get here, collection exists, so delete all documents
                self.collection.delete(where={"$exists": {"source_document": True}})
                self.collection.delete() # Delete everything in case the above filter didn't match
                logger.info(f"Reset existing collection: {VECTORSTORE_COLLECTION_NAME}")
            except:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=VECTORSTORE_COLLECTION_NAME,
                    metadata={"description": "Document chunks with semantic embeddings"}
                )
                logger.info(f"Created new collection: {VECTORSTORE_COLLECTION_NAME}")
            
            # Clear keyword search data
            self.documents = []
            self.document_ids = []
            self.tfidf_matrix = None
            
            # Verify collection is empty
            count = self.collection.count()
            logger.info(f"Collection cleared, document count: {count}")
            
            # Hard reload to ensure we're starting fresh
            try:
                self.collection = self.chroma_client.get_collection(VECTORSTORE_COLLECTION_NAME)
            except:
                self.collection = self.chroma_client.create_collection(
                    name=VECTORSTORE_COLLECTION_NAME,
                    metadata={"description": "Document chunks with semantic embeddings"}
                )
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def _load_existing_documents(self):
        """Load existing documents from ChromaDB for keyword search"""
        try:
            count = self.collection.count()
            if count > 0:
                logger.info(f"Loading {count} existing documents from collection")
                # Get all documents from collection
                results = self.collection.get()
                
                self.documents = results['documents']
                self.document_ids = results['ids']
                
                # Rebuild keyword search index
                if self.documents:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
                    logger.info(f"Rebuilt keyword search index with {len(self.documents)} documents")
                
        except Exception as e:
            logger.error(f"Failed to load existing documents: {e}")
            # Continue with empty documents list
            self.documents = []
            self.document_ids = []
    
    def update_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for existing document"""
        try:
            # Check if document exists
            results = self.collection.get(ids=[doc_id])
            if not results or not results['ids']:
                logger.error(f"Document {doc_id} not found in collection")
                return False
            
            # Get current metadata
            current_metadata = results['metadatas'][0] if results['metadatas'] else {}
            
            # Merge with new metadata (new values take precedence)
            updated_metadata = {**current_metadata, **metadata}
            
            # Update document
            self.collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )
            
            logger.info(f"Successfully updated metadata for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document metadata: {e}")
            return False
