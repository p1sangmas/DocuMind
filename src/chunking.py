"""Semantic text chunking module for intelligent document segmentation"""

import re
import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import EMBEDDING_MODEL, MAX_CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class SemanticChunker:
    """Semantic-aware text chunking using sentence embeddings"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        # Initialize with CPU device to avoid MPS issues
        import torch
        device = 'cpu'  # Force CPU for stability
        self.model = SentenceTransformer(model_name, device=device)
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.max_chunk_size = MAX_CHUNK_SIZE
        self.overlap_size = CHUNK_OVERLAP
        
    def chunk_by_semantic_similarity(self, text: str) -> List[str]:
        """Group sentences by semantic similarity"""
        logger.info("Starting semantic chunking process")
        
        # Clean and split text into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text]
            
        logger.info(f"Split text into {len(sentences)} sentences")
        
        # Generate embeddings for all sentences
        try:
            embeddings = self.model.encode(sentences, show_progress_bar=False)
            logger.info("Generated sentence embeddings")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return self._fallback_chunking(text)
        
        # Group sentences into chunks based on semantic similarity
        chunks = self._group_sentences_by_similarity(sentences, embeddings)
        
        # Ensure chunks don't exceed max size and add overlap
        final_chunks = self._post_process_chunks(chunks)
        
        logger.info(f"Created {len(final_chunks)} semantic chunks")
        return final_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling"""
        # Clean text
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out very short sentences and clean
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum sentence length
                cleaned_sentences.append(sentence)
                
        return cleaned_sentences
    
    def _group_sentences_by_similarity(self, sentences: List[str], embeddings: np.ndarray) -> List[str]:
        """Group sentences into chunks based on semantic similarity"""
        if len(sentences) == 0:
            return []
            
        chunks = []
        current_chunk_sentences = []
        current_chunk_embeddings = []
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if not current_chunk_sentences:
                # Start new chunk
                current_chunk_sentences.append(sentence)
                current_chunk_embeddings.append(embedding)
                continue
            
            # Calculate similarity with current chunk centroid
            chunk_centroid = np.mean(current_chunk_embeddings, axis=0).reshape(1, -1)
            sentence_embedding = embedding.reshape(1, -1)
            similarity = cosine_similarity(sentence_embedding, chunk_centroid)[0][0]
            
            # Check if sentence fits in current chunk
            potential_chunk = ' '.join(current_chunk_sentences + [sentence])
            
            if (similarity > self.similarity_threshold and 
                len(potential_chunk) <= self.max_chunk_size):
                # Add to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_embeddings.append(embedding)
            else:
                # Finalize current chunk and start new one
                if current_chunk_sentences:
                    chunks.append(' '.join(current_chunk_sentences))
                
                current_chunk_sentences = [sentence]
                current_chunk_embeddings = [embedding]
        
        # Add the last chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
            
        return chunks
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """Post-process chunks to ensure size limits and add overlap"""
        final_chunks = []
        
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split oversized chunks
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
        
        # Add overlap between consecutive chunks
        overlapped_chunks = self._add_overlap(final_chunks)
        
        return overlapped_chunks
    
    def _split_large_chunk(self, chunk: str) -> List[str]:
        """Split chunks that exceed max size"""
        words = chunk.split()
        sub_chunks = []
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length <= self.max_chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    sub_chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
        
        if current_chunk:
            sub_chunks.append(' '.join(current_chunk))
            
        return sub_chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks"""
        if len(chunks) <= 1:
            return chunks
            
        overlapped_chunks = [chunks[0]]  # First chunk as-is
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i-1]
            
            # Extract overlap from previous chunk
            previous_words = previous_chunk.split()
            overlap_words = previous_words[-self.overlap_size//10:]  # Approximate word count for overlap
            
            if overlap_words:
                overlap_text = ' '.join(overlap_words)
                overlapped_chunk = overlap_text + ' ' + current_chunk
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(current_chunk)
                
        return overlapped_chunks
    
    def _fallback_chunking(self, text: str) -> List[str]:
        """Fallback to simple chunking if semantic chunking fails"""
        logger.warning("Falling back to simple chunking")
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            
            if current_length + word_length <= self.max_chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
