"""Evaluation framework for measuring system performance and quality"""

import logging
import json
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Comprehensive evaluation using custom metrics inspired by RAGAS framework"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        # Initialize with CPU device to avoid MPS issues
        import torch
        device = 'cpu'  # Force CPU for stability
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.metrics = {
            'answer_relevancy': self._calculate_answer_relevancy,
            'faithfulness': self._calculate_faithfulness,
            'context_precision': self._calculate_context_precision,
            'context_recall': self._calculate_context_recall,
            'response_time': self._calculate_response_time,
            'source_attribution_accuracy': self._calculate_source_attribution
        }
    
    def evaluate_response(self, query: str, response: Dict[str, Any], 
                         context_documents: List[Dict[str, Any]], 
                         ground_truth: Optional[str] = None,
                         response_time: Optional[float] = None) -> Dict[str, float]:
        """Evaluate a single response across multiple metrics"""
        
        evaluation_results = {}
        
        for metric_name, metric_func in self.metrics.items():
            try:
                if metric_name == 'response_time':
                    score = metric_func(response_time) if response_time else 0.0
                elif metric_name == 'context_recall':
                    score = metric_func(query, response, context_documents, ground_truth) if ground_truth else 0.0
                else:
                    score = metric_func(query, response, context_documents)
                
                evaluation_results[metric_name] = score
                
            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {e}")
                evaluation_results[metric_name] = 0.0
        
        return evaluation_results
    
    def _calculate_answer_relevancy(self, query: str, response: Dict[str, Any], 
                                  context_documents: List[Dict[str, Any]]) -> float:
        """Calculate how relevant the answer is to the query"""
        answer = response.get('answer', '')
        
        if not answer or not query:
            return 0.0
        
        try:
            # Generate embeddings
            query_embedding = self.embedding_model.encode([query])
            answer_embedding = self.embedding_model.encode([answer])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, answer_embedding)[0][0]
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error calculating answer relevancy: {e}")
            return 0.0
    
    def _calculate_faithfulness(self, query: str, response: Dict[str, Any], 
                              context_documents: List[Dict[str, Any]]) -> float:
        """Calculate how faithful the answer is to the provided context"""
        answer = response.get('answer', '')
        
        if not answer or not context_documents:
            return 0.0
        
        try:
            # Combine all context documents
            context_text = " ".join([doc.get('content', '') for doc in context_documents])
            
            if not context_text:
                return 0.0
            
            # Generate embeddings
            answer_embedding = self.embedding_model.encode([answer])
            context_embedding = self.embedding_model.encode([context_text])
            
            # Calculate similarity
            similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
            
            # Check for direct quotes or citations (bonus for faithfulness)
            citation_bonus = 0.1 if '[Source:' in answer or 'according to' in answer.lower() else 0.0
            
            return max(0.0, min(1.0, (similarity + 1) / 2 + citation_bonus))
            
        except Exception as e:
            logger.error(f"Error calculating faithfulness: {e}")
            return 0.0
    
    def _calculate_context_precision(self, query: str, response: Dict[str, Any], 
                                   context_documents: List[Dict[str, Any]]) -> float:
        """Calculate precision of retrieved context"""
        if not context_documents:
            return 0.0
        
        try:
            answer = response.get('answer', '')
            
            # Calculate how many retrieved documents are actually relevant
            relevant_docs = 0
            
            for doc in context_documents:
                doc_content = doc.get('content', '')
                
                # Check if document content appears to be used in the answer
                # Simple heuristic: check for keyword overlap
                doc_words = set(doc_content.lower().split())
                answer_words = set(answer.lower().split())
                
                overlap = len(doc_words.intersection(answer_words))
                
                if overlap > min(10, len(doc_words) * 0.1):  # At least 10 words or 10% overlap
                    relevant_docs += 1
            
            precision = relevant_docs / len(context_documents)
            return precision
            
        except Exception as e:
            logger.error(f"Error calculating context precision: {e}")
            return 0.0
    
    def _calculate_context_recall(self, query: str, response: Dict[str, Any], 
                                context_documents: List[Dict[str, Any]], 
                                ground_truth: str) -> float:
        """Calculate recall of retrieved context against ground truth"""
        if not ground_truth or not context_documents:
            return 0.0
        
        try:
            # Combine all context documents
            context_text = " ".join([doc.get('content', '') for doc in context_documents])
            
            # Generate embeddings
            ground_truth_embedding = self.embedding_model.encode([ground_truth])
            context_embedding = self.embedding_model.encode([context_text])
            
            # Calculate similarity
            similarity = cosine_similarity(ground_truth_embedding, context_embedding)[0][0]
            
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error calculating context recall: {e}")
            return 0.0
    
    def _calculate_response_time(self, response_time: Optional[float]) -> float:
        """Calculate response time score (lower is better, normalized to 0-1)"""
        if response_time is None:
            return 0.0
        
        # Normalize response time: excellent < 2s, good < 5s, poor > 10s
        if response_time <= 2.0:
            return 1.0
        elif response_time <= 5.0:
            return 1.0 - (response_time - 2.0) / 3.0  # Linear decrease from 1.0 to 0.0
        elif response_time <= 10.0:
            return 0.5 - (response_time - 5.0) / 10.0  # Linear decrease from 0.5 to 0.0
        else:
            return 0.0
    
    def _calculate_source_attribution(self, query: str, response: Dict[str, Any], 
                                    context_documents: List[Dict[str, Any]]) -> float:
        """Calculate accuracy of source attribution"""
        answer = response.get('answer', '')
        sources = response.get('sources', [])
        
        if not sources:
            # Check if answer mentions sources but doesn't list them
            if '[Source:' in answer or 'according to' in answer.lower():
                return 0.5  # Partial credit for attempting attribution
            return 0.0
        
        # Check if all mentioned sources exist in context documents
        context_sources = set()
        for doc in context_documents:
            source_doc = doc.get('metadata', {}).get('source_document', '')
            if source_doc:
                context_sources.add(source_doc)
        
        correct_attributions = 0
        for source in sources:
            source_doc = source.get('document', '')
            if source_doc in context_sources:
                correct_attributions += 1
        
        return correct_attributions / len(sources) if sources else 0.0
    
    def generate_evaluation_report(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not evaluations:
            return {'error': 'No evaluations provided'}
        
        # Calculate average scores for each metric
        metrics_summary = {}
        for metric in self.metrics.keys():
            scores = [eval_result.get(metric, 0.0) for eval_result in evaluations]
            metrics_summary[metric] = {
                'average': np.mean(scores),
                'std_dev': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
        
        # Calculate overall performance score
        key_metrics = ['answer_relevancy', 'faithfulness', 'context_precision']
        overall_score = np.mean([metrics_summary[metric]['average'] for metric in key_metrics])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_summary)
        
        return {
            'overall_score': overall_score,
            'metrics_summary': metrics_summary,
            'total_evaluations': len(evaluations),
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, metrics_summary: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        # Answer relevancy
        if metrics_summary.get('answer_relevancy', {}).get('average', 0) < 0.7:
            recommendations.append("Consider improving query understanding and response generation")
        
        # Faithfulness
        if metrics_summary.get('faithfulness', {}).get('average', 0) < 0.8:
            recommendations.append("Focus on grounding responses more closely to source documents")
        
        # Context precision
        if metrics_summary.get('context_precision', {}).get('average', 0) < 0.6:
            recommendations.append("Improve document retrieval to fetch more relevant context")
        
        # Response time
        if metrics_summary.get('response_time', {}).get('average', 0) < 0.7:
            recommendations.append("Optimize system performance to reduce response times")
        
        return recommendations

class FeedbackProcessor:
    """Process and store user feedback for continuous improvement"""
    
    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = db_path
        self.retraining_threshold = 50
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    rating INTEGER,
                    corrections TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    user_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
    
    def process_user_feedback(self, query: str, response: str, rating: int, 
                            corrections: str = "", session_id: str = "", user_id: str = ""):
        """Process and store user feedback"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO feedback (query, response, rating, corrections, session_id, user_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (query, response, rating, corrections, session_id, user_id))
            
            logger.info(f"Stored user feedback: rating={rating}")
            
            # Check if retraining threshold is reached
            feedback_count = self._get_feedback_count()
            if feedback_count >= self.retraining_threshold:
                logger.info("Retraining threshold reached - analysis recommended")
                
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
    
    def store_system_metrics(self, metrics: Dict[str, float], metadata: Dict[str, Any] = None):
        """Store system performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for metric_name, metric_value in metrics.items():
                    conn.execute("""
                        INSERT INTO system_metrics (metric_name, metric_value, metadata)
                        VALUES (?, ?, ?)
                    """, (metric_name, metric_value, json.dumps(metadata) if metadata else None))
            
            logger.info(f"Stored {len(metrics)} system metrics")
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def _get_feedback_count(self) -> int:
        """Get total feedback count"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM feedback")
                return cursor.fetchone()[0]
        except Exception:
            return 0
    
    def get_feedback_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Analyze recent feedback for insights"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent feedback
                cursor = conn.execute("""
                    SELECT rating, corrections, timestamp 
                    FROM feedback 
                    WHERE datetime(timestamp) >= datetime('now', '-{} days')
                """.format(days))
                
                feedback_data = cursor.fetchall()
                
                if not feedback_data:
                    return {'message': 'No recent feedback available'}
                
                # Calculate statistics
                ratings = [row[0] for row in feedback_data if row[0] is not None]
                avg_rating = np.mean(ratings) if ratings else 0
                
                # Count feedback with corrections
                corrections_count = sum(1 for row in feedback_data if row[1] and row[1].strip())
                
                return {
                    'total_feedback': len(feedback_data),
                    'average_rating': avg_rating,
                    'corrections_provided': corrections_count,
                    'correction_rate': corrections_count / len(feedback_data) if feedback_data else 0,
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}")
            return {'error': str(e)}
