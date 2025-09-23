"""
🎯 Sparse Search Service (BM25 Keyword Search)
=============================================
Purpose: Exact keyword matching using BM25 algorithm with Bengali tokenization
 use BM25 for exact keyword matches that vector search might miss"
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import re
from rank_bm25 import BM25Okapi
from collections import Counter

# Local imports  
from app.core.config import settings
from app.models.schemas import QuestionResult, SearchType
from app.services.text_processor import text_processor

logger = logging.getLogger(__name__)

class SparseSearchService:
    """
    🔎 BM25-based sparse retrieval for exact keyword matching
    
    Why BM25:
    - Captures exact keyword importance (TF-IDF improved)
    - Handles document length normalization
    - Proven effective for keyword-based search
    - Complements vector search perfectly
    
    Example: Query "গীতাঞ্জলি" will strongly match documents containing exact word "গীতাঞ্জলি"
    """
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.is_loaded = False
        
        # BM25 parameters (tuned for Bengali literature)
        self.k1 = settings.BM25_K1  # Term frequency saturation (1.2)
        self.b = settings.BM25_B    # Length normalization (0.75)
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'total_search_time': 0,
            'avg_search_time': 0,
            'total_documents': 0,
            'avg_doc_length': 0
        }
        
        # Query cache for performance
        self.query_cache: Dict[str, List[Tuple[int, float]]] = {}
        self.cache_max_size = 50
    
    async def initialize(self) -> None:
        """
        🚀 Load documents and build BM25 index
        
        Process:
        1. Load processed documents from embeddings metadata
        2. Tokenize all documents using Bengali tokenizer  
        3. Build BM25 index for fast retrieval
        4. Calculate index statistics
        """
        if self.is_loaded:
            return
        
        logger.info("🔧 Initializing Sparse Search Service (BM25)...")
        
        try:
            # Load documents from embeddings metadata
            embeddings_path = settings.get_absolute_path(settings.EMBEDDINGS_PATH)
            
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['metadata']
            texts = data['texts']
            
            logger.info(f"📖 Loaded {len(self.documents)} documents for BM25 indexing")
            
            # Tokenize all documents for BM25
            logger.info("✂️ Tokenizing documents for BM25...")
            self.tokenized_docs = []
            
            for i, text in enumerate(texts):
                # Use our Bengali tokenizer
                tokens = text_processor.tokenize_bengali(text, remove_stopwords=True)
                self.tokenized_docs.append(tokens)
                
                if i % 100 == 0:
                    logger.debug(f"   Tokenized {i+1}/{len(texts)} documents")
            
            # Build BM25 index
            logger.info("🏗️ Building BM25 index...")
            self.bm25 = BM25Okapi(
                self.tokenized_docs,
                k1=self.k1,
                b=self.b
            )
            
            # Calculate statistics
            self.stats['total_documents'] = len(self.documents)
            doc_lengths = [len(doc) for doc in self.tokenized_docs]
            self.stats['avg_doc_length'] = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
            
            logger.info("✅ Sparse Search Service initialized successfully")
            logger.info(f"📊 Indexed {self.stats['total_documents']} documents")
            logger.info(f"📏 Average document length: {self.stats['avg_doc_length']:.1f} tokens")
            logger.info(f"⚙️ BM25 parameters: k1={self.k1}, b={self.b}")
            
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Sparse Search Service: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 5,
                    difficulty_filter: Optional[List[int]] = None) -> List[QuestionResult]:
        """
        🔍 Perform BM25 keyword search
        
        Args:
            query: Search query
            top_k: Number of results to return
            difficulty_filter: Filter by difficulty levels
            
        Returns:
            List of ranked search results
            
        Algorithm:
        1. Tokenize query using same Bengali tokenizer
        2. Compute BM25 scores for all documents
        3. Rank by BM25 scores
        4. Apply filters
        5. Return top-k results
        """
        if not self.is_loaded:
            await self.initialize()
        
        start_time = time.time()
        self.stats['total_searches'] += 1
        
        logger.debug(f"🔍 BM25 search for: {query[:50]}...")
        
        try:
            # Check cache first
            cache_key = f"{query.strip().lower()}_{top_k}_{difficulty_filter}"
            if cache_key in self.query_cache:
                scored_docs = self.query_cache[cache_key]
            else:
                # Tokenize query
                query_tokens = text_processor.tokenize_bengali(query, remove_stopwords=True)
                
                if not query_tokens:
                    logger.warning("No valid tokens found in query")
                    return []
                
                logger.debug(f"🔤 Query tokens: {query_tokens}")
                
                # Get BM25 scores
                scores = self.bm25.get_scores(query_tokens)
                
                # Create (index, score) pairs and sort by score
                scored_docs = [(i, float(score)) for i, score in enumerate(scores)]
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                # Cache results
                if len(self.query_cache) < self.cache_max_size:
                    self.query_cache[cache_key] = scored_docs
            
            # Convert to results with filtering
            results = []
            for doc_idx, score in scored_docs[:top_k * 3]:  # Get more for filtering
                
                if score <= 0.01:  # Skip very low scores
                    continue
                
                metadata = self.documents[doc_idx]
                
                # Apply difficulty filter
                if difficulty_filter and metadata.get('difficulty', 1) not in difficulty_filter:
                    continue
                
                result = QuestionResult(
                    id=f"bm25_{doc_idx}",
                    question_id=str(metadata.get('row_id', doc_idx)),
                    question=metadata.get('question', ''),
                    explanation=metadata.get('explanation', ''),
                    options={},
                    correct_answer=metadata.get('answer', ''),
                    difficulty=metadata.get('difficulty', 1),
                    relevance_score=float(min(score / 10.0, 1.0)),  # Normalize BM25 score
                    sparse_score=float(score),
                    text_length=metadata.get('text_length', 0),
                    has_explanation=metadata.get('has_explanation', False),
                    quality_score=metadata.get('quality_score', 0.0),
                    search_type="sparse"
                )
                
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            # Update performance stats
            search_time = time.time() - start_time
            self.stats['total_search_time'] += search_time
            self.stats['avg_search_time'] = self.stats['total_search_time'] / self.stats['total_searches']
            
            logger.debug(f"✅ BM25 search completed: {len(results)} results in {search_time*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ BM25 search failed: {e}")
            return []
    
    def get_term_frequencies(self, query: str) -> Dict[str, List[Tuple[str, int]]]:
        """
        📊 Get term frequencies for query analysis
        
        Useful for:
        - Understanding why certain documents rank higher
        - Query expansion suggestions
        - Debugging search results
        """
        if not self.is_loaded:
            return {}
        
        query_tokens = text_processor.tokenize_bengali(query, remove_stopwords=True)
        
        result = {}
        for token in query_tokens:
            # Find documents containing this token
            doc_freqs = []
            for i, doc_tokens in enumerate(self.tokenized_docs):
                freq = doc_tokens.count(token)
                if freq > 0:
                    question = self.documents[i].get('question', '')[:50] + '...'
                    doc_freqs.append((question, freq))
            
            # Sort by frequency
            doc_freqs.sort(key=lambda x: x[1], reverse=True)
            result[token] = doc_freqs[:10]  # Top 10
        
        return result
    
    def explain_score(self, query: str, doc_id: str) -> Dict[str, Any]:
        """
        📝 Explain BM25 score calculation for a specific document
        
        Useful for debugging and understanding ranking
        """
        if not self.is_loaded:
            return {}
        
        try:
            # Find document index
            doc_idx = None
            for i, metadata in enumerate(self.documents):
                if str(metadata.get('row_id')) == doc_id or f"bm25_{i}" == doc_id:
                    doc_idx = i
                    break
            
            if doc_idx is None:
                return {"error": "Document not found"}
            
            query_tokens = text_processor.tokenize_bengali(query, remove_stopwords=True)
            doc_tokens = self.tokenized_docs[doc_idx]
            
            explanation = {
                "document_id": doc_id,
                "query_tokens": query_tokens,
                "document_length": len(doc_tokens),
                "term_scores": {},
                "total_score": 0
            }
            
            # Calculate individual term contributions
            for token in query_tokens:
                tf = doc_tokens.count(token)  # Term frequency
                df = sum(1 for doc in self.tokenized_docs if token in doc)  # Document frequency
                idf = self.bm25.idf.get(token, 0)  # Inverse document frequency
                
                explanation["term_scores"][token] = {
                    "term_frequency": tf,
                    "document_frequency": df,
                    "inverse_document_frequency": idf,
                    "present": tf > 0
                }
            
            # Get total BM25 score
            scores = self.bm25.get_scores(query_tokens)
            explanation["total_score"] = float(scores[doc_idx])
            
            return explanation
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance and index statistics"""
        vocab_size = len(self.bm25.idf) if self.bm25 else 0
        
        return {
            **self.stats,
            'vocabulary_size': vocab_size,
            'cache_size': len(self.query_cache),
            'bm25_parameters': {'k1': self.k1, 'b': self.b},
            'is_loaded': self.is_loaded
        }
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("🗑️ BM25 search cache cleared")

# Global service instance
sparse_search_service = SparseSearchService()
