"""
🎯 Hybrid Search Service (RRF Fusion Engine)
===========================================
Purpose: Combine dense and sparse search using Reciprocal Rank Fusion
use RRF to optimally combine semantic and keyword search for superior accuracy"
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import time
from collections import defaultdict
import numpy as np

# Local imports
from app.core.config import settings
from app.models.schemas import QuestionResult, SearchType, SearchRequest
from app.services.vector_search import vector_search_service  
from app.services.sparse_search import sparse_search_service

logger = logging.getLogger(__name__)

class HybridSearchService:
    """
    ⚡ Advanced hybrid search combining multiple retrieval strategies
    
    Why Hybrid Search:
    - Dense (vector) search: Captures semantic similarity
    - Sparse (BM25) search: Captures exact keyword matches
    - RRF Fusion: Optimally combines both without score normalization issues
    
    Algorithm: Reciprocal Rank Fusion (RRF)
    - Proven superior to score-based fusion
    - No need to normalize different score ranges
    - Formula: RRF_score = Σ(1 / (k + rank_i)) for each search method
    
    Research: "The best fusion method for metasearch" - Cormack et al.
    """
    
    def __init__(self):
        self.rrf_k = settings.RRF_K  # RRF parameter (typically 60)
        self.dense_weight = settings.DENSE_WEIGHT
        self.sparse_weight = settings.SPARSE_WEIGHT
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'hybrid_searches': 0,
            'dense_only_searches': 0,
            'sparse_only_searches': 0,
            'total_search_time': 0,
            'avg_search_time': 0,
            'fusion_time': 0
        }
    
    async def search(self, request: SearchRequest) -> List[QuestionResult]:
        """
        🔍 Perform hybrid search with intelligent fusion
        
        Args:
            request: Complete search request with parameters
            
        Returns:
            Fused and ranked search results
            
        Process:
        1. Run dense and sparse searches in parallel
        2. Apply RRF fusion algorithm
        3. Merge metadata and scoring information
        4. Apply post-fusion filtering and ranking
        """
        start_time = time.time()
        self.stats['total_searches'] += 1
        
        logger.info(f"🔍 Hybrid search: '{request.query}' (type: {request.search_type})")
        
        try:
            if request.search_type == SearchType.DENSE:
                # Dense search only
                results = await self._dense_search_only(request)
                self.stats['dense_only_searches'] += 1
                
            elif request.search_type == SearchType.SPARSE:
                # Sparse search only
                results = await self._sparse_search_only(request)
                self.stats['sparse_only_searches'] += 1
                
            else:  # SearchType.HYBRID
                # Full hybrid search with RRF fusion
                results = await self._hybrid_search_rrf(request)
                self.stats['hybrid_searches'] += 1
            
            # Update performance stats
            search_time = time.time() - start_time
            self.stats['total_search_time'] += search_time
            self.stats['avg_search_time'] = self.stats['total_search_time'] / self.stats['total_searches']
            
            logger.info(f"✅ Hybrid search completed: {len(results)} results in {search_time*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []
    
    async def _dense_search_only(self, request: SearchRequest) -> List[QuestionResult]:
        """Execute dense search only"""
        return await vector_search_service.search(
            query=request.query,
            top_k=request.top_k,
            difficulty_filter=request.difficulty_filter
        )
    
    async def _sparse_search_only(self, request: SearchRequest) -> List[QuestionResult]:
        """Execute sparse search only"""
        return await sparse_search_service.search(
            query=request.query,
            top_k=request.top_k,
            difficulty_filter=request.difficulty_filter
        )
    
    async def _hybrid_search_rrf(self, request: SearchRequest) -> List[QuestionResult]:
        """
        ⚡ Execute hybrid search with RRF fusion
        
        RRF Algorithm:
        1. Get rankings from both search methods
        2. For each document: RRF_score = Σ(1/(k + rank)) across all methods
        3. Re-rank by RRF scores
        4. Merge document metadata
        """
        logger.debug("🔄 Running parallel dense and sparse searches...")
        
        # Run both searches in parallel for speed
        dense_task = vector_search_service.search(
            query=request.query,
            top_k=request.top_k * 2,  # Get more candidates for better fusion
            difficulty_filter=request.difficulty_filter
        )
        
        sparse_task = sparse_search_service.search(
            query=request.query,
            top_k=request.top_k * 2,  # Get more candidates for better fusion
            difficulty_filter=request.difficulty_filter
        )
        
        # Wait for both searches to complete
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        logger.debug(f"📊 Retrieved {len(dense_results)} dense + {len(sparse_results)} sparse results")
        
        # Apply RRF fusion
        fusion_start = time.time()
        fused_results = self._apply_rrf_fusion(dense_results, sparse_results, request.top_k)
        fusion_time = time.time() - fusion_start
        self.stats['fusion_time'] += fusion_time
        
        logger.debug(f"⚡ RRF fusion completed in {fusion_time*1000:.1f}ms")
        
        return fused_results
    
    def _apply_rrf_fusion(self, dense_results: List[QuestionResult], 
                         sparse_results: List[QuestionResult], 
                         top_k: int) -> List[QuestionResult]:
        """
        🧮 Apply Reciprocal Rank Fusion algorithm
        
        RRF Formula: score = Σ(1/(k + rank)) for each retrieval method
        
        Why RRF works:
        - No score normalization needed
        - Ranks matter more than raw scores
        - Proven effective in metasearch
        - Handles score distribution differences
        """
        
        # Create document registry
        doc_registry: Dict[str, QuestionResult] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)
        method_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_id = result.question_id  # Use question_id as unique identifier
            doc_registry[doc_id] = result
            
            # Calculate RRF contribution from dense search
            rrf_contribution = self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[doc_id] += rrf_contribution
            
            method_scores[doc_id]['dense_rank'] = rank + 1
            method_scores[doc_id]['dense_score'] = result.dense_score or result.relevance_score
            method_scores[doc_id]['dense_rrf'] = rrf_contribution
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result.question_id
            
            # Use sparse result if not seen, otherwise merge metadata
            if doc_id not in doc_registry:
                doc_registry[doc_id] = result
            else:
                # Merge sparse information into existing result
                doc_registry[doc_id].sparse_score = result.sparse_score
            
            # Calculate RRF contribution from sparse search
            rrf_contribution = self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[doc_id] += rrf_contribution
            
            method_scores[doc_id]['sparse_rank'] = rank + 1
            method_scores[doc_id]['sparse_score'] = result.sparse_score or result.relevance_score
            method_scores[doc_id]['sparse_rrf'] = rrf_contribution
        
        # Create final results with RRF scores
        final_results = []
        for doc_id, rrf_score in rrf_scores.items():
            result = doc_registry[doc_id]
            
            # Update result with fusion information
            result.rrf_score = float(rrf_score)
            result.relevance_score = float(rrf_score)  # Use RRF as primary score
            result.search_type = "hybrid"
            
            # Add method-specific information if available
            if doc_id in method_scores:
                scores = method_scores[doc_id]
                result.dense_score = scores.get('dense_score')
                result.sparse_score = scores.get('sparse_score')
            
            final_results.append(result)
        
        # Sort by RRF score (descending)
        final_results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        # Return top-k results
        return final_results[:top_k]
    
    def _boost_quality_scores(self, results: List[QuestionResult]) -> List[QuestionResult]:
        """
        📈 Apply quality-based score boosting
        
        Boosts:
        - High-quality explanations get slight boost
        - Questions with multiple choice options get boost
        - Balanced to not overwhelm relevance
        """
        for result in results:
            original_score = result.relevance_score
            boost = 0.0
            
            # Quality boost (up to 5%)
            if result.quality_score > 0.8:
                boost += 0.05
            elif result.quality_score > 0.5:
                boost += 0.02
            
            # Explanation boost (up to 3%)
            if result.has_explanation:
                boost += 0.03
            
            # Options boost (up to 2%)
            if result.options and len(result.options) >= 3:
                boost += 0.02
            
            # Apply boost (capped at 10% total)
            result.relevance_score = min(original_score * (1 + min(boost, 0.1)), 1.0)
        
        return results
    
    async def search_similar(self, doc_id: str, top_k: int = 5) -> List[QuestionResult]:
        """
        🔗 Find similar documents using hybrid approach
        
        Use case: "More like this" functionality
        """
        try:
            # Get similar documents from vector search
            similar_results = vector_search_service.get_similar_documents(doc_id, top_k * 2)
            
            if not similar_results:
                return []
            
            # Extract key terms from the original document for sparse search
            original_doc = None
            for result in similar_results:
                if result.id == doc_id or result.question_id == doc_id:
                    original_doc = result
                    break
            
            if original_doc:
                # Use original document's question as query for sparse search
                query = original_doc.question
                sparse_results = await sparse_search_service.search(query, top_k)
                
                # Simple fusion for similar documents (favor vector similarity)
                combined = {}
                
                # Add vector results with higher weight
                for i, result in enumerate(similar_results):
                    if result.id != doc_id and result.question_id != doc_id:
                        combined[result.question_id] = result
                        result.relevance_score *= 0.8  # Slightly reduce vector scores
                
                # Add sparse results with lower weight
                for result in sparse_results:
                    if result.question_id not in combined:
                        result.relevance_score *= 0.6  # Reduce sparse scores more
                        combined[result.question_id] = result
                
                # Sort and return
                final_results = list(combined.values())
                final_results.sort(key=lambda x: x.relevance_score, reverse=True)
                
                return final_results[:top_k]
            
            return similar_results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Similar document search failed: {e}")
            return []
    
    def explain_ranking(self, query: str, doc_id: str) -> Dict[str, Any]:
        """
        📝 Explain why a document was ranked at its position
        
        Useful for debugging and understanding search results
        """
        try:
            explanation = {
                "document_id": doc_id,
                "query": query,
                "rrf_parameters": {
                    "k": self.rrf_k,
                    "dense_weight": self.dense_weight,
                    "sparse_weight": self.sparse_weight
                },
                "component_scores": {}
            }
            
            # This would require running individual searches and tracking scores
            # Implementation depends on specific debugging needs
            
            return explanation
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'rrf_parameters': {
                'k': self.rrf_k,
                'dense_weight': self.dense_weight,
                'sparse_weight': self.sparse_weight
            },
            'avg_fusion_time_ms': (self.stats['fusion_time'] / max(1, self.stats['hybrid_searches'])) * 1000,
            'search_type_distribution': {
                'hybrid': self.stats['hybrid_searches'],
                'dense_only': self.stats['dense_only_searches'],
                'sparse_only': self.stats['sparse_only_searches']
            }
        }

# Global service instance
hybrid_search_service = HybridSearchService()
