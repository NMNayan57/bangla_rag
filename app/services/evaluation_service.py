"""
🎯 Evaluation Metrics Service
=============================
Purpose: Comprehensive evaluation framework for RAG system performance
implemented Hit@K and MRR metrics for rigorous performance evaluation"
"""

import asyncio
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Local imports
from app.core.config import settings
from app.models.schemas import SearchRequest, SearchType, QuestionResult
from app.services.hybrid_search import hybrid_search_service

logger = logging.getLogger(__name__)

@dataclass
class EvaluationQuery:
    """Single evaluation query with ground truth"""
    query: str
    relevant_doc_ids: List[str]  # Ground truth relevant documents
    difficulty: int
    category: str

@dataclass
class EvaluationResult:
    """Results for a single query evaluation"""
    query: str
    retrieved_doc_ids: List[str]
    relevant_doc_ids: List[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    mrr_score: float
    response_time_ms: float
    search_type: str
    ai_enhanced: bool

class EvaluationService:
    """
    📊 Comprehensive evaluation service for RAG system
    
    Metrics Implemented:
    - Hit@K: Percentage of queries with at least one relevant result in top-K
    - MRR (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant result
    - Response Time: Performance benchmarking
    - A/B Testing: With/Without AI enhancement comparison
    
    Why these metrics:
    - Hit@K: Measures recall at different cutoffs
    - MRR: Measures ranking quality (higher is better)
    - Industry standard for search evaluation
    """
    
    def __init__(self):
        self.evaluation_queries: List[EvaluationQuery] = []
        self.results_cache: Dict[str, List[EvaluationResult]] = {}
        
        # Performance tracking
        self.stats = {
            'total_evaluations': 0,
            'avg_hit_at_1': 0.0,
            'avg_hit_at_3': 0.0,
            'avg_hit_at_5': 0.0,
            'avg_mrr': 0.0,
            'avg_response_time': 0.0
        }
    
    def load_evaluation_dataset(self) -> None:
        """
        📋 Load or create evaluation queries from the dataset
        
        Strategy:
        - Sample representative queries from different categories
        - Create ground truth based on question similarity
        - Include varied difficulty levels
        """
        
        # Sample evaluation queries (would normally load from file)
        self.evaluation_queries = [
            EvaluationQuery(
                query="রবীন্দ্রনাথ গীতাঞ্জলি",
                relevant_doc_ids=["305", "470", "155"],  # Questions about Gitanjali
                difficulty=1,
                category="rabindranath_poetry"
            ),
            EvaluationQuery(
                query="নোবেল পুরস্কার",
                relevant_doc_ids=["305", "470"],  # Nobel Prize related
                difficulty=2,
                category="awards"
            ),
            EvaluationQuery(
                query="কাব্যগ্রন্থ",
                relevant_doc_ids=["305", "209", "448"],  # Poetry books
                difficulty=1,
                category="literature_types"
            ),
            EvaluationQuery(
                query="বিজ্ঞান বিষয়ক গ্রন্থ",
                relevant_doc_ids=["222"],  # Science books
                difficulty=2,
                category="science_literature"
            ),
            EvaluationQuery(
                query="চলিত ভাষা",
                relevant_doc_ids=["221"],  # Colloquial language
                difficulty=3,
                category="language_style"
            ),
            EvaluationQuery(
                query="গদ্য ছন্দ",
                relevant_doc_ids=["209"],  # Prose rhythm
                difficulty=3,
                category="literary_techniques"
            ),
            EvaluationQuery(
                query="প্রথম প্রকাশিত গ্রন্থ",
                relevant_doc_ids=["448"],  # First published book
                difficulty=2,
                category="literary_history"
            ),
            EvaluationQuery(
                query="আত্মজীবনী",
                relevant_doc_ids=["305", "222"],  # Autobiography
                difficulty=1,
                category="biography"
            )
        ]
        
        logger.info(f"📋 Loaded {len(self.evaluation_queries)} evaluation queries")
    
    def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick evaluation stats without running full tests"""
        return {
            "available_tests": {
                "quick_test": "Sample 3 queries only (30 seconds)",
                "single_evaluation": "Test one search method (1-2 minutes)",
                "ab_test": "Compare all methods (3-5 minutes)"
            },
            "test_queries_loaded": len(self.evaluation_queries) if self.evaluation_queries else 8,
            "evaluation_categories": [
                "rabindranath_poetry", "awards", "literature_types", 
                "science_literature", "language_style", "literary_techniques",
                "literary_history", "biography"
            ],
            "metrics_calculated": ["Hit@1", "Hit@3", "Hit@5", "MRR", "Response Time"],
            "user_guidance": {
                "for_end_users": "Just use the search interface - evaluation is for developers",
                "for_developers": "Use quick test for changes, full evaluation for comprehensive analysis",
                "for_interviews": "Shows systematic ML evaluation approach"
            }
        }

    async def quick_evaluation(self, search_type: SearchType, use_ai: bool = False) -> Dict[str, Any]:
        """Quick evaluation with just 3 queries for fast testing"""
        if not self.evaluation_queries:
            self.load_evaluation_dataset()
        
        logger.info(f"⚡ Running quick evaluation (3 queries): {search_type.value}, AI: {use_ai}")
        
        # Use only first 3 queries for quick test
        quick_queries = self.evaluation_queries[:3]
        original_queries = self.evaluation_queries
        self.evaluation_queries = quick_queries
        
        try:
            results = await self.evaluate_search_method(search_type, use_ai, top_k=3)
            metrics = self.calculate_aggregate_metrics(results)
            return {
                "test_type": "quick_evaluation",
                "queries_tested": len(quick_queries),
                "search_type": search_type.value,
                "ai_enhanced": use_ai,
                "metrics": metrics,
                "individual_results": [
                    {
                        "query": r.query,
                        "hit_at_1": r.hit_at_1,
                        "hit_at_3": r.hit_at_3,
                        "mrr_score": r.mrr_score,
                        "response_time_ms": r.response_time_ms
                    }
                    for r in results[:3]  # Show details for quick test
                ],
                "note": "Quick test with 3 queries only. Use full evaluation for comprehensive results."
            }
        finally:
            # Restore original queries
            self.evaluation_queries = original_queries
    
    async def evaluate_search_method(self, search_type: SearchType, use_ai: bool = False, 
                                   top_k: int = 5) -> List[EvaluationResult]:
        """
        🧪 Evaluate a specific search method
        
        Args:
            search_type: Type of search to evaluate
            use_ai: Whether to use AI enhancement
            top_k: Number of results to retrieve
            
        Returns:
            List of evaluation results for each query
        """
        
        if not self.evaluation_queries:
            self.load_evaluation_dataset()
        
        logger.info(f"🧪 Evaluating {search_type.value} search (AI: {use_ai}, top_k: {top_k})")
        
        results = []
        
        for eval_query in self.evaluation_queries:
            try:
                # Create search request
                request = SearchRequest(
                    query=eval_query.query,
                    top_k=top_k,
                    search_type=search_type,
                    use_llm=use_ai
                )
                
                # Perform search with timing
                start_time = time.time()
                search_results = await hybrid_search_service.search(request)
                response_time_ms = (time.time() - start_time) * 1000
                
                # Extract document IDs from results
                retrieved_doc_ids = [result.question_id for result in search_results]
                
                # Calculate metrics
                hit_at_1 = self._calculate_hit_at_k(retrieved_doc_ids, eval_query.relevant_doc_ids, 1)
                hit_at_3 = self._calculate_hit_at_k(retrieved_doc_ids, eval_query.relevant_doc_ids, 3)
                hit_at_5 = self._calculate_hit_at_k(retrieved_doc_ids, eval_query.relevant_doc_ids, 5)
                mrr_score = self._calculate_mrr(retrieved_doc_ids, eval_query.relevant_doc_ids)
                
                # Create evaluation result
                result = EvaluationResult(
                    query=eval_query.query,
                    retrieved_doc_ids=retrieved_doc_ids,
                    relevant_doc_ids=eval_query.relevant_doc_ids,
                    hit_at_1=hit_at_1,
                    hit_at_3=hit_at_3,
                    hit_at_5=hit_at_5,
                    mrr_score=mrr_score,
                    response_time_ms=response_time_ms,
                    search_type=search_type.value,
                    ai_enhanced=use_ai
                )
                
                results.append(result)
                
                logger.debug(f"✅ Query: {eval_query.query[:30]}... | Hit@1: {hit_at_1} | MRR: {mrr_score:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Evaluation failed for query '{eval_query.query}': {e}")
        
        self.stats['total_evaluations'] += len(results)
        logger.info(f"✅ Completed evaluation: {len(results)} queries processed")
        
        return results
    
    def _calculate_hit_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> bool:
        """
        🎯 Calculate Hit@K metric
        
        Hit@K = 1 if any relevant document appears in top-K results, else 0
        """
        top_k_retrieved = retrieved[:k]
        return any(doc_id in relevant for doc_id in top_k_retrieved)
    
    def _calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        📊 Calculate Mean Reciprocal Rank
        
        MRR = 1/rank of first relevant document (0 if no relevant docs found)
        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)  # Rank is 1-indexed
        return 0.0
    
    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        📈 Calculate aggregate metrics from evaluation results
        """
        if not results:
            return {}
        
        total_queries = len(results)
        
        # Calculate averages
        hit_at_1_avg = sum(r.hit_at_1 for r in results) / total_queries
        hit_at_3_avg = sum(r.hit_at_3 for r in results) / total_queries
        hit_at_5_avg = sum(r.hit_at_5 for r in results) / total_queries
        mrr_avg = sum(r.mrr_score for r in results) / total_queries
        response_time_avg = sum(r.response_time_ms for r in results) / total_queries
        
        return {
            'total_queries': total_queries,
            'hit_at_1': {
                'score': hit_at_1_avg,
                'percentage': f"{hit_at_1_avg * 100:.1f}%"
            },
            'hit_at_3': {
                'score': hit_at_3_avg,
                'percentage': f"{hit_at_3_avg * 100:.1f}%"
            },
            'hit_at_5': {
                'score': hit_at_5_avg,
                'percentage': f"{hit_at_5_avg * 100:.1f}%"
            },
            'mrr': {
                'score': mrr_avg,
                'interpretation': self._interpret_mrr(mrr_avg)
            },
            'response_time': {
                'avg_ms': response_time_avg,
                'avg_seconds': response_time_avg / 1000
            }
        }
    
    def _interpret_mrr(self, mrr: float) -> str:
        """Interpret MRR score"""
        if mrr >= 0.8:
            return "Excellent"
        elif mrr >= 0.6:
            return "Good"
        elif mrr >= 0.4:
            return "Fair"
        elif mrr >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    async def run_ab_test(self) -> Dict[str, Any]:
        """
        🔬 Run A/B test comparing different search configurations
        
        Tests:
        - Hybrid vs Dense vs Sparse
        - With AI vs Without AI
        - Different top-K values
        """
        
        logger.info("🔬 Starting comprehensive A/B testing...")
        
        ab_results = {}
        
        # Test 1: Search Method Comparison
        search_methods = [SearchType.HYBRID, SearchType.DENSE, SearchType.SPARSE]
        
        for search_type in search_methods:
            method_name = search_type.value
            
            # Without AI
            results_no_ai = await self.evaluate_search_method(search_type, use_ai=False)
            ab_results[f"{method_name}_no_ai"] = {
                'results': results_no_ai,
                'metrics': self.calculate_aggregate_metrics(results_no_ai)
            }
            
            # With AI (only for hybrid to save time/cost)
            if search_type == SearchType.HYBRID:
                results_with_ai = await self.evaluate_search_method(search_type, use_ai=True)
                ab_results[f"{method_name}_with_ai"] = {
                    'results': results_with_ai,
                    'metrics': self.calculate_aggregate_metrics(results_with_ai)
                }
        
        # Generate comparison summary
        comparison = self._generate_comparison_summary(ab_results)
        
        logger.info("✅ A/B testing completed")
        
        return {
            'test_timestamp': datetime.now().isoformat(),
            'detailed_results': ab_results,
            'comparison_summary': comparison,
            'recommendations': self._generate_recommendations(comparison)
        }
    
    def _generate_comparison_summary(self, ab_results: Dict) -> Dict[str, Any]:
        """Generate comparison summary between different methods"""
        
        summary = {
            'best_hit_at_1': {'method': '', 'score': 0},
            'best_hit_at_3': {'method': '', 'score': 0},
            'best_hit_at_5': {'method': '', 'score': 0},
            'best_mrr': {'method': '', 'score': 0},
            'fastest': {'method': '', 'time_ms': float('inf')},
            'method_ranking': []
        }
        
        for method_name, data in ab_results.items():
            metrics = data['metrics']
            
            # Track best performers
            if metrics['hit_at_1']['score'] > summary['best_hit_at_1']['score']:
                summary['best_hit_at_1'] = {'method': method_name, 'score': metrics['hit_at_1']['score']}
            
            if metrics['hit_at_3']['score'] > summary['best_hit_at_3']['score']:
                summary['best_hit_at_3'] = {'method': method_name, 'score': metrics['hit_at_3']['score']}
            
            if metrics['hit_at_5']['score'] > summary['best_hit_at_5']['score']:
                summary['best_hit_at_5'] = {'method': method_name, 'score': metrics['hit_at_5']['score']}
            
            if metrics['mrr']['score'] > summary['best_mrr']['score']:
                summary['best_mrr'] = {'method': method_name, 'score': metrics['mrr']['score']}
            
            if metrics['response_time']['avg_ms'] < summary['fastest']['time_ms']:
                summary['fastest'] = {'method': method_name, 'time_ms': metrics['response_time']['avg_ms']}
        
        return summary
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results"""
        
        recommendations = []
        
        # Performance recommendations
        best_overall = comparison.get('best_mrr', {}).get('method', '')
        if 'hybrid' in best_overall:
            recommendations.append("🎯 Hybrid search provides the best overall performance (highest MRR)")
        
        if 'with_ai' in best_overall:
            recommendations.append("🤖 AI enhancement improves search quality for complex queries")
        
        # Speed recommendations
        fastest = comparison.get('fastest', {}).get('method', '')
        fastest_time = comparison.get('fastest', {}).get('time_ms', 0)
        if fastest_time < 50:
            recommendations.append(f"⚡ {fastest} provides excellent response time (<50ms)")
        
        # General recommendations
        recommendations.extend([
            "📊 Use Hit@K metrics to monitor search quality over time",
            "🔬 Run regular A/B tests to validate system improvements",
            "🎛️ Consider user query patterns when choosing default search method"
        ])
        
        return recommendations
    
    def save_evaluation_report(self, results: Dict[str, Any], filepath: str = None) -> str:
        """Save detailed evaluation report to file"""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/evaluation/evaluation_report_{timestamp}.json"
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Evaluation report saved to {filepath}")
        return filepath
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, EvaluationResult):
            return {
                'query': obj.query,
                'retrieved_doc_ids': obj.retrieved_doc_ids,
                'relevant_doc_ids': obj.relevant_doc_ids,
                'hit_at_1': obj.hit_at_1,
                'hit_at_3': obj.hit_at_3,
                'hit_at_5': obj.hit_at_5,
                'mrr_score': obj.mrr_score,
                'response_time_ms': obj.response_time_ms,
                'search_type': obj.search_type,
                'ai_enhanced': obj.ai_enhanced
            }
        else:
            return obj

# Global service instance
evaluation_service = EvaluationService()
