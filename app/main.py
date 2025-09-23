"""
🎯 FastAPI Application - Bengali Literature RAG System
====================================================
Purpose: Production-ready API with hybrid search capabilities
 built a scalable FastAPI application with comprehensive error handling and monitoring"
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
#from app.services.evaluation_service import evaluation_service  # Commented for performance

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

# Local imports
from app.core.config import settings
from app.models.schemas import (
    SearchRequest, SearchResponse, QuestionResult, 
    HealthResponse, ErrorResponse, SearchSuggestionRequest, SearchSuggestionResponse
)
from app.services.hybrid_search import hybrid_search_service
from app.services.vector_search import vector_search_service
from app.services.sparse_search import sparse_search_service
from app.services.gpt_service import gpt_service  # GPT service import
from app.services.text_processor import text_processor

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    🚀 Application lifecycle management
    
    Startup:
    - Initialize all search services
    - Load embeddings and build indexes
    - Validate system health
    
    Shutdown:
    - Cleanup resources
    - Log final statistics
    """
    logger.info("🚀 Starting Bengali Literature RAG System...")
    
    try:
        # Initialize services in parallel for faster startup
        logger.info("📚 Initializing search services...")
        
        initialization_tasks = [
            vector_search_service.initialize(),
            sparse_search_service.initialize(),
            gpt_service.initialize()  # Add GPT initialization
        ]
        
        await asyncio.gather(*initialization_tasks)
        
        logger.info("✅ All services initialized successfully")
        
        # Validate system health
        health_status = await get_system_health()
        if not all(service == "healthy" for service in health_status.services.values()):
            logger.warning("⚠️ Some services are not fully healthy")
        
        logger.info("🎉 Bengali Literature RAG System is ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down Bengali Literature RAG System...")
    
    # Log final statistics
    vector_stats = vector_search_service.get_stats()
    sparse_stats = sparse_search_service.get_stats()
    hybrid_stats = hybrid_search_service.get_stats()
    gpt_stats = gpt_service.get_stats()  # Add GPT stats
    
    logger.info("📊 Final Statistics:")
    logger.info(f"   Vector searches: {vector_stats['total_searches']}")
    logger.info(f"   Sparse searches: {sparse_stats['total_searches']}")
    logger.info(f"   Hybrid searches: {hybrid_stats['total_searches']}")
    logger.info(f"   GPT enhancements: {gpt_stats['successful_calls']}")
    
    logger.info("👋 Goodbye!")

# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Advanced RAG system for Bengali literature with hybrid search capabilities",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Mount static files (for simple HTML interface later)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    logger.warning("Static files directory not found - UI will not be available")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors gracefully"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            message="An unexpected error occurred. Please try again later.",
            error_code="INTERNAL_ERROR",
            details={"path": str(request.url.path)} if settings.DEBUG else None
        ).dict()
    )

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bengali Literature RAG System API",
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "debug_gpt": "/debug/gpt",
        "evaluation_docs": "/docs/evaluation"
    }

@app.get("/docs/evaluation")
async def evaluation_docs():
    """📚 Evaluation system documentation"""
    return {
        "title": "RAG System Evaluation Guide",
        "overview": "Our system has comprehensive evaluation metrics for performance analysis",
        
        "when_to_use_evaluation": {
            "for_developers": [
                "Testing system performance after changes",
                "Comparing different search methods",
                "Validating improvements",
                "Research and analysis"
            ],
            "for_end_users": [
                "Not needed - just use the search interface at /ui",
                "Evaluation is for system optimization only"
            ],
            "for_interviews": [
                "Shows systematic ML approach",
                "Demonstrates evaluation methodology",
                "Proves system quality"
            ]
        },
        
        "evaluation_types": {
            "quick_test": {
                "purpose": "Fast system check (3 queries)",
                "time": "30 seconds",
                "use_case": "After code changes",
                "endpoint": "/api/evaluation/quick"
            },
            "full_evaluation": {
                "purpose": "Comprehensive analysis (8 queries)",
                "time": "1-2 minutes",
                "use_case": "Weekly performance check",
                "endpoint": "/api/evaluate"
            },
            "ab_testing": {
                "purpose": "Compare all methods",
                "time": "3-5 minutes", 
                "use_case": "Major system decisions",
                "endpoint": "/api/ab-test"
            }
        },
        
        "metrics_explained": {
            "hit_at_k": "Percentage of queries with relevant results in top-K",
            "mrr": "Mean Reciprocal Rank - higher is better (0-1)",
            "response_time": "Average search speed in milliseconds"
        },
        
        "recommendation": "Most users should just use /api/search or /ui. Evaluation is for system analysis only."
    }

@app.get("/debug/gpt", response_model=Dict[str, Any])
async def debug_gpt():
    """
    🐛 GPT Service Debug Endpoint
    
    Purpose: Detailed GPT service status and configuration
    Useful for: Troubleshooting GPT integration issues
    """
    return {
        "service_status": gpt_service.get_status(),
        "service_stats": gpt_service.get_stats(),
        "config_check": {
            "enable_gpt_bonus": settings.ENABLE_GPT_BONUS,
            "has_openai_key": bool(settings.OPENAI_API_KEY),
            "openai_key_preview": settings.OPENAI_API_KEY[:10] + "..." if settings.OPENAI_API_KEY else None,
            "gpt_model": settings.GPT_MODEL,
            "max_tokens": settings.GPT_MAX_TOKENS,
            "temperature": settings.GPT_TEMPERATURE
        },
        "runtime_info": {
            "client_initialized": gpt_service.client is not None,
            "is_available": gpt_service.is_available,
            "cache_size": len(gpt_service.response_cache),
            "cache_max_size": gpt_service.cache_max_size
        }
    }

@app.get("/api/evaluation/info")
async def get_evaluation_info():
    """📊 Get evaluation service information without running tests"""
    try:
        from app.services.evaluation_service import evaluation_service
        return evaluation_service.get_quick_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation info failed: {str(e)}")

@app.post("/api/evaluation/quick")
async def run_quick_evaluation(search_type: str = "hybrid", use_ai: bool = False):
    """⚡ Quick evaluation with 3 queries (30 seconds)"""
    try:
        from app.services.evaluation_service import evaluation_service
        from app.models.schemas import SearchType
        
        logger.info(f"⚡ Running quick evaluation: {search_type}, AI: {use_ai}")
        
        search_type_enum = SearchType(search_type)
        result = await evaluation_service.quick_evaluation(search_type_enum, use_ai)
        return result
    except Exception as e:
        logger.error(f"Quick evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Quick evaluation failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    🏥 Comprehensive health check endpoint
    
    Checks:
    - Service availability
    - Database connections  
    - System performance
    - Error rates
    """
    return await get_system_health()

async def get_system_health() -> HealthResponse:
    """Get detailed system health information"""
    try:
        services = {}
        system_info = {}
        database_stats = {}
        
        # Check vector search service
        try:
            if vector_search_service.is_loaded:
                services["vector_search"] = "healthy"
                vector_stats = vector_search_service.get_stats()
                database_stats["vector_search"] = {
                    "total_documents": vector_stats["total_documents"],
                    "total_searches": vector_stats["total_searches"],
                    "avg_search_time_ms": vector_stats.get("avg_search_time", 0) * 1000
                }
            else:
                services["vector_search"] = "initializing"
        except Exception as e:
            services["vector_search"] = f"error: {str(e)}"
        
        # Check sparse search service
        try:
            if sparse_search_service.is_loaded:
                services["sparse_search"] = "healthy"
                sparse_stats = sparse_search_service.get_stats()
                database_stats["sparse_search"] = {
                    "total_documents": sparse_stats["total_documents"],
                    "vocabulary_size": sparse_stats["vocabulary_size"],
                    "total_searches": sparse_stats["total_searches"]
                }
            else:
                services["sparse_search"] = "initializing"
        except Exception as e:
            services["sparse_search"] = f"error: {str(e)}"
        
        # Check hybrid search service
        try:
            hybrid_stats = hybrid_search_service.get_stats()
            services["hybrid_search"] = "healthy"
            system_info["hybrid_searches"] = hybrid_stats["total_searches"]
            system_info["avg_search_time_ms"] = hybrid_stats.get("avg_search_time", 0) * 1000
        except Exception as e:
            services["hybrid_search"] = f"error: {str(e)}"
        
        # Check GPT service
        try:
            if gpt_service.is_available:
                services["gpt_service"] = "healthy"
                gpt_stats = gpt_service.get_stats()
                database_stats["gpt_service"] = {
                    "total_calls": gpt_stats["total_calls"],
                    "successful_calls": gpt_stats["successful_calls"],
                    "success_rate": gpt_stats.get("success_rate_percent", "0%"),
                    "estimated_cost": f"${gpt_stats.get('estimated_cost_usd', 0):.4f}"
                }
            else:
                services["gpt_service"] = "disabled"
        except Exception as e:
            services["gpt_service"] = f"error: {str(e)}"
        
        # Check evaluation service (lazy check)
        try:
            from app.services.evaluation_service import evaluation_service
            services["evaluation_service"] = "available"
        except Exception:
            services["evaluation_service"] = "disabled"
        
        # System information
        system_info.update({
            "embedding_model": settings.EMBEDDING_MODEL,
            "search_types_available": ["hybrid", "dense", "sparse"],
            "max_top_k": settings.MAX_TOP_K,
            "default_top_k": settings.DEFAULT_TOP_K,
            "gpt_enabled": settings.ENABLE_GPT_BONUS,
            "evaluation_available": services.get("evaluation_service") == "available"
        })
        
        return HealthResponse(
            status="healthy" if all("healthy" in str(status) or status == "available" or status == "disabled" for status in services.values()) else "degraded",
            services=services,
            system_info=system_info,
            database_stats=database_stats
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            services={"system": f"error: {str(e)}"}
        )

@app.post("/api/search", response_model=SearchResponse)
async def search_questions(request: SearchRequest):
    """
    🔍 Main search endpoint with hybrid capabilities
    
    Features:
    - Hybrid search (dense + sparse + RRF fusion)
    - Dense-only search (semantic similarity)
    - Sparse-only search (keyword matching)
    - GPT-4o Mini enhancement (optional)
    - Difficulty filtering
    - Performance monitoring
    """
    start_time = time.time()
    
    logger.info(f"🔍 Search request: '{request.query}' (type: {request.search_type}, top_k: {request.top_k}, use_llm: {request.use_llm})")
    
    try:
        # Process query
        processed_query = text_processor.normalize_unicode(
            text_processor.clean_html(request.query)
        )
        
        # Perform search
        results = await hybrid_search_service.search(request)
        
        # Apply GPT enhancement if requested
        llm_enhanced = False
        if request.use_llm and gpt_service.is_available:
            try:
                logger.info("🤖 Applying GPT-4o Mini enhancement...")
                enhanced_results = await gpt_service.enhance_results(request.query, results)
                if enhanced_results:
                    results = enhanced_results
                    llm_enhanced = True
                    logger.info("✅ GPT enhancement successful")
            except Exception as e:
                logger.error(f"GPT enhancement failed: {e}")
        
        # Apply reranking if requested
        if request.use_llm and request.use_reranking and gpt_service.is_available:
            try:
                results = await gpt_service.rerank_results(request.query, results)
            except Exception as e:
                logger.error(f"GPT reranking failed: {e}")
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Prepare response
        filters_applied = {}
        if request.difficulty_filter:
            filters_applied["difficulty"] = request.difficulty_filter
        
        # Search statistics
        search_stats = {
            "query_tokens": len(text_processor.tokenize_bengali(processed_query)),
            "search_method": request.search_type.value,
            "results_found": len(results)
        }
        
        # Feature usage
        features_used = {
            "llm_enhancement": llm_enhanced,
            "reranking": request.use_reranking and llm_enhanced,
            "hybrid_fusion": request.search_type.value == "hybrid"
        }
        
        response = SearchResponse(
            query=request.query,
            processed_query=processed_query,
            search_type=request.search_type.value,
            results=results,
            total_results=len(results),
            response_time_ms=response_time_ms,
            filters_applied=filters_applied,
            search_stats=search_stats,
            features_used=features_used
        )
        
        logger.info(f"✅ Search completed: {len(results)} results in {response_time_ms:.1f}ms (LLM: {llm_enhanced})")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/api/evaluate", response_model=Dict[str, Any])
async def run_evaluation(search_type: str = "hybrid", use_ai: bool = False, top_k: int = 5):
    """
    📊 Full evaluation (1-2 minutes)
    Tests 8 queries across different categories
    """
    try:
        from app.services.evaluation_service import evaluation_service
        from app.models.schemas import SearchType
        
        logger.info(f"🧪 Starting full evaluation (estimated time: 1-2 minutes)")
        
        search_type_enum = SearchType(search_type)
        results = await evaluation_service.evaluate_search_method(search_type_enum, use_ai, top_k)
        metrics = evaluation_service.calculate_aggregate_metrics(results)
        
        return {
            'configuration': {
                'search_type': search_type,
                'ai_enhanced': use_ai,
                'top_k': top_k
            },
            'metrics': metrics,
            'total_queries_tested': len(results),
            'test_duration_note': "Full evaluation with 8 diverse queries",
            'individual_results': [
                {
                    'query': r.query,
                    'hit_at_1': r.hit_at_1,
                    'hit_at_3': r.hit_at_3,
                    'hit_at_5': r.hit_at_5,
                    'mrr_score': r.mrr_score,
                    'response_time_ms': r.response_time_ms
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/api/ab-test", response_model=Dict[str, Any])
async def run_ab_test():
    """
    🔬 Full A/B testing (3-5 minutes)
    Compares all search methods comprehensively
    """
    try:
        from app.services.evaluation_service import evaluation_service
        
        logger.info("🔬 Starting A/B testing (estimated time: 3-5 minutes)")
        ab_results = await evaluation_service.run_ab_test()
        
        # Save detailed report
        report_path = evaluation_service.save_evaluation_report(ab_results)
        
        return {
            'test_completed': True,
            'report_saved': report_path,
            'summary': ab_results.get('comparison_summary', {}),
            'recommendations': ab_results.get('recommendations', []),
            'detailed_metrics': {
                method: data['metrics'] 
                for method, data in ab_results.get('detailed_results', {}).items()
            },
            'note': 'Complete A/B test comparing all search methods'
        }
    except Exception as e:
        logger.error(f"❌ A/B testing failed: {e}")
        raise HTTPException(status_code=500, detail=f"A/B testing failed: {str(e)}")

@app.get("/api/similar/{doc_id}", response_model=SearchResponse)
async def find_similar_questions(doc_id: str, top_k: int = 5):
    """
    🔗 Find questions similar to a given question
    
    Use case: "More like this" functionality
    """
    start_time = time.time()
    
    logger.info(f"🔗 Finding similar questions to: {doc_id}")
    
    try:
        results = await hybrid_search_service.search_similar(doc_id, top_k)
        response_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=f"Similar to {doc_id}",
            search_type="similar",
            results=results,
            total_results=len(results),
            response_time_ms=response_time_ms,
            features_used={"similar_search": True}
        )
        
    except Exception as e:
        logger.error(f"❌ Similar search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar search failed: {str(e)}"
        )

@app.get("/api/stats", response_model=Dict[str, Any])
async def get_system_stats():
    """
    📊 Get comprehensive system performance statistics
    
    Useful for:
    - Performance monitoring
    - Usage analytics
    - System optimization
    """
    try:
        stats = {
            "vector_search": vector_search_service.get_stats(),
            "sparse_search": sparse_search_service.get_stats(),
            "hybrid_search": hybrid_search_service.get_stats(),
            "gpt_service": gpt_service.get_stats(),
            "system_info": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "bm25_parameters": {"k1": settings.BM25_K1, "b": settings.BM25_B},
                "rrf_k": settings.RRF_K,
                "gpt_model": settings.GPT_MODEL,
                "gpt_enabled": settings.ENABLE_GPT_BONUS
            }
        }
        
        # Add evaluation stats if available
        try:
            from app.services.evaluation_service import evaluation_service
            stats["evaluation_service"] = evaluation_service.stats
        except Exception:
            stats["evaluation_service"] = "not_loaded"
        
        return stats
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats retrieval failed: {str(e)}"
        )

@app.get("/ui")
async def web_interface():
    """
    🎨 Simple web interface for search
    
    Usage: Visit http://localhost:8000/ui
    """
    try:
        from fastapi.responses import FileResponse
        return FileResponse('static/index.html')
    except Exception as e:
        return {"message": "Web interface not available", "error": str(e)}

@app.post("/api/suggestions", response_model=SearchSuggestionResponse)
async def get_search_suggestions(request: SearchSuggestionRequest):
    """
    💡 Get search suggestions based on partial query
    
    Future enhancement: Could use query logs, popular searches, etc.
    """
    # Basic implementation - can be enhanced with ML-based suggestions
    suggestions = []
    
    # Common Bengali literature search terms
    common_terms = [
        "রবীন্দ্রনাথ ঠাকুর", "গীতাঞ্জলি", "নোবেল পুরস্কার", "কাব্যগ্রন্থ",
        "ভানুসিংহ ঠাকুর", "সোনার তরী", "বলাকা", "চোখের বালি",
        "ঘরে বাইরে", "গোরা", "যোগাযোগ", "শেষের কবিতা"
    ]
    
    query_lower = request.query.lower()
    
    # Find matching terms
    for term in common_terms:
        if query_lower in term.lower() or term.lower().startswith(query_lower):
            suggestions.append(term)
        
        if len(suggestions) >= request.limit:
            break
    
    return SearchSuggestionResponse(
        suggestions=suggestions,
        query=request.query
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
