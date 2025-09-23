"""
🎯 Pydantic Models & Schemas
============================
Purpose: Type-safe API models for Bengali Literature RAG System
 use Pydantic for automatic validation, serialization, and API documentation"
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class SearchType(str, Enum):
    """Search type enumeration for different retrieval strategies"""
    HYBRID = "hybrid"      # Dense + Sparse + RRF (Best performance)
    DENSE = "dense"        # Vector similarity only (Semantic understanding)
    SPARSE = "sparse"      # BM25 keyword matching only (Exact matches)

class DifficultyLevel(int, Enum):
    """Question difficulty levels for filtering"""
    EASY = 1
    MEDIUM = 2
    HARD = 3

class SearchRequest(BaseModel):
    """
    🔍 Search request model with validation
    
    Why comprehensive validation:
    - Prevents malformed queries from reaching search engine
    - Provides clear error messages to users
    - Enables automatic API documentation
    """
    query: str = Field(
        ..., 
        min_length=2, 
        max_length=500, 
        description="Search query in Bengali or English",
        examples=["রবীন্দ্রনাথ ঠাকুর", "গীতাঞ্জলি", "নোবেল পুরস্কার"]
    )
    
    top_k: int = Field(
        5, 
        ge=1, 
        le=20, 
        description="Number of results to return"
    )
    
    search_type: SearchType = Field(
        SearchType.HYBRID, 
        description="Type of search to perform"
    )
    
    difficulty_filter: Optional[List[DifficultyLevel]] = Field(
        None, 
        description="Filter results by difficulty levels"
    )
    
    use_llm: bool = Field(
        False, 
        description="Enable GPT-4o Mini explanation enhancement (bonus feature)"
    )
    
    use_reranking: bool = Field(
        False, 
        description="Enable intelligent reranking (bonus feature)"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Clean and validate search query"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Basic cleaning
        cleaned = v.strip()
        
        # Check for potentially malicious content
        if any(char in cleaned for char in ['<', '>', '{', '}', ';']):
            raise ValueError("Query contains invalid characters")
        
        return cleaned
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "রবীন্দ্রনাথ ঠাকুর গীতাঞ্জলি",
                "top_k": 5,
                "search_type": "hybrid",
                "difficulty_filter": [1, 2],
                "use_llm": False,
                "use_reranking": False
            }
        }

class QuestionResult(BaseModel):
    """
    📝 Individual search result with comprehensive metadata
    
    Why detailed metadata:
    - Enables rich UI display
    - Provides debugging information
    - Supports different ranking algorithms
    """
    
    # Core content
    id: str = Field(..., description="Unique document identifier")
    question_id: str = Field(..., description="Original question ID from dataset")
    question: str = Field(..., description="The main question text")
    explanation: str = Field(..., description="Detailed explanation or answer")
    options: Dict[str, str] = Field(default_factory=dict, description="Multiple choice options")
    correct_answer: str = Field("", description="Correct answer identifier")
    difficulty: int = Field(1, ge=1, le=3, description="Question difficulty (1-3)")
    
    # Scoring information (for debugging and optimization)
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Primary relevance score")
    dense_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Vector similarity score")
    sparse_score: Optional[float] = Field(None, ge=0.0, description="BM25 score")
    rrf_score: Optional[float] = Field(None, ge=0.0, description="Reciprocal Rank Fusion score")
    llm_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM enhancement score")
    
    # Metadata
    text_length: int = Field(0, ge=0, description="Length of searchable text")
    has_explanation: bool = Field(False, description="Whether question has detailed explanation")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Content quality assessment")
    search_type: str = Field("hybrid", description="Search method used to find this result")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_42",
                "question_id": "4645", 
                "question": "রবীন্দ্রনাথের 'গীতাঞ্জলী' কাব্যগ্রন্থের ইংরেজি অনুবাদ করেন কে?",
                "explanation": "গীতাঞ্জলি রবীন্দ্রনাথ ঠাকুরের ১৫৭টি গানের সংকলন। রবীন্দ্রনাথ নিজেই এই গ্রন্থের ইংরেজি অনুবাদ 'Song Offerings' করেন।",
                "options": {
                    "A": "টি.এস. এলিয়ট",
                    "B": "ডব্লিউ বি. ইয়েটস", 
                    "C": "রবীন্দ্রনাথ ঠাকুর",
                    "D": "বুদ্ধদেব বসু"
                },
                "correct_answer": "C",
                "difficulty": 2,
                "relevance_score": 0.95,
                "dense_score": 0.89,
                "sparse_score": 0.76,
                "rrf_score": 0.82
            }
        }

class SearchResponse(BaseModel):
    """
    📊 Complete search response with metadata and performance info
    
    Why comprehensive response:
    - Provides performance metrics for optimization
    - Enables A/B testing of different search strategies
    - Helps with debugging and system monitoring
    """
    
    # Query information
    query: str = Field(..., description="Original search query")
    processed_query: Optional[str] = Field(None, description="Query after preprocessing")
    search_type: str = Field(..., description="Search method used")
    
    # Results
    results: List[QuestionResult] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results found")
    
    # Performance metrics
    response_time_ms: float = Field(..., ge=0, description="Total response time in milliseconds")
    
    # Search metadata
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Filters applied to results")
    search_stats: Dict[str, Any] = Field(default_factory=dict, description="Search algorithm statistics")
    
    # Feature usage tracking
    features_used: Dict[str, bool] = Field(default_factory=dict, description="Which features were enabled")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "রবীন্দ্রনাথ গীতাঞ্জলি",
                "processed_query": "রবীন্দ্রনাথ গীতাঞ্জলি",
                "search_type": "hybrid",
                "results": [],
                "total_results": 5,
                "response_time_ms": 145.2,
                "filters_applied": {"difficulty": [1, 2]},
                "features_used": {"llm_enhancement": False, "reranking": False}
            }
        }

class HealthResponse(BaseModel):
    """System health check response"""
    status: str = Field("healthy", description="System status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field("1.0.0", description="API version")
    
    # Service status
    services: Dict[str, str] = Field(default_factory=dict, description="Individual service statuses")
    
    # System information  
    system_info: Dict[str, Any] = Field(default_factory=dict, description="System performance info")
    
    # Database stats
    database_stats: Dict[str, Any] = Field(default_factory=dict, description="Embedding database statistics")

class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: bool = Field(True, description="Error flag")
    message: str = Field(..., description="Human-readable error message")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class SearchSuggestionRequest(BaseModel):
    """Search suggestion request"""
    query: str = Field(..., min_length=1, max_length=100, description="Partial query for suggestions")
    limit: int = Field(5, ge=1, le=10, description="Maximum number of suggestions")

class SearchSuggestionResponse(BaseModel):
    """Search suggestions response"""
    suggestions: List[str] = Field(..., description="List of suggested queries")
    query: str = Field(..., description="Original query")

# Type aliases for complex types
SearchFilters = Dict[str, Union[str, int, List[int], bool]]
SearchMetadata = Dict[str, Union[str, int, float, bool]]
PerformanceMetrics = Dict[str, Union[int, float, str]]
