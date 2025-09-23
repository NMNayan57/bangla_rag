"""
🎯 Core Configuration Management
================================================
Purpose: Centralized settings with environment variable support
 pydantic-settings for type-safe configuration management"
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import os

class Settings(BaseSettings):
    """
    🔧 Application configuration with environment variable support
    
    Why Pydantic Settings:
    - Type validation at startup
    - Environment variable auto-loading  
    - Documentation via Field descriptions
    - IDE autocomplete support
    """
    
    # ===== API Configuration =====
    API_TITLE: str = Field("Bengali Literature RAG System", description="API title")
    API_VERSION: str = Field("1.0.0", description="API version")
    API_HOST: str = Field("0.0.0.0", description="API host")
    API_PORT: int = Field(8000, description="API port")
    DEBUG: bool = Field(False, description="Debug mode")
    
    # ===== Data Paths =====
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    RAW_DATA_PATH: str = Field("data/raw/questions.csv", description="Raw CSV file path")
    PROCESSED_DATA_PATH: str = Field("data/processed/clean_questions.csv", description="Processed CSV path")
    EMBEDDINGS_PATH: str = Field("data/processed/embeddings.pkl", description="Embeddings storage path")
    BM25_INDEX_PATH: str = Field("data/processed/bm25_index.pkl", description="BM25 index path")
    METADATA_PATH: str = Field("data/processed/metadata.json", description="Processing metadata path")
    
    # ===== ML Model Configuration =====
    EMBEDDING_MODEL: str = Field(
        "sentence-transformers/LaBSE", 
        description="Embedding model for multilingual support"
    )
    EMBEDDING_DIMENSION: int = Field(768, description="Embedding vector dimension")
    MAX_SEQUENCE_LENGTH: int = Field(512, description="Maximum input sequence length")
    BATCH_SIZE: int = Field(32, description="Batch size for embedding generation")
    
    # ===== Search Configuration =====
    DEFAULT_TOP_K: int = Field(5, description="Default number of search results")
    MAX_TOP_K: int = Field(20, description="Maximum allowed search results")
    MIN_QUERY_LENGTH: int = Field(2, description="Minimum query length")
    MAX_QUERY_LENGTH: int = Field(500, description="Maximum query length")
    
    # ===== BM25 Parameters =====
    BM25_K1: float = Field(1.2, description="BM25 term frequency saturation parameter")
    BM25_B: float = Field(0.75, description="BM25 length normalization parameter")
    
    # ===== Hybrid Search Configuration =====
    RRF_K: int = Field(60, description="Reciprocal Rank Fusion parameter")
    DENSE_WEIGHT: float = Field(0.5, description="Weight for dense retrieval in hybrid")
    SPARSE_WEIGHT: float = Field(0.5, description="Weight for sparse retrieval in hybrid")
    
    # ===== ChromaDB Configuration =====
    CHROMADB_HOST: str = Field("localhost", description="ChromaDB host")
    CHROMADB_PORT: int = Field(8001, description="ChromaDB port")
    COLLECTION_NAME: str = Field("bengali_literature", description="ChromaDB collection name")
    CHROMADB_PERSIST_DIR: str = Field("data/chromadb", description="ChromaDB persistence directory")
    
    # ===== GPT Bonus Configuration =====
    ENABLE_GPT_BONUS: bool = Field(False, description="Enable GPT-4o Mini bonus features")
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key for GPT features")
    GPT_MODEL: str = Field("gpt-4o-mini", description="GPT model to use")
    GPT_MAX_TOKENS: int = Field(150, description="Maximum tokens for GPT responses")
    GPT_TEMPERATURE: float = Field(0.3, description="GPT temperature for response generation")
    
    # ===== ALL MISSING FIELDS FROM YOUR .ENV =====
    ENABLE_HYBRID_SEARCH: bool = Field(True, description="Enable hybrid search")
    ENABLE_RERANKING: bool = Field(False, description="Enable reranking")
    OLLAMA_URL: str = Field("http://localhost:11434", description="Ollama URL")
    LLM_MODEL: str = Field("llama3.2:1b", description="LLM model")
    SEARCH_CACHE_TTL: int = Field(3600, description="Search cache TTL")
    MAX_SEARCH_RESULTS: int = Field(20, description="Max search results")
    REDIS_URL: str = Field("redis://localhost:6379", description="Redis URL")
    ENABLE_LLM: bool = Field(False, description="Enable LLM features")
    
    # ===== Performance & Caching =====
    CACHE_SIZE: int = Field(1000, description="In-memory cache size")
    REQUEST_TIMEOUT: int = Field(30, description="Request timeout in seconds")
    ENABLE_RESPONSE_CACHE: bool = Field(True, description="Enable response caching")
    CACHE_TTL: int = Field(3600, description="Cache TTL in seconds")
    
    # ===== Logging Configuration =====
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # ===== CORS Configuration =====
    CORS_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
        description="Allowed CORS origins"
    )
    CORS_METHODS: List[str] = Field(["GET", "POST"], description="Allowed CORS methods")
    CORS_HEADERS: List[str] = Field(["*"], description="Allowed CORS headers")
    
    @validator('EMBEDDING_MODEL')
    def validate_embedding_model(cls, v):
        """Validate embedding model name"""
        supported_models = [
            'sentence-transformers/LaBSE',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/all-MiniLM-L6-v2'
        ]
        if v not in supported_models:
            raise ValueError(f"Unsupported embedding model. Supported: {supported_models}")
        return v
    
    @validator('BM25_K1', 'BM25_B', 'DENSE_WEIGHT', 'SPARSE_WEIGHT')
    def validate_positive_float(cls, v):
        """Validate positive float parameters"""
        if v <= 0:
            raise ValueError("Parameter must be positive")
        return v
    
    @validator('DENSE_WEIGHT', 'SPARSE_WEIGHT')
    def validate_weight_range(cls, v):
        """Validate weight parameters are between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path"""
        return self.PROJECT_ROOT / relative_path
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get all data paths as absolute paths"""
        return {
            'raw_data': self.get_absolute_path(self.RAW_DATA_PATH),
            'processed_data': self.get_absolute_path(self.PROCESSED_DATA_PATH),
            'embeddings': self.get_absolute_path(self.EMBEDDINGS_PATH),
            'bm25_index': self.get_absolute_path(self.BM25_INDEX_PATH),
            'metadata': self.get_absolute_path(self.METADATA_PATH),
            'chromadb': self.get_absolute_path(self.CHROMADB_PERSIST_DIR)
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

def validate_setup() -> bool:
    """
    🔍 Validate system setup and requirements
    
    Purpose: Ensure all required files and dependencies are available
    Returns: True if setup is valid, raises exception otherwise
    """
    import importlib
    import sys
    
    # Check required Python packages (removed chromadb for now)
    required_packages = [
        'sentence_transformers',
        'pandas',
        'numpy',
        'sklearn',
        'rank_bm25'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {missing_packages}")
    
    # Check data directories
    paths = settings.get_data_paths()
    
    # Create directories if they don't exist
    for path_name, path in paths.items():
        if path_name != 'raw_data':  # Don't create raw_data directory
            path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate raw data exists
    if not paths['raw_data'].exists():
        raise FileNotFoundError(f"Raw data file not found: {paths['raw_data']}")
    
    return True

if __name__ == "__main__":
    """Test configuration loading"""
    print("🔧 Testing configuration...")
    print(f"Project root: {settings.PROJECT_ROOT}")
    print(f"Embedding model: {settings.EMBEDDING_MODEL}")
    print(f"Data paths: {settings.get_data_paths()}")
    
    try:
        validate_setup()
        print("✅ Configuration validation passed!")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
