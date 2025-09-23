"""
🎯 Generate LaBSE Embeddings (ChromaDB-Free)
==========================================
Purpose: Create multilingual embeddings without ChromaDB dependency issues
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any
import logging
import time
import json
from datetime import datetime

# ML imports
from sentence_transformers import SentenceTransformer

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from app.core.config import settings
from app.services.text_processor import text_processor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbeddingGenerator:
    """🧠 Simplified embedding generation without ChromaDB dependency issues"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self.stats = {'total_texts_processed': 0, 'total_processing_time': 0, 'model_load_time': 0}
    
    def load_model(self) -> None:
        """Load the sentence transformer model"""
        if self.model is not None:
            return
        
        logger.info(f"🚀 Loading embedding model: {self.model_name}")
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            self.stats['model_load_time'] = load_time
            
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"📊 Model info:")
            logger.info(f"   - Max sequence length: {self.model.max_seq_length}")
            logger.info(f"   - Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts"""
        if not texts:
            return np.array([])
        
        self.load_model()
        
        logger.info(f"🔄 Generating embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            processing_time = time.time() - start_time
            self.stats['total_texts_processed'] += len(texts)
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings in {processing_time:.2f}s")
            logger.info(f"📊 Average: {processing_time/len(texts)*1000:.2f}ms per text")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise
    
    def save_embeddings_and_data(self, embeddings: np.ndarray, texts: List[str], 
                                 metadata: List[Dict], output_dir: str) -> None:
        """Save embeddings and metadata"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving embeddings to {output_path}")
        
        # Save complete data as pickle
        embeddings_data = {
            'embeddings': embeddings,
            'texts': texts,
            'metadata': metadata,
            'model_name': self.model_name,
            'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'generation_timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
        
        embeddings_file = output_path / 'embeddings.pkl'
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = embeddings_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✅ Complete embeddings saved ({file_size:.2f} MB)")
        
        # Save additional formats
        np.save(output_path / 'embeddings_array.npy', embeddings)
        
        with open(output_path / 'texts.json', 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        
        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ All files saved successfully")

def process_dataset_simple(input_csv_path: str, output_dir: str) -> Dict[str, Any]:
    """🏭 Simplified pipeline without ChromaDB dependency"""
    logger.info("🚀 Starting simplified embedding generation pipeline...")
    
    # Load processed dataset
    df = pd.read_csv(input_csv_path)
    logger.info(f"📊 Loaded {len(df)} rows")
    
    # Prepare texts for embedding
    texts_to_embed = []
    metadata_list = []
    
    for idx, row in df.iterrows():
        searchable_text = row.get('search_text', '')
        
        if not searchable_text or len(str(searchable_text).strip()) < 10:
            logger.warning(f"Skipping row {idx}: insufficient content")
            continue
        
        texts_to_embed.append(str(searchable_text))
        
        metadata = {
            'row_id': int(idx),
            'question': str(row.get('question_clean', '')),
            'explanation': str(row.get('explanation_clean', '')),
            'difficulty': int(row.get('difficulty', 1)),
            'quality_score': float(row.get('explanation_quality', 0.0)),
            'text_length': len(str(searchable_text)),
            'has_explanation': len(str(row.get('explanation_clean', '')).strip()) > 10
        }
        
        if 'answer_clean' in row and row['answer_clean']:
            metadata['answer'] = str(row['answer_clean'])
        
        metadata_list.append(metadata)
    
    logger.info(f"📝 Prepared {len(texts_to_embed)} texts for embedding")
    
    # Generate embeddings
    embedding_generator = SimpleEmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(texts_to_embed, batch_size=32)
    
    # Save everything
    embedding_generator.save_embeddings_and_data(embeddings, texts_to_embed, metadata_list, output_dir)
    
    # Generate stats
    stats = {
        'total_documents': len(texts_to_embed),
        'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
        'model_used': embedding_generator.model_name,
        'processing_time': embedding_generator.stats['total_processing_time'],
        'generation_timestamp': datetime.now().isoformat()
    }
    
    logger.info("✅ Simplified embedding generation completed successfully!")
    
    return stats

if __name__ == "__main__":
    """Main execution"""
    project_root = Path(__file__).parent.parent
    processed_csv = project_root / 'data' / 'processed' / 'clean_questions.csv'
    output_directory = project_root / 'data' / 'processed'
    
    if not processed_csv.exists():
        logger.error(f"❌ Processed CSV not found: {processed_csv}")
        exit(1)
    
    try:
        stats = process_dataset_simple(str(processed_csv), str(output_directory))
        
        print("\n🎉 EMBEDDING GENERATION COMPLETED!")
        print(f"📊 Generated embeddings for {stats['total_documents']} documents")
        print(f"⚡ Model used: {stats['model_used']}")
        print(f"🚀 Processing time: {stats['processing_time']:.2f}s")
        print("🎯 READY FOR PHASE 2!")
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        exit(1)
