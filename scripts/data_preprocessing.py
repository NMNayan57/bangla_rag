"""
🎯 Bengali Literature Dataset Preprocessing
==================================================
Purpose: Clean HTML, handle NaN values, normalize Bengali text
Logic: Foundation for superior search accuracy
"
"""

import pandas as pd
import re
import html
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BengaliTextProcessor:
    """
    🧠 Advanced Bengali text processing for literature content
    
    Why this matters:
    - Bengali text has unique Unicode challenges
    - HTML tags are embedded in explanations  
    - Missing values need intelligent handling
    - Proper tokenization is crucial for search accuracy
    """
    
    def __init__(self):
        # Bengali-specific normalization patterns
        self.normalization_patterns = [
            (r'[০-৯]', self._convert_bengali_digits),  # Convert Bengali numerals
            (r'[‍‌]', ''),  # Remove zero-width joiners/non-joiners
            (r'\s+', ' '),  # Normalize multiple spaces
            (r'[।]+', '।'),  # Normalize Bengali periods
            (r'[,]+', ','),  # Normalize commas
        ]
        
        # HTML cleaning patterns (common in explanations)
        self.html_patterns = [
            (r'<br\s*/?>', '\n'),  # Convert <br> to newlines
            (r'<strong>(.*?)</strong>', r'**\1**'),  # Convert bold tags
            (r'<[^>]+>', ''),  # Remove all remaining HTML tags
            (r'&nbsp;', ' '),  # Replace non-breaking spaces
            (r'&[a-zA-Z]+;', ''),  # Remove HTML entities
            (r'&zwnj;', ''),  # Remove zero-width non-joiner
        ]
        
        # Quality indicators for explanation assessment
        self.quality_indicators = [
            'উৎস:', 'সূত্র:', 'তথ্যসূত্র:', 'গ্রন্থ:', 'কাব্য:', 'উপন্যাস:',
            'কবি:', 'লেখক:', 'রচনাকাল:', 'প্রকাশকাল:', 'সাল', 'খ্রিস্টাব্দ'
        ]
    
    def clean_html_content(self, text: str) -> str:
        """
        🧹 Remove HTML tags and clean content
        
        Why needed: Original CSV has HTML formatting in explanations
        Logic: Multi-step cleaning for robust results
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Step 1: HTML decode
        text = html.unescape(text)
        
        # Step 2: Use BeautifulSoup for robust HTML removal
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')
        
        # Step 3: Apply additional cleaning patterns
        for pattern, replacement in self.html_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Step 4: Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_bengali_text(self, text: str) -> str:
        """
        🔧 Normalize Bengali text for better search
        
        Why crucial: Bengali Unicode inconsistencies affect search accuracy
        Logic: Systematic normalization of common issues
        """
        if not text:
            return ""
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def _convert_bengali_digits(self, match) -> str:
        """Convert Bengali numerals to English for consistency"""
        bengali_digits = '০১২৩৪৫৬৭৮৯'
        english_digits = '0123456789'
        digit = match.group()
        return english_digits[bengali_digits.index(digit)]
    
    def assess_explanation_quality(self, text: str) -> float:
        """
        📊 Assess explanation quality (0.0 to 1.0)
        
        Why important: Helps prioritize high-quality content
        Logic: Multi-factor quality scoring
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        quality_score = 0.0
        
        # Length-based scoring
        length = len(text)
        if length > 50:
            quality_score += 0.2
        if length > 150:
            quality_score += 0.2
        if length > 300:
            quality_score += 0.1
        
        # Content quality indicators
        indicator_count = sum(1 for indicator in self.quality_indicators if indicator in text)
        quality_score += min(indicator_count * 0.1, 0.3)
        
        # Structure indicators (sentences, references)
        sentence_count = len([s for s in text.split('।') if s.strip()])
        if sentence_count >= 2:
            quality_score += 0.1
        if sentence_count >= 4:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def create_search_text(self, question: str, explanation: str, options: List[str]) -> str:
        """
        🔍 Create comprehensive searchable text
        
        Purpose: Combine all relevant text for better search coverage
        Logic: Question + explanation + options for complete context
        """
        search_parts = []
        
        if question:
            search_parts.append(question)
        
        if explanation and len(explanation.strip()) > 10:
            search_parts.append(explanation)
        
        # Add options if they exist
        for option in options:
            if option and option.strip():
                search_parts.append(option.strip())
        
        return ' '.join(search_parts)


def preprocess_dataset(input_path: str, output_dir: str) -> Dict[str, Any]:
    """
    🏭 Main preprocessing pipeline
    
    Purpose: Transform raw CSV into clean, searchable format
    Interview Answer: "I preprocess for Bengali-specific challenges that others miss"
    """
    logger.info(f"🚀 Starting dataset preprocessing from {input_path}")
    
    # Load raw data
    df = pd.read_csv(input_path)
    original_rows = len(df)
    logger.info(f"📊 Loaded {original_rows} rows with columns: {list(df.columns)}")
    
    # Initialize processor
    processor = BengaliTextProcessor()
    
    # Statistics tracking
    stats = {
        'original_rows': original_rows,
        'processed_rows': 0,
        'columns_processed': list(df.columns),
        'html_cleaned_count': 0,
        'missing_explanations_fixed': 0,
        'quality_distribution': {},
        'processing_timestamp': pd.Timestamp.now().isoformat()
    }
    
    logger.info("🧹 Cleaning and processing columns...")
    
    # Process Question column
    if 'Question' in df.columns:
        logger.info("Processing Question column...")
        df['question_clean'] = df['Question'].fillna('').astype(str).apply(processor.clean_html_content)
        df['question_clean'] = df['question_clean'].apply(processor.normalize_bengali_text)
        
    # Process Explanation column  
    if 'Explain' in df.columns:
        logger.info("Processing Explain column...")
        # Track HTML cleaning
        html_before = df['Explain'].fillna('').astype(str)
        df['explanation_clean'] = html_before.apply(processor.clean_html_content)
        df['explanation_clean'] = df['explanation_clean'].apply(processor.normalize_bengali_text)
        
        # Count HTML cleaning impact
        html_cleaned = sum(1 for i in range(len(df)) 
                          if html_before.iloc[i] != df['explanation_clean'].iloc[i])
        stats['html_cleaned_count'] = html_cleaned
        
        # Assess explanation quality
        df['explanation_quality'] = df['explanation_clean'].apply(processor.assess_explanation_quality)
        
        # Handle missing explanations
        missing_mask = (df['explanation_clean'].str.len() < 10) | df['explanation_clean'].isna()
        missing_count = missing_mask.sum()
        df.loc[missing_mask, 'explanation_clean'] = 'No detailed explanation available for this question.'
        df.loc[missing_mask, 'explanation_quality'] = 0.0
        stats['missing_explanations_fixed'] = int(missing_count)
        
        # Quality distribution
        quality_bins = pd.cut(df['explanation_quality'], 
                            bins=[0, 0.2, 0.5, 0.8, 1.0], 
                            labels=['Poor', 'Fair', 'Good', 'Excellent'],
                            include_lowest=True)
        stats['quality_distribution'] = quality_bins.value_counts().to_dict()
    
    # Process Options columns
    option_columns = [col for col in df.columns if col.startswith('Option')]
    if option_columns:
        logger.info(f"Processing {len(option_columns)} option columns...")
        
        # Clean each option column
        for col in option_columns:
            clean_col = f"{col.lower()}_clean"
            df[clean_col] = df[col].fillna('').astype(str).apply(processor.clean_html_content)
            df[clean_col] = df[clean_col].apply(processor.normalize_bengali_text)
        
        # Create options list for each row
        df['options_list'] = df[[f"{col.lower()}_clean" for col in option_columns]].values.tolist()
        df['options_list'] = df['options_list'].apply(lambda x: [opt for opt in x if opt.strip()])
    
    # Process Answer column
    if 'Answer' in df.columns:
        df['answer_clean'] = df['Answer'].fillna('').astype(str).apply(processor.clean_html_content)
    
    # Process Difficulty column
    if 'Difficulty' in df.columns:
        df['difficulty'] = pd.to_numeric(df['Difficulty'], errors='coerce').fillna(1).astype(int)
        df['difficulty'] = df['difficulty'].clip(1, 3)  # Ensure 1-3 range
    
    # Create comprehensive search text
    logger.info("🔍 Creating searchable text...")
    def create_search_text_row(row):
        question = row.get('question_clean', '')
        explanation = row.get('explanation_clean', '')
        options = row.get('options_list', [])
        return processor.create_search_text(question, explanation, options)
    
    df['search_text'] = df.apply(create_search_text_row, axis=1)
    df['search_text_length'] = df['search_text'].str.len()
    
    # Remove rows with insufficient content
    min_content_length = 20
    content_mask = df['search_text_length'] >= min_content_length
    df_clean = df[content_mask].copy()
    
    removed_rows = len(df) - len(df_clean)
    if removed_rows > 0:
        logger.warning(f"⚠️ Removed {removed_rows} rows with insufficient content")
    
    # Final statistics
    stats['processed_rows'] = len(df_clean)
    stats['removed_rows'] = removed_rows
    stats['avg_search_text_length'] = float(df_clean['search_text_length'].mean())
    stats['content_coverage'] = float(len(df_clean) / original_rows)
    
    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_csv_path = output_path / 'clean_questions.csv'
    df_clean.to_csv(processed_csv_path, index=False)
    
    # Save metadata
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ Preprocessing completed successfully!")
    logger.info(f"📊 Final stats: {stats['processed_rows']}/{original_rows} rows processed")
    logger.info(f"🎯 Quality distribution: {stats['quality_distribution']}")
    
    return stats


if __name__ == "__main__":
    """
    🎬 Main execution
    Usage: python scripts/data_preprocessing.py
    """
    import sys
    from pathlib import Path
    
    # Paths
    project_root = Path(__file__).parent.parent
    input_csv = project_root / 'data' / 'raw' / 'questions.csv'
    output_dir = project_root / 'data' / 'processed'
    
    # Validate input
    if not input_csv.exists():
        logger.error(f"❌ Input CSV not found: {input_csv}")
        logger.info("Please place your questions.csv in data/raw/ directory")
        sys.exit(1)
    
    # Run preprocessing
    try:
        stats = preprocess_dataset(str(input_csv), str(output_dir))
        logger.info("🎉 Preprocessing pipeline completed successfully!")
        print(f"\n📋 Processing Summary:")
        print(f"   Original rows: {stats['original_rows']}")
        print(f"   Processed rows: {stats['processed_rows']}")  
        print(f"   HTML cleaned: {stats['html_cleaned_count']}")
        print(f"   Missing explanations fixed: {stats['missing_explanations_fixed']}")
        print(f"   Content coverage: {stats['content_coverage']:.1%}")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        sys.exit(1)
