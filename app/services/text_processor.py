"""
🎯 Bengali Text Processing Service
===============================================
Purpose: Advanced text cleaning and normalization for Bengali literature
 handle Bengali-specific Unicode and tokenization challenges"
"""

import re
import html
from typing import List, Dict, Optional, Tuple
import unicodedata
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class BengaliTextProcessor:
    """
    🧠 Advanced Bengali text processing for literature content
    
    Why this is crucial:
    - Bengali Unicode has multiple representations for same characters
    - HTML content mixed with Bengali text in dataset
    - Proper tokenization affects search accuracy significantly
    - Literature text has specific formatting needs
    """
    
    def __init__(self):
        # Bengali Unicode normalization patterns
        self.unicode_patterns = [
            # Normalize Bengali numerals to English
            (r'[০-৯]', self._convert_bengali_digits),
            # Remove zero-width characters that cause matching issues
            (r'[\u200c\u200d]', ''),  # Zero-width non-joiner/joiner
            # Normalize punctuation
            (r'[।]+', '।'),  # Multiple Bengali periods
            (r'[,，]+', ','),  # Various comma types
            (r'[;；]+', ';'),  # Various semicolon types
        ]
        
        # HTML cleaning patterns (common in explanations)
        self.html_patterns = [
            (r'<br\s*/?>', '\n'),  # Convert breaks to newlines
            (r'<strong>(.*?)</strong>', r'**\1**'),  # Convert bold to markdown
            (r'<em>(.*?)</em>', r'*\1*'),  # Convert italic to markdown
            (r'<[^>]+>', ''),  # Remove all other HTML tags
            (r'&nbsp;', ' '),  # Non-breaking spaces
            (r'&zwnj;', ''),  # Zero-width non-joiner entity
            (r'&[a-zA-Z0-9]+;', ''),  # Other HTML entities
        ]
        
        # Bengali stopwords for search optimization
        self.bengali_stopwords = {
            'এবং', 'বা', 'কিন্তু', 'তবে', 'যদি', 'তাহলে', 'কারণ', 'কিংবা',
            'অথবা', 'এই', 'সেই', 'ওই', 'যে', 'যা', 'যার', 'যাদের', 'এর',
            'তার', 'তাদের', 'আমার', 'আমাদের', 'তুমার', 'তোমাদের', 'তিনি',
            'তারা', 'আমি', 'আমরা', 'তুমি', 'তোমরা', 'সে', 'এক', 'দুই', 'তিন'
        }
        
        # Content quality indicators
        self.quality_indicators = [
            'উৎস:', 'সূত্র:', 'তথ্যসূত্র:', 'গ্রন্থ:', 'কাব্য:', 'উপন্যাস:',
            'কবি:', 'লেখক:', 'রচনাকাল:', 'প্রকাশকাল:', 'সাল', 'খ্রিস্টাব্দ',
            'গ্রন্থাকারে', 'প্রকাশিত', 'রচিত', 'অনূদিত', 'সম্পাদিত'
        ]
    
    def clean_html(self, text: str) -> str:
        """
        🧹 Remove HTML tags and normalize content
        
        Args:
            text: Raw text that may contain HTML
            
        Returns:
            Clean text without HTML tags
            
        Why needed: Dataset has HTML formatting mixed with Bengali text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: HTML decode entities
        text = html.unescape(text)
        
        # Step 2: Use BeautifulSoup for robust HTML parsing
        try:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text(separator=' ')
        except Exception as e:
            logger.warning(f"BeautifulSoup parsing failed: {e}")
            # Fallback to regex-based cleaning
        
        # Step 3: Apply HTML cleaning patterns
        for pattern, replacement in self.html_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def normalize_unicode(self, text: str) -> str:
        """
        🔧 Normalize Bengali Unicode for consistent matching
        
        Args:
            text: Text with potential Unicode inconsistencies
            
        Returns:
            Normalized text
            
        Why crucial: Bengali has multiple Unicode representations for same visual character
        Example: 'কি' vs 'কী' - both valid but different Unicode points
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization (NFC - Canonical Decomposition + Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Apply Bengali-specific patterns
        for pattern, replacement in self.unicode_patterns:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        # Step 3: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _convert_bengali_digits(self, match) -> str:
        """Convert Bengali numerals to English for consistency"""
        bengali_digits = '০১২৩৪৫৬৭৮৯'
        english_digits = '0123456789'
        digit = match.group()
        return english_digits[bengali_digits.index(digit)]
    
    def tokenize_bengali(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """
        ✂️ Advanced Bengali tokenization
        
        Args:
            text: Bengali text to tokenize
            remove_stopwords: Whether to remove common Bengali stopwords
            
        Returns:
            List of tokens
            
        Why custom tokenizer: Standard tokenizers don't handle Bengali properly
        """
        if not text:
            return []
        
        # Step 1: Basic word boundary tokenization
        # Bengali doesn't use spaces consistently, so we use punctuation as boundaries
        tokens = re.findall(r'[^\s\।\,\;\:\!\?\.\(\)\[\]\{\}]+', text)
        
        # Step 2: Clean each token
        cleaned_tokens = []
        for token in tokens:
            token = token.strip()
            if len(token) >= 2:  # Filter very short tokens
                cleaned_tokens.append(token)
        
        # Step 3: Remove stopwords if requested
        if remove_stopwords:
            cleaned_tokens = [token for token in cleaned_tokens 
                            if token.lower() not in self.bengali_stopwords]
        
        return cleaned_tokens
    
    def assess_content_quality(self, text: str) -> float:
        """
        📊 Assess text quality for prioritization (0.0 to 1.0)
        
        Args:
            text: Text to assess
            
        Returns:
            Quality score between 0.0 and 1.0
            
        Purpose: Help prioritize high-quality content in search results
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        quality_score = 0.0
        
        # Length-based scoring (longer generally means more detailed)
        length = len(text)
        if length > 50:
            quality_score += 0.2
        if length > 150:
            quality_score += 0.2
        if length > 300:
            quality_score += 0.1
        
        # Quality indicator presence
        indicator_count = sum(1 for indicator in self.quality_indicators 
                            if indicator in text)
        quality_score += min(indicator_count * 0.1, 0.3)
        
        # Sentence structure (well-formed content has multiple sentences)
        sentence_count = len([s for s in text.split('।') if s.strip()])
        if sentence_count >= 2:
            quality_score += 0.1
        if sentence_count >= 4:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def create_searchable_text(self, question: str, explanation: str, 
                             options: List[str], answer: str = "") -> str:
        """
        🔍 Create comprehensive searchable text from all components
        
        Args:
            question: Main question text
            explanation: Detailed explanation
            options: Multiple choice options
            answer: Correct answer
            
        Returns:
            Combined searchable text
            
        Purpose: Combine all relevant text for maximum search coverage
        """
        text_parts = []
        
        # Add question (highest priority)
        if question and question.strip():
            text_parts.append(question.strip())
        
        # Add explanation (important context)
        if explanation and explanation.strip() and len(explanation.strip()) > 10:
            text_parts.append(explanation.strip())
        
        # Add options (important for matching)
        for option in options:
            if option and option.strip():
                # Remove option prefixes like "ক)", "খ)" etc.
                clean_option = re.sub(r'^[ক-ৎ]\)\s*', '', option.strip())
                if clean_option:
                    text_parts.append(clean_option)
        
        # Add answer if available
        if answer and answer.strip():
            text_parts.append(answer.strip())
        
        return ' '.join(text_parts)
    
    def process_full_text(self, text: str) -> Dict[str, any]:
        """
        🏭 Complete text processing pipeline
        
        Args:
            text: Raw text input
            
        Returns:
            Dictionary with processed versions and metadata
            
        Purpose: One-stop processing for all text cleaning needs
        """
        if not text:
            return {
                'original': '',
                'cleaned': '',
                'normalized': '',
                'tokens': [],
                'quality_score': 0.0,
                'length': 0,
                'has_html': False
            }
        
        original_text = str(text)
        
        # Processing pipeline
        html_cleaned = self.clean_html(original_text)
        normalized = self.normalize_unicode(html_cleaned)
        tokens = self.tokenize_bengali(normalized)
        quality = self.assess_content_quality(normalized)
        
        return {
            'original': original_text,
            'cleaned': html_cleaned,
            'normalized': normalized,
            'tokens': tokens,
            'quality_score': quality,
            'length': len(normalized),
            'has_html': '<' in original_text and '>' in original_text,
            'token_count': len(tokens)
        }

# Global text processor instance
text_processor = BengaliTextProcessor()

def preprocess_text_batch(texts: List[str]) -> List[Dict[str, any]]:
    """
    ⚡ Batch process multiple texts efficiently
    
    Args:
        texts: List of texts to process
        
    Returns:
        List of processing results
        
    Purpose: Efficient batch processing for large datasets
    """
    return [text_processor.process_full_text(text) for text in texts]

if __name__ == "__main__":
    """Test the text processor"""
    
    # Test cases
    test_texts = [
        'রবীন্দ্রনাথ ঠাকুর <strong>গীতাঞ্জলি</strong> কাব্যগ্রন্থের জন্য বিখ্যাত।',
        '<br />তিনি ১৯১৩ সালে নোবেল পুরস্কার পান।&nbsp;',
        'ভানুসিংহ ঠাকুর তাঁর ছদ্মনাম ছিল।'
    ]
    
    print("🧪 Testing Bengali Text Processor...")
    
    for i, text in enumerate(test_texts, 1):
        result = text_processor.process_full_text(text)
        print(f"\nTest {i}:")
        print(f"Original: {result['original']}")
        print(f"Cleaned: {result['cleaned']}")
        print(f"Normalized: {result['normalized']}")
        print(f"Tokens: {result['tokens']}")
        print(f"Quality: {result['quality_score']:.2f}")
