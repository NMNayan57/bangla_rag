"""
🎯 GPT-4o Mini Integration Service
=================================
Purpose: Intelligent explanation enhancement for Bengali literature
integrated GPT-4o Mini for context-aware explanation generation"
"""

import asyncio
import time
import hashlib
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

import httpx
from openai import AsyncOpenAI

# Local imports
from app.core.config import settings
from app.models.schemas import QuestionResult

logger = logging.getLogger(__name__)

class GPTService:
    """
    🤖 GPT-4o Mini service for intelligent explanation enhancement
    
    Features:
    - Context-aware Bengali explanation generation
    - Cost tracking and optimization
    - Response caching for efficiency
    - Graceful fallback when GPT fails
    """
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.is_available = False
        
        # Performance optimization: Response caching
        self.response_cache: Dict[str, str] = {}
        self.cache_max_size = 200
        
        # Cost and performance tracking
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'cache_hits': 0,
            'total_response_time': 0,
            'avg_response_time': 0,
            'total_tokens_used': 0,
            'estimated_cost_usd': 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize GPT service if enabled"""
        if not settings.ENABLE_GPT_BONUS or not settings.OPENAI_API_KEY:
            logger.info("🤖 GPT service disabled (no API key or feature disabled)")
            return
        
        try:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Test connection with minimal call
            test_response = await self.client.chat.completions.create(
                model=settings.GPT_MODEL,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            
            self.is_available = True
            logger.info("✅ GPT-4o Mini service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ GPT service initialization failed: {e}")
            self.is_available = False
    
    async def enhance_results(self, query: str, results: List[QuestionResult]) -> List[QuestionResult]:
        """
        🚀 Enhance search results with GPT-4o Mini intelligence
        
        Strategy:
        - Process top 2-3 results only (cost optimization)
        - Generate enhanced explanations for poor/missing content
        - Apply intelligent reranking
        - Mark AI-enhanced results clearly
        """
        if not self.is_available:
            logger.debug("GPT service not available, returning original results")
            return results
        
        if not results:
            return results
        
        logger.info(f"🤖 GPT enhancing top {min(3, len(results))} results")
        
        enhanced_results = []
        enhancement_count = 0
        
        # Process top 3 results for enhancement
        for i, result in enumerate(results[:3]):
            try:
                start_time = time.time()
                
                # Decide if this result needs enhancement
                needs_enhancement = self._needs_enhancement(result)
                
                if needs_enhancement:
                    logger.debug(f"🤖 Enhancing result {i+1}: {result.question[:50]}...")
                    
                    # Generate enhanced explanation
                    enhanced_explanation = await self._generate_explanation(
                        result.question, result.explanation, query
                    )
                    
                    if enhanced_explanation and enhanced_explanation.strip():
                        # Update result with enhanced content
                        result.explanation = enhanced_explanation.strip()
                        result.llm_score = result.relevance_score * 1.15  # 15% boost for AI enhancement
                        result.relevance_score = result.llm_score  # Update primary score
                        enhancement_count += 1
                        
                        processing_time = (time.time() - start_time) * 1000
                        logger.info(f"✅ Enhanced result {i+1} in {processing_time:.1f}ms")
                    else:
                        logger.warning(f"⚠️ GPT enhancement failed for result {i+1}")
                else:
                    logger.debug(f"✓ Result {i+1} has good explanation, skipping GPT")
                
                enhanced_results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Error enhancing result {i+1}: {e}")
                enhanced_results.append(result)  # Keep original on error
        
        # Add remaining results without processing
        enhanced_results.extend(results[3:])
        
        logger.info(f"🎯 GPT enhanced {enhancement_count} results successfully")
        return enhanced_results
    
    def _needs_enhancement(self, result: QuestionResult) -> bool:
        """
        🔍 Determine if a result needs GPT enhancement
        
        Enhancement criteria:
        - Missing or very short explanation
        - Low quality score
        - Generic or placeholder text
        """
        explanation = result.explanation or ""
        
        # Check for missing/poor explanations
        if not explanation or len(explanation.strip()) < 30:
            return True
        
        # Check for placeholder text
        placeholder_indicators = [
            "no detailed explanation available",
            "explanation not provided",
            "তথ্য পাওয়া যায়নি",
            "ব্যাখ্যা নেই"
        ]
        
        explanation_lower = explanation.lower()
        if any(indicator in explanation_lower for indicator in placeholder_indicators):
            return True
        
        # Check quality score
        if result.quality_score < 0.4:
            return True
        
        return False
    
    async def _generate_explanation(self, question: str, original_explanation: str, 
                                  user_query: str) -> Optional[str]:
        """
        💡 Generate enhanced explanation using GPT-4o Mini
        
        Context strategy:
        - Include user query for relevance
        - Use original explanation as base (if available)
        - Generate Bengali-focused content
        """
        # Check cache first
        cache_key = hashlib.md5(
            f"{question}_{user_query}_{original_explanation[:50]}".encode()
        ).hexdigest()
        
        if cache_key in self.response_cache:
            self.stats['cache_hits'] += 1
            return self.response_cache[cache_key]
        
        # Prepare context-aware prompt
        prompt = self._create_enhancement_prompt(question, original_explanation, user_query)
        
        # Call GPT
        enhanced_text = await self._call_gpt(prompt)
        
        # Cache successful results
        if enhanced_text and len(self.response_cache) < self.cache_max_size:
            self.response_cache[cache_key] = enhanced_text
        
        return enhanced_text
    
    def _create_enhancement_prompt(self, question: str, original: str, query: str) -> str:
        """Create context-aware prompt for explanation enhancement"""
        
        base_prompt = f"""আপনি একজন বাংলা সাহিত্য বিশেষজ্ঞ। নিচের প্রশ্নের জন্য একটি সংক্ষিপ্ত কিন্তু তথ্যবহুল ব্যাখ্যা দিন:

প্রশ্ন: {question}
ব্যবহারকারীর অনুসন্ধান: {query}"""
        
        if original and len(original.strip()) > 10:
            base_prompt += f"\nবর্তমান ব্যাখ্যা: {original}\n\nবর্তমান ব্যাখ্যাটি উন্নত করুন এবং আরও প্রাসঙ্গিক তথ্য যোগ করুন:"
        else:
            base_prompt += f"\n\nএই প্রশ্নের জন্য একটি সংক্ষিপ্ত কিন্তু সম্পূর্ণ ব্যাখ্যা দিন:"
        
        base_prompt += """

নির্দেশনা:
- শুধুমাত্র বাংলায় উত্তর দিন
- ২-৩ বাক্যে সীমাবদ্ধ রাখুন
- প্রাসঙ্গিক তথ্য ও প্রসঙ্গ অন্তর্ভুক্ত করুন
- সহজ ও স্পষ্ট ভাষা ব্যবহার করুন

ব্যাখ্যা:"""
        
        return base_prompt
    
    async def _call_gpt(self, prompt: str) -> Optional[str]:
        """Make GPT API call with error handling and cost tracking"""
        if not self.client:
            return None
        
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        try:
            response = await self.client.chat.completions.create(
                model=settings.GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "আপনি একজন বাংলা সাহিত্যের বিশেষজ্ঞ। সংক্ষিপ্ত, তথ্যবহুল এবং প্রাসঙ্গিক উত্তর দিন।"
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.GPT_MAX_TOKENS,
                temperature=settings.GPT_TEMPERATURE,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Extract and validate response
            result = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            # Update statistics
            response_time = time.time() - start_time
            self.stats['successful_calls'] += 1
            self.stats['total_response_time'] += response_time
            self.stats['avg_response_time'] = self.stats['total_response_time'] / self.stats['successful_calls']
            self.stats['total_tokens_used'] += tokens_used
            
            # Calculate cost (GPT-4o Mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens)
            # Simplified estimation
            cost_per_token = 0.0000375  # Average cost estimate
            self.stats['estimated_cost_usd'] += tokens_used * cost_per_token
            
            logger.debug(f"⚡ GPT response in {response_time*1000:.1f}ms, {tokens_used} tokens")
            
            return result
            
        except Exception as e:
            self.stats['failed_calls'] += 1
            logger.error(f"❌ GPT API call failed: {e}")
            return None
    
    async def rerank_results(self, query: str, results: List[QuestionResult]) -> List[QuestionResult]:
        """
        📈 Apply intelligent reranking using GPT insights
        
        Reranking strategy:
        - Boost GPT-enhanced results
        - Consider content quality
        - Maintain relevance order
        """
        if not results:
            return results
        
        logger.debug("📈 Applying GPT-powered reranking...")
        
        # Apply scoring boosts
        for result in results:
            original_score = result.relevance_score
            
            # GPT enhancement boost (already applied above, but we can add more context)
            if result.llm_score and result.llm_score > original_score:
                # Additional small boost for GPT-enhanced content
                result.relevance_score = min(result.relevance_score * 1.05, 1.0)
            
            # Quality boost
            if result.quality_score > 0.8:
                result.relevance_score = min(result.relevance_score * 1.02, 1.0)
            
            # Explanation length boost (comprehensive explanations are better)
            if result.explanation and len(result.explanation) > 200:
                result.relevance_score = min(result.relevance_score * 1.01, 1.0)
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.debug("✅ GPT reranking completed")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        success_rate = (self.stats['successful_calls'] / max(1, self.stats['total_calls'])) * 100
        cache_hit_rate = (self.stats['cache_hits'] / max(1, self.stats['total_calls'])) * 100
        
        return {
            **self.stats,
            'success_rate_percent': f"{success_rate:.1f}%",
            'cache_hit_rate_percent': f"{cache_hit_rate:.1f}%",
            'cache_size': len(self.response_cache),
            'avg_tokens_per_call': self.stats['total_tokens_used'] / max(1, self.stats['successful_calls']),
            'cost_per_call_usd': self.stats['estimated_cost_usd'] / max(1, self.stats['successful_calls']),
            'is_available': self.is_available,
            'model': settings.GPT_MODEL
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information"""
        return {
            "enabled": settings.ENABLE_GPT_BONUS,
            "available": self.is_available,
            "model": settings.GPT_MODEL,
            "features": {
                "explanation_enhancement": True,
                "intelligent_reranking": True,
                "cost_tracking": True,
                "response_caching": True
            },
            "performance": self.get_stats()
        }

# Global service instance
gpt_service = GPTService()
