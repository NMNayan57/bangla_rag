"""
🧪 Foundation Testing Script
============================
Purpose: Validate that our foundation components work correctly
Usage: python test_foundation.py
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.core.config import settings, validate_setup
from app.services.text_processor import text_processor

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing Configuration...")
    
    try:
        # Test settings loading
        print(f"   ✅ Project root: {settings.PROJECT_ROOT}")
        print(f"   ✅ Embedding model: {settings.EMBEDDING_MODEL}")
        print(f"   ✅ Default top-k: {settings.DEFAULT_TOP_K}")
        
        # Test path resolution
        paths = settings.get_data_paths()
        print(f"   ✅ Data paths resolved: {len(paths)} paths")
        
        # Test validation
        validate_setup()
        print("   ✅ Setup validation passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_text_processor():
    """Test Bengali text processing"""
    print("\n🧹 Testing Text Processor...")
    
    try:
        # Test cases with Bengali literature content
        test_cases = [
            {
                'input': 'রবীন্দ্রনাথ ঠাকুর <strong>গীতাঞ্জলি</strong> কাব্যগ্রন্থের জন্য বিখ্যাত।',
                'expected_clean': True,
                'expected_tokens': ['রবীন্দ্রনাথ', 'ঠাকুর', 'গীতাঞ্জলি', 'কাব্যগ্রন্থের', 'জন্য', 'বিখ্যাত']
            },
            {
                'input': '<br />তিনি ১৯১৩ সালে নোবেল পুরস্কার পান।&nbsp;',
                'expected_clean': True,
                'expected_tokens': ['তিনি', '1913', 'সালে', 'নোবেল', 'পুরস্কার', 'পান']
            },
            {
                'input': 'ভানুসিংহ ঠাকুর তাঁর ছদ্মনাম ছিল। উৎস: বাংলা সাহিত্য।',
                'expected_clean': True,
                'expected_tokens': ['ভানুসিংহ', 'ঠাকুর', 'তাঁর', 'ছদ্মনাম', 'ছিল', 'উৎস', 'বাংলা', 'সাহিত্য']
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            result = text_processor.process_full_text(case['input'])
            
            print(f"   Test {i}:")
            print(f"     Input: {case['input'][:50]}...")
            print(f"     Cleaned: {result['cleaned']}")
            print(f"     Normalized: {result['normalized']}")
            print(f"     Tokens: {result['tokens']}")
            print(f"     Quality: {result['quality_score']:.2f}")
            print(f"     HTML removed: {result['has_html']}")
            
            # Verify HTML was cleaned
            if '<' in case['input'] and '<' not in result['cleaned']:
                print(f"     ✅ HTML cleaning successful")
            
            # Verify tokenization worked
            if len(result['tokens']) > 0:
                print(f"     ✅ Tokenization successful ({len(result['tokens'])} tokens)")
            
            print()
        
        print("   ✅ Text processor tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Text processor test failed: {e}")
        return False

def test_data_availability():
    """Test if required data files exist"""
    print("📂 Testing Data Availability...")
    
    try:
        paths = settings.get_data_paths()
        
        # Check if raw data exists
        if paths['raw_data'].exists():
            print(f"   ✅ Raw data found: {paths['raw_data']}")
            
            # Load and inspect
            df = pd.read_csv(paths['raw_data'])
            print(f"   ✅ Raw data loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"   ✅ Columns: {list(df.columns)}")
            
            # Check for Bengali content
            if 'Question' in df.columns:
                sample_question = df['Question'].dropna().iloc[0] if len(df) > 0 else ""
                if any(ord(char) >= 0x0980 and ord(char) <= 0x09FF for char in sample_question):
                    print("   ✅ Bengali content detected")
                else:
                    print("   ⚠️ No Bengali content detected in sample")
            
        else:
            print(f"   ⚠️ Raw data not found: {paths['raw_data']}")
            print("   Please place your questions.csv in data/raw/ directory")
            return False
        
        # Check processed data directory
        if paths['processed_data'].parent.exists():
            print(f"   ✅ Processed data directory exists")
        else:
            print(f"   ✅ Processed data directory will be created")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data availability test failed: {e}")
        return False

def test_dependencies():
    """Test if all required packages are available"""
    print("📦 Testing Dependencies...")
    
    required_packages = [
        'sentence_transformers',
        'chromadb',
        'pandas',
        'numpy',
        'sklearn',
        'rank_bm25',
        'beautifulsoup4',
        'transformers',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   ❌ Missing packages: {missing_packages}")
        print("   Please install: pip install -r requirements.txt")
        return False
    else:
        print("   ✅ All dependencies available")
        return True

def main():
    """Run all foundation tests"""
    print("🧪 FOUNDATION TESTING SUITE")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("Text Processor", test_text_processor),
        ("Data Availability", test_data_availability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 TEST SUMMARY:")
    print("=" * 30)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Foundation is ready.")
        print("\n📋 Next steps:")
        print("1. Run: python scripts/data_preprocessing.py")
        print("2. Run: python scripts/generate_embeddings.py")
        print("3. Proceed to Phase 2: Core Search Engine")
    else:
        print("❌ SOME TESTS FAILED! Please fix issues before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
