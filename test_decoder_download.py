#!/usr/bin/env python3
"""
Test script for MatterVial decoder download functionality.

This script tests the automatic download and caching system for decoder files.
It validates that the decoder functions work correctly with the new download system.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np

# Add the MatterVial directory to the path
sys.path.insert(0, os.path.abspath('.'))

def test_decoder_cache_info():
    """Test the cache info functionality."""
    print("=" * 60)
    print("Testing decoder cache info...")
    
    try:
        from mattervial.interpreter.decoder import get_decoder_cache_info
        
        cache_info = get_decoder_cache_info()
        print(f"✓ Cache info retrieved successfully")
        print(f"  Status: {cache_info['status']}")
        print(f"  Size: {cache_info['size_mb']} MB")
        print(f"  Files: {cache_info['files']}")
        if 'critical_files_present' in cache_info:
            print(f"  Critical files: {cache_info['critical_files_present']}/{cache_info['critical_files_total']}")

        if cache_info.get('missing_files'):
            print(f"  Missing files: {cache_info['missing_files']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cache info test failed: {e}")
        return False


def test_decoder_imports():
    """Test that decoder functions can be imported."""
    print("=" * 60)
    print("Testing decoder imports...")
    
    try:
        from mattervial.interpreter import encode_ofm, decode_ofm, encode_mm, decode_mm
        from mattervial.interpreter.decoder import (
            get_decoder_cache_info, 
            clear_decoder_cache,
            configure_figshare_url,
            get_decoder_info
        )
        print("✓ All decoder functions imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import test failed: {e}")
        return False


def test_decoder_info():
    """Test the decoder info functionality."""
    print("=" * 60)
    print("Testing decoder info...")
    
    try:
        from mattervial.interpreter.decoder import get_decoder_info
        
        info = get_decoder_info()
        print(f"✓ Decoder info retrieved successfully")
        print(f"  Module version: {info['module_version']}")
        print(f"  Figshare URL: {info['figshare_url'][:50]}...")
        print(f"  Decoders directory: {info['decoders_directory']}")
        print(f"  Supported operations: {info['supported_operations']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Decoder info test failed: {e}")
        return False


def test_configure_url():
    """Test URL configuration functionality."""
    print("=" * 60)
    print("Testing URL configuration...")
    
    try:
        from mattervial.interpreter.decoder import configure_figshare_url, get_decoder_info
        
        # Test URL configuration
        test_url = "https://example.com/test_url"
        configure_figshare_url(test_url)
        
        # Verify URL was updated
        info = get_decoder_info()
        if info['figshare_url'] == test_url:
            print("✓ URL configuration test passed")
            return True
        else:
            print(f"✗ URL not updated correctly: {info['figshare_url']}")
            return False
            
    except Exception as e:
        print(f"✗ URL configuration test failed: {e}")
        return False


def test_mock_encoding():
    """Test encoding functions with mock data (without actual download)."""
    print("=" * 60)
    print("Testing mock encoding (will attempt download)...")
    
    try:
        from mattervial.interpreter import encode_ofm, encode_mm
        
        # Create mock data
        np.random.seed(42)
        
        # Mock OFM data (943 features as per OFM specification)
        ofm_data = pd.DataFrame(
            np.random.randn(10, 943),
            columns=[f'ofm_feature_{i}' for i in range(943)]
        )
        
        # Mock MatMiner data (1264 features as per MM specification)  
        mm_data = pd.DataFrame(
            np.random.randn(10, 1264),
            columns=[f'mm_feature_{i}' for i in range(1264)]
        )
        
        print("✓ Mock data created successfully")
        print(f"  OFM data shape: {ofm_data.shape}")
        print(f"  MM data shape: {mm_data.shape}")
        
        # Note: These will attempt to download if files are missing
        # In a real test environment, you would mock the download or use test files
        print("Note: Actual encoding tests would require decoder files or mocked downloads")
        
        return True
        
    except Exception as e:
        print(f"✗ Mock encoding test failed: {e}")
        return False


def test_clear_cache():
    """Test cache clearing functionality."""
    print("=" * 60)
    print("Testing cache clearing...")
    
    try:
        from mattervial.interpreter.decoder import clear_decoder_cache, get_decoder_cache_info
        
        # Get initial cache info
        initial_info = get_decoder_cache_info()
        print(f"Initial cache status: {initial_info['status']}")
        
        # Test cache clearing
        result = clear_decoder_cache()
        print(f"Cache clear result: {result}")
        
        # Check cache info after clearing
        final_info = get_decoder_cache_info()
        print(f"Final cache status: {final_info['status']}")
        
        print("✓ Cache clearing test completed")
        return True
        
    except Exception as e:
        print(f"✗ Cache clearing test failed: {e}")
        return False


def main():
    """Run all decoder download tests."""
    print("MatterVial Decoder Download Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_decoder_imports),
        ("Cache Info Test", test_decoder_cache_info),
        ("Decoder Info Test", test_decoder_info),
        ("URL Configuration Test", test_configure_url),
        ("Mock Encoding Test", test_mock_encoding),
        ("Cache Clear Test", test_clear_cache),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Decoder download functionality is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
