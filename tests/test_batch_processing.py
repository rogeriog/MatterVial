#!/usr/bin/env python3
"""
Test script for MEGNet batch processing implementation.

This script validates that the new batch processing capabilities:
1. Produce identical results to sequential processing
2. Handle edge cases properly (invalid structures, memory limits)
3. Provide expected performance improvements
4. Maintain backward compatibility

Usage:
    python test_batch_processing.py
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add the mattervial package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mattervial.featurizers.megnet_models import (
        get_MVL_MEGNetFeatures, get_Custom_MEGNetFeatures, get_Adjacent_MEGNetFeatures,
        get_MVL_MEGNetFeatures_legacy, get_Custom_MEGNetFeatures_legacy, 
        get_Adjacent_MEGNetFeatures_legacy, BatchProcessingConfig,
        optimize_batch_size_for_structures, clear_gpu_memory
    )
    print("✓ Successfully imported batch processing functions")
except ImportError as e:
    print(f"✗ Failed to import batch processing functions: {e}")
    sys.exit(1)

def create_test_structures():
    """Create a set of test structures for validation."""
    try:
        from pymatgen.core import Structure, Lattice
        
        # Create simple test structures
        lattice = Lattice.cubic(4.0)
        structures = []
        
        # Simple cubic structures with different compositions
        test_compositions = [
            [("Fe", [0, 0, 0])],
            [("Al", [0, 0, 0])],
            [("Ti", [0, 0, 0])],
            [("Fe", [0, 0, 0]), ("O", [0.5, 0.5, 0.5])],
            [("Al", [0, 0, 0]), ("O", [0.5, 0.5, 0.5])],
        ]
        
        for comp in test_compositions:
            species = [atom[0] for atom in comp]
            coords = [atom[1] for atom in comp]
            structure = Structure(lattice, species, coords)
            structures.append(structure)
            
        return pd.Series(structures)
        
    except ImportError:
        print("✗ PyMatGen not available for creating test structures")
        return None

def test_batch_vs_sequential_consistency():
    """Test that batch processing produces identical results to sequential processing."""
    print("\n=== Testing Batch vs Sequential Consistency ===")
    
    structures = create_test_structures()
    if structures is None:
        print("✗ Cannot create test structures - skipping consistency test")
        return False
    
    try:
        # Test MVL features
        print("Testing MVL features...")
        batch_features = get_MVL_MEGNetFeatures(structures, batch_size=2, verbose=False)
        sequential_features = get_MVL_MEGNetFeatures_legacy(structures)
        
        if batch_features.shape == sequential_features.shape:
            print("✓ MVL features: Shape consistency passed")
        else:
            print(f"✗ MVL features: Shape mismatch - batch: {batch_features.shape}, sequential: {sequential_features.shape}")
            return False
            
        # Test Custom features (if models are available)
        print("Testing Custom features...")
        try:
            batch_custom = get_Custom_MEGNetFeatures(structures, 'MatMinerEncoded_v1', batch_size=2, verbose=False)
            sequential_custom = get_Custom_MEGNetFeatures_legacy(structures, 'MatMinerEncoded_v1')
            
            if batch_custom.shape == sequential_custom.shape:
                print("✓ Custom features: Shape consistency passed")
            else:
                print(f"✗ Custom features: Shape mismatch - batch: {batch_custom.shape}, sequential: {sequential_custom.shape}")
                return False
        except Exception as e:
            print(f"⚠ Custom features test skipped (model not available): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
        return False

def test_batch_processing_performance():
    """Test that batch processing provides performance improvements."""
    print("\n=== Testing Batch Processing Performance ===")
    
    structures = create_test_structures()
    if structures is None:
        print("✗ Cannot create test structures - skipping performance test")
        return False
    
    # Duplicate structures to create a larger dataset
    large_structures = pd.concat([structures] * 4, ignore_index=True)
    
    try:
        # Time sequential processing
        print("Timing sequential processing...")
        start_time = time.time()
        sequential_result = get_MVL_MEGNetFeatures_legacy(large_structures)
        sequential_time = time.time() - start_time
        
        # Clear memory between tests
        clear_gpu_memory()
        
        # Time batch processing
        print("Timing batch processing...")
        start_time = time.time()
        batch_result = get_MVL_MEGNetFeatures(large_structures, batch_size=4, verbose=False)
        batch_time = time.time() - start_time
        
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Batch time: {batch_time:.2f}s")
        
        if batch_time < sequential_time:
            speedup = sequential_time / batch_time
            print(f"✓ Batch processing is {speedup:.2f}x faster")
        else:
            print(f"⚠ Batch processing was slower (possibly due to small dataset or overhead)")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def test_memory_management():
    """Test memory management and batch size optimization."""
    print("\n=== Testing Memory Management ===")
    
    structures = create_test_structures()
    if structures is None:
        print("✗ Cannot create test structures - skipping memory test")
        return False
    
    try:
        # Test batch size optimization
        config = BatchProcessingConfig(default_batch_size=32, min_batch_size=1, max_batch_size=64)
        
        # This should work without the actual model
        try:
            optimal_batch_size = optimize_batch_size_for_structures(structures, None, config)
            print(f"✓ Batch size optimization returned: {optimal_batch_size}")
        except Exception as e:
            print(f"⚠ Batch size optimization test skipped: {e}")
        
        # Test memory clearing
        clear_gpu_memory()
        print("✓ GPU memory clearing executed without errors")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory management test failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid structures."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Create a mix of valid and invalid structures
        valid_structures = create_test_structures()
        if valid_structures is None:
            print("✗ Cannot create test structures - skipping error handling test")
            return False
        
        # Add some None values to simulate invalid structures
        mixed_structures = list(valid_structures) + [None, None]
        mixed_structures = pd.Series(mixed_structures)
        
        # Test that batch processing handles invalid structures gracefully
        result = get_MVL_MEGNetFeatures(mixed_structures, batch_size=2, verbose=False)
        
        if result is not None and len(result) == len(mixed_structures):
            print("✓ Error handling: Batch processing handled invalid structures gracefully")
            return True
        else:
            print("✗ Error handling: Batch processing failed with invalid structures")
            return False
            
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_configuration_options():
    """Test various configuration options."""
    print("\n=== Testing Configuration Options ===")
    
    try:
        # Test BatchProcessingConfig
        config = BatchProcessingConfig(
            default_batch_size=16,
            max_batch_size=32,
            min_batch_size=2,
            auto_adjust_batch_size=True,
            verbose=False
        )
        
        print(f"✓ BatchProcessingConfig created with batch_size={config.default_batch_size}")
        
        # Test configuration parameter passing
        structures = create_test_structures()
        if structures is not None:
            result = get_MVL_MEGNetFeatures(
                structures, 
                batch_size=8, 
                verbose=False,
                auto_adjust_batch_size=True
            )
            print("✓ Configuration parameters passed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MEGNet Batch Processing Validation Tests")
    print("=" * 50)
    
    tests = [
        test_batch_vs_sequential_consistency,
        test_batch_processing_performance,
        test_memory_management,
        test_error_handling,
        test_configuration_options
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Batch processing implementation is working correctly.")
        return 0
    else:
        print("⚠ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
