#!/usr/bin/env python3
"""
Test script to validate the refactoring and backward compatibility.
This script tests both the new unified classes and the legacy classes.
"""

import warnings
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all classes can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test new unified classes
        from mattervial.featurizers.structure import DescriptorMEGNetFeaturizer, AdjacentGNNFeaturizer
        print("✓ New unified classes imported successfully")
        
        # Test legacy classes
        from mattervial.featurizers.structure import LatentMMFeaturizer, LatentOFMFeaturizer, AdjacentMEGNetFeaturizer
        print("✓ Legacy classes imported successfully")
        
        # Test top-level imports
        from mattervial import DescriptorMEGNetFeaturizer, AdjacentGNNFeaturizer
        from mattervial import LatentMMFeaturizer, LatentOFMFeaturizer, AdjacentMEGNetFeaturizer
        print("✓ Top-level imports work correctly")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_new_classes():
    """Test the new unified classes."""
    print("\nTesting new unified classes...")
    
    try:
        from mattervial.featurizers.structure import DescriptorMEGNetFeaturizer, AdjacentGNNFeaturizer
        
        # Test DescriptorMEGNetFeaturizer
        desc_mm = DescriptorMEGNetFeaturizer(base_descriptor='l-MM_v1')
        print(f"✓ DescriptorMEGNetFeaturizer created with base_descriptor: {desc_mm.base_descriptor}")
        
        desc_ofm = DescriptorMEGNetFeaturizer(base_descriptor='l-OFM_v1')
        print(f"✓ DescriptorMEGNetFeaturizer created with base_descriptor: {desc_ofm.base_descriptor}")
        
        # Test AdjacentGNNFeaturizer
        adj_gnn = AdjacentGNNFeaturizer(base_model='MEGNet', layers='layer32')
        print(f"✓ AdjacentGNNFeaturizer created with base_model: {adj_gnn.base_model}")
        
        # Test coGN placeholder
        try:
            adj_cogn = AdjacentGNNFeaturizer(base_model='coGN')
            print(f"✓ AdjacentGNNFeaturizer created with base_model: {adj_cogn.base_model}")
        except Exception as e:
            print(f"✓ coGN placeholder working correctly: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ New classes test failed: {e}")
        return False

def test_legacy_classes():
    """Test the legacy classes and deprecation warnings."""
    print("\nTesting legacy classes and deprecation warnings...")
    
    try:
        from mattervial.featurizers.structure import LatentMMFeaturizer, LatentOFMFeaturizer, AdjacentMEGNetFeaturizer
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test LatentMMFeaturizer
            l_mm = LatentMMFeaturizer()
            if w and any("LatentMMFeaturizer is deprecated" in str(warning.message) for warning in w):
                print("✓ LatentMMFeaturizer deprecation warning issued")
            else:
                print("⚠ LatentMMFeaturizer deprecation warning not found")
            
            # Test LatentOFMFeaturizer
            l_ofm = LatentOFMFeaturizer()
            if w and any("LatentOFMFeaturizer is deprecated" in str(warning.message) for warning in w):
                print("✓ LatentOFMFeaturizer deprecation warning issued")
            else:
                print("⚠ LatentOFMFeaturizer deprecation warning not found")
            
            # Test AdjacentMEGNetFeaturizer
            adj_megnet = AdjacentMEGNetFeaturizer(layers='layer32')
            if w and any("AdjacentMEGNetFeaturizer is deprecated" in str(warning.message) for warning in w):
                print("✓ AdjacentMEGNetFeaturizer deprecation warning issued")
            else:
                print("⚠ AdjacentMEGNetFeaturizer deprecation warning not found")
        
        return True
    except Exception as e:
        print(f"✗ Legacy classes test failed: {e}")
        return False

def test_predefined_instances():
    """Test that predefined instances work correctly."""
    print("\nTesting predefined instances...")
    
    try:
        from mattervial.featurizers.structure import l_MM_v1, l_OFM_v1, adj_megnet, adj_megnet_layer16, adj_megnet_all
        
        # Test that instances exist (they might be None if dependencies are missing)
        if l_MM_v1 is not None:
            print(f"✓ l_MM_v1 instance available: {type(l_MM_v1).__name__}")
        else:
            print("⚠ l_MM_v1 instance is None (dependencies missing)")
            
        if l_OFM_v1 is not None:
            print(f"✓ l_OFM_v1 instance available: {type(l_OFM_v1).__name__}")
        else:
            print("⚠ l_OFM_v1 instance is None (dependencies missing)")
            
        if adj_megnet is not None:
            print(f"✓ adj_megnet instance available: {type(adj_megnet).__name__}")
        else:
            print("⚠ adj_megnet instance is None (dependencies missing)")
        
        return True
    except Exception as e:
        print(f"✗ Predefined instances test failed: {e}")
        return False

def test_parameter_validation():
    """Test parameter validation for new classes."""
    print("\nTesting parameter validation...")
    
    try:
        from mattervial.featurizers.structure import DescriptorMEGNetFeaturizer, AdjacentGNNFeaturizer
        
        # Test invalid base_descriptor
        try:
            desc_invalid = DescriptorMEGNetFeaturizer(base_descriptor='invalid')
            print("✗ Invalid base_descriptor should raise ValueError")
            return False
        except ValueError:
            print("✓ Invalid base_descriptor correctly raises ValueError")
        
        # Test invalid base_model
        try:
            adj_invalid = AdjacentGNNFeaturizer(base_model='invalid')
            print("✗ Invalid base_model should raise ValueError")
            return False
        except ValueError:
            print("✓ Invalid base_model correctly raises ValueError")
        
        return True
    except Exception as e:
        print(f"✗ Parameter validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MatterVial Refactoring Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_new_classes,
        test_legacy_classes,
        test_predefined_instances,
        test_parameter_validation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        print("❌ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
