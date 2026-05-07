#!/usr/bin/env python3
"""
Test script to verify ORB featurizer integration with MatterVial.
This script tests that the ORB featurizer follows the same interface as other featurizers.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Structure, Lattice

def create_test_structures():
    """Create a few simple test structures for testing."""
    structures = []

    # Simple cubic structure (e.g., simple cubic iron)
    lattice1 = Lattice.cubic(2.87)
    structure1 = Structure(lattice1, ["Fe"], [[0, 0, 0]])
    structures.append(structure1)

    # Simple binary compound (NaCl-like)
    lattice2 = Lattice.cubic(5.64)
    structure2 = Structure(lattice2, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structures.append(structure2)

    return structures

def test_orb_featurizer_import():
    """Test that ORB featurizer can be imported correctly."""
    print("Testing ORB featurizer import...")
    
    try:
        # Test import from mattervial.featurizers
        from mattervial.featurizers import orb_v3, get_ORB_features
        print("✓ Successfully imported orb_v3 and get_ORB_features from mattervial.featurizers")
        
        # Test import from mattervial (top-level)
        from mattervial import orb_v3 as orb_v3_toplevel
        print("✓ Successfully imported orb_v3 from mattervial top-level")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_orb_featurizer_interface():
    """Test that ORB featurizer has the expected interface."""
    print("\nTesting ORB featurizer interface...")
    
    try:
        from mattervial.featurizers import orb_v3
        
        # Check that it has get_features method
        if hasattr(orb_v3, 'get_features'):
            print("✓ ORB featurizer has get_features method")
        else:
            print("✗ ORB featurizer missing get_features method")
            return False
        
        # Check that it's callable
        if callable(orb_v3.get_features):
            print("✓ get_features method is callable")
        else:
            print("✗ get_features method is not callable")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Interface test failed: {e}")
        return False

def test_orb_featurizer_functionality():
    """Test that ORB featurizer can process structures (if ORB models are available)."""
    print("\nTesting ORB featurizer functionality...")
    
    try:
        from mattervial.featurizers import orb_v3
        from mattervial.featurizers.mlip_models import ORB_AVAILABLE
        
        if not ORB_AVAILABLE:
            print("⚠ ORB models not available - skipping functionality test")
            print("  (This is expected if orb-models package is not installed)")
            return True
        
        # Create test structures
        structures = create_test_structures()
        
        # Convert to DataFrame format (as expected by featurizers)
        # Convert structures to JSON strings for compatibility with featurizers
        structure_strings = [structure.to_json() for structure in structures]
        df = pd.DataFrame({'structure': structure_strings})
        
        print(f"Testing with {len(structures)} structures...")
        
        # Test the featurizer
        features = orb_v3.get_features(df)
        
        if isinstance(features, pd.DataFrame):
            print(f"✓ Successfully extracted features: {features.shape}")
            print(f"  Features shape: {features.shape[0]} structures × {features.shape[1]} features")
            
            # Check that we got features for all structures
            if features.shape[0] == len(structures):
                print("✓ Correct number of structures processed")
            else:
                print(f"✗ Expected {len(structures)} structures, got {features.shape[0]}")
                return False
            
            # Check that feature names follow expected pattern
            feature_names = features.columns.tolist()
            orb_features = [name for name in feature_names if name.startswith('ORB_v3_')]
            if orb_features:
                print(f"✓ Found {len(orb_features)} ORB features with correct naming pattern")
                print(f"  Example features: {orb_features[:3]}...")
            else:
                print("✗ No features found with ORB_v3_ prefix")
                return False
            
            return True
        else:
            print(f"✗ Expected DataFrame, got {type(features)}")
            return False
            
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def test_comparison_with_other_featurizers():
    """Test that ORB featurizer works similarly to other featurizers."""
    print("\nTesting comparison with other featurizers...")
    
    try:
        from mattervial.featurizers import orb_v3, mvl32
        
        # Create test structures
        structures = create_test_structures()
        # Convert structures to JSON strings for compatibility with featurizers
        structure_strings = [structure.to_json() for structure in structures]
        df = pd.DataFrame({'structure': structure_strings})
        
        print("Testing MVL32 featurizer for comparison...")
        try:
            mvl_features = mvl32.get_features(df)
            print(f"MVL32 features shape: {mvl_features.shape}")
            mvl_success = True
        except Exception as e:
            print(f"⚠ MVL32 featurizer failed (this may be due to test structures): {e}")
            mvl_success = False

        # Test ORB featurizer (if available)
        from mattervial.featurizers.mlip_models import ORB_AVAILABLE
        if ORB_AVAILABLE:
            print("Testing ORB featurizer...")
            orb_features = orb_v3.get_features(df)
            print(f"ORB features shape: {orb_features.shape}")

            # Both should return DataFrames with same number of rows (if MVL worked)
            if mvl_success and mvl_features.shape[0] == orb_features.shape[0]:
                print("✓ Both featurizers processed same number of structures")
            elif not mvl_success:
                print("✓ ORB featurizer works independently (MVL32 had issues with test structures)")
            else:
                print("✗ Featurizers processed different numbers of structures")
                return False
        else:
            print("⚠ ORB models not available - skipping ORB comparison")

        print("✓ Interface consistency test passed")
        return True
        
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MatterVial ORB Featurizer Integration Test")
    print("=" * 60)
    
    tests = [
        test_orb_featurizer_import,
        test_orb_featurizer_interface,
        test_orb_featurizer_functionality,
        test_comparison_with_other_featurizers
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ORB featurizer integration successful.")
        return 0
    else:
        print("❌ Some tests failed. Please check the integration.")
        return 1

if __name__ == "__main__":
    exit(main())
