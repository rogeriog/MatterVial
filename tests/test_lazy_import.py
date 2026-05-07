#!/usr/bin/env python3
"""
Test script to verify lazy loading works correctly.
"""

print("Testing lazy import of mattervial.featurizers...")

try:
    import mattervial.featurizers
    print("✓ Successfully imported mattervial.featurizers")
    
    # Test getting available featurizers
    try:
        available = mattervial.featurizers.get_available_featurizers()
        print("✓ Available featurizers:", available)
    except AttributeError:
        print("⚠ get_available_featurizers not available")
    
    # Test getting featurizer errors
    try:
        errors = mattervial.featurizers.get_featurizer_errors()
        print("✓ Featurizer errors:", errors)
    except AttributeError:
        print("⚠ get_featurizer_errors not available")
    
    # Test accessing ORB featurizer
    try:
        orb = mattervial.featurizers.orb_v3
        if orb is not None:
            print("✓ ORB featurizer available:", type(orb))
        else:
            print("⚠ ORB featurizer is None")
    except AttributeError:
        print("⚠ ORB featurizer not available")
    
    # Test accessing other featurizers
    featurizer_names = ["mvl32", "mvl16", "l_MM_v1", "l_OFM_v1", "adj_megnet", "adj_megnet_layer16"]
    for name in featurizer_names:
        try:
            featurizer = getattr(mattervial.featurizers, name, None)
            if featurizer is not None:
                print(f"✓ {name} featurizer available:", type(featurizer))
            else:
                print(f"⚠ {name} featurizer is None")
        except AttributeError:
            print(f"⚠ {name} featurizer not available")
    
except ImportError as e:
    print(f"✗ Failed to import mattervial.featurizers: {e}")

print("Test completed.")
