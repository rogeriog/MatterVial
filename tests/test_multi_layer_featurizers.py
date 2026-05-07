#!/usr/bin/env python3
"""
Test script for multi-layer MVL and Adjacent MEGNet featurizers.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Structure, Lattice

def create_test_structures():
    """Create simple test structures."""
    structures = []
    
    # Simple cubic structure (Fe)
    lattice1 = Lattice.cubic(2.87)
    structure1 = Structure(lattice1, ["Fe"], [[0, 0, 0]])
    structures.append(structure1)
    
    # Binary compound (NaCl)
    lattice2 = Lattice.cubic(5.64)
    structure2 = Structure(lattice2, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structures.append(structure2)
    
    return structures

def test_mvl_featurizer():
    """Test MVLFeaturizer with different layer configurations."""
    print("=" * 60)
    print("Testing MVLFeaturizer")
    print("=" * 60)
    
    try:
        import mattervial.featurizers
        print("✓ Successfully imported mattervial.featurizers")
        
        # Create test data
        structures = create_test_structures()
        structure_strings = [s.to_json() for s in structures]
        df = pd.DataFrame({'structure': structure_strings})
        print(f"✓ Created test dataset with {len(structures)} structures")
        
        # Test backward compatibility instances
        print("\n--- Testing Backward Compatibility ---")
        
        # Test mvl32 (single layer)
        try:
            mvl32 = mattervial.featurizers.mvl32
            if mvl32 is not None:
                print("✓ mvl32 instance available")
                features32 = mvl32.get_features(df)
                print(f"  mvl32 features: {features32.shape}")
                print(f"  Sample columns: {list(features32.columns[:3])}")
            else:
                print("⚠ mvl32 instance is None")
        except Exception as e:
            print(f"✗ mvl32 failed: {e}")
        
        # Test mvl16 (single layer)
        try:
            mvl16 = mattervial.featurizers.mvl16
            if mvl16 is not None:
                print("✓ mvl16 instance available")
                features16 = mvl16.get_features(df)
                print(f"  mvl16 features: {features16.shape}")
                print(f"  Sample columns: {list(features16.columns[:3])}")
            else:
                print("⚠ mvl16 instance is None")
        except Exception as e:
            print(f"✗ mvl16 failed: {e}")
        
        # Test new multi-layer instance
        print("\n--- Testing Multi-Layer Functionality ---")
        
        try:
            mvl_all = mattervial.featurizers.mvl_all
            if mvl_all is not None:
                print("✓ mvl_all instance available")
                features_all = mvl_all.get_features(df)
                print(f"  mvl_all features: {features_all.shape}")
                print(f"  Sample columns: {list(features_all.columns[:5])}")
                
                # Check that we have features from both layers
                layer32_cols = [col for col in features_all.columns if col.startswith('layer32_')]
                layer16_cols = [col for col in features_all.columns if col.startswith('layer16_')]
                print(f"  layer32 features: {len(layer32_cols)}")
                print(f"  layer16 features: {len(layer16_cols)}")
                
                if len(layer32_cols) > 0 and len(layer16_cols) > 0:
                    print("✓ Multi-layer extraction successful")
                else:
                    print("✗ Multi-layer extraction failed - missing layer features")
            else:
                print("⚠ mvl_all instance is None")
        except Exception as e:
            print(f"✗ mvl_all failed: {e}")
        
        # Test custom layer configurations
        print("\n--- Testing Custom Configurations ---")
        
        try:
            # Test single layer with new interface
            custom_single = mattervial.featurizers.MVLFeaturizer(layers=['layer32'])
            features_custom = custom_single.get_features(df)
            print(f"✓ Custom single layer: {features_custom.shape}")
            
            # Test both layers with new interface
            custom_both = mattervial.featurizers.MVLFeaturizer(layers=['layer32', 'layer16'])
            features_both = custom_both.get_features(df)
            print(f"✓ Custom both layers: {features_both.shape}")
            
            # Test backward compatibility with string parameter
            custom_string = mattervial.featurizers.MVLFeaturizer(layers='layer16')
            features_string = custom_string.get_features(df)
            print(f"✓ Custom string parameter: {features_string.shape}")
            
        except Exception as e:
            print(f"✗ Custom configurations failed: {e}")
        
    except ImportError as e:
        print(f"✗ Failed to import mattervial.featurizers: {e}")

def test_adjacent_megnet_featurizer():
    """Test AdjacentMEGNetFeaturizer with different layer configurations."""
    print("\n" + "=" * 60)
    print("Testing AdjacentMEGNetFeaturizer")
    print("=" * 60)
    
    try:
        import mattervial.featurizers
        
        # Create test data
        structures = create_test_structures()
        structure_strings = [s.to_json() for s in structures]
        df = pd.DataFrame({'structure': structure_strings})
        
        # Test backward compatibility instances
        print("\n--- Testing Backward Compatibility ---")
        
        # Test adj_megnet (single layer)
        try:
            adj_megnet = mattervial.featurizers.adj_megnet
            if adj_megnet is not None:
                print("✓ adj_megnet instance available")
                # Note: This will likely fail due to missing model files, but we test the interface
                try:
                    features32 = adj_megnet.get_features(df)
                    print(f"  adj_megnet features: {features32.shape}")
                except Exception as e:
                    print(f"  adj_megnet execution failed (expected): {e}")
            else:
                print("⚠ adj_megnet instance is None")
        except Exception as e:
            print(f"✗ adj_megnet failed: {e}")
        
        # Test adj_megnet_layer16 (single layer)
        try:
            adj_megnet16 = mattervial.featurizers.adj_megnet_layer16
            if adj_megnet16 is not None:
                print("✓ adj_megnet_layer16 instance available")
                try:
                    features16 = adj_megnet16.get_features(df)
                    print(f"  adj_megnet_layer16 features: {features16.shape}")
                except Exception as e:
                    print(f"  adj_megnet_layer16 execution failed (expected): {e}")
            else:
                print("⚠ adj_megnet_layer16 instance is None")
        except Exception as e:
            print(f"✗ adj_megnet_layer16 failed: {e}")
        
        # Test new multi-layer instance
        print("\n--- Testing Multi-Layer Functionality ---")
        
        try:
            adj_megnet_all = mattervial.featurizers.adj_megnet_all
            if adj_megnet_all is not None:
                print("✓ adj_megnet_all instance available")
                try:
                    features_all = adj_megnet_all.get_features(df)
                    print(f"  adj_megnet_all features: {features_all.shape}")
                except Exception as e:
                    print(f"  adj_megnet_all execution failed (expected): {e}")
            else:
                print("⚠ adj_megnet_all instance is None")
        except Exception as e:
            print(f"✗ adj_megnet_all failed: {e}")
        
        # Test custom layer configurations
        print("\n--- Testing Custom Configurations ---")
        
        try:
            # Test interface creation (won't execute due to missing models)
            custom_single = mattervial.featurizers.AdjacentMEGNetFeaturizer(layers=['layer32'])
            print(f"✓ Custom single layer interface created")
            print(f"  Layers: {custom_single.layers}")
            
            custom_both = mattervial.featurizers.AdjacentMEGNetFeaturizer(layers=['layer32', 'layer16'])
            print(f"✓ Custom both layers interface created")
            print(f"  Layers: {custom_both.layers}")
            
            custom_string = mattervial.featurizers.AdjacentMEGNetFeaturizer(layers='layer16')
            print(f"✓ Custom string parameter interface created")
            print(f"  Layers: {custom_string.layers}")
            
        except Exception as e:
            print(f"✗ Custom configurations failed: {e}")
        
    except ImportError as e:
        print(f"✗ Failed to import mattervial.featurizers: {e}")

def test_available_featurizers():
    """Test the get_available_featurizers function."""
    print("\n" + "=" * 60)
    print("Testing Available Featurizers")
    print("=" * 60)
    
    try:
        import mattervial.featurizers
        
        available = mattervial.featurizers.get_available_featurizers()
        print("Available featurizers:")
        for name, status in available.items():
            print(f"  {name}: {status}")
        
        # Check that new instances are included
        expected_new = ['mvl_all', 'adj_megnet_all']
        for name in expected_new:
            if name in available:
                print(f"✓ New instance '{name}' is included")
            else:
                print(f"✗ New instance '{name}' is missing")
        
    except Exception as e:
        print(f"✗ Failed to test available featurizers: {e}")

def main():
    """Run all tests."""
    print("Multi-Layer Featurizer Test Suite")
    print("=" * 60)
    
    test_mvl_featurizer()
    test_adjacent_megnet_featurizer()
    test_available_featurizers()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
