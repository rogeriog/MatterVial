#!/usr/bin/env python3
"""
Test script with more realistic structures that MEGNet can handle.
"""

import pandas as pd
import sys
import os
from pymatgen.core import Structure, Lattice

# Add the parent directory to the path to import mattervial
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_realistic_structures():
    """Create more realistic structures that MEGNet can handle."""
    structures = []
    
    # 1. Silicon (diamond structure)
    lattice_si = Lattice.from_parameters(5.43, 5.43, 5.43, 90, 90, 90)
    structure_si = Structure(lattice_si, 
                            ["Si", "Si", "Si", "Si", "Si", "Si", "Si", "Si"],
                            [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0], [0.75, 0.75, 0.25],
                             [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0, 0.5, 0.5], [0.25, 0.75, 0.75]])
    structures.append(("Si (diamond)", structure_si))
    
    # 2. Gallium Arsenide (zincblende structure)
    lattice_gaas = Lattice.from_parameters(5.65, 5.65, 5.65, 90, 90, 90)
    structure_gaas = Structure(lattice_gaas,
                              ["Ga", "Ga", "Ga", "Ga", "As", "As", "As", "As"],
                              [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                               [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]])
    structures.append(("GaAs (zincblende)", structure_gaas))
    
    return structures

def test_feature_naming_with_realistic_structures():
    """Test feature naming with structures that MEGNet can actually process."""
    print("=" * 60)
    print("Testing Feature Names with Realistic Structures")
    print("=" * 60)
    
    try:
        import mattervial.featurizers
        
        # Create realistic test data
        sample_structures = create_realistic_structures()
        structure_data = []
        for name, structure in sample_structures:
            structure_data.append({
                'name': name,
                'structure': structure.to_json(),
                'formula': structure.formula
            })
        
        df = pd.DataFrame(structure_data)
        print(f"Created dataset with {len(df)} structures:")
        for _, row in df.iterrows():
            print(f"  {row['name']}: {row['formula']}")
        
        # Test single layer MVL featurizers
        print("\n--- Single Layer MVL Featurizers ---")
        
        try:
            mvl32 = mattervial.featurizers.mvl32
            if mvl32 is not None:
                print("Testing mvl32...")
                features32 = mvl32.get_features(df[['structure']])
                print(f"✓ mvl32 features: {features32.shape}")
                
                # Show sample feature names
                sample_cols = list(features32.columns[:5])
                print(f"  Sample columns: {sample_cols}")
                
                # Check naming pattern
                mvl32_cols = [col for col in features32.columns if col.startswith('MVL32_')]
                print(f"  MVL32_ prefixed columns: {len(mvl32_cols)} / {len(features32.columns)}")
                
        except Exception as e:
            print(f"mvl32 test failed: {e}")
        
        try:
            mvl16 = mattervial.featurizers.mvl16
            if mvl16 is not None:
                print("Testing mvl16...")
                features16 = mvl16.get_features(df[['structure']])
                print(f"✓ mvl16 features: {features16.shape}")
                
                # Show sample feature names
                sample_cols = list(features16.columns[:5])
                print(f"  Sample columns: {sample_cols}")
                
                # Check naming pattern
                mvl16_cols = [col for col in features16.columns if col.startswith('MVL16_')]
                print(f"  MVL16_ prefixed columns: {len(mvl16_cols)} / {len(features16.columns)}")
                
        except Exception as e:
            print(f"mvl16 test failed: {e}")
        
        # Test multi-layer MVL featurizer
        print("\n--- Multi-Layer MVL Featurizer ---")
        
        try:
            mvl_all = mattervial.featurizers.mvl_all
            if mvl_all is not None:
                print("Testing mvl_all...")
                features_all = mvl_all.get_features(df[['structure']])
                print(f"✓ mvl_all features: {features_all.shape}")
                
                # Show sample feature names
                sample_cols = list(features_all.columns[:10])
                print(f"  Sample columns: {sample_cols}")
                
                # Analyze feature naming
                mvl32_cols = [col for col in features_all.columns if col.startswith('MVL32_')]
                mvl16_cols = [col for col in features_all.columns if col.startswith('MVL16_')]
                redundant_cols = [col for col in features_all.columns if 'layer32_MVL32_' in col or 'layer16_MVL16_' in col]
                
                print(f"  MVL32_ prefixed columns: {len(mvl32_cols)}")
                print(f"  MVL16_ prefixed columns: {len(mvl16_cols)}")
                print(f"  Total features: {len(features_all.columns)}")
                print(f"  Redundant naming columns: {len(redundant_cols)}")
                
                if len(redundant_cols) > 0:
                    print(f"  ❌ Found redundant naming: {redundant_cols[:3]}...")
                else:
                    print(f"  ✅ Clean naming - no redundant prefixes")
                
                # Verify that we have features from both layers
                if len(mvl32_cols) > 0 and len(mvl16_cols) > 0:
                    print(f"  ✅ Successfully combined features from both layers")
                else:
                    print(f"  ❌ Missing features from one or both layers")
                
                # Show feature type distribution
                feature_types = {}
                for col in features_all.columns:
                    if '_' in col:
                        parts = col.split('_')
                        if len(parts) >= 2:
                            feature_type = f"{parts[0]}_{parts[1]}"
                            feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
                
                print(f"  Feature type distribution:")
                for ftype, count in sorted(feature_types.items()):
                    print(f"    {ftype}: {count} features")
                
        except Exception as e:
            print(f"mvl_all test failed: {e}")
        
    except ImportError as e:
        print(f"Failed to import mattervial.featurizers: {e}")

def main():
    """Run the realistic structure test."""
    print("Clean Feature Names Test with Realistic Structures")
    print("=" * 60)
    
    test_feature_naming_with_realistic_structures()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print("This test verifies that:")
    print("1. Single-layer featurizers produce clean names (MVL32_, MVL16_)")
    print("2. Multi-layer featurizers combine features without redundant prefixes")
    print("3. No layer32_MVL32_ or layer16_MVL16_ redundancy exists")
    print("4. Feature extraction works with realistic crystal structures")

if __name__ == "__main__":
    main()
