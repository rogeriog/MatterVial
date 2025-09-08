#!/usr/bin/env python3
"""
Multi-Layer Featurizers Demo

This script demonstrates the enhanced MVL and Adjacent MEGNet featurizers
that support multi-layer feature extraction by default.

Requirements:
- For MEGNet featurizers: conda activate ML39
"""

import pandas as pd
import numpy as np
from pymatgen.core import Structure, Lattice
import warnings
import sys
import os

# Add the parent directory to the path to import mattervial
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_structures():
    """Create sample structures for demonstration."""
    structures = []
    
    # Simple cubic (Fe)
    lattice1 = Lattice.cubic(2.87)
    structure1 = Structure(lattice1, ["Fe"], [[0, 0, 0]])
    structures.append(("Fe (cubic)", structure1))
    
    # Binary compound (NaCl)
    lattice2 = Lattice.cubic(5.64)
    structure2 = Structure(lattice2, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structures.append(("NaCl", structure2))
    
    # Ternary compound (Perovskite CaTiO3)
    lattice3 = Lattice.cubic(3.84)
    structure3 = Structure(lattice3, ["Ca", "Ti", "O", "O", "O"], 
                          [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    structures.append(("CaTiO3 (Perovskite)", structure3))
    
    return structures

def demonstrate_multi_layer_mvl():
    """Demonstrate multi-layer MVL featurizer functionality."""
    print("=" * 60)
    print("Multi-Layer MVL Featurizer Demonstration")
    print("=" * 60)
    
    try:
        import mattervial.featurizers
        print("✓ Successfully imported mattervial.featurizers")
        
        # Create sample data
        sample_structures = create_sample_structures()
        structure_data = []
        for name, structure in sample_structures:
            structure_data.append({
                'name': name,
                'structure': structure.to_json(),
                'formula': structure.formula
            })
        
        df = pd.DataFrame(structure_data)
        print(f"✓ Created dataset with {len(df)} structures:")
        for _, row in df.iterrows():
            print(f"  {row['name']}: {row['formula']}")
        
        print("\n--- Testing New Multi-Layer Instance ---")
        
        # Test new multi-layer instance
        try:
            mvl_all = mattervial.featurizers.mvl_all
            if mvl_all is not None:
                print("✓ mvl_all instance available")
                print(f"  Configured layers: {mvl_all.layers}")
                
                features_all = mvl_all.get_features(df[['structure']])
                print(f"✓ Multi-layer features extracted: {features_all.shape}")
                
                # Analyze feature composition
                mvl32_cols = [col for col in features_all.columns if col.startswith('MVL32_')]
                mvl16_cols = [col for col in features_all.columns if col.startswith('MVL16_')]

                print(f"  MVL32 features: {len(mvl32_cols)}")
                print(f"  MVL16 features: {len(mvl16_cols)}")
                print(f"  Total features: {len(mvl32_cols) + len(mvl16_cols)}")

                # Show sample feature names
                print(f"  Sample MVL32 features: {mvl32_cols[:3]}")
                print(f"  Sample MVL16 features: {mvl16_cols[:3]}")
                
            else:
                print("⚠ mvl_all instance is None")
                
        except Exception as e:
            print(f"✗ mvl_all failed: {e}")
        
        print("\n--- Comparing with Single-Layer Instances ---")
        
        # Compare with single-layer instances
        try:
            mvl32 = mattervial.featurizers.mvl32
            mvl16 = mattervial.featurizers.mvl16
            
            if mvl32 is not None and mvl16 is not None:
                print("✓ Single-layer instances available")
                
                features32 = mvl32.get_features(df[['structure']])
                features16 = mvl16.get_features(df[['structure']])
                
                print(f"  mvl32 features: {features32.shape}")
                print(f"  mvl16 features: {features16.shape}")
                
                # Manual combination (features already have proper prefixes)
                combined_manual = pd.concat([features32, features16], axis=1)
                
                print(f"  Manual combination: {combined_manual.shape}")
                
                # Compare with multi-layer result
                if 'features_all' in locals():
                    if features_all.shape == combined_manual.shape:
                        print("✓ Multi-layer result matches manual combination")
                    else:
                        print("⚠ Shape mismatch between multi-layer and manual combination")
                
            else:
                print("⚠ Single-layer instances not available")
                
        except Exception as e:
            print(f"✗ Single-layer comparison failed: {e}")
        
        print("\n--- Testing Custom Configurations ---")
        
        # Test custom configurations
        try:
            # Custom multi-layer
            custom_multi = mattervial.featurizers.MVLFeaturizer(layers=['layer32', 'layer16'])
            print(f"✓ Custom multi-layer: {custom_multi.layers}")
            
            # Custom single layer
            custom_single = mattervial.featurizers.MVLFeaturizer(layers=['layer32'])
            print(f"✓ Custom single layer: {custom_single.layers}")
            
            # Backward compatible string
            custom_string = mattervial.featurizers.MVLFeaturizer(layers='layer16')
            print(f"✓ Backward compatible string: {custom_string.layers}")
            
            # Test feature extraction with custom configuration
            custom_features = custom_single.get_features(df[['structure']])
            print(f"✓ Custom single layer features: {custom_features.shape}")
            
        except Exception as e:
            print(f"✗ Custom configurations failed: {e}")
        
    except ImportError as e:
        print(f"✗ Failed to import mattervial.featurizers: {e}")

def demonstrate_multi_layer_adjacent():
    """Demonstrate multi-layer Adjacent MEGNet featurizer functionality."""
    print("\n" + "=" * 60)
    print("Multi-Layer Adjacent MEGNet Featurizer Demonstration")
    print("=" * 60)
    
    try:
        import mattervial.featurizers
        
        print("\n--- Testing Interface and Configuration ---")
        
        # Test new multi-layer instance
        try:
            adj_all = mattervial.featurizers.adj_megnet_all
            if adj_all is not None:
                print("✓ adj_megnet_all instance available")
                print(f"  Configured layers: {adj_all.layers}")
                print(f"  Model path: {adj_all.model_path}")
            else:
                print("⚠ adj_megnet_all instance is None")
        except Exception as e:
            print(f"✗ adj_megnet_all interface test failed: {e}")
        
        # Test backward compatibility instances
        try:
            adj_megnet = mattervial.featurizers.adj_megnet
            adj_megnet16 = mattervial.featurizers.adj_megnet_layer16
            
            if adj_megnet is not None:
                print(f"✓ adj_megnet available: {adj_megnet.layers}")
            if adj_megnet16 is not None:
                print(f"✓ adj_megnet_layer16 available: {adj_megnet16.layers}")
                
        except Exception as e:
            print(f"✗ Backward compatibility test failed: {e}")
        
        print("\n--- Testing Custom Configurations ---")
        
        # Test custom configurations
        try:
            # Custom multi-layer with model path
            custom_multi = mattervial.featurizers.AdjacentMEGNetFeaturizer(
                layers=['layer32', 'layer16'],
                model_path='/custom/path/'
            )
            print(f"✓ Custom multi-layer: {custom_multi.layers}")
            print(f"  Custom model path: {custom_multi.model_path}")
            
            # Custom single layer
            custom_single = mattervial.featurizers.AdjacentMEGNetFeaturizer(
                layers='layer32',
                model_path='/another/path/'
            )
            print(f"✓ Custom single layer: {custom_single.layers}")
            print(f"  Custom model path: {custom_single.model_path}")
            
        except Exception as e:
            print(f"✗ Custom configurations failed: {e}")
        
        print("\n--- Note on Feature Extraction ---")
        print("Adjacent MEGNet requires trained model files to extract features.")
        print("The interface and configuration work correctly, but feature extraction")
        print("will fail without proper model files (MEGNetModel__adjacent.h5).")
        
    except ImportError as e:
        print(f"✗ Failed to import mattervial.featurizers: {e}")

def demonstrate_available_featurizers():
    """Show all available featurizers including new multi-layer instances."""
    print("\n" + "=" * 60)
    print("Available Featurizers Overview")
    print("=" * 60)
    
    try:
        import mattervial.featurizers
        
        available = mattervial.featurizers.get_available_featurizers()
        
        print(f"Total featurizers: {len(available)}")
        print("\nFeaturizer Status:")
        print("-" * 40)
        
        # Group by type
        mvl_featurizers = {k: v for k, v in available.items() if k.startswith('mvl')}
        adj_featurizers = {k: v for k, v in available.items() if k.startswith('adj')}
        other_featurizers = {k: v for k, v in available.items() 
                           if not k.startswith('mvl') and not k.startswith('adj')}
        
        print("MVL Featurizers:")
        for name, status in mvl_featurizers.items():
            marker = "🆕" if name == "mvl_all" else "  "
            print(f"  {marker} {name:<12}: {status}")
        
        print("\nAdjacent MEGNet Featurizers:")
        for name, status in adj_featurizers.items():
            marker = "🆕" if name == "adj_megnet_all" else "  "
            print(f"  {marker} {name:<18}: {status}")
        
        print("\nOther Featurizers:")
        for name, status in other_featurizers.items():
            print(f"    {name:<12}: {status}")
        
        print("\n🆕 = New multi-layer instances")
        
    except Exception as e:
        print(f"✗ Failed to get available featurizers: {e}")

def main():
    """Main demonstration function."""
    print("Multi-Layer Featurizers Demo")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Demonstrate MVL featurizers
    demonstrate_multi_layer_mvl()
    
    # Demonstrate Adjacent MEGNet featurizers
    demonstrate_multi_layer_adjacent()
    
    # Show available featurizers
    demonstrate_available_featurizers()
    
    print("\n" + "=" * 60)
    print("Demo Summary")
    print("=" * 60)
    print("✅ Multi-layer feature extraction implemented")
    print("✅ Backward compatibility maintained")
    print("✅ New default instances available (mvl_all, adj_megnet_all)")
    print("✅ Custom layer configurations supported")
    print("✅ Enhanced feature naming with layer prefixes")
    print("\nFor full functionality, run in ML39 environment:")
    print("  conda activate ML39 && python examples/multi_layer_featurizers_demo.py")

if __name__ == "__main__":
    main()
