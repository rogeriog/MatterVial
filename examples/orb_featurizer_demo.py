#!/usr/bin/env python3
"""
ORB Featurizer Demo Script

This script demonstrates how to use the integrated ORB featurizer in MatterVial.
It shows basic usage, comparison with other featurizers, and performance benchmarking.

Requirements:
- For ORB featurizer: conda activate /gpfs/scratch/acad/htforft/rgouvea/orb_env
- For other featurizers: conda activate ML39
"""

import pandas as pd
import numpy as np
import time
from pymatgen.core import Structure, Lattice
from pymatgen.ext.matproj import MPRester
import matplotlib.pyplot as plt
import warnings

def create_sample_structures():
    """Create a variety of sample structures for demonstration."""
    structures = []
    
    # Simple structures
    print("Creating sample structures...")
    
    # 1. Simple cubic (Fe)
    lattice = Lattice.cubic(2.87)
    structure = Structure(lattice, ["Fe"], [[0, 0, 0]])
    structures.append(("Fe (cubic)", structure))
    
    # 2. FCC (Al)
    lattice = Lattice.from_parameters(4.05, 4.05, 4.05, 90, 90, 90)
    structure = Structure(lattice, ["Al", "Al", "Al", "Al"], 
                         [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    structures.append(("Al (FCC)", structure))
    
    # 3. Binary compound (NaCl)
    lattice = Lattice.cubic(5.64)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structures.append(("NaCl", structure))
    
    # 4. Ternary compound (Perovskite CaTiO3)
    lattice = Lattice.cubic(3.84)
    structure = Structure(lattice, ["Ca", "Ti", "O", "O", "O"], 
                         [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    structures.append(("CaTiO3 (Perovskite)", structure))
    
    # 5. Complex structure (Spinel MgAl2O4)
    lattice = Lattice.cubic(8.08)
    structure = Structure(lattice, 
                         ["Mg", "Al", "Al", "O", "O", "O", "O"], 
                         [[0.125, 0.125, 0.125], [0.5, 0.5, 0.5], [0, 0, 0], 
                          [0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.75, 0.25, 0.25]])
    structures.append(("MgAl2O4 (Spinel)", structure))
    
    return structures

def demonstrate_orb_featurizer():
    """Demonstrate basic ORB featurizer usage."""
    print("\n" + "="*60)
    print("ORB Featurizer Demonstration")
    print("="*60)
    
    # Import MatterVial
    try:
        import mattervial.featurizers
        print("✓ Successfully imported mattervial.featurizers")
        
        # Check available featurizers
        if hasattr(mattervial.featurizers, 'get_available_featurizers'):
            available = mattervial.featurizers.get_available_featurizers()
            print(f"Available featurizers: {list(available.keys())}")
            
            # Show status of each featurizer
            for name, status in available.items():
                print(f"  {name}: {status}")
        
    except ImportError as e:
        print(f"✗ Failed to import mattervial.featurizers: {e}")
        return
    
    # Create sample structures
    sample_structures = create_sample_structures()
    
    # Convert to DataFrame format
    structure_data = []
    for name, structure in sample_structures:
        structure_data.append({
            'name': name,
            'structure': structure.to_json(),
            'formula': structure.formula,
            'num_atoms': len(structure)
        })
    
    df = pd.DataFrame(structure_data)
    print(f"\nCreated dataset with {len(df)} structures:")
    for _, row in df.iterrows():
        print(f"  {row['name']}: {row['formula']} ({row['num_atoms']} atoms)")
    
    # Test ORB featurizer
    print("\nTesting ORB featurizer...")
    try:
        orb_featurizer = mattervial.featurizers.orb_v3
        if orb_featurizer is not None:
            print("✓ ORB featurizer available")
            
            # Extract features
            start_time = time.time()
            orb_features = orb_featurizer.get_features(df[['structure']])
            extraction_time = time.time() - start_time
            
            print(f"✓ ORB features extracted successfully!")
            print(f"  Shape: {orb_features.shape}")
            print(f"  Time: {extraction_time:.2f} seconds")
            print(f"  Features per structure: {orb_features.shape[1]}")
            print(f"  Time per structure: {extraction_time/len(df):.2f} seconds")
            
            # Show sample features
            feature_names = orb_features.columns.tolist()
            print(f"  Sample feature names: {feature_names[:5]}...")
            
            # Show feature statistics
            print(f"  Feature value range: [{orb_features.min().min():.3f}, {orb_features.max().max():.3f}]")
            print(f"  Mean feature value: {orb_features.mean().mean():.3f}")
            
            return orb_features, df
            
        else:
            print("⚠ ORB featurizer is None (dependencies may be missing)")
            return None, df
            
    except Exception as e:
        print(f"✗ ORB featurizer failed: {e}")
        return None, df

def compare_featurizers(df):
    """Compare ORB featurizer with other available featurizers."""
    print("\n" + "="*60)
    print("Featurizer Comparison")
    print("="*60)
    
    import mattervial.featurizers
    
    featurizers_to_test = [
        ('ORB v3', 'orb_v3'),
        ('MVL32', 'mvl32'),
        ('MVL16', 'mvl16'),
        ('Latent MM v1', 'l_MM_v1'),
        ('Latent OFM v1', 'l_OFM_v1'),
    ]
    
    results = {}
    
    for name, attr_name in featurizers_to_test:
        print(f"\nTesting {name}...")
        try:
            featurizer = getattr(mattervial.featurizers, attr_name, None)
            if featurizer is not None:
                start_time = time.time()
                features = featurizer.get_features(df[['structure']])
                extraction_time = time.time() - start_time
                
                results[name] = {
                    'success': True,
                    'shape': features.shape,
                    'time': extraction_time,
                    'time_per_structure': extraction_time / len(df),
                    'features': features
                }
                
                print(f"  ✓ Success: {features.shape} features in {extraction_time:.2f}s")
                
            else:
                print(f"  ⚠ {name} featurizer is None")
                results[name] = {'success': False, 'reason': 'Featurizer is None'}
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[name] = {'success': False, 'reason': str(e)}
    
    # Summary table
    print(f"\n{'Featurizer':<20} {'Status':<10} {'Features':<12} {'Time (s)':<10} {'Time/struct':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        if result['success']:
            print(f"{name:<20} {'✓':<10} {result['shape'][1]:<12} {result['time']:<10.2f} {result['time_per_structure']:<12.3f}")
        else:
            print(f"{name:<20} {'✗':<10} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
    
    return results

def performance_benchmark():
    """Benchmark ORB featurizer performance with different dataset sizes."""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    import mattervial.featurizers
    
    # Check if ORB featurizer is available
    orb_featurizer = getattr(mattervial.featurizers, 'orb_v3', None)
    if orb_featurizer is None:
        print("⚠ ORB featurizer not available for benchmarking")
        return
    
    # Create datasets of different sizes
    base_structures = create_sample_structures()
    dataset_sizes = [1, 2, 5, 10]
    
    benchmark_results = []
    
    for size in dataset_sizes:
        print(f"\nBenchmarking with {size} structures...")
        
        # Create dataset by repeating structures
        structures_for_test = []
        for i in range(size):
            name, structure = base_structures[i % len(base_structures)]
            structures_for_test.append(structure.to_json())
        
        df_test = pd.DataFrame({'structure': structures_for_test})
        
        try:
            # Measure extraction time
            start_time = time.time()
            features = orb_featurizer.get_features(df_test)
            extraction_time = time.time() - start_time
            
            result = {
                'size': size,
                'time': extraction_time,
                'time_per_structure': extraction_time / size,
                'features_per_structure': features.shape[1],
                'success': True
            }
            
            print(f"  ✓ {size} structures: {extraction_time:.2f}s ({extraction_time/size:.3f}s per structure)")
            
        except Exception as e:
            result = {
                'size': size,
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ Failed with {size} structures: {e}")
        
        benchmark_results.append(result)
    
    # Summary
    print(f"\n{'Dataset Size':<12} {'Total Time (s)':<15} {'Time per Structure (s)':<20} {'Features':<10}")
    print("-" * 60)
    
    for result in benchmark_results:
        if result['success']:
            print(f"{result['size']:<12} {result['time']:<15.2f} {result['time_per_structure']:<20.3f} {result['features_per_structure']:<10}")
        else:
            print(f"{result['size']:<12} {'Failed':<15} {'N/A':<20} {'N/A':<10}")
    
    return benchmark_results

def main():
    """Main demonstration function."""
    print("MatterVial ORB Featurizer Demo")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Basic demonstration
    orb_features, df = demonstrate_orb_featurizer()
    
    if orb_features is not None:
        # Comparison with other featurizers
        comparison_results = compare_featurizers(df)
        
        # Performance benchmark
        benchmark_results = performance_benchmark()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        print(f"✓ ORB featurizer extracted {orb_features.shape[1]} features from {orb_features.shape[0]} structures")
        print("✓ Comparison with other featurizers completed")
        print("✓ Performance benchmark completed")
        
    else:
        print("\n" + "="*60)
        print("Demo completed with limitations")
        print("="*60)
        print("⚠ ORB featurizer was not available - check environment and dependencies")
        print("  To use ORB featurizer: conda activate /gpfs/scratch/acad/htforft/rgouvea/orb_env")

if __name__ == "__main__":
    main()
