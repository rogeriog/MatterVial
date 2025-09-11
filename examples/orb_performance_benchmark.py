#!/usr/bin/env python3
"""
ORB Featurizer Performance Benchmark

This script benchmarks the performance of the ORB featurizer with different dataset sizes
and demonstrates the chunked processing approach for large datasets.

Requirements:
- conda activate /gpfs/scratch/acad/htforft/rgouvea/orb_env
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice
import psutil
import os
from typing import List, Dict, Any

def create_diverse_structures(n_structures: int) -> List[Structure]:
    """Create a diverse set of structures for benchmarking."""
    structures = []
    
    # Base structure templates
    templates = [
        # Simple cubic (Fe)
        (Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]]),
        
        # FCC (Al)
        (Lattice.from_parameters(4.05, 4.05, 4.05, 90, 90, 90), 
         ["Al", "Al", "Al", "Al"], 
         [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]),
        
        # Binary (NaCl)
        (Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        
        # Ternary (Perovskite CaTiO3)
        (Lattice.cubic(3.84), ["Ca", "Ti", "O", "O", "O"], 
         [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]),
        
        # Complex (Spinel MgAl2O4)
        (Lattice.cubic(8.08), 
         ["Mg", "Al", "Al", "O", "O", "O", "O"], 
         [[0.125, 0.125, 0.125], [0.5, 0.5, 0.5], [0, 0, 0], 
          [0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.75, 0.25, 0.25]]),
    ]
    
    # Generate structures by cycling through templates
    for i in range(n_structures):
        lattice, species, coords = templates[i % len(templates)]
        
        # Add some variation by slightly perturbing lattice parameters
        if i > 0:
            scale_factor = 0.95 + 0.1 * (i % 10) / 10  # Scale between 0.95 and 1.05
            lattice = lattice.scale(scale_factor)
        
        structure = Structure(lattice, species, coords)
        structures.append(structure)
    
    return structures

def monitor_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
    }

def benchmark_single_run(orb_featurizer, df: pd.DataFrame, run_name: str) -> Dict[str, Any]:
    """Benchmark a single run of the ORB featurizer."""
    print(f"  Running {run_name}...")
    
    # Monitor memory before
    memory_before = monitor_memory_usage()
    
    # Run featurizer
    start_time = time.time()
    features = orb_featurizer.get_features(df)
    end_time = time.time()
    
    # Monitor memory after
    memory_after = monitor_memory_usage()
    
    # Calculate metrics
    total_time = end_time - start_time
    time_per_structure = total_time / len(df)
    memory_used = memory_after['rss_mb'] - memory_before['rss_mb']
    
    result = {
        'run_name': run_name,
        'n_structures': len(df),
        'total_time': total_time,
        'time_per_structure': time_per_structure,
        'features_shape': features.shape,
        'memory_before_mb': memory_before['rss_mb'],
        'memory_after_mb': memory_after['rss_mb'],
        'memory_used_mb': memory_used,
        'success': True
    }
    
    print(f"    ✓ {len(df)} structures in {total_time:.2f}s ({time_per_structure:.3f}s per structure)")
    print(f"    Memory: {memory_used:.1f} MB used, {memory_after['rss_mb']:.1f} MB total")
    
    return result

def scaling_benchmark():
    """Benchmark ORB featurizer performance with different dataset sizes."""
    print("=" * 60)
    print("Scaling Benchmark")
    print("=" * 60)
    
    # Import ORB featurizer
    try:
        import mattervial.featurizers
        orb_featurizer = mattervial.featurizers.orb_v3
        
        if orb_featurizer is None:
            print("✗ ORB featurizer not available")
            return []
        
        print("✓ ORB featurizer loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load ORB featurizer: {e}")
        return []
    
    # Test different dataset sizes
    dataset_sizes = [1, 2, 5, 10, 20, 50]
    results = []
    
    for size in dataset_sizes:
        print(f"\nTesting with {size} structures...")
        
        try:
            # Create dataset
            structures = create_diverse_structures(size)
            structure_strings = [s.to_json() for s in structures]
            df = pd.DataFrame({'structure': structure_strings})
            
            # Benchmark
            result = benchmark_single_run(orb_featurizer, df, f"{size}_structures")
            results.append(result)
            
        except Exception as e:
            print(f"  ✗ Failed with {size} structures: {e}")
            results.append({
                'run_name': f"{size}_structures",
                'n_structures': size,
                'success': False,
                'error': str(e)
            })
    
    return results

def chunked_processing_benchmark():
    """Benchmark chunked processing approach for large datasets."""
    print("\n" + "=" * 60)
    print("Chunked Processing Benchmark")
    print("=" * 60)
    
    # Import ORB featurizer
    try:
        import mattervial.featurizers
        orb_featurizer = mattervial.featurizers.orb_v3
        
        if orb_featurizer is None:
            print("✗ ORB featurizer not available")
            return []
        
    except Exception as e:
        print(f"✗ Failed to load ORB featurizer: {e}")
        return []
    
    # Create a moderately large dataset
    total_structures = 100
    chunk_sizes = [10, 20, 50, 100]  # Different chunk sizes to test
    
    print(f"Creating dataset with {total_structures} structures...")
    structures = create_diverse_structures(total_structures)
    structure_strings = [s.to_json() for s in structures]
    full_df = pd.DataFrame({'structure': structure_strings})
    print(f"✓ Dataset created")
    
    results = []
    
    for chunk_size in chunk_sizes:
        print(f"\nTesting chunked processing with chunk size {chunk_size}...")
        
        try:
            # Process in chunks
            start_time = time.time()
            memory_before = monitor_memory_usage()
            
            chunk_results = []
            n_chunks = (len(full_df) + chunk_size - 1) // chunk_size  # Ceiling division
            
            for i in range(0, len(full_df), chunk_size):
                chunk_df = full_df.iloc[i:i+chunk_size]
                chunk_features = orb_featurizer.get_features(chunk_df)
                chunk_results.append(chunk_features)
                
                print(f"    Processed chunk {len(chunk_results)}/{n_chunks} "
                      f"({len(chunk_df)} structures)")
            
            # Combine results
            combined_features = pd.concat(chunk_results, ignore_index=True)
            
            end_time = time.time()
            memory_after = monitor_memory_usage()
            
            # Calculate metrics
            total_time = end_time - start_time
            time_per_structure = total_time / len(full_df)
            memory_used = memory_after['rss_mb'] - memory_before['rss_mb']
            
            result = {
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'total_structures': len(full_df),
                'total_time': total_time,
                'time_per_structure': time_per_structure,
                'features_shape': combined_features.shape,
                'memory_used_mb': memory_used,
                'success': True
            }
            
            print(f"  ✓ Completed: {total_time:.2f}s total ({time_per_structure:.3f}s per structure)")
            print(f"    Memory used: {memory_used:.1f} MB")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ✗ Failed with chunk size {chunk_size}: {e}")
            results.append({
                'chunk_size': chunk_size,
                'success': False,
                'error': str(e)
            })
    
    return results

def generate_report(scaling_results: List[Dict], chunked_results: List[Dict]):
    """Generate a comprehensive performance report."""
    print("\n" + "=" * 60)
    print("Performance Report")
    print("=" * 60)
    
    # Scaling results
    if scaling_results:
        print("\nScaling Performance:")
        print(f"{'Size':<6} {'Time (s)':<10} {'Time/struct':<12} {'Memory (MB)':<12} {'Features':<10}")
        print("-" * 60)
        
        for result in scaling_results:
            if result['success']:
                print(f"{result['n_structures']:<6} "
                      f"{result['total_time']:<10.2f} "
                      f"{result['time_per_structure']:<12.3f} "
                      f"{result['memory_used_mb']:<12.1f} "
                      f"{result['features_shape'][1]:<10}")
            else:
                print(f"{result['n_structures']:<6} {'FAILED':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    # Chunked processing results
    if chunked_results:
        print("\nChunked Processing Performance:")
        print(f"{'Chunk Size':<12} {'Chunks':<8} {'Time (s)':<10} {'Time/struct':<12} {'Memory (MB)':<12}")
        print("-" * 64)
        
        for result in chunked_results:
            if result['success']:
                print(f"{result['chunk_size']:<12} "
                      f"{result['n_chunks']:<8} "
                      f"{result['total_time']:<10.2f} "
                      f"{result['time_per_structure']:<12.3f} "
                      f"{result['memory_used_mb']:<12.1f}")
            else:
                print(f"{result['chunk_size']:<12} {'FAILED':<8} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
    
    # Performance insights
    print("\nPerformance Insights:")
    
    if scaling_results:
        successful_scaling = [r for r in scaling_results if r['success']]
        if len(successful_scaling) > 1:
            # Calculate scaling efficiency
            times_per_struct = [r['time_per_structure'] for r in successful_scaling]
            avg_time_per_struct = np.mean(times_per_struct)
            std_time_per_struct = np.std(times_per_struct)
            
            print(f"  • Average time per structure: {avg_time_per_struct:.3f} ± {std_time_per_struct:.3f} seconds")
            
            # Memory efficiency
            memory_per_struct = [r['memory_used_mb'] / r['n_structures'] for r in successful_scaling if r['memory_used_mb'] > 0]
            if memory_per_struct:
                avg_memory_per_struct = np.mean(memory_per_struct)
                print(f"  • Average memory per structure: {avg_memory_per_struct:.2f} MB")
    
    if chunked_results:
        successful_chunked = [r for r in chunked_results if r['success']]
        if successful_chunked:
            best_chunk_size = min(successful_chunked, key=lambda x: x['time_per_structure'])
            print(f"  • Best chunk size for performance: {best_chunk_size['chunk_size']} "
                  f"({best_chunk_size['time_per_structure']:.3f}s per structure)")

def main():
    """Main benchmarking function."""
    print("ORB Featurizer Performance Benchmark")
    print("=" * 60)
    
    # Run scaling benchmark
    scaling_results = scaling_benchmark()
    
    # Run chunked processing benchmark
    chunked_results = chunked_processing_benchmark()
    
    # Generate comprehensive report
    generate_report(scaling_results, chunked_results)
    
    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
