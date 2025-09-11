#!/usr/bin/env python3
"""
ORB Featurizer Consistency Test

This script compares the output of the integrated ORB featurizer with the original
orb_extract_features.py script to ensure consistency.

Requirements:
- conda activate /gpfs/scratch/acad/htforft/rgouvea/orb_env
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pymatgen.core import Structure, Lattice
import subprocess

def create_test_dataset():
    """Create a small test dataset for comparison."""
    structures = []
    
    # Simple cubic structure (Fe)
    lattice1 = Lattice.cubic(2.87)
    structure1 = Structure(lattice1, ["Fe"], [[0, 0, 0]])
    structures.append(structure1)
    
    # Binary compound (NaCl)
    lattice2 = Lattice.cubic(5.64)
    structure2 = Structure(lattice2, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structures.append(structure2)
    
    # Ternary compound (Perovskite)
    lattice3 = Lattice.cubic(3.84)
    structure3 = Structure(lattice3, ["Ca", "Ti", "O", "O", "O"], 
                          [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    structures.append(structure3)
    
    # Create DataFrame
    structure_data = []
    for i, structure in enumerate(structures):
        structure_data.append({
            'id': f'test_structure_{i+1}',
            'structure': structure.to_json(),
            'formula': structure.formula
        })
    
    return pd.DataFrame(structure_data)

def test_integrated_featurizer(df):
    """Test the integrated ORB featurizer."""
    print("Testing integrated ORB featurizer...")
    
    try:
        import mattervial.featurizers
        orb_featurizer = mattervial.featurizers.orb_v3
        
        if orb_featurizer is None:
            print("✗ ORB featurizer not available")
            return None
        
        # Extract features
        features = orb_featurizer.get_features(df[['structure']])
        
        print(f"✓ Integrated featurizer: {features.shape} features extracted")
        print(f"  Feature columns: {len(features.columns)}")
        print(f"  Sample features: {list(features.columns[:5])}")
        
        return features
        
    except Exception as e:
        print(f"✗ Integrated featurizer failed: {e}")
        return None

def test_original_script(df):
    """Test the original orb_extract_features.py script."""
    print("Testing original ORB script...")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_input:
        input_file = temp_input.name
        df.to_csv(input_file, index=False)
    
    try:
        # Run the original script
        script_path = 'orb_extract_features.py'
        if not os.path.exists(script_path):
            print(f"✗ Original script not found: {script_path}")
            return None
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        
        # Run the script with 1 chunk for simplicity
        cmd = [
            'python', script_path,
            '--input_csv', input_file,
            '--num_chunks', '1'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0:
            print(f"✗ Original script failed:")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return None
        
        # Find the output file
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        chunk_file = f"orb_features_chunks_{base_name}/{base_name}_orb_chunk_001.csv"
        
        if not os.path.exists(chunk_file):
            print(f"✗ Output file not found: {chunk_file}")
            print(f"  Available files: {os.listdir('.')}")
            return None
        
        # Load the results
        original_results = pd.read_csv(chunk_file)
        
        # Extract only the ORB feature columns
        orb_columns = [col for col in original_results.columns if col.startswith('ORB_v3_')]
        original_features = original_results[orb_columns]
        
        print(f"✓ Original script: {original_features.shape} features extracted")
        print(f"  Feature columns: {len(original_features.columns)}")
        print(f"  Sample features: {list(original_features.columns[:5])}")
        
        # Cleanup
        os.unlink(input_file)
        import shutil
        shutil.rmtree(f"orb_features_chunks_{base_name}", ignore_errors=True)
        
        return original_features
        
    except Exception as e:
        print(f"✗ Original script test failed: {e}")
        # Cleanup
        try:
            os.unlink(input_file)
        except:
            pass
        return None

def compare_results(integrated_features, original_features):
    """Compare the results from both methods."""
    print("\nComparing results...")
    
    if integrated_features is None or original_features is None:
        print("✗ Cannot compare - one or both methods failed")
        return False
    
    # Check shapes
    print(f"Shape comparison:")
    print(f"  Integrated: {integrated_features.shape}")
    print(f"  Original:   {original_features.shape}")
    
    if integrated_features.shape != original_features.shape:
        print("✗ Shapes don't match")
        return False
    
    print("✓ Shapes match")
    
    # Check column names
    integrated_cols = set(integrated_features.columns)
    original_cols = set(original_features.columns)
    
    if integrated_cols != original_cols:
        print("✗ Column names don't match")
        print(f"  Only in integrated: {integrated_cols - original_cols}")
        print(f"  Only in original: {original_cols - integrated_cols}")
        return False
    
    print("✓ Column names match")
    
    # Sort columns for comparison
    common_cols = sorted(list(integrated_cols))
    integrated_sorted = integrated_features[common_cols]
    original_sorted = original_features[common_cols]
    
    # Check values
    try:
        # Allow for small numerical differences
        np.testing.assert_allclose(
            integrated_sorted.values, 
            original_sorted.values, 
            rtol=1e-10, 
            atol=1e-10
        )
        print("✓ Feature values match (within numerical precision)")
        
        # Calculate some statistics
        max_diff = np.abs(integrated_sorted.values - original_sorted.values).max()
        mean_diff = np.abs(integrated_sorted.values - original_sorted.values).mean()
        
        print(f"  Maximum difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Feature values don't match: {e}")
        
        # Show some statistics about the differences
        diff = np.abs(integrated_sorted.values - original_sorted.values)
        print(f"  Maximum difference: {diff.max():.2e}")
        print(f"  Mean difference: {diff.mean():.2e}")
        print(f"  Std difference: {diff.std():.2e}")
        
        # Show which features have the largest differences
        max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  Largest difference at row {max_diff_idx[0]}, column {common_cols[max_diff_idx[1]]}")
        print(f"    Integrated: {integrated_sorted.iloc[max_diff_idx]}")
        print(f"    Original: {original_sorted.iloc[max_diff_idx]}")
        
        return False

def main():
    """Main consistency test function."""
    print("ORB Featurizer Consistency Test")
    print("=" * 50)
    
    # Create test dataset
    print("Creating test dataset...")
    df = create_test_dataset()
    print(f"✓ Created dataset with {len(df)} structures")
    
    for _, row in df.iterrows():
        print(f"  {row['id']}: {row['formula']}")
    
    print()
    
    # Test both methods
    integrated_features = test_integrated_featurizer(df)
    original_features = test_original_script(df)
    
    # Compare results
    success = compare_results(integrated_features, original_features)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Consistency test PASSED!")
        print("The integrated ORB featurizer produces identical results to the original script.")
    else:
        print("❌ Consistency test FAILED!")
        print("The integrated ORB featurizer produces different results from the original script.")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
