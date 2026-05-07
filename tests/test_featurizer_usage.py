#!/usr/bin/env python3
"""
Test script to verify featurizers work correctly when used.
"""

import pandas as pd
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

print("Testing featurizer usage...")

try:
    import mattervial.featurizers
    print("✓ Successfully imported mattervial.featurizers")
    
    # Create test structures
    structures = create_test_structures()
    # Convert structures to JSON strings for compatibility with featurizers
    structure_strings = [structure.to_json() for structure in structures]
    df = pd.DataFrame({'structure': structure_strings})
    print(f"✓ Created {len(structures)} test structures")
    
    # Test ORB featurizer
    print("\nTesting ORB featurizer...")
    try:
        orb = mattervial.featurizers.orb_v3
        if orb is not None:
            print("✓ ORB featurizer instance available")
            features = orb.get_features(df)
            print(f"✓ ORB features extracted: {features.shape}")
        else:
            print("⚠ ORB featurizer is None")
    except Exception as e:
        print(f"✗ ORB featurizer failed: {e}")
    
    # Test MVL featurizer (should fail due to missing dependencies)
    print("\nTesting MVL featurizer...")
    try:
        mvl = mattervial.featurizers.mvl32
        if mvl is not None:
            print("✓ MVL featurizer instance available")
            features = mvl.get_features(df)
            print(f"✓ MVL features extracted: {features.shape}")
        else:
            print("⚠ MVL featurizer is None")
    except Exception as e:
        print(f"✗ MVL featurizer failed (expected): {e}")
    
except ImportError as e:
    print(f"✗ Failed to import mattervial.featurizers: {e}")

print("Test completed.")
