# MatterVial Examples

This directory contains example scripts and demonstrations of MatterVial's capabilities.

## Available Examples

### `multi_layer_featurizers_demo.py`
Demonstrates the multi-layer featurizer functionality:
- Shows how to use MVLFeaturizer and AdjacentMEGNetFeaturizer with multiple layers
- Tests backward compatibility with single-layer configurations
- Displays available featurizers and their status
- Provides interface testing for different configurations

**Key Features Demonstrated:**
- Multi-layer feature extraction (combining layer32 and layer16)
- Backward compatibility with existing single-layer code
- Error handling and availability checking
- Custom configuration options

### `orb_featurizer_demo.py`
Demonstrates ORB (Orbital-based Representation) featurizer usage:
- Shows how to set up and use the ORB featurizer
- Tests ORB model availability and environment requirements
- Demonstrates feature extraction from crystal structures
- Includes performance benchmarking capabilities

**Key Features Demonstrated:**
- ORB model initialization and configuration
- Structure preparation and featurization
- Environment dependency checking
- GPU/CPU device selection

### `orb_consistency_test.py`
Tests consistency and reproducibility of ORB featurizer:
- Validates that ORB features are consistent across runs
- Tests different precision modes and device configurations
- Ensures reproducibility of results
- Benchmarks performance across different settings

### `orb_performance_benchmark.py`
Benchmarks ORB featurizer performance:
- Measures featurization speed for different dataset sizes
- Compares GPU vs CPU performance
- Tests memory usage and scalability
- Provides timing analysis for optimization

## Usage Instructions

### Running the Examples

```bash
# Navigate to the examples directory
cd examples/

# Run multi-layer featurizer demo
python multi_layer_featurizers_demo.py

# Run ORB featurizer demo (requires ORB environment)
conda activate env_orb
python orb_featurizer_demo.py

# Run ORB consistency test
python orb_consistency_test.py

# Run ORB performance benchmark
python orb_performance_benchmark.py
```

### Environment Requirements

Different examples require different conda environments:

- **multi_layer_featurizers_demo.py**: Primary environment (`env_primary`)
- **orb_*.py scripts**: ORB environment (`env_orb`)

Make sure to activate the appropriate environment before running each example.

## Example Output

### Multi-layer Featurizer Demo
```
=== MatterVial Multi-Layer Featurizers Demo ===

--- Testing Interface and Configuration ---
✓ adj_megnet_all instance available
  Configured layers: ['layer32', 'layer16']
  Model path: ./

✓ adj_megnet available: ['layer32']
✓ adj_megnet_layer16 available: ['layer16']

--- Available Featurizers ---
MVL Featurizers:
  🆕 mvl_all      : Available
     mvl32       : Available  
     mvl16       : Available

Adjacent Featurizers:
  🆕 adj_megnet_all    : Available
     adj_megnet        : Available
     adj_megnet_layer16: Available
```

### ORB Featurizer Demo
```
=== ORB Featurizer Demonstration ===

✓ Successfully imported mattervial.featurizers
Available featurizers: ['l_MM_v1', 'l_OFM_v1', 'mvl32', 'mvl16', 'orb_v3']

--- ORB Featurizer Status ---
orb_v3: Available

--- Testing ORB Feature Extraction ---
Processing structure 1/3
Processing structure 2/3  
Processing structure 3/3
ORB features extracted: 256 features for 3 structures
Feature extraction completed successfully.
```

## Creating Custom Examples

When creating new examples, follow these patterns:

### Basic Structure
```python
#!/usr/bin/env python3
"""
Example script demonstrating [specific functionality].
"""

def main():
    """Main demonstration function."""
    print("=== Your Example Title ===")
    
    try:
        # Import MatterVial components
        import mattervial.featurizers
        
        # Your demonstration code here
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please check your environment setup.")
        return
    
    except Exception as e:
        print(f"Error during demonstration: {e}")
        return

if __name__ == "__main__":
    main()
```

### Error Handling
Always include proper error handling for:
- Import errors (missing dependencies)
- Environment issues (wrong conda environment)
- Data loading problems
- Model availability issues

### Documentation
Include clear docstrings and comments explaining:
- What the example demonstrates
- Required environment/dependencies
- Expected output
- Common issues and solutions

## Common Issues and Solutions

### Import Errors
```python
# Check featurizer availability
import mattervial.featurizers
available = mattervial.featurizers.get_available_featurizers()
print("Available featurizers:", available)
```

### Environment Issues
```python
# Check if specific featurizer is available
try:
    from mattervial.featurizers import ORBFeaturizer
    orb = ORBFeaturizer()
    print("ORB featurizer available")
except ImportError as e:
    print(f"ORB not available: {e}")
    print("Please activate the ORB environment: conda activate env_orb")
```

### Model File Issues
```python
# Check for required model files
import os
model_path = "./MEGNetModel__adjacent.h5"
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    print("Please train the adjacent model first.")
```

## Contributing Examples

When contributing new examples:

1. Follow the existing code style and structure
2. Include comprehensive error handling
3. Add clear documentation and comments
4. Test with different environments
5. Update this README with your example description
