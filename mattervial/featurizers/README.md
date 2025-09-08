# MatterVial Featurizers

This directory contains the core featurization modules of MatterVial, implementing various approaches to extract meaningful features from materials data.

## Module Overview

### Structure-based Featurizers (`structure.py`)

Contains featurizers that work with crystal structures (pymatgen Structure objects):

#### New Unified Classes (Recommended)
- **DescriptorMEGNetFeaturizer**: Unified interface for descriptor-oriented MEGNet features as described in methods section
  - `base_descriptor='l-MM_v1'`: Extracts 758 ℓ-MM (latent MatMiner) features
  - `base_descriptor='l-OFM_v1'`: Extracts 188 ℓ-OFM (latent Orbital Field Matrix) features
- **AdjacentGNNFeaturizer**: Unified interface for Adjacent GNN featurizers as described in methods section
  - `base_model='MEGNet'`: Uses MEGNet architecture (default)
  - `base_model='coGN'`: Uses coGN architecture (coming soon)

#### Other Featurizers
- **MVLFeaturizer**: Uses pretrained Materials Virtual Lab MEGNet models for feature extraction
- **ORBFeaturizer**: Uses ORB-v3 machine learning interatomic potential for feature extraction

#### Legacy Classes (Deprecated)
- **LatentMMFeaturizer (ℓ-MM)**: ⚠️ Deprecated - Use `DescriptorMEGNetFeaturizer(base_descriptor='l-MM_v1')`
- **LatentOFMFeaturizer (ℓ-OFM)**: ⚠️ Deprecated - Use `DescriptorMEGNetFeaturizer(base_descriptor='l-OFM_v1')`
- **AdjacentMEGNetFeaturizer**: ⚠️ Deprecated - Use `AdjacentGNNFeaturizer(base_model='MEGNet')`

### Composition-based Featurizers (`composition.py`)

Contains featurizers that work with material compositions (chemical formulas):

- **RoostModelFeaturizer**: Uses ROOST models for composition-based feature extraction

### SISSO Featurization (`sisso_featurization.py`)

Implements symbolic regression-based feature generation:

- **get_sisso_features()**: Applies SISSO formulas to create interpretable features from MatMiner descriptors

### Model Implementation Files

- `megnet_models.py`: Core MEGNet model operations and feature extraction functions
- `roost_models.py`: ROOST model implementation and utilities  
- `mlip_models.py`: Machine learning interatomic potential models (ORB)

## Usage Patterns

### Basic Structure Featurization (New Recommended Approach)

```python
from mattervial.featurizers import MVLFeaturizer, DescriptorMEGNetFeaturizer, AdjacentGNNFeaturizer

# Initialize featurizers using new unified classes
mvl_featurizer = MVLFeaturizer(layers=['layer32', 'layer16'])
desc_mm = DescriptorMEGNetFeaturizer(base_descriptor='l-MM_v1')  # ℓ-MM features
desc_ofm = DescriptorMEGNetFeaturizer(base_descriptor='l-OFM_v1')  # ℓ-OFM features
adj_gnn = AdjacentGNNFeaturizer(base_model='MEGNet', layers='layer32')

# Extract features
mvl_features = mvl_featurizer.get_features(structures)
l_mm_features = desc_mm.get_features(structures)  # 758 ℓ-MM features
l_ofm_features = desc_ofm.get_features(structures)  # 188 ℓ-OFM features

# Train and extract adjacent features
adj_gnn.train_adjacent_model(structures, targets, adjacent_model_path='./models/')
adj_features = adj_gnn.get_features(structures)
```

### Legacy Usage (Deprecated but Still Works)

```python
from mattervial.featurizers import LatentMMFeaturizer, LatentOFMFeaturizer, AdjacentMEGNetFeaturizer

# These will issue deprecation warnings
l_mm_legacy = LatentMMFeaturizer()  # Issues warning
l_ofm_legacy = LatentOFMFeaturizer()  # Issues warning
adj_legacy = AdjacentMEGNetFeaturizer(layers='layer32')  # Issues warning

# Extract features (functionality unchanged)
l_mm_features = l_mm_legacy.get_features(structures)
l_ofm_features = l_ofm_legacy.get_features(structures)
adj_legacy.train_adjacent_megnet(structures, targets)
adj_features = adj_legacy.get_features(structures)
```

### Composition-based Featurization

```python
from mattervial.featurizers import RoostModelFeaturizer

# Initialize ROOST featurizer
roost_featurizer = RoostModelFeaturizer(model_type='mpgap')

# Extract features from compositions
features = roost_featurizer.get_features(["Fe2O3", "Al2O3", "TiO2"])
```

### Adjacent Model Training

```python
from mattervial.featurizers import AdjacentMEGNetFeaturizer

# Initialize and train adjacent model
adj_featurizer = AdjacentMEGNetFeaturizer(layers='layer32')
adj_featurizer.train_adjacent_megnet(train_structures, train_targets)

# Extract features
features = adj_featurizer.get_features(test_structures)
```

## Environment Requirements

Different featurizers have different dependency requirements:

- **Primary Environment** (`env_primary.yml`): MEGNet-based featurizers, ROOST
- **ORB Environment** (`env_orb.yml`): ORB featurizer only
- **KGCNN Environment** (`env_kgcnn.yml`): coGN/coGNN models

See the main README.md for detailed environment setup instructions.

## Feature Output Formats

All featurizers return pandas DataFrames with consistent naming conventions:

- **MVL features**: `MVL32_PropertyName_Dataset_Index` or `MVL16_PropertyName_Dataset_Index`
- **Latent MM features**: `MEGNet_MatMiner_Index` 
- **Latent OFM features**: `MEGNet_OFMEncoded_v1_Index`
- **Adjacent features**: `Adjacent_MEGNet_LayerSize_Index`
- **ROOST features**: `ROOST_Index`
- **ORB features**: `ORB_Index`
- **SISSO features**: `SISSO_DatasetName_Index`

## Error Handling

All featurizers implement robust error handling:

- **ImportError**: Raised when required dependencies are not available
- **ValueError**: Raised for invalid input parameters or data
- **FileNotFoundError**: Raised when required model files are missing

Use the `get_available_featurizers()` function to check which featurizers are available in your current environment.

```python
from mattervial.featurizers import get_available_featurizers, get_featurizer_errors

# Check which featurizers are available
available = get_available_featurizers()
print("Available featurizers:", available)

# Check for any errors
errors = get_featurizer_errors()
if errors:
    print("Featurizer errors:", errors)
```

## Migration Guide

### Migrating from Legacy Classes

To align with the methods section terminology, we recommend migrating to the new unified classes:

#### LatentMMFeaturizer → DescriptorMEGNetFeaturizer

```python
# Old (deprecated)
from mattervial import LatentMMFeaturizer
l_mm = LatentMMFeaturizer()
features = l_mm.get_features(structures)

# New (recommended)
from mattervial import DescriptorMEGNetFeaturizer
desc_mm = DescriptorMEGNetFeaturizer(base_descriptor='l-MM_v1')
features = desc_mm.get_features(structures)
```

#### LatentOFMFeaturizer → DescriptorMEGNetFeaturizer

```python
# Old (deprecated)
from mattervial import LatentOFMFeaturizer
l_ofm = LatentOFMFeaturizer()
features = l_ofm.get_features(structures)

# New (recommended)
from mattervial import DescriptorMEGNetFeaturizer
desc_ofm = DescriptorMEGNetFeaturizer(base_descriptor='l-OFM_v1')
features = desc_ofm.get_features(structures)
```

#### AdjacentMEGNetFeaturizer → AdjacentGNNFeaturizer

```python
# Old (deprecated)
from mattervial import AdjacentMEGNetFeaturizer
adj_megnet = AdjacentMEGNetFeaturizer(layers='layer32')
adj_megnet.train_adjacent_megnet(structures, targets)
features = adj_megnet.get_features(structures)

# New (recommended)
from mattervial import AdjacentGNNFeaturizer
adj_gnn = AdjacentGNNFeaturizer(base_model='MEGNet', layers='layer32')
adj_gnn.train_adjacent_model(structures, targets)
features = adj_gnn.get_features(structures)
```

### Benefits of New Classes

1. **Alignment with Methods Section**: Class names match exactly what's described in the research paper
2. **Unified Interface**: Single class for multiple related functionalities
3. **Future Extensibility**: Easy to add new base models (coGN) and descriptors
4. **Consistent API**: Standardized parameter names and method signatures
5. **Backward Compatibility**: Legacy classes still work but issue deprecation warnings
