# MatterVial Interpreter

This directory contains the interpretability tools for understanding and analyzing features extracted by MatterVial featurizers.

## Overview

The MatterVial interpreter bridges the gap between high-level latent representations and interpretable chemical descriptors. It uses surrogate XGBoost models and SHAP analysis to decode the underlying chemical principles driving the predictions.

## Core Components

### `interpreter.py`
Main interpreter class providing:
- **Formula retrieval**: Get symbolic formulas for latent features (ℓ-MM, ℓ-OFM)
- **SISSO formula interpretation**: Understand SISSO-generated symbolic expressions
- **SHAP analysis**: Access feature importance analysis results
- **Visualization**: Display SVG plots for feature decomposition

### `decoder.py`
Encoding/decoding utilities for latent space features with automatic file management:
- **encode_ofm()**: Encode OFM features to latent space (ℓ-OFM)
- **decode_ofm()**: Decode latent features back to OFM space
- **encode_mm()**: Encode MatMiner features to latent space (ℓ-MM)
- **decode_mm()**: Decode latent features back to MatMiner space
- **Automatic downloads**: Decoder files (~50MB compressed) are automatically downloaded from Figshare when first needed
- **Cache management**: Local caching with utilities to check status and clear cache

### `feature_decomposition.py`
Advanced feature analysis tools:
- **SHAP decomposition**: Perform SHAP analysis on latent features
- **SISSO sample generation**: Generate datasets for SISSO symbolic regression
- **Feature importance analysis**: Analyze which interpretable features drive latent representations

## Data Structure

The interpreter relies on several data directories:

```
interpreter/
├── data/                    # SHAP analysis results and metrics
├── formulas/               # Standard feature formulas (JSON)
├── shap_values/           # SHAP importance values by feature type
├── shap_plots/            # SVG visualizations (auto-downloaded from Figshare)
└── decoders/              # Autoencoder models (auto-downloaded from Figshare)
    ├── MM__AutoEncoder.h5          # MatMiner autoencoder (130MB)
    ├── OFM__AutoEncoder.h5         # OFM autoencoder (62MB)
    ├── MM_scaler_autoencoder.pkl   # MatMiner scaler
    ├── OFM_scaler_autoencoder.pkl  # OFM scaler
    └── *_encoded_columns.txt       # Feature column names
```

## Usage Examples

### Basic Formula Retrieval

```python
from mattervial.interpreter import Interpreter

# Initialize interpreter
interpreter = Interpreter()

# Get formula for latent feature
formula_info = interpreter.get_formula("l-OFM_v1_1")
print(f"Formula: {formula_info['formula']}")
print(f"R² score: {formula_info['r2_score']}")
```

### SHAP Analysis

```python
# Get SHAP values for feature importance
shap_data = interpreter.get_shap_values("MEGNet_MatMiner_1")
print("Top contributing features:")
for feature, importance in shap_data['top_features'].items():
    print(f"  {feature}: {importance}")
```

### SISSO Formula Interpretation

```python
# Interpret SISSO-generated formulas
sisso_info = interpreter.get_formula("SISSO_matbench_dielectric_1")
print(f"Raw formula: {sisso_info['formula']}")
print(f"Formatted: {sisso_info['formatted_formula']}")
print(f"Base features: {sisso_info['features']}")
```

### Encoding/Decoding

```python
from mattervial.interpreter import encode_ofm, decode_ofm, encode_mm, decode_mm
import pandas as pd

# First usage automatically downloads decoder files (~50MB compressed)
ofm_data = pd.read_csv("ofm_features.csv")
latent_features = encode_ofm(ofm_data)  # Downloads files if needed

# Decode back to OFM space (uses cached files)
reconstructed_ofm = decode_ofm(latent_features)

# MatMiner encoding/decoding
mm_data = pd.read_csv("matminer_features.csv")
latent_mm = encode_mm(mm_data)
reconstructed_mm = decode_mm(latent_mm)
```

### Cache Management

```python
from mattervial.interpreter.decoder import get_decoder_cache_info, clear_decoder_cache

# Check decoder cache status
cache_info = get_decoder_cache_info()
print(f"Cache status: {cache_info['status']}")
print(f"Cache size: {cache_info['size_mb']} MB")
print(f"Files present: {cache_info['critical_files_present']}/{cache_info['critical_files_total']}")

# Clear cache to free disk space (files will re-download when needed)
clear_decoder_cache()
```

## Interpretability Workflow

The MatterVial interpretability framework follows this workflow:

1. **Feature Extraction**: Extract latent features using MatterVial featurizers
2. **SHAP Analysis**: Identify the top 30 most influential interpretable descriptors for each latent feature
3. **Symbolic Regression**: Use SISSO++ to generate symbolic formulas that correlate with latent features
4. **Visualization**: Generate plots and decompositions to understand feature relationships

## File Formats

### Formula Files
- **JSON format**: Standard formulas with R² scores and dimensional information
- **TXT format**: SISSO formulas with mathematical expressions

### SHAP Files  
- **JSON format**: Feature importance rankings and values
- **PKL format**: Raw SHAP analysis objects

### Visualization Files
- **SVG format**: Interactive plots with pan/zoom capabilities
- **PNG format**: Static visualizations for reports

## Advanced Usage

### Custom Formula Files

```python
# Use custom formula file
interpreter = Interpreter()
custom_formula = interpreter.get_formula(
    "custom_feature_1", 
    additional_formula_file="/path/to/custom_formulas.json"
)
```

### Batch Analysis

```python
# Analyze multiple features
features = ["MEGNet_MatMiner_1", "MEGNet_MatMiner_2", "MEGNet_MatMiner_3"]
for feature in features:
    try:
        formula = interpreter.get_formula(feature)
        shap_data = interpreter.get_shap_values(feature)
        print(f"{feature}: R² = {formula.get('r2_score', 'N/A')}")
    except Exception as e:
        print(f"{feature}: Error - {e}")
```

## Dependencies

The interpreter module requires:
- pandas, numpy: Data manipulation
- json: Formula file parsing  
- scikit-learn: Preprocessing utilities
- xgboost: Surrogate model training (for advanced features)
- shap: Feature importance analysis (for advanced features)

Most basic interpretation functions work with minimal dependencies, while advanced analysis requires the full ML stack.
