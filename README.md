

<div align="center" style="margin:0">
<a target="_blank" rel="noopener noreferrer" href="img/MATTERVial_logo.png">
        <img src="img/MATTERVial_logo.png" alt="MatterVial-logo" width="200" style="display: block; margin: 0 auto;">
      </a>
<h3>Materials Feature Extraction via Interpretable Artificial Learning</h3>
</div>

## Overview

MatterVial is a featurizer tool designed for materials science, leveraging both graph-neural networks (GNNs) and traditional feature engineering to extract valuable chemical information from materials structures and compositions. It aims to enhance the performance of materials property prediction models by generating meaningful features for a variety of machine learning tasks. MatterVial stands for **MAT**erials fea**T**u**R**e **E**xtraction **V**ia **I**nterpretable **A**rtificial **L**earning, evoking the metaphor of a vial containing distilled knowledge from materials data, representing our tool's ability to extract and contain valuable materials insights.


## Available Featurizers

MatterVial offers a diverse set of feature extraction tools:

### Graph Neural Network (GNN) Featurizers

-   **MVLFeaturizer**: Extracts features from pretrained MEGNet models, trained on diverse datasets. Offers reliable features suitable for a wide range of properties. Users can select different intermediate layers (16-neuron or 32-neuron) from the regression head of the MEGNet models.

-   **CustomMEGNetFeaturizer**: Loads custom MEGNet models from the `mattervial/custom_models/` directory. These models can incorporate encoded chemical information derived from extensive featurizers such as MatMiner (l-MM, `MatMinerEncoded_v1` model) and Orbital Field Matrix (l-OFM, `OFMEncoded_v1` model).

-   **AdjacentMEGNetFeaturizer**: Trains a MEGNet model on-the-fly using the user's dataset. This ensures that the extracted features are closely aligned with the specific dataset, reducing potential bias and improving model accuracy. Users can extract features from either the 16-neuron or 32-neuron layer.

### Composition-Based Featurizers
-   **RoostModelFeaturizer**: Featurizer for ROOST models, which are suitable for directly featurizing the composition of the material, without requiring a structure.
    - `roost_mpgap`: Roost model pretrained on the MatProject bandgap dataset.
    - `roost_oqmd_eform`: Roost model pretrained on the OQMD formation energy dataset.

### SISSO-based Feature Augmentation
-   **SISSO-based Feature Augmentation**: Applies formulas obtained via Symbolic Regression through SISSO method (https://doi.org/10.1063/5.0156620) to generate features based on previously computed MatMiner features.  

The creation of new features is based on mathematical combinations of existing features to predict different targets in several datasets, guided by a master formulas file.

### Latent Space Featurizers
-   **LatentMMFeaturizer**: Extracts latent space features from MatMiner features.
-   **LatentOFMFeaturizer**: Extracts latent space features from Orbital Field Matrix (OFM) features.

## Feature Interpretation

The package includes a tool to interpret the features extracted:
-   **FeatureInterpreter**: Allows to retrieve feature-specific information from JSON files in designated `mattervial/data` (for shap and metrics info) and `mattervial/formulas` (for formula info) folders.

## Installation

To install MatterVial, clone the repository and use the following command:

```bash
pip install -r requirements.txt
```

Ensure you have all the necessary dependencies installed, including TensorFlow, scikit-learn, the MEGNet library and Pytorch.

## Usage

### Basic Examples

#### MEGNet Featurizers
Here's an example of how to use MatterVial to extract features from a list of structures:

```python
import pandas as pd
from mattervial import MVLFeaturizer, AdjacentMEGNetFeaturizer
from mattervial import get_Custom_MEGNetFeatures

# Initialize featurizers
mvl32 = MVLFeaturizer(layer_name='layer32')
mvl16 = MVLFeaturizer(layer_name='layer16')
adj_megnet = AdjacentMEGNetFeaturizer(layer_name='layer32')

# Example structures
structures = pd.DataFrame({'structure': [...]})  # Replace '...' with your actual list of structures

# Extract features using MVLFeaturizer
features_32 = mvl32.get_features(structures)
features_16 = mvl16.get_features(structures)

# Train the AdjacentMEGNetFeaturizer on the fly
# Provide in 'targets' some property values corresponding to these structures
adj_megnet.train_adjacent_megnet(structures, targets=[...])

# Extract features using the trained AdjacentMEGNetFeaturizer
features_adj = adj_megnet.get_features(structures)

# Load features using custom model
features = get_Custom_MEGNetFeatures(structures, model_type='OFMEncoded_v1')
```

#### ROOST Featurizers
Here's an example of how to use MatterVial to extract features from a list of compositions:

```python
import pandas as pd
from mattervial import RoostModelFeaturizer

# Initialize ROOST featurizers
roost_mpgap = RoostModelFeaturizer(model_type='mpgap')
roost_oqmd_eform = RoostModelFeaturizer(model_type='oqmd_eform')

# Example compositions
compositions = pd.Series(["Fe2O3", "Al2O3"])

# Extract features using roost_mpgap
features_mpgap = roost_mpgap.get_features(compositions)

# Extract features using roost_oqmd_eform
features_oqmd_eform = roost_oqmd_eform.get_features(compositions)
```

#### SISSO Featurization

Here's an example of how to use MatterVial to extract features with SISSO:

```python
from mattervial.featurizers import get_sisso_features
# Assuming 'dataset_MatMinerFeaturized.csv' contains your initial featurized data
# and has a 'target' column and other feature columns.
sisso_features_df = get_sisso_features(input_csv_path="dataset_MatMinerFeaturized.csv", type="SISSO_FORMULAS_v1")

# Now 'sisso_features_df' contains only the newly generated SISSO_ features.
# You can merge this DataFrame with your original data if needed.
```

#### Feature Interpretation

Here's an example of how to use MatterVial to interpret the features extracted:

```python
from mattervial import FeatureInterpreter
import json

# Example with a single feature.
feature_info = FeatureInterpreter("MEGNet_MatMiner_1")
print("Single feature query:")
print(json.dumps(feature_info, indent=4))

# Example with multiple features and with all output enabled.
example_features = ["MEGNet_MatMiner_3", "MEGNet_MatMiner_5", "MEGNet_MatMiner_10"]
prefs = {"shap": True, "formula": True, "metrics": True, "model_info": True}
feature_info_list = FeatureInterpreter(example_features, preferences=prefs)
print("\nMultiple feature query:")
print(json.dumps(feature_info_list, indent=4))
```

## Contributions

We welcome contributions to improve the MatterVial tool, including adding more pretrained models, enhancing the featurization techniques, or improving the feature interpretation capabilities. Please feel free to submit pull requests or create issues for discussion.

## License

This project is licensed under the MIT License.

## Acknowledgments

The MatterVial tool is built on top of other software packages and publicly available GNN models such as MEGNet and ROOST. We also acknowledge the developers of the SISSO package which was used to augment MatMiner featurizers via symbolic regression.
