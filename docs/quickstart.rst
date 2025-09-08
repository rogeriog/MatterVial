Quick Start Guide
=================

This guide will help you get started with MatterVial quickly and efficiently.

Installation
------------

MatterVial uses different conda environments for different featurizers to manage conflicting dependencies.

Primary Environment
~~~~~~~~~~~~~~~~~~~

This environment supports the main featurizers including MEGNet-based models and ROOST:

.. code-block:: bash

   conda env create -f envs/env_primary.yml
   conda activate env_primary

ORB Environment
~~~~~~~~~~~~~~~

This specialized environment is required only for the ORB-v3 MLIP-based featurizer:

.. code-block:: bash

   conda env create -f envs/env_orb.yml
   conda activate env_orb

KGCNN Environment
~~~~~~~~~~~~~~~~~

This environment is required only for coGN/coGNN models:

.. code-block:: bash

   conda env create -f envs/env_kgcnn.yml
   conda activate env_kgcnn

Basic Usage
-----------

Structure-Based Featurization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from mattervial import MVLFeaturizer, LatentMMFeaturizer, LatentOFMFeaturizer

   # Prepare your structures (pymatgen Structure objects)
   structures = pd.Series([structure1, structure2, structure3])

   # Initialize featurizers
   mvl_featurizer = MVLFeaturizer()  # Default: both layer32 and layer16
   l_mm_featurizer = LatentMMFeaturizer()  # ℓ-MM featurizer
   l_ofm_featurizer = LatentOFMFeaturizer()  # ℓ-OFM featurizer

   # Extract features
   mvl_features = mvl_featurizer.get_features(structures)  # 240 features
   l_mm_features = l_mm_featurizer.get_features(structures)  # 758 features
   l_ofm_features = l_ofm_featurizer.get_features(structures)  # 188 features

   print(f"MVL features shape: {mvl_features.shape}")
   print(f"ℓ-MM features shape: {l_mm_features.shape}")
   print(f"ℓ-OFM features shape: {l_ofm_features.shape}")

Composition-Based Featurization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mattervial import RoostModelFeaturizer

   # Initialize ROOST featurizers
   roost_gap = RoostModelFeaturizer(model_type='mpgap')
   roost_eform = RoostModelFeaturizer(model_type='oqmd_eform')

   # Prepare compositions
   compositions = pd.Series(["Fe2O3", "Al2O3", "TiO2", "CaTiO3"])

   # Extract features
   gap_features = roost_gap.get_features(compositions)
   eform_features = roost_eform.get_features(compositions)

   print(f"Band gap features shape: {gap_features.shape}")
   print(f"Formation energy features shape: {eform_features.shape}")

Adjacent Model Training
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mattervial import AdjacentMEGNetFeaturizer

   # Initialize adjacent featurizer
   adj_featurizer = AdjacentMEGNetFeaturizer(layers='layer32')

   # Prepare training data
   train_structures = pd.Series([struct1, struct2, struct3])
   train_targets = [1.2, 2.3, 0.8]  # Target property values

   # Train the adjacent model
   adj_featurizer.train_adjacent_megnet(
       structures=train_structures,
       targets=train_targets,
       adjacent_model_path='./models/',
       max_epochs=100
   )

   # Extract features from the trained model
   test_structures = pd.Series([test_struct1, test_struct2])
   adj_features = adj_featurizer.get_features(test_structures)

   print(f"Adjacent features shape: {adj_features.shape}")

SISSO Feature Generation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mattervial.featurizers import get_sisso_features

   # Assuming you have a dataset with MatMiner features
   sisso_features = get_sisso_features(
       input_data="dataset_MatMinerFeaturized.csv",
       type="SISSO_FORMULAS_v1"
   )

   print(f"SISSO features shape: {sisso_features.shape}")

Feature Interpretation
---------------------

Understanding Your Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mattervial.interpreter import Interpreter

   # Initialize interpreter
   interpreter = Interpreter()

   # Get formula for a latent feature
   formula_info = interpreter.get_formula("l-OFM_v1_1")
   print(f"Formula: {formula_info['formula']}")
   print(f"R² score: {formula_info.get('r2_score', 'N/A')}")

   # Get SHAP values for feature importance
   shap_data = interpreter.get_shap_values("MEGNet_MatMiner_1")
   print("Top contributing features:")
   for feature, importance in list(shap_data.get('top_features', {}).items())[:5]:
       print(f"  {feature}: {importance}")

   # Interpret SISSO formulas
   sisso_info = interpreter.get_formula("SISSO_matbench_dielectric_1")
   print(f"SISSO formula: {sisso_info['formatted_formula']}")

Checking Available Featurizers
------------------------------

.. code-block:: python

   from mattervial.featurizers import get_available_featurizers, get_featurizer_errors

   # Check which featurizers are available in your environment
   available = get_available_featurizers()
   print("Available featurizers:")
   for name, status in available.items():
       print(f"  {name}: {status}")

   # Check for any errors
   errors = get_featurizer_errors()
   if errors:
       print("\nFeaturizer errors:")
       for name, error in errors.items():
           print(f"  {name}: {error}")

Common Workflows
---------------

Complete Feature Extraction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from mattervial import MVLFeaturizer, LatentMMFeaturizer, RoostModelFeaturizer
   from mattervial.featurizers import get_sisso_features

   # Prepare data
   structures = pd.Series([struct1, struct2, struct3])
   compositions = pd.Series(["Fe2O3", "Al2O3", "TiO2"])

   # Extract different types of features
   mvl_featurizer = MVLFeaturizer()
   l_mm_featurizer = LatentMMFeaturizer()
   roost_featurizer = RoostModelFeaturizer(model_type='mpgap')

   mvl_features = mvl_featurizer.get_features(structures)
   l_mm_features = l_mm_featurizer.get_features(structures)
   roost_features = roost_featurizer.get_features(compositions)

   # Combine features (ensure same number of samples)
   if len(structures) == len(compositions):
       combined_features = pd.concat([
           mvl_features, 
           l_mm_features, 
           roost_features
       ], axis=1)
       print(f"Combined features shape: {combined_features.shape}")

Feature Analysis and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mattervial.interpreter import Interpreter
   import pandas as pd

   # Extract features
   l_mm_featurizer = LatentMMFeaturizer()
   features = l_mm_featurizer.get_features(structures)

   # Analyze top features
   interpreter = Interpreter()
   feature_analysis = {}

   for col in features.columns[:10]:  # Analyze first 10 features
       try:
           formula_info = interpreter.get_formula(col)
           shap_data = interpreter.get_shap_values(col)
           
           feature_analysis[col] = {
               'r2_score': formula_info.get('r2_score', 'N/A'),
               'top_shap_feature': list(shap_data.get('top_features', {}).keys())[0] if shap_data.get('top_features') else 'N/A'
           }
       except Exception as e:
           feature_analysis[col] = {'error': str(e)}

   # Display analysis
   analysis_df = pd.DataFrame(feature_analysis).T
   print(analysis_df)

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **Import Errors**: Make sure you're using the correct conda environment
2. **Missing Model Files**: Some featurizers require pretrained models to be downloaded
3. **Memory Issues**: Large datasets may require batch processing
4. **GPU Issues**: ORB featurizer works best with GPU but falls back to CPU

Environment Debugging
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check your current environment
   import sys
   print(f"Python executable: {sys.executable}")

   # Test imports
   try:
       import mattervial
       print(f"MatterVial version: {mattervial.__version__}")
   except ImportError as e:
       print(f"MatterVial import error: {e}")

   # Check specific dependencies
   dependencies = ['pandas', 'numpy', 'sklearn', 'keras', 'megnet', 'pymatgen']
   for dep in dependencies:
       try:
           __import__(dep)
           print(f"✓ {dep} available")
       except ImportError:
           print(f"✗ {dep} not available")

Next Steps
----------

- Explore the :doc:`api/index` for detailed API documentation
- Check out :doc:`examples/index` for more comprehensive examples
- Learn about :doc:`interpretation` for understanding your features
- Read about :doc:`environments` for advanced environment management
