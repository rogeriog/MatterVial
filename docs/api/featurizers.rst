Featurizers
===========

This module contains all the featurization classes and functions for extracting features from materials data.

Structure-Based Featurizers
---------------------------

These featurizers work with crystal structures (pymatgen Structure objects).

Latent Space Featurizers
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: mattervial.featurizers.LatentMMFeaturizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mattervial.featurizers.LatentOFMFeaturizer
   :members:
   :undoc-members:
   :show-inheritance:

MEGNet-Based Featurizers
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: mattervial.featurizers.MVLFeaturizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mattervial.featurizers.AdjacentMEGNetFeaturizer
   :members:
   :undoc-members:
   :show-inheritance:

MLIP-Based Featurizers
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: mattervial.featurizers.ORBFeaturizer
   :members:
   :undoc-members:
   :show-inheritance:

Composition-Based Featurizers
-----------------------------

These featurizers work with material compositions (chemical formulas).

.. autoclass:: mattervial.featurizers.RoostModelFeaturizer
   :members:
   :undoc-members:
   :show-inheritance:

Symbolic Regression Features
---------------------------

.. autofunction:: mattervial.featurizers.get_sisso_features

Utility Functions
----------------

.. autofunction:: mattervial.featurizers.get_available_featurizers

.. autofunction:: mattervial.featurizers.get_featurizer_errors

Predefined Featurizer Instances
-------------------------------

For convenience, MatterVial provides predefined instances of commonly used featurizers:

Structure Featurizers
~~~~~~~~~~~~~~~~~~~~~

.. data:: mattervial.featurizers.l_MM_v1
   
   Predefined LatentMMFeaturizer instance for ℓ-MM features.

.. data:: mattervial.featurizers.l_OFM_v1
   
   Predefined LatentOFMFeaturizer instance for ℓ-OFM features.

.. data:: mattervial.featurizers.mvl32
   
   Predefined MVLFeaturizer instance for layer32 features only.

.. data:: mattervial.featurizers.mvl16
   
   Predefined MVLFeaturizer instance for layer16 features only.

.. data:: mattervial.featurizers.mvl_all
   
   Predefined MVLFeaturizer instance for both layer32 and layer16 features.

.. data:: mattervial.featurizers.adj_megnet
   
   Predefined AdjacentMEGNetFeaturizer instance for layer32 features.

.. data:: mattervial.featurizers.adj_megnet_layer16
   
   Predefined AdjacentMEGNetFeaturizer instance for layer16 features.

.. data:: mattervial.featurizers.adj_megnet_all
   
   Predefined AdjacentMEGNetFeaturizer instance for both layers.

.. data:: mattervial.featurizers.orb_v3
   
   Predefined ORBFeaturizer instance for ORB-v3 model.

Composition Featurizers
~~~~~~~~~~~~~~~~~~~~~~

.. data:: mattervial.featurizers.roost_mpgap
   
   Predefined RoostModelFeaturizer instance for Materials Project band gap model.

.. data:: mattervial.featurizers.roost_oqmd_eform
   
   Predefined RoostModelFeaturizer instance for OQMD formation energy model.
