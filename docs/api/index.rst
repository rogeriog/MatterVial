API Reference
=============

This section contains the complete API reference for MatterVial, organized by module and functionality.

.. toctree::
   :maxdepth: 2

   featurizers
   interpreter
   utilities

Core Modules
------------

.. autosummary::
   :toctree: _autosummary
   :template: module.rst

   mattervial.featurizers
   mattervial.interpreter

Featurizers Overview
-------------------

Structure-Based Featurizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   mattervial.featurizers.LatentMMFeaturizer
   mattervial.featurizers.LatentOFMFeaturizer
   mattervial.featurizers.MVLFeaturizer
   mattervial.featurizers.AdjacentMEGNetFeaturizer
   mattervial.featurizers.ORBFeaturizer

Composition-Based Featurizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   mattervial.featurizers.RoostModelFeaturizer

Symbolic Regression
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   mattervial.featurizers.get_sisso_features

Interpretability Tools
---------------------

.. autosummary::
   :toctree: _autosummary

   mattervial.interpreter.Interpreter
   mattervial.interpreter.encode_ofm
   mattervial.interpreter.decode_ofm

Utility Functions
----------------

.. autosummary::
   :toctree: _autosummary

   mattervial.featurizers.get_available_featurizers
   mattervial.featurizers.get_featurizer_errors
