# Lazy imports - these will be imported only when needed
# from .megnet_models import ( get_MVL_MEGNetFeatures, get_Custom_MEGNetFeatures,
#                            train_MEGNet_on_the_fly, get_Adjacent_MEGNetFeatures )
# from .mlip_models import get_ORB_features

import warnings

class DescriptorMEGNetFeaturizer:
    """
    Descriptor-oriented MEGNet featurizer for retrieving encoded features.

    This featurizer implements the DescriptorMEGNetFeaturizer class mentioned in the
    methods section. It provides a unified interface for retrieving OFM-encoded and
    MatMiner-encoded features from pretrained MEGNet models, allowing users to access
    both ℓ-MM (latent MatMiner) and ℓ-OFM (latent OFM) features through a single class.

    The class supports different base descriptors through the base_descriptor parameter:
    - 'l-MM_v1': Latent MatMiner features (758 features)
    - 'l-OFM_v1': Latent Orbital Field Matrix features (188 features)

    This unified approach provides flexibility in choosing the appropriate latent
    representation while maintaining a consistent interface for feature extraction.

    Attributes:
        base_descriptor (str): The base descriptor type ('l-MM_v1' or 'l-OFM_v1').

    Examples:
        >>> # Default: ℓ-MM features
        >>> desc_featurizer = DescriptorMEGNetFeaturizer()
        >>> features = desc_featurizer.get_features(structures)
        >>> print(features.shape)
        (100, 758)  # 758 ℓ-MM features

        >>> # ℓ-OFM features
        >>> desc_ofm = DescriptorMEGNetFeaturizer(base_descriptor='l-OFM_v1')
        >>> features = desc_ofm.get_features(structures)
        >>> print(features.shape)
        (100, 188)  # 188 ℓ-OFM features
    """

    def __init__(self, base_descriptor='l-MM_v1'):
        """
        Initialize DescriptorMEGNetFeaturizer.

        Args:
            base_descriptor (str): Base descriptor type to use. Options:
                - 'l-MM_v1' (default): Latent MatMiner features (758 features)
                - 'l-OFM_v1': Latent Orbital Field Matrix features (188 features)

        Raises:
            ValueError: If invalid base_descriptor is provided.

        Examples:
            >>> # Default ℓ-MM featurizer
            >>> featurizer = DescriptorMEGNetFeaturizer()

            >>> # ℓ-OFM featurizer
            >>> featurizer = DescriptorMEGNetFeaturizer(base_descriptor='l-OFM_v1')
        """
        # Validate base_descriptor
        valid_descriptors = ['l-MM_v1', 'l-OFM_v1']
        if base_descriptor not in valid_descriptors:
            raise ValueError(f"Invalid base_descriptor '{base_descriptor}'. Valid options: {valid_descriptors}")

        self.base_descriptor = base_descriptor

        # Initialize the appropriate backend featurizer
        if self.base_descriptor == 'l-MM_v1':
            self._backend_featurizer = LatentMMFeaturizer()
        elif self.base_descriptor == 'l-OFM_v1':
            self._backend_featurizer = LatentOFMFeaturizer()

    def get_features(self, structures):
        """
        Extract descriptor-oriented MEGNet features from crystal structures.

        This method retrieves encoded features from the MatterVial package using
        the specified base descriptor type. The features are extracted from pretrained
        MEGNet models that encode traditional descriptors into latent representations.

        Args:
            structures (pd.Series or pd.DataFrame or list): Input structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.

        Returns:
            pd.DataFrame: DataFrame containing descriptor-oriented MEGNet features.
                - For 'l-MM_v1': (n_structures, 758) with names like 'MEGNet_MatMiner_X'
                - For 'l-OFM_v1': (n_structures, 188) with names like 'MEGNet_OFMEncoded_v1_X'

        Raises:
            ImportError: If MEGNet dependencies are not available.
            ValueError: If the input structures are invalid or empty.

        Examples:
            >>> import pandas as pd
            >>> from pymatgen.core import Structure
            >>> structures = pd.Series([structure1, structure2])
            >>>
            >>> # Extract ℓ-MM features
            >>> desc_mm = DescriptorMEGNetFeaturizer(base_descriptor='l-MM_v1')
            >>> mm_features = desc_mm.get_features(structures)
            >>> print(f"ℓ-MM features: {mm_features.shape}")
            >>>
            >>> # Extract ℓ-OFM features
            >>> desc_ofm = DescriptorMEGNetFeaturizer(base_descriptor='l-OFM_v1')
            >>> ofm_features = desc_ofm.get_features(structures)
            >>> print(f"ℓ-OFM features: {ofm_features.shape}")
        """
        return self._backend_featurizer.get_features(structures)

class LatentMMFeaturizer:
    """
    Latent space featurizer for MatMiner features (ℓ-MM featurizer).

    .. deprecated:: 0.1.5
        LatentMMFeaturizer is deprecated. Use DescriptorMEGNetFeaturizer with
        base_descriptor='l-MM_v1' instead. This class will be removed in a future version.

    This featurizer extracts latent space features from MatMiner features using a
    pretrained MEGNet model. The model encodes traditional MatMiner features into
    a compressed latent representation that captures essential chemical information
    while reducing dimensionality.

    The ℓ-MM featurizer follows the procedure described in the MatterVial paper:
    MatMiner features are applied to the MP2018-stable dataset, followed by training
    an autoencoder to derive a latent space representation. The latent MatMiner
    features are then used as targets to train a GNN model that generates these
    features directly from structures.

    Attributes:
        model_type (str): The model type used ('MatMinerEncoded_v1').

    Examples:
        >>> # Deprecated usage (still works)
        >>> featurizer = LatentMMFeaturizer()
        >>> features = featurizer.get_features(structures)
        >>> print(features.shape)
        (100, 758)  # 758 latent MatMiner features

        >>> # Recommended new usage
        >>> featurizer = DescriptorMEGNetFeaturizer(base_descriptor='l-MM_v1')
        >>> features = featurizer.get_features(structures)
    """

    def __init__(self):
        """Initialize LatentMMFeaturizer with deprecation warning."""
        warnings.warn(
            "LatentMMFeaturizer is deprecated and will be removed in a future version. "
            "Use DescriptorMEGNetFeaturizer with base_descriptor='l-MM_v1' instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def get_features(self, structures):
        """
        Extract latent MatMiner features from crystal structures.

        Args:
            structures (pd.Series or pd.DataFrame or list): Input structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.

        Returns:
            pd.DataFrame: DataFrame containing latent MatMiner features with shape
                (n_structures, 758). Column names follow the pattern 'MEGNet_MatMiner_X'
                where X is the feature index.

        Raises:
            ImportError: If MEGNet dependencies are not available.
            ValueError: If the input structures are invalid or empty.

        Examples:
            >>> import pandas as pd
            >>> from pymatgen.core import Structure
            >>> structures = pd.Series([structure1, structure2])
            >>> featurizer = LatentMMFeaturizer()
            >>> features = featurizer.get_features(structures)
        """
        # Logic for latent space MatMiner feature extraction
        try:
            from .megnet_models import get_Custom_MEGNetFeatures
            return get_Custom_MEGNetFeatures(structures, model_type='MatMinerEncoded_v1')
        except ImportError as e:
            raise ImportError(f"MEGNet dependencies not available for LatentMMFeaturizer: {e}. "
                            f"Please install required packages (keras, megnet, etc.) or use a different environment.")

# Create featurizer instances with error handling
_featurizer_instances = {}
_featurizer_errors = {}

def _create_featurizer_safely(name, featurizer_class, *args, **kwargs):
    """Create a featurizer instance safely, catching import errors."""
    try:
        instance = featurizer_class(*args, **kwargs)
        _featurizer_instances[name] = instance
        return instance
    except ImportError as e:
        _featurizer_errors[name] = str(e)
        return None

# Try to create DescriptorMEGNetFeaturizer instances (new recommended approach)
l_MM_v1 = _create_featurizer_safely("l_MM_v1", DescriptorMEGNetFeaturizer, base_descriptor='l-MM_v1')

# Backward compatibility: create deprecated LatentMMFeaturizer instance
l_MM_v1_deprecated = _create_featurizer_safely("l_MM_v1_deprecated", LatentMMFeaturizer)

class LatentOFMFeaturizer:
    """
    Latent space featurizer for Orbital Field Matrix (OFM) features (ℓ-OFM featurizer).

    .. deprecated:: 0.1.5
        LatentOFMFeaturizer is deprecated. Use DescriptorMEGNetFeaturizer with
        base_descriptor='l-OFM_v1' instead. This class will be removed in a future version.

    This featurizer extracts latent space features from Orbital Field Matrix (OFM)
    features using a pretrained MEGNet model. The OFM featurizer captures valence
    electron interactions at each atomic site by employing a weighted vector outer
    product of one-hot encoded valence orbitals for every atom.

    The ℓ-OFM featurizer follows the procedure described in the MatterVial paper:
    The OFM featurizer is applied to the MP2018-stable dataset, followed by training
    an autoencoder to derive a latent space representation. The latent OFM features
    are then used as targets to train a GNN model that generates these features
    directly from structures.

    Attributes:
        model_type (str): The model type used ('OFMEncoded_v1').

    Examples:
        >>> # Deprecated usage (still works)
        >>> featurizer = LatentOFMFeaturizer()
        >>> features = featurizer.get_features(structures)
        >>> print(features.shape)
        (100, 188)  # 188 latent OFM features

        >>> # Recommended new usage
        >>> featurizer = DescriptorMEGNetFeaturizer(base_descriptor='l-OFM_v1')
        >>> features = featurizer.get_features(structures)
    """

    def __init__(self):
        """Initialize LatentOFMFeaturizer with deprecation warning."""
        warnings.warn(
            "LatentOFMFeaturizer is deprecated and will be removed in a future version. "
            "Use DescriptorMEGNetFeaturizer with base_descriptor='l-OFM_v1' instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def get_features(self, structures):
        """
        Extract latent Orbital Field Matrix (OFM) features from crystal structures.

        Args:
            structures (pd.Series or pd.DataFrame or list): Input structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.

        Returns:
            pd.DataFrame: DataFrame containing latent OFM features with shape
                (n_structures, 188). Column names follow the pattern 'MEGNet_OFMEncoded_v1_X'
                where X is the feature index.

        Raises:
            ImportError: If MEGNet dependencies are not available.
            ValueError: If the input structures are invalid or empty.

        Examples:
            >>> import pandas as pd
            >>> from pymatgen.core import Structure
            >>> structures = pd.Series([structure1, structure2])
            >>> featurizer = LatentOFMFeaturizer()
            >>> features = featurizer.get_features(structures)
        """
        # Logic for latent space OFM feature extraction
        try:
            from .megnet_models import get_Custom_MEGNetFeatures
            return get_Custom_MEGNetFeatures(structures, model_type='OFMEncoded_v1')
        except ImportError as e:
            raise ImportError(f"MEGNet dependencies not available for LatentOFMFeaturizer: {e}. "
                            f"Please install required packages (keras, megnet, etc.) or use a different environment.")

# Try to create DescriptorMEGNetFeaturizer instance for OFM (new recommended approach)
l_OFM_v1 = _create_featurizer_safely("l_OFM_v1", DescriptorMEGNetFeaturizer, base_descriptor='l-OFM_v1')

# Backward compatibility: create deprecated LatentOFMFeaturizer instance
l_OFM_v1_deprecated = _create_featurizer_safely("l_OFM_v1_deprecated", LatentOFMFeaturizer)

class MVLFeaturizer:
    """
    Materials Virtual Lab (MVL) featurizer using pretrained MEGNet models.

    This featurizer extracts features from pretrained MEGNet models provided by the
    Materials Virtual Lab. It utilizes five pretrained MEGNet models trained for
    formation energy, Fermi energy, elastic constants (KVRH and GVRH), and band gap
    regression on Materials Project datasets.

    The MVL featurizer allows extraction of features from different layers of the
    pretrained models, specifically from the MLP layers preceding the output
    (32-neuron and 16-neuron configurations). Features from both layers can be
    concatenated to provide a comprehensive representation.

    Attributes:
        layers (list): List of layer names to extract features from.
        layer_name (str or None): For backward compatibility, single layer name.

    Examples:
        >>> # Single layer extraction
        >>> mvl32 = MVLFeaturizer(layers='layer32')
        >>> features = mvl32.get_features(structures)
        >>> print(features.shape)
        (100, 160)  # 160 features from 32-neuron layer

        >>> # Multi-layer extraction (default)
        >>> mvl_all = MVLFeaturizer()  # Uses both layer32 and layer16
        >>> features = mvl_all.get_features(structures)
        >>> print(features.shape)
        (100, 240)  # 160 + 80 features combined
    """

    def __init__(self, layers=['layer32', 'layer16']):
        """
        Initialize MVLFeaturizer.

        Args:
            layers (list or str): List of layer names to extract features from,
                or single layer name string. Options: ['layer32', 'layer16'] (default),
                ['layer32'], ['layer16'], 'layer32', 'layer16'.

        Raises:
            ValueError: If invalid layer names are provided.

        Examples:
            >>> # Default: extract from both layers
            >>> featurizer = MVLFeaturizer()

            >>> # Single layer extraction
            >>> featurizer = MVLFeaturizer(layers='layer32')

            >>> # Custom multi-layer
            >>> featurizer = MVLFeaturizer(layers=['layer32', 'layer16'])
        """
        # Handle backward compatibility - convert single string to list
        if isinstance(layers, str):
            self.layers = [layers]
        else:
            self.layers = layers if isinstance(layers, list) else [layers]

        # Validate layer names
        valid_layers = ['layer32', 'layer16']
        for layer in self.layers:
            if layer not in valid_layers:
                raise ValueError(f"Invalid layer name '{layer}'. Valid options: {valid_layers}")

        # For backward compatibility, keep layer_name attribute for single layer
        self.layer_name = self.layers[0] if len(self.layers) == 1 else None

    def get_features(self, structures):
        """
        Extract features from pretrained MVL MEGNet models.

        Args:
            structures (pd.Series or pd.DataFrame or list): Input structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.

        Returns:
            pd.DataFrame: DataFrame containing MVL features. For single layer:
                - layer32: (n_structures, 160) features with names like 'MVL32_Eform_MP_2019_X'
                - layer16: (n_structures, 80) features with names like 'MVL16_Eform_MP_2019_X'
                For multiple layers: concatenated features from all specified layers.

        Raises:
            ImportError: If MEGNet dependencies are not available.
            ValueError: If the input structures are invalid or empty.

        Examples:
            >>> import pandas as pd
            >>> from pymatgen.core import Structure
            >>> structures = pd.Series([structure1, structure2])
            >>> mvl_featurizer = MVLFeaturizer(layers='layer32')
            >>> features = mvl_featurizer.get_features(structures)
            >>> print(features.columns[:3])
            ['MVL32_Eform_MP_2019_1', 'MVL32_Eform_MP_2019_2', 'MVL32_Eform_MP_2019_3']
        """
        try:
            from .megnet_models import get_MVL_MEGNetFeatures

            if len(self.layers) == 1:
                # Single layer - pass layer_name parameter for efficiency
                return get_MVL_MEGNetFeatures(structures, layer_name=self.layers[0])
            else:
                # Multiple layers - use single-pass optimization
                # Get all features in one pass (both layer32 and layer16)
                all_features = get_MVL_MEGNetFeatures(structures)

                # Filter to only the requested layers
                requested_columns = []
                for layer_name in self.layers:
                    layer_prefix = 'MVL32' if layer_name == 'layer32' else 'MVL16'
                    # Find columns that start with the layer prefix
                    layer_columns = [col for col in all_features.columns if col.startswith(layer_prefix)]
                    requested_columns.extend(layer_columns)

                # Return only the requested layer features
                return all_features[requested_columns]

        except ImportError as e:
            raise ImportError(f"MEGNet dependencies not available for MVLFeaturizer: {e}. "
                            f"Please install required packages (keras, megnet, etc.) or use a different environment.")

# Try to create MVLFeaturizer instances
# Backward compatibility: single-layer instances
mvl32 = _create_featurizer_safely("mvl32", MVLFeaturizer, layers='layer32')
mvl16 = _create_featurizer_safely("mvl16", MVLFeaturizer, layers='layer16')

# New default: multi-layer instance
mvl_all = _create_featurizer_safely("mvl_all", MVLFeaturizer, layers=['layer32', 'layer16'])

class AdjacentGNNFeaturizer:
    """
    Adjacent GNN featurizer that trains task-specific models on-the-fly.

    This featurizer implements the Adjacent GNN featurizer described in the methods
    section. It trains a GNN model on-the-fly using the user's dataset for each fold
    of the train-test split. This adjacent model captures task-specific data nuances,
    enhancing prediction accuracy by reducing potential bias from pretrained models.

    The featurizer supports different base GNN architectures through the base_model
    parameter, allowing users to choose between MEGNet and coGN models depending on
    their specific requirements and available computational resources.

    Attributes:
        base_model (str): The base GNN architecture to use ('MEGNet' or 'coGN').
        layers (list): List of layer names to extract features from.
        layer_name (str or None): For backward compatibility, single layer name.
        model_path (str): Path where the trained model will be saved/loaded.

    Examples:
        >>> # Default MEGNet-based adjacent featurizer
        >>> adj_featurizer = AdjacentGNNFeaturizer()
        >>> adj_featurizer.train_adjacent_model(structures, targets)
        >>> features = adj_featurizer.get_features(structures)

        >>> # coGN-based adjacent featurizer (when available)
        >>> adj_cogn = AdjacentGNNFeaturizer(base_model='coGN')
    """

    def __init__(self, base_model='MEGNet', layers=['layer32', 'layer16'], **kwargs):
        """
        Initialize AdjacentGNNFeaturizer.

        Args:
            base_model (str): Base GNN architecture to use. Options: 'MEGNet' (default), 'coGN'.
                - 'MEGNet': Uses MEGNet architecture for adjacent model training
                - 'coGN': Uses coGN architecture (placeholder - will be supported soon)
            layers (list or str): List of layer names to extract features from,
                or single layer name string. Options: ['layer32', 'layer16'] (default),
                ['layer32'], ['layer16'], 'layer32', 'layer16'.
            **kwargs: Additional arguments including:
                - model_path (str): Path where model files will be saved/loaded.
                  Defaults to './'.

        Raises:
            ValueError: If invalid base_model or layer names are provided.
            NotImplementedError: If coGN model is requested (temporary).

        Examples:
            >>> # Default MEGNet configuration
            >>> featurizer = AdjacentGNNFeaturizer()

            >>> # Single layer MEGNet with custom path
            >>> featurizer = AdjacentGNNFeaturizer(
            ...     base_model='MEGNet',
            ...     layers='layer32',
            ...     model_path='/path/to/models/'
            ... )

            >>> # coGN model (placeholder)
            >>> featurizer = AdjacentGNNFeaturizer(base_model='coGN')
        """
        # Validate base_model
        valid_models = ['MEGNet', 'coGN']
        if base_model not in valid_models:
            raise ValueError(f"Invalid base_model '{base_model}'. Valid options: {valid_models}")

        self.base_model = base_model

        # Handle backward compatibility - convert single string to list
        if isinstance(layers, str):
            self.layers = [layers]
        else:
            self.layers = layers if isinstance(layers, list) else [layers]

        # Validate layer names
        valid_layers = ['layer32', 'layer16']
        for layer in self.layers:
            if layer not in valid_layers:
                raise ValueError(f"Invalid layer name '{layer}'. Valid options: {valid_layers}")

        # For backward compatibility, keep layer_name attribute for single layer
        self.layer_name = self.layers[0] if len(self.layers) == 1 else None
        self.model_path = kwargs.get('model_path', './')

        # Initialize the appropriate backend featurizer
        if self.base_model == 'MEGNet':
            self._backend_featurizer = AdjacentMEGNetFeaturizer(layers=layers, **kwargs)
        elif self.base_model == 'coGN':
            # Placeholder for coGN implementation
            self._backend_featurizer = None

    def train_adjacent_model(self, structures, targets, **kwargs):
        """
        Train an adjacent GNN model on the provided dataset.

        This method trains a GNN model specifically on the user's dataset to
        capture task-specific patterns and nuances. The trained model is saved
        and can be used later for feature extraction.

        Args:
            structures (pd.Series or pd.DataFrame or list): Training structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.
            targets (array-like): Target values corresponding to the structures.
                Should be a 1D array or list with the same length as structures.
            **kwargs: Additional training parameters passed to the underlying
                GNN training function.

        Raises:
            NotImplementedError: If coGN model is requested (temporary).
            ImportError: If required dependencies are not available.
            ValueError: If structures and targets have mismatched lengths.

        Examples:
            >>> adj_featurizer = AdjacentGNNFeaturizer(base_model='MEGNet')
            >>> adj_featurizer.train_adjacent_model(
            ...     structures=train_structures,
            ...     targets=train_targets,
            ...     adjacent_model_path='./models/',
            ...     max_epochs=100
            ... )
        """
        if self.base_model == 'MEGNet':
            return self._backend_featurizer.train_adjacent_megnet(structures, targets, **kwargs)
        elif self.base_model == 'coGN':
            print("coGN support will be adapted soon to the public package!")
            raise NotImplementedError("coGN support is not yet available in the public package. "
                                    "Please use base_model='MEGNet' for now.")

    def get_features(self, structures, **kwargs):
        """
        Extract features from the trained adjacent GNN model.

        This method extracts features from the previously trained adjacent GNN
        model. The model must be trained first using train_adjacent_model() method.

        Args:
            structures (pd.Series or pd.DataFrame or list): Input structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.
            **kwargs: Additional parameters passed to the feature extraction function.

        Returns:
            pd.DataFrame: DataFrame containing adjacent GNN features. The exact
                format depends on the base_model used:
                - MEGNet: Features with names like 'Adjacent_MEGNet_LayerSize_Index'
                - coGN: Features with names like 'Adjacent_coGN_LayerSize_Index' (when available)

        Raises:
            NotImplementedError: If coGN model is requested (temporary).
            ImportError: If required dependencies are not available.
            FileNotFoundError: If the trained model file is not found.
            ValueError: If the input structures are invalid or empty.

        Examples:
            >>> # After training the model
            >>> adj_featurizer = AdjacentGNNFeaturizer(base_model='MEGNet')
            >>> adj_featurizer.train_adjacent_model(train_structures, train_targets)
            >>>
            >>> # Extract features
            >>> features = adj_featurizer.get_features(test_structures)
            >>> print(features.shape)
            (50, 32)  # 32 features from layer32
        """
        if self.base_model == 'MEGNet':
            return self._backend_featurizer.get_features(structures, **kwargs)
        elif self.base_model == 'coGN':
            print("coGN support will be adapted soon to the public package!")
            raise NotImplementedError("coGN support is not yet available in the public package. "
                                    "Please use base_model='MEGNet' for now.")

class AdjacentMEGNetFeaturizer:
    """
    Adjacent MEGNet featurizer that trains task-specific models on-the-fly.

    .. deprecated:: 0.1.5
        AdjacentMEGNetFeaturizer is deprecated. Use AdjacentGNNFeaturizer with
        base_model='MEGNet' instead. This class will be removed in a future version.

    This featurizer trains a MEGNet model on-the-fly using the user's dataset to
    ensure that extracted features are closely aligned with the specific task and
    dataset. This approach captures task-specific data nuances and can enhance
    prediction accuracy by reducing potential bias from pretrained models.

    The adjacent model is trained for each fold of the train-test split, making it
    particularly suitable for cross-validation scenarios. The model uses default
    MEGNet hyperparameters and can extract features from different layers of the
    trained network.

    Note:
        This class is now superseded by AdjacentGNNFeaturizer, which provides
        the same functionality with support for multiple GNN architectures.

    Attributes:
        layers (list): List of layer names to extract features from.
        layer_name (str or None): For backward compatibility, single layer name.
        model_path (str): Path where the trained model will be saved/loaded.

    Examples:
        >>> # Deprecated usage (still works)
        >>> adj_featurizer = AdjacentMEGNetFeaturizer(layers='layer32')
        >>> adj_featurizer.train_adjacent_megnet(structures, targets)
        >>> features = adj_featurizer.get_features(structures)

        >>> # Recommended new usage
        >>> adj_featurizer = AdjacentGNNFeaturizer(base_model='MEGNet', layers='layer32')
        >>> adj_featurizer.train_adjacent_model(structures, targets)
        >>> features = adj_featurizer.get_features(structures)
    """

    def __init__(self, layers=['layer32', 'layer16'], **kwargs):
        """
        Initialize AdjacentMEGNetFeaturizer.

        Args:
            layers (list or str): List of layer names to extract features from,
                or single layer name string. Options: ['layer32', 'layer16'] (default),
                ['layer32'], ['layer16'], 'layer32', 'layer16'.
            **kwargs: Additional arguments including:
                - model_path (str): Path where model files will be saved/loaded.
                  Defaults to './'.

        Raises:
            ValueError: If invalid layer names are provided.

        Examples:
            >>> # Deprecated usage (still works)
            >>> featurizer = AdjacentMEGNetFeaturizer()

            >>> # Recommended new usage
            >>> featurizer = AdjacentGNNFeaturizer(base_model='MEGNet')
        """
        # Issue deprecation warning
        warnings.warn(
            "AdjacentMEGNetFeaturizer is deprecated and will be removed in a future version. "
            "Use AdjacentGNNFeaturizer with base_model='MEGNet' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Handle backward compatibility - convert single string to list
        if isinstance(layers, str):
            self.layers = [layers]
        else:
            self.layers = layers if isinstance(layers, list) else [layers]

        # Validate layer names
        valid_layers = ['layer32', 'layer16']
        for layer in self.layers:
            if layer not in valid_layers:
                raise ValueError(f"Invalid layer name '{layer}'. Valid options: {valid_layers}")

        # For backward compatibility, keep layer_name attribute for single layer
        self.layer_name = self.layers[0] if len(self.layers) == 1 else None
        self.model_path = kwargs.get('model_path', './')

    def train_adjacent_megnet(self, structures, targets, **kwargs):
        """
        Train an adjacent MEGNet model on the provided dataset.

        This method trains a MEGNet model specifically on the user's dataset to
        capture task-specific patterns and nuances. The trained model is saved
        and can be used later for feature extraction.

        Args:
            structures (pd.Series or pd.DataFrame or list): Training structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.
            targets (array-like): Target values corresponding to the structures.
                Should be a 1D array or list with the same length as structures.
            **kwargs: Additional training parameters passed to the MEGNet training
                function, such as:
                - adjacent_model_path (str): Path to save the trained model
                - max_epochs (int): Maximum number of training epochs
                - validation_split (float): Fraction of data for validation

        Raises:
            ImportError: If MEGNet dependencies are not available.
            ValueError: If structures and targets have mismatched lengths.

        Examples:
            >>> adj_featurizer = AdjacentMEGNetFeaturizer()
            >>> adj_featurizer.train_adjacent_megnet(
            ...     structures=train_structures,
            ...     targets=train_targets,
            ...     adjacent_model_path='./models/',
            ...     max_epochs=100
            ... )
        """
        try:
            from .megnet_models import train_MEGNet_on_the_fly
            train_MEGNet_on_the_fly(structures, targets, **kwargs)
        except ImportError as e:
            raise ImportError(f"MEGNet dependencies not available for AdjacentMEGNetFeaturizer training: {e}. "
                            f"Please install required packages (keras, megnet, etc.) or use a different environment.")

    def get_features(self, structures, **kwargs):
        """
        Extract features from the trained adjacent MEGNet model.

        This method extracts features from the previously trained adjacent MEGNet
        model. The model must be trained first using train_adjacent_megnet() method.

        Args:
            structures (pd.Series or pd.DataFrame or list): Input structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.
            **kwargs: Additional parameters passed to the feature extraction function,
                such as:
                - model_path (str): Path where the trained model is located

        Returns:
            pd.DataFrame: DataFrame containing adjacent MEGNet features. For single layer:
                - layer32: (n_structures, 32) features with names like 'Adjacent_MEGNet_32_X'
                - layer16: (n_structures, 16) features with names like 'Adjacent_MEGNet_16_X'
                For multiple layers: concatenated features from all specified layers.

        Raises:
            ImportError: If MEGNet dependencies are not available.
            FileNotFoundError: If the trained model file is not found.
            ValueError: If the input structures are invalid or empty.

        Examples:
            >>> # After training the model
            >>> adj_featurizer = AdjacentMEGNetFeaturizer(layers='layer32')
            >>> adj_featurizer.train_adjacent_megnet(train_structures, train_targets)
            >>>
            >>> # Extract features
            >>> features = adj_featurizer.get_features(test_structures)
            >>> print(features.shape)
            (50, 32)  # 32 features from layer32
        """
        try:
            from .megnet_models import get_Adjacent_MEGNetFeatures
            import pandas as pd

            if len(self.layers) == 1:
                # Single layer - maintain original behavior
                return get_Adjacent_MEGNetFeatures(structures, layer_name=self.layers[0],
                                                   **kwargs)
            else:
                # Multiple layers - extract and combine features
                combined_features = []

                for layer_name in self.layers:
                    layer_features = get_Adjacent_MEGNetFeatures(structures, layer_name=layer_name,
                                                                **kwargs)

                    # Rename columns to avoid redundant layer information
                    # Original features already contain layer info, no need for additional prefix
                    combined_features.append(layer_features)

                # Combine all layer features
                result = pd.concat(combined_features, axis=1)
                return result

        except ImportError as e:
            raise ImportError(f"MEGNet dependencies not available for AdjacentMEGNetFeaturizer: {e}. "
                            f"Please install required packages (keras, megnet, etc.) or use a different environment.")

# Try to create AdjacentGNNFeaturizer instances (new recommended approach)
# Single-layer instances
adj_megnet = _create_featurizer_safely("adj_megnet", AdjacentGNNFeaturizer, base_model='MEGNet', layers='layer32')
adj_megnet_layer16 = _create_featurizer_safely("adj_megnet_layer16", AdjacentGNNFeaturizer, base_model='MEGNet', layers='layer16')

# Multi-layer instance (default)
adj_megnet_all = _create_featurizer_safely("adj_megnet_all", AdjacentGNNFeaturizer, base_model='MEGNet', layers=['layer32', 'layer16'])

# Backward compatibility: create deprecated AdjacentMEGNetFeaturizer instances
# These will issue deprecation warnings when used
adj_megnet_deprecated = _create_featurizer_safely("adj_megnet_deprecated", AdjacentMEGNetFeaturizer, layers='layer32')
adj_megnet_layer16_deprecated = _create_featurizer_safely("adj_megnet_layer16_deprecated", AdjacentMEGNetFeaturizer, layers='layer16')
adj_megnet_all_deprecated = _create_featurizer_safely("adj_megnet_all_deprecated", AdjacentMEGNetFeaturizer, layers=['layer32', 'layer16'])

class ORBFeaturizer:
    """
    ORB (Orbital-based Representation) featurizer for crystal structures.

    This featurizer uses the ORB-v3 machine learning interatomic potential (MLIP)
    to extract features from crystal structures. ORB models are based on orbital-based
    representations and can provide high-quality structural embeddings for materials.

    The ORB featurizer requires a specialized conda environment with the orb-models
    package installed. It supports GPU acceleration when available and offers different
    precision modes for computation.

    Attributes:
        model_name (str): Name of the ORB model to use.
        device (str): Device for computation ('cuda' or 'cpu').
        precision (str): Precision mode for the model.

    Examples:
        >>> orb_featurizer = ORBFeaturizer(model_name="ORB_v3")
        >>> features = orb_featurizer.get_features(structures)
        >>> print(features.shape)
        
    """

    def __init__(self, model_name="ORB_v3", device=None, precision="float32-high"):
        """
        Initialize ORBFeaturizer.

        Args:
            model_name (str): Name of the ORB model to use. Defaults to "ORB_v3".
            device (str, optional): Device to run the model on. If None,
                automatically detects CUDA availability. Options: 'cuda', 'cpu'.
            precision (str): Model precision mode. Defaults to "float32-high".
                Options depend on the ORB model implementation.

        Examples:
            >>> # Default configuration
            >>> featurizer = ORBFeaturizer()

            >>> # Custom configuration
            >>> featurizer = ORBFeaturizer(
            ...     model_name="ORB_v3",
            ...     device="cuda",
            ...     precision="float32-high"
            ... )
        """
        self.model_name = model_name
        self.device = device
        self.precision = precision

    def get_features(self, structures):
        """
        Extract ORB features from crystal structures.

        This method processes the input structures through the ORB model to extract
        orbital-based representations. The features capture important structural and
        chemical information that can be used for downstream machine learning tasks.

        Args:
            structures (pd.Series or pd.DataFrame or list): Input structures.
                Can be a pandas Series/DataFrame with 'structure' column containing
                pymatgen Structure objects, or a list of Structure objects.

        Returns:
            pd.DataFrame: DataFrame containing ORB features with shape
                (n_structures, n_features). The number of features depends on the
                ORB model configuration. Column names follow the pattern 'ORB_X'
                where X is the feature index.

        Raises:
            ImportError: If ORB dependencies (orb-models package) are not available.
            ValueError: If the input structures are invalid or empty.
            RuntimeError: If there are issues with model loading or computation.

        Examples:
            >>> import pandas as pd
            >>> from pymatgen.core import Structure
            >>> structures = pd.Series([structure1, structure2])
            >>> orb_featurizer = ORBFeaturizer()
            >>> features = orb_featurizer.get_features(structures)
            >>> print(f"Extracted {features.shape[1]} ORB features")
            Extracted 1792 ORB features
        """
        try:
            from .mlip_models import get_ORB_features
            return get_ORB_features(structures,
                                   model_name=self.model_name,
                                   device=self.device,
                                   precision=self.precision)
        except ImportError as e:
            raise ImportError(f"ORB dependencies not available for ORBFeaturizer: {e}. "
                            f"Please install orb-models package or use the ORB environment.")

# Try to create ORB featurizer instance
orb_v3 = _create_featurizer_safely("orb_v3", ORBFeaturizer, model_name="ORB_v3")

# Only export successfully created featurizers
__all__ = []
featurizer_names = ["l_MM_v1", "l_OFM_v1", "mvl32", "mvl16", "mvl_all",
                   "adj_megnet", "adj_megnet_layer16", "adj_megnet_all", "orb_v3"]
for name in featurizer_names:
    if name in _featurizer_instances:
        __all__.append(name)

# Add featurizer classes to exports
__all__.extend(["DescriptorMEGNetFeaturizer", "LatentMMFeaturizer", "LatentOFMFeaturizer", "MVLFeaturizer",
               "AdjacentGNNFeaturizer", "AdjacentMEGNetFeaturizer", "ORBFeaturizer"])

def get_available_featurizers():
    """Return a dictionary of available featurizers and their status."""
    status = {}
    for name in featurizer_names:
        if name in _featurizer_instances:
            status[name] = "Available"
        elif name in _featurizer_errors:
            status[name] = f"Error: {_featurizer_errors[name]}"
        else:
            status[name] = "Unknown"
    return status

def get_featurizer_errors():
    """Return a dictionary of featurizer creation errors."""
    return _featurizer_errors.copy()