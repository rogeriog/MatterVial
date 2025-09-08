from .roost_models import get_RoostFeatures

class RoostModelFeaturizer:
    """
    ROOST (Representation Learning from Stoichiometry) model featurizer.

    This featurizer uses pretrained ROOST models to extract features directly from
    material compositions without requiring structural information. ROOST models
    are particularly useful for composition-based property prediction tasks.

    The featurizer supports different pretrained ROOST models:
    - 'mpgap': Trained on Materials Project band gap dataset
    - 'oqmd_eform': Trained on OQMD formation energy dataset

    Attributes:
        model_type (str): Type of pretrained ROOST model to use.
        model_file (str): Path to custom model file (optional).
        embedding_filepath (str): Path to custom embedding file (optional).
        kwargs (dict): Additional keyword arguments.

    Examples:
        >>> # Using pretrained models
        >>> roost_gap = RoostModelFeaturizer(model_type='mpgap')
        >>> roost_eform = RoostModelFeaturizer(model_type='oqmd_eform')
        >>>
        >>> # Extract features from compositions
        >>> compositions = pd.Series(["Fe2O3", "Al2O3", "TiO2"])
        >>> features = roost_gap.get_features(compositions)
    """

    def __init__(self, model_type=None, model_file=None, embedding_filepath=None, **kwargs):
        """
        Initialize RoostModelFeaturizer.

        Args:
            model_type (str, optional): Type of pretrained ROOST model.
                Options: 'mpgap', 'oqmd_eform'. If None, must provide model_file.
            model_file (str, optional): Path to custom ROOST model file.
            embedding_filepath (str, optional): Path to custom embedding file.
            **kwargs: Additional keyword arguments passed to the ROOST model.

        Examples:
            >>> # Using pretrained model
            >>> featurizer = RoostModelFeaturizer(model_type='mpgap')

            >>> # Using custom model
            >>> featurizer = RoostModelFeaturizer(
            ...     model_file='/path/to/custom_model.pth',
            ...     embedding_filepath='/path/to/embeddings.json'
            ... )
        """
        self.model_type = model_type
        self.model_file = model_file
        self.embedding_filepath = embedding_filepath
        self.kwargs = kwargs

    def get_features(self, compositions):
        """
        Extract ROOST features from material compositions.

        Args:
            compositions (pd.Series or list): Material compositions as strings.
                Examples: ["Fe2O3", "Al2O3", "TiO2"] or pd.Series(["CaTiO3", "BaTiO3"]).

        Returns:
            pd.DataFrame: DataFrame containing ROOST features with shape
                (n_compositions, n_features). The number of features depends on
                the ROOST model architecture. Column names follow the pattern
                'ROOST_X' where X is the feature index.

        Raises:
            ImportError: If ROOST dependencies are not available.
            ValueError: If compositions are invalid or model_type is not recognized.
            FileNotFoundError: If custom model files are not found.

        Examples:
            >>> compositions = pd.Series(["Fe2O3", "Al2O3"])
            >>> roost_featurizer = RoostModelFeaturizer(model_type='mpgap')
            >>> features = roost_featurizer.get_features(compositions)
            >>> print(features.shape)
            (2, 128)  # 128 ROOST features for 2 compositions
        """
        return get_RoostFeatures(compositions, model_type=self.model_type,
                                 model_file=self.model_file,
                                 embedding_filepath=self.embedding_filepath,
                                 **self.kwargs)

roost_mpgap = RoostModelFeaturizer(model_type='mpgap')
roost_oqmd_eform = RoostModelFeaturizer(model_type='oqmd_eform')

__all__ = ("roost_mpgap", "roost_oqmd_eform")