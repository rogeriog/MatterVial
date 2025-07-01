from .roost_models import get_RoostFeatures

class RoostModelFeaturizer:
    """Featurizer for ROOST models."""

    def __init__(self, model_type=None, model_file=None, embedding_filepath=None, **kwargs):
        self.model_type = model_type
        self.model_file = model_file
        self.embedding_filepath = embedding_filepath
        self.kwargs = kwargs

    def get_features(self, compositions):
        return get_RoostFeatures(compositions, model_type=self.model_type,
                                 model_file=self.model_file,
                                 embedding_filepath=self.embedding_filepath,
                                 **self.kwargs)

roost_mpgap = RoostModelFeaturizer(model_type='mpgap')
roost_oqmd_eform = RoostModelFeaturizer(model_type='oqmd_eform')

__all__ = ("roost_mpgap", "roost_oqmd_eform")