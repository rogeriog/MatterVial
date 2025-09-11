# __init__.py

# Import modules with error handling
import warnings

# Always try to import structure module (contains lazy loading)
try:
    from .structure import *
    structure_available = True
except ImportError as e:
    structure_available = False
    warnings.warn(f"Structure featurizers not available: {e}")

# Try to import composition module
try:
    from .composition import *
    composition_available = True
except ImportError as e:
    composition_available = False
    warnings.warn(f"Composition featurizers not available: {e}")

# Try to import sisso_featurization module
try:
    from .sisso_featurization import *
    sisso_available = True
except ImportError as e:
    sisso_available = False
    warnings.warn(f"SISSO featurizers not available: {e}")

# Try to import mlip_models module
try:
    from .mlip_models import *
    mlip_available = True
except ImportError as e:
    mlip_available = False
    warnings.warn(f"MLIP models not available: {e}")

# Build __all__ from available modules
__all__ = []
if structure_available:
    try:
        __all__ += structure.__all__
    except NameError:
        pass
if composition_available:
    try:
        __all__ += composition.__all__
    except NameError:
        pass
if sisso_available:
    try:
        __all__ += sisso_featurization.__all__
    except NameError:
        pass
if mlip_available:
    try:
        __all__ += mlip_models.__all__
    except NameError:
        pass

# Add utility functions
if structure_available:
    try:
        from .structure import get_available_featurizers, get_featurizer_errors
        __all__.extend(["get_available_featurizers", "get_featurizer_errors"])
    except ImportError:
        pass