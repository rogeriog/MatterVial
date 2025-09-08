import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import torch
import os
# ORB model imports
try:
    import ase
    from orb_models.forcefield import atomic_system, pretrained
    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False
    print("Warning: ORB models not available. Please install orb-models to use ORBFeaturizer.")


class ORBModelHandler:
    """Handler for ORB model operations including model loading and feature extraction."""
    
    def __init__(self, model_name="ORB_v3", device=None, precision="float32-high"):
        if not ORB_AVAILABLE:
            raise ImportError("ORB models not available. Please install orb-models package.")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.orbff = None
        self.pooled_embeddings = {}
        self._hooks_registered = False
        
        self._load_model()
        self._register_hooks()
    
    def _load_model(self):
        """Load the ORB model."""
        if self.model_name == "ORB_v3":
            self.orbff = pretrained.orb_v3_conservative_inf_omat(
                device=self.device,
                precision=self.precision,
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _get_node_mlp_output_hook(self, layer_name):
        """Create a hook function for extracting embeddings from a specific layer."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.numel() > 0:
                atom_embeddings = output.detach()
                mean_pooled_embedding = torch.mean(atom_embeddings, dim=0)
                self.pooled_embeddings[layer_name] = mean_pooled_embedding
            else:
                self.pooled_embeddings[layer_name] = None
        return hook
    
    def _register_hooks(self):
        """Register forward hooks to extract embeddings from different layers."""
        if self._hooks_registered:
            return
        
        # Register hooks for different layers
        self.orbff.model.atom_emb.register_forward_hook(
            self._get_node_mlp_output_hook("initial_atom_embedding")
        )
        
        encoder_node_mlp = self.orbff.model._encoder._node_fn.mlp
        encoder_node_mlp.register_forward_hook(
            self._get_node_mlp_output_hook("encoder_node_features")
        )
        
        for i, gnn_layer in enumerate(self.orbff.model.gnn_stacks):
            node_mlp = gnn_layer._node_mlp.mlp
            node_mlp.register_forward_hook(
                self._get_node_mlp_output_hook(f"layer_{i}")
            )
        
        self._hooks_registered = True
    
    def extract_features_single(self, structure):
        """Extract ORB features from a single structure."""
        try:
            # Convert pymatgen Structure to ASE Atoms
            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)
            
            # Clear previous embeddings
            self.pooled_embeddings.clear()
            
            # Convert to ORB graph and run prediction
            graph = atomic_system.ase_atoms_to_atom_graphs(
                atoms, self.orbff.system_config, device=self.device
            )
            _ = self.orbff.predict(graph, split=False)
            
            # Extract features from embeddings
            features = {}
            for layer_name, embedding in self.pooled_embeddings.items():
                if embedding is not None:
                    embedding_np = embedding.cpu().numpy()
                    for neuron_idx, val in enumerate(embedding_np):
                        if layer_name.startswith("layer_"):
                            prefix, num = layer_name.split("_")
                            new_layer_name = f"{prefix}_{int(num) + 1}"
                            feature_name = f"{self.model_name}_{new_layer_name}_{neuron_idx + 1}"
                        else:
                            feature_name = f"{self.model_name}_{layer_name}_{neuron_idx + 1}"
                        features[feature_name] = val
                else:
                    # If embedding is None, fill with NaN for all neurons
                    # Assuming 256 neurons based on typical ORB output
                    for neuron_idx in range(256):
                        if layer_name.startswith("layer_"):
                            prefix, num = layer_name.split("_")
                            new_layer_name = f"{prefix}_{int(num) + 1}"
                            feature_name = f"{self.model_name}_{new_layer_name}_{neuron_idx + 1}"
                        else:
                            feature_name = f"{self.model_name}_{layer_name}_{neuron_idx + 1}"
                        features[feature_name] = float('nan')
            
            return features
            
        except Exception as e:
            print(f"Error processing structure: {e}")
            # Return NaN features for failed structures
            features = {}
            layer_names = ["initial_atom_embedding", "encoder_node_features"] + [f"layer_{i}" for i in range(10)]
            for layer_name in layer_names:
                for neuron_idx in range(256):
                    if layer_name.startswith("layer_"):
                        prefix, num = layer_name.split("_")
                        new_layer_name = f"{prefix}_{int(num) + 1}"
                        feature_name = f"{self.model_name}_{new_layer_name}_{neuron_idx + 1}"
                    else:
                        feature_name = f"{self.model_name}_{layer_name}_{neuron_idx + 1}"
                    features[feature_name] = float('nan')
            return features


def get_ORB_features(structures, model_name="ORB_v3", device=None, precision="float32-high"):
    """
    Extract ORB features from a list of structures.
    
    Parameters:
    - structures: List of pymatgen Structure objects or pandas Series/DataFrame with 'structure' column
    - model_name: Name of the ORB model to use (default: "ORB_v3")
    - device: Device to run the model on (default: auto-detect)
    - precision: Model precision (default: "float32-high")
    
    Returns:
    - DataFrame containing the extracted ORB features for each structure
    """
    if not ORB_AVAILABLE:
        raise ImportError("ORB models not available. Please install orb-models package.")
    
    # Handle different input formats
    if isinstance(structures, pd.DataFrame):
        if 'structure' in structures.columns:
            structure_list = [Structure.from_str(s, fmt='json') for s in structures['structure']]
        else:
            raise ValueError("DataFrame must contain a 'structure' column")
    elif isinstance(structures, pd.Series):
        structure_list = []
        for s in structures:
            if isinstance(s, str):
                # Handle JSON string
                try:
                    structure_list.append(Structure.from_str(s, fmt='json'))
                except:
                    raise ValueError(f"Invalid JSON structure string: {s}")
            elif isinstance(s, dict):
                # Handle dictionary format
                try:
                    structure_list.append(Structure.from_dict(s))
                except:
                    raise ValueError(f"Invalid structure dictionary: {s}")
            elif isinstance(s, Structure):
                # Handle pymatgen Structure directly
                structure_list.append(s)
            elif isinstance(s, ase.atoms.Atoms):
                # Handle ASE Atoms object
                try:
                    adaptor = AseAtomsAdaptor()
                    structure_list.append(adaptor.get_structure(s))
                except:
                    raise ValueError(f"Could not convert ASE Atoms to Structure: {s}")
            else:
                raise ValueError(f"Unsupported structure format: {type(s)}")
    elif isinstance(structures, list):
        structure_list = structures
    else:
        raise ValueError("Structures must be a list, pandas Series, or DataFrame with 'structure' column")
    
    # Initialize ORB model handler
    orb_handler = ORBModelHandler(model_name=model_name, device=device, precision=precision)
    
    # Extract features for all structures
    features_list = []
    for i, structure in enumerate(structure_list):
        if (i + 1) % 10 == 0 or i == len(structure_list) - 1:
            print(f"Processing structure {i + 1}/{len(structure_list)}")
        
        features = orb_handler.extract_features_single(structure)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    print(f"ORB features extracted: {features_df.shape[1]} features for {features_df.shape[0]} structures")
    
    return features_df


__all__ = ('ORBModelHandler', 'get_ORB_features')
