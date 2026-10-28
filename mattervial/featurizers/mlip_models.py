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


N_ELEMENTS = 118  # ORB one-hot dim (index = atomic number)


def _soft_onehot(structure):
    """(n_sites, 118) occupancy-weighted soft one-hot: S[i, Z] = occ_Z(site i)."""
    S = np.zeros((len(structure), N_ELEMENTS), dtype=np.float32)
    for i, site in enumerate(structure):
        for el, occ in site.species.items():
            S[i, el.Z] += float(occ)
    return S


class ORBModelHandler:
    """Handler for ORB model operations including model loading and feature extraction.

    Args:
        handle_disorder: If True, disordered sites (partial occupations) are
            featurized with occupancy-weighted soft one-hot atom embeddings
            (h_i = sum_Z occ_Z * W[Z]). For fully occupied sites the soft
            one-hot is an exact one-hot row, so ordered structures reduce to
            the stock embedding-table lookup bit-for-bit. If False, disordered
            structures are featurized with the majority species only, matching
            stock ORB behaviour.
    """

    def __init__(self, model_name="ORB_v3", device=None, precision="float32-high",
                 handle_disorder=False):
        if not ORB_AVAILABLE:
            raise ImportError("ORB models not available. Please install orb-models package.")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.handle_disorder = handle_disorder
        self.orbff = None
        self.pooled_embeddings = {}
        self._hooks_registered = False
        self._adaptor = AseAtomsAdaptor()

        self._load_model()
        if self.handle_disorder:
            self._patch_featurize_nodes()
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

    def _patch_featurize_nodes(self):
        """Patch featurize_nodes to feed the occupancy-weighted soft one-hot
        through the atom-embedding matrix instead of doing an integer lookup.

        Also fires any hooks registered on ``atom_emb`` manually so
        feature-extraction hooks behave identically to the stock path.
        """
        gns = self.orbff.model
        W = gns.atom_emb.embeddings.weight  # (num_elements+1, emb_size)

        def featurize_nodes_soft(batch):
            one_hot = batch.node_features["atomic_numbers_embedding"]
            atomic_embedding = one_hot.to(W.dtype) @ W[:N_ELEMENTS]
            for h in gns.atom_emb._forward_hooks.values():
                h(gns.atom_emb, (one_hot,), atomic_embedding)
            feature_names = [k for k in gns.node_feature_names if k != "feat"]
            return torch.cat(
                [atomic_embedding,
                 *[batch.node_features[k] for k in feature_names]], dim=-1)

        gns.featurize_nodes = featurize_nodes_soft

    def _structure_to_graph(self, structure):
        """pymatgen Structure -> ORB AtomGraphs.

        Under handle_disorder, disordered structures carry the soft one-hot
        occupancy vector via the ``atomic_numbers_embedding`` node-feature
        override (honoured natively by ase_atoms_to_atom_graphs). Ordered
        structures use the stock conversion.
        """
        if self.handle_disorder and not structure.is_ordered:
            numbers = np.array([
                max(site.species.items(), key=lambda kv: kv[1])[0].Z
                for site in structure], dtype=int)
            atoms = ase.Atoms(
                numbers=numbers,
                positions=np.asarray(structure.cart_coords),
                cell=np.asarray(structure.lattice.matrix),
                pbc=True,
            )
            atoms.info["node_features"] = {
                "atomic_numbers_embedding": torch.from_numpy(
                    _soft_onehot(structure))
            }
        else:
            atoms = self._adaptor.get_atoms(structure)
        return atomic_system.ase_atoms_to_atom_graphs(
            atoms, self.orbff.system_config, device=self.device)
    
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
        
        # Register hook for full atomic features (no pooling) from last GNN layer
        self.atomic_features_hook = None
        self.orbff.model.gnn_stacks[-1].register_forward_hook(
            self._get_full_atomic_features_hook()
        )
        
        self._hooks_registered = True
    
    def _get_full_atomic_features_hook(self):
        """Create a hook to capture full (non-pooled) atomic features from the last GNN layer."""
        def hook(module, input, output):
            # output is a tuple (nodes, edges)
            if isinstance(output, tuple) and len(output) >= 1:
                nodes = output[0]
                if isinstance(nodes, torch.Tensor):
                    self.atomic_features_hook = nodes.detach()
        return hook

    def extract_atomic_features_batch(self, structures, batch_size=256):
        """Extract per-atom features for a list of structures using batched GPU inference.

        Converts structures to ORB graphs sequentially within each sub-batch to avoid
        CUDA thread contention, then runs one forward pass per sub-batch.

        Returns list of np.ndarray of shape (n_atoms, 256), one per structure.
        In case of failure for an individual structure, the entry is a (0,256) array.
        """
        from orb_models.forcefield.base import batch_graphs as orb_batch_graphs

        n = len(structures)
        results = [np.zeros((0, 256), dtype=np.float32)] * n

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            # Convert sub-batch sequentially to avoid CUDA thread contention
            valid = []
            for i in range(start, end):
                try:
                    graph = self._structure_to_graph(structures[i])
                    valid.append((i, graph))
                except Exception:
                    pass

            if not valid:
                continue

            global_indices, valid_graphs = zip(*valid)

            self.atomic_features_hook = None
            batched = orb_batch_graphs(list(valid_graphs))

            with torch.no_grad():
                _ = self.orbff.model(batched)

            if self.atomic_features_hook is None:
                continue

            all_nodes = self.atomic_features_hook.cpu().numpy().astype(np.float32)
            n_atoms_per = batched.n_node.tolist()

            offset = 0
            for global_i, n_at in zip(global_indices, n_atoms_per):
                results[global_i] = all_nodes[offset: offset + n_at]
                offset += n_at

        return results

    def extract_features_single(self, structure):
        """Extract ORB features from a single structure."""
        try:
            # Clear previous embeddings
            self.pooled_embeddings.clear()

            # Convert to ORB graph and run prediction
            graph = self._structure_to_graph(structure)
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
    
    def extract_atomic_features_single(self, structure):
        """Extract full per-atom features from a single structure (no pooling).
        
        Returns numpy array of shape (n_atoms, 256).
        """
        try:
            # Clear previous embeddings
            self.pooled_embeddings.clear()
            self.atomic_features_hook = None

            # Convert to ORB graph and run forward through GNN only (no force/stress)
            graph = self._structure_to_graph(structure)
            with torch.no_grad():
                _ = self.orbff.model(graph)

            # Extract the full atomic features from the hook
            if self.atomic_features_hook is None:
                raise RuntimeError("Failed to capture atomic features via hook")
            
            atomic_features = self.atomic_features_hook.cpu().numpy()
            return atomic_features.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting atomic features from structure: {e}")
            # Return empty array on failure
            return np.array([], dtype=np.float32).reshape(0, 256)


def get_ORB_features(structures, model_name="ORB_v3", device=None,
                     precision="float32-high", handle_disorder=False):
    """
    Extract ORB features from a list of structures.
    
    Parameters:
    - structures: List of pymatgen Structure objects or pandas Series/DataFrame with 'structure' column
    - model_name: Name of the ORB model to use (default: "ORB_v3")
    - device: Device to run the model on (default: auto-detect)
    - precision: Model precision (default: "float32-high")
    - handle_disorder: If True, featurize disordered sites with
      occupancy-weighted soft one-hot embeddings (default: False)
    
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
    orb_handler = ORBModelHandler(model_name=model_name, device=device,
                                  precision=precision,
                                  handle_disorder=handle_disorder)
    
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
