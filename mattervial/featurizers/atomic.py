"""
Per-atom pre-readout embeddings from MatterVial GNN backends.

This module provides the AtomicFeaturizer class for extracting variable-length
per-atom node embeddings taken just before each model's pooling/readout stage.
Unlike the fixed-width featurizers, atomic representations have shape (n_atoms, hidden_dim)
where n_atoms varies per structure.

Supported backends:
- MEGNet (MVL × 6, l-MM, l-OFM, AdjacentMEGNet)
- coGN (KGCNN)
- ORB-v3 (PyTorch)
- ROOST (PyTorch, composition-based — broadcast to atomic sites)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Union, Optional
from pymatgen.core import Structure

__all__ = ['AtomicFeaturizer']


class AtomicFeaturizer:
    """Extract per-atom pre-pool embeddings from MatterVial GNN backends.

    Outputs are variable-shape per structure (n_atoms varies); kept
    separate from the fixed-width main featurizers.

    Attributes:
        models (list): List of model names to extract features from.
        verbose (bool): Whether to print progress messages.
    """

    KNOWN_MODELS = {
        # MEGNet MVL (one entry per pretrained MVL backbone)
        "MVL_Eform_MP_2019", "MVL_Efermi_MP_2019",
        "MVL_Bandgap_classifier_MP_2018", "MVL_Bandgap_MP_2018",
        "MVL_logK_MP_2019", "MVL_logG_MP_2019",
        # Custom MEGNets
        "l-MM_v1", "l-OFM_v1",
        "AdjacentMEGNet",
        # Other backbones
        "coGN",
        "ORB_v3",
        "ROOST_mpgap", "ROOST_oqmd_eform",
    }

    def __init__(
        self,
        models: Union[str, List[str]],
        *,
        adjacent_model_path: Optional[str] = None,
        cogn_weights_path: Optional[str] = None,
        roost_model_type: str = "mpgap",
        device: Optional[str] = None,
        verbose: bool = True,
        cache_file: Optional[str] = None,
        num_cores: Optional[int] = None,
        orb_handle_disorder: bool = False,
    ):
        if isinstance(models, str):
            models = [models]
        self.models = models
        self.verbose = verbose
        self.device = device
        self.adjacent_model_path = adjacent_model_path
        self.cogn_weights_path = cogn_weights_path
        self.roost_model_type = roost_model_type
        self.cache_file = cache_file
        self.num_cores = num_cores
        self.orb_handle_disorder = orb_handle_disorder

        invalid_models = set(models) - self.KNOWN_MODELS
        if invalid_models:
            raise ValueError(f"Invalid model names: {invalid_models}.")

        self._megnet_models_cache = {}
        self._orb_handler = None
        self._roost_handler = None
        
        # --- FIXED: Use a dictionary for hoisting to handle different architectures ---
        self._megnet_graph_hoist = {}

    def get_features(
        self, structures: Union[Structure, List[Structure], pd.Series]
        ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """
        Extract per-atom embeddings from structures.

        Args:
            structures: Single Structure, list of Structures, or pandas Series of Structures.

        Returns:
            If input is a single Structure: dict mapping model_name -> (n_atoms, hidden_dim) array.
            If input is multiple structures: list of dicts, one per structure, aligned to input order.

        Raises:
            ImportError: If required dependencies for a backend are not available.
            ValueError: If structures are invalid or empty.
        """
        # Normalize input
        single_structure = False
        if isinstance(structures, Structure):
            single_structure = True
            structures = [structures]
        elif isinstance(structures, pd.Series):
            structures = list(structures)
        elif not isinstance(structures, list):
            structures = list(structures)

        if not structures:
            raise ValueError("No structures provided.")

        if self.verbose:
            print(f"Extracting atomic features for {len(structures)} structure(s)...")

        n = len(structures)
        results = [{} for _ in range(n)]
        
        # Reset memory cache at the start of a new extraction batch
        self._precomputed_megnet_graphs = None
        self._megnet_graph_hoist = {} 

        for model_name in self.models:
            try:
                if model_name.startswith("MVL_"):
                    feats_list = self._extract_mvl_atomic_batch(structures, model_name)
                elif model_name in ("l-MM_v1", "l-OFM_v1"):
                    feats_list = self._extract_custom_megnet_atomic_batch(structures, model_name)
                elif model_name == "ORB_v3":
                    feats_list = self._extract_orb_atomic_batch(structures)
                elif model_name.startswith("ROOST_"):
                    feats_list = self._extract_roost_atomic_batch(structures, model_name)
                elif model_name == "AdjacentMEGNet":
                    # UPGRADED to batched method
                    feats_list = self._extract_adjacent_megnet_atomic_batch(structures)
                elif model_name == "coGN":
                    feats_list = [self._extract_cogn_atomic(s) for s in structures]
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                for i, feat in enumerate(feats_list):
                    results[i][model_name] = feat

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to extract {model_name}: {e}")
                import traceback; traceback.print_exc()
                for i in range(n):
                    results[i][model_name] = None

        if self.verbose:
            print("Feature extraction complete.")

        if single_structure:
            return results[0]
        return results

    def _extract_mvl_atomic(self, structure: Structure, model_name: str) -> np.ndarray:
        """Single-structure wrapper — delegates to the batched path."""
        return self._extract_mvl_atomic_batch([structure], model_name)[0]

    def _extract_mvl_atomic_batch(self, structures: List[Structure], model_name: str) -> List[np.ndarray]:
        try:
            from .megnet_models import predict_atomic_node_features, batch_convert_structures_to_graphs
            from megnet.utils.models import load_model as load_mvl_model
        except ImportError as e:
            raise ImportError(f"MEGNet dependencies required for MVL: {e}")

        mvl_model_map = {
            "MVL_Eform_MP_2019": "Eform_MP_2019",
            "MVL_Efermi_MP_2019": "Efermi_MP_2019",
            "MVL_Bandgap_classifier_MP_2018": "Bandgap_classifier_MP_2018",
            "MVL_Bandgap_MP_2018": "Bandgap_MP_2018",
            "MVL_logK_MP_2019": "logK_MP_2019",
            "MVL_logG_MP_2019": "logG_MP_2019",
        }
        mvl_id = mvl_model_map[model_name]
        if mvl_id not in self._megnet_models_cache:
            self._megnet_models_cache[mvl_id] = load_mvl_model(mvl_id)

        model = self._megnet_models_cache[mvl_id]
        
        # All MVL models share the "mvl" graph configuration
        if "mvl" not in self._megnet_graph_hoist:
            # For the disk cache, we use a suffix to prevent architecture collision
            mvl_cache = f"{self.cache_file}_mvl.pkl" if self.cache_file else None
            self._megnet_graph_hoist["mvl"] = batch_convert_structures_to_graphs(
                model, structures, verbose=self.verbose, cache_file=mvl_cache,
                num_cores=self.num_cores
            )

        return predict_atomic_node_features(
            model, structures, verbose=False, 
            precomputed_graphs=self._megnet_graph_hoist["mvl"]
        )

    def _extract_custom_megnet_atomic(
        self, structure: Structure, model_name: str
    ) -> np.ndarray:
        """Single-structure wrapper — delegates to the batched path."""
        return self._extract_custom_megnet_atomic_batch([structure], model_name)[0]

    def _extract_custom_megnet_atomic_batch(
        self, structures: List[Structure], model_name: str
    ) -> List[np.ndarray]:
        """Extract per-atom features from custom MEGNet model for a list of structures (batched)."""
        try:
            from .megnet_models import predict_atomic_node_features, load_model_scaler, batch_convert_structures_to_graphs
        except ImportError as e:
            raise ImportError(f"MEGNet dependencies required for {model_name}: {e}")

        model_type_map = {
            "l-MM_v1": "MatMinerEncoded_v1",
            "l-OFM_v1": "OFMEncoded_v1",
        }
        model_type = model_type_map[model_name]
        cache_key = f"custom_{model_type}"

        if cache_key not in self._megnet_models_cache:
            if self.verbose:
                print(f"    Loading custom MEGNet model: {model_name}")
            import os

            package_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if model_type == "MatMinerEncoded_v1":
                model_file = os.path.join(
                    package_base_dir, "featurizers", "custom_models", "MEGNetModel__MatMinerEncoded_v1.h5"
                )
                n_targets, neuron_layers = 758, (64, 128, 64)
            else:
                model_file = os.path.join(
                    package_base_dir, "featurizers", "custom_models", "MEGNetModel__OFMEncoded_v1.h5"
                )
                n_targets, neuron_layers = 188, (64, 128, 64)

            scaler_file = model_file.replace(".h5", "_scaler.pkl")
            model, _ = load_model_scaler(
                n_targets=n_targets,
                neuron_layers=neuron_layers,
                model_file=model_file,
                scaler_file=scaler_file,
            )
            self._megnet_models_cache[cache_key] = model

        model = self._megnet_models_cache[cache_key]
        
        # --- FIXED: Share the precomputed graphs between l-MM and l-OFM ---
        if "custom_latent" not in self._megnet_graph_hoist:
            custom_cache = f"{self.cache_file}_custom_latent.pkl" if self.cache_file else None
            self._megnet_graph_hoist["custom_latent"] = batch_convert_structures_to_graphs(
                model, structures, verbose=self.verbose, cache_file=custom_cache,
                num_cores=self.num_cores
            )

        return predict_atomic_node_features(
            model, structures, verbose=False, 
            precomputed_graphs=self._megnet_graph_hoist["custom_latent"],
            cache_file=self.cache_file
        )

    def _extract_adjacent_megnet_atomic_batch(self, structures: List[Structure]) -> List[np.ndarray]:
        """Extract per-atom features from AdjacentMEGNet model (batched)."""
        try:
            from .megnet_models import predict_atomic_node_features, load_model_scaler, batch_convert_structures_to_graphs
        except ImportError as e:
            raise ImportError(f"MEGNet dependencies required for AdjacentMEGNet: {e}")

        if not self.adjacent_model_path:
            raise ValueError(
                "adjacent_model_path must be provided for AdjacentMEGNet extraction."
            )

        import os

        model_file = os.path.join(
            self.adjacent_model_path, "MEGNetModel__adjacent.h5"
        )
        scaler_file = os.path.join(
            self.adjacent_model_path, "MEGNetModel__adjacent_scaler.pkl"
        )

        if not os.path.isfile(model_file):
            raise FileNotFoundError(
                f"AdjacentMEGNet model not found at {model_file}. "
                "Please train the model first."
            )

        # Load model (no caching for adjacent since path may vary)
        model, _ = load_model_scaler(
            model_file=model_file, scaler_file=scaler_file
        )

        # FIX 1 & 2: Memory Hoisting & Disk Caching
        if self._precomputed_megnet_graphs is None:
            self._precomputed_megnet_graphs = batch_convert_structures_to_graphs(
                model, structures, verbose=self.verbose, cache_file=self.cache_file,
                num_cores=self.num_cores
            )

        return predict_atomic_node_features(
            model, structures, verbose=False, 
            precomputed_graphs=self._precomputed_megnet_graphs,
            cache_file=self.cache_file
        )

    def _extract_cogn_atomic(self, structure: Structure) -> np.ndarray:
        """Extract per-atom features from coGN model (expanded from asymmetric unit)."""
        try:
            import tensorflow as tf
            from kgcnn.literature.coGN import make_model, model_default
            from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell
        except ImportError as e:
            raise ImportError(
                f"coGN dependencies required (kgcnn, tensorflow): {e}. "
                "Please activate the kgcnn environment."
            )

        if not self.cogn_weights_path:
            raise ValueError(
                "cogn_weights_path must be provided for coGN extraction."
            )

        import json
        import networkx as nx

        # Load model and weights
        if self.verbose:
            print(f"    Loading coGN model from {self.cogn_weights_path}")

        original_model = make_model(**model_default)
        original_model.load_weights(self.cogn_weights_path)

        # Build feature extractor sub-model
        try:
            final_node_embeddings_tensor = [
                layer.output[0]
                for layer in original_model.layers
                if layer.name.startswith("sequential_graph_network")
            ][0]
            readout_layer = [
                layer
                for layer in original_model.layers
                if layer.name.startswith("graph_network_multiplicity_readout")
            ][0]
        except IndexError:
            raise RuntimeError(
                "Could not find required layers in coGN model. "
                "Expected 'sequential_graph_network' and 'graph_network_multiplicity_readout'."
            )

        multiplicity_tensor = readout_layer.output[1]

        feature_extractor_model = tf.keras.Model(
            inputs=original_model.inputs,
            outputs=[final_node_embeddings_tensor, multiplicity_tensor],
            name="AtomicFeatureExtractor",
        )

        # Preprocess structure
        preprocessor = KNNAsymmetricUnitCell(24)
        nx_graph = preprocessor(structure)

        # Convert networkx graph to GraphList components
        node_indices = sorted(nx_graph.nodes())
        node_attributes = {
            "atomic_number": np.array(
                [nx_graph.nodes[i]["atomic_number"] for i in node_indices]
            ),
            "multiplicity": np.array(
                [nx_graph.nodes[i]["multiplicity"] for i in node_indices]
            ),
        }

        edge_indices = []
        edge_offsets = []
        for u, v, data in nx_graph.edges(data=True):
            edge_indices.append([u, v])
            edge_offsets.append(data["offset"])

        edge_attributes = {"offset": np.array(edge_offsets)} if edge_offsets else {}

        # Prepare input tensors
        try:
            from graphlist import GraphList
        except ImportError:
            raise ImportError(
                "graphlist module required for coGN. "
                "Please check your kgcnn environment setup."
            )

        graph_list = GraphList(
            num_nodes=[nx_graph.number_of_nodes()],
            num_edges=[nx_graph.number_of_edges()],
            edge_indices=[np.array(edge_indices)],
            node_attributes={k: [v] for k, v in node_attributes.items()},
            edge_attributes={k: [v] for k, v in edge_attributes.items()},
            graph_attributes={},
        )

        input_names = [inp.name for inp in feature_extractor_model.inputs]
        input_tensors = {}

        for input_name, attr_list in graph_list.node_attributes.items():
            if input_name in input_names:
                input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                    attr_list[0], [graph_list.num_nodes[0]]
                )

        for input_name, attr_list in graph_list.edge_attributes.items():
            if input_name in input_names:
                input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                    attr_list[0], [graph_list.num_edges[0]]
                )

        if "edge_indices" in input_names:
            reversed_indices = graph_list.edge_indices[0][:, [1, 0]]
            input_tensors["edge_indices"] = tf.RaggedTensor.from_row_lengths(
                reversed_indices, [graph_list.num_edges[0]]
            )

        # Predict
        outputs = feature_extractor_model.predict(input_tensors, verbose=0)
        node_embeddings = outputs[0].numpy()
        multiplicities = outputs[1].numpy().flatten()

        # Expand to all atoms in unit cell
        expanded_embeddings = np.repeat(node_embeddings, multiplicities.astype(int), axis=0)

        # Sanity check
        if len(expanded_embeddings) != len(structure):
            warnings.warn(
                f"coGN expansion mismatch: got {len(expanded_embeddings)} atoms, "
                f"expected {len(structure)}. Returning asymmetric-unit embeddings with warning."
            )
            return node_embeddings

        return expanded_embeddings

    def _extract_orb_atomic(self, structure: Structure) -> np.ndarray:
        """Single-structure wrapper — delegates to the batched path."""
        return self._extract_orb_atomic_batch([structure])[0]

    def _extract_orb_atomic_batch(
        self, structures: List[Structure], batch_size: int = 64
    ) -> List[np.ndarray]:
        """Extract per-atom features from ORB-v3 for a list of structures (batched)."""
        try:
            from .mlip_models import ORBModelHandler
        except ImportError as e:
            raise ImportError(
                f"ORB dependencies required: {e}. "
                "Please activate the orb_env environment."
            )

        if self._orb_handler is None:
            if self.verbose:
                print("    Initializing ORB-v3 model")
            self._orb_handler = ORBModelHandler(
                model_name="ORB_v3", device=self.device,
                handle_disorder=self.orb_handle_disorder,
            )

        return self._orb_handler.extract_atomic_features_batch(structures, batch_size=batch_size)

    def _extract_roost_atomic(
        self, structure: Structure, model_name: str
    ) -> np.ndarray:
        """Single-structure wrapper — delegates to the batched path."""
        return self._extract_roost_atomic_batch([structure], model_name)[0]

    def _extract_roost_atomic_batch(
        self, structures: List[Structure], model_name: str, batch_size: int = 64
    ) -> List[np.ndarray]:
        """Extract ROOST per-atom features for a list of structures (batched with composition dedup)."""
        model_type_map = {
            "ROOST_mpgap": "mpgap",
            "ROOST_oqmd_eform": "oqmd_eform",
        }
        model_type = model_type_map.get(model_name, self.roost_model_type)

        if self._roost_handler is None:
            if self.verbose:
                print(f"    Initializing ROOST model ({model_type})")
            self._roost_handler = _ROOSTHandler(model_type=model_type)

        return self._roost_handler.extract_atomic_features_batch(
            structures, batch_size=batch_size
        )


class _ROOSTHandler:
    """Helper class for ROOST feature extraction."""

    def __init__(self, model_type: str = "mpgap"):
        """Initialize ROOST model handler."""
        import os
        import torch
        from mattervial.packages.roost.roost.model import Roost
        from mattervial.packages.roost.roost.data import CompositionData

        self.model_type = model_type
        self.model = None
        self.elem_emb_len = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        if model_type == "mpgap":
            model_file = "mpgap_checkpoint-r0.pth.tar"
        elif model_type == "oqmd_eform":
            model_file = "oqmd_eform_checkpoint-r0.pth.tar"
        else:
            raise ValueError(f"Unknown ROOST model type: {model_type}")

        parent_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        # Try multiple possible locations for the model file
        possible_paths = [
            os.path.join(parent_dir, "packages", "roost", "custom_models", model_file),
            os.path.join(parent_dir, "packages", "roost", "roost", "custom_models", model_file),
            os.path.join(parent_dir, "featurizers", "custom_models", model_file),
        ]
        
        model_file = None
        for path in possible_paths:
            if os.path.isfile(path):
                model_file = path
                break
        
        if model_file is None:
            raise FileNotFoundError(
                f"ROOST model '{model_type}' not found. "
                f"Searched in: {possible_paths}. "
                f"ROOST model files are not included in the public repository. "
                f"Please download them from the ROOST repository."
            )

        # Create temporary CompositionData to get elem_emb_len
        import tempfile
        import pandas as pd

        # Try to find matscholar embedding file
        possible_fea_paths = [
            os.path.join(parent_dir, "packages", "roost", "custom_models", "matscholar-embedding.json"),
            os.path.join(parent_dir, "packages", "roost", "roost", "custom_models", "matscholar-embedding.json"),
            os.path.join(parent_dir, "featurizers", "custom_models", "matscholar-embedding.json"),
        ]
        fea_path = None
        for path in possible_fea_paths:
            if os.path.isfile(path):
                fea_path = path
                break
        if fea_path is None:
            raise FileNotFoundError(
                f"matscholar-embedding.json not found. Searched in: {possible_fea_paths}"
            )
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        temp_df = pd.DataFrame({"material_id": ["temp"], "composition": ["NaCl"]})
        temp_df.to_csv(temp_csv, index=False)

        tmp_dataset = CompositionData(data_path=temp_csv, fea_path=fea_path, task_dict={})
        self.elem_emb_len = tmp_dataset.elem_emb_len

        os.remove(temp_csv)

        # Build model
        self.model = Roost(
            task_dict={},
            robust=False,
            n_targets=[2],
            elem_emb_len=self.elem_emb_len,
            elem_fea_len=64,
            n_graph=3,
            device="gpu" if torch.cuda.is_available() else "cpu",
        )

        # Load weights
        checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Register hook to capture pre-pool element features
        self.elem_fea_hook = None
        self.model.material_nn.cry_pool[0].register_forward_pre_hook(
            self._get_elem_fea_hook()
        )

    def _get_elem_fea_hook(self):
        """Create a hook to capture element features before pooling."""
        def hook(module, input):
            self.elem_fea_hook = input[0].detach().cpu().numpy()

        return hook

    def extract_atomic_features(self, structure) -> np.ndarray:
        """Single-structure wrapper — delegates to the batched path."""
        return self.extract_atomic_features_batch([structure])[0]

    def extract_atomic_features_batch(self, structures, batch_size=64) -> List[np.ndarray]:
        """Extract element embeddings for a list of structures and broadcast to atomic sites.

        Uses composition deduplication: each unique composition is processed once and the
        resulting per-element embeddings are cached and reused for all structures sharing
        that composition. DataLoader batching is used for the actual forward passes.

        Returns a list of np.ndarray of shape (n_atoms, 64), one per structure.
        """
        import torch
        import pandas as pd
        import tempfile
        import os
        from pymatgen.core.composition import Composition
        from mattervial.packages.roost.roost.data import CompositionData, collate_batch
        from torch.utils.data import DataLoader

        # Try to find matscholar embedding file (resolve once)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_fea_paths = [
            os.path.join(parent_dir, "packages", "roost", "custom_models", "matscholar-embedding.json"),
            os.path.join(parent_dir, "packages", "roost", "roost", "custom_models", "matscholar-embedding.json"),
            os.path.join(parent_dir, "featurizers", "custom_models", "matscholar-embedding.json"),
        ]
        fea_path = None
        for path in possible_fea_paths:
            if os.path.isfile(path):
                fea_path = path
                break
        if fea_path is None:
            raise FileNotFoundError(
                f"matscholar-embedding.json not found. Searched in: {possible_fea_paths}"
            )

        # Build unique composition list (deduplication)
        comp_keys = [str(Composition(s.composition).reduced_formula) for s in structures]
        unique_comps = list(dict.fromkeys(comp_keys))  # preserves order, no duplicates

        # Build one CSV with all unique compositions
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        temp_df = pd.DataFrame({
            "material_id": [f"comp_{i}" for i in range(len(unique_comps))],
            "composition": unique_comps,
        })
        temp_df.to_csv(temp_csv, index=False)

        # Map from composition key -> (elem_symbols list, elem_embeddings array)
        comp_embedding_cache: dict = {}

        try:
            dataset = CompositionData(data_path=temp_csv, fea_path=fea_path, task_dict={})
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
            )

            dataset_index = 0
            with torch.no_grad():
                for batch in loader:
                    inputs, _targets, _comp_ids, _comp_formulas = batch
                    # collate_batch order: (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, crystal_atom_idx)
                    elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, crystal_atom_idx = inputs

                    self.elem_fea_hook = None
                    _ = self.model(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, crystal_atom_idx)

                    if self.elem_fea_hook is None:
                        raise RuntimeError("Failed to capture element features via hook")

                    all_elem_embs = self.elem_fea_hook  # (total_elements_in_batch, 64)

                    # crystal_atom_idx maps each element row to its crystal index (0..n_crystals-1)
                    crystal_idx_arr = crystal_atom_idx.cpu().numpy()

                    # elem_fea rows are ordered by crystal, count per crystal
                    n_crystals_in_batch = int(crystal_idx_arr.max()) + 1
                    for local_c in range(n_crystals_in_batch):
                        mask = crystal_idx_arr == local_c  # shape (N,) — same as all_elem_embs dim 0
                        crystal_elem_embs = all_elem_embs[mask]  # (n_elem_this_crystal, 64)
                        # The elements appear in the same order as in the composition
                        comp_key = unique_comps[dataset_index]
                        comp_obj = Composition(comp_key)
                        elem_symbols = [str(e) for e in comp_obj.elements]
                        comp_embedding_cache[comp_key] = (elem_symbols, crystal_elem_embs)
                        dataset_index += 1

        finally:
            os.remove(temp_csv)

        # Now broadcast cached embeddings to each structure's atomic sites
        results: List[np.ndarray] = []
        for struct, comp_key in zip(structures, comp_keys):
            if comp_key not in comp_embedding_cache:
                results.append(np.zeros((len(struct), 64), dtype=np.float32))
                continue

            elem_symbols, crystal_elem_embs = comp_embedding_cache[comp_key]
            atomic_embeddings = []
            
            for site in struct:
                # Initialize an empty vector for the site
                site_embedding = np.zeros(64, dtype=np.float32)
                valid_site = True
                
                # Iterate through all elements occupying this specific site
                # site.species.items() yields (Element/Specie, fractional_occupancy)
                for element, occupancy in site.species.items():
                    # Safely extract just the letters (e.g., 'Fe2+' -> 'Fe')
                    elem_symbol = "".join([c for c in str(element) if c.isalpha()])
                    
                    try:
                        elem_idx = elem_symbols.index(elem_symbol)
                        # Weight the element's embedding by its occupancy fraction and add to the site
                        site_embedding += crystal_elem_embs[elem_idx] * float(occupancy)
                    except ValueError:
                        # If an element is completely missing from the ROOST cache, invalidate the site
                        valid_site = False
                        break
                
                if valid_site:
                    atomic_embeddings.append(site_embedding)
                else:
                    # Inject NaNs to flag the error rather than silently assigning corrupted data
                    atomic_embeddings.append(np.full(64, np.nan, dtype=np.float32))
                    
            results.append(np.array(atomic_embeddings, dtype=np.float32))

        return results
