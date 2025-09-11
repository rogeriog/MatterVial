import numpy as np
from keras.models import Model
import warnings
from pickle import load, dump
import tensorflow as tf
import pandas as pd
import os
from typing import Tuple, Any, List

warnings.filterwarnings("ignore")

from megnet.utils.models import load_model
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping

### UNIFIED PROCESSING UTILITIES

def batch_convert_structures_to_graphs(model: MEGNetModel, structures: List,
                                       verbose: bool = True) -> Tuple[List, List, List]:
    """
    Convert multiple structures to graph inputs with error handling.

    Args:
        model: MEGNet model with a graph_converter.
        structures: List of pymatgen Structure objects.
        verbose: Whether to print processing information.

    Returns:
        Tuple of (valid_structures, graph_inputs, original_indices).
    """
    valid_structures = []
    graph_inputs = []
    original_indices = []

    if verbose:
        print(f"Converting {len(structures)} structures to graphs...")

    # Process structures individually to handle potential conversion errors
    for idx, structure in enumerate(structures):
        try:
            graph = model.graph_converter.convert(structure)
            graph_input = model.graph_converter.graph_to_input(graph)

            valid_structures.append(structure)
            graph_inputs.append(graph_input)
            original_indices.append(idx)
        except Exception as e:
            if verbose:
                print(f"Warning: Structure at index {idx} failed conversion: {e}")

    if verbose:
        successful_count = len(valid_structures)
        failed_count = len(structures) - successful_count
        print(f"Successfully converted {successful_count}/{len(structures)} structures.")
        if failed_count > 0:
            print(f"Failed to convert {failed_count} structures.")

    return valid_structures, graph_inputs, original_indices

def predict_structures_with_model(model: MEGNetModel, structures: List, verbose: bool = True) -> np.ndarray:
    """
    Performs prediction on Pymatgen Structure objects using the high-level
    model.predict_structures method, which handles graph conversion internally.
    This is used for final model predictions.
    """
    if not structures:
        return np.array([])
    if verbose:
        print(f"Predicting on {len(structures)} structures using model.predict_structures...")
    return model.predict_structures(structures)

def predict_graphs_with_model(model: Model, graph_inputs: List, verbose: bool = True) -> List:
    """
    Performs prediction on a list of pre-converted graph inputs.
    Returns a list of prediction results; for multi-output models, this will be a list of lists.
    """
    if not graph_inputs:
        return []
    if verbose:
        print(f"Predicting on {len(graph_inputs)} graphs sequentially...")
    
    # This loop now returns a list where each element can be a list of arrays (for multi-output)
    predictions = [model.predict(inp, verbose=0) for inp in graph_inputs]
    return predictions

def predict_intermediate_layers(model: MEGNetModel, graph_inputs: List, layer_indices: List[int],
                                verbose: bool = True) -> List[np.ndarray]:
    """
    Extracts features from multiple intermediate layers in a single model pass.
    This version correctly handles the list-of-lists output from a multi-output model.
    """
    outputs = [model.layers[i].output for i in layer_indices]
    multi_output_model = Model(inputs=model.input, outputs=outputs)
    
    # raw_predictions is now a list of lists, e.g., [[L1_g1, L2_g1], [L1_g2, L2_g2], ...]
    raw_predictions = predict_graphs_with_model(multi_output_model, graph_inputs, verbose=verbose)

    if not raw_predictions:
        return []

    # --- NEW LOGIC: Unpack and Concatenate ---
    # Transpose the list of lists to group predictions by layer.
    # e.g., from [[L1_g1, L2_g1], [L1_g2, L2_g2]] -> [(L1_g1, L1_g2), (L2_g1, L2_g2)]
    predictions_by_layer = zip(*raw_predictions)

    # Now, concatenate the predictions for each layer into a single numpy array.
    # Result: [np.array([L1_g1, L1_g2, ...]), np.array([L2_g1, L2_g2, ...])]
    concatenated_predictions = [np.concatenate(layer_preds, axis=0) for layer_preds in predictions_by_layer]
    
    return concatenated_predictions

### FUNCTIONS TO SETUP, EVALUATE AND TRAIN MEGNET MODELS

def model_setup(ntarget: int = None, **kwargs) -> MEGNetModel:
    """
    Sets up a MEGNetModel with a default architecture.
    """
    n1 = kwargs.get('n1', 64) 
    n2 = kwargs.get('n2', 32) 
    n3 = kwargs.get('n3', 16)
    nfeat_bond = kwargs.get('nfeat_bond', 100)
    r_cutoff = kwargs.get('r_cutoff', 5)
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = kwargs.get('gaussian_width', 0.5)
    graph_converter = CrystalGraph(cutoff=r_cutoff)

    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width,
                        ntarget=ntarget, **kwargs)
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary.splitlines()[-4])
    return model

def load_model_scaler(id: str = '', n_targets: int = 1, neuron_layers: Tuple[int] = (64, 32, 16),
                      **kwargs) -> Tuple[MEGNetModel, Any]:
    """
    Loads a pre-trained MEGNet model weights and a corresponding scaler.
    """
    n1, n2, n3 = neuron_layers
    model = model_setup(ntarget=n_targets, n1=n1, n2=n2, n3=n3, **kwargs)
    
    modelpath_id = kwargs.get("modeldir", "./") + id
    model_file = kwargs.get('model_file', f"{modelpath_id}_weights.h5")
    scaler_file = kwargs.get('scaler_file', f'{modelpath_id}_scaler.pkl')
    
    model.load_weights(model_file)
    try:
        scaler = load(open(scaler_file, 'rb'))
    except FileNotFoundError:
        scaler = None
    return (model, scaler)

def megnet_evaluate_structures(model: MEGNetModel, structures: List, targets=None, scaler=None, **kwargs):
    """
    Evaluate structures using a MEGNet model.
    """
    labels = kwargs.get('labels', [''] * len(structures))
    verbose = kwargs.get('verbose', True)

    if targets is None:
        noTargets = True
        target_values = np.ones(len(structures))
    else:
        noTargets = False
        target_values = targets.values if isinstance(targets, pd.DataFrame) else targets

    valid_structures, graph_inputs, valid_indices = batch_convert_structures_to_graphs(
        model, structures, verbose=verbose
    )

    if not valid_structures:
        return ([], np.array([])) if noTargets else ([], np.array([]), np.array([]), np.array([]))

    targets_valid = []
    labels_valid = []
    for idx in valid_indices:
        target_val = target_values[idx]
        if scaler is not None:
            targets_valid.append(np.nan_to_num(scaler.transform(target_val.reshape(1, -1))))
        else:
            targets_valid.append(target_val)
        labels_valid.append(labels[idx])

    # Use the correct prediction function for final outputs on Pymatgen structures
    ypred = predict_structures_with_model(model, valid_structures, verbose=verbose)

    y = np.array(targets_valid).squeeze()
    labels_out = np.array(labels_valid)

    if noTargets:
        return (valid_structures, ypred)
    else:
        return (valid_structures, ypred, y, labels_out)

def train_MEGNet_on_the_fly(structures, targets, **kwargs):
    """
    Trains a new MEGNet model on the provided structures and targets.

    This function properly configures the MEGNet model with Gaussian basis expansion
    parameters required for bond feature processing.

    Args:
        structures: Training structures (pandas Series/DataFrame or list)
        targets: Target values corresponding to the structures
        **kwargs: Additional parameters including:
            - adjacent_model_path (str): Path to save the trained model (default: '.')
            - max_epochs (int): Maximum training epochs (default: 100)
            - patience (int): Early stopping patience (default: 10)
            - r_cutoff (float): Cutoff radius for graph construction (default: 5.0)
            - nfeat_bond (int): Number of bond features for Gaussian expansion (default: 100)
            - gaussian_width (float): Width of Gaussian basis functions (default: 0.5)
            - n1, n2, n3 (int): Hidden layer sizes (defaults: 64, 32, 16)
    """
    from sklearn.preprocessing import MinMaxScaler
    targets = np.array(targets).reshape(-1, 1)
    scaler = MinMaxScaler()
    targets = scaler.fit_transform(targets)

    adjacent_model_path = kwargs.get('adjacent_model_path', '.')
    os.makedirs(adjacent_model_path, exist_ok=True)

    dump(scaler, open(os.path.join(adjacent_model_path, 'MEGNetModel__adjacent_scaler.pkl'), 'wb'))
    print('Scaler for the adjacent model saved to MEGNetModel__adjacent_scaler.pkl')

    # Training parameters
    max_epochs = kwargs.get('max_epochs', 100)
    patience = kwargs.get('patience', 10)
    early_stopping = EarlyStopping(monitor='val_mae', patience=patience, restore_best_weights=True)

    # Model architecture parameters (same as model_setup function)
    r_cutoff = kwargs.get('r_cutoff', 5.0)
    nfeat_bond = kwargs.get('nfeat_bond', 100)
    gaussian_width = kwargs.get('gaussian_width', 0.5)

    # Generate Gaussian centers for bond feature expansion
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)

    # Create graph converter
    graph_converter = CrystalGraph(cutoff=r_cutoff)

    # Split data for training and validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_index, val_index = next(kf.split(structures))

    train_structures, val_structures = structures.iloc[train_index], structures.iloc[val_index]
    train_targets, val_targets = targets[train_index], targets[val_index]

    # Create MEGNet model with proper Gaussian basis expansion parameters
    model = MEGNetModel(
        metrics=['mae'],
        graph_converter=graph_converter,
        centers=gaussian_centers,  # Required for Gaussian basis expansion
        width=gaussian_width,      # Required for Gaussian basis expansion
        ntarget=1,                 # Single target regression
        **{k: v for k, v in kwargs.items() if k in ['n1', 'n2', 'n3']}  # Pass architecture params
    )

    model.train(train_structures, train_targets, validation_structures=val_structures,
                validation_targets=val_targets, epochs=max_epochs, save_checkpoint=False, callbacks=[early_stopping])

    model_save_path = os.path.join(adjacent_model_path, 'MEGNetModel__adjacent.h5')
    model.save(model_save_path)
    print(f'On-the-fly MEGNet model saved to {model_save_path}')

def get_MVL_MEGNetFeatures(structures, **kwargs) -> pd.DataFrame:
    """
    Extracts features from pre-trained MEGNet models.

    This function can extract features from both layer 32 and layer 16 in a single pass
    for maximum efficiency, or extract features from a specific layer when requested.

    Args:
        structures: Input structures (pandas Series/DataFrame or list)
        layer_name (str, optional): Specific layer to extract ('layer32' or 'layer16').
                                   If not specified, extracts from both layers.
        verbose (bool): Whether to print processing information.

    Returns:
        pd.DataFrame: Features from the requested layer(s).
    """
    verbose = kwargs.get('verbose', True)
    layer_name = kwargs.get('layer_name', None)
    indexes = structures.index.to_list() if hasattr(structures, 'index') else list(range(len(structures)))
    structure_list = list(structures)

    # Define the layers we want to extract features from
    layer_config = {'MVL32': -3, 'MVL16': -2}

    # Determine which layers to extract based on layer_name parameter
    if layer_name is not None:
        if layer_name == 'layer32':
            requested_layers = {'MVL32': -3}
        elif layer_name == 'layer16':
            requested_layers = {'MVL16': -2}
        else:
            raise ValueError(f"Invalid layer_name '{layer_name}'. Valid options: 'layer32', 'layer16'")
    else:
        # Extract from both layers for backward compatibility
        requested_layers = layer_config

    layer_indices = list(requested_layers.values())

    model_names = ["Eform_MP_2019", 'Efermi_MP_2019', "Bandgap_classifier_MP_2018",
                   'Bandgap_MP_2018', 'logK_MP_2019', 'logG_MP_2019']

    # --- SINGLE GRAPH CONVERSION (from previous optimization) ---
    if verbose:
        print(f"Preparing graphs for all {len(model_names)} MVL models...")
    model_for_conversion = load_model(model_names[0])
    valid_structures, graph_inputs, valid_indices = batch_convert_structures_to_graphs(
        model_for_conversion, structure_list, verbose=verbose
    )

    if not graph_inputs:
        print("Warning: No structures could be converted. Returning empty DataFrame.")
        return pd.DataFrame(index=indexes)
    # --- END ---

    all_features_df_list = []
    for i, model_name in enumerate(model_names):
        if verbose:
            layer_desc = f"layer {layer_name}" if layer_name else "all layers"
            print(f"Processing model: {model_name} for {layer_desc}...")

        model = model_for_conversion if i == 0 else load_model(model_name)

        # --- SINGLE-PASS MULTI-LAYER PREDICTION ---
        # This predict call returns a list of predictions for the requested layers
        predictions_all_layers = predict_intermediate_layers(
            model, graph_inputs, layer_indices, verbose=verbose
        )
        # --- END ---

        # Process results for each requested layer
        for (suffix, layer_idx), layer_predictions in zip(requested_layers.items(), predictions_all_layers):
            n_features = layer_predictions.shape[-1]
            result_data = np.full((len(structure_list), n_features), np.nan)
            result_data[valid_indices] = layer_predictions.squeeze()

            columns = [f"{suffix}_{model_name}_{j+1}" for j in range(n_features)]
            feature_df = pd.DataFrame(result_data, columns=columns, index=indexes)
            all_features_df_list.append(feature_df)

        if verbose:
            layer_desc = f"layer {layer_name}" if layer_name else "all layers"
            print(f"Features for {layer_desc} calculated for model {model_name}.")

    return pd.concat(all_features_df_list, axis=1)
def get_Custom_MEGNetFeatures(structures, model_type: str, **kwargs) -> pd.DataFrame:
    """
    Extracts features from a custom-trained MEGNet model.
    """
    verbose = kwargs.get('verbose', True)
    try:
        package_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        package_base_dir = '.'

    if model_type == 'MatMinerEncoded_v1':
        model_file = kwargs.get('model_file', os.path.join(package_base_dir, 'featurizers', 'custom_models', 'MEGNetModel__MatMinerEncoded_v1.h5'))
        n_targets, neuron_layers, model_name = 758, (64, 128, 64), "MatMinerEncoded_v1"
    elif model_type == 'OFMEncoded_v1':
        model_file = kwargs.get('model_file', os.path.join(package_base_dir, 'featurizers', 'custom_models', 'MEGNetModel__OFMEncoded_v1.h5'))
        n_targets, neuron_layers, model_name = 188, (64, 128, 64), "OFMEncoded_v1"
    else:
        raise ValueError(f"model_type '{model_type}' not recognized.")

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")

    file_path_without_ext, _ = os.path.splitext(model_file)
    scaler_file = kwargs.get('scaler_file', file_path_without_ext + "_scaler.pkl")

    model, scaler = load_model_scaler(n_targets=n_targets, neuron_layers=neuron_layers,
                                      model_file=model_file, scaler_file=scaler_file, **kwargs)

    indexes = structures.index.to_list() if hasattr(structures, 'index') else list(range(len(structures)))
    structure_list = list(structures)

    if verbose:
        print(f"Processing {len(structure_list)} structures with {model_name} model...")

    valid_structures, graph_inputs, valid_indices = batch_convert_structures_to_graphs(model, structure_list, verbose=verbose)

    result_data = np.full((len(structure_list), n_targets), np.nan)

    if valid_structures:
        # Pass the Pymatgen structures to the correct prediction function for final outputs.
        predictions = predict_structures_with_model(model, valid_structures, verbose=verbose)
        
        if scaler:
            predictions = scaler.inverse_transform(predictions)
        
        result_data[valid_indices] = predictions

    columns = [f"MEGNet_{model_name}_{i+1}" for i in range(n_targets)]
    return pd.DataFrame(result_data, columns=columns, index=indexes)

def get_Adjacent_MEGNetFeatures(structures, layer_name: str = 'layer32', **kwargs) -> pd.DataFrame:
    """
    Extracts features from an 'adjacent' on-the-fly trained MEGNet model.
    """
    model_path = kwargs.get('model_path', '')
    model_file = os.path.join(model_path, kwargs.get('model_file', 'MEGNetModel__adjacent.h5'))
    scaler_file = os.path.join(os.path.dirname(model_file), kwargs.get('scaler_file', 'MEGNetModel__adjacent_scaler.pkl'))

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"{model_file} not found. Please train the model first.")
        
    verbose = kwargs.get('verbose', True)
    model_name = kwargs.get('model_name', 'AdjacentMEGNet')
    
    model, _ = load_model_scaler(model_file=model_file, scaler_file=scaler_file, **kwargs)
    
    layer_mapping = {'layer16': -2, 'layer32': -3}
    if layer_name not in layer_mapping:
        raise ValueError("Invalid layer_name. Choose 'layer16' or 'layer32'.")
    layer_idx = layer_mapping[layer_name]

    indexes = structures.index.to_list() if hasattr(structures, 'index') else list(range(len(structures)))
    structure_list = list(structures)
    
    if verbose:
        print(f"Processing {len(structure_list)} structures with {model_name} model...")

    valid_structures, graph_inputs, valid_indices = batch_convert_structures_to_graphs(model, structure_list, verbose=verbose)

    n_features = 32 if layer_name == 'layer32' else 16
    result_data = np.full((len(structure_list), n_features), np.nan)

    if graph_inputs:
        # predict_intermediate_layers expects a list of layer indices
        predictions = predict_intermediate_layers(model, graph_inputs, [layer_idx], verbose=verbose)
        # predictions is a list with one element (for the single layer)
        result_data[valid_indices] = predictions[0].squeeze()

    columns = [f"{model_name}_{layer_name}_{i + 1}" for i in range(n_features)]
    return pd.DataFrame(result_data, columns=columns, index=indexes)

### BACKWARD COMPATIBILITY WRAPPERS

def get_MVL_MEGNetFeatures_legacy(structures, layer_name='layer32'):
    return get_MVL_MEGNetFeatures(structures, layer_name=layer_name, verbose=False)

def get_Custom_MEGNetFeatures_legacy(structures, model_type, **kwargs):
    return get_Custom_MEGNetFeatures(structures, model_type, verbose=False, **kwargs)

def get_Adjacent_MEGNetFeatures_legacy(structures, **kwargs):
    return get_Adjacent_MEGNetFeatures(structures, verbose=False, **kwargs)

def megnet_evaluate_structures_legacy(model, structures, **kwargs):
    return megnet_evaluate_structures(model, structures, verbose=False, **kwargs)

__all__ = [
    'model_setup', 'load_model_scaler', 'megnet_evaluate_structures', 'train_MEGNet_on_the_fly',
    'get_MVL_MEGNetFeatures', 'get_Custom_MEGNetFeatures', 'get_Adjacent_MEGNetFeatures',
    'batch_convert_structures_to_graphs', 'predict_structures_with_model', 
    'predict_graphs_with_model',  'predict_intermediate_layers',
    # Legacy functions for backward compatibility
    'get_MVL_MEGNetFeatures_legacy', 'get_Custom_MEGNetFeatures_legacy',
    'get_Adjacent_MEGNetFeatures_legacy', 'megnet_evaluate_structures_legacy'
]