import numpy as np
from keras.models import Model
import warnings
from pickle import load, dump
import tensorflow as tf
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Any, List

warnings.filterwarnings("ignore")

from megnet.utils.models import load_model
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GraphBatchGenerator
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping

# ==============================================================================
# OPTIMIZED UNIFIED PROCESSING UTILITIES
# ==============================================================================

def batch_convert_structures_to_graphs(model: MEGNetModel, structures: List,
                                       verbose: bool = True) -> Tuple[List, List, List, List]:
    """
    Convert multiple structures to graph inputs using ThreadPool parallel processing.
    Uses the model's native graph_to_input to guarantee properly expanded bond features.
    """
    valid_structures = []
    graph_inputs = []
    raw_graphs = []
    original_indices = []

    if verbose:
        print(f"Converting {len(structures)} structures to graphs using ThreadPool...")

    def _thread_worker(args):
        idx, structure = args
        try:
            # Safely use the model's native converter
            graph = model.graph_converter.convert(structure)
            
            # Guard against isolated atoms crashing the generator
            if len(graph.get('index1', [])) == 0:
                return idx, structure, None, None, "0 bonds found (isolated atoms)"
                
            # Pre-calculate the Keras-ready input tensor list (EXPANDS GAUSSIAN BONDS)
            inp = model.graph_converter.graph_to_input(graph)
            return idx, structure, graph, inp, None
        except Exception as e:
            return idx, structure, None, None, str(e)

    worker_args = [(idx, struct) for idx, struct in enumerate(structures)]
    num_threads = min(32, multiprocessing.cpu_count() * 4) 
    
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(_thread_worker, arg): arg for arg in worker_args}
        for future in as_completed(futures):
            results.append(future.result())
            
    results.sort(key=lambda x: x[0])
    
    for idx, struct, graph, inp, err in results:
        if err is None:
            valid_structures.append(struct)
            raw_graphs.append(graph)
            graph_inputs.append(inp)
            original_indices.append(idx)

    if verbose:
        print(f"Successfully converted {len(valid_structures)}/{len(structures)} structures.")

    return valid_structures, graph_inputs, raw_graphs, original_indices


def predict_structures_with_model(model: MEGNetModel, structures: List, verbose: bool = True) -> np.ndarray:
    if not structures:
        return np.array([])
    if verbose:
        print(f"Predicting on {len(structures)} structures using model.predict_structures...")
    return model.predict_structures(structures)


def predict_graphs_with_model(model: Model, graph_inputs: List, verbose: bool = True) -> List:
    """Sequential fallback prediction."""
    if not graph_inputs:
        return []
    if verbose:
        print(f"Predicting sequentially (safe fallback mode)...")
    
    predictions = [model.predict(inp, verbose=0) for inp in graph_inputs]
    return predictions

def predict_intermediate_layers(model: MEGNetModel, raw_graphs: List, graph_inputs: List, layer_indices: List[int],
                                verbose: bool = True) -> List[np.ndarray]:
    """
    Extracts features from multiple intermediate layers.
    Strips the dummy batch dimension from graph_inputs and pipes them into
    GraphBatchGenerator for massive parallel execution. Safely handles RaggedTensors and Batch Padded Shapes.
    """
    if not graph_inputs:
        return []

    outputs = [model.layers[i].output for i in layer_indices]
    multi_output_model = Model(inputs=model.input, outputs=outputs)
    
    batch_size = 128
    
    try:
        if verbose:
            print(f"Extracting intermediate layers in batches of {batch_size}...")
            
        atom_features = [inp[0][0] for inp in graph_inputs]
        bond_features = [inp[1][0] for inp in graph_inputs]
        state_features = [inp[2][0] for inp in graph_inputs]
        index1_list = [inp[3][0] for inp in graph_inputs]
        index2_list = [inp[4][0] for inp in graph_inputs]
        
        # Generator requires dummy targets
        dummy_targets = np.zeros(len(graph_inputs))

        generator = GraphBatchGenerator(
            atom_features, bond_features, state_features, index1_list, index2_list,
            targets=dummy_targets, batch_size=batch_size, is_shuffle=False
        )
        
        # Predict using the generator
        predictions = multi_output_model.predict(generator, verbose=1 if verbose else 0)
        
        if predictions is None:
            return []

        # --- THE FIX: CLEANUP TENSORS AND BATCH DIMENSIONS ---
        def to_numpy_safe(x):
            if hasattr(x, 'to_tensor'):  # Convert RaggedTensor
                x = x.to_tensor()
            if hasattr(x, 'numpy'):      # Convert Tensor
                x = x.numpy()
            else:
                x = np.array(x)
            
            # If Keras returned shape (Num_Batches, Batch_Size, Features), reshape it!
            # e.g., (79, 128, 32) -> (10112, 32)
            if x.ndim == 3:
                x = np.vstack(x)
                
            # Slice off any padding added by the batch generator to match exact input length
            # e.g., 10112 -> 10000
            return x[:len(graph_inputs)]

        # Ensure multi-output predictions are wrapped in a list uniformly
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        return [to_numpy_safe(p) for p in predictions]

    except Exception as e:
        print(f"\n[DEBUG] ❌ Fast Batch Prediction Failed! Error: {str(e)}")
        print(f"[DEBUG] Triggering Safe Sequential Fallback...")
        
        raw_predictions = predict_graphs_with_model(multi_output_model, graph_inputs, verbose=verbose)
        
        if not raw_predictions:
            return []
            
        if len(layer_indices) == 1:
            return [np.concatenate(raw_predictions, axis=0)]
            
        predictions_by_layer = list(zip(*raw_predictions))
        return [np.concatenate(layer_preds, axis=0) for layer_preds in predictions_by_layer]
    
### FUNCTIONS TO SETUP, EVALUATE AND TRAIN MEGNET MODELS

def model_setup(ntarget: int = None, **kwargs) -> MEGNetModel:
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
    labels = kwargs.get('labels', [''] * len(structures))
    verbose = kwargs.get('verbose', True)

    if targets is None:
        noTargets = True
        target_values = np.ones(len(structures))
    else:
        noTargets = False
        target_values = targets.values if isinstance(targets, pd.DataFrame) else targets

    valid_structures, graph_inputs, raw_graphs, valid_indices = batch_convert_structures_to_graphs(
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

    ypred = predict_structures_with_model(model, valid_structures, verbose=verbose)

    y = np.array(targets_valid).squeeze()
    labels_out = np.array(labels_valid)

    if noTargets:
        return (valid_structures, ypred)
    else:
        return (valid_structures, ypred, y, labels_out)

def train_MEGNet_on_the_fly(structures, targets, **kwargs):
    from sklearn.preprocessing import MinMaxScaler
    targets = np.array(targets).reshape(-1, 1)
    scaler = MinMaxScaler()
    targets = scaler.fit_transform(targets)

    adjacent_model_path = kwargs.get('adjacent_model_path', '.')
    os.makedirs(adjacent_model_path, exist_ok=True)

    dump(scaler, open(os.path.join(adjacent_model_path, 'MEGNetModel__adjacent_scaler.pkl'), 'wb'))
    print('Scaler for the adjacent model saved to MEGNetModel__adjacent_scaler.pkl')

    max_epochs = kwargs.get('max_epochs', 100)
    patience = kwargs.get('patience', 10)
    early_stopping = EarlyStopping(monitor='val_mae', patience=patience, restore_best_weights=True)

    r_cutoff = kwargs.get('r_cutoff', 5.0)
    nfeat_bond = kwargs.get('nfeat_bond', 100)
    gaussian_width = kwargs.get('gaussian_width', 0.5)

    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    graph_converter = CrystalGraph(cutoff=r_cutoff)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_index, val_index = next(kf.split(structures))

    train_structures, val_structures = structures.iloc[train_index], structures.iloc[val_index]
    train_targets, val_targets = targets[train_index], targets[val_index]

    model = MEGNetModel(
        metrics=['mae'],
        graph_converter=graph_converter,
        centers=gaussian_centers,
        width=gaussian_width,
        ntarget=1,
        **{k: v for k, v in kwargs.items() if k in ['n1', 'n2', 'n3']}
    )

    model.train(train_structures, train_targets, validation_structures=val_structures,
                validation_targets=val_targets, epochs=max_epochs, save_checkpoint=False, callbacks=[early_stopping])

    model_save_path = os.path.join(adjacent_model_path, 'MEGNetModel__adjacent.h5')
    model.save(model_save_path)
    print(f'On-the-fly MEGNet model saved to {model_save_path}')

def get_MVL_MEGNetFeatures(structures, **kwargs) -> pd.DataFrame:
    verbose = kwargs.get('verbose', True)
    layer_name = kwargs.get('layer_name', None)
    indexes = structures.index.to_list() if hasattr(structures, 'index') else list(range(len(structures)))
    structure_list = list(structures)

    layer_config = {'MVL32': -3, 'MVL16': -2}

    if layer_name is not None:
        if layer_name == 'layer32':
            requested_layers = {'MVL32': -3}
        elif layer_name == 'layer16':
            requested_layers = {'MVL16': -2}
        else:
            raise ValueError(f"Invalid layer_name '{layer_name}'. Valid options: 'layer32', 'layer16'")
    else:
        requested_layers = layer_config

    layer_indices = list(requested_layers.values())

    model_names = ["Eform_MP_2019", 'Efermi_MP_2019', "Bandgap_classifier_MP_2018",
                   'Bandgap_MP_2018', 'logK_MP_2019', 'logG_MP_2019']

    if verbose:
        print(f"Preparing graphs for all {len(model_names)} MVL models...")
    model_for_conversion = load_model(model_names[0])
    
    valid_structures, graph_inputs, raw_graphs, valid_indices = batch_convert_structures_to_graphs(
        model_for_conversion, structure_list, verbose=verbose
    )

    if not graph_inputs:
        print("Warning: No structures could be converted. Returning empty DataFrame.")
        return pd.DataFrame(index=indexes)

    all_features_df_list = []
    for i, model_name in enumerate(model_names):
        if verbose:
            layer_desc = f"layer {layer_name}" if layer_name else "all layers"
            print(f"Processing model: {model_name} for {layer_desc}...")

        model = model_for_conversion if i == 0 else load_model(model_name)

        predictions_all_layers = predict_intermediate_layers(
            model, raw_graphs, graph_inputs, layer_indices, verbose=verbose
        )

        for (suffix, layer_idx), layer_predictions in zip(requested_layers.items(), predictions_all_layers):
            if layer_predictions is None or len(layer_predictions) == 0:
                continue
                
            n_features = layer_predictions.shape[-1]
            result_data = np.full((len(structure_list), n_features), np.nan)
            
            result_data[valid_indices] = layer_predictions.squeeze()

            columns = [f"{suffix}_{model_name}_{j+1}" for j in range(n_features)]
            feature_df = pd.DataFrame(result_data, columns=columns, index=indexes)
            all_features_df_list.append(feature_df)

        if verbose:
            layer_desc = f"layer {layer_name}" if layer_name else "all layers"
            print(f"Features for {layer_desc} calculated for model {model_name}.")

    if not all_features_df_list:
        return pd.DataFrame(index=indexes)
        
    return pd.concat(all_features_df_list, axis=1)

def get_Custom_MEGNetFeatures(structures, model_type: str, **kwargs) -> pd.DataFrame:
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

    valid_structures, graph_inputs, raw_graphs, valid_indices = batch_convert_structures_to_graphs(
        model, structure_list, verbose=verbose
    )

    result_data = np.full((len(structure_list), n_targets), np.nan)

    if valid_structures:
        predictions = predict_structures_with_model(model, valid_structures, verbose=verbose)
        
        if scaler:
            predictions = scaler.inverse_transform(predictions)
        
        result_data[valid_indices] = predictions

    columns = [f"MEGNet_{model_name}_{i+1}" for i in range(n_targets)]
    return pd.DataFrame(result_data, columns=columns, index=indexes)

def get_Adjacent_MEGNetFeatures(structures, layer_name: str = 'layer32', **kwargs) -> pd.DataFrame:
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

    valid_structures, graph_inputs, raw_graphs, valid_indices = batch_convert_structures_to_graphs(
        model, structure_list, verbose=verbose
    )

    n_features = 32 if layer_name == 'layer32' else 16
    result_data = np.full((len(structure_list), n_features), np.nan)

    if graph_inputs:
        predictions = predict_intermediate_layers(model, raw_graphs, graph_inputs, [layer_idx], verbose=verbose)
        if len(predictions) > 0:
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