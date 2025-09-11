import numpy as np
import tensorflow as tf
from pathlib import Path
from pymatgen.core import Structure, Lattice
import networkx as nx
import pandas as pd
import json

# KGCNN imports for model architecture and graph preprocessing
from kgcnn.literature.coGN import make_model, model_default
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell
from graphlist import GraphList # Custom GraphList class for KGCNN input format


def convert_nx_to_graph_list_components(nx_graph: nx.Graph) -> dict:
    """
    Converts a networkx.Graph object (output from KGCNN preprocessor)
    into the dictionary format required by the GraphList constructor.
    This ensures compatibility with the model's input expectations.
    """
    node_indices = sorted(nx_graph.nodes())
    node_attributes = {
        'atomic_number': np.array([nx_graph.nodes[i]['atomic_number'] for i in node_indices]),
        'multiplicity': np.array([nx_graph.nodes[i]['multiplicity'] for i in node_indices])
    }

    edge_indices = []
    edge_offsets = []
    for u, v, data in nx_graph.edges(data=True):
        edge_indices.append([u, v])
        edge_offsets.append(data['offset'])

    edge_attributes = {'offset': np.array(edge_offsets)} if edge_offsets else {}

    return {
        'num_nodes': nx_graph.number_of_nodes(),
        'num_edges': nx_graph.number_of_edges(),
        'edge_indices': np.array(edge_indices),
        'node_attributes': node_attributes,
        'edge_attributes': edge_attributes,
        'graph_attributes': nx_graph.graph # Often an empty dict, but included for completeness
    }


def get_input_tensors(model_inputs, graph_list):
    """
    Prepares a single GraphList object into a dictionary of TensorFlow RaggedTensors,
    matching the input signature expected by the KGCNN model.
    """
    input_names = [inp.name for inp in model_inputs]
    input_tensors = {}

    # Populate node attributes as RaggedTensors
    for input_name, attr_list in graph_list.node_attributes.items():
        if input_name in input_names:
            input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                attr_list[0], [graph_list.num_nodes[0]]
            )

    # Populate edge attributes as RaggedTensors
    for input_name, attr_list in graph_list.edge_attributes.items():
        if input_name in input_names:
            input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                attr_list[0], [graph_list.num_edges[0]]
            )

    # Handle edge indices, reversing them as required by some KGCNN layers
    if 'edge_indices' in input_names:
        reversed_indices = graph_list.edge_indices[0][:, [1, 0]]
        input_tensors['edge_indices'] = tf.RaggedTensor.from_row_lengths(
            reversed_indices, [graph_list.num_edges[0]]
        )

    return input_tensors


def featurize_structures_from_csv(csv_path: str, weights_path: str) -> pd.DataFrame:
    """
    Loads a CSV dataset with a 'structure' column containing JSON-dumped pymatgen structures,
    converts them, calculates features using a pre-trained KGCNN model, and returns
    a featurized pandas DataFrame. Handles variable output shapes from the readout layer
    and includes multiplicity information.

    Args:
        csv_path (str): Path to the input CSV file.
        weights_path (str): Path to the pre-trained KGCNN model weights file (.h5).

    Returns:
        pd.DataFrame: A DataFrame with extracted features and multiplicity for each structure.
                      Features and multiplicities are padded with NaNs if a structure yields
                      fewer components than the maximum observed.
    """
    # --- 1. Configuration and Model Loading ---
    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"Error: Model weights file not found at {weights_path}")

    print(f"Loading pre-trained coGN model architecture and weights from '{weights_file}'...")
    original_model = make_model(**model_default)
    original_model.load_weights(str(weights_file))

    # --- 2. Define Feature Extractor Model ---
    # Find layers by name prefix to handle potential numerical suffixes (e.g., _1, _2)
    # This logic needs to be applied consistently.
    try:
        final_node_embeddings_tensor = [layer.output[0] for layer in original_model.layers if layer.name.startswith('sequential_graph_network')][0]
        readout_layer = [layer for layer in original_model.layers if layer.name.startswith('graph_network_multiplicity_readout')][0]
    except IndexError:
        # Fallback or more specific error if layers aren't found as expected
        print("ERROR: Could not find 'sequential_graph_network' or 'graph_network_multiplicity_readout' layers by prefix.")
        print("Existing layers:", [layer.name for layer in original_model.layers])
        raise

    pooled_graph_features_tensor = readout_layer.input[0]
    final_prediction_output_dict_tensor = readout_layer.output[1]

    feature_extractor_model = tf.keras.Model(
        inputs=original_model.inputs,
        outputs=[final_node_embeddings_tensor, pooled_graph_features_tensor, final_prediction_output_dict_tensor],
        name="FeatureExtractor"
    )

    # Only print summary once to avoid clutter if called in a loop
    if not hasattr(featurize_structures_from_csv, 'model_summary_printed'):
        feature_extractor_model.summary()
        featurize_structures_from_csv.model_summary_printed = True # Mark as printed


    # --- Define feature dimensions (fixed for the feature dimension, but components can vary) ---
    num_features_per_component = 128

    # --- 3. Load and Process CSV ---
    print(f"Loading data from {csv_path}...")
    df_input = pd.read_csv(csv_path)
    if 'structure' not in df_input.columns:
        raise ValueError("CSV file must contain a 'structure' column with JSON-dumped pymatgen structures.")

    all_flattened_features = []
    all_multiplicities = [] # New list to store multiplicities
    max_num_components_observed = 0 # To track the maximum number of components for DataFrame columns

    preprocessor = KNNAsymmetricUnitCell(24)

    print("Processing structures and extracting features...")
    for index, row in df_input.iterrows():
        # print(f"\n--- Processing structure at index {index} (ID: {row.get('id', 'N/A')}) ---") # Too verbose for large datasets
        current_flattened_features = np.full(0, np.nan) # Initialize with empty NaN array
        current_multiplicity_values = np.full(0, np.nan) # Initialize with empty NaN array
        try:
            # Deserialize the pymatgen structure from JSON string
            structure_dict = json.loads(row['structure'])
            pmg_structure = Structure.from_dict(structure_dict)
            # print(f"Structure {index} loaded: {pmg_structure.formula}") # Too verbose

            # Preprocess the structure into a networkx graph
            nx_graph = preprocessor(pmg_structure)
            # print(f"Structure {index} preprocessed. Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}") # Too verbose

            # Convert the networkx graph into GraphList components
            graph_components = convert_nx_to_graph_list_components(nx_graph)
            # print(f"Graph components for structure {index}: num_nodes={graph_components['num_nodes']}, num_edges={graph_components['num_edges']}") # Too verbose

            # Construct the GraphList object, wrapping components in lists for batching
            graph_list = GraphList(
                num_nodes=[graph_components['num_nodes']],
                num_edges=[graph_components['num_edges']],
                edge_indices=[graph_components['edge_indices']],
                node_attributes={key: [value] for key, value in graph_components['node_attributes'].items()},
                edge_attributes={key: [value] for key, value in graph_components['edge_attributes'].items()},
                graph_attributes={key: [value] for key, value in graph_components['graph_attributes'].items()}
            )

            # Prepare input tensors for the feature extractor model
            input_tensors = get_input_tensors(feature_extractor_model.inputs, graph_list)
            # print(f"Input tensors prepared for structure {index}. Keys: {input_tensors.keys()}") # Too verbose

            # Predict and Extract Outputs
            extracted_outputs = feature_extractor_model.predict(input_tensors, verbose=0)

            final_prediction_output_dict = extracted_outputs[2]
            final_prediction_ragged = final_prediction_output_dict['features']
            multiplicity_ragged = final_prediction_output_dict['multiplicity'] # Extract multiplicity

            # print(f"Structure {index} - final_prediction_ragged shape: {final_prediction_ragged.shape}") # Too verbose
            # print(f"Structure {index} - multiplicity_ragged shape: {multiplicity_ragged.shape}") # Too verbose


            # Process features
            if final_prediction_ragged.shape[0] == 0 or (final_prediction_ragged.shape[0] > 0 and final_prediction_ragged.row_lengths()[0] == 0):
                print(f"WARNING: Structure {index} resulted in an empty or zero-length final_prediction_ragged tensor. Features will be NaNs.")
                # current_flattened_features and current_multiplicity_values are already initialized to empty NaN arrays
            else:
                final_prediction = final_prediction_ragged.numpy()
                current_multiplicity_values = multiplicity_ragged.numpy().flatten() # Flatten multiplicity
                # print(f"Structure {index} - final_prediction numpy shape: {final_prediction.shape}") # Too verbose
                # print(f"Structure {index} - multiplicity numpy values: {current_multiplicity_values}") # Too verbose

                current_num_components = final_prediction.shape[1]
                max_num_components_observed = max(max_num_components_observed, current_num_components)

                current_flattened_features = final_prediction.reshape(-1)

            all_flattened_features.append(current_flattened_features)
            all_multiplicities.append(current_multiplicity_values) # Store multiplicity

        except Exception as e:
            print(f"CRITICAL ERROR processing row {index}: {e}")
            # current_flattened_features and current_multiplicity_values are already initialized to empty NaN arrays
            all_flattened_features.append(current_flattened_features) # Append the initialized NaN array
            all_multiplicities.append(current_multiplicity_values) # Append the initialized NaN array

    # --- 4. Create Featurized DataFrame ---
    # Determine the total number of feature columns and multiplicity columns
    total_features_per_row = max_num_components_observed * num_features_per_component
    total_multiplicities_per_row = max_num_components_observed

    padded_features = []
    padded_multiplicities = []

    for i in range(len(all_flattened_features)):
        features = all_flattened_features[i]
        multiplicities = all_multiplicities[i]

        # Pad features
        if features.size < total_features_per_row:
            padded_features.append(np.pad(features, (0, total_features_per_row - features.size), 'constant', constant_values=np.nan))
        else:
            padded_features.append(features)

        # Pad multiplicities
        if multiplicities.size < total_multiplicities_per_row:
            padded_multiplicities.append(np.pad(multiplicities, (0, total_multiplicities_per_row - multiplicities.size), 'constant', constant_values=np.nan))
        else:
            padded_multiplicities.append(multiplicities)


    # Create column names for features
    feature_column_names = []
    for component_idx in range(max_num_components_observed):
        for neuron_idx in range(num_features_per_component):
            feature_column_names.append(f'coGN_ReadoutComponent{component_idx+1}_Feature{neuron_idx+1}')

    # Create column names for multiplicities
    multiplicity_column_names = []
    for component_idx in range(max_num_components_observed):
        multiplicity_column_names.append(f'coGN_ReadoutMultiplicity_Component{component_idx+1}')

    # Combine features and multiplicities into a single DataFrame
    df_features = pd.DataFrame(padded_features, columns=feature_column_names)
    df_multiplicities = pd.DataFrame(padded_multiplicities, columns=multiplicity_column_names)

    # Concatenate the two DataFrames
    df_featurized = pd.concat([df_features, df_multiplicities], axis=1)

    print("\n✅ Feature and multiplicity extraction complete!")
    return df_featurized

# --- Main execution block ---
if __name__ == "__main__":
    # input_csv_path = '/gpfs/scratch/acad/htforft/rgouvea/matbench_tests/data/matbench_perovskites/matbench_perovskites_featurizedMM2020Struct_mattervial.csv'
    # output_csv_prefix = 'matbench_perovskites_featurizedMM2020Struct_mattervial_coGNadj'
    # base_weights_path = './results_coGN/matbench_perovskites/{num}/weights.h5'

    input_csv_path = '/gpfs/scratch/acad/htforft/rgouvea/matbench_tests/data/matbench_jdft2d/matbench_jdft2d_featurizedMM2020_mattervial.csv'
    output_csv_prefix = 'matbench_jdft2d_featurizedMM2020Struct_mattervial_coGNadj'
    base_weights_path = './results_coGN/matbench_jdft2d/{num}/weights.h5'
    num_folds = 5 # Folds 0 to 4

    all_folds_featurized_dfs = []
    original_df_for_merge = None

    try:
        # Load the original DataFrame once to get non-feature columns (like 'structure', 'id', etc.)
        # We'll merge the featurized data back to this later.
        print(f"Loading original data from {input_csv_path} for merging...")
        original_df_for_merge = pd.read_csv(input_csv_path)
        
        # Ensure 'structure' column is present, as it's needed for featurization
        if 'structure' not in original_df_for_merge.columns:
            raise ValueError(f"The input CSV '{input_csv_path}' must contain a 'structure' column.")

        # Drop any existing feature columns from the original_df if they exist,
        # to avoid conflicts when concatenating new features.
        # This is a heuristic; adjust if your original CSV has specific columns you want to keep.
        cols_to_drop = [col for col in original_df_for_merge.columns if col.startswith(('coGN'))]
        if cols_to_drop:
            print(f"Dropping existing coGN/Multiplicity columns from original DataFrame: {cols_to_drop}")
            original_df_for_merge = original_df_for_merge.drop(columns=cols_to_drop)

        for fold_num in range(num_folds):
            weights_path = base_weights_path.format(num=fold_num)
            print(f"\n--- Processing with model fold {fold_num} (weights: {weights_path}) ---")

            # Call the featurization function
            # Pass the input_csv_path directly, as the function handles loading it.
            featurized_df_fold = featurize_structures_from_csv(input_csv_path, weights_path)

            # Append _fold{num} to all feature and multiplicity column names
            new_columns = {col: f"{col}_fold{fold_num}" for col in featurized_df_fold.columns}
            featurized_df_fold = featurized_df_fold.rename(columns=new_columns)

            all_folds_featurized_dfs.append(featurized_df_fold)
            print(f"Featurization for fold {fold_num} complete. DataFrame shape: {featurized_df_fold.shape}")

        # Concatenate all featurized DataFrames horizontally
        # We need to ensure the index aligns for concatenation.
        # If original_df_for_merge has an 'id' column, we can use that for merging.
        # Otherwise, we assume row order is consistent.
        if 'id' in original_df_for_merge.columns:
            # Use 'id' for merging to be robust against potential row reordering
            final_featurized_df = original_df_for_merge.copy()
            for df_fold in all_folds_featurized_dfs:
                # Ensure 'id' column is present in featurized_df_fold if it's needed for merging
                # For now, assuming featurized_df_fold only contains features and multiplicity
                # and we merge based on index. If 'id' is needed, it should be passed through featurize_structures_from_csv
                # or added back here. For simplicity, we'll rely on index alignment for now.
                # If the original_df_for_merge has a unique index, we can just concat.
                # If not, we might need to reset index and merge on 'id'.
                pass # We will concatenate directly assuming consistent row order.

            # Concatenate all featurized dataframes
            # Ensure all featurized_df_fold have the same index as original_df_for_merge
            # This is crucial for correct concatenation.
            # If original_df_for_merge has a default integer index, and featurize_structures_from_csv
            # processes rows in order, this should work.
            final_featurized_df = pd.concat([original_df_for_merge] + all_folds_featurized_dfs, axis=1)

        else:
            # If no 'id' column, just concatenate assuming row order is preserved
            final_featurized_df = pd.concat([original_df_for_merge] + all_folds_featurized_dfs, axis=1)


        # Construct the output file path
        input_path_obj = Path(input_csv_path)
        output_csv_path = input_path_obj.parent / f"{output_csv_prefix}.csv"

        print(f"\nSaving final combined featurized data to: {output_csv_path}")
        final_featurized_df.to_csv(output_csv_path, index=False)
        print(f"Final DataFrame shape: {final_featurized_df.shape}")

        # Check for NaNs in the final DataFrame
        nan_columns = final_featurized_df.isnull().sum()
        nan_columns = nan_columns[nan_columns > 0]
        if not nan_columns.empty:
            print("\nColumns with NaN values in final DataFrame:")
            print(nan_columns)
            print("\nRows with NaN values in final DataFrame (first 10 indices):")
            print(final_featurized_df[final_featurized_df.isnull().any(axis=1)].index.tolist()[:10])
        else:
            print("\nNo NaN values found in the final featurized DataFrame.")

    except FileNotFoundError as e:
        print(e)
        print("Please ensure your input CSV and model weights files exist at the specified paths.")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")