"""  
This script featurizes crystal structures from the 'matbench_mp_gap' dataset 
which is basically MP 2018 dataset filtered for stability using a pre-trained 
coGN (as the default adjacent model) model. It loads structures via matminer, 
preprocesses them into graph representations, extracts features and multiplicities 
using the coGN model, and saves the results to a CSV.  
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
from pymatgen.core import Structure, Lattice
import networkx as nx
import pandas as pd
import json
from matminer.datasets import load_dataset

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


def featurize_structures_from_dataframe(df_input: pd.DataFrame, weights_path: str) -> pd.DataFrame:
   """
   Loads a DataFrame with a 'structure' column containing pymatgen structures,
   converts them, calculates features using a pre-trained KGCNN model, and returns
   a featurized pandas DataFrame. Handles variable output shapes from the readout layer
   and includes multiplicity information.

   Args:
      df_input (pd.DataFrame): Input DataFrame containing a 'structure' column.
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
   try:
      final_node_embeddings_tensor = [layer.output[0] for layer in original_model.layers if layer.name.startswith('sequential_graph_network')][0]
      readout_layer = [layer for layer in original_model.layers if layer.name.startswith('graph_network_multiplicity_readout')][0]
   except IndexError:
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

   if not hasattr(featurize_structures_from_dataframe, 'model_summary_printed'):
      feature_extractor_model.summary()
      featurize_structures_from_dataframe.model_summary_printed = True


   num_features_per_component = 128

   # --- 3. Process DataFrame ---
   if 'structure' not in df_input.columns:
      raise ValueError("Input DataFrame must contain a 'structure' column with pymatgen structures.")

   all_flattened_features = []
   all_multiplicities = []
   max_num_components_observed = 0

   preprocessor = KNNAsymmetricUnitCell(24)

   print("Processing structures and extracting features...")
   for index, row in df_input.iterrows():
      current_flattened_features = np.full(0, np.nan)
      current_multiplicity_values = np.full(0, np.nan)
      try:
         pmg_structure = row['structure'] # Structure is already a pymatgen object

         nx_graph = preprocessor(pmg_structure)

         graph_components = convert_nx_to_graph_list_components(nx_graph)

         graph_list = GraphList(
               num_nodes=[graph_components['num_nodes']],
               num_edges=[graph_components['num_edges']],
               edge_indices=[graph_components['edge_indices']],
               node_attributes={key: [value] for key, value in graph_components['node_attributes'].items()},
               edge_attributes={key: [value] for key, value in graph_components['edge_attributes'].items()},
               graph_attributes={key: [value] for key, value in graph_components['graph_attributes'].items()}
         )

         input_tensors = get_input_tensors(feature_extractor_model.inputs, graph_list)

         extracted_outputs = feature_extractor_model.predict(input_tensors, verbose=0)

         final_prediction_output_dict = extracted_outputs[2]
         final_prediction_ragged = final_prediction_output_dict['features']
         multiplicity_ragged = final_prediction_output_dict['multiplicity']

         if final_prediction_ragged.shape[0] == 0 or (final_prediction_ragged.shape[0] > 0 and final_prediction_ragged.row_lengths()[0] == 0):
               print(f"WARNING: Structure {index} resulted in an empty or zero-length final_prediction_ragged tensor. Features will be NaNs.")
         else:
               final_prediction = final_prediction_ragged.numpy()
               current_multiplicity_values = multiplicity_ragged.numpy().flatten()

               current_num_components = final_prediction.shape[1]
               max_num_components_observed = max(max_num_components_observed, current_num_components)

               current_flattened_features = final_prediction.reshape(-1)

         all_flattened_features.append(current_flattened_features)
         all_multiplicities.append(current_multiplicity_values)

      except Exception as e:
         print(f"CRITICAL ERROR processing row {index}: {e}")
         all_flattened_features.append(current_flattened_features)
         all_multiplicities.append(current_multiplicity_values)

   # --- 4. Create Featurized DataFrame ---
   total_features_per_row = max_num_components_observed * num_features_per_component
   total_multiplicities_per_row = max_num_components_observed

   padded_features = []
   padded_multiplicities = []

   for i in range(len(all_flattened_features)):
      features = all_flattened_features[i]
      multiplicities = all_multiplicities[i]

      if features.size < total_features_per_row:
         padded_features.append(np.pad(features, (0, total_features_per_row - features.size), 'constant', constant_values=np.nan))
      else:
         padded_features.append(features)

      if multiplicities.size < total_multiplicities_per_row:
         padded_multiplicities.append(np.pad(multiplicities, (0, total_multiplicities_per_row - multiplicities.size), 'constant', constant_values=np.nan))
      else:
         padded_multiplicities.append(multiplicities)

   feature_column_names = []
   for component_idx in range(max_num_components_observed):
      for neuron_idx in range(num_features_per_component):
         feature_column_names.append(f'coGN_ReadoutComponent{component_idx+1}_Feature{neuron_idx+1}')

   multiplicity_column_names = []
   for component_idx in range(max_num_components_observed):
      multiplicity_column_names.append(f'coGN_ReadoutMultiplicity_Component{component_idx+1}')

   df_features = pd.DataFrame(padded_features, columns=feature_column_names)
   df_multiplicities = pd.DataFrame(padded_multiplicities, columns=multiplicity_column_names)

   df_featurized = pd.concat([df_features, df_multiplicities], axis=1)

   print("\n✅ Feature and multiplicity extraction complete!")
   return df_featurized

# --- Main execution block ---
if __name__ == "__main__":
   # Define your weights path
   # For demonstration, I'll use a placeholder. You'll need to provide your actual path.
   # base_weights_path = './results_coGN/matbench_jdft2d/{num}/weights.h5' # Example from your original script
   base_weights_path = './path/to/your/model/{num}/weights.h5' # Placeholder
   num_folds = 5 # Folds 0 to 4

   all_folds_featurized_dfs = []
   original_df_for_merge = None

   try:
      print("Loading 'mp_gap_dataset' using matminer...")
      # Load the dataset directly from matminer
      df_original = load_dataset("matbench_mp_gap")
      print(f"Dataset loaded. Shape: {df_original.shape}")
      print("First 5 rows of the loaded dataset:")
      print(df_original.head())

      # The 'structure' column from matminer.load_dataset is already pymatgen Structure objects.
      # So, no need for json.loads() or Structure.from_dict() in the featurization function.
      # We'll pass this DataFrame directly to the featurization function.
      original_df_for_merge = df_original.copy()

      # Drop any existing feature columns if they exist
      cols_to_drop = [col for col in original_df_for_merge.columns if col.startswith(('coGN'))]
      if cols_to_drop:
         print(f"Dropping existing coGN/Multiplicity columns from original DataFrame: {cols_to_drop}")
         original_df_for_merge = original_df_for_merge.drop(columns=cols_to_drop)

      for fold_num in range(num_folds):
         weights_path = base_weights_path.format(num=fold_num)
         print(f"\n--- Processing with model fold {fold_num} (weights: {weights_path}) ---")

         # Call the featurization function with the DataFrame
         featurized_df_fold = featurize_structures_from_dataframe(df_original, weights_path)

         # Append _fold{num} to all feature and multiplicity column names
         new_columns = {col: f"{col}_fold{fold_num}" for col in featurized_df_fold.columns}
         featurized_df_fold = featurized_df_fold.rename(columns=new_columns)

         all_folds_featurized_dfs.append(featurized_df_fold)
         print(f"Featurization for fold {fold_num} complete. DataFrame shape: {featurized_df_fold.shape}")

      # Concatenate all featurized DataFrames horizontally
      final_featurized_df = pd.concat([original_df_for_merge] + all_folds_featurized_dfs, axis=1)

      # Define output path (you might want to customize this)
      output_csv_path = Path("./mp_gap_featurized_coGN.csv")

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
      print("Please ensure your model weights files exist at the specified paths.")
   except ValueError as e:
      print(e)
   except Exception as e:
      print(f"An unexpected error occurred: {e}")