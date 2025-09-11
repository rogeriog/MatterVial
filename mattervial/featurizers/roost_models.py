import os
import types
import tempfile
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from mattervial.packages.roost.roost.data import CompositionData, collate_batch
from mattervial.packages.roost.roost.model import Roost
from pymatgen.core.composition import Composition
from pymatgen.core import Structure
import torch

def get_RoostFeatures(compositions_input, model_type=None, model_file=None, embedding_filepath=None, **kwargs): # Renamed input slightly for clarity
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
      print(f"CUDA device count: {torch.cuda.device_count()}")
      print(f"Current CUDA device: {torch.cuda.current_device()}")
      print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

   if model_type == 'oqmd_eform':
      model_file = 'oqmd_eform_checkpoint-r0.pth.tar'
      parent_dir = os.path.dirname(__file__)
      model_file = os.path.join(parent_dir, 'custom_models', model_file)
      print("Using OQMD_Eform ROOST model from:", model_file)
   elif model_type == 'mpgap':
      model_file = 'mpgap_checkpoint-r0.pth.tar'
      parent_dir = os.path.dirname(__file__)
      model_file = os.path.join(parent_dir, 'custom_models', model_file)
      print("Using MP_GAP ROOST model from:", model_file)
   elif model_file is not None: ## if a custom model is provided
      model_file = model_file
      print("Using custom model:", model_file)
      print("Specify fea_path if not using default (MatScholar).")
   else:
      raise ValueError("Please provide a valid model file or specify a model type ('mpgap' or 'oqmd_eform').")

   # Suffixes
   suffix = model_type if model_type is not None else kwargs.get('suffix', 'custom')

   # Define the paths to the data and featurizer embedding files.
   FEA_PATH = os.path.join(os.path.dirname(__file__), 'custom_models', 'matscholar-embedding.json')
   if embedding_filepath is not None:
      FEA_PATH = embedding_filepath
   # Define a hook factory to capture activations.
   def get_hook(name, storage):
      def hook(module, input, output):
         if isinstance(output, types.GeneratorType):
               output_list = list(output)
               output_clean = [o.detach() if hasattr(o, "detach") else o for o in output_list]
         else:
               output_clean = output.detach() if hasattr(output, "detach") else output
         storage[name] = output_clean
      return hook

   # Task settings and featurizer embedding file.
   task_dict = {}

   # --- START FIX ---
   # Validate input and CAPTURE ORIGINAL INDEX
   if not isinstance(compositions_input, pd.Series):
      raise ValueError("Input must be a pandas Series of pymatgen structures or compositions.")
   original_index = compositions_input.index # Capture the original index HERE
   compositions = compositions_input # Use a new variable if preferred, or reuse

   # if pymatgen structures, convert to compositions, PRESERVING the index
   if isinstance(compositions.iloc[0], Structure):
      compositions = pd.Series(
          [Composition(struc.composition) for struc in compositions],
          index=original_index # Explicitly set the index for the new Series
      )

   # Convert the pandas Series to a DataFrame with a column named 'composition'
   compositions_df = pd.DataFrame({'composition': compositions})
   # Use the captured original_index for the material_id column
   compositions_df['material_id'] = original_index

   # Create a temporary CSV file for the data.
   data_path = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
   # Save the compositions to the temporary CSV file, with 'material_id' as the first column.
   compositions_df.to_csv(data_path, index=False, columns=['material_id', 'composition'])
   # print("DataFrame saved to temp CSV:") # Debug print
   # print(compositions_df.head())


   # Create a temporary CompositionData to obtain element embedding length.
   tmp_dataset = CompositionData(data_path=data_path, fea_path=FEA_PATH, task_dict=task_dict)
   elem_emb_len = tmp_dataset.elem_emb_len
   n_targets = [2] # This seems arbitrary for feature extraction? Might not matter.
   robust = False
   elem_fea_len = kwargs.get('elem_fea_len', 64)
   n_graph = kwargs.get('n_graph', 3)

   # Instantiate Roost models with defined architecture.
   model_base = Roost(
      task_dict=task_dict,
      robust=robust,
      n_targets=n_targets,
      elem_emb_len=elem_emb_len,
      elem_fea_len=elem_fea_len,
      n_graph=n_graph,
      elem_heads=kwargs.get('elem_heads', 3),
      elem_gate=kwargs.get('elem_gate', [256]),
      elem_msg=kwargs.get('elem_msg', [256]),
      cry_heads=kwargs.get('cry_heads', 3),
      cry_gate=kwargs.get('cry_gate', [256]),
      cry_msg=kwargs.get('cry_msg', [256]),
      trunk_hidden=kwargs.get('trunk_hidden', [1024, 512]),
      out_hidden=kwargs.get('out_hidden', [256, 128, 64]),
      device="gpu" 
   )

   checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
   state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
   model_base.load_state_dict(state_dict)
   model_base.eval()
   print("Model loaded successfully!")

   # Prepare dictionaries for hook activations.
   activations = {}
   for name, module in model_base.named_modules():
      module.register_forward_hook(get_hook(name, activations))

   print(f"\nProcessing dataset: {data_path}")
   # df_orig = pd.read_csv(data_path) # No need to read again here
   # print(f"Loaded original data with {len(df_orig)} rows.")

   dataset = CompositionData(data_path=data_path, fea_path=FEA_PATH, task_dict=task_dict)
   # Consider increasing batch_size if memory allows, might speed up inference
   loader = DataLoader(dataset, batch_size=kwargs.get('batch_size', 32), shuffle=False, collate_fn=collate_batch)
   features_dict = {}  # Key: material_id (string); Value: feature dict

   with torch.no_grad():
      batch_num = 0
      for batch in loader:
         batch_num += 1
         inputs, targets, comp_ids, comp_formulas = batch # Roost returns tuples/lists

         elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, extra = inputs

         # Process the entire batch with the model
         activations.clear()
         _ = model_base(elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, extra)

         # Extract batch activations (assuming they have batch dimension first)
         output_nn_batch = activations.get('output_nns.0.acts.1', None)
         material_nn_batch = activations.get('material_nn', None)

         if output_nn_batch is not None:
             output_nn_batch_np = output_nn_batch.cpu().numpy()
         else:
             output_nn_batch_np = None

         if material_nn_batch is not None:
             material_nn_batch_np = material_nn_batch.cpu().numpy()
         else:
             material_nn_batch_np = None

         # Iterate through items in the batch
         for i in range(len(comp_ids)):
             comp_id_str = str(comp_ids[i]) # Convert ID to string for dict key
             comp_formula = comp_formulas[i]
             record = {}
             record["material_id"] = comp_id_str
             record["comp_formula"] = comp_formula

             if output_nn_batch_np is not None:
                 output_nn_act = output_nn_batch_np[i] # Get features for this specific sample
                 for j, val in enumerate(output_nn_act):
                    record[f"ROOST_{suffix}_LayerOutput_#{j+1:02d}"] = val

             if material_nn_batch_np is not None:
                 material_nn_act = material_nn_batch_np[i] # Get features for this specific sample
                 for j, val in enumerate(material_nn_act):
                    record[f"ROOST_{suffix}_LayerMaterialPooling_#{j+1:02d}"] = val

             features_dict[comp_id_str] = record

         if batch_num % 200 == 0: # Print progress more often if large dataset
             print(f"Processed batch {batch_num}...")

   # --- START FIX for Reconstruction ---
   # Now rebuild the features in the order of the *original* input index!
   index_list = list(original_index) # Use the captured original index
   material_id_list = [str(i) for i in index_list]  # String version for dict lookup

   features_list = []
   missing_ids = []
   feature_cols = [] # Keep track of expected feature columns

   # First pass to determine all possible feature columns from the first found entry
   for mid in material_id_list:
       if mid in features_dict:
           feature_cols = list(features_dict[mid].keys())
           feature_cols.remove("material_id")
           feature_cols.remove("comp_formula")
           break # Found one, assuming all others have the same structure

   # Second pass to build the list in the correct order
   for mid in material_id_list:
      if mid in features_dict:
         feat = features_dict[mid].copy()
      else:
         # Create a record with NaNs for missing entries
         missing_ids.append(mid)
         feat = {"material_id": mid, "comp_formula": np.nan}
         for col in feature_cols:
             feat[col] = np.nan
      features_list.append(feat)

   if missing_ids:
       print(f"Warning: Could not find features for the following material_ids: {missing_ids}")

   # Create the final DataFrame using the correctly ordered list and the original index
   RoostFeaturesDF = pd.DataFrame(features_list, index=index_list)

   # print(f"Index of DataFrame right before dropping columns: {RoostFeaturesDF.index.tolist()}") # Debug print

   # Remove material_id, comp_formula for final return (optional but typical)
   if "material_id" in RoostFeaturesDF.columns:
        RoostFeaturesDF.drop(columns=["material_id"], inplace=True)
   if "comp_formula" in RoostFeaturesDF.columns:
        RoostFeaturesDF.drop(columns=["comp_formula"], inplace=True)
   # --- END FIX for Reconstruction ---

   # Clean up temporary file
   try:
       os.remove(data_path)
   except OSError as e:
       print(f"Error removing temporary file {data_path}: {e}")

   return RoostFeaturesDF
