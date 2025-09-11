import os
import sys
import pickle
import re
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from typing import List

# --- GLOBAL CONFIGURATION VARIABLES ---
# Adjust these paths and parameters as needed for your setup

# General settings
DEBUG = False # Set to False to reduce console output from formula search
DIR_PREFIX = 'matbench_perovskites_mattervial_noORB'
# DIR_PREFIX = 'matbench_perovskites_mattervial'
MAX_BACKGROUND_SAMPLES = 300
MAX_SHAP_INSTANCES = 500
NSAMPLES = 300

# Fold settings (for scripts that iterate over folds or use a specific fold)
NUM_FOLDS = 1 # Number of folds to process for multi-fold operations
DEFAULT_FOLD_IDX = 0 # Default fold index for single-fold operations

# Paths for MODNet/MatterVial environment (SHAP calculation, clustering, formula retrieval)
# These paths are for DEFAULT_FOLD_IDX (e.g., fold 0 or fold 1 as per your original scripts)
# If NUM_FOLDS > 1, the script will try to infer paths for other folds based on these patterns.
# MODEL_PATH_TEMPLATE = './results/matbench_perovskites_megnet_mm_megnet_ofm_mvl_all_roost_oqmd_sisso_orb_v3_best_model_fold_{fold_num}.pkl'
# MODDATA_PATH_TEMPLATE = './folds/matbench_perovskites_megnet_mm_megnet_ofm_mvl_all_roost_oqmd_sisso_orb_v3_train_moddata_f{fold_num}'
MODEL_PATH_TEMPLATE = './results_final/MODNet_matbench_perovskites_megnet_mm_megnet_ofm_mvl_all_roost_oqmd_sisso_f{fold_num}_ga.pkl'
MODDATA_PATH_TEMPLATE = './folds_final/matbench_perovskites_megnet_mm_megnet_ofm_mvl_all_roost_oqmd_sisso_train_moddata_f{fold_num}'


# Paths for Dimensionality Reduction environment (after SHAP CSV is generated)
# This is the input CSV for DR plots
SHAP_CSV_FILE_DR = os.path.join(DIR_PREFIX, f"shap_values_fold_{DEFAULT_FOLD_IDX}.csv")
# This is the full featurized data CSV for top features DR mode
FEATURE_CSV_FILE_FULL_DATA = '/gpfs/scratch/acad/htforft/rgouvea/matbench_tests/data/matbench_perovskites/matbench_perovskites_featurizedMM2020Struct_mattervial.csv'
N_TOP_FEATURES_DR = 10 # Number of top features to use for dimensionality reduction plots
CLUSTERING_THRESHOLD_DISTANCE = 2.5 # Distance threshold for clustering
TRANSPARENCY_POWER_EXPONENT = 0.01 # Exponent for transparency in the scatter plot for shap importance of features

# --- Aesthetic Configurations ---
SUPERSCRIPTS = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'}

# Fixed color scheme for feature groups
FEATURE_GROUP_COLORS = {
    'ORB_v3': '#1f77b4',  # Blue
    'SISSO': '#ff7f0e',   # Orange
    'MVL': '#2ca02c',     # Green
    'ℓ-MM': '#d62728', # Red
    'ℓ-OFM': '#9467bd', # Purple
    'ROOST': '#8c564b',   # Brown
    'Other': '#7f7f7f'    # Gray
}

# Mapping for display names of feature classes (for bar plots)
FEATURE_CLASS_DISPLAY_NAMES = {
    'ORB_v3': 'ORB',
    'SISSO': 'SISSO',
    'MVL': 'MVL',
    'MEGNet_MatMinerEncoded': 'ℓ-MM',
    'MEGNet_OFMEncoded': 'ℓ-OFM',
    'ROOST': 'ROOST',
    'Other': 'Other'
}

# --- Conditional Imports ---
try:
   from modnet.models import EnsembleMODNetModel
   from modnet.preprocessing import MODData
   from mattervial.interpreter import Interpreter
   MODNET_MATTERVIAL_AVAILABLE = True
except ImportError:
   print("MODNet or MatterVial not found. SHAP calculation and formula retrieval will be skipped.")
   MODNET_MATTERVIAL_AVAILABLE = False

try:
   from scipy.cluster import hierarchy
   from sklearn.metrics.pairwise import cosine_similarity
   import scipy.spatial.distance
   CLUSTERING_AVAILABLE = True
except ImportError:
   print("Scipy or Scikit-learn not found. Clustering features will be skipped.")
   CLUSTERING_AVAILABLE = False

try:
   import umap
   from sklearn.decomposition import PCA
   from sklearn.manifold import TSNE
   from sklearn.preprocessing import StandardScaler
   from matplotlib.colors import to_rgba # Import here to avoid issues if matplotlib is missing
   from matplotlib.lines import Line2D # Import here
   DIMENSIONALITY_REDUCTION_AVAILABLE = True
except :
   print("UMAP, Scikit-learn, or Matplotlib not found. Dimensionality reduction plots will be skipped.")
   DIMENSIONALITY_REDUCTION_AVAILABLE = False


# --- Helper Functions for String Manipulation (from your scripts) ---

def to_superscript(s: str) -> str:
   return ''.join(SUPERSCRIPTS.get(char, char) for char in s)

def ofm_superscript(formula: str) -> str:
   return re.sub(r'\^([0-9+\-]+)', lambda m: to_superscript(m.group(1)), formula)

def substitute_feature_names_in_formula(formula: str) -> str:
   formula = re.sub(r'MEGNet_OFMEncoded', 'ℓ-OFM', formula)
   formula = re.sub(r'MEGNet_MatMinerEncoded', 'ℓ-MM', formula)
   return formula

def rename_feature_label(feature_name: str) -> str:
   """Converts an original feature name to its final display format for plots."""
   if feature_name.startswith('MEGNet_OFMEncoded_v1_'):
      return re.sub(r'^MEGNet_OFMEncoded_v1_(\d+)$', r'ℓ-OFM_v1_#\1', feature_name)
   if feature_name.startswith('MEGNet_MatMinerEncoded_v1_'):
      return re.sub(r'^MEGNet_MatMinerEncoded_v1_(\d+)$', r'ℓ-MM_v1_#\1', feature_name)
   
   # For other known types, add a # before the final number.
   if feature_name.startswith(('ORB_v3_', 'SISSO_', 'MVL32_', 'MVL16_', 'ROOST_')):
      # This regex finds the last underscore followed by digits and inserts a #
      return re.sub(r'(.*)_(\d+)$', r'\1_#\2', feature_name)

   return feature_name

def get_mattervial_search_key(feature_name: str) -> str:
   """Converts an original feature name to the key expected by MatterVial."""
   # MEGNet features are searched with l-OFM/l-MM prefix (non-cursive)
   if feature_name.startswith('MEGNet_OFMEncoded_'):
      return feature_name.replace('MEGNet_OFMEncoded', 'l-OFM')
   if feature_name.startswith('MEGNet_MatMinerEncoded_'):
      return feature_name.replace('MEGNet_MatMinerEncoded', 'l-MM')
   # Remove # if present in the feature name
   return feature_name.replace('#', '')
   

def shorten_formula(equation: str) -> str:
   equation = equation.strip()
   constant_match = re.match(r"^[+\-]?\s*\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", equation)
   if constant_match:
      matched = constant_match.group(0)
      equation = equation[len(matched):].strip()
      equation = re.sub(r'^\+\s*', '', equation)
   
   first_term, depth, i = "", 0, 0
   while i < len(equation):
      char = equation[i]
      if depth == 0 and i + 3 <= len(equation) and equation[i:i+3] in [" + ", " - "]:
         break
      if char == '(': depth += 1
      elif char == ')': depth -= 1
      first_term += char
      i += 1
   
   return first_term.strip() + " + ..." if i < len(equation) else first_term.strip()

# --- Feature and Formula Analysis Functions (from your scripts) ---

def is_sisso_feature(feature_name: str) -> bool:
   return feature_name.upper().startswith('SISSO')

def find_best_dimension(interp: 'Interpreter', search_key: str, min_r2: float = 0.8):
   if DEBUG: print(f"Searching for best dimension for '{search_key}' (min R²: {min_r2})")
   best_r2, best_dim, best_formula = -float('inf'), None, None
   
   for dim in range(1, 6):
      try:
         formula_data = interp.get_formula(search_key, dimension=dim)
         if "r2" in formula_data:
               r2 = float(formula_data["r2"])
               if DEBUG: print(f"  Dimension {dim}: R² = {r2:.3f}")
               if r2 >= min_r2:
                  if DEBUG: print(f"  → Selected dimension {dim} (R² = {r2:.3f})")
                  return dim, formula_data
               if r2 > best_r2:
                  best_r2, best_dim, best_formula = r2, dim, formula_data
         elif best_formula is None:
               if DEBUG: print(f"  Dimension {dim}: No R² available (likely SISSO)")
               best_dim, best_formula = dim, formula_data
      except Exception as e:
         if DEBUG: print(f"  Dimension {dim}: Error - {e}")
         continue
         
   if best_dim is not None:
      r2_info = f"R² = {best_formula.get('r2', 'N/A')}" if best_formula else "No R²"
      if DEBUG: print(f"  → Using highest R² at dimension {best_dim} ({r2_info})")
      return best_dim, best_formula
      
   if DEBUG: print(f"  → No formula available at any dimension")
   return None, None

def get_formatted_formula(interp: 'Interpreter', original_feature_name: str, mode: str = 'short') -> str:
   """Retrieves and formats the formula for a feature using its original name."""
   if not MODNET_MATTERVIAL_AVAILABLE:
      return "MatterVial not available for formula retrieval."
   try:
      search_key = get_mattervial_search_key(original_feature_name)
      
      if is_sisso_feature(search_key):
         dimension = 1
         formula_data = interp.get_formula(search_key, dimension=dimension)
      else:
         dimension, formula_data = find_best_dimension(interp, search_key)
         if formula_data is None:
               raise Exception("No suitable formula found")

      if "r2" in formula_data:
         formula_text = formula_data["compformula2d"]
         r2 = float(formula_data["r2"])
         if mode == 'short':
               formula_text = shorten_formula(formula_text)
         annotation_text = f"{formula_text}\nR²: {r2:.2f} (D{dimension})"
      else: # SISSO
         formula_text = formula_data["formula"]
         pattern = r'df\[\s*"([^"]+)"\s*\]'
         annotation_text = re.sub(pattern, r'\1 (norm)', formula_text)

      annotation_text = substitute_feature_names_in_formula(annotation_text)
      annotation_text = ofm_superscript(annotation_text)
      return annotation_text
   except Exception as e:
      if DEBUG: print(f"Error retrieving formula for '{original_feature_name}' (searched as '{search_key}'): {e}")
      return "No formula available"

def get_feature_class(feature_name: str) -> str:
   if feature_name.startswith('ORB_v3_'): return 'ORB_v3'
   if feature_name.startswith('SISSO_'): return 'SISSO'
   if feature_name.startswith('MVL32_') or feature_name.startswith('MVL16_'): return 'MVL'
   if feature_name.startswith('MEGNet_MatMinerEncoded_'): return 'MEGNet_MatMinerEncoded'
   if feature_name.startswith('MEGNet_OFMEncoded_'): return 'MEGNet_OFMEncoded'
   if feature_name.startswith('ROOST_'): return 'ROOST'
   return 'Other'

def get_formatted_formula_all_dimensions(interp: 'Interpreter', original_feature_name: str, mode: str = 'short') -> List[str]:  
   """  
   Retrieves and formats formulas for all available dimensions for a feature.  
   Returns a list of formatted formula strings.  
   """  
   if not MODNET_MATTERVIAL_AVAILABLE:
      return ["MatterVial not available for formula retrieval."]

   all_formulas = []  
   search_key = get_mattervial_search_key(original_feature_name)  

   if is_sisso_feature(search_key):  
      try:  
         formula_data = interp.get_formula(search_key, dimension=1)  
         formula_text = formula_data["formula"]  
         pattern = r'df\[\s*"([^"]+)"\s*\]'  
         annotation_text = re.sub(pattern, r'\1 (norm)', formula_text)  
         annotation_text = substitute_feature_names_in_formula(annotation_text)  
         annotation_text = ofm_superscript(annotation_text)  
         all_formulas.append(f"SISSO (D1): {annotation_text}")  
      except Exception as e:  
         if DEBUG: print(f"Error retrieving SISSO formula for '{original_feature_name}': {e}")  
         all_formulas.append("SISSO (D1): No formula available")  
   else:  
      for dim in range(1, 6): # Assuming dimensions 1 to 5 are relevant  
         try:  
               formula_data = interp.get_formula(search_key, dimension=dim)  
               if "r2" in formula_data:  
                  formula_text = formula_data["compformula2d"]  
                  r2 = float(formula_data["r2"])  
                  if mode == 'short':  
                     formula_text = shorten_formula(formula_text)  
                     
                  annotation_text = f"{formula_text}\nR²: {r2:.2f}"  
                  annotation_text = substitute_feature_names_in_formula(annotation_text)  
                  annotation_text = ofm_superscript(annotation_text)  
                  all_formulas.append(f"D{dim}: {annotation_text}")  
               else:  
                  if DEBUG: print(f"  Dimension {dim}: No R² available for '{original_feature_name}'")  
         except Exception as e:  
               if DEBUG: print(f"  Dimension {dim}: Error for '{original_feature_name}' - {e}")  
               continue  
   
   if not all_formulas:  
      all_formulas.append("No formulas available for any dimension")  
         
   return all_formulas

# --- Core Analysis and Plotting Functions (from your scripts) ---

def analyze_feature_importance(shap_values, df_features):
   individual_importance = pd.Series(np.mean(np.abs(shap_values), axis=0), index=df_features.columns)
   feature_to_class = individual_importance.index.map(get_feature_class)
   class_aggregated_importance = individual_importance.groupby(feature_to_class).sum().sort_values(ascending=False)
   
   top_features_per_class = {}
   for class_name in class_aggregated_importance.index:
      features_in_class = individual_importance[feature_to_class == class_name]
      if not features_in_class.empty:
         top_features_per_class[class_name] = features_in_class.idxmax()
         
   return individual_importance, class_aggregated_importance, top_features_per_class

def plot_class_aggregation_barchart(class_importance, fold_idx, dir_prefix, is_loaded):
   plt.figure(figsize=(10, max(6, len(class_importance) * 0.7)))
   
   # Map class names to display names for plotting
   display_class_names = [FEATURE_CLASS_DISPLAY_NAMES.get(cls, cls) for cls in class_importance.index]
   
   # Create a color list based on the fixed color scheme
   colors = [FEATURE_GROUP_COLORS.get(cls, FEATURE_GROUP_COLORS['Other']) for cls in display_class_names]

   sns.barplot(x=class_importance.values, y=display_class_names, palette=colors, orient='h')
   plt.title(f"Aggregated Feature Importance by Class (Fold {fold_idx})")
   plt.xlabel("Sum of Mean Absolute SHAP Values")
   plt.ylabel("Feature Classes")
   plt.yticks(fontsize=14) # Increase font size for y-axis labels
   plt.tight_layout()
   
   suffix = '_loaded' if is_loaded else ''
   plot_filename = os.path.join(dir_prefix, f"feature_importance_bar_class_aggregated_fold_{fold_idx}{suffix}.svg")
   plt.savefig(plot_filename)
   plt.close()
   print(f"Class-aggregated feature importance plot saved as: {plot_filename}")

def plot_beeswarm_with_formulas(shap_values, df_features, top_features, fold_idx, formula_display_mode, dir_prefix, is_loaded):  
   """Generates and saves a SHAP beeswarm plot with optional formula annotations."""  
   if not top_features:  
      print("No top features identified to generate a beeswarm plot.")  
      return  

   print(f"\n--- Generating Beeswarm Plot (Formulas: {formula_display_mode}) for Fold {fold_idx} ---")  
   
   # Filter data for the top features  
   selected_feature_indices = [df_features.columns.get_loc(f) for f in top_features]  
   shap_values_filtered = shap_values[:, selected_feature_indices]  
   df_features_filtered = df_features[top_features]  

   # Plotting setup  
   mpl.rcParams['svg.fonttype'] = 'none'  
   
   # Adjust figure size and font size based on formula_display_mode  
   if formula_display_mode == 'none':  
      fig_width, fig_height, label_fontsize = 10, max(6, len(top_features) * 0.5), 15  
   else:  
      fig_width, fig_height, label_fontsize, annotation_fontsize = 14, max(6, len(top_features) * 0.8), 10, 10  

   plt.figure(figsize=(fig_width, fig_height))  
   
   # SHAP plot determines the order of features based on importance  
   shap.summary_plot(shap_values_filtered, df_features_filtered, show=False, max_display=len(top_features))  
   
   fig = plt.gcf()  
   ax_beeswarm = plt.gca()  
   
   ax_beeswarm.tick_params(axis='y', labelsize=label_fontsize)  
   plt.tight_layout()  
   
   original_labels_from_plot = [lbl.get_text() for lbl in ax_beeswarm.get_yticklabels()]  
   
   # Create the final display names (with # and ℓ) and set them on the plot  
   display_labels = [rename_feature_label(orig_label) for orig_label in original_labels_from_plot]  
   ax_beeswarm.set_yticklabels(display_labels)  

   if formula_display_mode != 'none':  
      bbox = ax_beeswarm.get_position()  
      ann_ax = fig.add_axes([bbox.x1 + 0.02, bbox.y0, 1.0 - (bbox.x1 + 0.04), bbox.height])  
      ann_ax.set_ylim(ax_beeswarm.get_ylim())  
      ann_ax.axis("off")  

      interp = Interpreter()  
         
      # Get the y-positions of the ticks  
      tick_positions = ax_beeswarm.get_yticks()  

      # **FIX**: Iterate using the original feature names (in the correct plot order) for formula lookup  
      for original_feature, pos in zip(original_labels_from_plot, tick_positions):  
         annotation_text = get_formatted_formula(interp, original_feature, mode=formula_display_mode)  
         ann_ax.text(0.05, pos, annotation_text, va='center', ha='left', fontsize=annotation_fontsize, wrap=True)  

   plt.title(f"SHAP Beeswarm Plot for Top Features from Each Class (Fold {fold_idx})")  
   
   suffix = '_loaded' if is_loaded else ''  
   plot_filename = os.path.join(dir_prefix, f"shap_beeswarm_formulas_{formula_display_mode}_fold_{fold_idx}{suffix}.svg")  
   plt.savefig(plot_filename, bbox_inches='tight')  
   plt.close()  
   print(f"Beeswarm plot saved as: {plot_filename}")

def save_feature_formulas_to_text(top_features_list, fold_idx, dir_prefix, is_loaded):
   """
   Saves the top features from each cluster and their formatted formulas to a text file,
   including formulas for all available dimensions.
   """
   print(f"\n--- Saving Feature Formulas to Text File for Fold {fold_idx} ---")
   
   suffix = '_loaded' if is_loaded else ''
   output_filename = os.path.join(dir_prefix, f"top_feature_per_group_formulas_all_dimensions_fold_{fold_idx}{suffix}.txt")
   
   if not MODNET_MATTERVIAL_AVAILABLE:
      print("MatterVial not available. Skipping formula saving.")
      with open(output_filename, 'w', encoding='utf-8') as f:
         f.write("MatterVial not available for formula retrieval. Formulas could not be generated.\n")
      return

   interp = Interpreter() # Initialize Interpreter once
   
   with open(output_filename, 'w', encoding='utf-8') as f:
      f.write(f"--- Top Features from Each Group and Their Formulas (Fold {fold_idx}) ---\n\n")
      
      for feature_name in sorted(top_features_list): # Sort for consistent output
         display_name = rename_feature_label(feature_name)
         
         f.write(f"Feature: {display_name}\n")
         
         full_formulas_all_dims = get_formatted_formula_all_dimensions(interp, feature_name, mode='full')
         f.write(f"  Full Formulas (All Dimensions):\n")
         for formula_text in full_formulas_all_dims:
               f.write(f"    - {formula_text}\n")
         
         f.write("-" * 50 + "\n\n")
         
   print(f"Feature formulas saved to: {output_filename}")

# --- Script 1: Get SHAP and Feature Decomposition ---
def get_shap_and_feature_decomposition(
   model_path: str,
   moddata_path: str,
   dir_prefix: str,
   fold_idx: int
):
   if not MODNET_MATTERVIAL_AVAILABLE:
      print("MODNet or MatterVial not available. Cannot perform SHAP calculation and feature decomposition.")
      return

   os.makedirs(dir_prefix, exist_ok=True)
   
   shap_values_filename = os.path.join(dir_prefix, f"shap_values_fold_{fold_idx}.pkl")
   
   print("Loading models and data...")
   modnet_model = EnsembleMODNetModel.load(model_path)
   moddata = MODData.load(moddata_path)
   is_loaded = False

   if os.path.exists(shap_values_filename):
      print(f"Loading cached SHAP values from {shap_values_filename}")
      with open(shap_values_filename, 'rb') as f:
         shap_values = pickle.load(f)
      is_loaded = True
   else:
      print("Calculating SHAP values...")
      df_features = moddata.df_featurized
      shap_background = shap.kmeans(df_features, min(MAX_BACKGROUND_SAMPLES, len(df_features)))
      df_shap = df_features.sample(n=min(MAX_SHAP_INSTANCES, len(df_features)), random_state=42)
      
      base_model = modnet_model.model[0]
      def predictor(X):
         X_df = pd.DataFrame(data=X, columns=df_features.columns)
         x = X_df.replace([np.inf, -np.inf, np.nan], 0)[base_model.optimal_descriptors[:base_model.n_feat]].values
         x = np.nan_to_num(x)
         if base_model._scaler is not None:
               x = base_model._scaler.transform(x)
               x = np.nan_to_num(x, nan=-1)
         return np.array(base_model.model.predict(x))

      explainer = shap.KernelExplainer(predictor, shap_background)
      shap_values = explainer.shap_values(df_shap, nsamples=NSAMPLES)
      if isinstance(shap_values, list): shap_values = shap_values[0]
      
      with open(shap_values_filename, 'wb') as f:
         pickle.dump(shap_values, f)
      print(f"SHAP values calculated and saved to {shap_values_filename}")

   df_shap_current = moddata.df_featurized.sample(n=min(MAX_SHAP_INSTANCES, len(moddata.df_featurized)), random_state=42)
   
   individual_importance, class_importance, top_features_map = analyze_feature_importance(shap_values, df_shap_current)
   
   print("\nClass-aggregated feature importance:")
   print(class_importance)
   
   top_features_list = list(top_features_map.values())
   print("\nTop feature per class:")
   for feature in top_features_list:
      print(f"  - {feature}")

   plot_class_aggregation_barchart(class_importance, fold_idx, dir_prefix, is_loaded)
   
   save_feature_formulas_to_text(top_features_list, fold_idx, dir_prefix, is_loaded)

   plot_beeswarm_with_formulas(shap_values, df_shap_current, top_features_list, fold_idx, 'none', dir_prefix, is_loaded)
   plot_beeswarm_with_formulas(shap_values, df_shap_current, top_features_list, fold_idx, 'short', dir_prefix, is_loaded)
   plot_beeswarm_with_formulas(shap_values, df_shap_current, top_features_list, fold_idx, 'full', dir_prefix, is_loaded)

# --- Script 2: SHAP-based Feature Clustering ---

def plot_cluster_distance_histogram(linkage_matrix, num_features, fold_idx, dir_prefix, is_loaded, target_clusters=10):
    """
    Plots a histogram of distances at which clusters merge, highlighting the distance
    that yields approximately `target_clusters`.
    """
    print(f"\n--- Generating Cluster Distance Histogram for Fold {fold_idx} ---")
    
    # The distances are in the 3rd column of the linkage matrix
    distances = linkage_matrix[:, 2]
    
    # Sort distances in ascending order
    sorted_distances = np.sort(distances)
    
    # Calculate the number of clusters at each merge distance
    # Starting with num_features clusters, each merge reduces the count by 1
    num_clusters_at_distance = np.arange(num_features, 0, -1)[:len(sorted_distances)]

    plt.figure(figsize=(12, 7))
    plt.plot(sorted_distances, num_clusters_at_distance, marker='o', linestyle='-', color='blue', markersize=4)
    
    plt.title(f'Number of Clusters vs. Merge Distance (Fold {fold_idx})')
    plt.xlabel('Distance (1 - Cosine Similarity)')
    plt.ylabel('Number of Clusters')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight the distance for target_clusters
    if target_clusters > 0 and target_clusters <= num_features:
        # Find the distance that results in approximately target_clusters
        # We want the largest distance such that num_clusters_at_distance >= target_clusters
        # Or, more precisely, the distance where the number of clusters drops to target_clusters or just below.
        
        # Find index where num_clusters_at_distance first drops to or below target_clusters
        idx_target = np.where(num_clusters_at_distance <= target_clusters)[0]
        
        if len(idx_target) > 0:
            # Use the distance at this index
            highlight_distance = sorted_distances[idx_target[0]]
            
            # Plot a vertical line at this distance
            plt.axvline(x=highlight_distance, color='red', linestyle='--', linewidth=2, 
                        label=f'Approx. {target_clusters} Clusters at Distance {highlight_distance:.2f}')
            plt.legend()
            
            print(f"Highlighted distance for approx. {target_clusters} clusters: {highlight_distance:.2f}")
        else:
            print(f"Could not find a merge distance yielding exactly {target_clusters} clusters. Showing all merges.")
    
    plt.tight_layout()
    suffix = '_loaded' if is_loaded else ''
    plot_filename = os.path.join(dir_prefix, f"cluster_distance_histogram_fold_{fold_idx}{suffix}.svg")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Cluster distance histogram saved as: {plot_filename}")


def cluster_features_by_shap(shap_values, df_features, fold_idx, dir_prefix, is_loaded):
   """
   Performs hierarchical clustering on features based on their SHAP value patterns.
   Returns a DataFrame mapping features to cluster IDs.
   """
   if not CLUSTERING_AVAILABLE:
      print("Scipy or Scikit-learn not available. Skipping feature clustering.")
      return pd.DataFrame({'feature': df_features.columns, 'cluster_id': 1})

   print(f"\n--- Clustering features based on SHAP value patterns for Fold {fold_idx} ---")

   # Ensure SHAP values are finite and handle features with zero variance
   shap_values_T = shap_values.T

   feature_variances = np.var(shap_values_T, axis=1)
   non_constant_features_mask = feature_variances > 1e-9 

   if not np.any(non_constant_features_mask):
      print("Warning: All features have constant SHAP values. Cannot perform clustering.")
      return pd.DataFrame({'feature': df_features.columns, 'cluster_id': 1})

   shap_values_for_clustering = np.nan_to_num(shap_values_T[non_constant_features_mask], nan=0.0, posinf=0.0, neginf=0.0)
   features_for_clustering = df_features.columns[non_constant_features_mask]

   condensed_dist = scipy.spatial.distance.pdist(shap_values_for_clustering, metric='cosine')

   linkage_matrix = hierarchy.linkage(condensed_dist, method='ward')

   # Plot the cluster distance histogram
   plot_cluster_distance_histogram(linkage_matrix, len(features_for_clustering), fold_idx, dir_prefix, is_loaded, target_clusters=10)

   plt.figure(figsize=(20, max(12, len(features_for_clustering) * 0.2)))
   plt.title(f'Hierarchical Clustering Dendrogram of Features (Fold {fold_idx})')
   plt.xlabel('Distance (1 - Cosine Similarity)')
   plt.ylabel('Features')
   
   renamed_labels = [rename_feature_label(f) for f in features_for_clustering]

   hierarchy.dendrogram(
      linkage_matrix,
      labels=renamed_labels,
      orientation='left',
      leaf_font_size=10
   )
   plt.tight_layout()
   suffix = '_loaded' if is_loaded else ''
   plot_filename = os.path.join(dir_prefix, f"feature_cluster_dendrogram_fold_{fold_idx}{suffix}.svg")
   plt.savefig(plot_filename)
   plt.close()
   print(f"Feature clustering dendrogram saved as: {plot_filename}")

   distance_threshold = CLUSTERING_THRESHOLD_DISTANCE 
   clusters = hierarchy.fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
   
   df_clusters_result = pd.DataFrame({'feature': features_for_clustering, 'cluster_id': clusters})
   df_clusters_result = df_clusters_result.sort_values(by=['cluster_id', 'feature'])

   print(f"\nFeature clusters based on SHAP similarity (distance < {distance_threshold}):")
   for cluster_id in sorted(df_clusters_result['cluster_id'].unique()):
      print(f"\n--- Cluster {cluster_id} ---")
      features_in_cluster = df_clusters_result[df_clusters_result['cluster_id'] == cluster_id]['feature'].tolist()
      for feature in features_in_cluster:
         print(f"  - {feature}")
         
   excluded_features = df_features.columns[~non_constant_features_mask]
   if len(excluded_features) > 0:
      max_cluster_id = df_clusters_result['cluster_id'].max() if not df_clusters_result.empty else 0
      constant_cluster_id = max_cluster_id + 1
      df_excluded_clusters = pd.DataFrame({'feature': excluded_features, 'cluster_id': constant_cluster_id})
      df_clusters_result = pd.concat([df_clusters_result, df_excluded_clusters], ignore_index=True)
      print(f"\n--- Cluster {constant_cluster_id} (Constant SHAP Values) ---")
      for feature in excluded_features:
         print(f"  - {feature}")

   return df_clusters_result

def get_top_feature_per_cluster(df_clusters, individual_importance):
   """Identifies the most important feature within each cluster."""
   df_importance = individual_importance.to_frame(name='importance')
   df_merged = df_clusters.merge(df_importance, left_on='feature', right_index=True)
   
   top_features_idx = df_merged.groupby('cluster_id')['importance'].idxmax()
   top_features = df_merged.loc[top_features_idx]['feature'].tolist()
   
   return top_features

def plot_cluster_aggregation_barchart(df_clusters, individual_importance, fold_idx, dir_prefix, is_loaded):
    """
    Plots a bar chart of aggregated SHAP importance for each feature cluster,
    with descriptive labels including the top feature of each cluster.
    """
    print("\n--- Generating Cluster-Aggregated Importance Bar Plot ---")
    df_importance = individual_importance.to_frame(name='importance')
    df_merged = df_clusters.merge(df_importance, left_on='feature', right_index=True)
    
    # Calculate aggregated importance per cluster
    cluster_importance = df_merged.groupby('cluster_id')['importance'].sum()
    
    # Get the top feature for each cluster and its importance
    top_features_per_cluster_df = df_merged.loc[df_merged.groupby('cluster_id')['importance'].idxmax()]
    top_features_per_cluster_df = top_features_per_cluster_df.set_index('cluster_id')
    
    # Combine aggregated importance with top feature info
    plot_data = cluster_importance.to_frame(name='aggregated_importance')
    plot_data = plot_data.merge(top_features_per_cluster_df[['feature', 'importance']], 
                                left_index=True, right_index=True, how='left')
    plot_data = plot_data.sort_values(by='aggregated_importance', ascending=False)
    
    # Create descriptive labels for the y-axis
    y_labels = []
    for cluster_id, row in plot_data.iterrows():
        top_feat_display = rename_feature_label(row['feature'])
        y_labels.append(f"Cluster {cluster_id}: {top_feat_display}\n(Top Feature Importance: {row['importance']:.2f})")

    plt.figure(figsize=(12, max(8, len(plot_data) * 0.8))) # Adjust figure size dynamically
    sns.barplot(x=plot_data['aggregated_importance'].values, y=y_labels, palette='viridis', orient='h')
    
    plt.title(f"Aggregated Feature Importance by SHAP-based Cluster (Fold {fold_idx})")
    plt.xlabel("Sum of Mean Absolute SHAP Values")
    plt.ylabel("Feature Cluster")
    plt.yticks(fontsize=10) # Adjust font size for potentially longer labels
    plt.tight_layout()
    
    suffix = '_loaded' if is_loaded else ''
    plot_filename = os.path.join(dir_prefix, f"feature_importance_bar_cluster_aggregated_fold_{fold_idx}{suffix}.svg")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Cluster-aggregated feature importance plot saved as: {plot_filename}")

def plot_beeswarm_with_formulas_cluster(shap_values, df_features, top_features, fold_idx, formula_display_mode, dir_prefix, is_loaded):  
   """Generates and saves a SHAP beeswarm plot for the top feature of each cluster."""  
   if not top_features:  
      print("No top features identified to generate a beeswarm plot.")  
      return  

   print(f"\n--- Generating Beeswarm Plot for Top Features per Cluster (Formulas: {formula_display_mode}) for Fold {fold_idx} ---")  
   
   selected_feature_indices = [df_features.columns.get_loc(f) for f in top_features]  
   shap_values_filtered = shap_values[:, selected_feature_indices]  
   df_features_filtered = df_features[top_features]  

   mpl.rcParams['svg.fonttype'] = 'none'  
   
   if formula_display_mode == 'none':  
      fig_width, fig_height, label_fontsize = 10, max(6, len(top_features) * 0.5), 15  
   else:  
      fig_width, fig_height, label_fontsize, annotation_fontsize = 14, max(6, len(top_features) * 0.8), 10, 10  

   plt.figure(figsize=(fig_width, fig_height))  
   
   shap.summary_plot(shap_values_filtered, df_features_filtered, show=False, max_display=len(top_features))  
   
   fig = plt.gcf()  
   ax_beeswarm = plt.gca()  
   
   ax_beeswarm.tick_params(axis='y', labelsize=label_fontsize)  
   plt.tight_layout()  
   
   original_labels_from_plot = [lbl.get_text() for lbl in ax_beeswarm.get_yticklabels()]  
   
   display_labels = [rename_feature_label(orig_label) for orig_label in original_labels_from_plot]  
   ax_beeswarm.set_yticklabels(display_labels)  

   if formula_display_mode != 'none':  
      bbox = ax_beeswarm.get_position()  
      ann_ax = fig.add_axes([bbox.x1 + 0.02, bbox.y0, 1.0 - (bbox.x1 + 0.04), bbox.height])  
      ann_ax.set_ylim(ax_beeswarm.get_ylim())  
      ann_ax.axis("off")  

      interp = Interpreter()  
      tick_positions = ax_beeswarm.get_yticks()  

      for original_feature, pos in zip(original_labels_from_plot, tick_positions):  
         annotation_text = get_formatted_formula(interp, original_feature, mode=formula_display_mode)  
         ann_ax.text(0.05, pos, annotation_text, va='center', ha='left', fontsize=annotation_fontsize, wrap=True)  

   plt.title(f"SHAP Beeswarm for Top Features from Each Cluster (Fold {fold_idx})")  
   
   suffix = '_loaded' if is_loaded else ''  
   plot_filename = os.path.join(dir_prefix, f"shap_beeswarm_cluster_formulas_{formula_display_mode}_fold_{fold_idx}{suffix}.svg")  
   plt.savefig(plot_filename, bbox_inches='tight')  
   plt.close()  
   print(f"Beeswarm plot saved as: {plot_filename}")

def save_feature_formulas_to_text_cluster(top_features_list, fold_idx, dir_prefix, is_loaded):
   """
   Saves the top features from each cluster and their formatted formulas to a text file.
   """
   print(f"\n--- Saving Feature Formulas to Text File for Fold {fold_idx} ---")
   
   suffix = '_loaded' if is_loaded else ''
   output_filename = os.path.join(dir_prefix, f"top_cluster_feature_formulas_fold_{fold_idx}{suffix}.txt")
   
   if not MODNET_MATTERVIAL_AVAILABLE:
      print("MatterVial not available. Skipping formula saving.")
      with open(output_filename, 'w', encoding='utf-8') as f:
         f.write("MatterVial not available for formula retrieval. Formulas could not be generated.\n")
      return

   interp = Interpreter() # Initialize Interpreter once
   
   with open(output_filename, 'w', encoding='utf-8') as f:
      f.write(f"--- Top Features from Each Cluster and Their Formulas (Fold {fold_idx}) ---\n\n")
      
      for feature_name in sorted(top_features_list): # Sort for consistent output
         display_name = rename_feature_label(feature_name)
         full_formula = get_formatted_formula(interp, feature_name, mode='full')
         
         f.write(f"Feature: {display_name}\n")
         f.write(f"  Full Formula: {full_formula}\n")
         f.write("-" * 50 + "\n\n")
         
   print(f"Feature formulas saved to: {output_filename}")

def get_shap_and_feature_clustering(
   model_paths: List[str],
   moddata_paths: List[str],
   dir_prefix: str,
   num_folds: int
):
   if not MODNET_MATTERVIAL_AVAILABLE:
      print("MODNet or MatterVial not available. Cannot perform SHAP calculation and feature clustering.")
      return
   if not CLUSTERING_AVAILABLE:
      print("Scipy or Scikit-learn not available. Cannot perform feature clustering.")
      return

   os.makedirs(dir_prefix, exist_ok=True)

   print("Loading models and data...")
   modnet_models = [EnsembleMODNetModel.load(p) for p in model_paths]
   train_mds = [MODData.load(p) for p in moddata_paths]
   print(f"{len(modnet_models)} models and {len(train_mds)} MODData objects loaded.")

   for i in range(num_folds):
      print(f"\n{'='*10} Processing Fold {i} {'='*10}")
      shap_values_filename = os.path.join(dir_prefix, f"shap_values_fold_{i}.pkl")
      moddata = train_mds[i]
      is_loaded = False

      if os.path.exists(shap_values_filename):
         print(f"Loading cached SHAP values from {shap_values_filename}")
         with open(shap_values_filename, 'rb') as f:
               shap_values = pickle.load(f)
         is_loaded = True
      else:
         print("Calculating SHAP values...")
         df_features = moddata.df_featurized
         shap_background = shap.kmeans(df_features, min(MAX_BACKGROUND_SAMPLES, len(df_features)))
         df_shap = df_features.sample(n=min(MAX_SHAP_INSTANCES, len(df_features)), random_state=42)
         
         base_model = modnet_models[i].model[0]
         def predictor(X):
               X_df = pd.DataFrame(data=X, columns=df_features.columns)
               x = X_df.replace([np.inf, -np.inf, np.nan], 0)[base_model.optimal_descriptors[:base_model.n_feat]].values
               x = np.nan_to_num(x)
               if base_model._scaler is not None:
                  x = base_model._scaler.transform(x)
                  x = np.nan_to_num(x, nan=-1)
               return np.array(base_model.model.predict(x))

         explainer = shap.KernelExplainer(predictor, shap_background)
         shap_values = explainer.shap_values(df_shap, nsamples=NSAMPLES)
         if isinstance(shap_values, list): shap_values = shap_values[0]
         
         with open(shap_values_filename, 'wb') as f:
               pickle.dump(shap_values, f)
         print(f"SHAP values calculated and saved to {shap_values_filename}")

      df_shap_current = moddata.df_featurized.sample(n=min(MAX_SHAP_INSTANCES, len(moddata.df_featurized)), random_state=42)
      
      # --- NEW ANALYSIS FLOW ---

      # 1. Perform feature clustering based on SHAP patterns
      df_clusters = cluster_features_by_shap(shap_values, df_shap_current, i, dir_prefix, is_loaded)

      # 2. Calculate individual feature importance
      individual_importance_series, class_aggregated_importance_series, _ = analyze_feature_importance(shap_values, df_shap_current)
      
      plot_cluster_aggregation_barchart(df_clusters, individual_importance_series, i, dir_prefix, is_loaded) 
  
      # 4. Identify the top feature from each cluster  
      top_features_from_clusters = get_top_feature_per_cluster(df_clusters, individual_importance_series) 
      print("\nTop feature per cluster:")
      for feature in sorted(top_features_from_clusters):
         print(f"  - {feature}")

      # 5. Save feature formulas to a text file
      save_feature_formulas_to_text_cluster(top_features_from_clusters, i, dir_prefix, is_loaded)

      # 6. Generate beeswarm plots using only the top feature from each cluster
      plot_beeswarm_with_formulas_cluster(shap_values, df_shap_current, top_features_from_clusters, i, 'none', dir_prefix, is_loaded)
      plot_beeswarm_with_formulas_cluster(shap_values, df_shap_current, top_features_from_clusters, i, 'short', dir_prefix, is_loaded)
      plot_beeswarm_with_formulas_cluster(shap_values, df_shap_current, top_features_from_clusters, i, 'full', dir_prefix, is_loaded)

# --- Script 3: Create SHAP CSV ---
def create_shap_csv(num_folds: int, dir_prefix: str):
   """
   Loads SHAP values and feature names, then saves them to a CSV file for each fold.
   """
   os.makedirs(dir_prefix, exist_ok=True)

   for i in range(num_folds):
      print(f"\n{'='*10} Processing Fold {i} {'='*10}")
      shap_values_filename = os.path.join(dir_prefix, f"shap_values_fold_{i}.pkl")
      
      if not MODNET_MATTERVIAL_AVAILABLE:
         print("MODNet not available. Cannot load MODData to get feature names for CSV creation.")
         print("Please ensure modnet is installed or provide a way to get feature names.")
         return

      try:
         # Use the MODDATA_PATH_TEMPLATE for consistency, assuming it can be adapted for each fold
         moddata_path_for_fold = MODDATA_PATH_TEMPLATE.format(fold_num=i+1) # Assuming fold_num starts from 1 for MODData paths
         train_md = MODData.load(moddata_path_for_fold)
         df_features = train_md.df_featurized
      except FileNotFoundError:
         print(f"Warning: MODData for fold {i} not found at '{moddata_path_for_fold}'. Cannot retrieve feature names without it.")
         print("Please ensure the MODData files are in './folds/' or provide a way to get feature names.")
         return
      except Exception as e:
         print(f"An unexpected error occurred while loading MODData: {e}")
         return


      if os.path.exists(shap_values_filename):
         print(f"Loading cached SHAP values from {shap_values_filename}")
         with open(shap_values_filename, 'rb') as f:
               shap_values = pickle.load(f)
         
         if isinstance(shap_values, list):
               shap_values = shap_values[0] # Take the first element if it's a list (common for multi-output models)

         # Create a DataFrame from SHAP values
         # Ensure the number of columns in shap_values matches the number of features
         if shap_values.shape[1] == len(df_features.columns):
               shap_df = pd.DataFrame(data=shap_values, columns=df_features.columns)
               output_csv_filename = os.path.join(dir_prefix, f"shap_values_fold_{i}.csv")
               shap_df.to_csv(output_csv_filename, index=False)
               print(f"SHAP values and feature names saved to {output_csv_filename}")
         else:
               print(f"Error: Mismatch between SHAP values columns ({shap_values.shape[1]}) and feature names ({len(df_features.columns)}) for fold {i}.")
               print("Cannot create CSV. Please check the SHAP value calculation or feature data.")

      else:
         print(f"SHAP values file not found for fold {i} at {shap_values_filename}. Cannot create CSV.")
         print("Please run the original script to calculate SHAP values first.")

# --- Script 4: Dimensionality Reduction Plots (Full Features) ---

def substitute_megnet_feature_names_dr(feature_name: str) -> str:
   """
   Substitutes MEGNet feature prefixes with their special characters.
   This is used for individual feature names (e.g., in annotations).
   """
   name = re.sub(r'MEGNet_OFMEncoded', 'ℓ-OFM', feature_name)
   name = re.sub(r'MEGNet_MatMinerEncoded', 'ℓ-MM', name)
   return name

def get_feature_group_display_name_dr(feature_name: str) -> str:
   """
   Determines the feature group based on the feature name and returns its
   display name (with substitutions for MEGNet groups).
   This is used for legend labels.
   """
   if feature_name.startswith('ORB_'): return 'ORB'
   if feature_name.startswith('SISSO_'): return 'SISSO'
   if feature_name.startswith('MVL'): return 'MVL'
   if feature_name.startswith('ROOST_'): return 'ROOST'
   if feature_name.startswith('MEGNet_MatMinerEncoded'): return 'ℓ-MM'
   if feature_name.startswith('MEGNet_OFMEncoded'): return 'ℓ-OFM'
   return 'Other'

def plot_dimensionality_reduction_full(csv_filepath, method='UMAP', output_dir='matbench_perovskites_mattervial'):
   """
   Loads SHAP values from a CSV, performs specified dimensionality reduction,
   and generates a 2D plot colored by feature group and with transparency
   tuned by mean absolute SHAP value. Outputs both PNG and SVG.

   Args:
      csv_filepath (str): The path to the input CSV file containing SHAP values.
      method (str): The dimensionality reduction method to use ('UMAP', 'PCA', 'TSNE').
      output_dir (str): Directory to save the plots.
   """
   if not DIMENSIONALITY_REDUCTION_AVAILABLE:
      print(f"Dimensionality reduction libraries not available. Skipping {method} plot.")
      return

   print(f"Loading data from {csv_filepath} for {method}...")
   try:
      df_shap = pd.read_csv(csv_filepath)
      print(f"Successfully loaded {len(df_shap)} rows and {len(df_shap.columns)} columns.")
   except FileNotFoundError:
      print(f"Error: The file '{csv_filepath}' was not found. Please ensure the path is correct.")
      return
   except Exception as e:
      print(f"An error occurred while reading the CSV file: {e}")
      return

   # Calculate Mean Absolute SHAP values for each feature (column)
   mean_abs_shap_values = df_shap.abs().mean()

   # --- Alpha Scaling Adjustment ---
   transformed_shap = np.power(mean_abs_shap_values + 1e-9, TRANSPARENCY_POWER_EXPONENT) 

   min_transformed_shap = transformed_shap.min()
   max_transformed_shap = transformed_shap.max()

   if max_transformed_shap == min_transformed_shap:
      normalized_alpha = pd.Series(0.7, index=mean_abs_shap_values.index) # Default alpha if no variation
   else:
      normalized_alpha = (transformed_shap - min_transformed_shap) / (max_transformed_shap - min_transformed_shap)
   
   min_alpha_display = 0.2 # Minimum transparency (0.0 is fully transparent, 1.0 is fully opaque)
   normalized_alpha = normalized_alpha * (1 - min_alpha_display) + min_alpha_display
   # --- End Alpha Scaling Adjustment ---


   # Extract original feature names (column names)
   original_feature_names = df_shap.columns.tolist()
   
   # Apply substitutions to feature names for display (for annotations)
   display_feature_names = [substitute_megnet_feature_names_dr(name) for name in original_feature_names]

   # Get feature groups for each original column name (for coloring and legend)
   feature_groups = [get_feature_group_display_name_dr(name) for name in original_feature_names]

   # Select only numeric data for dimensionality reduction
   numeric_df_shap = df_shap.select_dtypes(include=np.number)
   if numeric_df_shap.empty:
      print("Error: No numeric columns found in the CSV for dimensionality reduction. Please check your data.")
      return

   # Transpose the SHAP values so features are rows and samples are columns
   df_shap_transposed = numeric_df_shap.T
   
   # Apply DR to the transposed data
   embedding = None
   if method == 'UMAP':
      print("Applying UMAP dimensionality reduction...")
      reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric='euclidean', random_state=42)
      embedding = reducer.fit_transform(df_shap_transposed)
   elif method == 'PCA':
      print("Applying PCA dimensionality reduction...")
      reducer = PCA(n_components=2, random_state=42)
      embedding = reducer.fit_transform(df_shap_transposed)
   elif method == 'TSNE':
      print("Applying t-SNE dimensionality reduction...")
      perplexity_val = min(30, len(df_shap_transposed) - 1)
      if perplexity_val <= 1:
         print("Warning: Not enough features for t-SNE with perplexity > 1. Skipping t-SNE.")
         return
      reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=1000)
      embedding = reducer.fit_transform(df_shap_transposed)
   else:
      print(f"Error: Unknown dimensionality reduction method '{method}'. Choose from 'UMAP', 'PCA', 'TSNE'.")
      return

   print(f"{method} reduction complete.")

   # Create a DataFrame for the embedding, including feature groups, display names, and normalized alpha
   df_feature_embedding = pd.DataFrame({
      f'{method}_1': embedding[:, 0],
      f'{method}_2': embedding[:, 1],
      'Feature_Group': feature_groups, # This now holds the substituted group names
      'Feature_Name_Display': display_feature_names, # This holds the substituted individual feature names
      'Mean_Abs_SHAP': mean_abs_shap_values.values,
      'Alpha_Value': normalized_alpha.values
   })

   # Plotting
   print(f"Generating {method} plot...")
   plt.figure(figsize=(14, 12))
   
   # Get unique feature groups for legend (these will now be the substituted names)
   unique_groups = df_feature_embedding['Feature_Group'].unique()
   
   # Use the fixed color scheme
   color_map = {group: FEATURE_GROUP_COLORS.get(group, FEATURE_GROUP_COLORS['Other']) for group in unique_groups}
   
   # Create a scatter plot for each feature group
   for group in unique_groups:
      group_data = df_feature_embedding[df_feature_embedding['Feature_Group'] == group]
      
      for idx, row in group_data.iterrows():
         base_color = color_map[group]
         rgba_color = to_rgba(base_color, alpha=row['Alpha_Value'])
         
         plt.scatter(
               row[f'{method}_1'], 
               row[f'{method}_2'], 
               color=rgba_color, 
               s=100,  # Size of the point
               edgecolors='none'  # No edge color for cleaner look
         )
   
   # Add a legend for the feature groups
   legend_elements = [Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=color_map[group], markersize=10, 
                           label=group) for group in unique_groups]
   
   plt.legend(handles=legend_elements, title='Feature Group', 
            bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
            fontsize='x-large', markerscale=2, title_fontsize='x-large')
   
   plt.title(f'{method} Projection of SHAP Feature Importances\n(Colored by Group, Transparency by Mean Abs SHAP)')
   plt.xlabel(f'{method}_1')
   plt.ylabel(f'{method}_2')
   plt.grid(True, linestyle='--', alpha=0.6)
   plt.tight_layout()

   # Save the plot as PNG
   os.makedirs(output_dir, exist_ok=True)
   png_plot_filename = os.path.join(output_dir, f"{method.lower()}_plot_shap_features_alpha.png")
   plt.savefig(png_plot_filename, bbox_inches='tight')
   plt.close()
   print(f"Plot saved to: {png_plot_filename}")

   # Save the plot as SVG (with editable fonts)
   svg_plot_filename = os.path.join(output_dir, f"{method.lower()}_plot_shap_features_alpha.svg")
   plt.figure(figsize=(14, 12)) # Recreate figure for SVG
   # Re-plot all points for the SVG version
   for group in unique_groups:
      group_data = df_feature_embedding[df_feature_embedding['Feature_Group'] == group]
      for idx, row in group_data.iterrows():
         base_color = color_map[group]
         rgba_color = to_rgba(base_color, alpha=row['Alpha_Value'])
         plt.scatter(
               row[f'{method}_1'], 
               row[f'{method}_2'], 
               color=rgba_color, 
               s=100,
               edgecolors='none'
         )
   plt.legend(handles=legend_elements, title='Feature Group', 
            bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
            fontsize='x-large', markerscale=2, title_fontsize='x-large')
   plt.title(f'{method} Projection of SHAP Feature Importances\n(Colored by Group, Transparency by Mean Abs SHAP)')
   plt.xlabel(f'{method}_1')
   plt.ylabel(f'{method}_2')
   plt.grid(True, linestyle='--', alpha=0.6)
   plt.tight_layout()
   plt.savefig(svg_plot_filename, bbox_inches='tight')
   plt.close()
   print(f"Plot saved to: {svg_plot_filename}")


   # Create an additional plot with annotations for top features (PNG and SVG)
   plt.figure(figsize=(14, 12))
   
   # Plot all points first (same as before)
   for group in unique_groups:
      group_data = df_feature_embedding[df_feature_embedding['Feature_Group'] == group]
      for idx, row in group_data.iterrows():
         base_color = color_map[group]
         rgba_color = to_rgba(base_color, alpha=row['Alpha_Value'])
         plt.scatter(
               row[f'{method}_1'], 
               row[f'{method}_2'], 
               color=rgba_color, 
               s=100,
               edgecolors='none'
         )
   
   # Annotate top N features by SHAP value
   top_n = N_TOP_FEATURES_DR  # Number of top features to annotate
   top_features = df_feature_embedding.nlargest(top_n, 'Mean_Abs_SHAP')
   
   for idx, row in top_features.iterrows():
      plt.annotate(
         row['Feature_Name_Display'], # This uses the already substituted name
         (row[f'{method}_1'], row[f'{method}_2']),
         xytext=(5, 5),  # Small offset from the point
         textcoords='offset points',
         fontsize=12,  # Increased font size for better visibility
         fontweight='bold',  # Bold text for emphasis
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
      )
   
   # Add the legend
   plt.legend(handles=legend_elements, title='Feature Group', 
            bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
            fontsize='x-large', markerscale=2, title_fontsize='x-large')
   
   plt.title(f'{method} Projection with Top {top_n} Features Annotated\n(Colored by Group, Transparency by Mean Abs SHAP)')
   plt.xlabel(f'{method}_1')
   plt.ylabel(f'{method}_2')
   plt.grid(True, linestyle='--', alpha=0.6)
   plt.tight_layout()
   
   # Save the annotated plot as PNG
   annotated_png_filename = os.path.join(output_dir, f"{method.lower()}_plot_shap_features_annotated.png")
   plt.savefig(annotated_png_filename, bbox_inches='tight')
   plt.close()
   print(f"Annotated plot saved to: {annotated_png_filename}")

   # Save the annotated plot as SVG
   annotated_svg_filename = os.path.join(output_dir, f"{method.lower()}_plot_shap_features_annotated.svg")
   plt.figure(figsize=(14, 12)) # Recreate figure for SVG
   # Re-plot all points for the SVG version
   for group in unique_groups:
      group_data = df_feature_embedding[df_feature_embedding['Feature_Group'] == group]
      for idx, row in group_data.iterrows():
         base_color = color_map[group]
         rgba_color = to_rgba(base_color, alpha=row['Alpha_Value'])
         plt.scatter(
               row[f'{method}_1'], 
               row[f'{method}_2'], 
               color=rgba_color, 
               s=100,
               edgecolors='none'
         )
   # Re-annotate top N features for SVG
   for idx, row in top_features.iterrows():
      plt.annotate(
         row['Feature_Name_Display'],
         (row[f'{method}_1'], row[f'{method}_2']),
         xytext=(5, 5),
         textcoords='offset points',
         fontsize=12,  # Increased font size for better visibility
         fontweight='bold',  # Bold text for emphasis
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
      )
   plt.legend(handles=legend_elements, title='Feature Group', 
            bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
            fontsize='x-large', markerscale=2, title_fontsize='x-large')
   plt.title(f'{method} Projection with Top {top_n} Features Annotated\n(Colored by Group, Transparency by Mean Abs SHAP)')
   plt.xlabel(f'{method}_1')
   plt.ylabel(f'{method}_2')
   plt.grid(True, linestyle='--', alpha=0.6)
   plt.tight_layout()
   plt.savefig(annotated_svg_filename, bbox_inches='tight')
   plt.close()
   print(f"Annotated plot saved to: {annotated_svg_filename}")

def dimensionality_reduction_plots_full(csv_file: str, dir_prefix: str):
   mpl.rcParams['svg.fonttype'] = 'none' # Ensure this is set for SVG output
   plot_dimensionality_reduction_full(csv_file, method='UMAP', output_dir=dir_prefix)
   plot_dimensionality_reduction_full(csv_file, method='PCA', output_dir=dir_prefix)
   plot_dimensionality_reduction_full(csv_file, method='TSNE', output_dir=dir_prefix)

# --- Script 5: Dimensionality Reduction Plots (Top Features) ---

def load_shap_values_csv_top(csv_file_path):
   """Load SHAP values from CSV file."""
   print(f"Loading SHAP values from {csv_file_path}")
   try:
      df_shap = pd.read_csv(csv_file_path)
      print(f"Successfully loaded SHAP values with {len(df_shap)} rows and {len(df_shap.columns)} columns.")
      return df_shap
   except Exception as e:
      print(f"Error loading SHAP values: {e}")
      raise

def load_feature_data_top(csv_file_path):
   """Load feature and target data from CSV file."""
   print(f"Loading data from {csv_file_path}")
   try:
      df = pd.read_csv(csv_file_path)
      
      # Assuming the target column is named 'target' or is the first column
      # Adjust this logic based on your actual CSV structure
      if 'target' in df.columns:
         target_col = 'target'
      else:
         # Assume first column is the target
         target_col = df.columns[0]
      
      print(f"Using '{target_col}' as target column")
      
      # Separate features and target
      y = df[target_col]
      X = df.drop(columns=[target_col])
      
      print(f"Loaded {X.shape[1]} features and {X.shape[0]} samples")
      return X, y, target_col
   except Exception as e:
      print(f"Error loading feature data: {e}")
      raise

def get_top_features_from_shap_csv_top(df_shap, n=20):
   """Get the top n most important features based on mean absolute SHAP values from CSV."""
   # Calculate mean absolute SHAP values for each feature (column)
   mean_abs_shap = df_shap.abs().mean()
   
   # Create a DataFrame with feature names and importance
   feature_importance = pd.DataFrame({
      'Feature': mean_abs_shap.index,
      'Importance': mean_abs_shap.values
   })
   
   # Sort by importance and get top n
   top_features = feature_importance.sort_values('Importance', ascending=False).head(n)
   print(f"Top {n} features by SHAP importance:")
   for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
      print(f"{i+1}. {feature}: {importance:.6f}")
   
   return top_features['Feature'].tolist()

def plot_dimensionality_reduction_top(X, y, method='UMAP', output_dir='matbench_perovskites_mattervial', fold=0):
   """
   Create dimensionality reduction plot (UMAP or t-SNE) colored by target value.
   
   Args:
      X (DataFrame): Feature matrix with only the top features
      y (Series): Target values
      method (str): 'UMAP' or 'TSNE'
      output_dir (str): Directory to save plots
      fold (int): Fold number for filename
   """
   if not DIMENSIONALITY_REDUCTION_AVAILABLE:
      print(f"Dimensionality reduction libraries not available. Skipping {method} plot for top features.")
      return

   print(f"Generating {method} plot...")
   
   # Scale the features
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   
   # Apply dimensionality reduction
   if method == 'UMAP':
      reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
      embedding = reducer.fit_transform(X_scaled)
   elif method == 'TSNE':
      reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1), n_iter=1000)
      embedding = reducer.fit_transform(X_scaled)
   else:
      raise ValueError(f"Unknown method: {method}")
   
   # Create DataFrame for plotting
   df_plot = pd.DataFrame({
      'x': embedding[:, 0],
      'y': embedding[:, 1],
      'target': y
   })
   
   # Create plot
   plt.figure(figsize=(10, 8))
   
   # Create a scatter plot with points colored by target value
   scatter = plt.scatter(
      df_plot['x'], 
      df_plot['y'],
      c=df_plot['target'],
      cmap='coolwarm',  # Purple to red colormap similar to the example
      s=50,  # Point size
      alpha=0.8,  # Slight transparency
      edgecolors='none'
   )
   
   # Add colorbar
   cbar = plt.colorbar(scatter)
   cbar.set_label('Target Value')
   
   # Set labels and title
   plt.title(f'{method} of Top {len(X.columns)} Features Colored by Target Value')
   plt.xlabel(f'{method.lower()} dimension 1')
   plt.ylabel(f'{method.lower()} dimension 2')
   
   # Save plots
   os.makedirs(output_dir, exist_ok=True)
   
   # Save as PNG
   png_filename = os.path.join(output_dir, f"{method.lower()}_top_features_fold_{fold}.png")
   plt.savefig(png_filename, bbox_inches='tight', dpi=300)
   
   # Save as SVG with editable text
   svg_filename = os.path.join(output_dir, f"{method.lower()}_top_features_fold_{fold}.svg")
   plt.savefig(svg_filename, bbox_inches='tight')
   
   plt.close()
   print(f"{method} plot saved to {png_filename} and {svg_filename}")

def save_top_features_data_top(X, y, target_col, output_dir, fold=0):
   """Save the top features and target data to a CSV file for reference."""
   output_file = os.path.join(output_dir, f"top_features_data_fold_{fold}.csv")
   
   # Create a DataFrame with top features and target
   df_top = X.copy()
   df_top[target_col] = y
   
   # Save to CSV
   df_top.to_csv(output_file, index=False)
   print(f"Top features data saved to {output_file}")

def dimensionality_reduction_plots_top_features(
   shap_csv_file: str,
   feature_csv_file: str,
   dir_prefix: str,
   fold: int,
   n_top_features: int
):
   if not DIMENSIONALITY_REDUCTION_AVAILABLE:
      print("Dimensionality reduction libraries not available. Skipping top features DR plots.")
      return

   mpl.rcParams['svg.fonttype'] = 'none' # Ensure this is set for SVG output

   # Load SHAP values from CSV
   df_shap = load_shap_values_csv_top(shap_csv_file)
   
   # Get top features based on SHAP values
   top_features = get_top_features_from_shap_csv_top(df_shap, n=n_top_features)
   
   # Load full feature data
   X_full, y, target_col = load_feature_data_top(feature_csv_file)
   
   # Check which top features are available in the feature data
   available_features = [f for f in top_features if f in X_full.columns]
   missing_features = [f for f in top_features if f not in X_full.columns]
   
   if missing_features:
      print(f"Warning: {len(missing_features)} top features are not available in the feature data:")
      for f in missing_features:
         print(f"  - {f}")
   
   print(f"Using {len(available_features)} available top features")
   
   # Filter X to only include available top features
   X = X_full[available_features]
   
   print(f"Feature matrix shape: {X.shape}")
   print(f"Target vector shape: {y.shape}")
   
   # Save the top features data for reference
   save_top_features_data_top(X, y, target_col, dir_prefix, fold)
   
   # Create UMAP visualization
   plot_dimensionality_reduction_top(X, y, method='UMAP', output_dir=dir_prefix, fold=fold)
   
   # Create t-SNE visualization
   plot_dimensionality_reduction_top(X, y, method='TSNE', output_dir=dir_prefix, fold=fold)
   
   print("Done!")


# --- Main Entry Point ---
if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Run different parts of the SHAP analysis and plotting pipeline.")
   parser.add_argument('--mode', type=str, required=True,
                     choices=['get_shap_and_feature_decomposition', 'get_shap_and_feature_clustering', 'create_shap_csv', 'dimensionality_reduction_plots_full', 'dimensionality_reduction_plots_top_features'],
                     help="Mode of execution: 'get_shap_and_feature_decomposition' for class-based analysis, 'get_shap_and_feature_clustering' for cluster-based analysis, 'create_shap_csv' to convert SHAP pkl to CSV, 'dimensionality_reduction_plots_full' for DR plots on all features, or 'dimensionality_reduction_plots_top_features' for DR plots on top features.")
   
   args = parser.parse_args()

   if args.mode == 'get_shap_and_feature_decomposition':
      get_shap_and_feature_decomposition(
         model_path=MODEL_PATH_TEMPLATE.format(fold_num=DEFAULT_FOLD_IDX),
         moddata_path=MODDATA_PATH_TEMPLATE.format(fold_num=DEFAULT_FOLD_IDX),
         dir_prefix=DIR_PREFIX,
         fold_idx=DEFAULT_FOLD_IDX
      )
   elif args.mode == 'get_shap_and_feature_clustering':
      # Construct model and moddata paths based on NUM_FOLDS and TEMPLATE
      model_paths = [MODEL_PATH_TEMPLATE.format(fold_num=i) for i in range(NUM_FOLDS)]
      moddata_paths = [MODDATA_PATH_TEMPLATE.format(fold_num=i) for i in range(NUM_FOLDS)]
      
      get_shap_and_feature_clustering(
         model_paths=model_paths,
         moddata_paths=moddata_paths,
         dir_prefix=DIR_PREFIX,
         num_folds=NUM_FOLDS
      )
   elif args.mode == 'create_shap_csv':
      create_shap_csv(
         num_folds=NUM_FOLDS,
         dir_prefix=DIR_PREFIX
      )
   elif args.mode == 'dimensionality_reduction_plots_full':
      dimensionality_reduction_plots_full(
         csv_file=SHAP_CSV_FILE_DR,
         dir_prefix=DIR_PREFIX
      )
   elif args.mode == 'dimensionality_reduction_plots_top_features':
      dimensionality_reduction_plots_top_features(
         shap_csv_file=SHAP_CSV_FILE_DR,
         feature_csv_file=FEATURE_CSV_FILE_FULL_DATA,
         dir_prefix=DIR_PREFIX,
         fold=DEFAULT_FOLD_IDX,
         n_top_features=N_TOP_FEATURES_DR
      )
   else:
      print("Invalid mode selected.")