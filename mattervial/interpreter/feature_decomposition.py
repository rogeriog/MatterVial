import os
import re
import json
import pickle # Keep this import as it might be used by other parts of mattervial
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb # Keep this import as it might be used by other parts of mattervial
import shap # Keep this import as it might be used by other parts of mattervial
import matplotlib.pyplot as plt # Keep this import as it might be used by other parts of mattervial
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shutil 
# ----------------------------------------------------------------
# Helper Functions (from previous script, kept for completeness)
# ----------------------------------------------------------------
def log_message(message, level="INFO"):
    """
    Print a timestamped log message with specified level.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def rename_cols(df):
    """
    Replace problematic characters in column names so that XGBoost accepts them.
    """
    df = df.copy()
    df.columns = [col.replace("[", "_")
                    .replace("]", "_")
                    .replace(" ", "_")
                    .replace("<", "_")
                    .replace(">", "_")
                    .replace(",", "_")
                    .replace("|", "_")
                    .replace(".", "_")
                  for col in df.columns]
    return df

# We will no longer use check_existing_outputs in its current form for SHAP decomposition
# def check_existing_outputs(target_col, out_dir):
#     """
#     Check if all output files for a target already exist.
#     Returns True if all files exist, False otherwise.
#     """
#     json_filename = os.path.join(out_dir, f"{target_col}_top_features.json")
#     plot_filename = os.path.join(out_dir, f"{target_col}_shap_plot.svg")
#     pickle_filename = os.path.join(out_dir, f"{target_col}_shap_values_explainer.pkl")

#     return all(os.path.exists(f) for f in [json_filename, plot_filename, pickle_filename])


def recursive_feature_elimination(X, y, target_threshold=300, drop_fraction=0.1, n_jobs=24):
    """
    Recursively drops a fraction (drop_fraction) of the current least important
    features using an XGBoost regressor until the number of predictors is reduced 
    to target_threshold.
    """
    log_message("Starting recursive feature elimination process...")
    X_current = X.copy()
    initial_num = X_current.shape[1]

    if initial_num <= target_threshold:
        log_message(f"Feature count ({initial_num}) is already <= target ({target_threshold}). Skipping elimination.", "INFO")
        return X_current

    log_message(f"Initial features: {initial_num}, Target features: {target_threshold}")

    iteration = 0
    while X_current.shape[1] > target_threshold:
        iteration += 1
        current_num = X_current.shape[1]
        log_message(f"RFE Iteration {iteration}: Current feature count = {current_num}")

        # Apply rename_cols helper function
        X_renamed = rename_cols(X_current.copy()) 

        X_numeric = X_renamed.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_numeric = pd.to_numeric(y, errors='coerce')

        if y_numeric.isnull().any():
            log_message(f"Warning: {y_numeric.isnull().sum()} NaN values found in target", "WARNING")

        model = xgb.XGBRegressor(n_jobs=n_jobs, random_state=1, objective='reg:squarederror')
        try:
            model.fit(X_numeric.values, y_numeric.values)
        except Exception as e:
            log_message(f"Error during XGBoost fit in RFE: {e}", "ERROR")
            return X_current # Return current features on error

        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        num_to_drop = max(1, int(current_num * drop_fraction))
        
        if current_num - num_to_drop < target_threshold:
            num_to_drop = current_num - target_threshold
        
        drop_features = list(X_current.columns[sorted_idx[:num_to_drop]])
        X_current.drop(columns=drop_features, inplace=True)
        log_message(f"  Dropped {num_to_drop} features. Remaining: {X_current.shape[1]}")

        if num_to_drop == 0:
            log_message("Warning: Calculated num_to_drop is 0. Breaking elimination loop.", "WARNING")
            break

    log_message(f"Recursive feature elimination completed. Final feature count: {X_current.shape[1]}")
    return X_current


def determine_output_folder(target_col):
    """
    Determines the subfolder name based on the target feature name.
    This function is now more flexible to handle new feature names.
    """
    target_low = target_col.lower()
    if target_low.startswith("mvl16_"): return "MVL16"
    if target_low.startswith("mvl32_"): return "MVL32"
    if target_low.startswith("roost_"):
        if "mp_gap" in target_low or "mpgap" in target_low: return "ROOST_mp_gap"
        if "oqmd" in target_low: return "ROOST_oqmd"
        return "ROOST_other"
    if target_low.startswith("megnet_matminerencoded_v1"): return "l-MM_v1"
    if target_low.startswith("megnet_ofmencoded_v1"): return "l-OFM_v1"
    if target_low.startswith("orb_v3"): return "ORB_v3"
    # Add more specific rules here if needed for other prefixes, e.g.:
    if target_low.startswith("cogn"): return "coGN"

    return target_col # Generic fallback, uses the full feature name as folder


# ----------------------------------------------------------------
# Core SHAP Decomposition Logic Function (from previous script)
# ----------------------------------------------------------------
def run_shap_decomposition(feature_prefix: str, base_dataset: str, n_cpus: int = 24, feature_suffix: str = None):  
    """  
    Performs SHAP feature decomposition for all target features matching a specified prefix  
    and an optional suffix.  
  
    Args:  
        feature_prefix (str): The prefix of the target features to decompose (e.g., 'coGN_ReadoutComponent1').  
        base_dataset (str): Path to the input CSV dataset.  
        n_cpus (int): Number of CPU cores to use for XGBoost.  
        feature_suffix (str, optional): An optional suffix that the target features must end with.  
                                        If None, no suffix filtering is applied. Defaults to None.  
    """  
    log_message("="*60)  
    log_message("STARTING SHAP FEATURE DECOMPOSITION ANALYSIS")  
    log_message(f"Running with params: feature_prefix='{feature_prefix}', base_dataset='{base_dataset}', n_cpus={n_cpus}, feature_suffix='{feature_suffix}'")  
    log_message("="*60)  
  
    # -----------------------  
    # 1. Load the data  
    # -----------------------  
    log_message("STEP 1: Loading data...")  
    data_path = base_dataset  
    if not os.path.exists(data_path):  
        log_message(f"Data file not found: {data_path}", "ERROR")  
        return  
  
    try:  
        df = pd.read_csv(data_path)  
        log_message(f"Successfully loaded data with shape: {df.shape}")  
    except Exception as e:  
        log_message(f"Error loading data: {e}", "ERROR")  
        return  
  
    # -------------------------------  
    # 2. Define Interpretable Features and Identify Target Features by Prefix  
    # -------------------------------  
    log_message("STEP 2: Identifying interpretable features and target features by prefix...")  
    MATMINER_PREFIXES = [  
        'AtomicOrbitals|', 'AtomicPackingEfficiency|', 'BandCenter|', 'ElementFraction|',  
        'ElementProperty|', 'IonProperty|', 'Miedema|', 'Stoichiometry|',  
        'TMetalFraction|', 'ValenceOrbital|', 'YangSolidSolution|',  
        'ElectronegativityDiff|', 'OxidationStates|', 'DensityFeatures|',  
        'GlobalSymmetryFeatures|', 'CoulombMatrix|', 'SineCoulombMatrix|',  
        'BondFractions|', 'StructuralHeterogeneity|', 'MaximumPackingEfficiency|',  
        'ChemicalOrdering|', 'XRDPowderPattern|', 'RadialDistributionFunction|',  
        'AGNIFingerPrint|', 'AverageBondAngle|', 'AverageBondLength|',  
        'BondOrientationParameter|', 'ChemEnvSiteFingerprint|', 'CoordinationNumber|',  
        'CrystalNNFingerprint|', 'GaussianSymmFunc|', 'GeneralizedRDF|',  
        'LocalPropertyDifference|', 'OPSiteFingerprint|', 'VoronoiFingerprint|'  
    ]  
    matminer_pattern = r'^(' + '|'.join([re.escape(p) for p in MATMINER_PREFIXES]) + ')'  
    megnet_ofm_pattern = r'^MEGNet_OFMEncoded_v1_'  
      
    interpretable_cols = [col for col in df.columns if (re.search(matminer_pattern, col) or re.search(megnet_ofm_pattern, col))]  
      
    if not interpretable_cols:  
        log_message("No interpretable features found! Please check MATMINER_PREFIXES and MEGNet_OFM_pattern.", "ERROR")  
        return  
    X_interpretable = df[interpretable_cols]  
    log_message(f"Found {len(interpretable_cols)} interpretable features.")  
  
    # Select all columns that start with the provided feature_prefix and optionally end with feature_suffix  
    if feature_suffix:  
        target_cols = sorted([col for col in df.columns if col.startswith(feature_prefix) and col.endswith(feature_suffix)])  
        log_message(f"Found {len(target_cols)} target features matching prefix '{feature_prefix}' and ending with '{feature_suffix}'.")  
    else:  
        target_cols = sorted([col for col in df.columns if col.startswith(feature_prefix)])  
        log_message(f"Found {len(target_cols)} target features matching prefix '{feature_prefix}' (no suffix filter).")  
      
    if not target_cols:  
        log_message(f"Error: No target features found with the specified prefix and/or suffix.", "ERROR")  
        log_message(f"Available columns include: {df.columns.tolist()[:10]}...", "INFO") # Show first 10 columns  
        return  
  
    base_output_dir = "SHAP_feature_decomposition"  
    os.makedirs(base_output_dir, exist_ok=True)  
  
    # -------------------------------  
    # 3. Process each target feature matching the prefix  
    # -------------------------------  
    log_message("STEP 3: Processing target features matching the prefix...")  
    total_targets_to_process = len(target_cols)  
      
    # Keep track of targets that were skipped because all files existed  
    skipped_targets = []  
    # Keep track of targets that were processed (or partially processed)  
    processed_targets = []  
  
    for i, target_col in enumerate(target_cols, 1):  
        log_message("="*80)  
        log_message(f"PROCESSING TARGET {i}/{total_targets_to_process}: {target_col}")  
        log_message("="*80)  
  
        group_folder = determine_output_folder(target_col)  
        out_dir = os.path.join(base_output_dir, group_folder)  
        os.makedirs(out_dir, exist_ok=True)  
  
        # Define file paths for the current target  
        json_filename = os.path.join(out_dir, f"{target_col}_top_features.json")  
        plot_filename = os.path.join(out_dir, f"{target_col}_shap_plot.svg")  
        pickle_filename = os.path.join(out_dir, f"{target_col}_shap_values_explainer.pkl")  
  
        # Check if ALL files exist for this target  
        if os.path.exists(json_filename) and os.path.exists(plot_filename) and os.path.exists(pickle_filename):  
            log_message(f"SKIPPING {target_col} - all output files already exist.", "INFO")  
            skipped_targets.append(target_col)  
            continue # Skip to the next target if all files are present  
  
        # If not all files exist, proceed with processing (or re-processing missing ones)  
        log_message(f"Initiating processing for {target_col} (some files might be missing or need regeneration).")  
        processed_targets.append(target_col) # Add to processed list if we're going to work on it  
  
        y = df[target_col]  
        X_temp = X_interpretable.copy()  
  
        X_reduced = recursive_feature_elimination(X_temp, y,  
                                                  target_threshold=300,  
                                                  drop_fraction=0.1,  
                                                  n_jobs=n_cpus)  
  
        predictors = rename_cols(X_reduced)  
        model = xgb.XGBRegressor(n_jobs=n_cpus, random_state=1, objective='reg:squarederror')  
          
        try:  
            X_numeric = predictors.apply(pd.to_numeric, errors='coerce').fillna(0)  
            y_numeric = pd.to_numeric(y, errors='coerce')  
            model.fit(X_numeric.values, y_numeric.values)  
        except Exception as e:  
            log_message(f"Error during final XGBoost fit for target {target_col}: {e}", "ERROR")  
            continue # Continue to next target if model fit fails  
  
        log_message("Computing SHAP values...")  
        explainer = shap.Explainer(model, X_numeric)  
        shap_values = explainer(X_numeric)  
  
        log_message("Saving outputs...")  
        # Plot - only save if file doesn't exist  
        if not os.path.exists(plot_filename):  
            plt.rcParams.update({'font.size': 16})  
            shap.summary_plot(shap_values.values, X_numeric, show=False)  
            plt.title(f"SHAP Summary for {target_col}")  
            plt.savefig(plot_filename, bbox_inches='tight')  
            plt.close()  
            log_message(f"Saved plot: {plot_filename}")  
        else:  
            log_message(f"Plot already exists: {plot_filename} - Skipping.")  
  
        # Pickle - only save if file doesn't exist  
        if not os.path.exists(pickle_filename):  
            with open(pickle_filename, 'wb') as f:  
                pickle.dump(shap_values, f)  
            log_message(f"Saved pickle: {pickle_filename}")  
        else:  
            log_message(f"Pickle already exists: {pickle_filename} - Skipping.")  
  
        # JSON - only save if file doesn't exist  
        if not os.path.exists(json_filename):  
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)  
            sorted_idx = np.argsort(mean_abs_shap)[::-1]  
            top_features = {shap_values.feature_names[idx]: float(mean_abs_shap[idx]) for idx in sorted_idx[:30]}  
            with open(json_filename, 'w') as f:  
                json.dump(top_features, f, indent=4)  
            log_message(f"Saved JSON: {json_filename}")  
        else:  
            log_message(f"JSON already exists: {json_filename} - Skipping.")  
          
        log_message(f"Finished output checks for {target_col}.")  
  
    log_message("="*60)  
    log_message(f"ANALYSIS FOR PREFIX '{feature_prefix}' COMPLETE")  
    log_message(f"Total targets found: {total_targets_to_process}")  
    log_message(f"Targets skipped (all files existed): {len(skipped_targets)} - {skipped_targets}")  
    log_message(f"Targets processed (or attempted): {len(processed_targets)} - {processed_targets}")  
    log_message("="*60)

def pca_based_sampler(pca_df, n_bins=10, alpha=1.0,
                      random_state=42, id_column="material_id",
                      max_pca_components=None):
    """
    Perform sampling on the provided PCA features dataframe.
    Returns:
      A list of sampled material_ids.
    """
    print(f"\nSampling from PCA features using random seed {random_state}")
    np.random.seed(random_state)

    # Get PCA component columns (by default all columns starting with "PC_")
    pca_columns = [col for col in pca_df.columns if col.startswith("PC_")]
    if max_pca_components is not None:
        try:
            max_pc = int(max_pca_components)
        except ValueError:
            max_pc = len(pca_columns)
        pca_columns = pca_columns[:max_pc]

    X_pca = pca_df[pca_columns].values
    n_components = len(pca_columns)
    print(f"Sampling using {n_components} PCA components on {len(pca_df)} samples")

    # Build bins on each component
    bins = []
    for i in range(n_components):
        component = X_pca[:, i]
        bin_edges = np.linspace(component.min(), component.max(), n_bins + 1)
        bin_indices = np.digitize(component, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bins.append(bin_indices)
    bins = np.stack(bins, axis=1)
    print(f"Binning array shape: {bins.shape}")

    # Generate unique key strings from the bin indices per sample
    bin_keys = ["_".join(map(str, b)) for b in bins]

    # Group sample indices by their bin key.
    bin_to_indices = {}
    for idx, key in enumerate(bin_keys):
        bin_to_indices.setdefault(key, []).append(idx)
    print(f"Found {len(bin_to_indices)} unique bins during sampling.")

    # For each bin, sample a fraction (alpha) of the indices (at least one)
    selected_indices = []
    for key, indices in bin_to_indices.items():
        n_samples_bin = max(1, int(len(indices) * alpha))
        sampled = np.random.choice(indices, size=n_samples_bin, replace=False)
        selected_indices.extend(sampled)
    print(f"Total selected samples: {len(selected_indices)}")

    # Return the id values corresponding to selected samples.
    selected_ids = pca_df.iloc[selected_indices][id_column].tolist()
    return selected_ids

def compute_pca_features(df, feature_columns, target_column,
                         id_column="material_id",
                         n_components=10,
                         include_target_in_pca=False,
                         target_weight=1.0,
                         random_state=42):
    """
    Compute PCA features based on the provided dataframe.
    The PCA is performed on the columns specified by feature_columns
    (optionally augmented by the target), and the resulting dataframe
    (with PCA columns and the id_column) is returned.
    """
    np.random.seed(random_state)
    print(f"Computing PCA using {len(feature_columns)} features for target '{target_column}' ...")

    # Extract features (from the top-feature columns)
    X = df[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Feature matrix shape after scaling: {X_scaled.shape}")

    # Optionally augment with target information
    if include_target_in_pca:
        print(f"Including target '{target_column}' with weight {target_weight}...")
        y = df[target_column].values.reshape(-1, 1)
        y_scaled = StandardScaler().fit_transform(y)
        X_augmented = np.hstack([X_scaled, target_weight * y_scaled])
        print(f"Augmented feature matrix shape: {X_augmented.shape}")
    else:
        X_augmented = X_scaled

    # Limit number of PCA components to available dimensions
    n_components_actual = min(n_components, X_augmented.shape[1])
    print(f"Performing PCA with {n_components_actual} components")
    pca = PCA(n_components=n_components_actual, random_state=random_state)
    X_pca = pca.fit_transform(X_augmented)
    print(f"Explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.3f}")

    # Create a dataframe with PCA results. PCA columns are named "PC_1", "PC_2", etc.
    pca_columns = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    # Keep the id column for later identification.
    pca_df[id_column] = df[id_column].values

    return pca_df

# ----------------------------------------------------------------
# Generate SISSO Samples
# ----------------------------------------------------------------
def generate_sisso_samples(
    base_dataset_path: str,
    shap_top_features_base_folder: str = "SHAP_feature_decomposition",
    sisso_output_folder: str = "sampled_data_for_sisso",
    pca_n_components: int = 20,
    pca_include_target: bool = True,
    pca_target_weight: float = 3.0,
    sampler_n_bins: int = 5,
    sampler_alpha: float = 0.01,
    sampler_max_pca_components: int = 3,
    random_seed: int = 42,
):
    """
    Generates SISSO-ready sampled datasets based on SHAP top features and PCA-based sampling.

    Args:
        base_dataset_path (str): Path to the main input CSV dataset.
        shap_top_features_base_folder (str): Base folder containing SHAP top-feature JSON files.
                                             (e.g., "SHAP_feature_decomposition")
        sisso_output_folder (str): Folder where SISSO-ready CSVs and feature mappings will be saved.
        pca_n_components (int): Number of PCA components to compute.
        pca_include_target (bool): Whether to include the target column in PCA.
        pca_target_weight (float): Weight for the target column if included in PCA.
        sampler_n_bins (int): Number of bins for PCA-based sampling.
        sampler_alpha (float): Fraction of samples to select per bin (at least one).
        sampler_max_pca_components (int): Maximum number of PCA components to use for sampling.
        random_seed (int): Random seed for reproducibility.
    """
    log_message("="*60)
    log_message("STARTING SISSO SAMPLE GENERATION")
    log_message(f"Base Dataset: {base_dataset_path}")
    log_message(f"SHAP Features Folder: {shap_top_features_base_folder}")
    log_message(f"SISSO Output Folder: {sisso_output_folder}")
    log_message("="*60)

    np.random.seed(random_seed) # Set global random seed for this function

    # Read the main dataset only once.
    if not os.path.exists(base_dataset_path):
        log_message(f"Base dataset file not found: {base_dataset_path}", "ERROR")
        return

    try:
        df_main = pd.read_csv(base_dataset_path)
        df_main = rename_cols(df_main) # Apply renaming to the main dataframe
        log_message(f"Successfully loaded base dataset with shape: {df_main.shape}")
    except Exception as e:
        log_message(f"Error loading base dataset: {e}", "ERROR")
        return

    os.makedirs(sisso_output_folder, exist_ok=True) # Ensure output folder exists once

    processed_sisso_targets = []
    skipped_sisso_targets = []

    # For each JSON file with top features...
    for root, dirs, files in os.walk(shap_top_features_base_folder):
        
        for file in files:
            if file.endswith("_top_features.json"):
                json_path = os.path.join(root, file)
                log_message(f"\nProcessing SHAP features file: {json_path}")

                # Extract the target column from the filename.
                pattern = r"^(.*)_top_features\.json$"
                match = re.match(pattern, file)
                if not match:
                    log_message(f"Could not parse target from filename: {file}. Skipping...", "WARNING")
                    continue

                target_column = match.group(1)
                log_message(f"Target column parsed from filename: {target_column}")

                # Define output file paths for the current target
                sampled_output_path = os.path.join(sisso_output_folder, f"SISSO_sampled_{target_column}.csv")
                mapping_json_path = os.path.join(sisso_output_folder, f"feature_mapping_{target_column}.json")

                # Check if output files already exist for SISSO
                if os.path.exists(sampled_output_path) and os.path.exists(mapping_json_path):
                    log_message(f"Output files for target '{target_column}' already exist. Skipping SISSO computation.", "INFO")
                    skipped_sisso_targets.append(target_column)
                    continue # Skip to the next file

                log_message(f"Generating SISSO samples for target: {target_column}")
                processed_sisso_targets.append(target_column)

                # Load the SHAP top features dictionary.
                try:
                    with open(json_path, "r") as f:
                        top_features_dict = json.load(f)
                except Exception as e:
                    log_message(f"Error loading top features JSON {json_path}: {e}", "ERROR")
                    continue

                top_feature_columns = list(top_features_dict.keys())
                log_message(f"Extracted {len(top_feature_columns)} top features.")

                # Ensure target column and all top features exist in the main dataframe
                missing_cols = [col for col in [target_column] + top_feature_columns if col not in df_main.columns]
                if missing_cols:
                    log_message(f"Missing required columns in base dataset for {target_column}: {missing_cols}. Skipping.", "ERROR")
                    continue

                log_message(f"Target distribution '{target_column}' in full dataset:\n{df_main[target_column].describe()}", "INFO")

                # Create a subset of the full dataset with necessary columns:
                # material_id, target, and the top features.
                required_cols = ["material_id", target_column] + top_feature_columns
                df_subset = df_main[required_cols].copy()

                # Use PCA on the top features to obtain sampled material_ids.
                pca_df = compute_pca_features(
                    df=df_subset,
                    feature_columns=top_feature_columns,
                    target_column=target_column,
                    id_column="material_id",
                    n_components=pca_n_components,
                    include_target_in_pca=pca_include_target,
                    target_weight=pca_target_weight,
                    random_state=random_seed
                )

                sampled_ids = pca_based_sampler(
                    pca_df=pca_df,
                    n_bins=sampler_n_bins,
                    alpha=sampler_alpha,
                    random_state=random_seed,
                    id_column="material_id",
                    max_pca_components=sampler_max_pca_components
                )
                log_message(f"Sampled {len(sampled_ids)} material IDs for target '{target_column}'.")

                # IMPORTANT:
                # The PCA is only used to select material_ids.
                # Now filter the original subset (df_subset) to keep only the sampled rows.
                final_df = df_subset[df_subset["material_id"].isin(sampled_ids)].copy()
                log_message(f"Number of rows after sampling: {len(final_df)}")

                log_message(f"Target distribution '{target_column}' in sampled dataset:\n{final_df[target_column].describe()}", "INFO")

                # Prepare SISSO input:
                #   1. Drop the material_id column.
                #   2. Rename the target column to "target".
                #   3. Keep only the top features (from the JSON; ignore the PCA features).
                #      Only up to 30 features are preserved.
                final_df = final_df.drop(columns=["material_id"])
                final_df = final_df.rename(columns={target_column: "target"})

                # Preserve the top features as they appear in top_feature_columns.
                # In case there are more than 30, only retain the first 30.
                n_features_to_keep = min(len(top_feature_columns), 30)
                selected_features = top_feature_columns[:n_features_to_keep]
                
                # Build mapping: new feature name -> original feature name.
                mapping_dict = {}
                for i, orig_col in enumerate(selected_features):
                    new_name = f"feature_{i+1}"
                    mapping_dict[new_name] = orig_col

                # Keep only the target and the selected feature columns.
                final_df = final_df[["target"] + selected_features].copy()
                # Rename these feature columns.
                rename_mapping = {orig: f"feature_{i+1}" for i, orig in enumerate(selected_features)}
                final_df = final_df.rename(columns=rename_mapping)

                # Optionally add a sample_id column.
                final_df.reset_index(drop=True, inplace=True)
                final_df.index.name = "sample_id"
                final_df.reset_index(inplace=True)

                # Save the mapping dictionary as JSON.
                try:
                    with open(mapping_json_path, "w") as jf:
                        json.dump(mapping_dict, jf, indent=4)
                    log_message(f"Feature mapping saved to {mapping_json_path}")
                except Exception as e:
                    log_message(f"Error saving feature mapping to {mapping_json_path}: {e}", "ERROR")

                # Save the final SISSO input CSV.
                # The first line is preceded by '#' for a comment.
                try:
                    with open(sampled_output_path, "w") as outf:
                        outf.write("#")
                        final_df.to_csv(outf, index=False, header=True)
                    log_message(f"SISSO-ready sampled data saved to {sampled_output_path}\n")
                except Exception as e:
                    log_message(f"Error saving SISSO sampled data to {sampled_output_path}: {e}", "ERROR")

    log_message("="*60)
    log_message("SISSO SAMPLE GENERATION COMPLETE")
    log_message(f"Total SHAP top feature JSONs found: {len(processed_sisso_targets) + len(skipped_sisso_targets)}")
    log_message(f"Targets skipped (SISSO outputs existed): {len(skipped_sisso_targets)} - {skipped_sisso_targets}")
    log_message(f"Targets processed (or attempted for SISSO): {len(processed_sisso_targets)} - {processed_sisso_targets}")
    log_message("="*60)
    # Dynamically determine the path to the help_scripts folder  
    # This script is in mattervial/interpreter/feature_decomposition.py  
    # We need to go up two levels to 'mattervial' and then down to 'help_scripts'  
    current_script_dir = os.path.dirname(os.path.abspath(__file__))  
    mattervial_root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))  
      
    source_script_path = os.path.join(mattervial_root_dir, 'mattervial', 'interpreter', 'help_scripts', 'populate_sisso_calcs.py')
    source2_script_path = os.path.join(mattervial_root_dir, 'mattervial', 'interpreter', 'help_scripts', 'get_sisso_final_formulas.py')
    destination_script_path = os.path.join(sisso_output_folder, "populate_sisso_calcs.py")  
    destination2_script_path = os.path.join(sisso_output_folder, "get_sisso_final_formulas.py")
    if os.path.exists(source_script_path): 
        try:  
            shutil.copy(source_script_path, destination_script_path)  
            shutil.copy(source2_script_path, destination2_script_path)
            log_message(f"Copied '{os.path.basename(source_script_path)}' to '{sisso_output_folder}'")
            log_message(f"Copied '{os.path.basename(source2_script_path)}' to '{sisso_output_folder}'")
            log_message("="*60)
            log_message("NEXT STEPS:")
            log_message("1. Review the files generated.")
            log_message("2. Edit the populate_sisso_calcs.py verifying the SISSO settings and editing for submission in your environment.")
            log_message("3. Submit the SISSO calculations using populate_sisso_calcs.py")
            log_message("4. Once SISSO calculations are complete, run get_sisso_formulas.py to extract the results")
            log_message("="*60)
        except Exception as e:  
            log_message(f"Error copying script '{source_script_path}' to '{sisso_output_folder}': {e}", "ERROR")  
    else:  
        log_message(f"Source script not found: {source_script_path}", "WARNING")


# ----------------------------------------------------------------
# Main execution block for command-line usage
# ----------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SHAP feature decomposition or generate SISSO samples.")
    
    # Subparsers for different functionalities
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for SHAP decomposition
    shap_parser = subparsers.add_parser('shap-decompose', help='Perform SHAP feature decomposition.')
    shap_parser.add_argument('--feature-prefix', type=str, required=True,
                             help='The prefix of the target features to decompose (e.g., MEGNet_MMencoded, coGN_v1).')
    shap_parser.add_argument('--base-dataset', type=str,
                             default='data/matbench_mp_gap/matbench_mp_gap_featurizedMM2020Struct_mattervial.csv',
                             help='Path to the input CSV dataset.')
    shap_parser.add_argument('--n-cpus', type=int, default=os.cpu_count(), help='Number of CPU cores to use for XGBoost.')

    # Subparser for SISSO sample generation
    sisso_parser = subparsers.add_parser('sisso-sample', help='Generate SISSO-ready sampled datasets.')
    sisso_parser.add_argument('--base-dataset', type=str,
                              default='data/matbench_mp_gap/matbench_mp_gap_featurizedMM2020Struct_mattervial.csv',
                              help='Path to the main input CSV dataset.')
    sisso_parser.add_argument('--shap-folder', type=str,
                              default='SHAP_feature_decomposition',
                              help='Base folder containing SHAP top-feature JSON files.')
    sisso_parser.add_argument('--output-folder', type=str,
                              default='sampled_data_for_sisso',
                              help='Folder where SISSO-ready CSVs and feature mappings will be saved.')
    sisso_parser.add_argument('--pca-n-components', type=int, default=20,
                              help='Number of PCA components to compute.')
    sisso_parser.add_argument('--pca-include-target', type=bool, default=True,
                              help='Whether to include the target column in PCA.')
    sisso_parser.add_argument('--pca-target-weight', type=float, default=3.0,
                              help='Weight for the target column if included in PCA.')
    sisso_parser.add_argument('--sampler-n-bins', type=int, default=5,
                              help='Number of bins for PCA-based sampling.')
    sisso_parser.add_argument('--sampler-alpha', type=float, default=0.01,
                              help='Fraction of samples to select per bin (at least one).')
    sisso_parser.add_argument('--sampler-max-pca-components', type=int, default=3,
                              help='Maximum number of PCA components to use for sampling.')
    sisso_parser.add_argument('--random-seed', type=int, default=42,
                              help='Random seed for reproducibility.')
    

    args = parser.parse_args()

    if args.command == 'shap-decompose':
        run_shap_decomposition(
            feature_prefix=args.feature_prefix,
            base_dataset=args.base_dataset,
            n_cpus=args.n_cpus
        )
    elif args.command == 'sisso-sample':
        generate_sisso_samples(
            base_dataset_path=args.base_dataset,
            shap_top_features_base_folder=args.shap_folder,
            sisso_output_folder=args.output_folder,
            pca_n_components=args.pca_n_components,
            pca_include_target=args.pca_include_target,
            pca_target_weight=args.pca_target_weight,
            sampler_n_bins=args.sampler_n_bins,
            sampler_alpha=args.sampler_alpha,
            sampler_max_pca_components=args.sampler_max_pca_components,
            random_seed=args.random_seed,
        )
    else:
        parser.print_help()