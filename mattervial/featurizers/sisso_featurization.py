import os
import json
import pandas as pd
import numpy as np

def safe_div(numerator, denominator):
   return np.where(np.abs(denominator) > 1e-12, numerator / denominator, numerator / 0.001)

def get_sisso_features(input_data, formulas_file_path=None, robust_scaler_params_path=None, type=None):
    """
    Calculate SISSO-based features for materials using symbolic regression formulas.

    This function applies SISSO (Sure Independence Screening and Sparsifying Operator)
    formulas to generate new features based on existing MatMiner features. The SISSO++
    framework generates symbolic expressions that approximate target material properties
    across multiple datasets by transforming MatMiner features using mathematical operators.

    The SISSO-based feature augmentation creates features through mathematical combinations
    of existing features to predict different targets in several datasets, guided by a
    master formulas file. This approach provides interpretable feature engineering that
    can enhance model performance.

    Args:
        input_data (str or pd.DataFrame): Input data containing MatMiner features.
            Can be either a path to a CSV file or a pandas DataFrame with featurized data.
        formulas_file_path (str, optional): Path to the SISSO formulas file. If None,
            will be determined automatically based on the 'type' parameter.
        robust_scaler_params_path (str, optional): Path to the robust scaler JSON file
            containing scaling parameters for input features. If None, will be determined
            automatically based on the 'type' parameter.
        type (str, optional): Predefined formula set type. Currently supports:
            - "SISSO_FORMULAS_v1": Uses the default SISSO formulas and scaler parameters.
            If provided, formulas_file_path and robust_scaler_params_path are set automatically.

    Returns:
        pd.DataFrame: DataFrame containing only the newly calculated SISSO features.
            Column names follow the pattern 'SISSO_<dataset>_<feature_index>' where
            dataset corresponds to the original training dataset and feature_index
            is the sequential number of the SISSO feature.

    Raises:
        FileNotFoundError: If the specified formulas file or scaler file is not found.
        ValueError: If the input data format is invalid or required columns are missing.
        KeyError: If required features for SISSO formulas are not present in input data.

    Examples:
        >>> # Using predefined formula set
        >>> sisso_features = get_sisso_features(
        ...     input_data="dataset_MatMinerFeaturized.csv",
        ...     type="SISSO_FORMULAS_v1"
        ... )
        >>> print(sisso_features.shape)
        (1000, 300)  # 300 SISSO features for 1000 materials

        >>> # Using DataFrame input
        >>> import pandas as pd
        >>> df = pd.read_csv("featurized_data.csv")
        >>> sisso_features = get_sisso_features(df, type="SISSO_FORMULAS_v1")

        >>> # Using custom formula files
        >>> sisso_features = get_sisso_features(
        ...     input_data=df,
        ...     formulas_file_path="/path/to/custom_formulas.txt",
        ...     robust_scaler_params_path="/path/to/custom_scaler.json"
        ... )

    Note:
        The SISSO formulas are derived from 15 different datasets and provide versatile
        feature engineering capabilities. The formulas use mathematical operators including
        addition, subtraction, multiplication, division, sine, cosine, exponential, and
        logarithm to create new features from existing MatMiner descriptors.
    """
    print(f"--- Starting GET SISSO FEATURES ---")

    # If type is provided, set file paths automatically.
    if type is not None:
        if type == "SISSO_FORMULAS_v1":
            sisso_prefix = "SISSO"
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            formulas_file_path = os.path.join(curr_dir, "formulas", "SISSO_FORMULAS_v1.txt")
            robust_scaler_params_path = os.path.join(curr_dir, "formulas", "robust_scaler_mpgap_for_sisso.json")
            print(f"Using type '{type}', formulas file set to: {formulas_file_path}")
            print(f"Using type '{type}', robust scaler file set to: {robust_scaler_params_path}")
        elif type == "SISSO_FORMULAS_v2":
            sisso_prefix = "SISSOv2"
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            formulas_file_path = os.path.join(curr_dir, "formulas", "SISSO_FORMULAS_v2.txt")
            robust_scaler_params_path = os.path.join(curr_dir, "formulas", "robust_scaler_mpgap_for_sisso.json")
            print(f"Using type '{type}', formulas file set to: {formulas_file_path}")
            print(f"Using type '{type}', robust scaler file set to: {robust_scaler_params_path}")
        else:
            raise ValueError(f"Unrecognized type: {type}.")
    else:
        # Ensure both file paths are provided if type is not given.
        if not (formulas_file_path and robust_scaler_params_path):
            raise ValueError("Either provide a type or both formulas_file_path and robust_scaler_params_path.")

    print(f"Master Formulas File: {formulas_file_path}")
    print(f"Robust Scaler Params: {robust_scaler_params_path}")

    try:
        # 1. Load the data
        print("Loading data...")
        if isinstance(input_data, str):
            print(f"Loading data from CSV file: {input_data}")
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            print("Using provided DataFrame")
            df = input_data.copy()
        else:
            raise ValueError("input_data must be either a path to a CSV file or a pandas DataFrame")

        target_column = 'target'
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        # Identify feature columns (exclude target and known nonfeature columns).
        feature_columns = [col for col in df.columns if col not in [target_column, 'composition', 'material_id', 'structure']]
        if not feature_columns:
            raise ValueError("No feature columns found in the input data.")
        print(f"Feature columns identified: {feature_columns}")

        X = df[feature_columns]

        # 2. Load and apply RobustScaler parameters.
        print(f"Loading scaling parameters from '{robust_scaler_params_path}'...")
        try:
            with open(robust_scaler_params_path, 'r') as f:
                scaling_params = json.load(f)
            print("Scaling parameters loaded.")
        except Exception as e:
            raise FileNotFoundError(f"Error loading scaling parameters: {e}")

        # Scale only those features that are present in the scaling parameters.
        common_features = [f for f in X.columns if f in scaling_params]
        for feature in common_features:
            median = scaling_params[feature]['median']
            iqr = scaling_params[feature]['iqr']
            X.loc[:, feature] = (X[feature] - median) / iqr

        # Create a scaled features DataFrame.
        X_scaled_df = pd.DataFrame(X, columns=X.columns, index=X.index)

        # 3. Load all formulas from the master formulas file.
        print(f"Loading all SISSO formulas from master file '{formulas_file_path}'...")
        all_formulas = {}      # Dictionary to hold { feature_name: formula }
        current_dataset = None # Tracks the current section (dataset) in the formulas file.
        feature_counter = {}   # Keeps track of the formula numbering per dataset.

        with open(formulas_file_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                # If the line is a dataset header, note the dataset and reset the counter.
                if stripped.startswith("# Dataset:"):
                    current_dataset = stripped.split(":", 1)[1].strip()
                    feature_counter[current_dataset] = 0
                    print(f"Found dataset section: {current_dataset}")
                elif stripped.startswith("Feature"):
                    if current_dataset is None:
                        print(f"Warning: Found a formula before any dataset section. Skipping line: {stripped}")
                        continue
                    parts = stripped.split(":", 1)
                    if len(parts) != 2:
                        print(f"Warning: Skipping an invalid formula line: {stripped}")
                        continue
                    feature_counter[current_dataset] += 1
                    feature_name = f"{sisso_prefix}_{current_dataset}_{feature_counter[current_dataset]}"
                    formula = parts[1].strip()
                    # Replace '^' with '**' to conform to Python's exponentiation operator.
                    formula = formula.replace("^", "**")
                    all_formulas[feature_name] = formula

        if not all_formulas:
            raise ValueError("No SISSO formulas found in the master formulas file.")

        print(f"Loaded {len(all_formulas)} formulas from the master file.")

        # 4. Evaluate each formula to generate SISSO features.
        print("Calculating SISSO features...")
        eval_env = {
            "df": X_scaled_df,
            "np": np,
            "abs": abs,
            "ln": np.log,
            "sqrt": np.sqrt,
            "cbrt": np.cbrt,
            "sin": np.sin,
            "cos": np.cos,
            "pow": pow,
            "safe_div": safe_div,
        }

        sisso_features_df = pd.DataFrame(index=X_scaled_df.index)
        for feature_name, formula in all_formulas.items():
            print(f"  Evaluating {feature_name}: {formula}")
            try:
                feature_values = eval(formula, eval_env)
                sisso_features_df[feature_name] = feature_values
                print(f"    -> {feature_name} computed successfully.")
            except Exception as e:
                print(f"    Error calculating {feature_name}: {e}. Skipping this feature.")
                continue

        print("--- Finished GET SISSO FEATURES ---")
        return sisso_features_df

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("--- GET SISSO FEATURES Failed ---")
    except ValueError as e:
        print(f"Value error: {e}")
        print("--- GET SISSO FEATURES Failed ---")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("--- GET SISSO FEATURES Failed ---")

__all__ = ("get_sisso_features",)