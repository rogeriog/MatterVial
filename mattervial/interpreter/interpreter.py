import os
import json
import glob

def load_all_json_files(directory):
    """
    Loads and merges all JSON files in the supplied directory.
    Assumes that each JSON file contains a dictionary.
    """
    json_files = glob.glob(os.path.join(directory, '*.json'))
    combined_data = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Merge the JSON file contents into the combined dict.
                # If there are duplicate keys across files, later files will override earlier values.
                combined_data.update(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return combined_data

def FeatureInterpreter(features, preferences=None):
    """
    Retrieves feature-specific information from JSON files in designated
    'data' (for shap and metrics info) and 'formulas' (for formula info) folders.

    Parameters:
        features (str or list of str):
            A feature name or a list of feature names. For example,
            "MEGNet_MatMiner_1" or ["MEGNet_MatMiner_3", "MEGNet_MatMiner_5"].
        preferences (dict, optional):
            A dictionary to customize what information is returned.
            The keys can be:
              - 'shap': (bool) include shap analysis info (default True)
              - 'formula': (bool) include formula info (default True)
              - 'metrics': (bool) include model metrics info (default False)
              - 'model_info': (bool) include model architecture/hyperparameters info (default False)

    Returns:
        dict: A dictionary where each key is one of the provided feature names and the
              value is a dictionary containing the retrieved information.
    """
    # Normalize the input so that we always have a list of feature names.
    if isinstance(features, str):
        features = [features]

    # Define default preferences
    default_preferences = {
        "shap": True,
        "formula": True,
        "metrics": False,
        "model_info": False,
    }
    if preferences is None:
        preferences = default_preferences
    else:
        # Update missing keys with default values.
        for key, default_val in default_preferences.items():
            if key not in preferences:
                preferences[key] = default_val

    # Define the directories (assumed to be siblings of this file)
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, 'data')
    formulas_dir = os.path.join(current_dir, 'formulas')

    # Load and merge JSON files within each folder.
    shap_and_metrics_data = load_all_json_files(data_dir)
    formula_data = load_all_json_files(formulas_dir)

    # Prepare the result dictionary
    result = {}

    # For each feature the user is interested in:
    for feature in features:
        result_entry = {}
        found = False  # Flag to indicate if we found data for that feature

        # Search for a match in the shap_and_metrics_data dictionary.
        # Here we use a simple case-insensitive check to see if one of the keys
        # in the JSON data appears in the provided feature name.
        for key, details in shap_and_metrics_data.items():
            if key.lower() in feature.lower():
                found = True
                if preferences.get("shap"):
                    result_entry["shap_analysis"] = details.get("shap_analysis", {})
                if preferences.get("metrics"):
                    result_entry["model_metrics"] = details.get("model_metrics", {})
                if preferences.get("model_info"):
                    result_entry["model_info"] = details.get("model_info", {})
                # If you want to handle multiple matches per feature, consider appending 
                # the current result_entry details to a list. For now, we break after the first match.
                break

        # Similarly, search in formula_data for a match.
        for key, details in formula_data.items():
            if key.lower() in feature.lower():
                found = True
                if preferences.get("formula"):
                    result_entry["formula_info"] = {
                        "formula": details.get("formula", ""),
                        "description": details.get("description", "")
                    }
                break

        # If no data was found for the given feature, record an error message.
        if not found:
            result_entry["error"] = f"No data found for feature '{feature}'"
        result[feature] = result_entry

    return result

# Example usage:
if __name__ == "__main__":
    # Example with a single feature.
    feature_info = FeatureInterpreter("MEGNet_MatMiner_1")
    print("Single feature query:")
    print(json.dumps(feature_info, indent=4))

    # Example with multiple features and with all output enabled.
    example_features = ["MEGNet_MatMiner_3", "MEGNet_MatMiner_5", "MEGNet_MatMiner_10"]
    prefs = {"shap": True, "formula": True, "metrics": True, "model_info": True}
    feature_info_list = FeatureInterpreter(example_features, preferences=prefs)
    print("\nMultiple feature query:")
    print(json.dumps(feature_info_list, indent=4))
