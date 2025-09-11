import os
import re
import json
import pprint
import requests
import tarfile
from pathlib import Path
import tempfile
# For SVG plotting – in a Jupyter environment these will render the SVG.
try:
    from IPython.display import SVG, display
except ImportError:
    SVG = None
    display = None

def print_formula_metrics(info):
    """
    Given a dictionary with evaluation metrics, print them in a nicely formatted way.
    Expected keys (if present) are: rmse, mae, r2 and target_stats.
    """
    if "rmse" in info and "mae" in info and "r2" in info:
        print("Metrics showing the performance of the SISSO approximation to the feature:")
        print(f"  RMSE       : {info['rmse']}")
        print(f"  MAE        : {info['mae']}")
        print(f"  R²         : {info['r2']}")
        if "target_stats" in info and isinstance(info["target_stats"], dict):
            print("Target Stats:")
            for key, value in info["target_stats"].items():
                print(f"    {key:5}: {value}")
    else:
        print("No evaluation metrics available.")

class Interpreter:
    """
    MatterVial feature interpreter for understanding and analyzing extracted features.

    The Interpreter class provides tools to bridge the gap between high-level latent
    representations and interpretable chemical descriptors. It leverages surrogate
    XGBoost models and SHAP analysis to decode the underlying chemical principles
    driving the predictions from MatterVial's featurizers.

    The interpreter supports:
    - Formula retrieval for latent features (ℓ-MM, ℓ-OFM)
    - SISSO formula interpretation
    - SHAP value analysis for feature importance
    - SVG plot visualization for feature decomposition

    This interpretability framework extends to both pretrained models and adjacent
    GNN models trained on-the-fly, enabling users to understand the chemical
    principles behind feature representations.

    Attributes:
        base_dir (str): Base directory for interpreter data files.
        formulas_path (str): Path to the standard formulas JSON file.
        formulas (dict): Loaded standard formulas dictionary.
        sisso_formulas_path (str): Path to SISSO formulas file.
        robust_scaler_path (str): Path to robust scaler parameters file.

    Examples:
        >>> interpreter = Interpreter()
        >>> formula_info = interpreter.get_formula("l-OFM_v1_1")
        >>> shap_data = interpreter.get_shap_values("MEGNet_MatMiner_1")
    """

    def __init__(self, base_dir=None):
        """
        Initialize the MatterVial Interpreter.

        Args:
            base_dir (str, optional): Base directory containing interpreter data files.
                If None, uses the directory of this file. The directory should contain:
                - formulas/all_formulas.json: Standard feature formulas
                - shap_values/: SHAP analysis results
                - ../featurizers/formulas/: SISSO formulas and scaler parameters

        Examples:
            >>> # Default initialization
            >>> interpreter = Interpreter()

            >>> # Custom base directory
            >>> interpreter = Interpreter(base_dir="/path/to/interpreter/data")
        """
        if base_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.base_dir = base_dir

        # Path for standard formulas (for example, MVL16_Bandgap_classifier_MP_2018_1)
        self.formulas_path = os.path.join(self.base_dir, "formulas", "all_formulas.json")
        self.formulas = self._load_formulas(self.formulas_path)

        # Attribute to aditional formulas
        self._additional_formulas = None

        # Paths for SISSO formulas and robust scaler parameters.
        # Adjust these paths if your directory layout is different.
        self.featurizers_dir = os.path.join(os.path.dirname(self.base_dir), "featurizers")
        self.sisso_formulas_path = os.path.join(self.featurizers_dir, "formulas", "SISSO_FORMULAS_v1.txt")
        self.robust_scaler_path = os.path.join(self.featurizers_dir, "formulas", "robust_scaler_mpgap_for_sisso.json")
        self._sisso_formulas = None
        self._robust_scaler_params = None
        # Figshare download configuration
        self.figshare_url = "https://figshare.com/ndownloader/files/57756415"
        self.shap_plots_dir = os.path.join(self.base_dir, "shap_plots")
        self.archive_name = "shap_plots_archive.tar.gz"
        
        
    def _ensure_shap_plots_available(self):
        """
        Ensure SHAP plots are available locally. Download and extract if needed.
        """
        if os.path.exists(self.shap_plots_dir) and os.listdir(self.shap_plots_dir):
            return  # Already available
        else: # create the directory if it doesn't exist
            os.makedirs(self.shap_plots_dir, exist_ok=True)    
        print("SHAP plots not found locally. Downloading from Figshare...")
        self._download_and_extract_shap_plots()
        
    def _download_and_extract_shap_plots(self):
        """
        Download SHAP plots archive from Figshare and extract it.
        """
        try:
            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = os.path.join(temp_dir, self.archive_name)
                
                # Download with progress bar
                response = requests.get(self.figshare_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(archive_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
                
                print(f"\nDownload complete. Extracting to {self.base_dir}...")
                
                # Extract the archive
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(path=self.base_dir)
                
                print("SHAP plots extracted successfully!")
                
        except Exception as e:
            raise RuntimeError(f"Failed to download SHAP plots: {e}")
    
    
    # ------------------------------
    # Standard helper methods
    # ------------------------------
    def _load_formulas(self, file_path):
        """
        Load all standard formulas from the JSON file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Formulas file not found: {file_path}")
        with open(file_path, "r") as f:
            formulas = json.load(f)
        return formulas

    def _generate_name_variants(self, feature_name):
        """
        Generate variants of the feature name that differ only in the zero‐padding at the end.
        For example:
          "MVL16_Bandgap_classifier_MP_2018_1" →
            ["MVL16_Bandgap_classifier_MP_2018_1",
             "MVL16_Bandgap_classifier_MP_2018_01",
             "MVL16_Bandgap_classifier_MP_2018_001"]
        If no trailing digits are found, returns a one-element list.
        """
        variants = [feature_name]
        m = re.search(r"^(.*_)(\d+)$", feature_name)
        if m:
            prefix = m.group(1)
            num_val = int(m.group(2))
            variants = [
                f"{prefix}{num_val}",
                f"{prefix}{num_val:02d}",
                f"{prefix}{num_val:03d}"
            ]
        return variants

    def _find_in_collection(self, feature_name, collection):
        """
        Try the original feature name and its zero‐padded variants in the given dictionary.
        Returns the matching key or None if not found.
        """
        variants = self._generate_name_variants(feature_name)
        for name in variants:
            if name in collection:
                return name
        return None

    def _find_file_with_variants(self, base_dir, feature_name, file_suffix):
        """
        Search for a file within base_dir matching one of feature_name's variants (considering zero‐padding)
        appended with the given file_suffix.
        Returns the full file path if found; otherwise None.
        """
        variants = self._generate_name_variants(feature_name)
        for name in variants:
            filename = f"{name}{file_suffix}"
            file_path = os.path.join(base_dir, filename)
            if os.path.exists(file_path):
                return file_path
        return None

    def _format_formula(self, formula):
        """
        Returns a multi-line version of the formula string by inserting newline characters
        before each plus sign.
        For example, transforms:
          "c0 + a0 * (...) + a1 * (...)"
        into:
          "c0 + a0 * (...) 
           + a1 * (...)"
        """
        # Strip leading/trailing whitespace.
        formula = formula.strip()
        # Split the formula by " + " (assuming this delimiter is used consistently)
        terms = formula.split(" + ")
        # Reconstruct the formula: first term remains on the same line; others get a preceding newline and plus.
        if len(terms) > 1:
            formatted = terms[0] + " + " + "\n+ ".join(terms[1:])
        else:
            formatted = terms[0]
        return formatted

    def _round_formula_numbers(self, formula):
        """
        Rounds every float number in the formula string to two decimals.
        This helps reduce the length of the expression when large numbers are involved.
        """
        # The regex finds numbers with a decimal point (including negative numbers).
        pattern = re.compile(r"(-?\d+\.\d+)")
        rounded_formula = pattern.sub(lambda m: f"{float(m.group(0)):.2f}", formula)
        return rounded_formula

    def _make_compformula_2d(self, entry):
        """
        Generate a 2-decimal-substituted version of the formula with proper operator signs.
        If a constant's value is lower than 0.1 (in absolute value), it will be formatted in
        scientific notation with one digit in the significand (e.g., 0.0234532 -> '2.3e-2').
        """
        import re

        def _fix_constant_naming(constants):
            fixed_constants = {}
            for k, v in constants.items():
                if re.match(r"a\d+", k):
                    idx = int(k[1:]) - 1
                    fixed_constants[f"a{idx}"] = v
                else:
                    fixed_constants[k] = v
            return fixed_constants

        if not isinstance(entry, dict) or "formula" not in entry or "constants" not in entry:
            raise ValueError("Invalid formula entry")

        formula = entry["formula"]
        constants = _fix_constant_naming(entry["constants"])

        # Format constants: use scientific notation if the absolute value is lower than 0.1
        rounded_constants = {}
        for name, val in constants.items():
            val_num = float(val)
            if abs(val_num) < 0.1:
                # Format using scientific notation with one decimal digit
                s = f"{val_num:.1e}"
                parts = s.split("e")
                # Convert exponent to int to remove any leading zeros (e.g., 'e-02' becomes 'e-2')
                exponent = int(parts[1])
                s = f"{parts[0]}e{exponent}"
            else:
                s = f"{val_num:.2f}"
            rounded_constants[name] = s

        sorted_keys = sorted(rounded_constants.keys(), key=len, reverse=True)

        # Replace constants in the formula
        for const in sorted_keys:
            val = rounded_constants[const]
            formula = re.sub(rf"\b{re.escape(const)}\b", val, formula)

        # Clean up '+ -' → '- '
        formula = re.sub(r'\+\s*-', '- ', formula)

        return formula

    def _find_best_dimension(self, entry):
        """
        Find the dimension with the lowest max_AE (maximum absolute error).
        Returns the dimension key and the corresponding data.
        """
        best_dim = None
        best_mae = float('inf')
        
        for dim_key, dim_data in entry.items():
            if dim_key.startswith('dim_') and isinstance(dim_data, dict):
                if 'max_AE' in dim_data and dim_data['max_AE'] < best_mae:
                    best_mae = dim_data['max_AE']
                    best_dim = dim_key
        
        return best_dim, entry[best_dim] if best_dim else None

    # ------------------------------
    # get_formula – Standard and SISSO with Robust Scaler Extraction
    # ------------------------------
    def get_formula(self, feature_name, dimension=None, verbose=False, additional_formula_file=None): # Added additional_formula_file  
        """  
        Returns the formula information for the given feature name.  
        For multi-dimensional formulas, defaults to the dimension with lowest max_AE,  
        but accepts specific dimension requests via the dimension parameter.  
          
        Args:  
            feature_name (str): The name of the feature  
            dimension (int, optional): Specific dimension to retrieve (1, 2, 3, 4, 5)  
            verbose (bool, optional): If True, prints detailed information. Defaults to False.  
            additional_formula_file (str, optional): Path to an additional JSON file to search if the feature is not found in the primary formulas.  
        """  
        if feature_name.startswith("SISSO"):  
            # ... (SISSO logic remains the same) ...  
            if self._sisso_formulas is None:  
                self._sisso_formulas = self._load_sisso_formulas()  
            if feature_name in self._sisso_formulas:  
                raw_formula = self._sisso_formulas[feature_name]  
                # Extract feature names from expressions like df["..."]  
                extracted_features = re.findall(r'df\["([^"]+)"\]', raw_formula)  
                robust_scaler_data = {}  
                robust_params = self.get_robust_scaler_params()  
                for feat in extracted_features:  
                    robust_scaler_data[feat] = robust_params.get(feat, "NOT FOUND")  
                formatted_formula = self._format_formula(raw_formula)  
  
                if verbose:  
                    print(f"SISSO formula for {feature_name} retrieved successfully:")  
                    print("\n--- Formatted SISSO Formula ---")  
                    print(formatted_formula)  
                    print("\n--- Base Features ---")  
                    print("\n".join(extracted_features))  
                    print("\n--- Scaler Data for Base Features ---")  
                    for feat, data in robust_scaler_data.items():  
                        print(f"{feat} : {pprint.pformat(data)}")  
                  
                return {  
                    "formula": raw_formula,  
                    "formatted_formula": formatted_formula,  
                    "features": extracted_features,  
                    "robust_scaler_data": robust_scaler_data  
                }  
            else:  
                raise ValueError(f"SISSO formula not found for feature ID '{feature_name}'.")  
        else:  
            key = self._find_in_collection(feature_name, self.formulas)  
            target_collection = self.formulas # Start with the primary collection  
  
            if key is None and additional_formula_file:  
                # If not found in primary, try loading and searching the additional file  
                if self._additional_formulas is None or self._additional_formulas_path != additional_formula_file:  
                    # Load only if not already loaded or if a different file is specified  
                    try:  
                        self._additional_formulas = self._load_formulas(additional_formula_file)  
                        self._additional_formulas_path = additional_formula_file # Store path to avoid reloading  
                    except FileNotFoundError as e:  
                        print(f"Warning: Additional formula file not found: {e}. Skipping search in this file.")  
                        self._additional_formulas = None # Reset to ensure it's not used if file not found  
                  
                if self._additional_formulas:  
                    key = self._find_in_collection(feature_name, self._additional_formulas)  
                    if key is not None:  
                        target_collection = self._additional_formulas # Switch to the additional collection  
  
            if key is not None:  
                entry = target_collection[key] # Use the determined target_collection  
                  
                # ... (Rest of the standard formula logic remains the same) ...  
                # Check if this is a multi-dimensional formula  
                if isinstance(entry, dict) and any(k.startswith('dim_') for k in entry.keys()):  
                    if dimension is not None:  
                        # User requested specific dimension  
                        requested_dim = f"dim_{dimension}"  
                        if requested_dim in entry:  
                            selected_entry = entry[requested_dim]  
                            if verbose:  
                                print(f"Standard formula for {feature_name} (dimension {dimension}) retrieved successfully:")  
                        else:  
                            available_dims = [k.replace('dim_', '') for k in entry.keys() if k.startswith('dim_')]  
                            raise ValueError(f"Dimension {dimension} not found for feature '{feature_name}'. Available dimensions: {available_dims}")  
                    else:  
                        # Default to best dimension (lowest max_AE)  
                        best_dim, selected_entry = self._find_best_dimension(entry)  
                        if selected_entry is None:  
                            raise ValueError(f"No valid dimensions found for feature '{feature_name}'")  
                        dim_num = best_dim.replace('dim_', '')  
                        if verbose:  
                            print(f"Standard formula for {feature_name} (best dimension: {dim_num}, max_AE: {selected_entry.get('max_AE', 'N/A')}) retrieved successfully:")  
                else:  
                    # Single formula (legacy format)  
                    if dimension is not None and verbose:  
                        print(f"Warning: Dimension parameter ignored for single-formula feature '{feature_name}'")  
                    selected_entry = entry  
                    if verbose:  
                        print(f"Standard formula for {feature_name} retrieved successfully:")  
                  
                # Process the selected entry  
                if verbose:  
                    print("--- SISSO estimated formula ---")  
                  
                if isinstance(selected_entry, dict) and "formula" in selected_entry:  
                    selected_entry["formatted_formula"] = self._format_formula(selected_entry["formula"])  
                  
                # Print the formatted formula directly.  
                if verbose:  
                    if isinstance(selected_entry, dict) and "formatted_formula" in selected_entry:  
                        print(selected_entry["formatted_formula"])  
                    else:  
                        print(selected_entry)  
                      
                if verbose and isinstance(selected_entry, dict) and "constants" in selected_entry:  
                    print("\n--- Constants ---")  
                    for const_name, const_value in selected_entry["constants"].items():  
                        print(f"{const_name}: {const_value}")  
                          
                if verbose:  
                    print("\n--- Evaluation Metrics ---")  
                    print_formula_metrics(selected_entry)  
                  
                if isinstance(selected_entry, dict) and "formula" in selected_entry and "constants" in selected_entry:  
                    selected_entry["compformula2d"] = self._make_compformula_2d(selected_entry)  
                    selected_entry["compformula2d"] = re.sub(r"\+\s*(-[\d.]+)", r"- \1", selected_entry["compformula2d"])  
                      
                return selected_entry  
            else:  
                raise ValueError(f"Feature '{feature_name}' (or any valid zero-padded variant) not found in standard formulas or the additional formula file.")

    def _load_sisso_formulas(self):
        """
        Reads the SISSO master formulas file and builds a dictionary mapping SISSO feature IDs
        to their Python expressions.
        Expected file format:
          # Dataset: <dataset_name>
          Feature 1: <Python expression>
          Feature 2: <Python expression>
          ...
        The keys are formatted as:
          SISSO_<dataset_name>_<feature_number>
        Any '^' characters are replaced with '**'.
        """
        if not os.path.exists(self.sisso_formulas_path):
            raise FileNotFoundError(f"SISSO formulas file not found: {self.sisso_formulas_path}")
        sisso_dict = {}
        current_dataset = None
        counter = 0
        with open(self.sisso_formulas_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("# Dataset:"):
                    current_dataset = stripped.split(":", 1)[1].strip()
                    counter = 0  # Reset counter for the new dataset
                elif stripped.startswith("Feature"):
                    if current_dataset is None:
                        continue  # Skip if not within a dataset section.
                    parts = stripped.split(":", 1)
                    if len(parts) != 2:
                        continue
                    counter += 1
                    feature_id = f"SISSO_{current_dataset}_{counter}"
                    formula = parts[1].strip().replace("^", "**")
                    sisso_dict[feature_id] = formula
        if not sisso_dict:
            raise ValueError("No SISSO formulas were found in the SISSO formulas file.")
        return sisso_dict

    # ------------------------------
    # SVG and SHAP methods
    # ------------------------------
    def display_svg(self, feature_name, plot_type="shap_summary"):
        """
        Displays the SVG plot for the feature.
        
        If SHAP plots are not available locally, they will be automatically 
        downloaded from Figshare (~160MB compressed, ~531MB extracted).
        
        Args:
            feature_name (str): Name of the feature to visualize
            plot_type (str): Type of plot ('shap_summary' or 'shap_waterfall')
            
        Note:
            First-time usage will trigger a download of SHAP plots from Figshare.
            Subsequent calls will use the cached local files.
        """
        # Ensure SHAP plots are available
        self._ensure_shap_plots_available()
        # Find the SVG file
        prefix = feature_name.split("_")[0]
        svg_dir = os.path.join(self.base_dir, "shap_plots", prefix)
        file_suffix = f"_{plot_type}.svg"
        svg_path = self._find_file_with_variants(svg_dir, feature_name, file_suffix)
        if svg_path is None:
            raise FileNotFoundError(f"SVG file not found for feature '{feature_name}' in {svg_dir}")
        if SVG is not None and display is not None:
            display(SVG(filename=svg_path))
        else:
            print(f"SVG file is available at: {svg_path}")

    def get_shap_values(self, feature_name, shap_type="top_features"):
        """
        Load and return the SHAP top features for the given feature by checking different variants.
        """
        prefix = feature_name.split("_")[0]
        shap_dir = os.path.join(self.base_dir, "shap_values", prefix)
        file_suffix = f"_{shap_type}.json"
        shap_path = self._find_file_with_variants(shap_dir, feature_name, file_suffix)
        if shap_path is None:
            raise FileNotFoundError(f"SHAP file not found for feature '{feature_name}' in {shap_dir}")
        with open(shap_path, "r") as f:
            shap_data = json.load(f)
        return shap_data

    def get_robust_scaler_params(self):
        """
        Loads and returns the robust scaler normalization parameters from the JSON file.
        """
        if self._robust_scaler_params is None:
            if not os.path.exists(self.robust_scaler_path):
                raise FileNotFoundError(f"Robust scaler file not found: {self.robust_scaler_path}")
            with open(self.robust_scaler_path, "r") as f:
                self._robust_scaler_params = json.load(f)
        return self._robust_scaler_params

# ------------------------------
# Example usage:
# ------------------------------
if __name__ == "__main__":
    interpreter = Interpreter()

    # Standard formula lookup.
    feature_standard = "MVL16_Bandgap_classifier_MP_2018_1"
    try:
        standard_formula = interpreter.get_formula(feature_standard)
    except ValueError as e:
        print(e)

    # Multi-dimensional formula lookup (default to best)
    try:
        best_formula = interpreter.get_formula("l-OFM_v1_1")
    except ValueError as e:
        print(e)

    # Multi-dimensional formula lookup (specific dimension)
    try:
        dim3_formula = interpreter.get_formula("l-OFM_v1_1", dimension=3)
    except ValueError as e:
        print(e)

    # SISSO formula lookup.
    sisso_feature = "SISSO_matbench_dielectric_1"
    try:
        sisso_formula = interpreter.get_formula(sisso_feature)
    except ValueError as e:
        print(e)

    # Attempt to display the SVG plot.
    svg_feature = feature_standard  # Using the same standard feature as an example.
    try:
        print(f"\nAttempting to display the SVG plot for {svg_feature} ...")
        interpreter.display_svg(svg_feature, plot_type="shap_summary")
    except Exception as e:
        print(e)