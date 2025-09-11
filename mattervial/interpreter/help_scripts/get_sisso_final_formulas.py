"""
This script scans the base folder "sisso_calcs" for subfolders of the form
    sisso_calc_<feature>
Each folder is expected to have:
  • a models/ subfolder (if the calculation ran)
  • either a JSON mapping file named feature_mapping_<feature>.json
    or a global feature_mapping.json in the sisso_calcs/ directory.

Inside models/ there are several model files (e.g. "train_dim_1_model_0.dat", …)
whose header lines include:
  – The regression formula (first header line)
  – A line with statistics including "RMSE:" and "Max AE:".

After the header, sample data is printed with columns including “Property Value”
and “Property Value (EST)”. This script:
  • Processes *all* model files in each sisso_calc folder
  • Substitutes feature tokens in the symbolic formula using the mapping file.
  • Computes an explicit formula ("compformula") by replacing constant tokens
    (c0, a0, a1, …) with their corresponding numerical values.
  • Extracts RMSE and Max AE from the header.
  • Reads the sample data to compute additional metrics: Mean Absolute Error (MAE)
    and R²; plus it also computes statistics for the target property – mean, std, min, max.

All results are written to an "all_formulas.json" file inside the sisso_calcs folder.
"""

import os
import glob
import re
import json
import math
from collections import OrderedDict

def natural_keys(text):
    """
    Split text into list of strings and ints for natural sorting.
    e.g. "train_dim_2" → ["train_dim_", 2, ""]
    """
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r'(\d+)', text)
    ]

# Determine script directory so we can load mappings from here:
# Determine script directory so we can load mappings from here:
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

base_dir = "sisso_calcs"
all_formulas = {}

# 1) Find all sisso_calc_<feature> folders
sisso_folders = glob.glob(os.path.join(base_dir, "sisso_calc_*"))
if not sisso_folders:
    print("No sisso_calc_ folders found in", base_dir)
    exit(1)

for folder_path in sisso_folders:
    folder_name  = os.path.basename(folder_path)
    feature_name = folder_name[len("sisso_calc_"):]
    print("\nProcessing feature:", feature_name)

    # 2) Load mapping JSON from script_dir
    mapping_file = f"feature_mapping_{feature_name}.json"
    mapping_path = os.path.join(script_dir, mapping_file)
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r") as mf:
                feature_mapping = json.load(mf)
            print("  Loaded mapping:", mapping_path,
                  f"({len(feature_mapping)} entries)")
        except Exception as e:
            print("  ❌ Error loading mapping file", mapping_path, e)
            feature_mapping = {}
    else:
        print("  ❌ Mapping file not found:", mapping_path)
        feature_mapping = {}

    # 3) Collect all model files
    models_dir = os.path.join(folder_path, "models")
    if not os.path.isdir(models_dir):
        print("  Skipping", folder_name, "(no models/ folder)")
        continue

    model_files = glob.glob(os.path.join(models_dir,
                                         "train_dim_*_model_*.dat"))
    if not model_files:
        print("  No model files found in", models_dir)
        continue

    # 4) Process each model file, keyed by dim_<N>
    models_info = OrderedDict()
    for mfile in sorted(model_files,
                        key=lambda p: natural_keys(os.path.basename(p))):
        mname = os.path.basename(mfile)
        # parse the dimension number
        mo = re.search(r"train_dim_(\d+)_model", mname)
        dim_key = f"dim_{mo.group(1)}" if mo else mname
        print("  Processing", mname, "→", dim_key)

        # --- a) read header lines ---
        header_lines = []
        with open(mfile, "r") as f:
            for line in f:
                if line.startswith("#"):
                    header_lines.append(line.strip())
                else:
                    break

        # --- b) extract RMSE & Max AE ---
        local_rmse = None
        max_ae     = None
        for h in header_lines:
            if "RMSE:" in h:
                m = re.search(r"RMSE:\s*([\d\.\-eE]+)", h)
                if m: local_rmse = float(m.group(1))
            if "Max AE:" in h:
                m = re.search(r"Max AE:\s*([\d\.\-eE]+)", h)
                if m: max_ae = float(m.group(1))
        if max_ae is None:
            print("    ❌ No Max AE found; skipping", mname)
            continue

        # --- c) find formula_line & constants_line ---
        formula_line   = None
        constants_line = ""
        for h in header_lines:
            txt = h.lstrip("# ").rstrip()
            if formula_line is None and txt:
                formula_line = txt
            if txt.lower().startswith("all"):
                constants_line = txt
            if formula_line and constants_line:
                break

        if not formula_line:
            print("    ❌ No formula line; skipping", mname)
            continue

        # --- d) parse constants_line into c0, a1, a2, … ---
        parts = [p.strip() for p in constants_line.split(",") if p.strip()]
        if parts and parts[0].lower().startswith("all"):
            const_vals = parts[1:]
        else:
            const_vals = parts

        const_mapping = {}
        if const_vals:
            const_mapping["c0"] = const_vals[0]
            for i, v in enumerate(const_vals[1:], start=1):
                const_mapping[f"a{i}"] = v

        # --- e) substitute feature tokens in the formula ---
        def feature_repl(m):
            tok = m.group(0)
            return feature_mapping.get(tok, tok)

        final_formula = re.sub(r"\bfeature_\d+\b",
                               feature_repl,
                               formula_line)

        # --- f) build explicit compformula by replacing c0, aX tokens ---
        def const_repl(m):
            tok = m.group(0)
            if tok == "c0":
                return const_mapping.get("c0", tok)
            mi = re.match(r"a(\d+)", tok)
            if mi:
                idx = int(mi.group(1)) + 1
                return const_mapping.get(f"a{idx}", tok)
            return tok

        compformula = re.sub(r"\b(c0|a\d+)\b",
                             const_repl,
                             final_formula)

        # --- g) read sample data, compute MAE, R², target stats ---
        y_true = []
        y_est  = []
        data_started = False
        with open(mfile, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    data_started = True
                if data_started and line.strip():
                    cols = [p.strip() for p in line.split(",")]
                    if len(cols) >= 3:
                        try:
                            y_true.append(float(cols[1]))
                            y_est .append(float(cols[2]))
                        except ValueError:
                            pass

        mae = None
        r2  = None
        if y_true and y_est:
            errs = [abs(t-e) for t,e in zip(y_true, y_est)]
            mae = sum(errs) / len(errs)
            μ   = sum(y_true) / len(y_true)
            ss_tot = sum((t-μ)**2 for t in y_true)
            ss_res = sum((t-e)**2 for t,e in zip(y_true, y_est))
            r2     = 1 - (ss_res/ss_tot) if ss_tot else (1.0 if ss_res==0 else 0.0)

        target_stats = {}
        if y_true:
            n   = len(y_true)
            μ   = sum(y_true)/n
            σ   = math.sqrt(sum((t-μ)**2 for t in y_true)/n)
            target_stats = {
                "mean": μ,
                "std":  σ,
                "min":  min(y_true),
                "max":  max(y_true)
            }

        # --- h) store this dimension’s info ---
        models_info[dim_key] = {
            "formula":      final_formula,
            "compformula":  compformula,
            "constants":    const_mapping,
            "max_AE":       max_ae,
            "rmse":         local_rmse,
            "mae":          mae,
            "r2":           r2,
            "target_stats": target_stats
        }

        print(f"    » {dim_key}: formula={final_formula}, compformula={compformula}")

    # end model loop

    all_formulas[feature_name] = models_info
    print(f"  Collected {len(models_info)} model(s) for feature '{feature_name}'")

# 5) sort features naturally and write out JSON
ordered = OrderedDict(
    sorted(all_formulas.items(), key=lambda kv: natural_keys(kv[0]))
)
out_path = os.path.join(base_dir, "all_formulas.json")
with open(out_path, "w") as outf:
    json.dump(ordered, outf, indent=4)

print("\nFinal formulas for all features written to", out_path)
