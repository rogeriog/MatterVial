"""
This script is to prepare sisso calculations after the shap decomposition
and pca-based sampling of the base dataset. 
"""

import os
import glob

# Base directory for the calculations; adjust if needed.
base_dir = "sisso_calcs"
os.makedirs(base_dir, exist_ok=True)

# 1. Find CSV files in the current folder that match the pattern.
#    For example: "SISSO_sampled_MVL32_Bandgap_classifier_MP_2018_1.csv"
csv_files = glob.glob("SISSO_sampled_*.csv")

if not csv_files:
    print("No CSV files matching the pattern were found!")
    exit(1)

# 2. Build a mapping from feature name to CSV file name.
#    Extract the feature name by removing the prefix "SISSO_sampled_" and suffix ".csv".
features = {}
for csv_file in csv_files:
    if csv_file.startswith("SISSO_sampled_") and csv_file.endswith(".csv"):
        feature_name = csv_file[len("SISSO_sampled_"):-len(".csv")]
        features[feature_name] = csv_file

print(f"Found {len(features)} CSV files to process:")
for feat, csv_file in features.items():
    print(f"  - {feat} -> {csv_file}")
print()

# 3. JSON template - using double braces to escape them, only {csv_file} will be substituted
json_template = """{{
   "data_file": "../../{csv_file}",
   "property_key": "target",
   "desc_dim": 5,
   "n_sis_select": 15,
   "max_rung": 2,
   "n_residual": 2,
   "calc_type": "regression",
   "min_abs_feat_val": 1e-05,
   "max_abs_feat_val": 100000000.0,
   "n_models_store": 1,
   "leave_out_frac": 0,
   "leave_out_inds": [],
   "opset": [
      "add",
      "sub",
      "abs_diff",
      "mult",
      "div",
      "inv",
      "abs",
      "exp",
      "log",
      "sq",
      "cb",
      "six_pow",
      "sqrt",
      "cbrt",
      "neg_exp"
   ],
   "data_file_relative_to_json": true
}}"""

# 4. Create subdirectories and write the sisso.json file in each subfolder.
print("Setting up calculation directories...")
for feat, csv_file in features.items():
    folder_name = "sisso_calc_" + feat
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    json_path = os.path.join(folder_path, "sisso.json")

    # Write the JSON file if it isn't already there.
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            f.write(json_template.format(csv_file=csv_file))
        print(f"  ✓ Created {json_path}")
    else:
        print(f"  → {json_path} already exists, skipping creation")

print()

# 5. Check which jobs have already been processed and which are pending.
print("Checking job status...")
processed_jobs = []
pending_jobs = []

for feat in features.keys():
    folder_name = "sisso_calc_" + feat
    folder_path = os.path.join(base_dir, folder_name)
    models_path = os.path.join(folder_path, "models")

    if os.path.exists(models_path):
        processed_jobs.append(folder_name)
        print(f"  ✓ {folder_name} - ALREADY PROCESSED (models folder exists)")
    else:
        pending_jobs.append(folder_name)
        print(f"  ⏳ {folder_name} - PENDING")

print()
print(f"Summary: {len(processed_jobs)} already processed, {len(pending_jobs)} pending")

if not pending_jobs:
    print("\n🎉 All jobs have already been processed! No new calculations needed.")
    exit(0)

if processed_jobs:
    print(f"\n📋 Already processed jobs ({len(processed_jobs)}):")
    for job in sorted(processed_jobs):
        print(f"  - {job}")

print(f"\n🚀 Jobs to be scheduled ({len(pending_jobs)}):")
for job in sorted(pending_jobs):
    print(f"  - {job}")

# Optional: sort the pending jobs by name.
pending_jobs.sort()
# import sys; sys.exit()
# 6. Divide the pending jobs into 10 groups.
num_groups = 10
groups = [[] for _ in range(num_groups)]
for i, folder in enumerate(pending_jobs):
    groups[i % num_groups].append(folder)

print(f"\n📦 Distributing {len(pending_jobs)} pending jobs across {num_groups} run files...")

# 7. Create 10 run files. Each file will have its own SBATCH header with job-name set to sisso_run#.
created_files = []
for idx, group in enumerate(groups):
    if not group:  # Skip empty groups
        continue

    # Name the shell file with 2-digit numbering (01, 02, …)
    run_filename = f"run_sisso_{idx+1:02d}.sh"
    run_filepath = os.path.join(base_dir, run_filename)

    # Create a header with a job-name that depends on the group number.
    SBATCH_header = f"""#!/bin/bash
#
#SBATCH --job-name=sisso_run{idx+1}             # Job name
#SBATCH --output=%j_run_sisso{idx+1}.out                # Output file (%j appends the job ID)
#SBATCH --partition=shared                       
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=24              
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G                        
#SBATCH --time=18:00:00                   
#SBATCH --account=htforft                

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"

# Load necessary modules (adjust versions as needed)
module purge
module load EasyBuild/2024a OpenMPI/5.0.3-GCC-13.3.0 FlexiBLAS/3.4.4-GCC-13.3.0

# Activate your conda environment (change the path if necessary)
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh

# Change to the working directory where your scripts/data reside
cd /gpfs/home/acad/ucl-modl/rgouvea/scratch/matbench_tests/sampled_data_for_sisso/sisso_calcs
###########################################################################
echo "Starting batch execution of sisso++"
# Activate conda environment for sisso run
conda activate sissopp_env

"""
    with open(run_filepath, "w") as shf:
        shf.write(SBATCH_header)
        shf.write("\n")
        for folder in group:
            shf.write(f"echo 'Entering folder: {folder}'\n")
            shf.write(f"cd {folder}\n")
            shf.write("LD_PRELOAD=/gpfs/softs/easybuild/2024a/software/GCCcore/13.3.0/lib64/libstdc++.so.6 mpiexec -np 24 sisso++ sisso.json\n")
            shf.write("cd - > /dev/null\n\n")
        shf.write("echo 'All tasks completed.'\n")
    os.chmod(run_filepath, 0o755)
    created_files.append((run_filepath, len(group)))
    print(f"  ✓ {run_filename} - {len(group)} jobs")

print(f"\n📄 Created {len(created_files)} run files:")
for filepath, job_count in created_files:
    print(f"  - {os.path.basename(filepath)} ({job_count} jobs)")

print(f"\n🎯 Ready to submit! Use: sbatch <run_file>")
