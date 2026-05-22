#!/bin/bash
# run_weekly_update.sh
# Weekly cron wrapper for UbiqCore E2-E3 updates.

# Navigate to the repo root
cd /home/brianna/apps/ubiqcore || exit 1

# Activate the miniconda environment
source /home/brianna/miniconda3/etc/profile.d/conda.sh
conda activate base # Adjust if the environment is named differently, but looking at miniconda3/envs, base is often what's used if no env is specified

echo "Starting PDB update process at $(date)..."

# Ensure dependencies are installed (optional but good for cron robustness)
pip install -r update_pdb/requirements-updater.txt

# Run the update scripts
echo "Running update_pdb_e2_e3.py..."
python update_pdb/scripts/update_pdb_e2_e3.py

echo "Running download_pdb_files.py..."
python update_pdb/scripts/download_pdb_files.py

echo "Running download_pdbrenum_files.py..."
python update_pdb/scripts/download_pdbrenum_files.py

echo "Update process finished at $(date)."
