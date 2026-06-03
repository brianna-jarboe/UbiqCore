import os
from pathlib import Path

# main dataset file (remain in-repo by default)
MAIN_DATASET_FILE = "util/data/combined_prediction_training_data.csv"

# Support using an external data directory for large, generated files.
# Set the environment variable UBIQCORE_DATA_DIR to point to an external
# folder (example: /home/brianna/data/ubiqcore). When set, the app will
# read cached CIF/PDB files and update CSVs from this external location.
_DEFAULT_EXT_DATA = Path("/home/brianna/data/ubiqcore")
_EXT_DATA = os.environ.get("UBIQCORE_DATA_DIR") or (
	str(_DEFAULT_EXT_DATA) if _DEFAULT_EXT_DATA.exists() else None
)

if _EXT_DATA:
	PDB_FILES = os.path.join(_EXT_DATA, "pdb_files")
	PDB_CIF_FILES = os.path.join(_EXT_DATA, "pdb_file_cache")
	PDBRENUM_CIF_FILES = os.path.join(_EXT_DATA, "pdbrenum_file_cache")
	PDB_STRUCTURE_TABLE_FOLDER = os.path.join(_EXT_DATA, "pdb_updates")
else:
	# Backwards-compatible defaults (repo-local paths)
	PDB_FILES = "util/resources/pdb_files"
	PDB_CIF_FILES = "util/data/pdb_updates/pdb_file_cache"
	PDBRENUM_CIF_FILES = "util/data/pdb_updates/pdbrenum_file_cache"
	PDB_STRUCTURE_TABLE_FOLDER = "util/data/pdb_updates"

# pdb updater script and input folder (repo-local)
PDB_UPDATER = "update_pdb"
