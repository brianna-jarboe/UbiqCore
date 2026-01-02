from pathlib import Path
import os
import glob
import pandas as pd
from datetime import datetime



def get_latest_pdb_table(folder_path):
    # Get list of CSV files matching the specific pattern
    pattern = str(Path(folder_path) / "pdb_structures_detailed_update_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        return "N/A", None

    # Find the latest file based on the timestamp in the filename
    latest_file = max(csv_files, key=lambda x: Path(x).stem.split('_')[-1])
    
    # Extract timestamp from filename (format: YYYYMMDD)
    timestamp_str = Path(latest_file).stem.split('_')[-1]
    try:
        date_obj = datetime.strptime(timestamp_str, '%Y%m%d')
        date_time = date_obj.strftime('%Y-%m-%d')
    except ValueError:
        # Fallback to file modification time if timestamp parsing fails
        mod_time = os.path.getmtime(latest_file)
        date_time = pd.to_datetime(mod_time, unit='s').strftime('%Y-%m-%d')
    
    return date_time, latest_file
