# Download biological assembly and asymmetric unit mmCIF files for each PDB ID in the latest update CSV.
import requests
import logging
import sys
import time
from pathlib import Path
import pandas as pd

def setup_logging(log_dir: Path):
	log_dir.mkdir(parents=True, exist_ok=True)
	log_file = log_dir / f"download_pdb_files_{time.strftime('%Y%m%d_%H%M%S')}.log"
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	for h in logger.handlers[:]:
		logger.removeHandler(h)
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.INFO)
	fh = logging.FileHandler(log_file)
	fh.setLevel(logging.INFO)
	fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	ch.setFormatter(fmt)
	fh.setFormatter(fmt)
	logger.addHandler(ch)
	logger.addHandler(fh)
	return logger

def requests_retry_session(retries=5, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504), session=None):
	from requests.adapters import HTTPAdapter
	from urllib3.util.retry import Retry
	session = session or requests.Session()
	retry = Retry(
		total=retries,
		read=retries,
		connect=retries,
		backoff_factor=backoff_factor,
		status_forcelist=status_forcelist,
		allowed_methods=frozenset(['GET', 'POST']),
		raise_on_status=False,
	)
	adapter = HTTPAdapter(max_retries=retry)
	session.mount("http://", adapter)
	session.mount("https://", adapter)
	return session

def get_latest_csv(results_dir: Path) -> Path:
	files = sorted(results_dir.glob("pdb_structures_detailed_update_*.csv"), reverse=True)
	if not files:
		raise FileNotFoundError("No pdb_structures_detailed_update_*.csv found in results dir")
	return files[0]

def get_pdb_ids_from_csv(csv_path: Path) -> set:
	df = pd.read_csv(csv_path, usecols=["PDB_ID"])
	return set(df["PDB_ID"].astype(str).str.upper())

def download_file(url: str, dest: Path, session, logger):
	try:
		resp = session.get(url, timeout=60)
		if resp.status_code == 200:
			with open(dest, "wb") as f:
				f.write(resp.content)
			logger.info(f"Downloaded: {dest.name}")
			return True
		else:
			logger.warning(f"Failed to download {url} (status {resp.status_code})")
	except Exception as e:
		logger.warning(f"Exception downloading {url}: {e}")
	return False

def main():
	project_dir = Path(__file__).resolve().parents[2]
	results_dir = project_dir / "util" / "data" / "pdb_updates"
	cache_dir = results_dir / "pdb_file_cache"
	log_dir = results_dir
	cache_dir.mkdir(parents=True, exist_ok=True)
	logger = setup_logging(log_dir)

	csv_path = get_latest_csv(results_dir)
	logger.info(f"Using CSV: {csv_path}")
	pdb_ids = get_pdb_ids_from_csv(csv_path)
	logger.info(f"Found {len(pdb_ids)} PDB IDs")

	session = requests_retry_session()

	for i, pdb_id in enumerate(sorted(pdb_ids)):
		pdb_id_lc = pdb_id.lower()
		# Asymmetric unit mmCIF
		asym_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
		asym_dest = cache_dir / f"{pdb_id}.cif"
		# Biological assembly mmCIF (1st assembly)
		bio_url = f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif"
		bio_dest = cache_dir / f"{pdb_id}-assembly1.cif"

		if not asym_dest.exists():
			logger.info(f"[{i+1}/{len(pdb_ids)}] Downloading asymmetric unit for {pdb_id}")
			download_file(asym_url, asym_dest, session, logger)
		else:
			logger.info(f"[{i+1}/{len(pdb_ids)}] Asymmetric unit already cached: {asym_dest.name}")

		if not bio_dest.exists():
			logger.info(f"[{i+1}/{len(pdb_ids)}] Downloading biological assembly for {pdb_id}")
			download_file(bio_url, bio_dest, session, logger)
		else:
			logger.info(f"[{i+1}/{len(pdb_ids)}] Biological assembly already cached: {bio_dest.name}")

		time.sleep(0.2)  # be nice to RCSB

	logger.info("Download complete.")

if __name__ == "__main__":
	main()
