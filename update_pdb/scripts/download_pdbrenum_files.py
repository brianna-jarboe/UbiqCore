# Download renumbered mmCIF files from PDBrenum for each PDB ID in the latest update CSV.
import requests
import logging
import sys
import time
import gzip
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_logging(log_dir: Path):
	log_dir.mkdir(parents=True, exist_ok=True)
	log_file = log_dir / f"download_pdbrenum_files_{time.strftime('%Y%m%d_%H%M%S')}.log"
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

def download_and_decompress_file(url: str, dest: Path, session, logger, timeout: int = 120):
	"""Download a gzipped file from URL and decompress to dest."""
	try:
		resp = session.get(url, timeout=timeout)
		if resp.status_code == 200:
			try:
				decompressed = gzip.decompress(resp.content)
				with open(dest, "wb") as f:
					f.write(decompressed)
				logger.info(f"Downloaded and decompressed: {dest.name}")
				return True
			except Exception as e:
				logger.warning(f"Failed to decompress {url}: {e}")
				return False
		else:
			logger.debug(f"Failed to download {url} (status {resp.status_code})")
	except Exception as e:
		logger.debug(f"Exception downloading {url}: {e}")
	return False

def main():
	project_dir = Path(__file__).resolve().parents[2]
	results_dir = project_dir / "util" / "data" / "pdb_updates"
	cache_dir = results_dir / "pdbrenum_file_cache"
	log_dir = results_dir
	cache_dir.mkdir(parents=True, exist_ok=True)
	logger = setup_logging(log_dir)

	csv_path = get_latest_csv(results_dir)
	logger.info(f"Using CSV: {csv_path}")
	pdb_ids = get_pdb_ids_from_csv(csv_path)
	logger.info(f"Found {len(pdb_ids)} PDB IDs")

	session = requests_retry_session()
	pdbrenum_base = "https://dunbrack.fccc.edu/PDBrenum"

	# Prepare download tasks
	tasks = []
	for pdb_id in sorted(pdb_ids):
		pdb_id_lower = pdb_id.lower()
		
		# Asymmetric unit mmCIF (renumbered)
		asym_cif_url = f"{pdbrenum_base}/output_mmCIF/{pdb_id_lower}_renum.cif.gz"
		asym_cif_dest = cache_dir / f"{pdb_id}.cif"
		
		# Biological assembly mmCIF (renumbered, 1st assembly)
		bio_cif_url = f"{pdbrenum_base}/output_mmCIF_assembly/{pdb_id_lower}-assembly1_renum.cif.gz"
		bio_cif_dest = cache_dir / f"{pdb_id}-assembly1.cif"

		if not asym_cif_dest.exists():
			tasks.append((asym_cif_url, asym_cif_dest, pdb_id, "asymmetric"))
		else:
			logger.debug(f"Already cached: {asym_cif_dest.name}")

		if not bio_cif_dest.exists():
			tasks.append((bio_cif_url, bio_cif_dest, pdb_id, "assembly"))
		else:
			logger.debug(f"Already cached: {bio_cif_dest.name}")

	logger.info(f"Downloading {len(tasks)} files using 4 parallel workers...")

	downloaded = 0
	failed = 0
	with ThreadPoolExecutor(max_workers=4) as executor:
		futures = {
			executor.submit(download_and_decompress_file, url, dest, session, logger): (pdb_id, file_type)
			for url, dest, pdb_id, file_type in tasks
		}
		for future in as_completed(futures):
			pdb_id, file_type = futures[future]
			try:
				if future.result():
					downloaded += 1
				else:
					failed += 1
			except Exception as e:
				logger.error(f"Task failed for {pdb_id} ({file_type}): {e}")
				failed += 1

	logger.info(f"PDBrenum download complete. Downloaded: {downloaded}, Failed: {failed}, Total tasks: {len(tasks)}")

if __name__ == "__main__":
	main()
