import requests
import json
import time
import pandas as pd
from pathlib import Path
import logging
import sys
import datetime
from typing import List, Dict, Set, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # <-- fix: use urllib3.util.retry, not urllib3.util_retry
# --- Shared helpers ---------------------------------------------------------

def setup_logging(results_dir: Path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"pdb_update_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # clear handlers
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
    return logger, log_file

def requests_retry_session(
    retries=5,
    backoff_factor=0.5,
    status_forcelist=(500, 502, 503, 504),
    session=None,
):
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

def read_uniprot_ids(file_path: Path) -> List[str]:
    ids: List[str] = []
    with open(file_path, "r") as f:
        for line in f:
            clean = line.strip()
            if clean and not clean.startswith("//"):
                ids.append(clean)
    return ids

# --- Constants ---------------------------------------

UBIQUITIN_UNIPROT_IDS: Set[str] = {"P0CG47", "P0CG48", "P62988"}
UBIQUITIN_GENE_NAMES: Set[str] = {"UBB", "UBC"}

# --- UniProt → PDB mapping ---------------------

def map_uniprot_to_pdb_uniprot_service(uniprot_ids: List[str], batch_size: int = 50) -> Dict[str, Set[str]]:
    """Current UniProt mapping: UniProtKB_AC-ID -> PDB."""
    mapping_url = "https://rest.uniprot.org/idmapping/run"
    sess = requests_retry_session()
    up_to_pdb: Dict[str, Set[str]] = {u: set() for u in uniprot_ids}

    chunk_size = min(batch_size, 100)
    for i in range(0, len(uniprot_ids), chunk_size):
        chunk = uniprot_ids[i:i + chunk_size]
        logging.info(f"UniProt mapping batch {i//chunk_size + 1}: {len(chunk)} IDs")

        params = {
            "from": "UniProtKB_AC-ID",
            "to": "PDB",
            "ids": ",".join(chunk),
        }
        try:
            resp = sess.post(mapping_url, data=params, timeout=60)
            resp.raise_for_status()
            job_id = resp.json()["jobId"]
        except Exception as e:
            logging.error(f"Failed to submit UniProt mapping job: {e}")
            continue

        status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
        for _ in range(30):
            st = sess.get(status_url, timeout=30)
            st.raise_for_status()
            st_data = st.json()
            if "results" in st_data:
                stream_url = f"https://rest.uniprot.org/idmapping/stream/{job_id}"
                rs = sess.get(stream_url, timeout=60)
                rs.raise_for_status()
                rs_json = rs.json()
                if "results" in rs_json:
                    for r in rs_json["results"]:
                        frm = r.get("from")
                        to = r.get("to")
                        if not frm or not to:
                            continue
                        pdb_id = to.split(":")[0].upper()
                        if frm in up_to_pdb:
                            up_to_pdb[frm].add(pdb_id)
                break
            if st_data.get("jobStatus") == "RUNNING":
                time.sleep(2)
                continue
            logging.error(f"UniProt mapping job failed: {st_data}")
            break

        time.sleep(1)

    return up_to_pdb

def map_uniprot_to_pdb_rcsb_fulltext(uniprot_ids: List[str]) -> Dict[str, Set[str]]:
    """
    Additional mapping via RCSB search API, using full-text query
    as pdb_query_offline.search_pdb_for_uniprot, but called from here.

    For each UniProt ID:
      - full-text search for the ID string,
      - collect entry identifiers (PDB IDs).
    """
    base_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    sess = requests_retry_session()
    up_to_pdb: Dict[str, Set[str]] = {u: set() for u in uniprot_ids}

    for idx, uid in enumerate(uniprot_ids):
        logging.info(f"RCSB full-text mapping {idx+1}/{len(uniprot_ids)}: {uid}")

        query = {
            "query": {
                "type": "terminal",
                "label": "full_text",
                "service": "full_text",
                "parameters": {
                    "value": uid
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": 100
                },
                "results_content_type": ["experimental"],
                "sort": [
                    {
                        "sort_by": "score",
                        "direction": "desc"
                    }
                ],
                "scoring_strategy": "combined"
            }
        }

        try:
            r = sess.post(base_url, json=query, timeout=20)
            # RCSB may return 204 for no hits; treat as empty
            if r.status_code == 204:
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logging.error(f"RCSB full-text mapping for {uid} failed: {e}")
            continue

        for item in data.get("result_set", []):
            pdb_id = item.get("identifier")
            if pdb_id:
                up_to_pdb[uid].add(pdb_id.upper())

        # Be nice to the server
        time.sleep(1.0)

    return up_to_pdb

def map_uniprot_to_pdb(uniprot_ids: List[str], batch_size: int = 50) -> Dict[str, Set[str]]:
    """
    Combined mapping:
    - UniProt ID mapping service (primary, curated).
    - RCSB full-text search for the UniProt string (secondary, broad).
    Returns union of both per UniProt ID.
    """
    logging.info("Starting UniProt→PDB mapping via UniProt service...")
    up_uniprot = map_uniprot_to_pdb_uniprot_service(uniprot_ids, batch_size=batch_size)

    logging.info("Starting UniProt→PDB mapping via RCSB full-text search...")
    up_rcsb = map_uniprot_to_pdb_rcsb_fulltext(uniprot_ids)

    combined: Dict[str, Set[str]] = {u: set() for u in uniprot_ids}
    for uid in uniprot_ids:
        combined[uid].update(up_uniprot.get(uid, set()))
        combined[uid].update(up_rcsb.get(uid, set()))

    logging.info("Combined mapping complete")
    return combined

# --- JSON detail retrieval via RCSB Entry Polymer Entity -------------------

def fetch_pdb_details_json(pdb_ids: List[str]) -> List[Dict]:
    """
    Fetch entry-level details for given PDB IDs via RCSB Entry Polymer Entity API.
    Uses: https://data.rcsb.org/rest/v1/core/entry/{entry_id}
          https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}
    """
    sess = requests_retry_session()
    entries: List[Dict] = []

    for i, pdb_id in enumerate(pdb_ids):
        if i % 50 == 0:
            logging.info(f"JSON API: {i+1}/{len(pdb_ids)} {pdb_id}")

        try:
            # 1) Entry-level info
            entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            er = sess.get(entry_url, timeout=30)
            if er.status_code != 200:
                logging.debug(f"Entry JSON {pdb_id} status {er.status_code}")
                continue
            entry_json = er.json()

            # 2) Determine polymer entity IDs from entry summary
            #    rcsb_entry_container_identifiers.polymer_entity_ids
            pe_ids = (entry_json.get("rcsb_entry_container_identifiers", {})
                                 .get("polymer_entity_ids") or [])

            polymer_entities: List[Dict] = []
            for ent_id in pe_ids:
                pe_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{ent_id}"
                pr = sess.get(pe_url, timeout=30)
                if pr.status_code != 200:
                    logging.debug(f"Polymer entity JSON {pdb_id}/{ent_id} status {pr.status_code}")
                    continue
                pe_json = pr.json()
                polymer_entities.append(pe_json)

            # 3) Nonpolymer entities via nonpolymer_entity endpoint
            #    rcsb_entry_container_identifiers.nonpolymer_entity_ids
            np_ids = (entry_json.get("rcsb_entry_container_identifiers", {})
                                  .get("nonpolymer_entity_ids") or [])
            nonpolymer_entities: List[Dict] = []
            for np_id in np_ids:
                np_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{np_id}"
                nr = sess.get(np_url, timeout=30)
                if nr.status_code != 200:
                    logging.debug(f"Nonpolymer entity JSON {pdb_id}/{np_id} status {nr.status_code}")
                    continue
                np_json = nr.json()
                nonpolymer_entities.append(np_json)

            # Build a structure that looks like GraphQL's entry object
            entries.append({
                "entry": {
                    "id": pdb_id,
                    "rcsb_accession_info": {
                        "initial_release_date": entry_json.get("rcsb_accession_info", {}).get("initial_release_date")
                    },
                },
                "rcsb_entry_info": {
                    "resolution_combined": [entry_json.get("rcsb_entry_info", {}).get("resolution_combined")]
                    if entry_json.get("rcsb_entry_info", {}).get("resolution_combined") is not None
                    else [],
                    "experimental_method": entry_json.get("rcsb_entry_info", {}).get("experimental_method") or [],
                    "polymer_entity_count_protein": entry_json.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0),
                },
                # polymer_entities and nonpolymer_entities will be post-processed below
                "polymer_entities_raw": polymer_entities,
                "nonpolymer_entities_raw": nonpolymer_entities,
            })

            if i % 20 == 19:
                time.sleep(1)

        except Exception as e:
            logging.debug(f"JSON API error for {pdb_id}: {e}")
            continue

    logging.info(f"JSON API returned base info for {len(entries)} entries")
    return entries

def convert_ep_entry_to_offline_schema(raw_entries: List[Dict]) -> List[Dict]:
    """
    Convert JSON 'entry' + polymer_entity/nonpolymer_entity REST objects into the
    pseudo-GraphQL structure expected by process_entries_like_offline.
    """
    converted: List[Dict] = []

    for rec in raw_entries:
        base_entry = {
            "entry": rec["entry"],
            "rcsb_entry_info": rec["rcsb_entry_info"],
            "polymer_entities": [],
            "nonpolymer_entities": [],
        }

        # Map polymer_entity JSON into the fields used in process_entries_like_offline
        for pe in rec.get("polymer_entities_raw", []):
            # entity_id and uniprot_ids
            cont_ids = pe.get("rcsb_polymer_entity_container_identifiers", {}) or {}
            entity_id = cont_ids.get("entity_id")
            uniprot_ids = cont_ids.get("uniprot_ids") or []

            # type
            entity_poly_type = (pe.get("entity_poly", {}) or {}).get("type")

            # organism
            org_name = None
            if pe.get("rcsb_entity_source_organism"):
                # this is a list in JSON API
                src_list = pe["rcsb_entity_source_organism"]
                if isinstance(src_list, list) and src_list:
                    org_name = src_list[0].get("ncbi_scientific_name")

            # description
            desc = (pe.get("rcsb_polymer_entity", {}) or {}).get("pdbx_description")

            # gene names
            gene_src = pe.get("entity_src_gen") or []
            gene_names = []
            for g in gene_src:
                gn = g.get("pdbx_gene_src_gene")
                if gn:
                    gene_names.append(gn)

            # EC numbers
            ec_list = pe.get("rcsb_ec") or []
            ec_numbers = [ec.get("number") for ec in ec_list if ec.get("number")]

            base_entry["polymer_entities"].append({
                "entity_poly": {"type": entity_poly_type} if entity_poly_type else {},
                "rcsb_entity_source_organism": {"ncbi_scientific_name": org_name} if org_name else {},
                "rcsb_polymer_entity": {"pdbx_description": desc} if desc else {},
                "rcsb_polymer_entity_container_identifiers": {
                    "entity_id": entity_id,
                    "uniprot_ids": uniprot_ids,
                },
                "entity_src_gen": [{"pdbx_gene_src_gene": g} for g in gene_names] if gene_names else [],
                "rcsb_ec": [{"number": ec} for ec in ec_numbers] if ec_numbers else [],
            })

        # Map nonpolymer_entity JSON
        for np in rec.get("nonpolymer_entities_raw", []):
            # prefer chemical name, fallback to descriptor
            desc = np.get("pdbx_description")
            if not desc:
                desc = ((np.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("name")
            base_entry["nonpolymer_entities"].append({
                "pdbx_description": desc,
                "nonpolymer_comp": {
                    "chem_comp": {
                        "name": ((np.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("name")
                    }
                } if np.get("nonpolymer_comp") else {},
            })

        converted.append(base_entry)

    return converted

# --- Process GraphQL -----------

def process_entries_like_offline(entries: List[Dict], e2_ids: Set[str], e3_ids: Set[str]) -> List[Dict]:
    """Mimic parse_cif_file logic but from GraphQL JSON."""
    processed: List[Dict] = []

    for entry_data in entries:
        if not entry_data or "entry" not in entry_data:
            continue

        entry = entry_data["entry"]
        info = entry_data.get("rcsb_entry_info", {}) or {}

        experimental_methods = info.get("experimental_method") or []
        resolution_combined = info.get("resolution_combined") or []
        resolution = resolution_combined[0] if resolution_combined else None

        pdb_data = {
            "PDB_ID": entry["id"],
            "Entry_ID": entry["id"],
            "Number_of_Distinct_Protein_Entities": info.get("polymer_entity_count_protein", 0),
            "Refinement_Resolution": resolution,
            "Resolution_Method": ", ".join(experimental_methods) if experimental_methods else None,
            "Release_Date": (entry.get("rcsb_accession_info") or {}).get("initial_release_date"),
            "Entity_ID": [],
            "Source_Organism": [],
            "Gene_Name": [],
            "Macromolecule_Name": [],
            "Accession_Codes": [],
            "EC_Number": [],
            "Polymer_Type": [],
            "E2_in_entry": [],
            "E3_in_entry": [],
            "ubiquitin_in_structure": [],
            "Nonpolymer_Entities": [],
        }

        ub_entities: Set[str] = set()

        # Nonpolymer names 
        nonpoly_names: Set[str] = set()
        for np in (entry_data.get("nonpolymer_entities") or []):
            desc = np.get("pdbx_description")
            if desc:
                nonpoly_names.add(desc)
            else:
                chem = ((np.get("nonpolymer_comp") or {}).get("chem_comp") or {})
                nm = chem.get("name")
                if nm:
                    nonpoly_names.add(nm)
        if nonpoly_names:
            pdb_data["Nonpolymer_Entities"] = ",".join(sorted(nonpoly_names))

        # Polymer entities 
        for poly in (entry_data.get("polymer_entities") or []):
            if not poly:
                continue

            cont = poly.get("rcsb_polymer_entity_container_identifiers") or {}
            entity_id = cont.get("entity_id")
            if not entity_id:
                continue
            pdb_data["Entity_ID"].append(entity_id)

            # Polymer type
            poly_type = (poly.get("entity_poly") or {}).get("type")
            if poly_type:
                pdb_data["Polymer_Type"].append(f"{entity_id}:{poly_type}")

            # Organism
            org = (poly.get("rcsb_entity_source_organism") or {}).get("ncbi_scientific_name")
            if org:
                pdb_data["Source_Organism"].append(f"{entity_id}:{org}")

            # Gene names
            gene_names_list: List[str] = []
            for src in (poly.get("entity_src_gen") or []):
                g = src.get("pdbx_gene_src_gene")
                if g:
                    gene_names_list.append(str(g))
            if gene_names_list:
                pdb_data["Gene_Name"].append(f"{entity_id}:{','.join(gene_names_list)}")

            # Macromolecule name
            desc = (poly.get("rcsb_polymer_entity") or {}).get("pdbx_description")
            if desc:
                pdb_data["Macromolecule_Name"].append(f"{entity_id}:{desc}")

            # EC numbers
            ec_numbers: List[str] = []
            for ec in (poly.get("rcsb_ec") or []):
                num = ec.get("number")
                if num:
                    ec_numbers.append(num)
            if ec_numbers:
                pdb_data["EC_Number"].append(f"{entity_id}:{','.join(ec_numbers)}")

            # UniProt IDs
            uniprot_ids: Set[str] = set(cont.get("uniprot_ids") or [])
            if uniprot_ids:
                pdb_data["Accession_Codes"].append(f"{entity_id}:{','.join(sorted(uniprot_ids))}")

            entity_uniprots = list(uniprot_ids)
            has_e2_or_e3 = any((u in e2_ids) or (u in e3_ids) for u in entity_uniprots)

            # E2/E3
            for uid in entity_uniprots:
                if uid in e2_ids and uid not in pdb_data["E2_in_entry"]:
                    pdb_data["E2_in_entry"].append(uid)
                if uid in e3_ids and uid not in pdb_data["E3_in_entry"]:
                    pdb_data["E3_in_entry"].append(uid)

            # Ubiquitin classification
            # 1) UniProt-based, only if not also E2/E3
            if not has_e2_or_e3:
                for uid in entity_uniprots:
                    if uid in UBIQUITIN_UNIPROT_IDS:
                        key = f"{entity_id}:{uid}"
                        if key not in ub_entities:
                            pdb_data["ubiquitin_in_structure"].append(uid)
                            ub_entities.add(key)

                # 2) Gene-name-based
                for g in gene_names_list:
                    if g and g.strip().upper() in UBIQUITIN_GENE_NAMES:
                        key = f"{entity_id}:GENE:{g}"
                        if key not in ub_entities:
                            pdb_data["ubiquitin_in_structure"].append(g)
                            ub_entities.add(key)

                # 3) Name-based
                if desc:
                    dlow = desc.strip().lower()
                    if dlow == "ubiquitin" or "polyubiquitin" in dlow:
                        key = f"{entity_id}:NAME:{desc}"
                        if key not in ub_entities:
                            pdb_data["ubiquitin_in_structure"].append(desc)
                            ub_entities.add(key)

        processed.append(pdb_data)

    return processed

# --- UniProt helpers --------------------------------------------------------

def fetch_uniprot_gene_names(uniprot_ids: Set[str]) -> Dict[str, str]:
    """
    Map UniProt IDs to their primary/official gene names using the UniProt
    ID mapping service: from=UniProtKB_AC-ID, to=Gene_Name.

    Returns: dict[UniProt ID] = gene name (string).
    """
    if not uniprot_ids:
        return {}

    sess = requests_retry_session()
    mapping_url = "https://rest.uniprot.org/idmapping/run"
    all_ids = sorted(uniprot_ids)
    gene_map: Dict[str, str] = {}

    chunk_size = 400
    for i in range(0, len(all_ids), chunk_size):
        chunk = all_ids[i:i + chunk_size]
        logging.info(f"UniProt gene-mapping batch {i//chunk_size + 1}: {len(chunk)} IDs")

        # Submit mapping job: directly to Gene_Name
        params = {
            "from": "UniProtKB_AC-ID",
            "to": "Gene_Name",
            "ids": ",".join(chunk),
        }

        try:
            run_resp = sess.post(mapping_url, data=params, timeout=60)
            run_resp.raise_for_status()
            job_id = run_resp.json()["jobId"]
        except Exception as e:
            logging.warning(f"Failed to submit UniProt gene-mapping job: {e}")
            continue

        status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"

        # poll until results are ready
        for _ in range(30):
            st = sess.get(status_url, timeout=30)
            st.raise_for_status()
            st_data = st.json()
            if "results" in st_data:
                stream_url = f"https://rest.uniprot.org/idmapping/stream/{job_id}"
                try:
                    rs = sess.get(stream_url, timeout=120)
                    rs.raise_for_status()
                    rs_json = rs.json()
                except Exception as e:
                    logging.warning(f"Gene-mapping stream failed: {e}")
                    break

                for r in rs_json.get("results", []):
                    frm = r.get("from")   # UniProt accession
                    to_val = r.get("to")  # mapped Gene_Name string
                    if not frm or not to_val:
                        continue
                    # UniProt may return duplicates; last one wins (typically same value)
                    gene_map[frm] = str(to_val)
                break

            if st_data.get("jobStatus") == "RUNNING":
                time.sleep(2)
                continue

            logging.warning(f"UniProt gene-mapping job failed: {st_data}")
            break

        time.sleep(1)

    logging.info(f"Mapped {len(gene_map)} UniProt IDs to gene names via mapping API")
    return gene_map

# --- Save to CSVs ---------------------------------------------

def save_results(
    results_dir: Path,
    all_structure_details: List[Dict],
    e2_ids: Set[str],
    e3_ids: Set[str],
):
    if not all_structure_details:
        logging.warning("No structure details to save")
        return

    # Build UniProt → gene-name mapping from all E2/E3 IDs
    uniprot_to_gene = fetch_uniprot_gene_names(e2_ids.union(e3_ids))

    df = pd.DataFrame(all_structure_details)

    # New columns based on existing E2_in_entry / E3_in_entry lists
    def normalize_id_list(col: pd.Series) -> pd.Series: 
        def to_list(v):
            if isinstance(v, (list, tuple, set)):
                return list(v)
            if v is None:
                return []
            s = str(v).strip()
            if s in ("", "[]"):
                return []
            # try to parse list-like strings: "['P63279']"
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [s]
        return col.apply(to_list)

    # Ensure E2_in_entry / E3_in_entry are list-like internally
    if "E2_in_entry" in df.columns:
        e2_lists = normalize_id_list(df["E2_in_entry"])
    else:
        e2_lists = pd.Series([[]] * len(df))
    if "E3_in_entry" in df.columns:
        e3_lists = normalize_id_list(df["E3_in_entry"])
    else:
        e3_lists = pd.Series([[]] * len(df))

    # New UniProt ID columns (string, comma-separated)
    df["E2_UniProtIDs"] = e2_lists.apply(lambda xs: ",".join(sorted(set(str(x) for x in xs))) if xs else "")
    df["E3_UniProtIDs"] = e3_lists.apply(lambda xs: ",".join(sorted(set(str(x) for x in xs))) if xs else "")

    # New Gene name columns using UniProt mapping
    def ids_to_genes(id_list):
        genes = {uniprot_to_gene[uid] for uid in id_list if uid in uniprot_to_gene}
        return ",".join(sorted(genes)) if genes else ""

    df["E2_Gene_Names"] = e2_lists.apply(ids_to_genes)
    df["E3_Gene_Names"] = e3_lists.apply(ids_to_genes)

    # Choose output column order: use new columns instead of E2_in_entry / E3_in_entry
    columns_order = [
        "PDB_ID",
        "Entry_ID",
        "Number_of_Distinct_Protein_Entities",
        "Refinement_Resolution",
        "Resolution_Method",
        "Release_Date",
        "E2_UniProtIDs",
        "E2_Gene_Names",
        "E3_UniProtIDs",
        "E3_Gene_Names",
        "ubiquitin_in_structure",
        "Nonpolymer_Entities",
        "Entity_ID",
        "Source_Organism",
        "Gene_Name",
        "Macromolecule_Name",
        "Accession_Codes",
        "EC_Number",
        "Polymer_Type",
    ]
    df = df[[c for c in columns_order if c in df.columns]]

    # --- pretty / cleaning for detailed CSV --------------------------------
    list_like_cols = [
        "ubiquitin_in_structure",
        "Nonpolymer_Entities",
        "Entity_ID",
        "Source_Organism",
        "Gene_Name",
        "Macromolecule_Name",
        "Accession_Codes",
        "EC_Number",
        "Polymer_Type",
    ]

    def join_list_cell(v):
        if isinstance(v, (list, tuple, set)):
            return ",".join(str(x) for x in v)
        return v

    for col in list_like_cols:
        if col in df.columns:
            df[col] = df[col].apply(join_list_cell)

    # simplify ISO-like datetime to YYYY-MM-DD
    if "Release_Date" in df.columns:
        df["Release_Date"] = (
            df["Release_Date"]
            .astype(str)
            .str.extract(r"^(\d{4}-\d{2}-\d{2})")[0]
            .fillna(df["Release_Date"])
        )

    # clean Refinement_Resolution
    if "Refinement_Resolution" in df.columns:
        def clean_resolution_val(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return v
            s = str(v).strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip().strip("'").strip('"')
            return s
        df["Refinement_Resolution"] = df["Refinement_Resolution"].apply(clean_resolution_val)

    # clean Resolution_Method: strip brackets and compress/remap
    def clean_resolution_method(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return v
        s = str(v).strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if (inner.startswith("'") and inner.endswith("'")) or (inner.startswith('"') and inner.endswith('"')):
                inner = inner[1:-1]
            s = inner
        # normalize commas/spaces
        s = " ".join(s.replace(",", " ").split())
        # for your use case, you opted to remove spaces entirely
        s = s.replace(" ", "")
        return s or None

    if "Resolution_Method" in df.columns:
        df["Resolution_Method"] = df["Resolution_Method"].apply(clean_resolution_method)
    # ------------------------------------------------------------------------

    ts = datetime.datetime.now().strftime("%Y%m%d")
    csv_file = results_dir / f"pdb_structures_detailed_update_{ts}.csv"
    df.to_csv(csv_file, index=False)
    logging.info(f"Saved main CSV: {csv_file}")

    # Helper: interpret list-like columns correctly for filters.
    def col_has_nonempty_list(col: pd.Series) -> pd.Series:
        def is_nonempty(v):
            if v is None:
                return False
            if isinstance(v, (list, tuple, set)):
                return len(v) > 0
            s = str(v).strip()
            if s in ("", "[]"):
                return False
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, (list, tuple, set)):
                    return len(parsed) > 0
            except Exception:
                return s != ""
            return False
        return col.apply(is_nonempty)

    # For E2/E3 filters we now use the UniProt ID string columns
    def has_nonempty_ids(col: pd.Series) -> pd.Series:
        return col.astype(str).str.strip().ne("")

    has_e2 = has_nonempty_ids(df["E2_UniProtIDs"]) if "E2_UniProtIDs" in df.columns else pd.Series(False, index=df.index)
    has_e3 = has_nonempty_ids(df["E3_UniProtIDs"]) if "E3_UniProtIDs" in df.columns else pd.Series(False, index=df.index)
    has_ub = col_has_nonempty_list(df["ubiquitin_in_structure"]) if "ubiquitin_in_structure" in df.columns else pd.Series(False, index=df.index)

    def save_filtered(cond, name: str):
        sub = df[cond]
        if not sub.empty:
            out = results_dir / name
            sub.to_csv(out, index=False)
            logging.info(f"Saved {len(sub)} rows to {out}")
    
    # not needed for now
    """
    save_filtered(has_e2, "pdb_structures_with_e2_update.csv")
    save_filtered(has_e3, "pdb_structures_with_e3_update.csv")
    save_filtered(has_ub, "pdb_structures_with_ubiquitin_update.csv")
    save_filtered(has_e2 & has_e3, "pdb_structures_with_e2_and_e3_update.csv")
    save_filtered(has_e2 & has_ub, "pdb_structures_with_e2_and_ubiquitin_update.csv")
    save_filtered(has_e3 & has_ub, "pdb_structures_with_e3_and_ubiquitin_update.csv")
    save_filtered(has_e2 & has_e3 & has_ub, "pdb_structures_with_e2_e3_and_ubiquitin_update.csv")
    """

    logging.info("\nSummary:")
    logging.info(f"Total unique PDB structures: {len(df)}")
    logging.info(f"PDB structures containing E2 (by UniProt IDs): {len(df[has_e2])}")
    logging.info(f"PDB structures containing E3 (by UniProt IDs): {len(df[has_e3])}")
    logging.info(f"PDB structures containing Ubiquitin: {len(df[has_ub])}")
    logging.info(f"PDB structures containing both E2 and E3: {len(df[has_e2 & has_e3])}")
    logging.info(f"PDB structures containing E2, E3, and Ubiquitin: {len(df[has_e2 & has_e3 & has_ub])}")

# --- main -------------------------------------------------------------------

def main():
    project_dir = Path(__file__).resolve().parents[2]
    e2_file = project_dir / "update_pdb" / "input_data" / "E2_list.txt"
    e3_file = project_dir / "update_pdb" / "input_data" / "E3_list.txt"

    results_dir = project_dir / "util" / "data" / "pdb_updates"

    logger, log_file = setup_logging(results_dir)
    logging.info(f"Logging to {log_file}")

    if not e2_file.exists() or not e3_file.exists():
        logging.error("Missing input files:")
        if not e2_file.exists():
            logging.error(f"  - {e2_file}")
        if not e3_file.exists():
            logging.error(f"  - {e3_file}")
        return

    e2_ids = set(read_uniprot_ids(e2_file))
    e3_ids = set(read_uniprot_ids(e3_file))
    logging.info(f"Read {len(e2_ids)} E2 IDs and {len(e3_ids)} E3 IDs")

    all_uniprot = sorted(e2_ids.union(e3_ids))
    logging.info(f"Total unique UniProt IDs: {len(all_uniprot)}")

    # UniProt → PDB
    up_to_pdb = map_uniprot_to_pdb(all_uniprot, batch_size=50)
    all_pdb_ids = sorted({pdb for s in up_to_pdb.values() for pdb in s})
    logging.info(f"Total unique PDB IDs mapped: {len(all_pdb_ids)}")

    if not all_pdb_ids:
        logging.warning("No PDB IDs found from UniProt mapping")
        return

    # Process ALL mapped PDB IDs (no artificial max limit)
    logging.info(f"Processing all {len(all_pdb_ids)} PDB IDs")

    # PDB → details (JSON API instead of GraphQL)
    raw_entries = fetch_pdb_details_json(all_pdb_ids)
    if not raw_entries:
        logging.warning("No detailed data retrieved from JSON API")
        return

    # Convert JSON objects into structure compatible with process_entries_like_offline
    gql_like_entries = convert_ep_entry_to_offline_schema(raw_entries)

    # Process into same schema as offline CIF parsing
    details = process_entries_like_offline(gql_like_entries, e2_ids, e3_ids)

    # Save outputs
    save_results(results_dir, details, e2_ids, e3_ids)

    logging.info("processing complete.")

if __name__ == "__main__":
    main()
