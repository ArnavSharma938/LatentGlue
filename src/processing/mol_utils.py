import logging
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
import pandas as pd
import requests

logger = logging.getLogger("MolUtils")
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.INFO)
logger.propagate = False
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

def is_missing(x):
    if x is None: return True
    if pd.isna(x): return True
    s = str(x).strip()
    return s == "" or s.lower() in {"na", "n/a", "nan", "null", "none"}

def parallel_process(func, data_list, n_cores=None):
    return list(map(func, data_list))

def parallel_io_process(func, data_list, max_workers=20):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, data_list))
    return results

def _fetch_single_molglue(uid):
    if not uid or pd.isna(uid): return uid, None
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            taxon_id = data.get("organism", {}).get("taxonId")
            entry_type = data.get("entryType", "")
            if taxon_id != 9606 or "reviewed" not in entry_type.lower():
                return uid, None
            seq = data.get("sequence", {}).get("value")
            if seq:
                return uid, {"sequence": seq}
    except Exception as e:
        logger.error(f"Error fetching {uid}: {str(e)}")
    return uid, None

def fetch_data_and_annotations(uniprot_ids: set[str]) -> dict[str, dict]:
    ids_list = list(uniprot_ids)
    logger.info(f"Fetching sequences and annotations for {len(ids_list)} unique UniProt IDs...")
    fetch_results = parallel_io_process(_fetch_single_molglue, ids_list)
    results = {}
    for uid, data in fetch_results:
        if data:
            results[uid] = data
    return results

def canonicalize_smiles(smiles: str) -> str:
    if not smiles or pd.isna(smiles):
        return ""
    try:
        cleaned = str(smiles).strip()
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return ""
        # Generate canonical non-isomeric SMILES, aligned with MoLFormer-XL
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        return ""

class DatasetTracker:
    def __init__(self, title):
        self.title = title

    def record(self, step_name, count):
        logger.info(f"TRACKER [{self.title}]: {step_name} -> {count}")
