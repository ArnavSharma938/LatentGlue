import os
import re
import logging
import pandas as pd
import requests
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from src.processing.mol_utils import parallel_process, parallel_io_process, DatasetTracker, is_missing
logger = logging.getLogger("SubSet")
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.INFO)
logger.propagate = False
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

def log_stage(current, total, description):
    separator = "=" * 60
    logger.info(separator)
    logger.info(f" STAGE {current}/{total}: {description}")
    logger.info(separator)

def log_summary(script_name, metrics):
    logger.info("=" * 60)
    logger.info(f" SUMMARY: {script_name}")
    for key, value in metrics.items():
        logger.info(f" - {key}: {value}")
    logger.info("=" * 60)

def strip_cell(x: str) -> str:
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()

def norm_header(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    norm = lambda s: re.sub(r"\s+", " ", s.strip()).lower()
    norm_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in norm_map:
            return norm_map[key]
    return candidates[0]

MGDB_DROP_COLS = [
    "Unnamed: 34", "Unnamed: 35", "name", "Covalent warhead",
    "Protein-targeting ligand", "Application", "Status", "Cite",
    "Reference", "Patent Number", "Application Title", "Synonyms",
    "PubChem CID", "Molecular Weight", "IUPAC Name", "InChI",
    "InChI Key", "ChEMBL", "DrugBank ID", "CAS",
    "Advantage", "None-protein Name", "Target name", "Molecular Formula",
    "PDB ID", "E3 ligase", "E3 ligase-nonProtein", "MolWeight", "ExactMass",
    "ClogP", "TPSA", "HBACount", "HBDCount",
    "RotatableBondCount", "RingCount", "AliphaticRings", "AromaticRings",
    "AliphaticHeteroRings", "AromaticHeteroRings", "HeavyAtoms", "HeteroAtoms",
    "SpiroAtoms", "BridgeheadAtoms", "Caco_2", "BBB", "Pgp_inhibitor",
    "Pgp_substrate", "CYP1A2_inhibitor", "CYP3A4_inhibitor", "CYP2B6_inhibitor",
    "CYP2C9_inhibitor", "CYP2C19_inhibitor", "CYP2D6_inhibitor",
    "CYP1A2_substrate", "CYP3A4_substrate", "CYP2B6_substrate", "CYP2C9_substrate",
    "CYP2C19_substrate", "CYP2D6_substrate", "CLp_c", "CLr", "Neurotoxicity",
    "DILI", "hERG_10uM", "Respiratory_toxicity", "Skin_corrosion",
    "Skin_irritation", "Skin_sensitisation", "Ames", "Mouse_carcinogenicity",
    "Rat_carcinogenicity", "Rodents_carcinogenicity", "Biodegradability", "AOT", "Type", 
    "MoA", "Mechanism", "Mechanism annotation", 
    "MOA", "MOA type", "Mode of action", "ModeOfAction", "Scaffold", 
    "Target Repeat", "Target Disorder", "Effector Repeat", "Effector Disorder",
    "Target REPEAT", "Target DISORDER", "Effector REPEAT", "Effector DISORDER", "Subtype"
]

def clean_seq(s):
    if pd.isna(s): return s
    return str(s).replace(',', '').replace('，', '').strip()

def _fetch_single_uniprot_mgdb(uid):
    """Helper for parallel UniProt + AlphaFold + structure feature fetching for MGDB."""
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            taxon_id = data.get("organism", {}).get("taxonId")
            entry_type = data.get("entryType", "")
            if taxon_id == 9606 and "reviewed" in entry_type.lower():
                seq = data.get("sequence", {}).get("value")
                if seq:
                    return uid, {
                        "sequence": seq
                    }
    except Exception as e:
        logger.error(f"Error fetching {uid}: {str(e)}")
    return uid, None

def mgdb_clean_smiles(s: str) -> str:
    if is_missing(s):
        return ""
    cleaned = strip_cell(s)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None: return ""
    Chem.SanitizeMol(mol)
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    else:
        mol = frags[0]
    clean_s = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return clean_s

def mgdb_fetch_uniprot_annotations(uniprot_ids: set[str]) -> dict[str, dict]:
    ids_list = [uid for uid in uniprot_ids if not is_missing(uid)]
    logger.info(f"Fetching sequences and annotations for {len(ids_list)} unique UniProt IDs...")
    fetch_results = parallel_io_process(_fetch_single_uniprot_mgdb, ids_list)
    results = {}
    for uid, data in fetch_results:
        if data:
            results[uid] = data
    return results

def mgdb_preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    log_stage(1, 4, "Column Level Filtering")
    initial_cols = len(df.columns)
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(strip_cell)
    drop_norm = {norm_header(c) for c in MGDB_DROP_COLS}
    cols_to_drop = [c for c in df.columns if norm_header(c) in drop_norm]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    logger.info(f"Dropped {initial_cols - len(df.columns)} specified columns.")
    return df

def mgdb_preprocess_rows(df: pd.DataFrame) -> pd.DataFrame:
    log_stage(2, 4, "Row Level Filtering")
    initial_count = len(df)
    logger.info(f"Starting row processing with {initial_count} initial entries.")

    smiles_col = pick_col(df, ["SMILES", "Smiles", "Canonical SMILES"])
    logger.info("Cleaning SMILES...")
    df.loc[:, smiles_col] = parallel_process(mgdb_clean_smiles, df[smiles_col].tolist())
    
    df_valid_smiles = df[df[smiles_col] != ""].copy()
    diff = len(df) - len(df_valid_smiles)
    logger.info(f"SMILES Cleaned: Removed {diff} entries.")
    df = df_valid_smiles

    df = df[~df[smiles_col].str.lower().isin(['stabilizer', 'degrader'])].copy()
    logger.info(f"Removed disorganized rows from MGDB. Current count: {len(df)}")

    target_uniprot_col = pick_col(df, ["Uniprot", "Target Uniprot", "Target UniProt", "Target Uniprot ID"])
    target_seq_col = pick_col(df, ["Target Sequence", "Protein Sequence", "Target protein sequence", "Sequence"])
    eff_uniprot_col = pick_col(df, ["Effector Uniprot", "Effector UniProt", "Partner Uniprot", "Partner UniProt"])
    eff_seq_col = pick_col(df, ["Effector Sequence", "Partner Sequence", "Effector protein sequence", "Partner protein sequence"])

    logger.info("Cleaning protein sequences...")
    df.loc[:, target_seq_col] = parallel_process(clean_seq, df[target_seq_col].tolist())
    df.loc[:, eff_seq_col] = parallel_process(clean_seq, df[eff_seq_col].tolist())

    pre_multi = len(df)
    df = df[~df[target_uniprot_col].str.contains(r'[;/]', na=False)].copy()
    df = df[~df[eff_uniprot_col].str.contains(r'[;/]', na=False)].copy()
    logger.info(f"Removed multi-target/effector entries in MGDB: Removed {pre_multi - len(df)} rows.")

    prot_mask = (
        (~df[target_uniprot_col].map(is_missing)) &
        (~df[target_seq_col].map(is_missing)) &
        (~df[eff_uniprot_col].map(is_missing)) &
        (~df[eff_seq_col].map(is_missing))
    )
    df_prot = df[prot_mask].copy()
    diff = len(df) - len(df_prot)
    logger.info(f"Protein Pairs: Removed {diff} entries missing basic protein details.")
    df = df_prot
    
    try:
        gene_col = pick_col(df, ["Gene Name", "Target Gene"])
        u_col = target_uniprot_col
        s_col = target_seq_col
        eff_gene_col = pick_col(df, ["E3 Gene Name", "Effector Gene", "E3 ligase"])
        eff_u_col = eff_uniprot_col

        ikzf1_mask = (df[gene_col] == 'IKZF1')
        if ikzf1_mask.any():
            logger.info(f"Fixing {ikzf1_mask.sum()} IKZF1 rows: Setting Target Uniprot to Q13422")
            df.loc[ikzf1_mask, u_col] = 'Q13422'
            valid_ikzf1 = df[(df[gene_col] == 'IKZF1') & (df[s_col].str.len() > 50)]
            if not valid_ikzf1.empty:
                correct_seq = valid_ikzf1.iloc[0][s_col]
                df.loc[ikzf1_mask, s_col] = correct_seq

        crbn_mask = (df[eff_u_col] == 'Q13422') & (df[eff_gene_col].str.contains('CRBN', case=False, na=False))
        if crbn_mask.any():
            logger.info(f"Fixing {crbn_mask.sum()} CRBN Effector rows: Setting Effector Uniprot to Q96SW2")
            df.loc[crbn_mask, eff_u_col] = 'Q96SW2'

        ar_mask = (df[u_col] == 'P10275')
        if ar_mask.any():
            logger.info("Refetching P10275 sequence from UniProt to ensure correctness...")
            try:
                resp = requests.get("https://rest.uniprot.org/uniprotkb/P10275.fasta")
                if resp.status_code == 200:
                    ar_seq = "".join(resp.text.split('\n')[1:])
                    df.loc[ar_mask, s_col] = ar_seq
                    logger.info("Updated P10275 sequence.")
            except Exception as ex:
                logger.warning(f"Failed to fetch P10275: {ex}")

    except Exception as e:
        logger.warning(f"Failed to apply final data fixes: {e}")

    # Filter out known AlphaFold failures
    af_excludes = {'A9UF02', 'P42858'}
    df = df[~df[target_uniprot_col].isin(af_excludes)].copy()
    df = df[~df[eff_uniprot_col].isin(af_excludes)].copy()

    # Human UniProt API Validation & Annotation
    log_stage(3, 4, "Biological Consistency & Annotation")
    target_uids = set(df[target_uniprot_col].unique())
    effector_uids = set(df[eff_uniprot_col].unique())
    all_uids = target_uids.union(effector_uids)
    
    uniprot_data = mgdb_fetch_uniprot_annotations(all_uids)
    
    valid_mask = (df[target_uniprot_col].isin(uniprot_data)) & (df[eff_uniprot_col].isin(uniprot_data))
    df_valid = df[valid_mask].copy()
    
    df_valid[target_seq_col] = df_valid[target_uniprot_col].map(lambda x: uniprot_data[x]['sequence'])
    df_valid[eff_seq_col] = df_valid[eff_uniprot_col].map(lambda x: uniprot_data[x]['sequence'])
    
    diff = len(df) - len(df_valid)
    logger.info(f"Biological Consistency: Removed {diff} non-human or invalid entries.")
    df = df_valid

    log_stage(4, 4, "Biological Context Filtering")
    func_col = pick_col(df, ["Function", "Mechanism", "Mechanism annotation", "MOA", "MOA type"])
    
    def is_degrader(v: str) -> bool:
        if is_missing(v): return False
        s = re.sub(r"[^a-z]+", "", strip_cell(v).lower())
        return s == "degrader"

    df['_is_degrader'] = df[func_col].map(is_degrader)
    
    has_mech_mask = ~df[func_col].map(is_missing)
    df_mech_exists = df[has_mech_mask].copy()
    diff = len(df) - len(df_mech_exists)
    logger.info(f"Mechanism Exists: Removed {diff} entries with missing annotations.")
    df = df_mech_exists

    df_only_degrader = df[df['_is_degrader']].copy()
    diff = len(df) - len(df_only_degrader)
    logger.info(f"Mechanism Type: Removed {diff} non-degrader entries.")
    df = df_only_degrader
    tracker.record("MGDB: Cleaned & Validated", len(df))

    df = df.sort_values(by='_is_degrader', ascending=False)
    initial_dupes = len(df)
    df = df.drop_duplicates(subset=[smiles_col, target_uniprot_col, eff_uniprot_col], keep='first')
    diff = initial_dupes - len(df)
    logger.info(f"Duplicate Removal: Removed {diff} identical systems.")
    tracker.record("MGDB: Final Deduplicated", len(df))

    df = df.drop(columns=['_is_degrader', func_col], errors='ignore')


    log_summary("MGDB Processor", {
        "Initial Entries": initial_count,
        "Final Entries": len(df),
        "Total Removed": initial_count - len(df)
    })
    return df

def process_mgdb():
    input_path = os.path.join("data", "subdata", "MGDB.csv")
    output_path = os.path.join("data", "subdata", "Processed_MGDB.csv")
    
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    tracker.record("MGDB: Raw Records", len(df))
    df = mgdb_preprocess_columns(df)
    df = mgdb_preprocess_rows(df)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")

LIGASE_MAP = {
    'CRBN': 'Q96SW2', 'DDB1': 'Q16531', 'DCAF15': 'Q66K64', 'DCAF16': 'Q9NXF7',
    'VHL': 'P40337', 'β-TrCP': 'Q9Y297', 'MDM2': 'Q00987', 'RNF126': 'Q9BV68',
    'TRIM21': 'P19474', 'KLHDC3': 'Q9BQ90', 'LZTR1': 'Q8N653', 'ASB8': 'Q9H765',
    'FBXO4': 'Q9UKT5', 'UBE2D1': 'P51668', 'ZFP91': 'Q96JP5', 'KBTBD4': 'Q9NVX7',
    'CHIP': 'Q9UNE7', 'DCAF11': 'Q8TEB1', 'TRIM25': 'Q14258'
}

EXCLUDE_LIGASES = {'COl1', 'UBE2', 'TIR1'}
EXCLUDE_IDS = {'A0A804HI94', 'A9UF02', 'P42858'}

def tpddb_clean_smiles(s: str) -> str:
    if is_missing(s):
        return ""
    try:
        cleaned = str(s).strip()
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None: return ""
        Chem.SanitizeMol(mol)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        else:
            mol = frags[0]
        clean_s = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return clean_s
    except:
        return ""

def _fetch_single_uniprot_tpd(uid):
    """Helper for parallel UniProt + AlphaFold + structure feature fetching in TPDDB."""
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
                return uid, {
                    "sequence": seq
                }
        elif response.status_code == 404:
            logger.warning(f"UniProt ID not found: {uid}")
    except Exception as e:
        logger.error(f"Error fetching {uid}: {str(e)}")
    return uid, None

def tpddb_fetch_uniprot_data(uniprot_ids: set[str]) -> dict[str, dict]:
    ids_list = list(uniprot_ids)
    logger.info(f"Fetching sequences and annotations for {len(ids_list)} unique UniProt IDs...")
    fetch_results = parallel_io_process(_fetch_single_uniprot_tpd, ids_list)
    results = {}
    for uid, data in fetch_results:
        if data:
            results[uid] = data
    return results

def tpddb_preprocess_tier1(df: pd.DataFrame) -> pd.DataFrame:
    log_stage(1, 3, "Basic Filtering")
    tracker.record("TPDDB: Raw Records", len(df))
    initial_count = len(df)

    smiles_col = pick_col(df, ['SMILES'])
    target_id_col = pick_col(df, ['Target ID'])
    ligase_col = pick_col(df, ['Ligase'])
    mech_annot_col = pick_col(df, ['Mechanism annotation', 'Mechanism Annotation', 'Subtype'])

    logger.info("Cleaning SMILES...")
    df.loc[:, smiles_col] = parallel_process(tpddb_clean_smiles, df[smiles_col].tolist())
    
    df = df[df[smiles_col] != ""]
    logger.info(f"Cleaned SMILES. Current count: {len(df)}")

    df = df[~df[smiles_col].str.lower().isin(['stabilizer', 'degrader'])]
    logger.info(f"Removed disorganized rows. Current count: {len(df)}")

    df = df[~df[target_id_col].map(is_missing)]
    df = df[~df[ligase_col].map(is_missing)]
    logger.info(f"Removed missing Target ID or Ligase. Current count: {len(df)}")

    df = df[~df[mech_annot_col].map(is_missing)]
    df = df[df[mech_annot_col].str.lower().str.contains('degrader', na=False)]
    logger.info(f"Filtered for 'degrader' mechanism. Current count: {len(df)}")

    df = df.drop_duplicates(subset=[smiles_col, target_id_col, ligase_col], keep='first')
    logger.info(f"Removed duplicate rows. Current count: {len(df)}")

    cols_to_drop = ['Source', 'TPD NAME', 'PubChem synonyms', 'PubChem Synonyms', 'Fomula']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    logger.info(f"Dropped Tier 1 columns. Total removed in Tier 1: {initial_count - len(df)}")
    
    return df

def tpddb_preprocess_tier2(df: pd.DataFrame) -> pd.DataFrame:
    log_stage(2, 3, "Complex Filtering & Expansion")
    
    target_sym_col = pick_col(df, ['Target Symbol'])
    target_id_col = pick_col(df, ['Target ID'])
    ligase_col = pick_col(df, ['Ligase'])

    pre_expand_count = len(df)
    df = df[~df[target_sym_col].str.contains(';', na=False)]
    df = df[~df[target_id_col].str.contains('/', na=False)]
    logger.info(f"Removed multi-target entries. Removed {pre_expand_count - len(df)} rows. Current count: {len(df)}")

    df['Effector UniProt'] = df[ligase_col].map(LIGASE_MAP)
    df = df[~df[ligase_col].isin(EXCLUDE_LIGASES)]
    df = df[~df['Effector UniProt'].map(is_missing)]
    
    pre_exclude_count = len(df)
    df = df[~df[target_id_col].isin(EXCLUDE_IDS)]
    df = df[~df['Effector UniProt'].isin(EXCLUDE_IDS)]
    logger.info(f"Mapped Ligases and cleaned special cases. Excluded {pre_exclude_count - len(df)} rows. Current count: {len(df)}")
    
    return df

def tpddb_fetch_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    log_stage(3, 3, "Sequence Fetching & Validation")
    
    unique_ids = set(df['Target ID'].unique()).union(set(df['Effector UniProt'].unique()))
    uniprot_data = tpddb_fetch_uniprot_data(unique_ids)
    
    valid_mask = (df['Target ID'].isin(uniprot_data)) & (df['Effector UniProt'].isin(uniprot_data))
    df_valid = df[valid_mask].copy()
    
    df_valid['Target Sequence'] = df_valid['Target ID'].map(lambda x: uniprot_data[x]['sequence'])
    df_valid['Effector Sequence'] = df_valid['Effector UniProt'].map(lambda x: uniprot_data[x]['sequence'])
    
    logger.info(f"Filtered for Human-only and successful sequence retrieval. Current count: {len(df_valid)}")
    df = df_valid
    tracker.record("TPDDB: Validated SubSet", len(df))
    
    initial_dupes = len(df)
    smiles_col = pick_col(df, ['SMILES'])
    df = df.drop_duplicates(subset=[smiles_col, 'Target Sequence', 'Effector Sequence'], keep='first')
    diff = initial_dupes - len(df)
    logger.info(f"Removed duplicate sequences. Removed {diff} entries. Final count: {len(df)}")
    tracker.record("TPDDB: Final Deduplicated", len(df))
    
    return df

def process_tpddb():
    input_path = os.path.join("data", "subdata", "TPDDB.csv")
    output_path = os.path.join("data", "subdata", "Processed_TPDDB.csv")
    
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    df = tpddb_preprocess_tier1(df)
    df = tpddb_preprocess_tier2(df)
    df = tpddb_fetch_and_validate(df)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.to_csv(output_path, index=False)
    log_summary("TPDDB Processor", {
        "Final Entries": len(df),
        "Output": output_path
    })


tracker = DatasetTracker("Training Set")

if __name__ == "__main__":
    logger.info("Processing MGDB compounds...")
    process_mgdb()
    print()
    logger.info("Processing TPDDB compounds...")
    process_tpddb()
