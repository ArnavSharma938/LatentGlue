import os
import logging
import pandas as pd
import re
from src.processing.mol_utils import DatasetTracker

logger = logging.getLogger("ActivitySet")
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.INFO)
logger.propagate = False
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

def log_stage(current, total, description):
    separator = "=" * 60
    logger.info(separator)
    logger.info(f" STAGE {current}/{total}: {description}")
    logger.info(separator)

def is_invalid_cell_line(val):
    if pd.isna(val): return True
    s = str(val).strip()
    return s == "" or all(c == "." for c in s)

def normalize_cell_line(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    if re.match(r'^HEK[-\s]?293[T]?$', s, re.IGNORECASE):
        return 'HEK293T'
    if re.match(r'^HT[-\s]?1080$', s, re.IGNORECASE):
        return 'HT-1080 cell'
    return s

def clean_to_pure_numeric(val):
    if pd.isna(val): return None
    s = str(val).strip()
    if re.search(r'[><=≤≥x\-–—/]', s, re.IGNORECASE):
        return None
    
    numeric_part = re.sub(r'[^0-9.]', '', s)
    try:
        if numeric_part == "": return None
        test_val = s.replace('nM', '', 1).strip()
        float(test_val) 
        return float(numeric_part)
    except ValueError:
        return None

def process_mgdb_activity():
    log_stage(1, 3, "Processing MGDB Activity Data")
    
    data_dir = os.path.join('data', 'subdata')
    activity_file = os.path.join(data_dir, 'Activity-MGDB.csv')
    compound_file = os.path.join(data_dir, 'Processed_MGDB.csv')
    
    if not os.path.exists(activity_file) or not os.path.exists(compound_file):
        logger.warning("MGDB activity or compound file missing. Skipping.")
        return None

    activity_data = pd.read_csv(activity_file)
    tracker.record("MGDB: Raw Activity", len(activity_data))
    
    df = activity_data.copy()
    
    cols_to_drop = ['name', 'Target', 'DOI', 'Patent number', 'Patent title', 
                    'Disease', 'Picture', 'Assay', 'Description',
                    'Assay Method', 'Administration Time', 'Efficacy Type']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df = df[df['Efficacy Data'].str.strip() == 'DC50'].copy()
    tracker.record("MGDB: DC50 Only", len(df))
    
    allowed_cells = {'HEK293T', 'HT-1080 cell'}
    df = df[~df['Model'].map(is_invalid_cell_line)].copy()
    df['Model'] = df['Model'].map(normalize_cell_line)
    df = df[df['Model'].isin(allowed_cells)].copy()
    tracker.record("MGDB: Cell-Line Validated", len(df))
    
    df = df[df['Units'].str.strip() == 'nM'].copy()
    
    df['Result'] = df['Result'].map(clean_to_pure_numeric)
    df = df.dropna(subset=['Result']).copy()
    tracker.record("MGDB: Numeric Quality", len(df))
    
    df = df.rename(columns={
        'ID': 'Compound ID',
        'Model': 'Cell Line',
        'Result': 'Value'
    })
    df['Source'] = 'MGDB'
    
    keep_cols = ['Compound ID', 'Cell Line', 'Value', 'Units', 'Source']
    df = df[keep_cols]
    
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    logger.info(f"MGDB Activity processing complete. {len(df)} rows prepared.")
    return df

tracker = DatasetTracker("Activity Dataset")

if __name__ == "__main__":
    final_file = os.path.join('data', 'GlueDegradDB-Activity.csv')
    df_mg = process_mgdb_activity()
    
    log_stage(2, 2, "Saving MGDB Activity Data")
    dfs_to_combine = []
    if df_mg is not None: dfs_to_combine.append(df_mg)
    
    if dfs_to_combine:
        df_combined = pd.concat(dfs_to_combine, ignore_index=True)
        df_combined.to_csv(final_file, index=False)
        tracker.record("Unified: Clean Activity", len(df_combined))
        logger.info(f"Unified activity data saved to {final_file}. Total rows: {len(df_combined)}")
        tracker.record("Unified Activity Saved", len(df_combined))
    else:
        logger.error("No activity data to unify.")
