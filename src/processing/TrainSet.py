import random
import os
import logging
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from src.processing.mol_utils import (
    parallel_process,
    canonicalize_smiles, DatasetTracker,
)

logger = logging.getLogger("TrainSet")
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

def perform_data_split(df):
    logger.info("Performing strict component-level split via clusters...")
    
    adj = {}
    def add_edge(u, v):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        
    for _, row in df.iterrows():
        s = f"S_{row['SMILES']}"
        t = f"T_{row['Target UniProt']}"
        e = f"E_{row['Effector UniProt']}"
        add_edge(s, t)
        add_edge(t, e)
        add_edge(e, s)

    visited = set()
    clusters = []
    nodes = list(adj.keys())
    
    for start_node in nodes:
        if start_node not in visited:
            cluster = []
            stack = [start_node]
            visited.add(start_node)
            while stack:
                u = stack.pop()
                cluster.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            clusters.append(cluster)
            
    logger.info(f"Identified {len(clusters)} independent ternary clusters.")

    node_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = i
            
    cluster_weights = [0] * len(clusters)
    for _, row in df.iterrows():
        s = f"S_{row['SMILES']}"
        cluster_weights[node_to_cluster[s]] += 1

    indices = list(range(len(clusters)))
    random.seed(42)
    random.shuffle(indices)
    
    total_rows = len(df)
    target_train = 0.90 * total_rows
    target_val = 0.05 * total_rows
    
    cluster_split = {}
    current_train = 0
    current_val = 0
    
    for idx in indices:
        weight = cluster_weights[idx]
        if current_train < target_train:
            cluster_split[idx] = 'train'
            current_train += weight
        elif current_val < target_val:
            cluster_split[idx] = 'val'
            current_val += weight
        else:
            cluster_split[idx] = 'test'

    df['split'] = [cluster_split[node_to_cluster[f"S_{s}"]] for s in df['SMILES']]
    
    dist = df['split'].value_counts(normalize=True) * 100
    logger.info(f"Strict Cluster Split Distribution: {dist.to_dict()}")
    return df

def finalize_database():
    log_stage(1, 4, "Unify and Deduplicate")
    mgdb_path = r'data/subdata/Processed_MGDB.csv'
    tpddb_path = r'data/subdata/Processed_TPDDB.csv'
    eval_path = r'data/GlueDegradDB-Eval.csv'

    logger.info("Reading MGDB and TPDDB data...")
    df_mgdb = pd.read_csv(mgdb_path)
    mgdb_mapping = {
        'ID': 'Compound ID',
        'Smiles': 'SMILES',
        'Gene Name': 'Target',
        'Uniprot': 'Target UniProt',
        'Sequence': 'Target Sequence',
        'E3 Gene Name': 'Effector',
        'Effector Uniprot': 'Effector UniProt',
        'Effector Sequence': 'Effector Sequence',
        'Function': 'Function'
    }
    df_mgdb.rename(columns=mgdb_mapping, inplace=True)
    df_mgdb['Source'] = 'MGDB'

    df_tpddb = pd.read_csv(tpddb_path)
    tpddb_mapping = {
        'TPD ID': 'Compound ID',
        'SMILES': 'SMILES',
        'Target Symbol': 'Target',
        'Target ID': 'Target UniProt',
        'Target Sequence': 'Target Sequence',
        'Ligase': 'Effector',
        'Effector UniProt': 'Effector UniProt',
        'Effector Sequence': 'Effector Sequence',
        'Subtype': 'Function'
    }
    df_tpddb.rename(columns=tpddb_mapping, inplace=True)
    df_tpddb['Source'] = 'TPDDB'

    dfs = [df_mgdb, df_tpddb]
    common_columns = [
        'Compound ID', 'SMILES', 'Target', 'Target UniProt', 'Target Sequence',
        'Effector', 'Effector UniProt', 'Effector Sequence', 'Source'
    ]

    df_unified = pd.concat([df[common_columns] for df in dfs], ignore_index=True)
    logger.info(f"Total rows before deduplication: {len(df_unified)}")
    tracker.record("Unified: Combined Sources", len(df_unified))

    activity_path = r'data/GlueDegradDB-Activity.csv'
    if os.path.exists(activity_path):
        logger.info("Merging activity data (DC50) into unified dataset...")
        df_activity = pd.read_csv(activity_path)

        df_activity['Compound ID'] = df_activity['Compound ID'].astype(str)
        df_unified['Compound ID'] = df_unified['Compound ID'].astype(str)
        
        df_unified = pd.merge(
            df_unified, 
            df_activity[['Compound ID', 'Source', 'Cell Line', 'Value', 'Units']], 
            on=['Compound ID', 'Source'], 
            how='left'
        )
        logger.info(f"Total rows after activity join: {len(df_unified)}")
    else:
        logger.warning(f"Activity data not found at {activity_path}. Skipping join.")

    df_unified['SMILES'] = df_unified['SMILES'].astype(str).str.strip()
    df_unified['Target UniProt'] = df_unified['Target UniProt'].astype(str).str.strip()
    df_unified['Effector UniProt'] = df_unified['Effector UniProt'].astype(str).str.strip()

    gene_map = {
        'β-TrCP': 'BTRC',
        'BTRCP': 'BTRC',
        'CK1α': 'CSNK1A1',
        'β-Catenin': 'CTNNB1',
        '14-3-3σ': 'SFN',
        'BCR/ABL fusion': 'BCR-ABL1'
    }

    log_stage(2, 4, "Augmentation & Normalization")
    logger.info("Standardizing SMILES...")
    df_unified.loc[:, 'SMILES'] = parallel_process(canonicalize_smiles, df_unified['SMILES'].tolist())

    if 'Target' in df_unified.columns:
        logger.info("Normalizing Target/Effector names...")
        df_unified['Target'] = df_unified['Target'].replace({'EEF2K': 'eEF2K', 'hAG-2': 'AGR2'})
        df_unified['Target'] = df_unified['Target'].replace(gene_map)
        df_unified['Effector'] = df_unified['Effector'].replace(gene_map)

    subset_cols = ['SMILES', 'Target UniProt', 'Effector UniProt']
    if 'Cell Line' in df_unified.columns:
        subset_cols.extend(['Cell Line', 'Value'])
        
    df = df_unified.drop_duplicates(subset=subset_cols, keep='first').copy()
    logger.info(f"Total rows after final deduplication: {len(df)}")
    tracker.record("Unified: Final Refined Database", len(df))

    log_stage(3, 4, "Finalizing Datasets")

    df = perform_data_split(df)

    df.insert(0, 'ID', range(1, len(df) + 1))

    log_stage(6, 6, "Separating Activity & Classification Data")
    
    activity_csv = r'data/GlueDegradDB-Activity.csv'
    classification_csv = r'data/GlueDegradDB.csv'
    
    if 'Value' in df.columns:
        df_activity_final = df[df['Value'].notna()].copy()
        
        df_classification_final = df[df['Value'].isna()].copy()
        
        drop_cols_act = [
            'split', 'MolWeight', 'ExactMass', 'HeavyAtoms', 'HeteroAtoms', 
            'ClogP', 'TPSA', 'HBACount', 'HBDCount', 'RotatableBondCount', 
            'RingCount', 'AliphaticRings', 'AromaticRings', 'AliphaticHeteroRings', 
            'AromaticHeteroRings', 'SpiroAtoms', 'BridgeheadAtoms', 'Scaffold', 
            'Target Repeat', 'Target Disorder', 'Effector Repeat', 'Effector Disorder', 
            'MoA', 'Function', 'Covalent'
        ]
        df_activity_final = df_activity_final.drop(columns=[c for c in drop_cols_act if c in df_activity_final.columns])
        
        drop_cols_cls = [
            'Cell Line', 'Value', 'Units', 'Covalent', 'MoA', 'Function',
            'MolWeight', 'ExactMass', 'HeavyAtoms', 'HeteroAtoms', 
            'ClogP', 'TPSA', 'HBACount', 'HBDCount', 'RotatableBondCount', 
            'RingCount', 'AliphaticRings', 'AromaticRings', 'AliphaticHeteroRings', 
            'AromaticHeteroRings', 'SpiroAtoms', 'BridgeheadAtoms', 'Scaffold', 
            'Target Repeat', 'Target Disorder', 'Effector Repeat', 'Effector Disorder'
        ]
        df_classification_final = df_classification_final.drop(columns=[c for c in drop_cols_cls if c in df_classification_final.columns])

        if 'Target' in df_activity_final.columns:
            mask = df_activity_final['Target'].isin(['CDK2', 'WIZ'])
            dropped_count = len(df_activity_final) - mask.sum()
            df_activity_final = df_activity_final[mask].copy()
            logger.info(f"Filtered Activity set to CDK2 and WIZ only. Dropped {dropped_count} non-target rows.")

        if {'Target', 'Effector', 'split'}.issubset(df_classification_final.columns):
            holdout_mask = (
                df_classification_final['split'].astype(str).eq('train')
                & df_classification_final['Target'].astype(str).isin(['CDK2', 'WIZ'])
                & df_classification_final['Effector'].astype(str).eq('CRBN')
            )
            held_out_count = int(holdout_mask.sum())
            if held_out_count > 0:
                df_classification_final = df_classification_final[~holdout_mask].copy()
                logger.info(
                    "Removed %d train rows for CDK2/WIZ-CRBN from the classification set "
                    "to keep the activity evaluation family held out.",
                    held_out_count,
                )

        logger.info(f"Saving {len(df_activity_final)} rows to Activity Dataset (Regression)...")
        df_activity_final.to_csv(activity_csv, index=False)
        tracker.record("Final: Activity Set (Regression)", len(df_activity_final))
        
        logger.info(f"Saving {len(df_classification_final)} rows to Classification Dataset (Binary)...")
        df_classification_final.to_csv(classification_csv, index=False)
        tracker.record("Final: Classification Set", len(df_classification_final))
        
        if 'split' in df_classification_final.columns:
            dist = df_classification_final['split'].value_counts(normalize=True) * 100
            logger.info(f"New Classification Split Distribution: {dist.to_dict()}")
        
        if os.path.exists(eval_path):
            df_eval = pd.read_csv(eval_path)
            pre_leak = len(df_eval)
            
            keys_cls = set(zip(df_classification_final['SMILES'], df_classification_final['Target UniProt'], df_classification_final['Effector UniProt']))
            keys_act = set(zip(df_activity_final['SMILES'], df_activity_final['Target UniProt'], df_activity_final['Effector UniProt']))
            all_train_keys = keys_cls.union(keys_act)
            
            eval_s = df_eval['SMILES'].astype(str).str.strip()
            eval_t = df_eval['Target UniProt'].astype(str).str.strip()
            eval_e = df_eval['Effector UniProt'].astype(str).str.strip()
            
            leak_mask = [(s, t, e) in all_train_keys for s, t, e in zip(eval_s, eval_t, eval_e)]
            df_eval_clean = df_eval[~pd.Series(leak_mask, index=df_eval.index)].copy()
            
            if len(df_eval_clean) < pre_leak:
                logger.info(f"Leakage Control: Removed {pre_leak - len(df_eval_clean)} overlapping complexes from EvalSet.")
                df_eval_clean.to_csv(eval_path, index=False)
            else:
                logger.info("No leakage found.")
        
        log_summary("TrainSet Processor", {
            "Activity Rows": len(df_activity_final),
            "Classification Rows": len(df_classification_final),
            "Activity CSV": activity_csv,
            "Classification CSV": classification_csv
        })
        
    else:
        logger.warning("No 'Value' column found. Saving all as classification.")
        df.to_csv(classification_csv, index=False)

    tracker.record("Final Database Saved", len(df))

tracker = DatasetTracker("Training Set")

if __name__ == "__main__":
    finalize_database()
