import os
import re
import logging
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from src.processing.mol_utils import (
    parallel_process,
    canonicalize_smiles, DatasetTracker,
    fetch_data_and_annotations, is_missing
)

logger = logging.getLogger("EvalSet")
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

def process_evalset():
    input_path = os.path.join("data", "subdata", "MolGlueDB.csv")
    output_path = os.path.join("data", "GlueDegradDB-Eval.csv")

    TARGET_MAPPING = {
        'ALK': 'Q9UM73', 'AR': 'P10275', 'ARID2': 'Q68CP9',
        'BCL6': 'P41182', 'BRD4': 'O60885', 'BRD9': 'Q9H8M2',
        'BTK': 'Q06187', 'CDK4': 'P11802', 'CDO1': 'Q16878',
        'CK1α': 'P48729', 'CYP19A1 aromatase': 'P11511',
        'E2F2': 'Q14209', 'FIZ1': 'Q96SL8', 'GEMIN3': 'Q9UHI6',
        'GSPT1': 'P15170', 'GSPT2': 'Q8IYD1', 'IDO1': 'P14902',
        'IKZF1': 'Q13422', 'IKZF2': 'Q9UKS7', 'IKZF3': 'Q9UKT9',
        'IRF4': 'Q15306', 'LRRK2': 'Q5S007', 'Lin28': 'Q9H9Z2',
        'NEK7': 'Q8TDX7', 'NFKB1': 'P19838', 'PDE5': 'O76074',
        'PDE6D': 'O43924', 'PHGDH': 'O43175', 'PKMYT1': 'Q99640',
        'PLZF': 'Q05516', 'RBM39': 'Q14498', 'RIOK2': 'Q9BVS4',
        'SALL4': 'Q9UJQ4', 'SMARCA2': 'P51531', 'SUMO1': 'P63165',
        'WEE1': 'P30291', 'WIZ': 'O95785', 'XPO1': 'O14980',
        'ZBTB16': 'Q05516', 'ZFP91': 'Q96JP5', 'ZKSC5': 'Q9Y2L8',
        'ZNF276': 'Q8N554', 'ZNF653': 'Q96CK0', 'ZNF654': 'Q8IZM8',
        'ZNF787': 'Q6DD87', 'c-Myc': 'P01106', 'eRF1': 'P62495',
        'β-catenin': 'P35222'
    }
    cols_to_drop = [
        "IsActive", "AntiCellProliferation", "TernaryEC50", "PK", "ADMET_profile", 
        "SafetyPharmacology", "InVivoPD", "SourceAddress_Website", "PDB", 
        "glueCodeInPDB", "CryoEM", "Formula", "MolWeight", "ExactMass", 
        "clogP", "ClogP", "logS", "tPSA", "TPSA", "HBACount", "HBDCount", "RotatableBondCount", 
        "Rings", "RingCount", "AliphaticRings", "AromaticRings", "AliphaticHeteroRings", 
        "AromaticHeteroRings", "HeavyAtoms", "HeteroAtoms", "SpiroAtoms", 
        "BridgeheadAtoms", "Name", "IUPAC", "StdInChl", "StdInChIKey", 
        "Pharmacophore", "Core", "ResearchStage", "TherapeuticUsage", 
        "logP_exp", "logD_exp", "logS_exp", "KineticSolubility", 
        "ThermodynamicSolubility", "RecruitingProtein_Affinity", 
        "PrimaryTargetDegInfo", "DegronType", "SecondaryTarget", "SecondaryTargetDegInfo",
        "Scaffold", "Target Repeat", "Target Disorder", "Effector Repeat", "Effector Disorder",
        "Function", "MoA", "ModeOfAction"
    ]
    
    df = pd.read_csv(input_path)
    tracker.record("Raw Database (MolGlueDB)", len(df))
    initial_count = len(df)

    log_stage(1, 5, "Column & Row Cleanup")
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        logger.info(f"Dropped {len(existing_cols_to_drop)} initial columns.")
    
    if 'SecondaryTarget' in df.columns:
        pre_filt = len(df)
        df = df[df['SecondaryTarget'].map(is_missing)].copy()
        logger.info(f"Removed {pre_filt - len(df)} rows with SecondaryTarget data.")
        
    if 'PrimaryTarget' in df.columns:
        pre_filt = len(df)
        df = df[~df['PrimaryTarget'].str.contains(';', na=False)].copy()
        logger.info(f"Removed {pre_filt - len(df)} rows with multi-target PrimaryTarget.")

    if 'RecruitingProtein' in df.columns:
        pre_filt = len(df)
        df = df[~df['RecruitingProtein'].str.contains(';', na=False)].copy()
        logger.info(f"Removed {pre_filt - len(df)} rows with multi-effector RecruitingProtein.")

    for up_col in ['PrimaryTarget_UniProtID', 'RecruitingProtein_UniProtID']:
        if up_col in df.columns:
            pre_filt = len(df)
            df = df[~df[up_col].str.contains(r'[;/]', na=False)].copy()
            logger.info(f"Removed {pre_filt - len(df)} rows with multi-ID in {up_col}.")
            label = "Unique Primary Target" if "Primary" in up_col else "Unique Ligase ID"
            tracker.record(label, len(df))

    excluded_targets = ["CDK12-CCNK(cyclin K)", "AUX/IAA", "JAZ degron"]
    if 'PrimaryTarget' in df.columns:
        for target in excluded_targets:
            pre_filt = len(df)
            df = df[~df['PrimaryTarget'].str.contains(re.escape(target), na=False, case=False)].copy()
            logger.info(f"Removed {pre_filt - len(df)} rows with excluded target '{target}'.")

        df['PrimaryTarget'] = df['PrimaryTarget'].str.replace(r'\(aromatase\)', ' aromatase', regex=True, case=False)
        df['PrimaryTarget'] = df['PrimaryTarget'].str.replace(r'\s*\([^)]*\)', '', regex=True)
        df['PrimaryTarget'] = df['PrimaryTarget'].str.strip()
        logger.info("Cleaned parentheticals from PrimaryTarget.")

        df['PrimaryTarget_UniProtID'] = df['PrimaryTarget'].map(TARGET_MAPPING.get)
        pre_drop = len(df)
        df = df.dropna(subset=['PrimaryTarget_UniProtID'])
        logger.info(f"Mapped PrimaryTarget to UniProt ID. Removed {pre_drop - len(df)} unmapped targets.")
        tracker.record("UniProt Mapped", len(df))

    log_stage(2, 5, "SMILES Standardization")
    if 'SMILES' in df.columns:
        logger.info("Standardizing SMILES...")
        df.loc[:, 'SMILES'] = parallel_process(canonicalize_smiles, df['SMILES'].tolist())
        pre_filt = len(df)
        df = df[df['SMILES'] != ""].copy()
        logger.info(f"Canonicalized SMILES. Removed {pre_filt - len(df)} invalid SMILES.")
        tracker.record("Standardized SMILES", len(df))

    log_stage(3, 5, "Early Deduplication")
    subset_cols = ['SMILES', 'PrimaryTarget_UniProtID', 'RecruitingProtein_UniProtID']
    existing_subset = [c for c in subset_cols if c in df.columns]
    if len(existing_subset) == 3:
        pre_dedup = len(df)
        df = df.drop_duplicates(subset=existing_subset, keep='first')
        logger.info(f"Early deduplication removed {pre_dedup - len(df)} rows.")
        tracker.record("Unique Architectures", len(df))
    else:
        logger.warning("Skipping early deduplication: Not all required columns are present.")

    log_stage(4, 5, "External Annotations")
    
    # Filter out known AlphaFold failures
    af_excludes = {'A9UF02', 'P42858'}
    if 'PrimaryTarget_UniProtID' in df.columns:
        df = df[~df['PrimaryTarget_UniProtID'].isin(af_excludes)].copy()
    if 'RecruitingProtein_UniProtID' in df.columns:
        df = df[~df['RecruitingProtein_UniProtID'].isin(af_excludes)].copy()

    unique_ids = set()
    if 'PrimaryTarget_UniProtID' in df.columns:
        unique_ids.update(df['PrimaryTarget_UniProtID'].dropna().unique())
    if 'RecruitingProtein_UniProtID' in df.columns:
        unique_ids.update(df['RecruitingProtein_UniProtID'].dropna().unique())
    
    data_map = fetch_data_and_annotations(unique_ids)
    
    df['Target Sequence'] = df['PrimaryTarget_UniProtID'].map(lambda x: data_map.get(x, {}).get("sequence"))
    df['Effector Sequence'] = df['RecruitingProtein_UniProtID'].map(lambda x: data_map.get(x, {}).get("sequence"))
    
    pre_filt = len(df)
    df = df[~df['Target Sequence'].map(is_missing)]
    df = df[~df['Effector Sequence'].map(is_missing)]
    logger.info(f"Filtered for Human-only and successful sequence retrieval. Removed {pre_filt - len(df)} rows.")
    tracker.record("Human Bio-Validated", len(df))


    log_stage(5, 5, "Final Polish & Column Renaming")
    rename_map = {
        'DATAID': 'Compound ID',
        'PrimaryTarget': 'Target',
        'PrimaryTarget_UniProtID': 'Target UniProt',
        'RecruitingProtein': 'Effector',
        'RecruitingProtein_UniProtID': 'Effector UniProt'
    }
    df = df.rename(columns=rename_map)

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    df.to_csv(output_path, index=False)
    log_summary("EvalSet Processor", {
        "Initial Entries": initial_count,
        "Final Entries": len(df),
        "Total Removed": initial_count - len(df)
    })
    tracker.record("Evolution Set Saved", len(df))

tracker = DatasetTracker("Evaluation Set")

if __name__ == "__main__":
    process_evalset()
