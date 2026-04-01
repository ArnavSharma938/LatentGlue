import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from Bio import Align
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.mol_utils import canonicalize_smiles as dataset_canonicalize_smiles

DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "GlueDegradDB.csv"
DEFAULT_ACTIVITY_CSV = PROJECT_ROOT / "data" / "GlueDegradDB-Activity.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "results" / "sequence_sim"
THRESHOLDS = (0.10, 0.20, 0.30, 0.70)

def build_aligner():
    aligner = Align.PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner

def alignment_identity(alignment):
    target_coords = alignment.coordinates[0]
    query_coords = alignment.coordinates[1]
    matches = 0
    columns = 0
    for idx in range(target_coords.size - 1):
        target_start = int(target_coords[idx])
        target_end = int(target_coords[idx + 1])
        query_start = int(query_coords[idx])
        query_end = int(query_coords[idx + 1])
        target_step = target_end - target_start
        query_step = query_end - query_start
        if target_step > 0 and query_step > 0:
            if target_step != query_step:
                raise RuntimeError("Unexpected diagonal step length mismatch in alignment coordinates.")
            target_chunk = alignment.target[target_start:target_end]
            query_chunk = alignment.query[query_start:query_end]
            matches += sum(a == b for a, b in zip(target_chunk, query_chunk))
            columns += target_step
        else:
            columns += max(target_step, query_step)
    if columns <= 0:
        return 0.0
    return float(matches) / float(columns)

def top_k_identity_against_refs(query, refs, aligner, max_k):
    scored = []
    for ref in refs:
        identity = alignment_identity(aligner.align(query, ref)[0])
        scored.append((identity, ref))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top = {}
    for rank in range(1, max_k + 1):
        picked = scored[rank - 1] if len(scored) >= rank else (0.0, "")
        top[rank] = (float(picked[0]), str(picked[1]))
    return top

def compute_identity_map(queries, refs, aligner, max_k):
    result = {}
    for query in sorted(set(str(value) for value in queries)):
        result[query] = top_k_identity_against_refs(query, refs, aligner, max_k)
    return result

def canonicalize_smiles(smiles):
    canonical = dataset_canonicalize_smiles(smiles)
    if not canonical:
        return "", None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return "", None
    return canonical, mol

def build_fingerprint_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def build_train_reference_map(smiles_values, fpgen):
    refs = {}
    for smiles in sorted(set(str(value) for value in smiles_values)):
        canonical, mol = canonicalize_smiles(smiles)
        if canonical and mol is not None:
            refs.setdefault(canonical, fpgen.GetFingerprint(mol))
    if not refs:
        raise ValueError("No valid train ligands were available for fingerprint generation.")
    return refs

def top_k_similarity_against_refs(query_smiles, ref_fp_map, fpgen, max_k):
    canonical, mol = canonicalize_smiles(query_smiles)
    if not canonical or mol is None:
        return {
            rank: {"smiles": "", "valid": False, "exact": False, "ref": "", "score": 0.0}
            for rank in range(1, max_k + 1)
        }
    query_fp = fpgen.GetFingerprint(mol)
    scored = []
    for ref_smiles, ref_fp in ref_fp_map.items():
        score = float(DataStructs.TanimotoSimilarity(query_fp, ref_fp))
        scored.append((score, ref_smiles))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top = {}
    exact = bool(canonical in ref_fp_map)
    for rank in range(1, max_k + 1):
        picked = scored[rank - 1] if len(scored) >= rank else (0.0, "")
        top[rank] = {
            "smiles": canonical,
            "valid": True,
            "exact": exact and rank == 1,
            "ref": str(picked[1]),
            "score": float(picked[0]),
        }
    return top

def load_splits(train_csv_path, activity_csv_path):
    train_df = pd.read_csv(train_csv_path)
    activity_df = pd.read_csv(activity_csv_path)
    if "split" not in train_df.columns:
        raise ValueError(f"{train_csv_path} must contain a `split` column.")
    train_split_df = train_df[train_df["split"].astype(str) == "train"].reset_index(drop=True)
    val_split_df = train_df[train_df["split"].astype(str) == "val"].reset_index(drop=True)
    if len(train_split_df) == 0:
        raise ValueError("Training split is empty.")
    if len(val_split_df) == 0:
        raise ValueError("Validation split is empty.")
    if len(activity_df) == 0:
        raise ValueError("Activity CSV is empty.")
    return train_split_df, val_split_df, activity_df

def annotate(df, target_map, effector_map, ligand_map, rank):
    annotated = df.copy()
    annotated["target_score"] = annotated["Target Sequence"].astype(str).map(lambda seq: float(target_map[str(seq)][rank][0]))
    annotated["target_ref"] = annotated["Target Sequence"].astype(str).map(lambda seq: str(target_map[str(seq)][rank][1]))
    annotated["effector_score"] = annotated["Effector Sequence"].astype(str).map(lambda seq: float(effector_map[str(seq)][rank][0]))
    annotated["effector_ref"] = annotated["Effector Sequence"].astype(str).map(lambda seq: str(effector_map[str(seq)][rank][1]))
    annotated["protein_score"] = annotated[["target_score", "effector_score"]].max(axis=1)
    annotated["protein_side"] = annotated.apply(
        lambda row: "target" if float(row["target_score"]) >= float(row["effector_score"]) else "effector",
        axis=1,
    )
    annotated["protein_ref"] = annotated.apply(
        lambda row: row["target_ref"] if row["protein_side"] == "target" else row["effector_ref"],
        axis=1,
    )
    ligand_rows = [
        ligand_map[str(smiles)][rank]
        for smiles in annotated["SMILES"].astype(str).tolist()
    ]
    ligand_df = pd.DataFrame(ligand_rows)
    for column in ligand_df.columns:
        annotated[f"ligand_{column}"] = ligand_df[column]
    return annotated

def annotate_activity(df, target_map, ligand_map, rank):
    annotated = df.copy()
    annotated["target_score"] = annotated["Target Sequence"].astype(str).map(lambda seq: float(target_map[str(seq)][rank][0]))
    annotated["target_ref"] = annotated["Target Sequence"].astype(str).map(lambda seq: str(target_map[str(seq)][rank][1]))
    ligand_rows = [
        ligand_map[str(smiles)][rank]
        for smiles in annotated["SMILES"].astype(str).tolist()
    ]
    ligand_df = pd.DataFrame(ligand_rows)
    for column in ligand_df.columns:
        annotated[f"ligand_{column}"] = ligand_df[column]
    return annotated

def summarize_hits(df, col_prefix, thresholds):
    out = {}
    for threshold in thresholds:
        if col_prefix == "protein":
            target_mask = df["target_score"] >= threshold
            effector_mask = df["effector_score"] >= threshold
            either_mask = target_mask | effector_mask
            both_mask = target_mask & effector_mask
            out[f"{int(threshold * 100)}pct"] = {
                "target": int(target_mask.sum()),
                "effector": int(effector_mask.sum()),
                "either": int(either_mask.sum()),
                "both": int(both_mask.sum()),
                "below": int((~either_mask).sum()),
            }
        else:
            mask = df["ligand_score"] >= threshold
            out[f"{int(threshold * 100)}pct"] = {
                "at_or_above": int(mask.sum()),
                "below": int((~mask).sum()),
            }
    return out

def summarize_range(series):
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "max": float(series.max()),
    }

def build_split_report(df):
    return {
        "counts": {
            "rows": int(len(df)),
            "targets": int(df["Target Sequence"].astype(str).nunique()),
            "effectors": int(df["Effector Sequence"].astype(str).nunique()),
            "smiles": int(df["SMILES"].astype(str).nunique()),
            "canonical_smiles": int(df.loc[df["ligand_smiles"] != "", "ligand_smiles"].nunique()),
        },
        "protein": {
            "hits": summarize_hits(df, "protein", THRESHOLDS),
            "scores": {
                "target": summarize_range(df["target_score"]),
                "effector": summarize_range(df["effector_score"]),
                "row": summarize_range(df["protein_score"]),
            },
        },
        "ligand": {
            "hits": summarize_hits(df, "ligand", THRESHOLDS),
            "scores": summarize_range(df["ligand_score"]),
            "valid": int(df["ligand_valid"].sum()),
            "invalid": int((~df["ligand_valid"]).sum()),
            "exact": int(df["ligand_exact"].sum()),
        },
    }

def summarize_target_hits(df, thresholds):
    out = {}
    for threshold in thresholds:
        mask = df["target_score"] >= threshold
        out[f"{int(threshold * 100)}pct"] = {
            "at_or_above": int(mask.sum()),
            "below": int((~mask).sum()),
        }
    return out

def build_activity_split_report(df):
    return {
        "counts": {
            "rows": int(len(df)),
            "targets": int(df["Target Sequence"].astype(str).nunique()),
            "smiles": int(df["SMILES"].astype(str).nunique()),
            "canonical_smiles": int(df.loc[df["ligand_smiles"] != "", "ligand_smiles"].nunique()),
        },
        "target": {
            "hits": summarize_target_hits(df, THRESHOLDS),
            "scores": summarize_range(df["target_score"]),
        },
        "ligand": {
            "hits": summarize_hits(df, "ligand", THRESHOLDS),
            "scores": summarize_range(df["ligand_score"]),
            "valid": int(df["ligand_valid"].sum()),
            "invalid": int((~df["ligand_valid"]).sum()),
            "exact": int(df["ligand_exact"].sum()),
        },
    }

def build_report(train_ref_split, rank, val_df):
    return {
        "rank": int(rank),
        "train_ref_split": train_ref_split,
        "protein_method": {
            "library": "Biopython PairwiseAligner",
            "mode": "global",
            "match": 2.0,
            "mismatch": -1.0,
            "open_gap": -10.0,
            "extend_gap": -0.5,
            "identity": "matches / alignment_columns",
            "thresholds": [float(value) for value in THRESHOLDS],
        },
        "ligand_method": {
            "library": "RDKit",
            "fingerprint": "Morgan",
            "radius": 2,
            "fp_size": 2048,
            "similarity": "Tanimoto",
            "thresholds": [float(value) for value in THRESHOLDS],
            "canonicalization": "RDKit canonical non-isomeric SMILES via src.processing.mol_utils.canonicalize_smiles",
        },
        "val": build_split_report(val_df),
    }

def build_activity_summary(activity_annotated):
    valid = activity_annotated[activity_annotated["ligand_valid"]]
    wiz_mask = activity_annotated["Target"].astype(str) == "WIZ"
    cdk2_mask = activity_annotated["Target"].astype(str) == "CDK2"
    wiz_valid = activity_annotated[wiz_mask & activity_annotated["ligand_valid"]]
    cdk2_valid = activity_annotated[cdk2_mask & activity_annotated["ligand_valid"]]
    wiz_target_score = float(activity_annotated.loc[wiz_mask, "target_score"].iloc[0])
    cdk2_target_score = float(activity_annotated.loc[cdk2_mask, "target_score"].iloc[0])
    return {
        "overall": {
            "mean_ligand_tanimoto_to_train": float(valid["ligand_score"].mean()),
            "mean_target_sequence_identity_to_train": float(activity_annotated["target_score"].mean()),
        },
        "wiz_crbn": {
            "mean_ligand_tanimoto_to_train": float(wiz_valid["ligand_score"].mean()) if len(wiz_valid) > 0 else None,
            "target_sequence_identity_to_train": wiz_target_score,
        },
        "cdk2_crbn": {
            "mean_ligand_tanimoto_to_train": float(cdk2_valid["ligand_score"].mean()) if len(cdk2_valid) > 0 else None,
            "target_sequence_identity_to_train": cdk2_target_score,
        },
    }

def build_activity_report(train_ref_split, rank, activity_df):
    return {
        "rank": int(rank),
        "train_ref_split": train_ref_split,
        "target_method": {
            "library": "Biopython PairwiseAligner",
            "mode": "global",
            "match": 2.0,
            "mismatch": -1.0,
            "open_gap": -10.0,
            "extend_gap": -0.5,
            "identity": "matches / alignment_columns",
            "thresholds": [float(value) for value in THRESHOLDS],
        },
        "ligand_method": {
            "library": "RDKit",
            "fingerprint": "Morgan",
            "radius": 2,
            "fp_size": 2048,
            "similarity": "Tanimoto",
            "thresholds": [float(value) for value in THRESHOLDS],
            "canonicalization": "RDKit canonical non-isomeric SMILES via src.processing.mol_utils.canonicalize_smiles",
        },
        "activity": build_activity_split_report(activity_df),
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Write nearest-train and second-nearest-train protein/ligand similarity reports."
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--activity-csv", type=Path, default=DEFAULT_ACTIVITY_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, activity_df = load_splits(args.train_csv, args.activity_csv)
    aligner = build_aligner()
    fpgen = build_fingerprint_generator()
    max_k = 2
    train_target_refs = sorted(set(train_df["Target Sequence"].astype(str)))
    train_effector_refs = sorted(set(train_df["Effector Sequence"].astype(str)))
    train_ref_fp_map = build_train_reference_map(train_df["SMILES"], fpgen)
    val_target_map = compute_identity_map(val_df["Target Sequence"], train_target_refs, aligner, max_k)
    val_effector_map = compute_identity_map(val_df["Effector Sequence"], train_effector_refs, aligner, max_k)
    val_ligand_map = {
        str(smiles): top_k_similarity_against_refs(smiles, train_ref_fp_map, fpgen, max_k)
        for smiles in sorted(set(val_df["SMILES"].astype(str)))
    }
    activity_target_map = compute_identity_map(activity_df["Target Sequence"], train_target_refs, aligner, max_k)
    activity_ligand_map = {
        str(smiles): top_k_similarity_against_refs(smiles, train_ref_fp_map, fpgen, max_k)
        for smiles in sorted(set(activity_df["SMILES"].astype(str)))
    }
    for rank, filename in (
        (1, "nearest_train_neighbor_report.json"),
        (2, "second_nearest_train_neighbor_report.json"),
    ):
        val_annotated = annotate(val_df, val_target_map, val_effector_map, val_ligand_map, rank)
        report = build_report("GlueDegradDB split == train", rank, val_annotated)
        out_path = args.out_dir / filename
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"Wrote report to {out_path}")
    activity_annotated_rank1 = None
    for rank, filename in (
        (1, "activity_nearest_train_neighbor_report.json"),
        (2, "activity_second_nearest_train_neighbor_report.json"),
    ):
        activity_annotated = annotate_activity(activity_df, activity_target_map, activity_ligand_map, rank)
        if rank == 1:
            activity_annotated_rank1 = activity_annotated
        report = build_activity_report("GlueDegradDB split == train", rank, activity_annotated)
        out_path = args.out_dir / filename
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"Wrote report to {out_path}")

    summary = build_activity_summary(activity_annotated_rank1)
    summary_path = args.out_dir / "activity_similarity_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Wrote activity summary to {summary_path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
