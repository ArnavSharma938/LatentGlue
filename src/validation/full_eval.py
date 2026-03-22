"""
1. Component effective dimensionality
   - compute spectral effective dimension for target, effector, and ligand
     embeddings from the checkpointed encoder on both the GlueDegradDB val split
     and GlueDegradDB-Eval.csv
   - compare three representation families per component: frozen mean-pooled
     backbones, projected mean-pooled features, and LatentGlue pooled features

2. Activity prediction
   - build one vector per ternary from the checkpointed encoder
   - fit simple downstream probes with target-balanced cross-validation
   - report whether the learned representation supports potency prediction on
     held-out folds
   Baselines used here are:
   - `activity/frozen_feature_*`: concatenated mean-pooled frozen backbone
     features
   - `activity/baseline_*`: concatenated mean-pooled projected token features
   - `activity/latentglue_*`: concatenated attention-pooled target / effector /
     ligand summaries from the trained encoder

3. Retrieval
   - build fixed-context retrieval tasks from the GlueDegradDB `val` split and
     `GlueDegradDB-Eval.csv`
   - positives are observed ligands for a target-effector context
   - negatives are ligands sampled from other contexts
   - compare `frozen_mean`, `projected_mean`, and `latentglue` under the same
     grouped context-holdout protocol
   - fit the same low-rank bilinear scorer for each representation family
   - report per-context and macro AUROC/AUPRC as a fair representation-level
     retrieval evaluation

4. Eval-set ligand attention
   - compute ligand-pooling attention over the full set of unique ligands in
     `GlueDegradDB-Eval.csv`
   - summarize attention concentration and spread across the eval ligand set
   - render a representative panel grid spanning molecule size and attention
     diffuseness bins, with one representative ligand per occupied bin

Data used:
- section 1: `data/GlueDegradDB.csv` with `split == "val"` and `data/GlueDegradDB-Eval.csv`
- section 2: `data/GlueDegradDB-Activity.csv`
- section 3: `data/GlueDegradDB.csv` with `split == "val"` and `data/GlueDegradDB-Eval.csv`
- section 4: `data/GlueDegradDB-Eval.csv`

Run this first:
sudo apt-get update
sudo apt-get install -y libxrender1
sudo apt-get install -y libxext6 libsm6
"""

import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.model.dataset import collate_ternary
from src.model.model import LatentGlueEncoder
from src.model.train import AUTOCAST_DTYPE
from src.validation.in_train_eval import (activity_targets, build_target_balanced_folds, masked_mean_pool, spectral_effective_dimension,
)

DEFAULT_DEVICE = "cuda"
DEFAULT_TRAIN_CSV = "data/GlueDegradDB.csv"
DEFAULT_EVAL_CSV = "data/GlueDegradDB-Eval.csv"
DEFAULT_ACTIVITY_CSV = "data/GlueDegradDB-Activity.csv"
DEFAULT_OUTPUT_JSON = "data/results/full_eval.json"
DEFAULT_ATTENTION_FIGURE_PATH = "data/results/ligand_attention.png"
DEFAULT_EFFECTIVE_DIM_FIGURE_PATH = "data/results/effective_dim_eval.png"
DEFAULT_ACTIVITY_FIGURE_PATH = "data/results/activity_prediction.png"
DEFAULT_RETRIEVAL_FIGURE_PATH = "data/results/retrieval.png"
DEFAULT_CHECKPOINT_URL = "https://huggingface.co/ArnavSharma938/LatentGlue"
DEFAULT_CHECKPOINT_FILENAME = "LatentGlue.pt"
COMPONENT_NAMES = ("target", "effector", "ligand")
TERNARY_COLUMNS = ("Target Sequence", "Effector Sequence", "SMILES")
EVAL_REQUIRED_COLUMNS = (
    "Compound ID",
    "SMILES",
    "Target",
    "Effector",
    "Effector UniProt",
    "Target Sequence",
    "Effector Sequence",
)
# `val` is bounded by the USP28/FBW7 context; `eval` retains the original budget.
RETRIEVAL_NEGATIVES_PER_CONTEXT_VAL = 45
RETRIEVAL_NEGATIVES_PER_CONTEXT_EVAL = 64
RETRIEVAL_BATCH_SIZE = 32
RETRIEVAL_N_FOLDS = 5
RETRIEVAL_BILINEAR_RANK = 64
RETRIEVAL_TRAIN_EPOCHS = 250
RETRIEVAL_PATIENCE = 30
RETRIEVAL_LR = 1e-3
RETRIEVAL_WEIGHT_DECAY = 1e-4
RETRIEVAL_SEED = 17
ATTENTION_PANEL_COLUMNS = 3
ATTENTION_PANEL_SIZE_LABELS = ("small", "medium", "large")
ATTENTION_PANEL_FOCUS_LABELS = ("focused", "balanced", "diffuse")
SUMMARY_FIGURE_DPI = 300
PLOT_COLORS = {
    "Frozen": "#7A7A7A",
    "Projected": "#4C78A8",
    "LatentGlue": "#E07A5F",
}

def read_df(path, required=(), split=None):
    path = str(path or "").strip()
    if not path:
        raise ValueError("CSV path must be provided.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    if split is not None:
        if "split" not in df.columns:
            raise ValueError(f"{path} is missing required column: split")
        df = df[df["split"].astype(str) == str(split)].reset_index(drop=True)
    if len(df) == 0:
        detail = f" after filtering split={split}" if split is not None else ""
        raise ValueError(f"{path} is empty{detail}.")
    return df

def load_models(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = LatentGlueEncoder(device=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    encoder.eval()
    return encoder

def parse_hf_repo_id(checkpoint_path):
    checkpoint_path = str(checkpoint_path).strip()
    if checkpoint_path.startswith("hf://"):
        repo_id = checkpoint_path[len("hf://") :]
        return repo_id if repo_id.count("/") == 1 else ""
    if (
        checkpoint_path.count("/") == 1
        and "\\" not in checkpoint_path
        and not checkpoint_path.startswith(".")
        and not os.path.isabs(checkpoint_path)
        and " " not in checkpoint_path
    ):
        return checkpoint_path
    parsed = urlparse(checkpoint_path)
    if parsed.scheme in {"http", "https"} and parsed.netloc.endswith("huggingface.co"):
        parts = [part for part in parsed.path.split("/") if part]
        return "/".join(parts[:2]) if len(parts) >= 2 else ""
    return ""

def resolve_checkpoint_path(checkpoint_path):
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        raise ValueError(
            "Checkpoint path must be provided as a local file, a Hugging Face URL, or an hf://owner/repo reference."
        )
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path
    repo_id = parse_hf_repo_id(checkpoint_path)
    if repo_id:
        return hf_hub_download(repo_id=repo_id, filename=DEFAULT_CHECKPOINT_FILENAME)
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. Use a local file, a Hugging Face repo id "
        f"like owner/repo, a Hugging Face URL, or hf://owner/repo."
    )

def trim_special_tokens(x, mask):
    x, mask = x[:, 1:, :], mask[:, 1:].clone()
    eos = mask.sum(dim=1) - 1
    mask[torch.arange(x.size(0), device=x.device), eos] = False
    return x, mask

def concat_components(components):
    return torch.cat(list(components), dim=-1).numpy()

def resolve_batch_size(batch_size, n_rows):
    if n_rows <= 0:
        raise ValueError("Batch size cannot be resolved for an empty dataset.")
    if batch_size is None:
        raise ValueError("batch_size must be a positive integer.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return batch_size

def ensure_output_dir(output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

def load_matplotlib():
    import sys
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt

def sample_standard_deviation(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1))

def activity_probe_cv_with_folds(features, values, targets, alpha=1.0, n_folds=5):
    features = np.asarray(features, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets)
    spearman_scores = []
    rmse_scores = []
    for fold_indices in build_target_balanced_folds(targets, values, n_folds=n_folds):
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[fold_indices] = False
        if train_mask.sum() == 0 or fold_indices.size == 0:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(features[train_mask], values[train_mask])
        predictions = model.predict(features[fold_indices])
        spearman_score = spearmanr(values[fold_indices], predictions).statistic
        if spearman_score is None or np.isnan(spearman_score):
            raise RuntimeError(
                "Activity evaluation produced an undefined Spearman correlation for at least one fold."
            )
        spearman_scores.append(float(spearman_score))
        rmse_scores.append(float(np.sqrt(mean_squared_error(values[fold_indices], predictions))))
    if not spearman_scores:
        raise RuntimeError("Activity evaluation did not produce any valid cross-validation folds.")
    return {
        "spearman": float(np.mean(spearman_scores)),
        "rmse": float(np.mean(rmse_scores)),
        "spearman_folds": [float(score) for score in spearman_scores],
        "rmse_folds": [float(score) for score in rmse_scores],
        "spearman_std": sample_standard_deviation(spearman_scores),
        "rmse_std": sample_standard_deviation(rmse_scores),
        "n_folds": float(len(spearman_scores)),
    }

def summarize_activity_cv_representation(display_name, probe_metrics):
    return {
        "display_name": str(display_name),
        "spearman": float(probe_metrics["spearman"]),
        "rmse": float(probe_metrics["rmse"]),
        "spearman_std": float(probe_metrics["spearman_std"]),
        "rmse_std": float(probe_metrics["rmse_std"]),
        "spearman_folds": [float(score) for score in probe_metrics["spearman_folds"]],
        "rmse_folds": [float(score) for score in probe_metrics["rmse_folds"]],
        "n_folds": float(probe_metrics["n_folds"]),
    }

class InMemoryTernaryDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.target_seqs = self.df["Target Sequence"].astype(str).tolist()
        self.effector_seqs = self.df["Effector Sequence"].astype(str).tolist()
        self.smiles = self.df["SMILES"].astype(str).str.strip().tolist()
        if "Effector UniProt" in self.df.columns:
            self.effector_bucket_keys = self.df["Effector UniProt"].astype(str).tolist()
        else:
            self.effector_bucket_keys = list(self.effector_seqs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "target_seq": self.target_seqs[idx],
            "effector_seq": self.effector_seqs[idx],
            "smiles": self.smiles[idx],
            "effector_id": self.effector_bucket_keys[idx],
        }

@torch.no_grad()
def collect_representations(df, encoder, batch_size, autocast_enabled=True):
    loader = DataLoader(
        InMemoryTernaryDataset(df),
        batch_size=resolve_batch_size(batch_size, len(df)),
        shuffle=False,
        collate_fn=collate_ternary,
    )
    reps = {name: [[] for _ in COMPONENT_NAMES] for name in ("latentglue", "projected_mean", "frozen_mean")}
    for batch in loader:
        cache_keys = (batch["target_seq"], batch["effector_seq"], batch["smiles"])
        with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
            prepared = encoder.prepare_inputs(batch["target_seq"], batch["effector_seq"], batch["smiles"])
            target_toks, target_mask, effector_toks, effector_mask, ligand_toks, ligand_mask = prepared
            cached_backbones = (
                encoder._get_cached_teacher_backbone_batch(0, cache_keys[0], target_toks, target_mask),
                encoder._get_cached_teacher_backbone_batch(1, cache_keys[1], effector_toks, effector_mask),
                encoder._get_cached_teacher_backbone_batch(2, cache_keys[2], ligand_toks, ligand_mask),
            )
            components, pooled, masks, _ = encoder(
                target_toks,
                effector_toks,
                ligand_toks,
                target_mask,
                effector_mask,
                ligand_mask,
                cached_backbones=cached_backbones,
                compute_pools=True,
            )
        projected = tuple(masked_mean_pool(component, mask) for component, mask in zip(components, masks))
        frozen = tuple(masked_mean_pool(*trim_special_tokens(backbone, mask)) for backbone, mask in zip(cached_backbones, (target_mask, effector_mask, ligand_mask)))
        for name, values in (("latentglue", pooled), ("projected_mean", projected), ("frozen_mean", frozen)):
            for idx, value in enumerate(values):
                reps[name][idx].append(value.detach().float().cpu())
    return {name: tuple(torch.cat(chunks, dim=0) for chunks in groups) for name, groups in reps.items()}

def evaluate_effective_dimensions(reps, dataset_name):
    frozen_names = ("esmc_mean_pool", "esmc_mean_pool", "molformer_xl_mean_pool")
    projected_names = ("projected_mean_target", "projected_mean_effector", "projected_mean_ligand")
    metrics = {}
    for idx, component in enumerate(COMPONENT_NAMES):
        latent_key = f"effective_dim/{dataset_name}/latentglue_{component}"
        frozen_key = f"effective_dim/{dataset_name}/{frozen_names[idx]}_{component}"
        projected_key = f"effective_dim/{dataset_name}/{projected_names[idx]}"
        metrics[latent_key] = spectral_effective_dimension(reps["latentglue"][idx])
        metrics[frozen_key] = spectral_effective_dimension(reps["frozen_mean"][idx])
        metrics[projected_key] = spectral_effective_dimension(reps["projected_mean"][idx])
    return metrics

def evaluate_activity(activity_df, reps):
    if activity_df is None:
        raise ValueError("Activity dataframe must be provided.")
    missing_columns = [column for column in ("Value", "Target") if column not in activity_df.columns]
    if missing_columns:
        raise ValueError(f"Activity dataframe is missing required columns: {missing_columns}")
    y, targets = activity_targets(activity_df)
    representation_specs = (
        ("frozen_feature", "Frozen", "activity/frozen_feature", concat_components(reps["frozen_mean"])),
        ("baseline", "Projected", "activity/baseline", concat_components(reps["projected_mean"])),
        ("latentglue", "LatentGlue", "activity/latentglue", concat_components(reps["latentglue"])),
    )
    metrics = {}
    activity_cv = {
        "split_strategy": "target-balanced 5-fold cross-validation on GlueDegradDB-Activity",
        "error_bar": "sample standard deviation across folds",
        "representations": {},
    }
    for rep_key, display_name, prefix, features in representation_specs:
        probe_metrics = activity_probe_cv_with_folds(features, y, targets, n_folds=5)
        metrics[f"{prefix}_spearman"] = float(probe_metrics["spearman"])
        metrics[f"{prefix}_rmse"] = float(probe_metrics["rmse"])
        activity_cv["representations"][rep_key] = summarize_activity_cv_representation(display_name, probe_metrics)
    activity_cv["n_folds"] = float(
        activity_cv["representations"]["latentglue"]["n_folds"]
    )
    metrics["activity_cv"] = activity_cv
    return metrics

def get_representations(repr_cache, cache_key, df, encoder, batch_size, autocast_enabled):
    if df is None:
        raise ValueError(f"Dataframe for cache key `{cache_key}` must be provided.")
    if cache_key not in repr_cache:
        repr_cache[cache_key] = collect_representations(
            df,
            encoder,
            batch_size,
            autocast_enabled=autocast_enabled,
        )
    return repr_cache[cache_key]

def canonicalize_smiles(smiles):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)

def smiles_atom_spans(smiles):
    spans = []
    i = 0
    two_char_atoms = {"Al", "Br", "Ca", "Cl", "Li", "Na", "Se", "Si"}
    single_char_atoms = set("BCFNOPSIbcnops")
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            end = smiles.find("]", i)
            if end < 0:
                raise ValueError(f"Unclosed bracket atom in SMILES: {smiles}")
            spans.append((i, end + 1))
            i = end + 1
            continue
        if i + 1 < len(smiles) and smiles[i : i + 2] in two_char_atoms:
            spans.append((i, i + 2))
            i += 2
            continue
        if ch in single_char_atoms:
            spans.append((i, i + 1))
        i += 1
    return spans

def distribute_token_weight_to_atoms(token_start, token_end, atom_spans, token_weight, atom_weights):
    overlaps = [
        atom_idx
        for atom_idx, (atom_start, atom_end) in enumerate(atom_spans)
        if atom_start < token_end and token_start < atom_end
    ]
    if overlaps:
        share = float(token_weight) / float(len(overlaps))
        for atom_idx in overlaps:
            atom_weights[atom_idx] += share
        return True

    # SMILES tokenizers emit syntax tokens such as bond markers, ring digits,
    # and parentheses. These do not correspond to atoms, so they are excluded
    # from the atom-level projection.
    return False

def token_weights_to_atom_weights(smiles, token_offsets, token_weights):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to parse SMILES for atom mapping: {smiles}")

    atom_spans = smiles_atom_spans(smiles)
    if len(atom_spans) != mol.GetNumAtoms():
        raise RuntimeError(
            f"SMILES atom span mismatch for {smiles}: parsed {len(atom_spans)} spans vs RDKit {mol.GetNumAtoms()} atoms."
        )

    atom_weights = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
    assigned_weight = 0.0
    for (start, end), weight in zip(token_offsets, token_weights):
        weight = float(weight)
        was_assigned = distribute_token_weight_to_atoms(int(start), int(end), atom_spans, weight, atom_weights)
        if was_assigned:
            assigned_weight += weight
    if assigned_weight <= 0.0:
        raise RuntimeError(f"No atom-overlapping token weights were available for {smiles}.")
    return mol, atom_weights

def extract_seed_attention_weights(pool_module, x, mask=None):
    query = pool_module.query_norm(pool_module.seed.expand(x.size(0), -1, -1))
    key_value = pool_module.key_norm(x)
    key_padding_mask = None if mask is None else ~mask.bool()
    pooled, weights = pool_module.attn(
        query,
        key_value,
        key_value,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=True,
    )
    return pooled[:, 0, :], weights[:, 0, :]

@torch.no_grad()
def compute_ligand_attention(encoder, smiles, autocast_enabled):
    canonical_smiles = canonicalize_smiles(smiles)
    tokenized = encoder.mol_tokenizer(
        [canonical_smiles],
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    ligand_toks = tokenized["input_ids"].to(encoder.device)
    ligand_mask_raw = tokenized["attention_mask"].bool().to(encoder.device)

    with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
        ligand_tokens, ligand_mask, _ = encoder.forward_ligand(
            ligand_toks,
            ligand_mask_raw,
            role_id=2,
        )
        _, token_weights = extract_seed_attention_weights(encoder.ligand_pool, ligand_tokens, mask=ligand_mask)
        token_weights = token_weights[0, ligand_mask[0]].detach().float().cpu().numpy().astype(np.float32)

    token_offsets = [
        (int(start), int(end))
        for start, end in tokenized["offset_mapping"][0].tolist()
        if int(end) > int(start)
    ]
    if len(token_offsets) != len(token_weights):
        raise RuntimeError(
            f"Token offset mismatch for {canonical_smiles}: got {len(token_offsets)} offsets vs {len(token_weights)} attention weights."
        )

    mol, atom_weights = token_weights_to_atom_weights(canonical_smiles, token_offsets, token_weights)
    total_weight = float(atom_weights.sum())
    if total_weight > 0:
        atom_weights = atom_weights / total_weight
    return {
        "smiles": canonical_smiles,
        "mol": mol,
        "token_weights": token_weights,
        "token_offsets": token_offsets,
        "atom_weights": atom_weights,
    }

def compute_attention_distribution_metrics(atom_weights):
    atom_weights = np.asarray(atom_weights, dtype=np.float32)
    if atom_weights.ndim != 1 or atom_weights.size == 0:
        raise ValueError("atom_weights must be a non-empty 1D vector.")
    total = float(atom_weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("atom_weights must sum to a positive finite value.")

    probs = (atom_weights / total).astype(np.float32)
    positive_probs = probs[probs > 0]
    entropy = float(-(positive_probs * np.log(positive_probs)).sum())
    max_entropy = float(np.log(len(probs))) if len(probs) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
    effective_atom_count = float(np.exp(entropy))
    participation_ratio = float(1.0 / np.square(probs).sum())
    sorted_probs = np.sort(probs)[::-1]

    return {
        "atom_count": float(len(probs)),
        "top1_atom_weight": float(sorted_probs[0]),
        "top3_atom_weight": float(sorted_probs[: min(3, len(sorted_probs))].sum()),
        "top5_atom_weight": float(sorted_probs[: min(5, len(sorted_probs))].sum()),
        "entropy": entropy,
        "normalized_entropy": float(normalized_entropy),
        "effective_atom_count": effective_atom_count,
        "effective_atom_fraction": float(effective_atom_count / len(probs)),
        "participation_ratio": participation_ratio,
    }

def build_eval_ligand_attention_records(eval_df):
    missing_columns = [column for column in ("Compound ID", "SMILES", "Target", "Effector") if column not in eval_df.columns]
    if missing_columns:
        raise ValueError(f"Eval ligand attention requires columns: {missing_columns}")

    unique_records = {}
    for row in eval_df.to_dict("records"):
        canonical_smiles = canonicalize_smiles(row["SMILES"])
        record = unique_records.get(canonical_smiles)
        if record is None:
            unique_records[canonical_smiles] = {
                "compound_ids": {str(row["Compound ID"]).strip()},
                "targets": {str(row["Target"]).strip()},
                "effectors": {str(row["Effector"]).strip()},
                "smiles": canonical_smiles,
                "eval_row_count": 1,
            }
            continue

        record["eval_row_count"] += 1
        record["compound_ids"].add(str(row["Compound ID"]).strip())
        record["targets"].add(str(row["Target"]).strip())
        record["effectors"].add(str(row["Effector"]).strip())

    ordered_records = []
    for canonical_smiles, record in unique_records.items():
        compound_ids = sorted(value for value in record["compound_ids"] if value)
        targets = sorted(value for value in record["targets"] if value)
        effectors = sorted(value for value in record["effectors"] if value)
        ordered_records.append(
            {
                "compound_id": compound_ids[0] if compound_ids else "",
                "compound_ids": compound_ids,
                "targets": targets,
                "effectors": effectors,
                "smiles": canonical_smiles,
                "eval_row_count": int(record["eval_row_count"]),
            }
        )
    ordered_records.sort(key=lambda item: (item["compound_id"], item["smiles"]))
    return ordered_records

def assign_attention_quantile_bins(values, labels):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.asarray([], dtype=object)
    n_bins = min(len(labels), int(values.size))
    rank_series = pd.Series(values).rank(method="first")
    quantile_bins = pd.qcut(rank_series, q=n_bins, labels=labels[:n_bins])
    return quantile_bins.astype(str).to_numpy()

def summarize_attention_metric_distribution(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty attention metric distribution.")
    quantiles = np.quantile(values, [0.1, 0.5, 0.9]).astype(np.float32)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q10": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q90": float(quantiles[2]),
    }

def select_representative_attention_panels(panel_data):
    if not panel_data:
        raise ValueError("panel_data must contain at least one ligand panel.")

    metrics_df = pd.DataFrame(
        {
            "panel_idx": np.arange(len(panel_data), dtype=np.int64),
            "atom_count": [float(panel["atom_count"]) for panel in panel_data],
            "normalized_entropy": [float(panel["normalized_entropy"]) for panel in panel_data],
        }
    )
    metrics_df["size_bin"] = assign_attention_quantile_bins(metrics_df["atom_count"].to_numpy(), ATTENTION_PANEL_SIZE_LABELS)
    metrics_df["focus_bin"] = assign_attention_quantile_bins(metrics_df["normalized_entropy"].to_numpy(), ATTENTION_PANEL_FOCUS_LABELS)

    for row in metrics_df.itertuples(index=False):
        panel_data[int(row.panel_idx)]["size_bin"] = str(row.size_bin)
        panel_data[int(row.panel_idx)]["focus_bin"] = str(row.focus_bin)

    atom_scale = max(float(metrics_df["atom_count"].std(ddof=0)), 1.0)
    entropy_scale = max(float(metrics_df["normalized_entropy"].std(ddof=0)), 1e-6)

    selected_indices = []
    for focus_bin in ATTENTION_PANEL_FOCUS_LABELS:
        for size_bin in ATTENTION_PANEL_SIZE_LABELS:
            cell = metrics_df[
                (metrics_df["focus_bin"] == focus_bin) &
                (metrics_df["size_bin"] == size_bin)
            ]
            if cell.empty:
                continue
            atom_center = float(cell["atom_count"].median())
            entropy_center = float(cell["normalized_entropy"].median())
            distances = (
                np.square((cell["atom_count"] - atom_center) / atom_scale) +
                np.square((cell["normalized_entropy"] - entropy_center) / entropy_scale)
            )
            selected_panel_idx = int(cell.loc[distances.idxmin(), "panel_idx"])
            panel = panel_data[selected_panel_idx]
            panel["size_bin"] = size_bin
            panel["focus_bin"] = focus_bin
            selected_indices.append(selected_panel_idx)

    return [panel_data[idx] for idx in selected_indices]

def render_molecule_attention_image(mol, atom_weights, norm, cmap, width=420, height=320):
    import io
    import matplotlib.image as mpimg
    from rdkit.Chem.Draw import rdMolDraw2D

    prepared = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.padding = 0.04

    highlight_atoms = list(range(prepared.GetNumAtoms()))
    highlight_colors = {}
    highlight_radii = {}
    for atom_idx in highlight_atoms:
        rgba = cmap(norm(float(atom_weights[atom_idx])))
        highlight_colors[atom_idx] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
        highlight_radii[atom_idx] = 0.32 + 0.18 * float(norm(float(atom_weights[atom_idx])))

    drawer.DrawMolecule(
        prepared,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    return mpimg.imread(io.BytesIO(drawer.GetDrawingText()), format="png")

def save_ligand_attention_figure(panel_data, output_path, title):
    matplotlib, plt = load_matplotlib()
    from matplotlib import colors

    if not panel_data:
        raise ValueError("panel_data must contain at least one ligand panel.")

    ensure_output_dir(output_path)

    cmap = matplotlib.colormaps.get_cmap("coolwarm")

    n_cols = min(ATTENTION_PANEL_COLUMNS, len(panel_data))
    n_rows = int(np.ceil(len(panel_data) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows))
    try:
        axes = np.asarray(axes, dtype=object).reshape(-1)

        for col_idx, panel in enumerate(panel_data):
            ax = axes[col_idx]
            panel_weights = np.asarray(panel["atom_weights"], dtype=np.float32)
            if not panel_weights.size:
                raise ValueError(
                    f"Ligand attention figure cannot be generated from empty atom weights for {panel['ligand_name']}."
                )
            vmin = float(panel_weights.min())
            vmax = float(panel_weights.max())
            if vmax <= vmin:
                vmax = vmin + 1e-6
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            image = render_molecule_attention_image(panel["mol"], panel_weights, norm=norm, cmap=cmap)
            ax.imshow(image)
            ax.axis("off")
            subtitle = panel.get("panel_subtitle", "")
            ax.set_title(
                panel["ligand_name"] if not subtitle else f"{panel['ligand_name']}\n{subtitle}",
                fontsize=10,
            )

            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.046, pad=0.03)
            colorbar.set_label("Relative attention")

        for ax in axes[len(panel_data) :]:
            ax.axis("off")

        fig.suptitle(title, fontsize=12)
        fig.text(0.5, 0.92, "Each ligand panel uses its own color scale.", ha="center", fontsize=9)
        fig.subplots_adjust(left=0.03, right=0.98, bottom=0.06, top=0.88, wspace=0.25, hspace=0.35)
        fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)

def evaluate_ligand_attention(encoder, eval_df, autocast_enabled, output_path):
    if not output_path:
        raise ValueError("attention_figure_path must be provided.")
    ligand_records = build_eval_ligand_attention_records(eval_df)
    if not ligand_records:
        raise RuntimeError("Eval ligand attention did not find any unique ligands.")

    panel_data = []
    for record in ligand_records:
        panel = compute_ligand_attention(
            encoder=encoder,
            smiles=record["smiles"],
            autocast_enabled=autocast_enabled,
        )
        panel.update(record)
        panel.update(compute_attention_distribution_metrics(panel["atom_weights"]))
        panel["ligand_name"] = f"CID {record['compound_id']}" if record["compound_id"] else "Unlabeled ligand"
        panel["panel_subtitle"] = ""
        panel_data.append(panel)

    representative_panels = select_representative_attention_panels(panel_data)
    for panel in representative_panels:
        panel["panel_subtitle"] = f"{panel['size_bin']} | {panel['focus_bin']}"

    attention_bin_counts = {
        focus_bin: {
            size_bin: float(
                sum(
                    panel["focus_bin"] == focus_bin and panel["size_bin"] == size_bin
                    for panel in panel_data
                )
            )
            for size_bin in ATTENTION_PANEL_SIZE_LABELS
        }
        for focus_bin in ATTENTION_PANEL_FOCUS_LABELS
    }
    occupied_bin_count = float(
        sum(
            count > 0.0
            for focus_counts in attention_bin_counts.values()
            for count in focus_counts.values()
        )
    )

    summary = {
        "atom_count": summarize_attention_metric_distribution([panel["atom_count"] for panel in panel_data]),
        "top1_atom_weight": summarize_attention_metric_distribution([panel["top1_atom_weight"] for panel in panel_data]),
        "top3_atom_weight": summarize_attention_metric_distribution([panel["top3_atom_weight"] for panel in panel_data]),
        "normalized_entropy": summarize_attention_metric_distribution([panel["normalized_entropy"] for panel in panel_data]),
        "effective_atom_fraction": summarize_attention_metric_distribution([panel["effective_atom_fraction"] for panel in panel_data]),
        "participation_ratio": summarize_attention_metric_distribution([panel["participation_ratio"] for panel in panel_data]),
    }

    result = {
        "dataset_name": "GlueDegradDB-Eval",
        "num_eval_rows": float(len(eval_df)),
        "num_unique_ligands": float(len(panel_data)),
        "representative_panel_count": float(len(representative_panels)),
        "occupied_bin_count": occupied_bin_count,
        "figure_path": output_path,
        "summary": summary,
        "attention_bin_counts": attention_bin_counts,
        "representative_panels": [
            {
                "compound_id": panel["compound_id"],
                "smiles": panel["smiles"],
                "targets": panel["targets"],
                "effectors": panel["effectors"],
                "eval_row_count": float(panel["eval_row_count"]),
                "atom_count": float(panel["atom_count"]),
                "top1_atom_weight": float(panel["top1_atom_weight"]),
                "top3_atom_weight": float(panel["top3_atom_weight"]),
                "normalized_entropy": float(panel["normalized_entropy"]),
                "effective_atom_fraction": float(panel["effective_atom_fraction"]),
                "participation_ratio": float(panel["participation_ratio"]),
                "size_bin": panel["size_bin"],
                "focus_bin": panel["focus_bin"],
            }
            for panel in representative_panels
        ],
    }
    save_ligand_attention_figure(
        representative_panels,
        output_path=output_path,
        title="Representative ligand attention across GlueDegradDB-Eval",
    )
    return result

def format_figure_value(value):
    value = float(value)
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.3f}"

def compute_soft_zoom_limits(values, step, lower_bound=0.0, lower_pad=0.45, upper_pad=0.18, min_span=1.0):
    values = np.asarray(values, dtype=np.float32)
    value_min = float(values.min())
    value_max = float(values.max())
    span = max(value_max - value_min, float(min_span))
    lower = max(float(lower_bound), value_min - float(lower_pad) * span)
    upper = value_max + float(upper_pad) * span
    step = float(step)
    lower = float(step * np.floor(lower / step))
    upper = float(step * np.ceil(upper / step))
    if upper <= lower:
        upper = lower + step
    return lower, upper

def effective_dimension_metric_key(dataset_name, representation_key, component):
    if representation_key == "frozen_mean":
        backbone_name = "molformer_xl_mean_pool" if component == "ligand" else "esmc_mean_pool"
        return f"effective_dim/{dataset_name}/{backbone_name}_{component}"
    return f"effective_dim/{dataset_name}/{representation_key}_{component}"

def hide_chart_spines(ax):
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)

def annotate_bar_values(ax, bars, values, offset):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            format_figure_value(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

def save_effective_dimensionality_figure(metrics, output_path):
    if not output_path:
        raise ValueError("effective_dim_figure_path must be provided.")
    _, plt = load_matplotlib()
    ensure_output_dir(output_path)

    family_specs = (
        ("Frozen", "frozen_mean"),
        ("Projected", "projected_mean"),
        ("LatentGlue", "latentglue"),
    )
    component_positions = np.arange(len(COMPONENT_NAMES), dtype=np.float32)
    bar_width = 0.22
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    try:
        max_value = 0.0
        min_value = float("inf")
        for offset_idx, (display_name, rep_key) in enumerate(family_specs):
            values = [
                float(metrics[effective_dimension_metric_key("eval", rep_key, component)])
                for component in COMPONENT_NAMES
            ]
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))
            bar_positions = component_positions + (offset_idx - 1) * bar_width
            bars = ax.bar(
                bar_positions,
                values,
                width=bar_width,
                color=PLOT_COLORS[display_name],
                label=display_name,
            )
            annotate_bar_values(ax, bars, values, offset=max_value * 0.012 + 4.0)

        ax.set_xticks(component_positions)
        ax.set_xticklabels([component.title() for component in COMPONENT_NAMES])
        ax.set_ylabel("Spectral effective dimension")
        ax.set_title("Effective dimensionality on GlueDegradDB-Eval")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=3)
        hide_chart_spines(ax)
        y_min, y_max = compute_soft_zoom_limits(
            [min_value, max_value],
            step=50.0,
            lower_bound=0.0,
            lower_pad=0.25,
            upper_pad=0.12,
            min_span=120.0,
        )
        ax.set_ylim(y_min, y_max)
        fig.tight_layout()
        fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)

def save_activity_prediction_figure(activity_metrics, output_path):
    if not output_path:
        raise ValueError("activity_figure_path must be provided.")
    _, plt = load_matplotlib()
    ensure_output_dir(output_path)

    rep_order = ("frozen_feature", "baseline", "latentglue")
    rep_metrics = activity_metrics["representations"]
    labels = [str(rep_metrics[key]["display_name"]) for key in rep_order]
    colors = [PLOT_COLORS[label] for label in labels]
    y_positions = np.arange(len(rep_order), dtype=np.float32)

    spearman_means = np.asarray([rep_metrics[key]["spearman"] for key in rep_order], dtype=np.float32)
    rmse_means = np.asarray([rep_metrics[key]["rmse"] for key in rep_order], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    try:
        panel_specs = (
            (axes[0], spearman_means, "Spearman", "Higher is better", 0.04),
            (axes[1], rmse_means, "RMSE", "Lower is better", 0.05),
        )

        for ax, means, title, xlabel, min_span in panel_specs:
            bars = ax.barh(
                y_positions,
                means,
                color=colors,
                height=0.62,
            )
            x_min, x_max = compute_soft_zoom_limits(
                means,
                step=0.02,
                lower_bound=0.0,
                lower_pad=0.40,
                upper_pad=0.16,
                min_span=min_span,
            )
            pad = max((x_max - x_min) * 0.02, 0.008)
            ax.set_xlim(x_min, x_max)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels)
            ax.grid(axis="x", alpha=0.25)
            hide_chart_spines(ax)
            for bar, value in zip(bars, means):
                ax.text(
                    float(value) + pad,
                    bar.get_y() + bar.get_height() / 2.0,
                    format_figure_value(value),
                    va="center",
                    ha="left",
                    fontsize=9,
                )

        axes[0].invert_yaxis()

        fig.suptitle("Activity prediction on GlueDegradDB-Activity", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)

def save_retrieval_figure(retrieval_metrics, output_path):
    if not output_path:
        raise ValueError("retrieval_figure_path must be provided.")
    _, plt = load_matplotlib()
    ensure_output_dir(output_path)

    family_specs = (
        ("Frozen", "frozen_mean"),
        ("Projected", "projected_mean"),
        ("LatentGlue", "latentglue"),
    )
    panel_specs = (
        ("eval", "macro_context_auroc", "Eval AUROC"),
        ("eval", "macro_context_auprc", "Eval AUPRC"),
        ("val", "macro_context_auroc", "Val AUROC"),
        ("val", "macro_context_auprc", "Val AUPRC"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
    try:
        for ax, (split_name, metric_name, title) in zip(axes.reshape(-1), panel_specs):
            labels = [display_name for display_name, _ in family_specs]
            values = np.asarray(
                [
                    retrieval_metrics[split_name]["representations"][rep_key][metric_name]
                    for _display_name, rep_key in family_specs
                ],
                dtype=np.float32,
            )
            bars = ax.bar(
                np.arange(len(labels), dtype=np.float32),
                values,
                width=0.65,
                color=[PLOT_COLORS[label] for label in labels],
            )
            axis_floor = 0.0 if metric_name.endswith("auprc") else 0.0
            y_min, y_max = compute_soft_zoom_limits(
                values,
                step=0.02,
                lower_bound=axis_floor,
                lower_pad=0.55,
                upper_pad=0.18,
                min_span=0.05,
            )
            ax.set_ylim(y_min, min(1.0, y_max))
            ax.set_xticks(np.arange(len(labels), dtype=np.float32))
            ax.set_xticklabels(labels)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
            hide_chart_spines(ax)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + ax.get_ylim()[1] * 0.025,
                    format_figure_value(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        fig.suptitle("Retrieval across eval and val splits", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)

def generate_summary_figures(metrics, effective_dim_figure_path, activity_figure_path, retrieval_figure_path):
    save_effective_dimensionality_figure(metrics, output_path=effective_dim_figure_path)
    save_activity_prediction_figure(metrics["activity_cv"], output_path=activity_figure_path)
    save_retrieval_figure(metrics["retrieval"], output_path=retrieval_figure_path)
    return {
        "effective_dim_eval": str(effective_dim_figure_path),
        "activity_prediction": str(activity_figure_path),
        "retrieval": str(retrieval_figure_path),
        "ligand_attention": str(metrics["ligand_attention"]["figure_path"]),
    }

@dataclass(frozen=True)
class RetrievalCandidateRecord:
    row_id: int
    compound_id: str
    smiles: str

@dataclass(frozen=True)
class RetrievalContextRecord:
    key: str
    target_name: str
    effector_name: str
    effector_uniprot: str
    target_sequence: str
    effector_sequence: str
    positives: tuple[RetrievalCandidateRecord, ...]
    negatives: tuple[RetrievalCandidateRecord, ...]

@dataclass(frozen=True)
class RetrievalRepresentationFamily:
    target: np.ndarray
    effector: np.ndarray
    ligand: np.ndarray

class LowRankBilinearScorer(torch.nn.Module):
    def __init__(self, target_dim, effector_dim, ligand_dim, rank):
        super().__init__()
        self.target_proj = torch.nn.Linear(target_dim, rank, bias=False)
        self.effector_proj = torch.nn.Linear(effector_dim, rank, bias=False)
        self.ligand_from_target_proj = torch.nn.Linear(ligand_dim, rank, bias=False)
        self.ligand_from_effector_proj = torch.nn.Linear(ligand_dim, rank, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, target, effector, ligand):
        target_ligand = (self.target_proj(target) * self.ligand_from_target_proj(ligand)).sum(dim=-1)
        effector_ligand = (self.effector_proj(effector) * self.ligand_from_effector_proj(ligand)).sum(dim=-1)
        return target_ligand + effector_ligand + self.bias

def _safe_float(value):
    if value is None:
        return float("nan")
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)

def build_retrieval_context_records(dataset_df, negatives_per_context, rng_seed, dataset_label):
    required = {
        "Compound ID",
        "SMILES",
        "Target",
        "Effector",
        "Effector UniProt",
        "Target Sequence",
        "Effector Sequence",
    }
    missing = required.difference(dataset_df.columns)
    if missing:
        raise ValueError(f"{dataset_label} is missing required columns: {sorted(missing)}")

    df = dataset_df.copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    blank_smiles_mask = df["SMILES"].astype(str).str.strip() == ""
    if bool(blank_smiles_mask.any()):
        blank_count = int(blank_smiles_mask.sum())
        raise ValueError(f"{dataset_label} contains {blank_count} rows with blank SMILES.")
    group_cols = ["Target", "Effector", "Effector UniProt", "Target Sequence", "Effector Sequence"]

    rng = np.random.default_rng(rng_seed)
    contexts = []
    grouped = df.groupby(group_cols, sort=False)
    for group_idx, (group_key, group_df) in enumerate(grouped, start=1):
        target_name, effector_name, effector_uniprot, target_seq, effector_seq = group_key
        positives = tuple(
            RetrievalCandidateRecord(
                row_id=int(row["row_id"]),
                compound_id=str(row["Compound ID"]),
                smiles=str(row["SMILES"]).strip(),
            )
            for row in group_df.to_dict("records")
        )

        positive_row_ids = [record.row_id for record in positives]
        negative_pool = df.loc[~df["row_id"].isin(positive_row_ids)]
        if len(negative_pool) < int(negatives_per_context):
            raise ValueError(
                f"{dataset_label} context {target_name}/{effector_name} has only {len(negative_pool)} "
                f"available negatives, fewer than requested {int(negatives_per_context)}."
            )

        sample_size = int(negatives_per_context)
        sampled_idx = rng.choice(negative_pool.index.to_numpy(), size=sample_size, replace=False)
        negatives = tuple(
            RetrievalCandidateRecord(
                row_id=int(row["row_id"]),
                compound_id=str(row["Compound ID"]),
                smiles=str(row["SMILES"]).strip(),
            )
            for row in negative_pool.loc[sampled_idx].to_dict("records")
        )

        contexts.append(
            RetrievalContextRecord(
                key=f"ctx_{group_idx:03d}_{target_name}_{effector_name}",
                target_name=str(target_name),
                effector_name=str(effector_name),
                effector_uniprot=str(effector_uniprot),
                target_sequence=str(target_seq),
                effector_sequence=str(effector_seq),
                positives=positives,
                negatives=negatives,
            )
        )
    return contexts

def build_retrieval_pair_dataframe(contexts):
    rows = []
    for context in contexts:
        for label, candidates in ((1, context.positives), (0, context.negatives)):
            for candidate in candidates:
                rows.append(
                    {
                        "context_key": context.key,
                        "candidate_row_id": int(candidate.row_id),
                        "Compound ID": candidate.compound_id,
                        "SMILES": candidate.smiles,
                        "label": int(label),
                        "Target": context.target_name,
                        "Effector": context.effector_name,
                        "Effector UniProt": context.effector_uniprot,
                        "Target Sequence": context.target_sequence,
                        "Effector Sequence": context.effector_sequence,
                    }
                )
    return pd.DataFrame(rows)

def build_retrieval_context_folds(contexts, n_folds):
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")
    if len(contexts) < n_folds:
        raise ValueError(f"Need at least {n_folds} contexts, found {len(contexts)}.")
    folds = [[] for _ in range(n_folds)]
    fold_loads = [0 for _ in range(n_folds)]
    context_infos = sorted(
        (
            (
                context.key,
                len(context.positives) + len(context.negatives),
                len(context.positives),
            )
            for context in contexts
        ),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    for context_key, pair_count, _positive_count in context_infos:
        fold_idx = min(range(n_folds), key=lambda idx: (fold_loads[idx], len(folds[idx])))
        folds[fold_idx].append(context_key)
        fold_loads[fold_idx] += pair_count
    return [sorted(fold) for fold in folds if fold]

def build_retrieval_feature_families(pair_df, encoder, batch_size, autocast_enabled):
    reps = collect_representations(
        pair_df,
        encoder,
        batch_size=batch_size,
        autocast_enabled=autocast_enabled,
    )
    return {
        name: RetrievalRepresentationFamily(
            target=reps[name][0].numpy().astype(np.float32),
            effector=reps[name][1].numpy().astype(np.float32),
            ligand=reps[name][2].numpy().astype(np.float32),
        )
        for name in ("frozen_mean", "projected_mean", "latentglue")
    }

def build_retrieval_standardization_stats(features):
    mean = features.mean(axis=0, keepdims=True).astype(np.float32)
    std = features.std(axis=0, keepdims=True).astype(np.float32)
    if np.any(std <= 1e-6):
        zero_var_count = int(np.sum(std <= 1e-6))
        raise ValueError(f"Encountered {zero_var_count} near-constant retrieval feature dimensions.")
    return mean, std

def apply_retrieval_standardization(features, mean, std):
    return ((features - mean) / std).astype(np.float32)

def choose_retrieval_validation_contexts(train_contexts, rng_seed):
    unique_contexts = np.unique(train_contexts)
    if len(unique_contexts) < 5:
        raise ValueError(f"Need at least 5 training contexts to form a validation split, found {len(unique_contexts)}.")
    rng = np.random.default_rng(rng_seed)
    shuffled = unique_contexts.copy()
    rng.shuffle(shuffled)
    val_count = int(round(0.2 * len(shuffled)))
    if val_count < 1 or val_count >= len(shuffled):
        raise RuntimeError(
            f"Validation split size must be between 1 and {len(shuffled) - 1}, computed {val_count}."
        )
    return np.sort(shuffled[:val_count])

def seed_retrieval_fold(fold_seed):
    fold_seed = int(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

def fit_and_score_retrieval_bilinear(representation, labels, context_keys, train_mask, test_mask, fold_seed, device):
    scorer_device = torch.device(device)
    if not np.any(train_mask):
        raise ValueError("Retrieval training mask is empty.")
    if not np.any(test_mask):
        raise ValueError("Retrieval test mask is empty.")
    train_contexts = context_keys[train_mask]
    val_contexts = choose_retrieval_validation_contexts(train_contexts, fold_seed)
    val_mask = train_mask & np.isin(context_keys, val_contexts)
    core_train_mask = train_mask & ~val_mask
    if not np.any(core_train_mask):
        raise RuntimeError("Retrieval core training mask is empty after validation split.")
    if not np.any(val_mask):
        raise RuntimeError("Retrieval validation mask is empty after validation split.")
    if len(np.unique(labels[val_mask])) != 2:
        raise RuntimeError("Retrieval validation split must contain both positive and negative labels.")

    target_mean, target_std = build_retrieval_standardization_stats(representation.target[core_train_mask])
    effector_mean, effector_std = build_retrieval_standardization_stats(representation.effector[core_train_mask])
    ligand_mean, ligand_std = build_retrieval_standardization_stats(representation.ligand[core_train_mask])

    target_train = torch.from_numpy(apply_retrieval_standardization(representation.target[core_train_mask], target_mean, target_std)).to(scorer_device)
    effector_train = torch.from_numpy(apply_retrieval_standardization(representation.effector[core_train_mask], effector_mean, effector_std)).to(scorer_device)
    ligand_train = torch.from_numpy(apply_retrieval_standardization(representation.ligand[core_train_mask], ligand_mean, ligand_std)).to(scorer_device)
    y_train = torch.from_numpy(labels[core_train_mask].astype(np.float32)).to(scorer_device)

    seed_retrieval_fold(fold_seed)
    model = LowRankBilinearScorer(
        target_dim=representation.target.shape[1],
        effector_dim=representation.effector.shape[1],
        ligand_dim=representation.ligand.shape[1],
        rank=RETRIEVAL_BILINEAR_RANK,
    ).to(scorer_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RETRIEVAL_LR, weight_decay=RETRIEVAL_WEIGHT_DECAY)
    positive_count = float(max(labels[core_train_mask].sum(), 1.0))
    negative_count = float(max((core_train_mask.sum() - labels[core_train_mask].sum()), 1.0))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_count / positive_count], device=scorer_device, dtype=torch.float32),
    )

    target_val = torch.from_numpy(apply_retrieval_standardization(representation.target[val_mask], target_mean, target_std)).to(scorer_device)
    effector_val = torch.from_numpy(apply_retrieval_standardization(representation.effector[val_mask], effector_mean, effector_std)).to(scorer_device)
    ligand_val = torch.from_numpy(apply_retrieval_standardization(representation.ligand[val_mask], ligand_mean, ligand_std)).to(scorer_device)
    y_val = labels[val_mask].astype(np.int64)

    best_score = float("-inf")
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    patience = 0
    batch_size = min(256, int(core_train_mask.sum()))

    for _epoch in range(RETRIEVAL_TRAIN_EPOCHS):
        model.train()
        permutation = torch.randperm(target_train.size(0), device=scorer_device)
        for batch_start in range(0, target_train.size(0), batch_size):
            batch_idx = permutation[batch_start : batch_start + batch_size]
            logits = model(target_train[batch_idx], effector_train[batch_idx], ligand_train[batch_idx])
            loss = criterion(logits, y_train[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(target_val, effector_val, ligand_val)
            val_scores = torch.sigmoid(val_logits).detach().cpu().numpy()
            current_score = float(roc_auc_score(y_val, val_scores))

        if current_score > best_score + 1e-4:
            best_score = current_score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= RETRIEVAL_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        target_test = torch.from_numpy(apply_retrieval_standardization(representation.target[test_mask], target_mean, target_std)).to(scorer_device)
        effector_test = torch.from_numpy(apply_retrieval_standardization(representation.effector[test_mask], effector_mean, effector_std)).to(scorer_device)
        ligand_test = torch.from_numpy(apply_retrieval_standardization(representation.ligand[test_mask], ligand_mean, ligand_std)).to(scorer_device)
        test_logits = model(target_test, effector_test, ligand_test)
        return torch.sigmoid(test_logits).detach().cpu().numpy().astype(np.float32)

def compute_retrieval_context_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if len(np.unique(labels)) != 2:
        raise ValueError("Retrieval context metrics require both positive and negative labels.")
    return {
        "candidate_count": float(len(labels)),
        "positive_count": float(int(labels.sum())),
        "negative_count": float(int((labels == 0).sum())),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }

def evaluate_retrieval_representation(representation, pair_df, folds, device):
    y = pair_df["label"].to_numpy(dtype=np.int64)
    context_keys = pair_df["context_key"].astype(str).to_numpy()
    out_of_fold_scores = np.full(len(pair_df), np.nan, dtype=np.float32)
    for fold_idx, test_contexts in enumerate(folds, start=1):
        test_mask = np.isin(context_keys, np.asarray(test_contexts, dtype=object))
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise RuntimeError(f"Fold {fold_idx} produced an empty train/test split.")

        out_of_fold_scores[test_mask] = fit_and_score_retrieval_bilinear(
            representation=representation,
            labels=y,
            context_keys=context_keys,
            train_mask=train_mask,
            test_mask=test_mask,
            fold_seed=RETRIEVAL_SEED + fold_idx,
            device=device,
        )

    if np.isnan(out_of_fold_scores).any():
        raise RuntimeError("Some retrieval pairs did not receive out-of-fold scores.")

    scored_df = pair_df.copy()
    scored_df["score"] = out_of_fold_scores

    per_context = {}
    macro_aurocs = []
    macro_auprcs = []
    for context_key, group_df in scored_df.groupby("context_key", sort=False):
        labels = group_df["label"].to_numpy(dtype=np.int64)
        scores = group_df["score"].to_numpy(dtype=np.float32)
        context_metrics = compute_retrieval_context_metrics(labels, scores)
        per_context[str(context_key)] = {
            "target_name": str(group_df["Target"].iloc[0]),
            "effector_name": str(group_df["Effector"].iloc[0]),
            "effector_uniprot": str(group_df["Effector UniProt"].iloc[0]),
            **{key: _safe_float(value) for key, value in context_metrics.items()},
        }
        macro_aurocs.append(context_metrics["auroc"])
        macro_auprcs.append(context_metrics["auprc"])

    if not macro_aurocs or not macro_auprcs:
        raise RuntimeError("Retrieval evaluation did not produce any valid per-context metrics.")

    return {
        "macro_context_auroc": float(np.mean(macro_aurocs)),
        "macro_context_auprc": float(np.mean(macro_auprcs)),
        "context_count": float(len(per_context)),
        "per_context": per_context,
    }

def evaluate_retrieval(dataset_df, dataset_name, encoder, batch_size, autocast_enabled, device, negatives_per_context):
    contexts = build_retrieval_context_records(
        dataset_df,
        negatives_per_context=negatives_per_context,
        rng_seed=RETRIEVAL_SEED,
        dataset_label=dataset_name,
    )
    if not contexts:
        raise RuntimeError(f"No retrieval contexts were produced for {dataset_name}.")

    pair_df = build_retrieval_pair_dataframe(contexts)
    folds = build_retrieval_context_folds(contexts, RETRIEVAL_N_FOLDS)
    retrieval_batch_size = resolve_batch_size(batch_size, len(pair_df))
    feature_families = build_retrieval_feature_families(
        pair_df=pair_df,
        encoder=encoder,
        batch_size=retrieval_batch_size,
        autocast_enabled=autocast_enabled,
    )

    results = {
        "config": {
            "negatives_per_context": float(negatives_per_context),
            "batch_size": float(retrieval_batch_size),
            "n_folds": float(len(folds)),
            "rng_seed": float(RETRIEVAL_SEED),
            "scorer": (
                f"LowRankBilinearScorer(rank={RETRIEVAL_BILINEAR_RANK}) "
                f"+ BCEWithLogitsLoss + AdamW(lr={RETRIEVAL_LR}, weight_decay={RETRIEVAL_WEIGHT_DECAY})"
            ),
            "split_strategy": f"grouped context holdout cross-validation on {dataset_name}",
        },
        "dataset": {
            "name": dataset_name,
            "num_rows": float(len(dataset_df)),
            "num_contexts": float(len(contexts)),
            "num_pairs": float(len(pair_df)),
            "context_positive_count_mean": float(np.mean([len(context.positives) for context in contexts])),
            "context_negative_count_mean": float(np.mean([len(context.negatives) for context in contexts])),
        },
        "folds": {
            "context_counts": [float(len(fold)) for fold in folds],
            "pair_counts": [float(int(pair_df["context_key"].isin(fold).sum())) for fold in folds],
        },
        "representations": {},
    }
    for name, representation in feature_families.items():
        results["representations"][name] = evaluate_retrieval_representation(
            representation=representation,
            pair_df=pair_df,
            folds=folds,
            device=device,
        )
    return results
def main(
    checkpoint_path="",
    device=DEFAULT_DEVICE,
    batch_size=RETRIEVAL_BATCH_SIZE,
    train_csv_path=DEFAULT_TRAIN_CSV,
    eval_csv_path=DEFAULT_EVAL_CSV,
    activity_csv_path=DEFAULT_ACTIVITY_CSV,
    output_json_path=DEFAULT_OUTPUT_JSON,
    attention_figure_path=DEFAULT_ATTENTION_FIGURE_PATH,
    effective_dim_figure_path=DEFAULT_EFFECTIVE_DIM_FIGURE_PATH,
    activity_figure_path=DEFAULT_ACTIVITY_FIGURE_PATH,
    retrieval_figure_path=DEFAULT_RETRIEVAL_FIGURE_PATH,
):
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path or os.environ.get("LATENTGLUE_CHECKPOINT", "") or DEFAULT_CHECKPOINT_URL
    )
    encoder = load_models(checkpoint_path, device=device)
    autocast_enabled = str(device).startswith("cuda")
    val_df = read_df(
        train_csv_path,
        required=("Target", "Effector", "Target Sequence", "Effector Sequence", "SMILES", "split"),
        split="val",
    )
    eval_df = read_df(eval_csv_path, EVAL_REQUIRED_COLUMNS)
    activity_df = read_df(activity_csv_path, (*TERNARY_COLUMNS, "Value", "Target"))
    repr_cache = {}

    val_reps = get_representations(
        repr_cache,
        "val",
        val_df,
        encoder,
        batch_size,
        autocast_enabled,
    )
    eval_reps = get_representations(
        repr_cache,
        "eval",
        eval_df,
        encoder,
        batch_size,
        autocast_enabled,
    )

    metrics = {}
    metrics.update(evaluate_effective_dimensions(val_reps, dataset_name="val"))
    metrics.update(evaluate_effective_dimensions(eval_reps, dataset_name="eval"))
    activity_reps = get_representations(
        repr_cache,
        "activity",
        activity_df,
        encoder,
        batch_size,
        autocast_enabled,
    )
    metrics.update(evaluate_activity(activity_df, activity_reps))
    metrics["retrieval"] = {
        "val": evaluate_retrieval(
            dataset_df=val_df,
            dataset_name="GlueDegradDB val split",
            encoder=encoder,
            batch_size=batch_size,
            autocast_enabled=autocast_enabled,
            device=device,
            negatives_per_context=RETRIEVAL_NEGATIVES_PER_CONTEXT_VAL,
        ),
        "eval": evaluate_retrieval(
            dataset_df=eval_df,
            dataset_name="GlueDegradDB-Eval",
            encoder=encoder,
            batch_size=batch_size,
            autocast_enabled=autocast_enabled,
            device=device,
            negatives_per_context=RETRIEVAL_NEGATIVES_PER_CONTEXT_EVAL,
        ),
    }
    metrics["ligand_attention"] = evaluate_ligand_attention(
        encoder=encoder,
        eval_df=eval_df,
        autocast_enabled=autocast_enabled,
        output_path=attention_figure_path,
    )
    metrics["summary_figures"] = generate_summary_figures(
        metrics,
        effective_dim_figure_path=effective_dim_figure_path,
        activity_figure_path=activity_figure_path,
        retrieval_figure_path=retrieval_figure_path,
    )

    if not output_json_path:
        raise ValueError("output_json_path must be provided.")
    ensure_output_dir(output_json_path)
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics

if __name__ == "__main__":
    main()
