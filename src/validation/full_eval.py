"""
1. Component effective dimensionality
   - compute spectral effective dimension for target, effector, and ligand
     embedding and compare between frozen, projected mean, and LatentGlue

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

3. Val-set ligand and protein attention
   - compute ligand-pooling attention over the full set of unique ligands
   - compute target- and effector-pooling attention over the unique proteins
   - summarize attention concentration and spread across ligand atoms and
     protein residues
"""

import json
import os
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.model.dataset import collate_ternary
from src.model.model import LatentGlueEncoder
from src.processing.mol_utils import canonicalize_smiles as dataset_canonicalize_smiles
from src.model.train import AUTOCAST_DTYPE
from src.validation.in_train_eval import (activity_targets, masked_mean_pool, spectral_effective_dimension,
)

DEFAULT_DEVICE = "cuda"
DEFAULT_TRAIN_CSV = "data/GlueDegradDB.csv"
DEFAULT_ACTIVITY_CSV = "data/GlueDegradDB-Activity.csv"
DEFAULT_OUTPUT_JSON = "data/results/full_eval.json"
DEFAULT_LIGAND_ATTENTION_FIGURE_PATH = "data/results/ligand_attention.png"
DEFAULT_PROTEIN_ATTENTION_FIGURE_PATH = "data/results/protein_attention.png"
DEFAULT_EFFECTIVE_DIM_FIGURE_PATH = "data/results/effective_dim_val.png"
DEFAULT_ACTIVITY_FIGURE_PATH = "data/results/activity_prediction.png"
DEFAULT_CHECKPOINT_URL = "https://huggingface.co/AnonPeerRev/LatentGlue"
DEFAULT_CHECKPOINT_FILENAME = "LatentGlue.pt"
COMPONENT_NAMES = ("target", "effector", "ligand")
TERNARY_COLUMNS = ("Target Sequence", "Effector Sequence", "SMILES")
DEFAULT_BATCH_SIZE = 32
EVAL_SEEDS = (17, 18, 19)
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

def build_seeded_target_balanced_folds(targets, values, n_folds=5, seed=0):
    targets = np.asarray(targets)
    values = np.asarray(values)
    rng = np.random.default_rng(int(seed))
    folds = [[] for _ in range(int(n_folds))]
    for target in sorted(set(targets.tolist())):
        target_indices = np.where(targets == target)[0]
        ordered = target_indices[np.argsort(values[target_indices], kind="stable")]
        fold_offset = int(rng.integers(int(n_folds))) if ordered.size else 0
        for offset, index in enumerate(ordered.tolist()):
            folds[(fold_offset + offset) % int(n_folds)].append(index)
    return [np.array(sorted(fold), dtype=np.int64) for fold in folds if fold]

def activity_probe_cv_with_folds(features, values, targets, alpha=1.0, n_folds=5, seed=0):
    features = np.asarray(features, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets)
    spearman_scores = []
    rmse_scores = []
    for fold_indices in build_seeded_target_balanced_folds(targets, values, n_folds=n_folds, seed=seed):
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

def activity_probe_cv_with_oof_predictions(features, values, targets, alpha=1.0, n_folds=5, seed=0):
    features = np.asarray(features, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets)
    spearman_scores = []
    rmse_scores = []
    out_of_fold_predictions = np.full(len(values), np.nan, dtype=np.float32)
    fold_ids = np.full(len(values), -1, dtype=np.int64)
    valid_fold_count = 0
    for fold_idx, fold_indices in enumerate(
        build_seeded_target_balanced_folds(targets, values, n_folds=n_folds, seed=seed),
        start=1,
    ):
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[fold_indices] = False
        if train_mask.sum() == 0 or fold_indices.size == 0:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(features[train_mask], values[train_mask])
        predictions = model.predict(features[fold_indices]).astype(np.float32)
        out_of_fold_predictions[fold_indices] = predictions
        fold_ids[fold_indices] = int(fold_idx)
        spearman_score = spearmanr(values[fold_indices], predictions).statistic
        if spearman_score is None or np.isnan(spearman_score):
            raise RuntimeError(
                "Activity evaluation produced an undefined Spearman correlation for at least one fold."
            )
        spearman_scores.append(float(spearman_score))
        rmse_scores.append(float(np.sqrt(mean_squared_error(values[fold_indices], predictions))))
        valid_fold_count += 1
    if not spearman_scores:
        raise RuntimeError("Activity evaluation did not produce any valid cross-validation folds.")
    if np.isnan(out_of_fold_predictions).any():
        raise RuntimeError("Activity evaluation did not produce out-of-fold predictions for every row.")
    return {
        "spearman": float(np.mean(spearman_scores)),
        "rmse": float(np.mean(rmse_scores)),
        "spearman_folds": [float(score) for score in spearman_scores],
        "rmse_folds": [float(score) for score in rmse_scores],
        "spearman_std": sample_standard_deviation(spearman_scores),
        "rmse_std": sample_standard_deviation(rmse_scores),
        "n_folds": float(valid_fold_count),
        "oof_predictions": out_of_fold_predictions,
        "fold_ids": fold_ids,
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

def aggregate_activity_representation(display_name, seed_metrics):
    spearman_values = [float(metrics["spearman"]) for metrics in seed_metrics]
    rmse_values = [float(metrics["rmse"]) for metrics in seed_metrics]
    return {
        "display_name": str(display_name),
        "spearman": float(np.mean(spearman_values)),
        "rmse": float(np.mean(rmse_values)),
        "spearman_std": sample_standard_deviation(spearman_values),
        "rmse_std": sample_standard_deviation(rmse_values),
        "n_seeds": float(len(seed_metrics)),
    }

def summarize_activity_subset(values, predictions):
    values = np.asarray(values, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    if values.size == 0 or predictions.size == 0 or values.size != predictions.size:
        raise ValueError("values and predictions must be non-empty arrays with matching shapes.")
    spearman_score = spearmanr(values, predictions).statistic
    if spearman_score is None or np.isnan(spearman_score):
        spearman_score = 0.0
    rmse_score = float(np.sqrt(mean_squared_error(values, predictions)))
    return {
        "n_rows": float(len(values)),
        "spearman": float(spearman_score),
        "rmse": rmse_score,
    }

def summarize_activity_per_complex(activity_df, values, predictions):
    complex_df = activity_df.copy().reset_index(drop=True)
    if "Effector" not in complex_df.columns:
        complex_df["Effector"] = complex_df["Effector Sequence"].astype(str)
    complex_df["activity_target"] = np.asarray(values, dtype=np.float32)
    complex_df["prediction"] = np.asarray(predictions, dtype=np.float32)
    per_complex = {}
    grouped = complex_df.groupby(["Target", "Effector"], sort=True)
    for (target_name, effector_name), group_df in grouped:
        subset_metrics = summarize_activity_subset(
            group_df["activity_target"].to_numpy(dtype=np.float32),
            group_df["prediction"].to_numpy(dtype=np.float32),
        )
        per_complex[f"{target_name}__{effector_name}"] = {
            "target_name": str(target_name),
            "effector_name": str(effector_name),
            **subset_metrics,
        }
    return per_complex

def aggregate_activity_per_complex(seed_entries):
    all_keys = sorted({key for entry in seed_entries for key in entry.keys()})
    aggregated = {}
    for complex_key in all_keys:
        complex_seed_entries = [entry[complex_key] for entry in seed_entries if complex_key in entry]
        aggregated[str(complex_key)] = {
            "target_name": str(complex_seed_entries[0]["target_name"]),
            "effector_name": str(complex_seed_entries[0]["effector_name"]),
            "n_rows": float(np.mean([item["n_rows"] for item in complex_seed_entries])),
            "spearman": float(np.mean([item["spearman"] for item in complex_seed_entries])),
            "spearman_std": sample_standard_deviation([item["spearman"] for item in complex_seed_entries]),
            "rmse": float(np.mean([item["rmse"] for item in complex_seed_entries])),
            "rmse_std": sample_standard_deviation([item["rmse"] for item in complex_seed_entries]),
        }
    return aggregated

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

def evaluate_activity(activity_df, reps, seeds=EVAL_SEEDS):
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
        "error_bar": "sample standard deviation across seeds",
        "within_seed_error_bar": "sample standard deviation across folds",
        "aggregation": "mean across repeated seeded cross-validation runs",
        "seeds": [float(seed) for seed in seeds],
        "representations": {},
        "per_seed_summaries": {},
    }
    per_rep_seed_metrics = {rep_key: [] for rep_key, *_rest in representation_specs}
    per_rep_seed_complex_metrics = {rep_key: [] for rep_key, *_rest in representation_specs}
    for seed in seeds:
        seed_summary = {"representations": {}}
        for rep_key, display_name, _prefix, features in representation_specs:
            probe_metrics = activity_probe_cv_with_oof_predictions(features, y, targets, n_folds=5, seed=seed)
            per_rep_seed_metrics[rep_key].append(probe_metrics)
            seed_summary["representations"][rep_key] = summarize_activity_cv_representation(display_name, probe_metrics)
            seed_summary["representations"][rep_key]["per_complex"] = summarize_activity_per_complex(
                activity_df,
                y,
                probe_metrics["oof_predictions"],
            )
            per_rep_seed_complex_metrics[rep_key].append(seed_summary["representations"][rep_key]["per_complex"])
        seed_summary["n_folds"] = float(seed_summary["representations"]["latentglue"]["n_folds"])
        activity_cv["per_seed_summaries"][str(int(seed))] = seed_summary

    for rep_key, display_name, prefix, _features in representation_specs:
        aggregate_metrics = aggregate_activity_representation(display_name, per_rep_seed_metrics[rep_key])
        metrics[f"{prefix}_spearman"] = float(aggregate_metrics["spearman"])
        metrics[f"{prefix}_rmse"] = float(aggregate_metrics["rmse"])
        aggregate_metrics["per_complex"] = aggregate_activity_per_complex(per_rep_seed_complex_metrics[rep_key])
        activity_cv["representations"][rep_key] = aggregate_metrics

    activity_cv["n_folds"] = float(activity_cv["per_seed_summaries"][str(int(seeds[0]))]["n_folds"])
    activity_cv["n_seeds"] = float(len(seeds))
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
    canonical_smiles = dataset_canonicalize_smiles(smiles)
    if not canonical_smiles:
        raise ValueError(f"Invalid SMILES: {smiles}")
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

def build_ligand_attention_records(dataset_df):
    missing_columns = [column for column in ("Compound ID", "SMILES", "Target", "Effector") if column not in dataset_df.columns]
    if missing_columns:
        raise ValueError(f"Ligand attention requires columns: {missing_columns}")

    unique_records = {}
    for row in dataset_df.to_dict("records"):
        canonical_smiles = dataset_canonicalize_smiles(row["SMILES"])
        if not canonical_smiles:
            raise ValueError(f"Invalid SMILES: {row['SMILES']}")
        record = unique_records.get(canonical_smiles)
        if record is None:
            unique_records[canonical_smiles] = {
                "compound_ids": {str(row["Compound ID"]).strip()},
                "targets": {str(row["Target"]).strip()},
                "effectors": {str(row["Effector"]).strip()},
                "smiles": canonical_smiles,
                "dataset_row_count": 1,
            }
            continue

        record["dataset_row_count"] += 1
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
                "dataset_row_count": int(record["dataset_row_count"]),
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

def select_representative_attention_examples(example_data, count_key, target_count=9):
    if not example_data:
        raise ValueError("example_data must contain at least one attention example.")

    metrics_df = pd.DataFrame(
        {
            "example_idx": np.arange(len(example_data), dtype=np.int64),
            "count": [float(example[count_key]) for example in example_data],
            "normalized_entropy": [float(example["normalized_entropy"]) for example in example_data],
        }
    )
    metrics_df["size_bin"] = assign_attention_quantile_bins(metrics_df["count"].to_numpy(), ATTENTION_PANEL_SIZE_LABELS)
    metrics_df["focus_bin"] = assign_attention_quantile_bins(metrics_df["normalized_entropy"].to_numpy(), ATTENTION_PANEL_FOCUS_LABELS)

    for row in metrics_df.itertuples(index=False):
        example_data[int(row.example_idx)]["size_bin"] = str(row.size_bin)
        example_data[int(row.example_idx)]["focus_bin"] = str(row.focus_bin)

    count_scale = max(float(metrics_df["count"].std(ddof=0)), 1.0)
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
            count_center = float(cell["count"].median())
            entropy_center = float(cell["normalized_entropy"].median())
            distances = (
                np.square((cell["count"] - count_center) / count_scale) +
                np.square((cell["normalized_entropy"] - entropy_center) / entropy_scale)
            )
            selected_example_idx = int(cell.loc[distances.idxmin(), "example_idx"])
            example = example_data[selected_example_idx]
            example["size_bin"] = size_bin
            example["focus_bin"] = focus_bin
            selected_indices.append(selected_example_idx)

    selected_indices = list(dict.fromkeys(int(idx) for idx in selected_indices))
    if len(selected_indices) < min(int(target_count), len(example_data)):
        overall_count_center = float(metrics_df["count"].median())
        overall_entropy_center = float(metrics_df["normalized_entropy"].median())
        remaining_candidates = []
        selected_index_set = set(selected_indices)
        for row in metrics_df.itertuples(index=False):
            if int(row.example_idx) in selected_index_set:
                continue
            distance = (
                np.square((float(row.count) - overall_count_center) / count_scale) +
                np.square((float(row.normalized_entropy) - overall_entropy_center) / entropy_scale)
            )
            remaining_candidates.append((float(distance), int(row.example_idx)))
        for _distance, example_idx in sorted(remaining_candidates, key=lambda item: item[0]):
            selected_indices.append(int(example_idx))
            if len(selected_indices) >= min(int(target_count), len(example_data)):
                break

    return [example_data[idx] for idx in selected_indices[: min(int(target_count), len(example_data))]]

def evaluate_ligand_attention(encoder, dataset_df, dataset_name, autocast_enabled, output_path):
    ligand_records = build_ligand_attention_records(dataset_df)
    if not ligand_records:
        raise RuntimeError(f"Ligand attention did not find any unique ligands for {dataset_name}.")

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

    representative_panels = select_representative_attention_examples(panel_data, count_key="atom_count")
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
        "dataset_name": str(dataset_name),
        "num_dataset_rows": float(len(dataset_df)),
        "num_unique_ligands": float(len(panel_data)),
        "representative_panel_count": float(len(representative_panels)),
        "occupied_bin_count": occupied_bin_count,
        "summary": summary,
        "attention_bin_counts": attention_bin_counts,
        "representative_panels": [
            {
                "compound_id": panel["compound_id"],
                "smiles": panel["smiles"],
                "targets": panel["targets"],
                "effectors": panel["effectors"],
                "dataset_row_count": float(panel["dataset_row_count"]),
                "atom_count": float(panel["atom_count"]),
                "top1_atom_weight": float(panel["top1_atom_weight"]),
                "top3_atom_weight": float(panel["top3_atom_weight"]),
                "normalized_entropy": float(panel["normalized_entropy"]),
                "effective_atom_fraction": float(panel["effective_atom_fraction"]),
                "participation_ratio": float(panel["participation_ratio"]),
                "size_bin": panel["size_bin"],
                "focus_bin": panel["focus_bin"],
                "atom_weights": [float(value) for value in panel["atom_weights"]],
            }
            for panel in representative_panels
        ],
    }
    return result

@torch.no_grad()
def compute_protein_attention(encoder, sequence, role_id, autocast_enabled):
    sequence = str(sequence).strip()
    if not sequence:
        raise ValueError("Protein sequence must be non-empty.")

    protein_toks = encoder.esm_model._tokenize([sequence]).to(encoder.device)
    protein_mask_raw = protein_toks != encoder.esm_model.tokenizer.pad_token_id
    pool_module = encoder.target_pool if int(role_id) == 0 else encoder.effector_pool

    with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
        protein_tokens, protein_mask, _ = encoder.forward_protein(
            protein_toks,
            protein_mask_raw,
            role_id=int(role_id),
        )
        _, token_weights = extract_seed_attention_weights(pool_module, protein_tokens, mask=protein_mask)
        residue_weights = token_weights[0, protein_mask[0]].detach().float().cpu().numpy().astype(np.float32)

    if len(sequence) != len(residue_weights):
        raise RuntimeError(
            f"Protein attention length mismatch: sequence has {len(sequence)} residues but got {len(residue_weights)} weights."
        )

    total_weight = float(residue_weights.sum())
    if total_weight <= 0.0:
        raise RuntimeError("Protein attention weights must sum to a positive value.")
    residue_weights = residue_weights / total_weight
    return {
        "sequence": sequence,
        "residue_weights": residue_weights,
    }

def compute_protein_attention_distribution_metrics(residue_weights):
    residue_weights = np.asarray(residue_weights, dtype=np.float32)
    if residue_weights.ndim != 1 or residue_weights.size == 0:
        raise ValueError("residue_weights must be a non-empty 1D vector.")

    positive_probs = residue_weights[residue_weights > 0]
    entropy = float(-(positive_probs * np.log(positive_probs)).sum())
    max_entropy = float(np.log(len(residue_weights))) if len(residue_weights) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
    effective_residue_count = float(np.exp(entropy))
    participation_ratio = float(1.0 / np.square(residue_weights).sum())
    sorted_probs = np.sort(residue_weights)[::-1]

    return {
        "residue_count": float(len(residue_weights)),
        "top1_residue_weight": float(sorted_probs[0]),
        "top10_residue_weight": float(sorted_probs[: min(10, len(sorted_probs))].sum()),
        "entropy": entropy,
        "normalized_entropy": float(normalized_entropy),
        "effective_residue_count": effective_residue_count,
        "effective_residue_fraction": float(effective_residue_count / len(residue_weights)),
        "participation_ratio": participation_ratio,
    }

def build_protein_attention_records(dataset_df, component_name):
    component_name = str(component_name)
    if component_name == "target":
        name_col = "Target"
        seq_col = "Target Sequence"
        partner_col = "Effector"
        uniprot_col = "Target UniProt"
    elif component_name == "effector":
        name_col = "Effector"
        seq_col = "Effector Sequence"
        partner_col = "Target"
        uniprot_col = "Effector UniProt"
    else:
        raise ValueError(f"Unsupported protein attention component: {component_name}")

    required_columns = [name_col, seq_col, partner_col]
    missing_columns = [column for column in required_columns if column not in dataset_df.columns]
    if missing_columns:
        raise ValueError(f"Protein attention for {component_name} requires columns: {missing_columns}")

    unique_records = {}
    for row in dataset_df.to_dict("records"):
        sequence = str(row[seq_col]).strip()
        if not sequence:
            raise ValueError(f"Blank {component_name} sequence encountered in attention evaluation.")
        record = unique_records.get(sequence)
        if record is None:
            unique_records[sequence] = {
                "protein_name": str(row[name_col]).strip(),
                "uniprot": str(row.get(uniprot_col, "")).strip(),
                "sequence": sequence,
                "dataset_row_count": 1,
                "partners": {str(row[partner_col]).strip()},
            }
            continue

        record["dataset_row_count"] += 1
        record["partners"].add(str(row[partner_col]).strip())

    ordered_records = []
    for sequence, record in unique_records.items():
        partners = sorted(value for value in record["partners"] if value)
        ordered_records.append(
            {
                "protein_name": record["protein_name"],
                "uniprot": record["uniprot"],
                "sequence": sequence,
                "dataset_row_count": int(record["dataset_row_count"]),
                "partners": partners,
            }
        )
    ordered_records.sort(key=lambda item: (item["protein_name"], item["uniprot"], item["sequence"]))
    return ordered_records

def evaluate_protein_attention_component(encoder, dataset_df, dataset_name, component_name, autocast_enabled):
    records = build_protein_attention_records(dataset_df, component_name=component_name)
    if not records:
        raise RuntimeError(f"Protein attention did not find any unique {component_name} sequences for {dataset_name}.")

    example_data = []
    role_id = 0 if component_name == "target" else 1
    for record in records:
        example = compute_protein_attention(
            encoder=encoder,
            sequence=record["sequence"],
            role_id=role_id,
            autocast_enabled=autocast_enabled,
        )
        example.update(record)
        example.update(compute_protein_attention_distribution_metrics(example["residue_weights"]))
        example_data.append(example)

    representative_examples = select_representative_attention_examples(example_data, count_key="residue_count")

    attention_bin_counts = {
        focus_bin: {
            size_bin: float(
                sum(
                    example["focus_bin"] == focus_bin and example["size_bin"] == size_bin
                    for example in example_data
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
        "residue_count": summarize_attention_metric_distribution([example["residue_count"] for example in example_data]),
        "top1_residue_weight": summarize_attention_metric_distribution([example["top1_residue_weight"] for example in example_data]),
        "top10_residue_weight": summarize_attention_metric_distribution([example["top10_residue_weight"] for example in example_data]),
        "normalized_entropy": summarize_attention_metric_distribution([example["normalized_entropy"] for example in example_data]),
        "effective_residue_fraction": summarize_attention_metric_distribution([example["effective_residue_fraction"] for example in example_data]),
        "participation_ratio": summarize_attention_metric_distribution([example["participation_ratio"] for example in example_data]),
    }

    return {
        "component_name": str(component_name),
        "dataset_name": str(dataset_name),
        "num_dataset_rows": float(len(dataset_df)),
        "num_unique_sequences": float(len(example_data)),
        "representative_entry_count": float(len(representative_examples)),
        "occupied_bin_count": occupied_bin_count,
        "summary": summary,
        "attention_bin_counts": attention_bin_counts,
        "representative_entries": [
            {
                "protein_name": example["protein_name"],
                "uniprot": example["uniprot"],
                "partners": example["partners"],
                "dataset_row_count": float(example["dataset_row_count"]),
                "residue_count": float(example["residue_count"]),
                "top1_residue_weight": float(example["top1_residue_weight"]),
                "top10_residue_weight": float(example["top10_residue_weight"]),
                "normalized_entropy": float(example["normalized_entropy"]),
                "effective_residue_fraction": float(example["effective_residue_fraction"]),
                "participation_ratio": float(example["participation_ratio"]),
                "size_bin": example["size_bin"],
                "focus_bin": example["focus_bin"],
                "residue_weights": [float(value) for value in example["residue_weights"]],
            }
            for example in representative_examples
        ],
    }

def evaluate_protein_attention(encoder, dataset_df, dataset_name, autocast_enabled):
    return {
        "dataset_name": str(dataset_name),
        "target": evaluate_protein_attention_component(
            encoder=encoder,
            dataset_df=dataset_df,
            dataset_name=dataset_name,
            component_name="target",
            autocast_enabled=autocast_enabled,
        ),
        "effector": evaluate_protein_attention_component(
            encoder=encoder,
            dataset_df=dataset_df,
            dataset_name=dataset_name,
            component_name="effector",
            autocast_enabled=autocast_enabled,
        ),
    }

def summarize_top_ligand_atoms(smiles, atom_weights, top_k=3):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return []

    atom_weights = np.asarray(atom_weights, dtype=np.float32)
    if atom_weights.size != mol.GetNumAtoms():
        return []

    top_indices = np.argsort(atom_weights)[::-1][: min(int(top_k), atom_weights.size)]
    return [
        {
            "atom_index": int(atom_idx),
            "atom_symbol": str(mol.GetAtomWithIdx(int(atom_idx)).GetSymbol()),
            "weight": float(atom_weights[int(atom_idx)]),
        }
        for atom_idx in top_indices
    ]

def select_protein_gallery_entries(protein_attention_result, target_count=9):
    combined_entries = []
    for component_name in ("target", "effector"):
        component = protein_attention_result[component_name]
        for entry in component["representative_entries"]:
            record = dict(entry)
            record["component_name"] = component_name
            combined_entries.append(record)
    if not combined_entries:
        return []

    if len(combined_entries) <= int(target_count):
        return combined_entries

    gallery_examples = [dict(entry) for entry in combined_entries]
    selected = select_representative_attention_examples(
        gallery_examples,
        count_key="residue_count",
        target_count=target_count,
    )
    return selected

def save_ligand_attention_figure(ligand_attention, output_path):
    if not output_path:
        raise ValueError("ligand_attention_figure_path must be provided.")

    _, plt = load_matplotlib()
    ensure_output_dir(output_path)

    ligand_entries = [dict(entry) for entry in ligand_attention["representative_panels"]]
    if not ligand_entries:
        raise RuntimeError("Ligand attention figure requires at least one representative ligand entry.")

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 10.5))
    axes = np.asarray(axes).reshape(-1)
    try:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            hide_chart_spines(ax)

        for panel_idx, entry in enumerate(ligand_entries[:9]):
            ax = axes[panel_idx]
            atom_weights = np.asarray(entry["atom_weights"], dtype=np.float32)
            ax.imshow(atom_weights[np.newaxis, :], aspect="auto", cmap="inferno", interpolation="nearest")
            ax.set_title(
                f"Ligand {panel_idx + 1}: {entry.get('compound_id', '') or 'unlabeled'}\n"
                f"{entry['size_bin']} | {entry['focus_bin']}",
                fontsize=10,
            )
            top_atoms = summarize_top_ligand_atoms(entry["smiles"], atom_weights, top_k=3)
            top_atom_text = ", ".join(
                f"{atom['atom_symbol']}{atom['atom_index']}:{atom['weight']:.2f}"
                for atom in top_atoms
            ) or "n/a"
            metric_text = (
                f"atoms={int(round(entry['atom_count']))}  top1={entry['top1_atom_weight']:.2f}\n"
                f"top3={entry['top3_atom_weight']:.2f}  H={entry['normalized_entropy']:.2f}\n"
                f"top atoms: {top_atom_text}"
            )
            ax.text(
                0.02,
                0.02,
                metric_text,
                transform=ax.transAxes,
                fontsize=8.5,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5},
            )
            ax.set_xlabel("Ligand atom index", fontsize=8)

        for empty_ax in axes[len(ligand_entries[:9]) : 9]:
            empty_ax.axis("off")

        fig.suptitle(
            "Val-split ligand attention representatives",
            fontsize=14,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
        fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)

def save_protein_attention_figure(protein_attention, output_path):
    if not output_path:
        raise ValueError("protein_attention_figure_path must be provided.")

    _, plt = load_matplotlib()
    ensure_output_dir(output_path)

    protein_entries = [dict(entry) for entry in select_protein_gallery_entries(protein_attention, target_count=9)]
    if not protein_entries:
        raise RuntimeError("Protein attention figure requires at least one representative protein entry.")

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.5))
    axes = np.asarray(axes).reshape(-1)
    try:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            hide_chart_spines(ax)

        for protein_idx, entry in enumerate(protein_entries[:9]):
            ax = axes[protein_idx]
            residue_weights = np.asarray(entry["residue_weights"], dtype=np.float32)
            ax.imshow(residue_weights[np.newaxis, :], aspect="auto", cmap="inferno", interpolation="nearest")
            ax.set_title(
                f"{entry['component_name'].title()} {protein_idx + 1}: {entry['protein_name']}\n"
                f"{entry['size_bin']} | {entry['focus_bin']}",
                fontsize=10,
            )
            metric_text = (
                f"res={int(round(entry['residue_count']))}  top1={entry['top1_residue_weight']:.2f}\n"
                f"top10={entry['top10_residue_weight']:.2f}  H={entry['normalized_entropy']:.2f}"
            )
            ax.text(
                0.02,
                0.90,
                metric_text,
                transform=ax.transAxes,
                fontsize=8.5,
                va="top",
                ha="left",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 2.5},
            )
            ax.set_xlabel(f"{entry['component_name'].title()} sequence position", fontsize=8)

        for empty_ax in axes[len(protein_entries[:9]) : 9]:
            empty_ax.axis("off")

        fig.suptitle("Val-split protein attention representatives", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
        fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)

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
                float(metrics[effective_dimension_metric_key("val", rep_key, component)])
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
        ax.set_title("Effective dimensionality on GlueDegradDB val split")
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

def generate_summary_figures(metrics, effective_dim_figure_path, activity_figure_path):
    save_effective_dimensionality_figure(metrics, output_path=effective_dim_figure_path)
    save_activity_prediction_figure(metrics["activity_cv"], output_path=activity_figure_path)
    return {
        "effective_dim_val": str(effective_dim_figure_path),
        "activity_prediction": str(activity_figure_path),
    }
def main(
    checkpoint_path="",
    device=DEFAULT_DEVICE,
    batch_size=DEFAULT_BATCH_SIZE,
    train_csv_path=DEFAULT_TRAIN_CSV,
    activity_csv_path=DEFAULT_ACTIVITY_CSV,
    output_json_path=DEFAULT_OUTPUT_JSON,
    ligand_attention_figure_path=DEFAULT_LIGAND_ATTENTION_FIGURE_PATH,
    protein_attention_figure_path=DEFAULT_PROTEIN_ATTENTION_FIGURE_PATH,
    effective_dim_figure_path=DEFAULT_EFFECTIVE_DIM_FIGURE_PATH,
    activity_figure_path=DEFAULT_ACTIVITY_FIGURE_PATH,
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
    metrics = {}
    metrics.update(evaluate_effective_dimensions(val_reps, dataset_name="val"))
    activity_reps = get_representations(
        repr_cache,
        "activity",
        activity_df,
        encoder,
        batch_size,
        autocast_enabled,
    )
    metrics.update(evaluate_activity(activity_df, activity_reps, seeds=EVAL_SEEDS))
    metrics["ligand_attention"] = evaluate_ligand_attention(
        encoder=encoder,
        dataset_df=val_df,
        dataset_name="GlueDegradDB val split",
        autocast_enabled=autocast_enabled,
        output_path=ligand_attention_figure_path,
    )
    metrics["protein_attention"] = evaluate_protein_attention(
        encoder=encoder,
        dataset_df=val_df,
        dataset_name="GlueDegradDB val split",
        autocast_enabled=autocast_enabled,
    )
    save_ligand_attention_figure(
        metrics["ligand_attention"],
        output_path=ligand_attention_figure_path,
    )
    save_protein_attention_figure(
        metrics["protein_attention"],
        output_path=protein_attention_figure_path,
    )
    metrics["summary_figures"] = generate_summary_figures(
        metrics,
        effective_dim_figure_path=effective_dim_figure_path,
        activity_figure_path=activity_figure_path,
    )
    metrics["summary_figures"]["ligand_attention"] = str(ligand_attention_figure_path)
    metrics["summary_figures"]["protein_attention"] = str(protein_attention_figure_path)

    if not output_json_path:
        raise ValueError("output_json_path must be provided.")
    ensure_output_dir(output_json_path)
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics

if __name__ == "__main__":
    main()
