"""
1. Component effective dimensionality + Uniformity
   - compute spectral effective dimension for target, effector, and ligand
     embeddings from the checkpointed encoder on both the GlueDegradDB val split
     and GlueDegradDB-Eval.csv
   - compare three representation families per component: frozen mean-pooled
     backbones, projected mean-pooled features, and LatentGlue pooled features
   - compute per-component uniformity on L2-normalized embeddings; lower is
     better and indicates more even use of the hypersphere
   - log uniformity for the same three representation families per component:
     frozen mean-pooled backbones, projected mean-pooled features, and
     LatentGlue pooled features

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

Data used:
- section 1: `data/GlueDegradDB.csv` with `split == "val"` and `data/GlueDegradDB-Eval.csv`
- section 2: `data/GlueDegradDB-Activity.csv`
"""

import json
import os
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.model.dataset import TernaryDataset, collate_ternary
from src.model.model import LatentGlueEncoder
from src.model.train import AUTOCAST_DTYPE
from src.validation.in_train_eval import (activity_probe_metrics, activity_targets, build_target_balanced_folds, masked_mean_pool, spectral_effective_dimension,
)
_ = Axes3D

DEFAULT_DEVICE = "cuda"
DEFAULT_TRAIN_CSV = "data/GlueDegradDB.csv"
DEFAULT_EVAL_CSV = "data/GlueDegradDB-Eval.csv"
DEFAULT_ACTIVITY_CSV = "data/GlueDegradDB-Activity.csv"
DEFAULT_OUTPUT_JSON = "data/results/full_eval.json"
DEFAULT_FIGURE_DIR = "data/results/full_eval_figures"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoint", "LatentGlue.pt")
DEFAULT_CHECKPOINT_REPO_ID = "ArnavSharma938/LatentGlue"
DEFAULT_CHECKPOINT_FILENAME = "LatentGlue.pt"
COMPONENT_NAMES = ("target", "effector", "ligand")
TERNARY_COLUMNS = ("Target Sequence", "Effector Sequence", "SMILES")

def read_df(path, required=(), split=None):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if split is not None:
        if "split" not in df.columns:
            return None
        df = df[df["split"].astype(str) == str(split)].reset_index(drop=True)
    return df if len(df) and all(column in df.columns for column in required) else None

def load_checkpoint(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = LatentGlueEncoder(device=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    encoder.eval()
    return encoder

def parse_hf_repo_id(checkpoint_path):
    checkpoint_path = str(checkpoint_path).strip()
    parsed = urlparse(checkpoint_path)
    if parsed.scheme in {"http", "https"} and parsed.netloc.endswith("huggingface.co"):
        parts = [part for part in parsed.path.split("/") if part]
        return "/".join(parts[:2]) if len(parts) >= 2 else ""
    return checkpoint_path if checkpoint_path.count("/") == 1 and "\\" not in checkpoint_path and not os.path.splitext(checkpoint_path)[1] else ""

def resolve_checkpoint_path(checkpoint_path):
    checkpoint_path = str(checkpoint_path or "").strip()
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path
    repo_id = parse_hf_repo_id(checkpoint_path)
    if repo_id:
        return hf_hub_download(repo_id=repo_id, filename=DEFAULT_CHECKPOINT_FILENAME)
    if checkpoint_path:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return hf_hub_download(repo_id=DEFAULT_CHECKPOINT_REPO_ID, filename=DEFAULT_CHECKPOINT_FILENAME)

def trim_special_tokens(x, mask):
    x, mask = x[:, 1:, :], mask[:, 1:].clone()
    eos = mask.sum(dim=1) - 1
    mask[torch.arange(x.size(0), device=x.device), eos] = False
    return x, mask

def concat_components(components):
    return torch.cat(list(components), dim=-1).numpy()

def resolve_batch_size(batch_size, n_rows):
    return max(1, n_rows if batch_size is None or int(batch_size) <= 0 else int(batch_size))

@torch.no_grad()
def collect_representations(df, encoder, batch_size, autocast_enabled=True):
    loader = DataLoader(
        TernaryDataset(csv_path=None, df=df),
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

def representation_uniformity(z):
    if z.size(0) < 2:
        return float("nan")
    pairwise_sqdist = torch.pdist(F.normalize(z.float(), dim=-1), p=2).pow(2)
    if pairwise_sqdist.numel() == 0:
        return float("nan")
    return float(torch.log(torch.exp(-2.0 * pairwise_sqdist).mean()).item())

def evaluate_uniformity(reps, dataset_name):
    frozen_names = ("esmc_mean_pool", "esmc_mean_pool", "molformer_xl_mean_pool")
    metrics = {}
    for idx, component in enumerate(COMPONENT_NAMES):
        latent_key = f"uniformity/{dataset_name}/latentglue_{component}"
        frozen_key = f"uniformity/{dataset_name}/{frozen_names[idx]}_{component}"
        projected_key = f"uniformity/{dataset_name}/projected_mean_{component}"
        metrics[latent_key] = representation_uniformity(reps["latentglue"][idx])
        metrics[frozen_key] = representation_uniformity(reps["frozen_mean"][idx])
        metrics[projected_key] = representation_uniformity(reps["projected_mean"][idx])
    return metrics

def evaluate_activity(activity_df, reps):
    if activity_df is None or "Value" not in activity_df.columns or "Target" not in activity_df.columns:
        return {}
    y, targets = activity_targets(activity_df)
    trained_scores = activity_probe_metrics("activity/latentglue", concat_components(reps["latentglue"]), y, targets)
    baseline_scores = activity_probe_metrics("activity/baseline", concat_components(reps["projected_mean"]), y, targets)
    frozen_scores = activity_probe_metrics("activity/frozen_feature", concat_components(reps["frozen_mean"]), y, targets)
    return {**trained_scores, **baseline_scores, **frozen_scores}

def get_representations(repr_cache, cache_key, df, encoder, batch_size, autocast_enabled):
    if df is None:
        return None
    if cache_key not in repr_cache:
        repr_cache[cache_key] = collect_representations(df, encoder, batch_size, autocast_enabled=autocast_enabled)
    return repr_cache[cache_key]

def format_metric(value):
    return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)

def activity_probe_predictions(features, y, targets, alpha=1.0, n_folds=5):
    features = np.asarray(features, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    targets = np.asarray(targets)
    predictions = np.full(len(y), np.nan, dtype=np.float32)
    for fold_indices in build_target_balanced_folds(targets, y, n_folds=n_folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[fold_indices] = False
        if train_mask.sum() == 0 or fold_indices.size == 0:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(features[train_mask], y[train_mask])
        predictions[fold_indices] = model.predict(features[fold_indices]).astype(np.float32)
    return predictions

def component_baseline_prefix(component):
    return "esmc_mean_pool" if component in {"target", "effector"} else "molformer_xl_mean_pool"

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def save_figure(fig, output_path, use_tight_layout=True):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if use_tight_layout:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, wspace=0.28, hspace=0.28)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_grouped_bars(ax, categories, series_map, title, ylabel):
    labels = list(series_map.keys())
    values = [np.asarray(series_map[label], dtype=np.float32) for label in labels]
    x = np.arange(len(categories), dtype=np.float32)
    width = 0.8 / max(len(labels), 1)
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2.0) * width
        ax.bar(x + offset, values[idx], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)

def sphere_projection_coords(z, max_points=600, seed=0):
    x = F.normalize(z.float(), dim=-1).cpu().numpy()
    if len(x) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(x) > max_points:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), size=max_points, replace=False)]
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[1] >= 3:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        coords = x @ vt[:3].T
    else:
        coords = np.pad(x, ((0, 0), (0, max(0, 3 - x.shape[1]))))
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (coords / norms).astype(np.float32)

def draw_unit_sphere(ax):
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 24)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#E2E8F0", alpha=0.10, linewidth=0, shade=False)
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, color="#94A3B8", linewidth=0.70, alpha=0.90)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect((1, 1, 1))
    ax.set_facecolor("#F8FAFC")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def plot_hypersphere_geometry_figure(metrics, reps, dataset_name, figure_dir):
    if reps is None or dataset_name != "eval":
        return
    component_colors = {"target": "#1D4ED8", "effector": "#C2410C", "ligand": "#0F766E"}
    fig = plt.figure(figsize=(12.5, 4.4))
    for col_idx, component in enumerate(COMPONENT_NAMES):
        ax = fig.add_subplot(1, len(COMPONENT_NAMES), col_idx + 1, projection="3d")
        coords = sphere_projection_coords(reps["latentglue"][col_idx], seed=col_idx)
        draw_unit_sphere(ax)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=10,
            alpha=0.72,
            color=component_colors[component],
            edgecolors="none",
        )
        ax.set_title(component.capitalize(), fontsize=11)
    save_figure(fig, os.path.join(figure_dir, "hypersphere_geometry_eval.png"), use_tight_layout=False)

def plot_uniformity_figure(metrics, figure_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset_name, title in zip(axes, ("val", "eval"), ("GlueDegradDB val split", "GlueDegradDB-Eval")):
        if f"uniformity/{dataset_name}/latentglue_target" not in metrics:
            ax.axis("off")
            continue
        series_map = {
            "Frozen mean": [metrics[f"uniformity/{dataset_name}/{component_baseline_prefix(component)}_{component}"] for component in COMPONENT_NAMES],
            "Projected mean": [metrics[f"uniformity/{dataset_name}/projected_mean_{component}"] for component in COMPONENT_NAMES],
            "LatentGlue": [metrics[f"uniformity/{dataset_name}/latentglue_{component}"] for component in COMPONENT_NAMES],
        }
        plot_grouped_bars(ax, COMPONENT_NAMES, series_map, title, "Uniformity (lower is better)")
    axes[0].legend(loc="upper right")
    save_figure(fig, os.path.join(figure_dir, "uniformity_bars.png"))

def plot_activity_scatter_figure(activity_df, activity_reps, metrics, figure_dir):
    if activity_df is None or activity_reps is None:
        return
    y, targets = activity_targets(activity_df)
    feature_specs = [
        ("latentglue", "LatentGlue", "activity/latentglue_spearman", "activity/latentglue_rmse"),
        ("projected_mean", "Projected mean", "activity/baseline_spearman", "activity/baseline_rmse"),
        ("frozen_mean", "Frozen feature", "activity/frozen_feature_spearman", "activity/frozen_feature_rmse"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    valid_min, valid_max = float(np.min(y)), float(np.max(y))
    for ax, (rep_name, label, spearman_key, rmse_key) in zip(axes, feature_specs):
        predictions = activity_probe_predictions(concat_components(activity_reps[rep_name]), y, targets)
        valid = ~np.isnan(predictions)
        ax.scatter(y[valid], predictions[valid], s=18, alpha=0.75, color="#7F1D1D")
        ax.plot([valid_min, valid_max], [valid_min, valid_max], linestyle="--", linewidth=1.0, color="#334155")
        ax.set_title(f"{label}\nSpearman={metrics.get(spearman_key, float('nan')):.3f} | RMSE={metrics.get(rmse_key, float('nan')):.3f}")
        ax.set_xlabel("Observed DC50 Assay Activity")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted DC50 Assay Activity")
    save_figure(fig, os.path.join(figure_dir, "activity_prediction_scatter.png"))

def plot_activity_scatter_latentglue_figure(activity_df, activity_reps, figure_dir):
    if activity_df is None or activity_reps is None:
        return
    y, targets = activity_targets(activity_df)
    predictions = activity_probe_predictions(concat_components(activity_reps["latentglue"]), y, targets)
    valid = ~np.isnan(predictions)
    valid_min, valid_max = float(np.min(y)), float(np.max(y))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(y[valid], predictions[valid], s=18, alpha=0.75, color="#7F1D1D")
    ax.plot([valid_min, valid_max], [valid_min, valid_max], linestyle="--", linewidth=1.0, color="#334155")
    ax.set_title("LatentGlue")
    ax.set_xlabel("Observed DC50 Assay Activity")
    ax.set_ylabel("Predicted DC50 Assay Activity")
    ax.grid(alpha=0.2)
    save_figure(fig, os.path.join(figure_dir, "activity_prediction_scatter_latentglue.png"))

def generate_figures(metrics, figure_dir, activity_df, activity_reps, val_reps, eval_reps):
    if not figure_dir:
        return
    ensure_dir(figure_dir)
    plot_effective_dimension_figure(metrics, figure_dir)
    plot_uniformity_figure(metrics, figure_dir)
    plot_activity_scatter_figure(activity_df, activity_reps, metrics, figure_dir)
    plot_activity_scatter_latentglue_figure(activity_df, activity_reps, figure_dir)
    plot_hypersphere_geometry_figure(metrics, eval_reps, "eval", figure_dir)

def print_metric_group(title, metrics, groups):
    separator = "=" * 88
    print(f"\n\n{separator}\n== {title} ==\n{separator}")
    printed = False
    for prefix, label in groups:
        subset = {key: value for key, value in metrics.items() if key.startswith(prefix)}
        if not subset:
            continue
        printed = True
        print(f"\n{label}")
        for key in sorted(subset):
            print(f"  {key.split('/')[-1]}: {format_metric(subset[key])}")
    if not printed:
        print("  No metrics computed.")

def print_metrics(metrics):
    print_metric_group(
        "Component Effective Dimensionality",
        metrics,
        [
            ("effective_dim/val/latentglue", "GlueDegradDB val split | LatentGlue"),
            ("effective_dim/val/esmc_mean_pool", "GlueDegradDB val split | ESM-C"),
            ("effective_dim/val/molformer_xl_mean_pool", "GlueDegradDB val split | MoLFormer-XL"),
            ("effective_dim/val/projected_mean", "GlueDegradDB val split | Projected mean pool"),
            ("effective_dim/eval/latentglue", "GlueDegradDB-Eval | LatentGlue"),
            ("effective_dim/eval/esmc_mean_pool", "GlueDegradDB-Eval | ESM-C"),
            ("effective_dim/eval/molformer_xl_mean_pool", "GlueDegradDB-Eval | MoLFormer-XL"),
            ("effective_dim/eval/projected_mean", "GlueDegradDB-Eval | Projected mean pool"),
        ],
    )
    print_metric_group(
        "Component Uniformity",
        metrics,
        [
            ("uniformity/val/latentglue", "GlueDegradDB val split | LatentGlue"),
            ("uniformity/val/esmc_mean_pool", "GlueDegradDB val split | ESM-C"),
            ("uniformity/val/molformer_xl_mean_pool", "GlueDegradDB val split | MoLFormer-XL"),
            ("uniformity/val/projected_mean", "GlueDegradDB val split | Projected mean pool"),
            ("uniformity/eval/latentglue", "GlueDegradDB-Eval | LatentGlue"),
            ("uniformity/eval/esmc_mean_pool", "GlueDegradDB-Eval | ESM-C"),
            ("uniformity/eval/molformer_xl_mean_pool", "GlueDegradDB-Eval | MoLFormer-XL"),
            ("uniformity/eval/projected_mean", "GlueDegradDB-Eval | Projected mean pool"),
        ],
    )
    print_metric_group(
        "Activity Prediction",
        metrics,
        [
            ("activity/latentglue", "LatentGlue"),
            ("activity/baseline", "Projected-token mean pool baseline"),
            ("activity/frozen_feature", "Frozen-feature baseline"),
        ],
    )
def main(
    checkpoint_path="",
    device=DEFAULT_DEVICE,
    batch_size=0,
    train_csv_path=DEFAULT_TRAIN_CSV,
    eval_csv_path=DEFAULT_EVAL_CSV,
    activity_csv_path=DEFAULT_ACTIVITY_CSV,
    output_json_path=DEFAULT_OUTPUT_JSON,
    figure_dir=DEFAULT_FIGURE_DIR,
):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path or os.environ.get("LATENTGLUE_CHECKPOINT", "") or DEFAULT_CHECKPOINT_REPO_ID)
    encoder = load_checkpoint(checkpoint_path, device=device)
    autocast_enabled = str(device).startswith("cuda")
    val_df = read_df(train_csv_path, (*TERNARY_COLUMNS, "split"), split="val")
    eval_df = read_df(eval_csv_path, TERNARY_COLUMNS)
    activity_df = read_df(activity_csv_path, (*TERNARY_COLUMNS, "Value", "Target"))
    repr_cache = {}
    val_reps = None
    eval_reps = None

    metrics = {}
    if val_df is not None:
        val_reps = get_representations(repr_cache, "val", val_df, encoder, batch_size, autocast_enabled)
        metrics.update(evaluate_effective_dimensions(val_reps, dataset_name="val"))
        metrics.update(evaluate_uniformity(val_reps, dataset_name="val"))
    if eval_df is not None:
        eval_reps = get_representations(repr_cache, "eval", eval_df, encoder, batch_size, autocast_enabled)
        metrics.update(evaluate_effective_dimensions(eval_reps, dataset_name="eval"))
        metrics.update(evaluate_uniformity(eval_reps, dataset_name="eval"))
    activity_reps = get_representations(repr_cache, "activity", activity_df, encoder, batch_size, autocast_enabled)
    metrics.update(evaluate_activity(activity_df, activity_reps))

    print_metrics(metrics)
    generate_figures(metrics, figure_dir, activity_df, activity_reps, val_reps, eval_reps)
    if output_json_path:
        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics

if __name__ == "__main__":
    main()
