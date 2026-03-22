"""
Debug and evaluation helpers shared by training and full validation.

`train_subset/recon/*`, `val_set/recon/*`, `eval/recon/*`
    Masked reconstruction quality for the three reconstruction rotations.
    - `mse_target`, `mse_effector`, `mse_ligand`: masked-token mean squared
      error for that reconstruction direction
    - `cosine_target`, `cosine_effector`, `cosine_ligand`: masked-token cosine
      alignment for that reconstruction direction
    - `mse_macro`, `cosine_macro`: simple averages across the three directions

`*/geometry/*`
    Representation spread diagnostics.
    - `erank_pool_target`, `erank_pool_effector`, `erank_pool_ligand`:
      effective rank of the pooled encoder representations
    - `erank_pred_masked`: effective rank of the masked predictions produced
      during reconstruction

`eval/fairness/*`
    Effector-group fairness diagnostics.
    - `effector_macro_mse`, `effector_macro_cosine`: macro averages across
      sufficiently supported effector groups
    - `effector_worst_mse`, `effector_worst_cosine`: worst supported effector
      group under the same aggregation

`activity/frozen_feature_*`
    Downstream probe results on concatenated mean-pooled frozen backbone
    features. This is the fixed no-training reference.

`activity/baseline_*`
    Downstream probe results on concatenated mean-pooled projected token
    features from the trained encoder. This isolates projection-space quality
    without learned attention pooling.

`activity/latentglue_*`
    Downstream probe results on concatenated attention-pooled target /
    effector / ligand summaries from the trained encoder. This is the main
    learned-representation evaluation used during training.

`activity/delta_*`
    Improvement of `activity/latentglue_*` over `activity/baseline_*`.
    Positive Spearman deltas and negative RMSE deltas are better.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COMPONENT_LABELS = ("target", "effector", "ligand")

def masked_mean_pool(x, mask):
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype)
    return (x.float() * mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom

def build_target_balanced_folds(targets, values, n_folds=5):
    targets = np.asarray(targets)
    values = np.asarray(values)
    folds = [[] for _ in range(n_folds)]
    for target in sorted(set(targets.tolist())):
        target_indices = np.where(targets == target)[0]
        ordered = target_indices[np.argsort(values[target_indices])]
        for offset, index in enumerate(ordered.tolist()):
            folds[offset % n_folds].append(index)
    return [np.array(sorted(fold), dtype=np.int64) for fold in folds if fold]

def _regression_cv_metrics(features, values, targets, model_factory, n_folds=5):
    features = np.asarray(features, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets)
    folds = build_target_balanced_folds(targets, values, n_folds=n_folds)
    spearman_scores = []
    rmse_scores = []
    for fold_indices in folds:
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[fold_indices] = False
        if train_mask.sum() == 0 or fold_indices.size == 0:
            continue
        model = model_factory()
        model.fit(features[train_mask], values[train_mask])
        predictions = model.predict(features[fold_indices])
        score = spearmanr(values[fold_indices], predictions).statistic
        if score is None or np.isnan(score):
            raise RuntimeError(
                "Activity evaluation produced an undefined Spearman correlation for at least one fold."
            )
        spearman_scores.append(float(score))
        mse = mean_squared_error(values[fold_indices], predictions)
        rmse_scores.append(float(np.sqrt(mse)))
    if not spearman_scores:
        raise RuntimeError("Activity evaluation did not produce any valid cross-validation folds.")
    return {
        "spearman": float(np.mean(spearman_scores)),
        "rmse": float(np.mean(rmse_scores)),
    }

def ridge_cv_metrics(features, values, targets, alpha=1.0, n_folds=5):
    return _regression_cv_metrics(
        features,
        values,
        targets,
        model_factory=lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        n_folds=n_folds,
    )

def ridge_cv_predictions(features, values, targets, alpha=1.0, n_folds=5):
    features = np.asarray(features, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets)
    predictions = np.full(len(values), np.nan, dtype=np.float32)
    for fold_indices in build_target_balanced_folds(targets, values, n_folds=n_folds):
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[fold_indices] = False
        if train_mask.sum() == 0 or fold_indices.size == 0:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(features[train_mask], values[train_mask])
        predictions[fold_indices] = model.predict(features[fold_indices]).astype(np.float32)
    return predictions

# Log-scaled DC50*
def activity_targets(activity_df):
    values = activity_df["Value"].astype(float).clip(lower=1e-8)
    targets = activity_df["Target"].astype(str).to_numpy()
    y = np.log10(values.to_numpy(dtype=np.float32))
    return y, targets

def activity_probe_metrics(prefix, features, y, targets):
    ridge_scores = ridge_cv_metrics(features, y, targets)
    return {
        f"{prefix}_spearman": ridge_scores["spearman"],
        f"{prefix}_rmse": ridge_scores["rmse"],
    }

def summarize_effector_fairness(records, prefix="eval/fairness", min_support=5):
    grouped = {}
    for record in records:
        effector = str(record["effector"])
        bucket = grouped.setdefault(
            effector,
            {"mse_sum": 0.0, "cosine_sum": 0.0, "token_count": 0, "example_count": 0},
        )
        weight = max(int(record.get("token_count", 0)), 1)
        bucket["mse_sum"] += float(record["mse"]) * weight
        bucket["cosine_sum"] += float(record["cosine"]) * weight
        bucket["token_count"] += weight
        bucket["example_count"] += 1

    valid_groups = []
    for bucket in grouped.values():
        if bucket["example_count"] < min_support:
            continue
        valid_groups.append(
            {
                "mse": bucket["mse_sum"] / max(bucket["token_count"], 1),
                "cosine": bucket["cosine_sum"] / max(bucket["token_count"], 1),
            }
        )

    if not valid_groups:
        return {
            f"{prefix}/effector_macro_mse": 0.0,
            f"{prefix}/effector_worst_mse": 0.0,
            f"{prefix}/effector_macro_cosine": 0.0,
            f"{prefix}/effector_worst_cosine": 0.0,
        }

    macro_mse = sum(group["mse"] for group in valid_groups) / len(valid_groups)
    worst_mse = max(group["mse"] for group in valid_groups)
    macro_cosine = sum(group["cosine"] for group in valid_groups) / len(valid_groups)
    worst_cosine = min(group["cosine"] for group in valid_groups)
    return {
        f"{prefix}/effector_macro_mse": macro_mse,
        f"{prefix}/effector_worst_mse": worst_mse,
        f"{prefix}/effector_macro_cosine": macro_cosine,
        f"{prefix}/effector_worst_cosine": worst_cosine,
    }

def spectral_effective_dimension(z):
    if z.numel() == 0:
        return 0.0
    if z.ndim > 2:
        z = z.reshape(-1, z.size(-1))
    z = z.detach().float()
    if z.size(0) <= 1:
        return 0.0
    variances = z.var(dim=0, unbiased=False)
    total_variance = variances.sum().clamp(min=1e-8)
    probs = variances / total_variance
    entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum()
    return float(torch.exp(entropy).item())

def summarize_geometry_metrics(pooled_batches, pred_masked_chunks, prefix):
    if not pooled_batches:
        return {
            f"{prefix}/geometry/erank_pool_target": 0.0,
            f"{prefix}/geometry/erank_pool_effector": 0.0,
            f"{prefix}/geometry/erank_pool_ligand": 0.0,
            f"{prefix}/geometry/erank_pred_masked": 0.0,
        }

    t_pool = torch.cat([batch[0] for batch in pooled_batches], dim=0)
    e_pool = torch.cat([batch[1] for batch in pooled_batches], dim=0)
    l_pool = torch.cat([batch[2] for batch in pooled_batches], dim=0)
    pred_masked = (
        torch.cat(pred_masked_chunks, dim=0)
        if pred_masked_chunks
        else torch.empty((0, t_pool.size(-1)), dtype=t_pool.dtype)
    )
    return {
        f"{prefix}/geometry/erank_pool_target": spectral_effective_dimension(t_pool),
        f"{prefix}/geometry/erank_pool_effector": spectral_effective_dimension(e_pool),
        f"{prefix}/geometry/erank_pool_ligand": spectral_effective_dimension(l_pool),
        f"{prefix}/geometry/erank_pred_masked": spectral_effective_dimension(pred_masked),
    }

def masked_token_stats(predicted_component, teacher_component, drop_mask):
    mask = drop_mask.bool()
    token_count = int(mask.sum().item())
    if token_count == 0:
        batch_size = drop_mask.size(0)
        nan_vec = torch.full((batch_size,), float("nan"), device=drop_mask.device)
        zero = torch.zeros((), device=drop_mask.device, dtype=torch.float32)
        return {
            "mse": zero,
            "cosine": zero,
            "token_count": 0,
            "example_mse": nan_vec,
            "example_cosine": nan_vec,
            "example_counts": torch.zeros(batch_size, device=drop_mask.device, dtype=torch.long),
        }

    predicted = predicted_component.float()
    teacher = teacher_component.float()
    token_mse = (predicted - teacher).pow(2).mean(dim=-1)
    token_cosine = F.cosine_similarity(predicted, teacher, dim=-1)
    mask_float = mask.float()
    counts = mask.sum(dim=1)
    denom = counts.clamp(min=1).float()
    example_mse = torch.where(
        counts > 0,
        (token_mse * mask_float).sum(dim=1) / denom,
        torch.full((mask.size(0),), float("nan"), device=mask.device, dtype=torch.float32),
    )
    example_cosine = torch.where(
        counts > 0,
        (token_cosine * mask_float).sum(dim=1) / denom,
        torch.full((mask.size(0),), float("nan"), device=mask.device, dtype=torch.float32),
    )
    return {
        "mse": (token_mse * mask_float).sum() / mask_float.sum().clamp(min=1.0),
        "cosine": (token_cosine * mask_float).sum() / mask_float.sum().clamp(min=1.0),
        "token_count": token_count,
        "example_mse": example_mse,
        "example_cosine": example_cosine,
        "example_counts": counts,
    }

def init_component_sums():
    return {
        name: {"mse_sum": 0.0, "cosine_sum": 0.0, "token_count": 0}
        for name in COMPONENT_LABELS
    }

def update_component_sums(component_sums, component_name, mse, cosine, token_count):
    bucket = component_sums[component_name]
    bucket["mse_sum"] += float(mse) * token_count
    bucket["cosine_sum"] += float(cosine) * token_count
    bucket["token_count"] += int(token_count)

def finalize_component_metrics(component_sums, prefix):
    """Finalize component-wise and macro reconstruction metrics."""
    metrics = {}
    mse_values = []
    cosine_values = []
    for component_name in COMPONENT_LABELS:
        bucket = component_sums[component_name]
        denom = max(bucket["token_count"], 1)
        mse = bucket["mse_sum"] / denom
        cosine = bucket["cosine_sum"] / denom
        metrics[f"{prefix}/recon/mse_{component_name}"] = mse
        metrics[f"{prefix}/recon/cosine_{component_name}"] = cosine
        mse_values.append(mse)
        cosine_values.append(cosine)
    metrics[f"{prefix}/recon/mse_macro"] = sum(mse_values) / len(mse_values)
    metrics[f"{prefix}/recon/cosine_macro"] = sum(cosine_values) / len(cosine_values)
    return metrics

def _largest_remainder_allocation(total, counts):
    if total <= 0 or not counts:
        return {key: 0 for key in counts}
    total_available = sum(counts.values())
    if total_available <= total:
        return dict(counts)
    raw = {key: total * value / total_available for key, value in counts.items()}
    alloc = {key: min(counts[key], int(raw[key])) for key in counts}
    remaining = total - sum(alloc.values())
    if remaining <= 0:
        return alloc
    order = sorted(
        counts,
        key=lambda key: (raw[key] - alloc[key], counts[key]),
        reverse=True,
    )
    for key in order:
        if remaining <= 0:
            break
        capacity = counts[key] - alloc[key]
        if capacity <= 0:
            continue
        alloc[key] += 1
        remaining -= 1
    return alloc

def build_train_subset_df(csv_path, subset_size=256, seed=42):
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()
    if len(df) <= subset_size:
        return df.reset_index(drop=True)

    eff_counts = df["Effector UniProt"].astype(str).value_counts()

    def bin_name(effector_id):
        count = int(eff_counts.get(str(effector_id), 0))
        if count >= 100:
            return "head"
        if count >= 10:
            return "torso"
        return "tail"

    df["effector_bin"] = df["Effector UniProt"].astype(str).map(bin_name)
    if "Source" not in df.columns:
        df["Source"] = "unknown"

    bin_counts = df["effector_bin"].value_counts().to_dict()
    bin_targets = _largest_remainder_allocation(subset_size, bin_counts)
    for bin_label in ("torso", "tail"):
        available = bin_counts.get(bin_label, 0)
        minimum = min(32, available)
        if bin_targets.get(bin_label, 0) >= minimum:
            continue
        deficit = minimum - bin_targets.get(bin_label, 0)
        donor_order = [label for label in ("head", "torso", "tail") if label != bin_label]
        for donor in donor_order:
            reducible = max(
                0,
                bin_targets.get(donor, 0) - min(32, bin_counts.get(donor, 0))
                if donor in ("torso", "tail")
                else bin_targets.get(donor, 0),
            )
            take = min(deficit, reducible)
            if take <= 0:
                continue
            bin_targets[donor] -= take
            bin_targets[bin_label] = bin_targets.get(bin_label, 0) + take
            deficit -= take
            if deficit == 0:
                break

    sampled_parts = []
    for offset, bin_label in enumerate(("head", "torso", "tail")):
        bin_df = df[df["effector_bin"] == bin_label].copy()
        target_count = min(bin_targets.get(bin_label, 0), len(bin_df))
        if target_count <= 0:
            continue
        source_counts = bin_df["Source"].astype(str).value_counts().to_dict()
        source_targets = _largest_remainder_allocation(target_count, source_counts)
        for source_idx, (source_name, source_target) in enumerate(sorted(source_targets.items())):
            if source_target <= 0:
                continue
            source_df = bin_df[bin_df["Source"].astype(str) == source_name]
            sampled_parts.append(
                source_df.sample(
                    n=min(source_target, len(source_df)),
                    replace=False,
                    random_state=seed + (offset * 17) + source_idx,
                )
            )

    subset_df = pd.concat(sampled_parts, ignore_index=True)
    if len(subset_df) < subset_size:
        picked_ids = set(subset_df["ID"].tolist()) if "ID" in subset_df.columns else set()
        remaining_df = df[~df["ID"].isin(picked_ids)].copy() if picked_ids else df.copy()
        shortfall = min(subset_size - len(subset_df), len(remaining_df))
        if shortfall > 0:
            subset_df = pd.concat(
                [
                    subset_df,
                    remaining_df.sample(n=shortfall, replace=False, random_state=seed + 999),
                ],
                ignore_index=True,
            )

    return subset_df.head(subset_size).reset_index(drop=True)
