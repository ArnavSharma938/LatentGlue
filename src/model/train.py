"""
1. One masked-reconstruction forward pass
   - tokenize the raw target sequence, effector sequence, and ligand SMILES
   - run a teacher pass on the full unmasked ternary through the frozen
     backbones, then through the trainable projection heads, role embeddings,
     and pooling heads
   - choose one rotation: target-masked, effector-masked, or ligand-masked
   - sample contiguous masked spans on the selected component while leaving at
     least a small visible remainder
   - apply raw mask tokens to that component only
   - re-encode only the masked branch from masked raw inputs
   - reuse detached teacher latents for the two visible branches so the context
     branches do not move just to solve the masked loss
   - overwrite masked positions in the selected latent stream with learned
     component-specific predictor mask embeddings
   - concatenate the target, effector, and ligand latent streams into one
     ternary sequence and feed that sequence to the relational predictor
   - compare the predictor output against the detached teacher latents only at
     masked positions

3. Loss and update
   - compute masked reconstruction loss from the selected component
   - add SIGReg during training to keep the pooled and predicted manifolds
     well-behaved
   - backpropagate through the trainable encoder path and the predictor

4. Periodic evaluation
   - on `diag_interval`, run deterministic reconstruction diagnostics on the
     fixed train subset, validation set, and eval set
   - on `activity_interval_epochs`, run downstream activity probes on the
     learned encoder representation
   - save checkpoints during training and rescore them later with
     `src.validation.full_eval`
"""

import math
import os
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from .model import LatentGlueEncoder, RelationalPredictor
from .dataset import TernaryDataset, collate_ternary
from src.validation.in_train_eval import (
    COMPONENT_LABELS,
    activity_probe_metrics,
    activity_targets,
    build_train_subset_df,
    finalize_component_metrics,
    init_component_sums,
    masked_mean_pool,
    masked_token_stats,
    summarize_effector_fairness,
    summarize_geometry_metrics,
    update_component_sums,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

DEFAULT_WARMUP_RATIO = 0.10
DEFAULT_MIN_LR = 5e-6
MASK_FRACTION = 0.60
MASK_BLOCKS = 3
AUTOCAST_DTYPE = torch.bfloat16
MASK_MIN_VISIBLE_TOKENS = 1
COMPONENT_NAMES = ("Target", "Effector", "Ligand")

@torch.no_grad()
def save_latentglue_checkpoint(encoder, predictor, optimizer, scheduler, epoch, step, name=None):
    os.makedirs("checkpoints", exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "encoder_state_dict": encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if name:
        filename = f"{name}_step_{step}.pt"
    else:
        filename = f"latentglue_step_{step}.pt"
    path = os.path.join("checkpoints", filename)
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")

class IsotropicGaussianRegularizer(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        if proj.size(0) <= 1:
            return torch.tensor(0.0, device=proj.device, dtype=proj.dtype)
        sketch = torch.randn(proj.size(-1), 256, device=proj.device, dtype=proj.dtype)
        sketch = sketch.div_(sketch.norm(p=2, dim=0).clamp(min=1e-8))
        x_t = (proj @ sketch).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return (err @ self.weights).mean()

def random_partition(total, parts, min_each=0, generator=None):
    if parts <= 0:
        return []
    if parts == 1:
        return [int(total)]

    base_total = parts * min_each
    if total < base_total:
        raise ValueError(f"Cannot partition total={total} into parts={parts} with min_each={min_each}")

    remaining = total - base_total
    if remaining == 0:
        return [int(min_each)] * parts

    weights = torch.rand(parts, generator=generator)
    weights = weights / weights.sum().clamp(min=1e-8)
    raw = weights * remaining
    ints = torch.floor(raw).long()
    shortfall = int(remaining - ints.sum().item())
    if shortfall > 0:
        fractional = raw - ints.float()
        order = torch.argsort(fractional, descending=True)
        ints[order[:shortfall]] += 1
    return [int(value) + int(min_each) for value in ints.tolist()]

def generate_contiguous_block_mask(
    valid_mask,
    generator=None,
    mask_fraction=MASK_FRACTION,
    num_blocks=MASK_BLOCKS,
    min_visible_tokens=MASK_MIN_VISIBLE_TOKENS,
):
    drop_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    masked_counts = []
    valid_counts = []

    for row_idx in range(valid_mask.size(0)):
        valid_len = int(valid_mask[row_idx].sum().item())
        valid_counts.append(valid_len)
        if valid_len <= min_visible_tokens:
            masked_counts.append(0)
            continue

        max_mask_total = max(valid_len - min_visible_tokens, 0)
        requested = max(1, int(round(valid_len * mask_fraction)))
        mask_total = min(requested, max_mask_total)
        if mask_total <= 0:
            masked_counts.append(0)
            continue

        block_count = min(int(num_blocks), mask_total)
        block_lengths = random_partition(mask_total, block_count, min_each=1, generator=generator)

        gap_budget = valid_len - mask_total
        min_internal_gap = 1 if gap_budget >= (block_count - 1) else 0
        base_gaps = [0] + [min_internal_gap] * max(block_count - 1, 0) + [0]
        extra_gap_budget = gap_budget - (block_count - 1) * min_internal_gap
        extra_gaps = random_partition(extra_gap_budget, block_count + 1, min_each=0, generator=generator)
        gaps = [base_gaps[i] + extra_gaps[i] for i in range(block_count + 1)]

        cursor = gaps[0]
        for block_idx, block_len in enumerate(block_lengths):
            drop_mask[row_idx, cursor : cursor + block_len] = True
            cursor += block_len + gaps[block_idx + 1]

        masked_counts.append(mask_total)

    drop_mask &= valid_mask.bool()
    return drop_mask, masked_counts, valid_counts

def expand_truncated_drop_mask_to_raw(raw_mask, truncated_drop_mask):
    raw_drop_mask = torch.zeros_like(raw_mask, dtype=torch.bool)
    truncated_width = truncated_drop_mask.size(1)
    raw_drop_mask[:, 1 : 1 + truncated_width] = truncated_drop_mask
    raw_drop_mask &= raw_mask.bool()
    return raw_drop_mask

def apply_raw_mask_tokens(tokens, raw_drop_mask, mask_token_id):
    masked_tokens = tokens.clone()
    masked_tokens[raw_drop_mask] = int(mask_token_id)
    return masked_tokens

def build_mask_generator(mask_seed=None):
    if mask_seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(mask_seed))
    return generator

def build_status_mask(masked_component, component_lengths, drop_mask_tr, device):
    len_t, len_e, len_l = component_lengths
    total_len = len_t + len_e + len_l
    status_mask = torch.zeros(
        (drop_mask_tr.size(0), total_len),
        device=device,
        dtype=torch.bool,
    )
    offsets = (0, len_t, len_t + len_e)
    width = component_lengths[int(masked_component)]
    offset = offsets[int(masked_component)]
    status_mask[:, offset : offset + width][drop_mask_tr] = True
    return status_mask

def run_predictor_for_masked_component(
    predictor,
    student_components,
    student_masks,
    masked_component,
    drop_mask_tr,
):
    component_lengths = tuple(component.size(1) for component in student_components)
    full_seq = torch.cat(student_components, dim=1)
    padding_mask = torch.cat(student_masks, dim=1)
    status_mask = build_status_mask(
        masked_component,
        component_lengths,
        drop_mask_tr,
        device=full_seq.device,
    )

    predicted_seq = predictor(
        full_seq,
        component_lengths=component_lengths,
        status_mask=status_mask,
        rotation_index=masked_component,
        padding_mask=padding_mask,
    )
    predicted_component = torch.split(predicted_seq, list(component_lengths), dim=1)[int(masked_component)]
    return predicted_component

def run_masked_reconstruction(
    encoder,
    predictor,
    regularizer,
    target_seqs,
    effector_seqs,
    smiles,
    rotation_index,
    mask_seed=None,
    autocast_enabled=True,
    mask_fraction=MASK_FRACTION,
    num_mask_blocks=MASK_BLOCKS,
    compute_sigreg=True,
):
    with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
        target_toks, target_mask_raw, effector_toks, effector_mask_raw, ligand_toks, ligand_mask_raw = encoder.prepare_inputs(
            target_seqs,
            effector_seqs,
            smiles,
        )

        teacher_components, teacher_pooled, teacher_masks, _ = encoder(
            target_toks,
            effector_toks,
            ligand_toks,
            target_mask_raw,
            effector_mask_raw,
            ligand_mask_raw,
            cache_keys=(target_seqs, effector_seqs, smiles),
        )

        # 1. Masking Configuration
        masked_component = int(rotation_index % 3)
        teacher_component = teacher_components[masked_component]
        component_valid_mask = teacher_masks[masked_component]

        generator = build_mask_generator(mask_seed)
        drop_mask_tr, masked_counts, valid_counts = generate_contiguous_block_mask(
            component_valid_mask,
            generator=generator,
            mask_fraction=mask_fraction,
            num_blocks=num_mask_blocks,
        )

        # 2. Raw Token Masking
        raw_tokens = [target_toks, effector_toks, ligand_toks]
        raw_masks = [target_mask_raw, effector_mask_raw, ligand_mask_raw]
        masked_raw_tokens = list(raw_tokens)
        raw_drop_mask = expand_truncated_drop_mask_to_raw(raw_masks[masked_component], drop_mask_tr)
        mask_token_id = encoder.get_component_mask_token_id(masked_component)
        masked_raw_tokens[masked_component] = apply_raw_mask_tokens(
            masked_raw_tokens[masked_component],
            raw_drop_mask,
            mask_token_id=mask_token_id,
        )

        # 3. Student Forward Pass (masked branch only)
        student_components = [component.detach() for component in teacher_components]
        student_masks = list(teacher_masks)
        student_component, student_mask, _ = encoder.forward_component(
            masked_component,
            masked_raw_tokens[masked_component],
            raw_masks[masked_component],
        )
        student_components[masked_component] = student_component
        student_masks[masked_component] = student_mask

        # 4. Identity Masking
        masked_student_component = student_components[masked_component].clone()
        mask_embeddings = predictor.get_mask_embedding(
            masked_component,
            batch_size=masked_student_component.size(0),
            device=masked_student_component.device,
            dtype=masked_student_component.dtype,
        ).expand(-1, masked_student_component.size(1), -1)
        if mask_embeddings.shape != masked_student_component.shape:
            raise RuntimeError(
                "Predictor mask embedding shape mismatch: "
                f"got {tuple(mask_embeddings.shape)}, expected {tuple(masked_student_component.shape)}. "
                "Mask embeddings must match the encoder token dimension before predictor projection."
            )
        masked_student_component[drop_mask_tr] = mask_embeddings[drop_mask_tr]
        student_components[masked_component] = masked_student_component

        # 5. Prediction & Reconstruction
        predicted_component = run_predictor_for_masked_component(
            predictor,
            student_components,
            student_masks,
            masked_component=masked_component,
            drop_mask_tr=drop_mask_tr,
        )
        teacher_component = teacher_component.detach()
        masked_stats = masked_token_stats(predicted_component, teacher_component, drop_mask_tr)
        predicted_masked = predicted_component[drop_mask_tr]
        teacher_masked = teacher_component[drop_mask_tr]

        if masked_stats["token_count"] == 0:
            mse_loss = torch.zeros((), device=teacher_components[0].device, dtype=teacher_components[0].dtype)
            if compute_sigreg:
                t_pool, e_pool, l_pool = teacher_pooled
                sigreg_loss = (regularizer(t_pool) + regularizer(e_pool) + regularizer(l_pool)) / 3.0
            else:
                sigreg_loss = torch.zeros((), device=teacher_components[0].device, dtype=teacher_components[0].dtype)
        else:
            mse_loss = masked_stats["mse"].to(dtype=teacher_components[0].dtype)
            if compute_sigreg:
                t_pool, e_pool, l_pool = teacher_pooled
                teacher_reg = (regularizer(t_pool) + regularizer(e_pool) + regularizer(l_pool)) / 3.0
                sigreg_loss = (teacher_reg + regularizer(predicted_masked)) / 2.0
            else:
                sigreg_loss = torch.zeros((), device=teacher_components[0].device, dtype=teacher_components[0].dtype)

    masked_token_count = int(sum(masked_counts))
    return {
        "mse_loss": mse_loss,
        "sigreg_loss": sigreg_loss,
        "teacher_pooled": teacher_pooled,
        "teacher_masked": teacher_masked,
        "predicted_masked": predicted_masked,
        "masked_component": masked_component,
        "token_count": masked_token_count,
        "cosine": masked_stats["cosine"],
        "example_mse": masked_stats["example_mse"],
        "example_cosine": masked_stats["example_cosine"],
        "example_counts": masked_stats["example_counts"],
        "teacher_bad": float(torch.isnan(teacher_masked).any().item() or torch.isinf(teacher_masked).any().item()),
        "pred_bad": float(torch.isnan(predicted_masked).any().item() or torch.isinf(predicted_masked).any().item()),
    }

def train_step(
    encoder,
    predictor,
    regularizer,
    optimizer,
    scheduler,
    scaler,
    target_seqs,
    effector_seqs,
    smiles,
    rotation_index,
    trainable_params,
    clip_norm=1.0,
    autocast_enabled=True,
    reg_lambda=0.50,
    mask_fraction=MASK_FRACTION,
    num_mask_blocks=MASK_BLOCKS,
):
    outputs = run_masked_reconstruction(
        encoder,
        predictor,
        regularizer,
        target_seqs,
        effector_seqs,
        smiles,
        rotation_index=rotation_index,
        mask_seed=None,
        autocast_enabled=autocast_enabled,
        mask_fraction=mask_fraction,
        num_mask_blocks=num_mask_blocks,
    )

    mse_loss = outputs["mse_loss"]
    sigreg_loss = outputs["sigreg_loss"]
    total_loss = mse_loss + reg_lambda * sigreg_loss
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
    else:
        total_loss.backward()

    prot_proj_grad = module_grad_norm(encoder.prot_proj)
    mol_proj_grad = module_grad_norm(encoder.mol_proj)
    metrics = {"train/loss_total": total_loss.detach()}
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm)
    step_was_skipped = False
    if scaler is not None:
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        step_was_skipped = scaler.get_scale() < prev_scale
    else:
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    metrics.update(
        {
            "train/loss_recon": mse_loss.detach(),
            "train/loss_sigreg": sigreg_loss.detach(),
            "train/recon_cosine": float(outputs["cosine"].detach().float().item() if isinstance(outputs["cosine"], torch.Tensor) else outputs["cosine"]),
            "train/grad_norm/global": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "train/grad_norm/prot_proj": prot_proj_grad,
            "train/grad_norm/mol_proj": mol_proj_grad,
            "train/amp_step_skipped": float(step_was_skipped),
            "train/masked_component": float(outputs["masked_component"]),
            "train/nan_inf_teacher": outputs["teacher_bad"],
            "train/nan_inf_pred": outputs["pred_bad"],
        }
    )
    return metrics

def module_grad_norm(module):
    sq_terms = [
        param.grad.detach().float().pow(2).sum()
        for param in module.parameters()
        if param.grad is not None
    ]
    if not sq_terms:
        return 0.0
    return float(torch.stack(sq_terms).sum().sqrt().item())

def build_fixed_subset_loader(csv_path, batch_size, subset_size=256, num_workers=None):
    subset_df = build_train_subset_df(csv_path, subset_size=subset_size, seed=42)
    dataset = TernaryDataset(csv_path=None, df=subset_df)
    if num_workers is None:
        num_workers = max(0, os.cpu_count() - 2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ternary,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

@torch.no_grad()
def evaluate_dataset(
    encoder,
    predictor,
    regularizer,
    loader,
    prefix,
    autocast_enabled=True,
    mask_fraction=MASK_FRACTION,
    num_mask_blocks=MASK_BLOCKS,
    include_fairness=False,
):
    encoder.eval()
    predictor.eval()
    regularizer.eval()

    component_sums = init_component_sums()
    pooled_batches = []
    pred_masked_chunks = []
    fairness_records = []
    teacher_bad = 0.0
    pred_bad = 0.0
    for batch_idx, batch in enumerate(loader):
        for component_id, component_name in enumerate(COMPONENT_LABELS):
            mask_seed = 1000 + batch_idx * 11 + component_id
            outputs = run_masked_reconstruction(
                encoder,
                predictor,
                regularizer,
                batch["target_seq"],
                batch["effector_seq"],
                batch["smiles"],
                rotation_index=component_id,
                mask_seed=mask_seed,
                autocast_enabled=autocast_enabled,
                mask_fraction=mask_fraction,
                num_mask_blocks=num_mask_blocks,
                compute_sigreg=False,
            )
            update_component_sums(
                component_sums,
                component_name,
                float(outputs["mse_loss"].detach().float().item()),
                float(outputs["cosine"].detach().float().item()),
                outputs["token_count"],
            )
            teacher_bad = max(teacher_bad, outputs["teacher_bad"])
            pred_bad = max(pred_bad, outputs["pred_bad"])
            if component_id == 0:
                pooled_batches.append(tuple(pool.detach().float().cpu() for pool in outputs["teacher_pooled"]))
            if outputs["predicted_masked"].numel() > 0:
                pred_masked_chunks.append(outputs["predicted_masked"].detach().float().cpu())
            if include_fairness:
                example_counts = outputs["example_counts"].detach().cpu().tolist()
                example_mse = outputs["example_mse"].detach().float().cpu().tolist()
                example_cos = outputs["example_cosine"].detach().float().cpu().tolist()
                for row_idx, example_count in enumerate(example_counts):
                    if int(example_count) <= 0:
                        continue
                    fairness_records.append(
                        {
                            "effector": batch["effector_id"][row_idx],
                            "mse": example_mse[row_idx],
                            "cosine": example_cos[row_idx],
                            "token_count": int(example_count),
                        }
                    )
    metrics = {}
    if prefix == "train_subset":
        subset_metrics = finalize_component_metrics(component_sums, prefix)
        for component_name in COMPONENT_LABELS:
            metrics[f"{prefix}/recon/mse_{component_name}"] = subset_metrics[f"{prefix}/recon/mse_{component_name}"]
    else:
        metrics.update(finalize_component_metrics(component_sums, prefix))
        metrics.update(summarize_geometry_metrics(pooled_batches, pred_masked_chunks, prefix))
        metrics[f"{prefix}/nan_inf_teacher"] = teacher_bad
        metrics[f"{prefix}/nan_inf_pred"] = pred_bad
    if include_fairness:
        metrics.update(summarize_effector_fairness(fairness_records, prefix=f"{prefix}/fairness"))
    return metrics

@torch.no_grad()
def collect_activity_representations(activity_df, encoder, batch_size, autocast_enabled=True):
    dataset = TernaryDataset(csv_path=None, df=activity_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ternary)
    trained_chunks = []
    baseline_chunks = []

    for batch in loader:
        with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
            target_toks, target_mask_raw, effector_toks, effector_mask_raw, ligand_toks, ligand_mask_raw = encoder.prepare_inputs(
                batch["target_seq"],
                batch["effector_seq"],
                batch["smiles"],
            )
            components, pooled, masks, _ = encoder(
                target_toks,
                effector_toks,
                ligand_toks,
                target_mask_raw,
                effector_mask_raw,
                ligand_mask_raw,
                compute_pools=True,
                cache_keys=(batch["target_seq"], batch["effector_seq"], batch["smiles"]),
            )
            trained_chunks.append(
                torch.cat([pooled[0], pooled[1], pooled[2]], dim=-1).detach().float().cpu()
            )

        baseline_chunks.append(
            torch.cat(
                [
                    masked_mean_pool(components[0], masks[0]),
                    masked_mean_pool(components[1], masks[1]),
                    masked_mean_pool(components[2], masks[2]),
                ],
                dim=-1,
            ).detach().float().cpu()
        )

    return torch.cat(trained_chunks, dim=0).numpy(), torch.cat(baseline_chunks, dim=0).numpy()

@torch.no_grad()
def collect_frozen_activity_representations(activity_df, encoder, batch_size, autocast_enabled=True):
    dataset = TernaryDataset(csv_path=None, df=activity_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ternary)
    frozen_chunks = []

    def truncate_backbone(x, mask):
        batch_size_local = x.size(0)
        x_trunc = x[:, 1:, :]
        mask_trunc = mask[:, 1:].clone()
        eos_indices = mask_trunc.sum(dim=1) - 1
        mask_trunc[torch.arange(batch_size_local, device=x.device), eos_indices] = False
        return x_trunc, mask_trunc

    def masked_mean(x, mask):
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype)
        return (x.float() * mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom

    for batch in loader:
        with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
            target_toks, target_mask_raw, effector_toks, effector_mask_raw, ligand_toks, ligand_mask_raw = encoder.prepare_inputs(
                batch["target_seq"],
                batch["effector_seq"],
                batch["smiles"],
            )
            target_backbone = encoder._encode_protein_backbone(target_toks, target_mask_raw)
            effector_backbone = encoder._encode_protein_backbone(effector_toks, effector_mask_raw)
            ligand_backbone = encoder._encode_ligand_backbone(ligand_toks, ligand_mask_raw)

        target_backbone, target_mask = truncate_backbone(target_backbone, target_mask_raw)
        effector_backbone, effector_mask = truncate_backbone(effector_backbone, effector_mask_raw)
        ligand_backbone, ligand_mask = truncate_backbone(ligand_backbone, ligand_mask_raw)

        frozen_chunks.append(
            torch.cat(
                [
                    masked_mean(target_backbone, target_mask),
                    masked_mean(effector_backbone, effector_mask),
                    masked_mean(ligand_backbone, ligand_mask),
                ],
                dim=-1,
            ).detach().float().cpu()
        )

    return torch.cat(frozen_chunks, dim=0).numpy()

@torch.no_grad()
def evaluate_activity_suite(activity_csv_path, encoder, batch_size, autocast_enabled=True):
    if not activity_csv_path or not os.path.exists(activity_csv_path):
        return {}
    activity_df = pd.read_csv(activity_csv_path)
    if len(activity_df) == 0 or "Value" not in activity_df.columns or "Target" not in activity_df.columns:
        return {}

    was_training_encoder = encoder.training
    encoder.eval()
    try:
        trained_repr, baseline_repr = collect_activity_representations(
            activity_df,
            encoder,
            batch_size,
            autocast_enabled=autocast_enabled,
        )
    finally:
        if was_training_encoder:
            encoder.train()

    y, targets = activity_targets(activity_df)
    trained_scores = activity_probe_metrics("activity/latentglue", trained_repr, y, targets)
    baseline_scores = activity_probe_metrics("activity/baseline", baseline_repr, y, targets)
    return {
        **trained_scores,
        **baseline_scores,
        "activity/delta_spearman": trained_scores["activity/latentglue_spearman"] - baseline_scores["activity/baseline_spearman"],
        "activity/delta_rmse": trained_scores["activity/latentglue_rmse"] - baseline_scores["activity/baseline_rmse"],
    }

@torch.no_grad()
def evaluate_frozen_feature_activity_baseline(activity_csv_path, encoder, batch_size, autocast_enabled=True):
    if not activity_csv_path or not os.path.exists(activity_csv_path):
        return {}
    activity_df = pd.read_csv(activity_csv_path)
    if len(activity_df) == 0 or "Value" not in activity_df.columns or "Target" not in activity_df.columns:
        return {}

    was_training = encoder.training
    encoder.eval()
    try:
        frozen_repr = collect_frozen_activity_representations(
            activity_df,
            encoder,
            batch_size,
            autocast_enabled=autocast_enabled,
        )
    finally:
        if was_training:
            encoder.train()

    y, targets = activity_targets(activity_df)
    return activity_probe_metrics("activity/frozen_feature", frozen_repr, y, targets)

def get_optimizer(encoder, predictor, lr=2e-4):
    decay = []
    no_decay = []
    for model in (encoder, predictor):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lowered = name.lower()
            if any(token in lowered for token in ("bias", "norm", "embed", "seed")):
                no_decay.append(param)
            else:
                decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-3, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-7,
    )

def resolve_warmup_steps(total_steps, warmup_ratio=DEFAULT_WARMUP_RATIO):
    if total_steps <= 1:
        return 1
    resolved = int(math.ceil(float(total_steps) * float(warmup_ratio)))
    return max(1, min(total_steps - 1, resolved))

def get_scheduler(optimizer, warmup_steps, total_steps, base_lr=2e-4, min_lr=DEFAULT_MIN_LR):
    def lr_lambda(step_idx):
        if step_idx <= warmup_steps:
            return float(step_idx) / float(max(1, warmup_steps))
        progress = min(
            1.0,
            max(0.0, float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))),
        )
        floor = min_lr / base_lr
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def format_component_name(component_index):
    return COMPONENT_NAMES[int(component_index) % len(COMPONENT_NAMES)]

def print_train_metrics(step, metrics):
    print(f"[TRAIN] Step {step}")
    print(
        f"  Loss: {float(metrics.get('train/loss_total', 0.0)):.6f} total | "
        f"{float(metrics.get('train/loss_recon', 0.0)):.6f} recon | "
        f"{float(metrics.get('train/loss_sigreg', 0.0)):.6f} sigreg"
    )
    print(f"  Recon: {float(metrics.get('train/recon_cosine', 0.0)):.6f} cosine")
    print(
        f"  Mask: {format_component_name(metrics.get('train/masked_component', 0.0))}"
    )
    print(
        f"  Grad: {float(metrics.get('train/grad_norm/global', 0.0)):.3f}"
    )
    print(
        f"  ProjGrad: {float(metrics.get('train/grad_norm/prot_proj', 0.0)):.3f} | "
        f"{float(metrics.get('train/grad_norm/mol_proj', 0.0)):.3f}"
    )
    print(f"  AMP: skipped={int(metrics.get('train/amp_step_skipped', 0.0))}")
    print(
        f"  NaN / Inf: teacher={int(metrics.get('train/nan_inf_teacher', 0.0))} | "
        f"pred={int(metrics.get('train/nan_inf_pred', 0.0))}"
    )


def print_val_metrics(train_subset_metrics, val_metrics):
    print("[VAL_SET]")
    print(
        f"  TrainSubset-MSE: {float(train_subset_metrics.get('train_subset/recon/mse_target', 0.0)):.6f} | "
        f"{float(train_subset_metrics.get('train_subset/recon/mse_effector', 0.0)):.6f} | "
        f"{float(train_subset_metrics.get('train_subset/recon/mse_ligand', 0.0)):.6f}"
    )
    print(
        f"  Recon-MSE: {float(val_metrics.get('val_set/recon/mse_macro', 0.0)):.6f} | "
        f"{float(val_metrics.get('val_set/recon/mse_target', 0.0)):.6f} | "
        f"{float(val_metrics.get('val_set/recon/mse_effector', 0.0)):.6f} | "
        f"{float(val_metrics.get('val_set/recon/mse_ligand', 0.0)):.6f}"
    )
    print(
        f"  Recon-Cos: {float(val_metrics.get('val_set/recon/cosine_macro', 0.0)):.6f} | "
        f"{float(val_metrics.get('val_set/recon/cosine_target', 0.0)):.6f} | "
        f"{float(val_metrics.get('val_set/recon/cosine_effector', 0.0)):.6f} | "
        f"{float(val_metrics.get('val_set/recon/cosine_ligand', 0.0)):.6f}"
    )
    print(
        f"  eRank: {float(val_metrics.get('val_set/geometry/erank_pool_target', 0.0)):.1f} | "
        f"{float(val_metrics.get('val_set/geometry/erank_pool_effector', 0.0)):.1f} | "
        f"{float(val_metrics.get('val_set/geometry/erank_pool_ligand', 0.0)):.1f} | "
        f"{float(val_metrics.get('val_set/geometry/erank_pred_masked', 0.0)):.1f}"
    )
    print(
        f"  NaN / Inf: teacher={int(val_metrics.get('val_set/nan_inf_teacher', 0.0))} | "
        f"pred={int(val_metrics.get('val_set/nan_inf_pred', 0.0))}"
    )


def print_eval_metrics(eval_metrics):
    print("[EVAL_SET]")
    print(
        f"  Recon-MSE: {float(eval_metrics.get('eval/recon/mse_macro', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/recon/mse_target', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/recon/mse_effector', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/recon/mse_ligand', 0.0)):.6f}"
    )
    print(
        f"  Recon-Cos: {float(eval_metrics.get('eval/recon/cosine_macro', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/recon/cosine_target', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/recon/cosine_effector', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/recon/cosine_ligand', 0.0)):.6f}"
    )
    print(
        f"  eRank: {float(eval_metrics.get('eval/geometry/erank_pool_target', 0.0)):.1f} | "
        f"{float(eval_metrics.get('eval/geometry/erank_pool_effector', 0.0)):.1f} | "
        f"{float(eval_metrics.get('eval/geometry/erank_pool_ligand', 0.0)):.1f} | "
        f"{float(eval_metrics.get('eval/geometry/erank_pred_masked', 0.0)):.1f}"
    )
    print(
        f"  Effector-Fairness: {float(eval_metrics.get('eval/fairness/effector_macro_mse', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/fairness/effector_worst_mse', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/fairness/effector_macro_cosine', 0.0)):.6f} | "
        f"{float(eval_metrics.get('eval/fairness/effector_worst_cosine', 0.0)):.6f}"
    )
    print(
        f"  NaN / Inf: teacher={int(eval_metrics.get('eval/nan_inf_teacher', 0.0))} | "
        f"pred={int(eval_metrics.get('eval/nan_inf_pred', 0.0))}"
    )

def train(
    csv_path,
    epochs=40, # The reported checkpoint ran with this but manually exited at epoch 18, with the saved checkpoint being post-epoch 4
    batch_size=64,
    lr=1e-4,
    device=None,
    autocast_enabled=True,
    lamb=0.50,
    diag_interval=40,
    complex_balance_power=1.0,
    mask_fraction=MASK_FRACTION,
    num_mask_blocks=MASK_BLOCKS,
    eval_csv_path="data/GlueDegradDB-Eval.csv",
    activity_csv_path="data/GlueDegradDB-Activity.csv",
    train_subset_size=256,
    activity_interval_epochs=1,
    run_activity_eval=True,
    warmup_ratio=DEFAULT_WARMUP_RATIO,
    min_lr=DEFAULT_MIN_LR,
):
    assert torch.cuda.is_available(), "CUDA is required for this trainer."
    if device is None:
        device = "cuda"

    from .dataset import get_dataloader

    encoder = LatentGlueEncoder(device=device)
    predictor = RelationalPredictor().to(device)
    regularizer = IsotropicGaussianRegularizer().to(device)
    model_params = list(encoder.parameters()) + list(predictor.parameters())
    total_params = sum(param.numel() for param in model_params)
    trainable_count = sum(param.numel() for param in model_params if param.requires_grad)
    print(
        f"\n[MODEL] LatentGlue: "
        f"{trainable_count / 1e6:.2f}M trainable / {total_params / 1e6:.2f}M total "
        f"({100.0 * trainable_count / max(total_params, 1):.2f}%)\n"
    )

    train_loader = get_dataloader(
        csv_path,
        batch_size=batch_size,
        split="train",
        weighted=True,
        complex_balance_power=complex_balance_power,
    )
    val_loader = get_dataloader(
        csv_path,
        batch_size=batch_size,
        split="validation",
        weighted=False,
        shuffle=False,
    )
    train_subset_loader = build_fixed_subset_loader(
        csv_path,
        batch_size=min(batch_size, train_subset_size),
        subset_size=train_subset_size,
    )
    eval_loader = None
    if eval_csv_path and os.path.exists(eval_csv_path):
        eval_loader = get_dataloader(
            eval_csv_path,
            batch_size=batch_size,
            split="val",
            weighted=False,
            shuffle=False,
        )

    optimizer = get_optimizer(encoder, predictor, lr=lr)
    trainable_params = [param for group in optimizer.param_groups for param in group["params"]]

    total_steps = epochs * len(train_loader)
    warmup_steps = resolve_warmup_steps(total_steps, warmup_ratio=warmup_ratio)
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps, base_lr=lr, min_lr=min_lr)

    wandb.init(
        project="LatentGlue",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "effective_batch_size": batch_size,
            "lr": lr,
            "architecture": "LatentGlue-gmlr-v1",
            "lamb": lamb,
            "complex_balance_power": complex_balance_power,
            "mask_fraction": mask_fraction,
            "mask_blocks": num_mask_blocks,
            "diag_interval": diag_interval,
            "train_subset_size": train_subset_size,
            "autocast_dtype": str(AUTOCAST_DTYPE).replace("torch.", ""),
            "activity_interval_epochs": activity_interval_epochs,
            "run_activity_eval": run_activity_eval,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "min_lr": min_lr,
        },
    )

    if run_activity_eval:
        frozen_feature_activity_metrics = evaluate_frozen_feature_activity_baseline(
            activity_csv_path,
            encoder,
            batch_size=min(batch_size, 128),
            autocast_enabled=autocast_enabled,
        )
        if frozen_feature_activity_metrics:
            wandb.log(frozen_feature_activity_metrics, step=0)

    use_grad_scaler = autocast_enabled and AUTOCAST_DTYPE == torch.float16
    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=True,
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100,
        )
        if use_grad_scaler
        else None
    )

    global_step = 0
    effective_diag_interval = max(1, int(diag_interval))

    for epoch in range(epochs):
        encoder.train()
        predictor.train()
        regularizer.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            metrics = train_step(
                encoder,
                predictor,
                regularizer,
                optimizer,
                scheduler,
                scaler,
                batch["target_seq"],
                batch["effector_seq"],
                batch["smiles"],
                rotation_index=global_step,
                trainable_params=trainable_params,
                clip_norm=1.0,
                autocast_enabled=autocast_enabled,
                reg_lambda=lamb,
                mask_fraction=mask_fraction,
                num_mask_blocks=num_mask_blocks,
            )

            pbar.set_postfix({"loss": f"{float(metrics['train/loss_total']):.4f}"})
            wandb.log(metrics, step=global_step)

            should_validate = global_step > 0 and global_step % effective_diag_interval == 0
            if should_validate:
                print("")
                print_train_metrics(global_step, metrics)

                train_subset_metrics = evaluate_dataset(
                    encoder,
                    predictor,
                    regularizer,
                    train_subset_loader,
                    prefix="train_subset",
                    autocast_enabled=autocast_enabled,
                    mask_fraction=mask_fraction,
                    num_mask_blocks=num_mask_blocks,
                )
                val_metrics = evaluate_dataset(
                    encoder,
                    predictor,
                    regularizer,
                    val_loader,
                    prefix="val_set",
                    autocast_enabled=autocast_enabled,
                    mask_fraction=mask_fraction,
                    num_mask_blocks=num_mask_blocks,
                )
                wandb.log({**train_subset_metrics, **val_metrics}, step=global_step)
                print_val_metrics(train_subset_metrics, val_metrics)

                if eval_loader is not None and len(eval_loader) > 0:
                    eval_metrics = evaluate_dataset(
                        encoder,
                        predictor,
                        regularizer,
                        eval_loader,
                        prefix="eval",
                        autocast_enabled=autocast_enabled,
                        mask_fraction=mask_fraction,
                        num_mask_blocks=num_mask_blocks,
                        include_fairness=True,
                    )
                    wandb.log(eval_metrics, step=global_step)
                    print_eval_metrics(eval_metrics)
                encoder.train()
                predictor.train()
                regularizer.train()
                print("=" * 85)

            global_step += 1

        if run_activity_eval and activity_interval_epochs > 0 and ((epoch + 1) % activity_interval_epochs == 0):
            activity_metrics = evaluate_activity_suite(
                activity_csv_path,
                encoder,
                batch_size=min(batch_size, 128),
                autocast_enabled=autocast_enabled,
            )
            if activity_metrics:
                wandb.log(activity_metrics, step=global_step)
            encoder.train()
            predictor.train()
            regularizer.train()

        save_latentglue_checkpoint(
            encoder,
            predictor,
            optimizer,
            scheduler,
            epoch,
            global_step,
            name=f"epoch_{epoch}",
        )

if __name__ == "__main__":
    train(csv_path="data/GlueDegradDB.csv")
