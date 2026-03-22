"""Screen GlueDegradDB-Filter with a context-gated centroid scorer."""

import heapq
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.dataset import collate_ternary
from src.model.model import LatentGlueEncoder
from src.model.train import AUTOCAST_DTYPE
from src.validation.full_eval import (
    LowRankBilinearScorer,
    apply_retrieval_standardization,
    build_retrieval_context_records,
    build_retrieval_pair_dataframe,
    build_retrieval_standardization_stats,
    choose_retrieval_validation_contexts,
)

DEFAULT_DEVICE = "cuda"
DEFAULT_CHECKPOINT_REPO_ID = "ArnavSharma938/LatentGlue"
DEFAULT_CHECKPOINT_FILENAME = "LatentGlue.pt"
DEFAULT_SCREENING_DATASET_REPO_ID = "ArnavSharma938/GlueDegradDB-Filter"
DEFAULT_SCREENING_FILENAME = "GlueDegradDB-Filter.csv"
DEFAULT_SCREENING_LOCAL_CSV = "data/screening/GlueDegradDB-Filter.csv"
DEFAULT_TRAIN_CSV = "data/GlueDegradDB.csv"
DEFAULT_OUTPUT_DIR = "data/results/screen"
DEFAULT_BASIS_PATH = "data/results/screen/context_projection.pt"
LEGACY_BASIS_PATH = "data/results/screen/retrieval_probe.pt"
DEFAULT_PROJECTION_CACHE_ROOT = "data/results/screen/projection_cache"
DEFAULT_TOP_K = 10_000
DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_BATCH_SIZE = 512
DEFAULT_PROJECTION_BATCH_CANDIDATES = (4096, 3072, 2048, 1536, 1024, 768, 512)
DEFAULT_SCORE_BATCH_ROWS = 131_072
DEFAULT_TRAIN_NEGATIVES = 64
DEFAULT_PROJECTION_RANK = 64
DEFAULT_TRAIN_EPOCHS = 250
DEFAULT_PATIENCE = 30
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 17
DEFAULT_MIN_REFERENCE_LIGANDS = 8
SCORE_MODE = "latentglue_context_centroid_v1"
LEGACY_SCORE_MODE = "latentglue_retrieval_probe_v1"
TRAIN_COLUMNS = (
    "Compound ID",
    "SMILES",
    "Target",
    "Effector",
    "Effector UniProt",
    "Target Sequence",
    "Effector Sequence",
    "split",
)
SCREENING_COLUMNS = (
    "id",
    "smiles",
    "mw",
    "hbd",
    "hba",
    "rot_bonds",
    "net_charge",
    "ring_count",
    "fsp3",
    "tpsa",
    "aromatic_rings",
    "total_stereo",
    "undefined_stereo",
    "logp",
)
SCREENING_SCORE_COLUMNS = ("smiles",)
ALPHA_SYNUCLEIN_SEQUENCE = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
KRAS_G12D_SEQUENCE = "MTEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM"
VHL_SEQUENCE = "MPRRAENWDEAEVGAEEAGVEEYGPEEDGGEESGAEESGPEESGPEELGAEEEMEAGRPRPVLRSVNSREPSQVIFCNRSPRVVLPVWLNFDGEPQPYPTLPPGTGRRIHSYRGHLWLFRDAGTHDGLLVNQTELFVPSLNVDGQPIFANITLPVYTLKERCLQVVRSLVKPENYRRLDIVRSLYEDLEDHPNVQKDLERLTQERIAHQRMGD"
CRBN_SEQUENCE = "MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQDSKEAKKPNIINFDTSLPTSHTYLGADMEEFHGRTLHDDDSCQVIPVLPQVMMILIPGQTLPLQLFHPQEVSMVRNLIQKDRTFAVLAYSNVQEREAQFGTTAEIYAYREEQDFGIEIVKVKAIGRQRFKVLELRTQSDGIQQAKVQILPECVLPSTMSAVQLESLNKCQIFPSKPVSREDQCSYKWWQKYQKRKFHCANLTSWPRWLYSLYDAETLMDRIKKQLREWDENLKDDSLPSNPIDFSYRVAACLPIDDVLRIQLLKIGSAIQRLRCELDIMNKCTSLCCKQCQETEITTKNEIFSLSLCGPMAAYVNPHGYVHETLTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICASHIGWKFTATKKDMSPQKFWGLTRSALLPTIPDTEDEISPDKVILCL"


@dataclass(frozen=True)
class ContextSpec:
    name: str
    target_name: str
    effector_name: str
    target_sequence: str
    effector_sequence: str
    reference_effector: str


@dataclass(frozen=True)
class ProjectionBundle:
    scorer: LowRankBilinearScorer
    target_mean: torch.Tensor
    target_std: torch.Tensor
    effector_mean: torch.Tensor
    effector_std: torch.Tensor
    ligand_mean: torch.Tensor
    ligand_std: torch.Tensor
    path: str
    meta: dict


@dataclass(frozen=True)
class ScreenContext:
    spec: ContextSpec
    target_proj: torch.Tensor
    effector_proj: torch.Tensor
    centroid_target: torch.Tensor
    centroid_effector: torch.Tensor
    reference_count: int
    signature: str


class InMemoryTernaryDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.target_seqs = df["Target Sequence"].astype(str).tolist()
        self.effector_seqs = df["Effector Sequence"].astype(str).tolist()
        self.smiles = df["SMILES"].astype(str).str.strip().tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            "target_seq": self.target_seqs[idx],
            "effector_seq": self.effector_seqs[idx],
            "smiles": self.smiles[idx],
            "effector_id": self.effector_seqs[idx],
        }


DEFAULT_CONTEXTS = {
    "alpha": ContextSpec(
        "ALPHA_SYNC_CRBN",
        "Alpha-synuclein",
        "CRBN",
        ALPHA_SYNUCLEIN_SEQUENCE,
        CRBN_SEQUENCE,
        "CRBN",
    ),
    "kras": ContextSpec(
        "KRAS_G12D_VHL",
        "KRAS G12D",
        "VHL",
        KRAS_G12D_SEQUENCE,
        VHL_SEQUENCE,
        "VHL",
    ),
}


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def file_signature(path):
    return {
        "path": os.path.abspath(str(path)),
        "size_bytes": int(os.path.getsize(path)),
        "mtime": safe_float(os.path.getmtime(path), default=0.0),
    }


def signature_token(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def array_signature(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def seed_all(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def clean_screen_chunk(chunk):
    chunk = chunk[chunk["smiles"].notna()].copy()
    chunk["smiles"] = chunk["smiles"].astype(str).str.strip()
    return chunk[chunk["smiles"] != ""].reset_index(drop=True)


def parse_hf_repo_id(value):
    value = str(value).strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc.endswith("huggingface.co"):
        parts = [part for part in parsed.path.split("/") if part]
        return "/".join(parts[:2]) if len(parts) >= 2 else ""
    return value if value.count("/") == 1 and "\\" not in value and not os.path.splitext(value)[1] else ""


def resolve_checkpoint_path(path):
    path = str(path or "").strip()
    if path and os.path.isfile(path):
        return path
    repo_id = parse_hf_repo_id(path)
    if repo_id:
        return hf_hub_download(repo_id=repo_id, filename=DEFAULT_CHECKPOINT_FILENAME)
    if path:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return hf_hub_download(repo_id=DEFAULT_CHECKPOINT_REPO_ID, filename=DEFAULT_CHECKPOINT_FILENAME)


def resolve_screening_csv_path():
    if os.path.isfile(DEFAULT_SCREENING_LOCAL_CSV):
        return DEFAULT_SCREENING_LOCAL_CSV
    return hf_hub_download(
        repo_id=DEFAULT_SCREENING_DATASET_REPO_ID,
        repo_type="dataset",
        filename=DEFAULT_SCREENING_FILENAME,
    )


def read_df(path, required, split=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if split is not None:
        if "split" not in df.columns:
            raise ValueError(f"{path} is missing required column: split")
        df = df[df["split"].astype(str) == str(split)].copy().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{path} is empty.")
    if bool((df["SMILES"].astype(str).str.strip() == "").any()):
        raise ValueError(f"{path} contains blank SMILES.")
    return df


def load_encoder(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = LatentGlueEncoder(device=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    encoder.eval()
    return encoder


def clear_encoder_caches(encoder):
    if hasattr(encoder, "_protein_backbone_cache"):
        encoder._protein_backbone_cache.clear()
    if hasattr(encoder, "_ligand_backbone_cache"):
        encoder._ligand_backbone_cache.clear()


def latent_arrays(df, encoder, batch_size, autocast_enabled):
    loader = DataLoader(
        InMemoryTernaryDataset(df),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_ternary,
    )
    target_reps, effector_reps, ligand_reps = [], [], []
    for batch in loader:
        with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
            prepared = encoder.prepare_inputs(batch["target_seq"], batch["effector_seq"], batch["smiles"])
            target_toks, target_mask, effector_toks, effector_mask, ligand_toks, ligand_mask = prepared
            target_bb = encoder._get_cached_teacher_backbone_batch(0, batch["target_seq"], target_toks, target_mask)
            effector_bb = encoder._get_cached_teacher_backbone_batch(1, batch["effector_seq"], effector_toks, effector_mask)
            ligand_bb = encoder._get_cached_teacher_backbone_batch(2, batch["smiles"], ligand_toks, ligand_mask)
            target_component, target_mask_tr, _ = encoder.forward_component(0, target_toks, target_mask, cached_backbone=target_bb)
            effector_component, effector_mask_tr, _ = encoder.forward_component(1, effector_toks, effector_mask, cached_backbone=effector_bb)
            ligand_component, ligand_mask_tr, _ = encoder.forward_component(2, ligand_toks, ligand_mask, cached_backbone=ligand_bb)
            target_reps.append(encoder.target_pool(target_component, mask=target_mask_tr).detach().float().cpu())
            effector_reps.append(encoder.effector_pool(effector_component, mask=effector_mask_tr).detach().float().cpu())
            ligand_reps.append(encoder.ligand_pool(ligand_component, mask=ligand_mask_tr).detach().float().cpu())
    clear_encoder_caches(encoder)
    return tuple(
        torch.cat(chunks, dim=0).numpy().astype(np.float32)
        for chunks in (target_reps, effector_reps, ligand_reps)
    )


def pair_df(df, negatives, label):
    return build_retrieval_pair_dataframe(
        build_retrieval_context_records(
            df,
            negatives_per_context=int(negatives),
            rng_seed=DEFAULT_SEED,
            dataset_label=label,
        )
    )


def to_std_tensor(array, mean, std, device):
    return torch.from_numpy(apply_retrieval_standardization(array, mean, std)).to(device)


def fit_projection_basis(target, effector, ligand, labels, context_keys, device):
    val_contexts = choose_retrieval_validation_contexts(np.asarray(context_keys, dtype=object), DEFAULT_SEED)
    val_mask = np.isin(context_keys, val_contexts)
    train_mask = ~val_mask
    if not np.any(train_mask) or not np.any(val_mask) or len(np.unique(labels[val_mask])) != 2:
        raise RuntimeError("Projection-basis training did not produce a valid train/validation split.")
    target_mean, target_std = build_retrieval_standardization_stats(target[train_mask])
    effector_mean, effector_std = build_retrieval_standardization_stats(effector[train_mask])
    ligand_mean, ligand_std = build_retrieval_standardization_stats(ligand[train_mask])
    scorer_device = torch.device(device)
    train_t = to_std_tensor(target[train_mask], target_mean, target_std, scorer_device)
    train_e = to_std_tensor(effector[train_mask], effector_mean, effector_std, scorer_device)
    train_l = to_std_tensor(ligand[train_mask], ligand_mean, ligand_std, scorer_device)
    y_train = torch.from_numpy(labels[train_mask].astype(np.float32)).to(scorer_device)
    val_t = to_std_tensor(target[val_mask], target_mean, target_std, scorer_device)
    val_e = to_std_tensor(effector[val_mask], effector_mean, effector_std, scorer_device)
    val_l = to_std_tensor(ligand[val_mask], ligand_mean, ligand_std, scorer_device)
    y_val = labels[val_mask].astype(np.int64)
    seed_all(DEFAULT_SEED)
    scorer = LowRankBilinearScorer(
        target.shape[1],
        effector.shape[1],
        ligand.shape[1],
        DEFAULT_PROJECTION_RANK,
    ).to(scorer_device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    positives = float(max(labels[train_mask].sum(), 1.0))
    negatives = float(max(train_mask.sum() - labels[train_mask].sum(), 1.0))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], device=scorer_device)
    )
    best = {
        "auroc": float("-inf"),
        "auprc": 0.0,
        "state": {key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()},
    }
    patience = 0
    train_batch = min(256, int(train_mask.sum()))
    for _ in range(DEFAULT_TRAIN_EPOCHS):
        scorer.train()
        order = torch.randperm(train_t.size(0), device=scorer_device)
        for start in range(0, train_t.size(0), train_batch):
            idx = order[start : start + train_batch]
            loss = criterion(scorer(train_t[idx], train_e[idx], train_l[idx]), y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scorer.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(scorer(val_t, val_e, val_l)).detach().cpu().numpy()
        auroc = float(roc_auc_score(y_val, val_scores))
        if auroc > best["auroc"] + 1e-4:
            best = {
                "auroc": auroc,
                "auprc": float(average_precision_score(y_val, val_scores)),
                "state": {key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= DEFAULT_PATIENCE:
                break
    scorer.load_state_dict(best["state"])
    scorer.eval()
    meta = {
        "internal_validation_auroc": best["auroc"],
        "internal_validation_auprc": best["auprc"],
        "context_count": int(len(np.unique(context_keys))),
        "pair_count": int(len(labels)),
        "score_mode": SCORE_MODE,
    }
    return ProjectionBundle(
        scorer=scorer,
        target_mean=torch.from_numpy(target_mean).to(scorer_device),
        target_std=torch.from_numpy(target_std).to(scorer_device),
        effector_mean=torch.from_numpy(effector_mean).to(scorer_device),
        effector_std=torch.from_numpy(effector_std).to(scorer_device),
        ligand_mean=torch.from_numpy(ligand_mean).to(scorer_device),
        ligand_std=torch.from_numpy(ligand_std).to(scorer_device),
        path=DEFAULT_BASIS_PATH,
        meta=meta,
    )


def save_projection_bundle(bundle):
    os.makedirs(os.path.dirname(bundle.path), exist_ok=True)
    torch.save(
        {
            "target_dim": int(bundle.target_mean.size(1)),
            "effector_dim": int(bundle.effector_mean.size(1)),
            "ligand_dim": int(bundle.ligand_mean.size(1)),
            "rank": DEFAULT_PROJECTION_RANK,
            "score_mode": SCORE_MODE,
            "target_mean": bundle.target_mean.detach().cpu(),
            "target_std": bundle.target_std.detach().cpu(),
            "effector_mean": bundle.effector_mean.detach().cpu(),
            "effector_std": bundle.effector_std.detach().cpu(),
            "ligand_mean": bundle.ligand_mean.detach().cpu(),
            "ligand_std": bundle.ligand_std.detach().cpu(),
            "scorer_state_dict": {key: value.detach().cpu() for key, value in bundle.scorer.state_dict().items()},
            "meta": dict(bundle.meta),
        },
        bundle.path,
    )


def load_projection_bundle(path, device):
    payload = torch.load(path, map_location=device)
    mode = str(payload.get("score_mode", "")).strip()
    if mode not in {SCORE_MODE, LEGACY_SCORE_MODE}:
        raise ValueError(f"Incompatible basis format: {mode}")
    scorer = LowRankBilinearScorer(
        payload["target_dim"],
        payload["effector_dim"],
        payload["ligand_dim"],
        payload["rank"],
    ).to(device)
    scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
    scorer.eval()
    meta = dict(payload.get("meta", {}))
    if "internal_validation_auroc" not in meta and "validation_auroc" in meta:
        meta["internal_validation_auroc"] = meta["validation_auroc"]
    if "internal_validation_auprc" not in meta and "validation_auprc" in meta:
        meta["internal_validation_auprc"] = meta["validation_auprc"]
    meta["score_mode"] = SCORE_MODE
    return ProjectionBundle(
        scorer=scorer,
        target_mean=payload["target_mean"].to(device),
        target_std=payload["target_std"].to(device),
        effector_mean=payload["effector_mean"].to(device),
        effector_std=payload["effector_std"].to(device),
        ligand_mean=payload["ligand_mean"].to(device),
        ligand_std=payload["ligand_std"].to(device),
        path=path,
        meta=meta,
    )


def migrate_bundle_path(bundle, path):
    return ProjectionBundle(
        scorer=bundle.scorer,
        target_mean=bundle.target_mean,
        target_std=bundle.target_std,
        effector_mean=bundle.effector_mean,
        effector_std=bundle.effector_std,
        ligand_mean=bundle.ligand_mean,
        ligand_std=bundle.ligand_std,
        path=path,
        meta=dict(bundle.meta),
    )


def load_or_train_projection_bundle(encoder, checkpoint_path, train_csv_path, batch_size, autocast_enabled):
    expected = {
        "checkpoint_signature": file_signature(checkpoint_path),
        "train_csv_signature": file_signature(train_csv_path),
    }
    for path in (DEFAULT_BASIS_PATH, LEGACY_BASIS_PATH):
        if not os.path.isfile(path):
            continue
        try:
            bundle = load_projection_bundle(path, encoder.device)
            if any(bundle.meta.get(key) != value for key, value in expected.items()):
                raise ValueError("Cached basis fingerprint mismatch.")
            if path != DEFAULT_BASIS_PATH:
                bundle = migrate_bundle_path(bundle, DEFAULT_BASIS_PATH)
                save_projection_bundle(bundle)
            print(
                f"Loaded context projection basis: {bundle.path} | "
                f"internal AUROC {safe_float(bundle.meta.get('internal_validation_auroc')):.3f} | "
                f"internal AUPRC {safe_float(bundle.meta.get('internal_validation_auprc')):.3f}"
            )
            return bundle
        except Exception as exc:
            print(f"Retraining context projection basis: {exc}")
    train_df = read_df(train_csv_path, TRAIN_COLUMNS, split="train")
    train_pairs = pair_df(train_df, DEFAULT_TRAIN_NEGATIVES, "GlueDegradDB train split")
    print(
        f"Training context projection basis on {len(train_pairs):,} pairs "
        f"from {train_pairs['context_key'].nunique():,} contexts."
    )
    target, effector, ligand = latent_arrays(train_pairs, encoder, batch_size, autocast_enabled)
    bundle = fit_projection_basis(
        target,
        effector,
        ligand,
        train_pairs["label"].to_numpy(dtype=np.int64),
        train_pairs["context_key"].astype(str).to_numpy(),
        device=encoder.device,
    )
    bundle = ProjectionBundle(
        scorer=bundle.scorer,
        target_mean=bundle.target_mean,
        target_std=bundle.target_std,
        effector_mean=bundle.effector_mean,
        effector_std=bundle.effector_std,
        ligand_mean=bundle.ligand_mean,
        ligand_std=bundle.ligand_std,
        path=DEFAULT_BASIS_PATH,
        meta={**bundle.meta, **expected},
    )
    save_projection_bundle(bundle)
    clear_encoder_caches(encoder)
    print(
        f"Saved context projection basis: {bundle.path} | "
        f"internal AUROC {bundle.meta['internal_validation_auroc']:.3f} | "
        f"internal AUPRC {bundle.meta['internal_validation_auprc']:.3f}"
    )
    return bundle


def tokenize_protein(encoder, seqs):
    tokens = encoder.esm_model._tokenize(seqs).to(encoder.device)
    return tokens, tokens != encoder.esm_model.tokenizer.pad_token_id


def standardize_torch(x, mean, std):
    return (x.float() - mean.to(x.device, dtype=torch.float32)) / std.to(x.device, dtype=torch.float32)


@torch.no_grad()
def prepare_context_projections(encoder, contexts, bundle, autocast_enabled):
    prepared = []
    for spec in contexts:
        with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
            target_toks, target_mask = tokenize_protein(encoder, [spec.target_sequence])
            effector_toks, effector_mask = tokenize_protein(encoder, [spec.effector_sequence])
            target_bb = encoder._get_cached_teacher_backbone_batch(0, [spec.target_sequence], target_toks, target_mask)
            effector_bb = encoder._get_cached_teacher_backbone_batch(1, [spec.effector_sequence], effector_toks, effector_mask)
            target_component, target_mask_tr, _ = encoder.forward_component(
                0, target_toks, target_mask, cached_backbone=target_bb
            )
            effector_component, effector_mask_tr, _ = encoder.forward_component(
                1, effector_toks, effector_mask, cached_backbone=effector_bb
            )
            target_repr = standardize_torch(
                encoder.target_pool(target_component, mask=target_mask_tr),
                bundle.target_mean,
                bundle.target_std,
            )
            effector_repr = standardize_torch(
                encoder.effector_pool(effector_component, mask=effector_mask_tr),
                bundle.effector_mean,
                bundle.effector_std,
            )
            target_proj = bundle.scorer.target_proj(target_repr).detach().float().cpu().reshape(-1)
            effector_proj = bundle.scorer.effector_proj(effector_repr).detach().float().cpu().reshape(-1)
        prepared.append(
            ScreenContext(
                spec=spec,
                target_proj=target_proj,
                effector_proj=effector_proj,
                centroid_target=torch.empty(0, dtype=torch.float32),
                centroid_effector=torch.empty(0, dtype=torch.float32),
                reference_count=0,
                signature="",
            )
        )
    clear_encoder_caches(encoder)
    return prepared


@torch.no_grad()
def ligand_projections(encoder, smiles_batch, bundle, autocast_enabled):
    smiles_batch = [str(smiles).strip() for smiles in smiles_batch]
    with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
        mol = encoder.mol_tokenizer(
            smiles_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(encoder.device)
        ligand_component, ligand_mask_tr, _ = encoder.forward_ligand(
            mol["input_ids"], mol["attention_mask"].bool(), role_id=2
        )
        ligand_repr = standardize_torch(
            encoder.ligand_pool(ligand_component, mask=ligand_mask_tr),
            bundle.ligand_mean,
            bundle.ligand_std,
        )
        target_proj = (
            bundle.scorer.ligand_from_target_proj(ligand_repr).detach().float().cpu().numpy().astype(np.float16)
        )
        effector_proj = (
            bundle.scorer.ligand_from_effector_proj(ligand_repr).detach().float().cpu().numpy().astype(np.float16)
        )
    return target_proj, effector_proj


def project_smiles_batches(encoder, smiles_list, bundle, batch_size, autocast_enabled):
    smiles = [str(smiles).strip() for smiles in smiles_list if str(smiles).strip()]
    if not smiles:
        raise ValueError("Reference ligand set is empty after cleaning.")
    target_chunks, effector_chunks = [], []
    for start in range(0, len(smiles), int(batch_size)):
        end = min(start + int(batch_size), len(smiles))
        batch_target, batch_effector = ligand_projections(encoder, smiles[start:end], bundle, autocast_enabled)
        target_chunks.append(batch_target)
        effector_chunks.append(batch_effector)
    clear_encoder_caches(encoder)
    return (
        np.concatenate(target_chunks, axis=0).astype(np.float32, copy=False),
        np.concatenate(effector_chunks, axis=0).astype(np.float32, copy=False),
    )


def l2_normalize_rows(array, eps=1e-12):
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, eps, None)


def build_context_centroid(context, reference_target_proj, reference_effector_proj):
    context_target = context.target_proj.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
    context_effector = context.effector_proj.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
    gated_target = np.asarray(reference_target_proj, dtype=np.float32) * context_target
    gated_effector = np.asarray(reference_effector_proj, dtype=np.float32) * context_effector
    phi = np.concatenate([gated_target, gated_effector], axis=1)
    phi = l2_normalize_rows(phi)
    centroid = phi.mean(axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(centroid))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError(f"Failed to build a valid centroid for {context.spec.name}.")
    centroid = centroid / norm
    rank = int(context.target_proj.numel())
    return centroid[:rank], centroid[rank:]


def build_reference_smiles_by_effector(train_df):
    reference_smiles = {}
    for effector, group in train_df.groupby("Effector", sort=False):
        seen = set()
        smiles_list = []
        for smiles in group["SMILES"].astype(str).str.strip():
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            smiles_list.append(smiles)
        if smiles_list:
            reference_smiles[str(effector)] = smiles_list
    return reference_smiles


def attach_context_centroids(
    encoder,
    prepared_contexts,
    reference_smiles_by_effector,
    bundle,
    batch_size,
    autocast_enabled,
):
    projection_cache = {}
    attached = []
    for context in prepared_contexts:
        effector_key = str(context.spec.reference_effector)
        reference_smiles = list(reference_smiles_by_effector.get(effector_key, []))
        if len(reference_smiles) < DEFAULT_MIN_REFERENCE_LIGANDS:
            raise RuntimeError(
                f"{context.spec.name} requires at least {DEFAULT_MIN_REFERENCE_LIGANDS} "
                f"reference ligands for {effector_key}, found {len(reference_smiles)}."
            )
        if effector_key not in projection_cache:
            projection_cache[effector_key] = project_smiles_batches(
                encoder,
                reference_smiles,
                bundle,
                batch_size,
                autocast_enabled,
            )
        reference_target_proj, reference_effector_proj = projection_cache[effector_key]
        centroid_target, centroid_effector = build_context_centroid(
            context,
            reference_target_proj,
            reference_effector_proj,
        )
        signature = signature_token(
            {
                "context": context.spec.name,
                "reference_effector": effector_key,
                "reference_count": int(len(reference_smiles)),
                "array_signature": array_signature(
                    context.target_proj.detach().cpu().numpy(),
                    context.effector_proj.detach().cpu().numpy(),
                    centroid_target,
                    centroid_effector,
                ),
            }
        )
        attached.append(
            ScreenContext(
                spec=context.spec,
                target_proj=context.target_proj,
                effector_proj=context.effector_proj,
                centroid_target=torch.from_numpy(np.asarray(centroid_target, dtype=np.float32)),
                centroid_effector=torch.from_numpy(np.asarray(centroid_effector, dtype=np.float32)),
                reference_count=int(len(reference_smiles)),
                signature=signature,
            )
        )
    clear_encoder_caches(encoder)
    return attached


def projection_cache_signature(bundle, checkpoint_path, csv_path):
    return {
        "score_mode": SCORE_MODE,
        "basis_signature": file_signature(bundle.path),
        "checkpoint_signature": file_signature(checkpoint_path),
        "screening_csv_signature": file_signature(csv_path),
        "rank": int(DEFAULT_PROJECTION_RANK),
    }


def projection_cache_paths(signature):
    cache_dir = os.path.join(DEFAULT_PROJECTION_CACHE_ROOT, signature_token(signature))
    return cache_dir, os.path.join(cache_dir, "meta.json")


def load_projection_cache_meta(signature):
    cache_dir, meta_path = projection_cache_paths(signature)
    if not os.path.isfile(meta_path):
        return None, cache_dir, meta_path
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("signature") != signature:
        return None, cache_dir, meta_path
    shards = meta.get("shards", [])
    shard_paths_ok = all(
        os.path.isfile(shard["target_path"]) and os.path.isfile(shard["effector_path"]) for shard in shards
    )
    if not shard_paths_ok:
        return None, cache_dir, meta_path
    return meta, cache_dir, meta_path


def save_projection_cache_meta(meta_path, meta):
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


def autotune_projection_batch_size(encoder, bundle, csv_path, autocast_enabled):
    if not str(encoder.device).startswith("cuda"):
        return DEFAULT_BATCH_SIZE
    sample = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=list(SCREENING_SCORE_COLUMNS),
        chunksize=max(DEFAULT_PROJECTION_BATCH_CANDIDATES),
    ):
        cleaned = clean_screen_chunk(chunk)
        if len(cleaned) == 0:
            continue
        sample = cleaned["smiles"].tolist()
        break
    if not sample:
        return DEFAULT_BATCH_SIZE
    for candidate in DEFAULT_PROJECTION_BATCH_CANDIDATES:
        if len(sample) < candidate:
            continue
        try:
            ligand_projections(encoder, sample[:candidate], bundle, autocast_enabled)
            clear_encoder_caches(encoder)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return int(candidate)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            clear_encoder_caches(encoder)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return DEFAULT_BATCH_SIZE


def build_projection_cache(encoder, bundle, checkpoint_path, csv_path, chunk_size, autocast_enabled):
    signature = projection_cache_signature(bundle, checkpoint_path, csv_path)
    meta, cache_dir, meta_path = load_projection_cache_meta(signature)
    if meta and meta.get("complete"):
        print(
            f"Using cached ligand projections: {cache_dir} | "
            f"{int(meta.get('rows_projected', 0)):,} ligands | {len(meta.get('shards', []))} shards"
        )
        return meta
    batch_size = (
        int(meta.get("projection_batch_size", DEFAULT_BATCH_SIZE))
        if meta
        else autotune_projection_batch_size(encoder, bundle, csv_path, autocast_enabled)
    )
    start_chunk = int(meta.get("next_chunk", 1)) if meta else 1
    rows_projected = int(meta.get("rows_projected", 0)) if meta else 0
    shards = list(meta.get("shards", [])) if meta else []
    meta = {
        "signature": signature,
        "projection_batch_size": int(batch_size),
        "chunk_size": int(chunk_size),
        "rows_projected": int(rows_projected),
        "next_chunk": int(start_chunk),
        "complete": False,
        "shards": shards,
    }
    os.makedirs(cache_dir, exist_ok=True)
    reader = pd.read_csv(csv_path, usecols=list(SCREENING_SCORE_COLUMNS), chunksize=chunk_size)
    if start_chunk > 1:
        print(f"Resuming projection cache build from chunk {start_chunk}.")
    for chunk_idx, chunk in enumerate(reader, start=1):
        if chunk_idx < start_chunk:
            continue
        chunk = clean_screen_chunk(chunk)
        if len(chunk) == 0:
            meta["next_chunk"] = int(chunk_idx + 1)
            save_projection_cache_meta(meta_path, meta)
            continue
        target_path = os.path.join(cache_dir, f"ligand_target_proj_{chunk_idx:05d}.npy")
        effector_path = os.path.join(cache_dir, f"ligand_effector_proj_{chunk_idx:05d}.npy")
        target_proj = np.empty((len(chunk), DEFAULT_PROJECTION_RANK), dtype=np.float16)
        effector_proj = np.empty((len(chunk), DEFAULT_PROJECTION_RANK), dtype=np.float16)
        start = 0
        total_batches = (len(chunk) + batch_size - 1) // batch_size
        with tqdm(total=total_batches, desc=f"Projection chunk {chunk_idx}", unit="batch", ncols=100, leave=True) as pbar:
            while start < len(chunk):
                end = min(start + batch_size, len(chunk))
                try:
                    batch_target, batch_effector = ligand_projections(
                        encoder,
                        chunk["smiles"].iloc[start:end].tolist(),
                        bundle,
                        autocast_enabled,
                    )
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or batch_size <= DEFAULT_BATCH_SIZE:
                        raise
                    batch_size = max(DEFAULT_BATCH_SIZE, batch_size // 2)
                    meta["projection_batch_size"] = int(batch_size)
                    clear_encoder_caches(encoder)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    total_batches = (len(chunk) - start + batch_size - 1) // batch_size
                    pbar.reset(total=total_batches)
                    pbar.set_description(f"Projection chunk {chunk_idx} (bs={batch_size})")
                    print(f"Lowered projection batch size to {batch_size} after CUDA OOM.")
                    continue
                target_proj[start:end] = batch_target
                effector_proj[start:end] = batch_effector
                start = end
                pbar.update(1)
                pbar.set_postfix(projected=f"{(rows_projected + start) / 1e6:.2f}M", batch_size=batch_size)
        np.save(target_path, target_proj, allow_pickle=False)
        np.save(effector_path, effector_proj, allow_pickle=False)
        shards.append(
            {
                "chunk_idx": int(chunk_idx),
                "row_start": int(rows_projected),
                "row_count": int(len(chunk)),
                "target_path": os.path.abspath(target_path),
                "effector_path": os.path.abspath(effector_path),
            }
        )
        rows_projected += int(len(chunk))
        meta.update({"rows_projected": int(rows_projected), "next_chunk": int(chunk_idx + 1), "shards": shards})
        save_projection_cache_meta(meta_path, meta)
        clear_encoder_caches(encoder)
        print(
            f"Cached chunk {chunk_idx}: {len(chunk):,} ligands | "
            f"total projected {rows_projected / 1e6:.2f}M"
        )
    meta["complete"] = True
    save_projection_cache_meta(meta_path, meta)
    print(
        f"Built ligand projection cache: {cache_dir} | "
        f"{rows_projected:,} ligands | batch size {batch_size}"
    )
    return meta


def checkpoint_paths(output_dir):
    root = os.path.join(output_dir, "checkpoints")
    return root, os.path.join(root, "meta.json"), os.path.join(root, "heaps.pkl")


def load_screen_state(output_dir, context_names, signature):
    checkpoint_dir, meta_path, heaps_path = checkpoint_paths(output_dir)
    if not (os.path.exists(meta_path) and os.path.exists(heaps_path)):
        return 1, 0, None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        with open(heaps_path, "rb") as handle:
            heaps = pickle.load(handle)
        if meta.get("contexts") != context_names or meta.get("signature") != signature:
            raise ValueError("Checkpoint metadata mismatch.")
        return int(meta["chunk_num"]) + 1, int(meta.get("rows_scored", 0)), heaps
    except Exception as exc:
        print(f"Starting fresh: {exc}")
        return 1, 0, None


def save_screen_state(output_dir, chunk_num, rows_scored, heaps, context_names, signature, top_k, batch_size, chunk_size):
    checkpoint_dir, meta_path, heaps_path = checkpoint_paths(output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "chunk_num": int(chunk_num),
                "rows_scored": int(rows_scored),
                "contexts": list(context_names),
                "signature": dict(signature),
                "top_k": int(top_k),
                "batch_size": int(batch_size),
                "chunk_size": int(chunk_size),
            },
            handle,
        )
    with open(heaps_path, "wb") as handle:
        pickle.dump(heaps, handle)


def clear_screen_state(output_dir):
    checkpoint_dir, meta_path, heaps_path = checkpoint_paths(output_dir)
    for path in (meta_path, heaps_path):
        if os.path.exists(path):
            os.remove(path)
    if os.path.isdir(checkpoint_dir):
        try:
            os.rmdir(checkpoint_dir)
        except OSError:
            pass


def score_centroid_batch(batch_target, batch_effector, target_proj, effector_proj, centroid_target, centroid_effector):
    gated_target = batch_target * target_proj
    gated_effector = batch_effector * effector_proj
    numerator = gated_target.matmul(centroid_target) + gated_effector.matmul(centroid_effector)
    norms = torch.sqrt(
        torch.clamp(gated_target.pow(2).sum(dim=1) + gated_effector.pow(2).sum(dim=1), min=1e-12)
    )
    return numerator / norms


def select_candidate_indices(values, heap, top_k):
    if len(values) == 0:
        return np.asarray([], dtype=np.int64)
    if len(heap) < top_k:
        take = min(top_k - len(heap), len(values))
        candidate_idx = np.arange(len(values), dtype=np.int64) if take >= len(values) else np.argpartition(values, -take)[-take:]
    else:
        candidate_idx = np.flatnonzero(values > heap[0][0])
        if len(candidate_idx) > top_k:
            candidate_idx = candidate_idx[np.argpartition(values[candidate_idx], -top_k)[-top_k:]]
    if len(candidate_idx) > 1:
        candidate_idx = candidate_idx[np.argsort(values[candidate_idx])[::-1]]
    return candidate_idx


def update_heap(heap, score, row_idx, top_k):
    item = (float(score), -int(row_idx))
    if len(heap) < top_k:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def finalize_heap(heap, context):
    ranked = sorted(heap, reverse=True)
    entries, rank, prev = [], 0, None
    for position, (score, neg_row_idx) in enumerate(ranked, start=1):
        if prev is None or score != prev:
            rank = position
        entries.append(
            {
                "row_idx": int(-neg_row_idx),
                "rank": int(rank),
                "context": context.spec.name,
                "target_name": context.spec.target_name,
                "effector_name": context.spec.effector_name,
                "reference_ligand_count": int(context.reference_count),
                "centroid_score": float(score),
            }
        )
        prev = score
    return entries


def enrich_ranked_rows(csv_path, ranked_by_context, chunk_size):
    ordered_row_indices = sorted({entry["row_idx"] for entries in ranked_by_context.values() for entry in entries})
    base_rows = {}
    cursor = 0
    global_row_idx = 0
    reader = pd.read_csv(csv_path, usecols=lambda column: column in SCREENING_COLUMNS, chunksize=chunk_size)
    for chunk in reader:
        chunk = clean_screen_chunk(chunk)
        if len(chunk) == 0:
            continue
        chunk_start = global_row_idx
        chunk_end = chunk_start + len(chunk)
        while cursor < len(ordered_row_indices) and ordered_row_indices[cursor] < chunk_end:
            row_idx = ordered_row_indices[cursor]
            if row_idx >= chunk_start:
                base_rows[row_idx] = chunk.iloc[row_idx - chunk_start].to_dict()
            cursor += 1
        global_row_idx = chunk_end
        if cursor >= len(ordered_row_indices):
            break
    missing = [row_idx for row_idx in ordered_row_indices if row_idx not in base_rows]
    if missing:
        raise RuntimeError(f"Failed to enrich {len(missing)} ranked ligands from {csv_path}.")
    return {
        context_name: pd.DataFrame(
            [{**base_rows[entry["row_idx"]], **{key: value for key, value in entry.items() if key != "row_idx"}} for entry in entries]
        )
        for context_name, entries in ranked_by_context.items()
    }


def screen(bundle, contexts, projection_meta, csv_path, output_dir, top_k):
    signature = {
        "score_mode": SCORE_MODE,
        "projection_cache_signature": projection_meta["signature"],
        "context_signatures": {context.spec.name: context.signature for context in contexts},
        "top_k": int(top_k),
    }
    context_names = [context.spec.name for context in contexts]
    start_chunk, rows_scored, heaps = load_screen_state(output_dir, context_names, signature)
    heaps = heaps or {name: [] for name in context_names}
    scorer_device = bundle.scorer.bias.device
    runtime_contexts = [
        (
            context,
            context.target_proj.to(device=scorer_device, dtype=torch.float32),
            context.effector_proj.to(device=scorer_device, dtype=torch.float32),
            context.centroid_target.to(device=scorer_device, dtype=torch.float32),
            context.centroid_effector.to(device=scorer_device, dtype=torch.float32),
        )
        for context in contexts
    ]
    shards = projection_meta["shards"]
    if start_chunk > 1:
        print(f"Resuming ranked screening from projection shard {start_chunk}.")
    for shard_idx, shard in enumerate(shards, start=1):
        if shard_idx < start_chunk:
            continue
        target_mm = np.load(shard["target_path"], mmap_mode="r")
        effector_mm = np.load(shard["effector_path"], mmap_mode="r")
        row_start = int(shard["row_start"])
        row_count = int(shard["row_count"])
        total_batches = (row_count + DEFAULT_SCORE_BATCH_ROWS - 1) // DEFAULT_SCORE_BATCH_ROWS
        print(
            f"\n[GLOBAL] Scored: {rows_scored / 1e6:.2f}M | Projection shard {shard_idx}: {row_count:,} ligands | "
            + " | ".join(f"{name}: {min(len(heaps[name]), top_k)}" for name in context_names)
        )
        with tqdm(total=total_batches, desc=f"Shard {shard_idx}", unit="batch", ncols=100, leave=True) as pbar:
            for start in range(0, row_count, DEFAULT_SCORE_BATCH_ROWS):
                end = min(start + DEFAULT_SCORE_BATCH_ROWS, row_count)
                batch_target = torch.from_numpy(np.asarray(target_mm[start:end], dtype=np.float32)).to(scorer_device)
                batch_effector = torch.from_numpy(np.asarray(effector_mm[start:end], dtype=np.float32)).to(scorer_device)
                for context, target_proj, effector_proj, centroid_target, centroid_effector in runtime_contexts:
                    context_scores = score_centroid_batch(
                        batch_target,
                        batch_effector,
                        target_proj,
                        effector_proj,
                        centroid_target,
                        centroid_effector,
                    ).detach().cpu().numpy().astype(np.float32)
                    candidate_idx = select_candidate_indices(context_scores, heaps[context.spec.name], top_k)
                    global_row_idx = row_start + start + candidate_idx
                    for row_idx, score in zip(global_row_idx.tolist(), context_scores[candidate_idx].tolist()):
                        update_heap(heaps[context.spec.name], score, row_idx, top_k)
                rows_scored += int(end - start)
                pbar.update(1)
                pbar.set_postfix(
                    scored=f"{rows_scored / 1e6:.2f}M",
                    **{name: min(len(heaps[name]), top_k) for name in context_names},
                )
        save_screen_state(
            output_dir,
            shard_idx,
            rows_scored,
            heaps,
            context_names,
            signature,
            top_k,
            DEFAULT_SCORE_BATCH_ROWS,
            int(projection_meta["chunk_size"]),
        )
    ranked_by_context = {context.spec.name: finalize_heap(heaps[context.spec.name], context) for context in contexts}
    enriched = enrich_ranked_rows(csv_path, ranked_by_context, int(projection_meta["chunk_size"]))
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for context in contexts:
        ranked_df = enriched[context.spec.name]
        path = os.path.join(output_dir, f"{context.spec.name}_top{top_k}.csv")
        ranked_df.to_csv(path, index=False)
        outputs.append((context.spec, path, ranked_df))
    clear_screen_state(output_dir)
    return outputs


def main():
    checkpoint_path = resolve_checkpoint_path(DEFAULT_CHECKPOINT_REPO_ID)
    screening_csv = resolve_screening_csv_path()
    train_df = read_df(DEFAULT_TRAIN_CSV, TRAIN_COLUMNS, split="train")
    encoder = load_encoder(checkpoint_path, device=DEFAULT_DEVICE)
    autocast_enabled = str(DEFAULT_DEVICE).startswith("cuda")
    bundle = load_or_train_projection_bundle(
        encoder,
        checkpoint_path,
        DEFAULT_TRAIN_CSV,
        DEFAULT_BATCH_SIZE,
        autocast_enabled,
    )
    reference_smiles_by_effector = build_reference_smiles_by_effector(train_df)
    prepared_contexts = prepare_context_projections(
        encoder,
        [DEFAULT_CONTEXTS["alpha"], DEFAULT_CONTEXTS["kras"]],
        bundle,
        autocast_enabled,
    )
    contexts = attach_context_centroids(
        encoder,
        prepared_contexts,
        reference_smiles_by_effector,
        bundle,
        DEFAULT_BATCH_SIZE,
        autocast_enabled,
    )
    projection_meta = build_projection_cache(
        encoder,
        bundle,
        checkpoint_path,
        screening_csv,
        DEFAULT_CHUNK_SIZE,
        autocast_enabled,
    )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Screening CSV: {screening_csv}")
    print(f"Context projection basis: {bundle.path}")
    print(
        f"Projection cache: {os.path.dirname(projection_meta['shards'][0]['target_path']) if projection_meta['shards'] else DEFAULT_PROJECTION_CACHE_ROOT} | "
        f"{int(projection_meta['rows_projected']):,} ligands"
    )
    print(
        "Basis diagnostics: "
        f"internal AUROC {safe_float(bundle.meta.get('internal_validation_auroc')):.3f} | "
        f"internal AUPRC {safe_float(bundle.meta.get('internal_validation_auprc')):.3f}"
    )
    print(
        "Reference centroids: "
        + " | ".join(
            f"{context.spec.name}: {context.reference_count} {context.spec.reference_effector} ligands"
            for context in contexts
        )
    )
    for context, path, ranked in screen(
        bundle,
        contexts,
        projection_meta,
        csv_path=screening_csv,
        output_dir=DEFAULT_OUTPUT_DIR,
        top_k=DEFAULT_TOP_K,
    ):
        print(f"Saved {len(ranked)} ranked ligands for {context.name} -> {path}")


if __name__ == "__main__":
    main()
