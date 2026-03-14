"""
1. Full-model screening path

   - for screening speed, it does not call `run_masked_reconstruction(...)`
     directly inside the hot loop; instead it reproduces the exact
     ligand-masked path with fixed-context precomputation
   - this is a screening script for the full checkpoint, not an encoder-only
     embedding export

2. Ranking rationale
   - the screened library in `data/screening/GlueDegradDB-Filter.csv` contains
     ligands only, so each candidate is evaluated under a fixed target /
     effector context injected by this script
   - the primary score is `ligand_score = -ligand_masked_mse`
   - this follows the training objective directly: lower ligand reconstruction
     MSE means the model finds that ternary more internally compatible
   - lower MSE is a model ranking signal, not a direct wet-lab guarantee
"""

import heapq
import json
import os
import pickle
from dataclasses import dataclass
from itertools import count
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from src.model.model import LatentGlueEncoder, RelationalPredictor
from src.model.train import (
    AUTOCAST_DTYPE,
    MASK_BLOCKS,
    MASK_FRACTION,
    apply_raw_mask_tokens,
    build_mask_generator,
    expand_truncated_drop_mask_to_raw,
    generate_contiguous_block_mask,
    run_predictor_for_masked_component,
)
from src.validation.in_train_eval import masked_token_stats

DEFAULT_DEVICE = "cuda"
DEFAULT_CHECKPOINT_REPO_ID = "ArnavSharma938/LatentGlue"
DEFAULT_CHECKPOINT_FILENAME = "LatentGlue.pt"
DEFAULT_SCREENING_DATASET_REPO_ID = "ArnavSharma938/GlueDegradDB-Filter"
DEFAULT_SCREENING_FILENAME = "GlueDegradDB-Filter.csv"
DEFAULT_OUTPUT_DIR = "data/results/screen"
DEFAULT_TOP_K = 10_000
DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_BATCH_SIZE = 512
DEFAULT_MASK_SEED = 50_000
DEFAULT_MAX_SCREENED_LIGANDS = 5_000_000
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

ALPHA_SYNUCLEIN_SEQUENCE = (
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
)
KRAS_G12D_SEQUENCE = (
    "MTEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM"
)
VHL_SEQUENCE = (
    "MPRRAENWDEAEVGAEEAGVEEYGPEEDGGEESGAEESGPEESGPEELGAEEEMEAGRPRPVLRSVNSREPSQVIFCNRSPRVVLPVWLNFDGEPQPYPTLPPGTGRRIHSYRGHLWLFRDAGTHDGLLVNQTELFVPSLNVDGQPIFANITLPVYTLKERCLQVVRSLVKPENYRRLDIVRSLYEDLEDHPNVQKDLERLTQERIAHQRMGD"
)
CRBN_SEQUENCE = (
    "MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQDSKEAKKPNIINFDTSLPTSHTYLGADMEEFHGRTLHDDDSCQVIPVLPQVMMILIPGQTLPLQLFHPQEVSMVRNLIQKDRTFAVLAYSNVQEREAQFGTTAEIYAYREEQDFGIEIVKVKAIGRQRFKVLELRTQSDGIQQAKVQILPECVLPSTMSAVQLESLNKCQIFPSKPVSREDQCSYKWWQKYQKRKFHCANLTSWPRWLYSLYDAETLMDRIKKQLREWDENLKDDSLPSNPIDFSYRVAACLPIDDVLRIQLLKIGSAIQRLRCELDIMNKCTSLCCKQCQETEITTKNEIFSLSLCGPMAAYVNPHGYVHETLTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICASHIGWKFTATKKDMSPQKFWGLTRSALLPTIPDTEDEISPDKVILCL"
)

@dataclass(frozen=True)
class ContextSpec:
    name: str
    target_name: str
    effector_name: str
    target_sequence: str
    effector_sequence: str

@dataclass(frozen=True)
class PreparedContext:
    spec: ContextSpec
    target_component: torch.Tensor
    target_mask_tr: torch.Tensor
    effector_component: torch.Tensor
    effector_mask_tr: torch.Tensor

@dataclass(frozen=True)
class PreparedLigandBatch:
    smiles: tuple[str, ...]
    ligand_tokens: torch.Tensor
    ligand_mask_raw: torch.Tensor
    teacher_component: torch.Tensor
    teacher_mask_tr: torch.Tensor

DEFAULT_CONTEXTS = {
    "alpha": ContextSpec(
        name="ALPHA_SYNC_CRBN",
        target_name="Alpha-synuclein",
        effector_name="CRBN",
        target_sequence=ALPHA_SYNUCLEIN_SEQUENCE,
        effector_sequence=CRBN_SEQUENCE,
    ),
    "kras": ContextSpec(
        name="KRAS_G12D_VHL",
        target_name="KRAS G12D",
        effector_name="VHL",
        target_sequence=KRAS_G12D_SEQUENCE,
        effector_sequence=VHL_SEQUENCE,
    ),
}

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

def resolve_screening_csv_path():
    return hf_hub_download(
        repo_id=DEFAULT_SCREENING_DATASET_REPO_ID,
        repo_type="dataset",
        filename=DEFAULT_SCREENING_FILENAME,
    )

def load_latentglue_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = LatentGlueEncoder(device=device)
    predictor = RelationalPredictor().to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    predictor.load_state_dict(checkpoint["predictor_state_dict"], strict=True)
    encoder.eval()
    predictor.eval()
    return encoder, predictor

def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def should_keep_score(heap, ligand_score, top_k):
    if len(heap) < top_k:
        return True
    return float(ligand_score) > float(heap[0][0])

def update_top_k(heap, record, ligand_score, top_k, screen_order):
    entry = (float(ligand_score), -int(screen_order), record)
    if len(heap) < top_k:
        heapq.heappush(heap, entry)
        return
    if float(ligand_score) > float(heap[0][0]):
        heapq.heapreplace(heap, entry)

@torch.no_grad()
def prepare_protein_inputs(encoder, sequences):
    tokens = encoder.esm_model._tokenize(sequences).to(encoder.device)
    mask = tokens != encoder.esm_model.tokenizer.pad_token_id
    return tokens, mask

@torch.no_grad()
def prepare_ligand_inputs(encoder, smiles_batch):
    mol_inputs = encoder.mol_tokenizer(
        smiles_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(encoder.device)
    return mol_inputs["input_ids"], mol_inputs["attention_mask"].bool()

@torch.no_grad()
def prepare_fixed_context(encoder, context, autocast_enabled):
    with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
        target_toks, target_mask_raw = prepare_protein_inputs(encoder, [context.target_sequence])
        effector_toks, effector_mask_raw = prepare_protein_inputs(encoder, [context.effector_sequence])
        target_backbone = encoder._get_cached_teacher_backbone_batch(
            0,
            [context.target_sequence],
            target_toks,
            target_mask_raw,
        )
        effector_backbone = encoder._get_cached_teacher_backbone_batch(
            1,
            [context.effector_sequence],
            effector_toks,
            effector_mask_raw,
        )
        target_component, target_mask_tr, _ = encoder.forward_component(
            0,
            target_toks,
            target_mask_raw,
            cached_backbone=target_backbone,
        )
        effector_component, effector_mask_tr, _ = encoder.forward_component(
            1,
            effector_toks,
            effector_mask_raw,
            cached_backbone=effector_backbone,
        )
    return PreparedContext(
        spec=context,
        target_component=target_component.detach(),
        target_mask_tr=target_mask_tr.detach(),
        effector_component=effector_component.detach(),
        effector_mask_tr=effector_mask_tr.detach(),
    )

@torch.no_grad()
def prepare_ligand_batch(encoder, smiles_batch, autocast_enabled):
    smiles_batch = tuple(smiles_batch)
    ligand_toks, ligand_mask_raw = prepare_ligand_inputs(encoder, list(smiles_batch))
    return encode_ligand_teacher_batch(
        encoder,
        smiles_batch,
        ligand_toks,
        ligand_mask_raw,
        autocast_enabled=autocast_enabled,
    )

@torch.no_grad()
def encode_ligand_teacher_batch(encoder, smiles_batch, ligand_toks, ligand_mask_raw, autocast_enabled):
    with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
        ligand_backbone = encoder._get_cached_teacher_backbone_batch(
            2,
            smiles_batch,
            ligand_toks,
            ligand_mask_raw,
        )
        teacher_component, teacher_mask_tr, _ = encoder.forward_component(
            2,
            ligand_toks,
            ligand_mask_raw,
            cached_backbone=ligand_backbone,
        )
    return PreparedLigandBatch(
        smiles=smiles_batch,
        ligand_tokens=ligand_toks,
        ligand_mask_raw=ligand_mask_raw,
        teacher_component=teacher_component.detach(),
        teacher_mask_tr=teacher_mask_tr.detach(),
    )

@torch.no_grad()
def score_prepared_context(
    encoder,
    predictor,
    prepared_context,
    prepared_ligands,
    autocast_enabled,
    mask_seed_base,
    mask_fraction=MASK_FRACTION,
    num_mask_blocks=MASK_BLOCKS,
):
    batch_size = len(prepared_ligands.smiles)
    with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=AUTOCAST_DTYPE):
        teacher_component = prepared_ligands.teacher_component
        component_valid_mask = prepared_ligands.teacher_mask_tr
        generator = build_mask_generator(mask_seed_base)
        drop_mask_tr, _masked_counts, _valid_counts = generate_contiguous_block_mask(
            component_valid_mask,
            generator=generator,
            mask_fraction=mask_fraction,
            num_blocks=num_mask_blocks,
        )
        raw_drop_mask = expand_truncated_drop_mask_to_raw(prepared_ligands.ligand_mask_raw, drop_mask_tr)
        masked_ligand_tokens = apply_raw_mask_tokens(
            prepared_ligands.ligand_tokens,
            raw_drop_mask,
            mask_token_id=encoder.get_component_mask_token_id(2),
        )
        student_ligand, student_ligand_mask, _ = encoder.forward_component(
            2,
            masked_ligand_tokens,
            prepared_ligands.ligand_mask_raw,
        )

        target_component = prepared_context.target_component.expand(batch_size, -1, -1).detach()
        effector_component = prepared_context.effector_component.expand(batch_size, -1, -1).detach()
        target_mask = prepared_context.target_mask_tr.expand(batch_size, -1)
        effector_mask = prepared_context.effector_mask_tr.expand(batch_size, -1)
        masked_student_ligand = student_ligand.clone()
        mask_embeddings = predictor.get_mask_embedding(
            2,
            batch_size=batch_size,
            device=masked_student_ligand.device,
            dtype=masked_student_ligand.dtype,
        ).expand(-1, masked_student_ligand.size(1), -1)
        masked_student_ligand[drop_mask_tr] = mask_embeddings[drop_mask_tr]
        predicted_component = run_predictor_for_masked_component(
            predictor,
            (target_component, effector_component, masked_student_ligand),
            (target_mask, effector_mask, student_ligand_mask),
            masked_component=2,
            drop_mask_tr=drop_mask_tr,
        )
        masked_stats = masked_token_stats(predicted_component, teacher_component.detach(), drop_mask_tr)
    ligand_mse = masked_stats["example_mse"].detach().float().cpu().numpy().astype(np.float32)
    return {
        "ligand_mse": ligand_mse,
        "ligand_score": -ligand_mse,
    }

def clear_ligand_cache(encoder):
    if hasattr(encoder, "_ligand_backbone_cache"):
        encoder._ligand_backbone_cache.clear()

def build_output_record(context, row, ligand_mse, ligand_score):
    record = row._asdict()
    record.update(
        {
            "context": context.name,
            "target_name": context.target_name,
            "effector_name": context.effector_name,
            "ligand_mse": safe_float(ligand_mse),
            "ligand_score": safe_float(ligand_score),
        }
    )
    return record

def finalize_top_k(heap):
    ranked_entries = sorted(heap, key=lambda entry: (entry[0], entry[1]), reverse=True)
    ranked_records = []
    current_rank = 0
    previous_score = None
    for position, (ligand_score, _neg_order, record) in enumerate(ranked_entries, start=1):
        if previous_score is None or ligand_score != previous_score:
            current_rank = position
        record["rank"] = current_rank
        ranked_records.append(record)
        previous_score = ligand_score
    return pd.DataFrame(ranked_records)

def screening_usecols(column_name):
    return column_name in SCREENING_COLUMNS

def get_checkpoint_paths(output_dir):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    return (
        checkpoint_dir,
        os.path.join(checkpoint_dir, "meta.json"),
        os.path.join(checkpoint_dir, "heaps.pkl"),
    )

def kept_counts_text(contexts, heaps, top_k):
    return " | ".join(
        f"{context.name}: {min(len(heaps[context.name]), top_k)}"
        for context in contexts
    )

def kept_counts_dict(contexts, heaps, top_k):
    return {
        context.name: min(len(heaps[context.name]), top_k)
        for context in contexts
    }

def save_screening_checkpoint(
    output_dir,
    chunk_num,
    rows_scored,
    heaps,
    contexts,
    top_k,
    batch_size,
    chunk_size,
    max_screened_ligands,
):
    checkpoint_dir, checkpoint_meta, checkpoint_heaps = get_checkpoint_paths(output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(checkpoint_meta, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "chunk_num": int(chunk_num),
                "rows_scored": int(rows_scored),
                "contexts": [context.name for context in contexts],
                "top_k": int(top_k),
                "batch_size": int(batch_size),
                "chunk_size": int(chunk_size),
                "max_screened_ligands": int(max_screened_ligands),
            },
            handle,
        )
    with open(checkpoint_heaps, "wb") as handle:
        pickle.dump(heaps, handle)

def load_screening_checkpoint(output_dir, contexts):
    checkpoint_dir, checkpoint_meta, checkpoint_heaps = get_checkpoint_paths(output_dir)
    if not (os.path.exists(checkpoint_meta) and os.path.exists(checkpoint_heaps)):
        return 1, 0, None, 0
    try:
        with open(checkpoint_meta, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        with open(checkpoint_heaps, "rb") as handle:
            heaps = pickle.load(handle)
        expected_contexts = [context.name for context in contexts]
        if meta.get("contexts") != expected_contexts:
            raise ValueError(
                f"Checkpoint contexts {meta.get('contexts')} do not match current contexts {expected_contexts}."
            )
        return (
            int(meta["chunk_num"]) + 1,
            int(meta.get("rows_scored", 0)),
            heaps,
            int(meta.get("max_screened_ligands", 0)),
        )
    except Exception as exc:
        print(f"Failed to load screening checkpoint, starting fresh: {exc}")
        return 1, 0, None, 0

def clear_screening_checkpoint(output_dir):
    checkpoint_dir, checkpoint_meta, checkpoint_heaps = get_checkpoint_paths(output_dir)
    for path in (checkpoint_meta, checkpoint_heaps):
        if os.path.exists(path):
            os.remove(path)
    if os.path.isdir(checkpoint_dir):
        try:
            os.rmdir(checkpoint_dir)
        except OSError:
            pass

def screen_contexts(
    prepared_contexts,
    encoder,
    predictor,
    csv_path,
    output_dir,
    batch_size,
    chunk_size,
    top_k,
    autocast_enabled,
    mask_seed,
    max_screened_ligands=DEFAULT_MAX_SCREENED_LIGANDS,
):
    context_specs = [prepared_context.spec for prepared_context in prepared_contexts]
    start_chunk, rows_scored, saved_heaps, checkpoint_max_screened_ligands = load_screening_checkpoint(output_dir, context_specs)
    if checkpoint_max_screened_ligands and checkpoint_max_screened_ligands != int(max_screened_ligands):
        print(
            f"Checkpoint ligand cap {checkpoint_max_screened_ligands} does not match current cap "
            f"{int(max_screened_ligands)}. Starting fresh."
        )
        start_chunk, rows_scored, saved_heaps = 1, 0, None
    heaps = saved_heaps if saved_heaps is not None else {context.name: [] for context in context_specs}
    screen_order = count(rows_scored)
    reader = pd.read_csv(csv_path, chunksize=chunk_size, usecols=screening_usecols)
    if start_chunk > 1:
        print(f"Resuming from Chunk {start_chunk}...")

    for chunk_idx, chunk in enumerate(reader, start=1):
        if rows_scored >= max_screened_ligands:
            print(f"Reached screening cap of {max_screened_ligands:,} ligands.")
            break
        if chunk_idx < start_chunk:
            continue
        if "smiles" not in chunk.columns:
            raise ValueError(f"{csv_path} must contain a 'smiles' column.")
        chunk = chunk[chunk["smiles"].notna()].copy()
        chunk["smiles"] = chunk["smiles"].astype(str).str.strip()
        chunk = chunk[chunk["smiles"] != ""].reset_index(drop=True)
        remaining = max_screened_ligands - rows_scored
        if remaining <= 0:
            print(f"Reached screening cap of {max_screened_ligands:,} ligands.")
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].reset_index(drop=True)
        if len(chunk) == 0:
            save_screening_checkpoint(
                output_dir,
                chunk_idx,
                rows_scored,
                heaps,
                context_specs,
                top_k,
                batch_size,
                chunk_size,
                max_screened_ligands,
            )
            continue

        total_batches = (len(chunk) + batch_size - 1) // batch_size
        print(
            f"\n[GLOBAL] Scored: {rows_scored/1e6:.2f}M | "
            f"Chunk {chunk_idx}: {len(chunk):,} ligands | "
            f"Kept -> {kept_counts_text(context_specs, heaps, top_k)}"
        )

        with tqdm(total=total_batches, desc=f"Chunk {chunk_idx}", unit="batch", ncols=100, leave=True) as pbar:
            for batch_idx, start in enumerate(range(0, len(chunk), batch_size), start=1):
                batch_df = chunk.iloc[start : start + batch_size].reset_index(drop=True)
                smiles_batch = batch_df["smiles"].tolist()
                batch_rows = list(batch_df.itertuples(index=False, name="ScreeningRow"))
                batch_orders = [next(screen_order) for _ in batch_rows]
                batch_seed = mask_seed + (chunk_idx - 1) * chunk_size + start
                ligand_toks, ligand_mask_raw = prepare_ligand_inputs(encoder, smiles_batch)
                prepared_ligands = encode_ligand_teacher_batch(
                    encoder,
                    tuple(smiles_batch),
                    ligand_toks,
                    ligand_mask_raw,
                    autocast_enabled=autocast_enabled,
                )

                for context_idx, prepared_context in enumerate(prepared_contexts):
                    context = prepared_context.spec
                    scores = score_prepared_context(
                        encoder,
                        predictor,
                        prepared_context,
                        prepared_ligands,
                        autocast_enabled=autocast_enabled,
                        mask_seed_base=batch_seed + context_idx,
                    )
                    heap = heaps[context.name]
                    for row, order, ligand_mse, ligand_score in zip(
                        batch_rows,
                        batch_orders,
                        scores["ligand_mse"],
                        scores["ligand_score"],
                    ):
                        if not should_keep_score(heap, ligand_score, top_k):
                            continue
                        record = build_output_record(context, row, ligand_mse, ligand_score)
                        update_top_k(heap, record, ligand_score, top_k, order)
                clear_ligand_cache(encoder)
                rows_scored += len(batch_rows)
                pbar.update(1)
                pbar.set_postfix(scored=f"{rows_scored/1e6:.2f}M", **kept_counts_dict(context_specs, heaps, top_k))

        save_screening_checkpoint(
            output_dir,
            chunk_idx,
            rows_scored,
            heaps,
            context_specs,
            top_k,
            batch_size,
            chunk_size,
            max_screened_ligands,
        )

    os.makedirs(output_dir, exist_ok=True)
    results = []
    for context in context_specs:
        ranked_df = finalize_top_k(heaps[context.name])
        output_path = os.path.join(output_dir, f"{context.name}_top{top_k}.csv")
        ranked_df.to_csv(output_path, index=False)
        results.append((context, output_path, ranked_df))
    clear_screening_checkpoint(output_dir)
    return results

def main():
    checkpoint = DEFAULT_CHECKPOINT_REPO_ID
    csv_path = resolve_screening_csv_path()
    output_dir = DEFAULT_OUTPUT_DIR
    device = DEFAULT_DEVICE
    batch_size = DEFAULT_BATCH_SIZE
    chunk_size = DEFAULT_CHUNK_SIZE
    top_k = DEFAULT_TOP_K
    mask_seed = DEFAULT_MASK_SEED
    max_screened_ligands = DEFAULT_MAX_SCREENED_LIGANDS
    context_keys = ("alpha", "kras")
    contexts = [DEFAULT_CONTEXTS[key] for key in context_keys]
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    encoder, predictor = load_latentglue_models(checkpoint_path, device=device)
    autocast_enabled = str(device).startswith("cuda")
    prepared_contexts = [prepare_fixed_context(encoder, context, autocast_enabled=autocast_enabled) for context in contexts]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Screening CSV: {csv_path}")
    print(f"Output dir: {output_dir}")
    print(f"Contexts: {', '.join(context.name for context in contexts)}")
    print(f"Top K per context: {top_k}")
    print(f"Max Screened Ligands: {max_screened_ligands:,}")
    print(f"Batch Size: {batch_size} | Chunk Size: {chunk_size} | Mask Seed: {mask_seed}")

    for context, output_path, ranked_df in screen_contexts(
        prepared_contexts,
        encoder,
        predictor,
        csv_path=csv_path,
        output_dir=output_dir,
        batch_size=batch_size,
        chunk_size=chunk_size,
        top_k=top_k,
        autocast_enabled=autocast_enabled,
        mask_seed=mask_seed,
        max_screened_ligands=max_screened_ligands,
    ):
        print(f"Saved {len(ranked_df)} ranked ligands for {context.name} -> {output_path}")

if __name__ == "__main__":
    main()
