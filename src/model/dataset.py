import os
import random
from collections import Counter
import pandas as pd
import torch

class TernaryDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path=None, split=None, df=None):
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_path)

        if split and df is None and "split" in self.df.columns:
            normalized_split = "validation" if split == "validation" else split
            self.df = self.df[self.df["split"] == normalized_split].reset_index(drop=True)
            if len(self.df) == 0 and split == "validation":
                self.df = pd.read_csv(csv_path)
                self.df = self.df[self.df["split"] == "val"].reset_index(drop=True)

        print(f"Split:{split if split else 'ALL'} ({len(self.df)} entries)")

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

def collate_ternary(batch):
    if not batch:
        return {}
    return {
        "target_seq": [item["target_seq"] for item in batch],
        "effector_seq": [item["effector_seq"] for item in batch],
        "smiles": [item["smiles"] for item in batch],
        "effector_id": [item["effector_id"] for item in batch],
    }

class StratifiedBucketSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, bin_size=512, complex_balance_power=1.0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.balance_power = complex_balance_power

        effector_ids = dataset.effector_bucket_keys
        counts = Counter(effector_ids)
        # Weight samples by 1/(count^power) for tunable stratification
        self.weights = torch.tensor([1.0 / (counts[id] ** self.balance_power) for id in effector_ids], dtype=torch.float)

        self.bins = {}
        for pos_idx, (target_seq, effector_seq, smiles) in enumerate(
            zip(dataset.target_seqs, dataset.effector_seqs, dataset.smiles)
        ):
            bucket_id = (len(target_seq) + len(effector_seq) + len(smiles)) // bin_size
            self.bins.setdefault(bucket_id, []).append(pos_idx)

    def __iter__(self):
        batches = []
        for indices in self.bins.values():
            bin_idx = torch.tensor(indices)
            sampled = bin_idx[torch.multinomial(self.weights[bin_idx], len(indices), replacement=True)].tolist()
            for start in range(0, len(sampled), self.batch_size):
                if start + self.batch_size <= len(sampled):
                    batches.append(sampled[start : start + self.batch_size])

        random.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.bins.values())

def get_dataloader(
    csv_path,
    batch_size=128,
    shuffle=True,
    split=None,
    num_workers=None,
    weighted=True,
    complex_balance_power=1.0,
):
    if num_workers is None:
        num_workers = max(0, os.cpu_count() - 2)

    dataset = TernaryDataset(csv_path, split=split)

    if split == "train" and weighted:
        sampler = StratifiedBucketSampler(
            dataset,
            batch_size=batch_size,
            complex_balance_power=complex_balance_power,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_ternary,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ternary,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=(split == "train"),
    )
