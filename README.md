<h1 align="center">Learning Transferable Representations of Molecular Glues in Latent Space</h1>

<div align="center">

[![Training Set](https://img.shields.io/badge/HuggingFace-Training%20Set-C2410C?style=flat)](https://huggingface.co/datasets/ArnavSharma938/GlueDegradDB)
[![Activity Set](https://img.shields.io/badge/HuggingFace-Activity%20Set-B45309?style=flat)](https://huggingface.co/datasets/ArnavSharma938/GlueDegradDB-Activity)
[![Open License](https://img.shields.io/badge/Open%20License-MIT-7F1D1D?style=flat)](LICENSE)
[![Open Weights](https://img.shields.io/badge/HuggingFace-Open%20Weights-4338CA?style=flat)](https://huggingface.co/ArnavSharma938/LatentGlue)

</div>

> [!NOTE]
> **MIT License:**
> LatentGlue is freely available for **any use** under the MIT License, contributions are always welcome!<br><br>
> For any inquiries related to this project, contact **Arnav Sharma** at [arnavsharma.0914@gmail.com](mailto:arnavsharma.0914@gmail.com).

## Overview
LatentGlue is a 635 million-parameter self-supervised representation learning model for molecular glues. It uses frozen ESM-C protein features and frozen MoLFormer-XL ligand features, projects them into a shared 768-dimensional latent space, summarizes each component with seed-attention pooling, and is trained with masked latent reconstruction over target-effector-ligand ternaries in a concatenated complex.

LatentGlue achieves RMSE 0.575 and Spearman 0.469, versus 0.693 / 0.359 for the Frozen baseline and 0.730 / 0.334 for the Projected baseline, a **17% RMSE reduction and 31% Spearman improvement** over Frozen, and **21% / 40%** over Projected. Gains are consistent across all three random seeds and both target–effector complexes (CDK2–CRBN and WIZ–CRBN), indicating that the latent structure learned during pre-training transfers to activity ranking even with a lightweight linear probe (single-digit kilobyte size). These gains correlate with effective dimensionality: frozen ESM-C protein features are severely collapsed (~200/768 effective dims, 26% utilization), while LatentGlue expands them ~3× to ~600/768 — aligning with the already well-spread MoLFormer-XL ligand features (~641/768 frozen, ~649/768 LatentGlue). A richer protein embedding is evidently necessary to discriminate fine-grained glue activity. Complementing this, LatentGlue attention is highly sparse: the top-3 ligand atoms capture on average **95% of attention weight** out of a median of 29 heavy atoms (~10% effective atom fraction), and the top-10 protein residues capture **80–90% of attention** out of hundreds to thousands of residues (~3% effective residue fraction). This sparsity is consistent with molecular glue biology, where a compact pharmacophore and a small binding interface drive ternary complex formation.

**Training to the released checkpoint (epoch 4) on a 4 vCPU, 32 GB RAM, 1× A100 80GB [Thunder Compute](https://www.thundercompute.com/) instance took under 60 minutes ($0.78). Open weights are available on [HuggingFace](https://huggingface.co/ArnavSharma938/LatentGlue).**

## Case Study

The case study applies LatentGlue to large-scale molecular glue screening, with the implementation in `src/casestudy/inference.py`. Starting from 104 million commercially accessible compounds from the [Enamine REAL database](https://enamine.net/compound-collections/real-compounds/real-database-subsets), the pipeline first performs chemistry-based filtering to a set of ~35 million compounds and then uses LatentGlue to screen and rank the top 10,000 candidate ligands for the Alpha-synuclein wild-type and KRAS G12D target contexts. This study is more experimental for now as the screening pipeline does not implement hard chemistry filters.

* **Top Ligand for A-Syn (Rank 2)**: COC(C)C(C)n1cc(COc2cccc(C(=O)NC3CCc4ccccc4NC3=O)c2)nn1
* **Top Ligand for KRAS G12D (Rank 1)**: CNC(=O)C1CC(O)CN1C(=O)CC(NC(=O)c1csnc1C)c1ccc(-c2ccccc2)cc1

High-ranked ligands have extremely strong chemistry fundamentals with major failures requiring minor chemistry edits.

## Data
Available datasets include **[GlueDegradDB](https://huggingface.co/datasets/ArnavSharma938/GlueDegradDB)**, the training dataset; **[GlueDegradDB-Eval](https://huggingface.co/datasets/ArnavSharma938/GlueDegradDB-Eval)**, an evaluation set (separate from the validation split within GlueDegradDB, which has no component overlap); **[GlueDegradDB-Activity](https://huggingface.co/datasets/ArnavSharma938/GlueDegradDB-Activity)**, degradation profiles; and **[GlueDegradDB-Filter](https://huggingface.co/datasets/ArnavSharma938/GlueDegradDB-Filter)**, a 35M-molecule molecular glue degrader candidate set.

## Setup
Clone the repository

```powershell
git clone https://github.com/ArnavSharma938/LatentGlue.git
```

Create a virtual environment

```powershell
uv venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate # Linux/macOS
```

Install dependencies

```powershell
uv pip install -r requirements.txt
```

Start training

```powershell
python scripts/run_train.py
```

> [!TIP]
> RDKit can be unstable on certain Linux distributions; accordingly, `scripts/run_processing.py` may fail in those environments. The full outputs from a verified run are provided directly in the `data/` directory and HuggingFace for immediate use.

## Citation
If you find this repository useful, please cite it as software while the paper is in peer review:

```bibtex
@software{sharma2026latentglue,
  author = {Sharma, Arnav},
  title = {Learning Transferable Representations of Molecular Glues in Latent Space},
  year = {2026},
  url = {https://github.com/ArnavSharma938/LatentGlue},
}
```
