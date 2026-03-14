# MolSSD Project Context

## Project Overview
- Molecular Scale Space Diffusion (MolSSD): first application of scale-space diffusion theory to molecular generation
- Application: novel UV-absorbing molecular materials for protective coatings
- Target venue: Nature
- Two-phase strategy: Phase A (beat ML baselines) → Phase B (UV-absorber discovery)

## Repository Structure
```
scale_diffusion_mols/
├── CLAUDE.md                    # This file
├── ROADMAP.txt                  # Step-by-step actionable roadmap
├── references.bib               # Shared BibTeX references
├── papers/                      # Source papers (PDFs)
│   ├── 2603.08709v1.pdf         # SSD paper (core math framework)
│   ├── 2212.09748v2.pdf         # DiT paper (transformer backbone)
│   ├── s41598-025-96185-2 (1).pdf # MSADN paper (multi-scale adversarial)
│   └── 2505.08762v2.pdf         # OMol25 paper (dataset + eSEN)
├── paper_draft/main.tex         # Nature-style manuscript (placeholder tables to fill)
├── research_proposal/proposal.tex
├── technical_whitepaper/whitepaper.tex
├── theory_explainer/
│   └── what_we_created.tex      # 2000+ line document: all 7 contributions
├── cloud_training_plan/
│   └── cloud_training_plan.tex  # Cloud pre-training strategy + cost analysis
├── molssd/                      # Core Python package
│   ├── core/
│   │   ├── coarsening.py        # Graph Laplacian, spectral clustering, coarsening matrix
│   │   ├── diffusion.py         # MolSSDDiffusion: forward/reverse process
│   │   ├── degradation.py       # DegradationOperator: M_t and M_{1:t}
│   │   ├── lanczos.py           # Lanczos tridiag, posterior eigendecomp, non-iso sampling
│   │   └── noise_schedules.py   # CosineSchedule, LinearSchedule, ResolutionSchedule
│   ├── models/
│   │   ├── egnn.py              # E(n)-equivariant GNN blocks (building block)
│   │   ├── embeddings.py        # Time, resolution, atom type embeddings
│   │   ├── flexi_net.py         # Molecular Flexi-Net (hierarchical multi-resolution)
│   │   └── conditioning.py      # Property/spectrum conditioning + classifier-free guidance
│   ├── data/
│   │   ├── qm9_loader.py        # QM9 dataset (130K mols, ≤29 atoms, 5 types)
│   │   ├── geom_drugs_loader.py # GEOM-Drugs dataset (304K mols, ≤181 atoms, 10 types)
│   │   └── omol25_loader.py     # OMol25 dataset (83M mols, ≤350 atoms, 16 types)
│   ├── training/
│   │   ├── trainer.py           # MolSSDTrainer: full training loop
│   │   ├── losses.py            # Min-SNR-5 weighted MSE + type cross-entropy
│   │   ├── ema.py               # Exponential moving average of model weights
│   │   └── optimizers.py        # AdamW + warmup/cosine decay scheduler
│   ├── evaluation/
│   │   ├── metrics.py           # Atom/mol stability, validity, uniqueness, novelty, JSD
│   │   └── sampling.py          # Reverse-process sampling (single + batch)
│   ├── tests/                   # 48 passing unit tests (1.4s)
│   │   ├── test_coarsening.py
│   │   ├── test_degradation.py
│   │   ├── test_diffusion.py
│   │   ├── test_equivariance.py
│   │   ├── test_lanczos.py
│   │   └── test_noise_schedules.py
│   └── configs/
│       ├── defaults.yaml
│       ├── qm9_unconditional.yaml
│       └── geom_drugs_unconditional.yaml
├── scripts/
│   ├── train_qm9.py             # QM9 training entry point
│   ├── train_geom_drugs.py      # GEOM-Drugs training (with transfer learning)
│   ├── sample.py                # Generate molecules from checkpoint
│   ├── evaluate.py              # Compute generation metrics
│   ├── run_evaluation.py        # One-command: sample → evaluate → comparison table
│   ├── run_ablations.py         # Step A4: automated ablation studies
│   └── validate_contributions.py # Empirical validation of contributions 1-4
├── validation_results/
│   └── validation_results.json
├── data/                        # Cached data
│   └── molssd_processed/        # Preprocessed QM9 (train 1.3GB, val, test)
└── checkpoints/                 # QM9 training checkpoints (every 5K steps)
```

## Architecture Overview

```
Molecular Flexi-Net (our model, 10.25M params)
│
├── Input: noisy positions x_t, atom types, timestep t, resolution level k
│
├── Atom Type Embedding (learned, 128-dim)
├── Timestep + Resolution Embedding (sinusoidal + learned → MLP → 128-dim)
│
├── Per-level EGNN stacks (only level k is active at timestep t):
│   ├── Level 0: 4 EGNN blocks × 128-dim  (full atomic, ~29 nodes for QM9)
│   ├── Level 1: 3 EGNN blocks × 256-dim  (coarsened, ~10 nodes)
│   ├── Level 2: 3 EGNN blocks × 384-dim  (coarser, ~3 nodes)
│   └── Level 3: 2 EGNN blocks × 512-dim  (coarsest, ~1 node)
│
├── Output heads (per-level):
│   ├── Position noise prediction: Linear(hidden → 3)
│   └── Atom type prediction: Linear(hidden → num_atom_types)
│
└── Pooling/Unpooling layers (for U-Net extension, not yet active)
```

**EGNN** (E(n)-Equivariant GNN) is the message-passing building block inside
Flexi-Net. It preserves rotational/translational symmetry. The novelty is the
multi-resolution hierarchy + dynamic level activation (Contribution 5).

## Conditioning Pipeline (for UV-absorber generation)

```
Phase 1 (OMol25 pre-training):
    HOMO-LUMO gap (scalar) → ScalarPropertyEncoder → c ∈ R^256 → adaLN → Flexi-Net

Phase 3 (UV-absorber fine-tuning):
    UV spectrum (600-dim) → SpectrumEncoder (1D CNN) ─┐
    λ_max, ε, logP, etc. → ScalarPropertyEncoder ────┤→ Fusion → c ∈ R^256 → adaLN
                                                       └──────────────────────────────
    Same injection point (adaLN), same dimension. Backbone weights transfer.

Classifier-Free Guidance:
    Training: drop conditioning → null embedding with prob p=0.15
    Sampling: ε_guided = (1+w)·ε_cond − w·ε_uncond, w=2.5
```

Module: `molssd/models/conditioning.py`

## Core Method
MolSSD adapts Scale Space Diffusion (SSD, arXiv:2603.08709) from images to molecular graphs:
- Forward process: x_t = M_{1:t} x_0 + sigma_t * epsilon, where M is spectral graph coarsening
- Molecular scale space: Full atoms → functional groups → fragments → chromophore skeleton → centroid
- Non-isotropic posterior at resolution-changing steps (Lanczos algorithm)
- Architecture: Molecular Flexi-Net (hierarchical E(n)-equivariant GNN)
- Pre-training on OMol25 (100M+ DFT calculations, 83M molecules)
- Conditioning on UV absorption spectra, photostability, coating compatibility

## Seven Contributions
1. **Spectral graph coarsening as molecular blurring** — graph Laplacian eigenvectors define multi-resolution hierarchy (replaces BRICS/Murcko)
2. **Non-isotropic posterior** — Σ = σ²I − (σ⁴/σₜ²)MₜᵀMₜ at resolution-changing steps (domain-general, not molecule-specific)
3. **SE(3) equivariance proof** — center-of-mass aggregation preserves equivariance through entire coarsening hierarchy
4. **Information-theoretic grounding** — Info_mol(t) = (N_k/N) × (SNR(t)/SNR(0)) monotonically decreases
5. **Molecular Flexi-Net** — dynamic level activation, 3-5× compute savings at coarse levels
6. **Bridge: physical CG ↔ generative diffusion** — connects to renormalization group theory
7. **UV-absorber conditional generation** — spectrum-conditioned with classifier-free guidance

## Current Status (updated 2026-03-14)

### Completed ✓
- Core math modules: coarsening, diffusion, degradation, Lanczos, noise schedules
- Model architecture: EGNN blocks, embeddings, Flexi-Net (10.25M params)
- Conditioning module: ScalarPropertyEncoder, SpectrumEncoder, classifier-free guidance
- Data pipelines: QM9 loader, GEOM-Drugs loader, OMol25 loader (all with precomputed coarsening)
- Training infrastructure: losses (Min-SNR-5), EMA, optimizer, scheduler, trainer
- Training optimization: vectorized forward diffusion at all resolution levels (14× speedup)
- Scripts: train_qm9, train_geom_drugs, sample, evaluate, run_evaluation, run_ablations
- Evaluation: atom/mol stability, validity, uniqueness, novelty, bond length/angle JSD
- 48 unit tests passing (1.4s)
- Empirical validation of contributions 1-4 on 500 real QM9 molecules (all PASS)
- Theory explainer document (2000+ lines) with diagrams, references, validation
- Cloud training plan with cost analysis (cloud_training_plan/cloud_training_plan.tex)
- Paper draft structure with placeholder tables
- SBIR pitch materials (NSF, NIST, DOE)

### In Progress 🔄
- **Step A2: QM9 training** — currently at step ~163K/300K (54%), ~2h remaining
  - 10.25M param Flexi-Net, batch_size=128, cosine schedule, T=1000
  - Position loss: 0.62-0.65 (decreasing), type loss: 0.0000 (converged)
  - Speed: ~12-20 steps/s on RTX 4090 (was 1.4 steps/s before optimization)
  - Checkpoints saved every 5K steps in ./checkpoints/

### Not Yet Done
- Step A3: Evaluate trained model (sample 10K mols → comparison table) — script ready
- Step A4: Ablation studies — script ready (run_ablations.py)
- Step A5: GEOM-Drugs training — loader + script ready, needs data download
- Step A6: OMol25 pre-training — loader ready, needs Globus access + cloud GPU
- Step B1: Curate UV-absorber dataset (50K mols + TD-DFT spectra)
- Step B2: Conditional training with spectrum conditioning
- Step B3-B5: Generation campaigns, screening, wet-lab validation

## Phase A: Beat the Baselines (CURRENT PRIORITY)

### Step A1: Training Loop ✓ DONE
- Loss function: Min-SNR-5 weighted MSE (position) + cross-entropy (atom types)
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- EMA: decay=0.9999
- Gradient clipping: max norm 1.0
- LR schedule: linear warmup (5K steps) + cosine decay
- Mixed precision (bf16), checkpointing, W&B support
- Vectorized forward diffusion (no per-molecule loop) — 14× speedup

### Step A2: Train on QM9 🔄 IN PROGRESS
- Training at step ~163K/300K, ETA ~2h
- batch_size=128, num_workers=0 (data in memory)
- Checkpoints every 5K steps + auto-resume from checkpoint_latest.pt

### Step A3: Fill the Comparison Table — READY TO RUN
```bash
python3 scripts/run_evaluation.py --checkpoint checkpoints/checkpoint_final.pt --num-molecules 10000
```
Target baselines:
| Model | Atom Stab % | Mol Stab % | Valid % | Unique % | Time (min) |
|-------|------------|-----------|--------|---------|-----------|
| EDM | 98.7 | 82.0 | 98.7 | 99.5 | 28.4 |
| GeoLDM | 98.9 | 89.4 | 93.8 | 99.4 | — |
| MiDi | 99.0 | 83.3 | 99.0 | — | — |
| GCDM | 99.0 | 85.7 | — | — | — |
| EQGAT-diff | 98.7 | 81.3 | 98.5 | 99.4 | — |
| MolDiff | 98.2 | 78.5 | 96.1 | 99.3 | — |
| **MolSSD** | **??** | **??** | **??** | **??** | **??** |

MolSSD targets: ≥98% atom stab, ≥82% mol stab, ≥99% validity, 2-3× faster than EDM

### Step A4: Ablation Studies — READY TO RUN
```bash
python3 scripts/run_ablations.py --max-steps 100000
```

### Step A5: Scale to GEOM-Drugs — LOADER + SCRIPT READY
- 304K drug-like molecules, up to 181 atoms, 10 atom types
- Loader: `molssd/data/geom_drugs_loader.py`
- Script: `scripts/train_geom_drugs.py` (supports --pretrained for transfer learning)
- Needs: download rdkit_folder.tar.gz from Harvard Dataverse

### Step A6: Pre-train on OMol25 — LOADER READY
- 83M molecules, 2-350 atoms, 16 atom types
- Loader: `molssd/data/omol25_loader.py`
- UV-relevant properties: HOMO-LUMO gap, orbital energies, partial charges
- Needs: Globus access (apply now, 1-2 week approval), cloud GPU (A100)
- See: cloud_training_plan/cloud_training_plan.tex for full strategy

## Phase B: UV-Absorber Discovery (after Phase A succeeds)

### Step B1: Curate UV-Absorber Dataset
- 50K+ molecules from ChEMBL/PubChem/ZINC
- Classes: benzotriazoles, benzophenones, triazines, cyanoacrylates, oxanilides, cinnamates
- TD-DFT spectra (CAM-B3LYP/6-311+G(d,p))
- No existing UV spectrum dataset — must be curated + computed

### Step B2: Conditional Training
- Conditioning module BUILT: `molssd/models/conditioning.py`
- ScalarPropertyEncoder: HOMO-LUMO gap, λ_max, ε_max, logP, photostability
- SpectrumEncoder: 1D CNN on discretized spectrum (200-800nm, 600 bins)
- Classifier-free guidance: dropout p=0.15, guidance weight w=2.5
- 3-phase swap: pre-train on HOMO-LUMO gap → fine-tune with full spectrum

### Step B3-B5: Generation, Screening, Wet-Lab
(unchanged from original plan)

## Key Research Gaps Addressed
1. No information-theoretic basis for multi-scale molecular generation
2. Ad-hoc molecular fragmentation (BRICS, Murcko are heuristic)
3. No formal non-isotropic posteriors in molecular coarse-graining
4. Computational inefficiency at full atomic resolution for all denoising steps
5. Lack of interpretable intermediate states
6. No unified theory connecting physical coarse-graining and generative modeling
7. No generative design framework for UV-absorbing materials

## OMol25 UV-Relevant Properties
OMol25 is ground-state DFT (no excited states), but contains these UV-relevant properties:
- **HOMO-LUMO gap**: primary proxy for λ_max (E = hc/λ, gap 3.1eV ≈ 400nm, 4.4eV ≈ 280nm)
- **Orbital energies**: identifies n→π*, π→π* transition candidates
- **Partial charges** (Mulliken, Löwdin, NBO): indicates charge-transfer chromophores
- **Density matrix**: encodes π-delocalization / conjugation extent
- NOT included: TD-DFT excited states, oscillator strengths, full spectra, photostability

## Key External Resources
- SSD code: https://github.com/prateksha/ScaleSpaceDiffusion
- EDM code: https://github.com/ehoogeboom/e3_diffusion_for_molecules
- FairChem/eSEN: https://github.com/facebookresearch/fairchem
- OMol25 dataset: via fairchem package + Globus access
- GEOM-Drugs: Harvard Dataverse doi:10.7910/DVN/JNGTDF

## Build & Run Commands
```bash
# Run tests
python3 -m pytest molssd/tests/ -v

# Validate contributions 1-4 on QM9
python3 scripts/validate_contributions.py

# Train on QM9 (auto-resumes from checkpoint)
python3 scripts/train_qm9.py --batch-size 128 --max-steps 300000 --num-workers 0

# Evaluate trained model (Step A3)
python3 scripts/run_evaluation.py --checkpoint checkpoints/checkpoint_final.pt --num-molecules 10000

# Run ablation studies (Step A4)
python3 scripts/run_ablations.py --max-steps 100000

# Train on GEOM-Drugs with transfer learning (Step A5)
python3 scripts/train_geom_drugs.py --pretrained checkpoints/checkpoint_final.pt --batch-size 32

# System uses python3 (not python)
```

## Environment
- Python 3.12.3 with python3 command
- PyTorch 2.10.0+cu128 with CUDA
- PyTorch Geometric (PyG) 2.7.0
- RDKit 2025.09.5
- GPU: NVIDIA RTX 4090 Laptop (16.4 GB VRAM, ~40% utilized during training)
- No LaTeX installation on system — compile .tex files locally
- WSL2 on Windows (32GB host RAM)
- WSL config: 24GB memory, 8GB swap, autoMemoryReclaim=gradual, sparseVhd=true
- DataLoader: num_workers=0 recommended (data in memory, avoids WSL OOM)

## Development Notes
- SE(3) equivariance must be preserved through coarsening (use center-of-mass aggregation)
- Min-SNR-5 loss weighting + ConvexDecay(0.5) resolution schedule
- eSEN is ground-state only; TD-DFT needed for excited states/absorption spectra
- The non-isotropic posterior is domain-general (images, point clouds, audio — not molecule-specific)
- Graph Laplacian on molecules is known; the novelty is using it as the blur operator inside a diffusion forward process
- Training bottleneck is CPU collation, not GPU compute (QM9 molecules are small)
- For GEOM-Drugs / OMol25, GPU will be fully utilized (larger molecules)
- Always commit AND push after completing work (user preference)

## Training Performance Notes
- Original training speed: 1.4 steps/s (per-molecule Python loop bottleneck)
- After optimization: 12-20 steps/s (14× speedup via vectorized forward diffusion)
- Key optimization: precompute coarsened positions/types/edges in dataset, eliminate per-molecule loop
- Resolution level distribution (T=1000): 55% level 0, 40% level 1, 5% level 2-3
- GPU memory: ~3.4/16 GB VRAM used (model is small, QM9 molecules are tiny)
- RTX 4090 will be fully utilized on GEOM-Drugs (molecules up to 181 atoms)
