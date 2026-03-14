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
│   └── what_we_created.tex      # 2000+ line document: all 7 contributions with
│                                # diagrams, validation results, references,
│                                # "what's known vs new" boxes, benchmarking plan
├── molssd/                      # Core Python package
│   ├── core/
│   │   ├── coarsening.py        # Graph Laplacian, spectral clustering, coarsening matrix
│   │   ├── diffusion.py         # MolSSDDiffusion: forward/reverse process
│   │   ├── degradation.py       # DegradationOperator: M_t and M_{1:t}
│   │   ├── lanczos.py           # Lanczos tridiag, posterior eigendecomp, non-iso sampling
│   │   └── noise_schedules.py   # CosineSchedule, LinearSchedule, ResolutionSchedule
│   ├── models/
│   │   ├── egnn.py              # E(n)-equivariant GNN blocks
│   │   ├── embeddings.py        # Time, resolution, atom type embeddings
│   │   └── flexi_net.py         # Molecular Flexi-Net (hierarchical multi-resolution)
│   ├── data/
│   │   └── qm9_loader.py        # QM9MolSSD dataset, preprocessing, caching
│   ├── training/                # (training loop — to be implemented)
│   ├── evaluation/              # (metrics — to be implemented)
│   ├── tests/                   # 48 passing unit tests
│   │   ├── test_coarsening.py
│   │   ├── test_degradation.py
│   │   ├── test_diffusion.py
│   │   ├── test_equivariance.py
│   │   ├── test_lanczos.py
│   │   └── test_noise_schedules.py
│   └── configs/                 # Hydra YAML configs
├── scripts/
│   └── validate_contributions.py # Empirical validation of contributions 1-4 on QM9
├── validation_results/
│   └── validation_results.json   # Results from validate_contributions.py
├── ai_multifunctional_coatings/  # SBIR pitch materials, mind maps
├── sbir/                         # SBIR-related documents
├── data/                         # Cached QM9 data (train 1.0GB, val 203MB, test 125MB)
└── checkpoints/                  # Model checkpoints (empty — no training yet)
```

## Core Method
MolSSD adapts Scale Space Diffusion (SSD, arXiv:2603.08709) from images to molecular graphs:
- Forward process: x_t = M_{1:t} x_0 + sigma_t * epsilon, where M is spectral graph coarsening
- Molecular scale space: Full atoms → functional groups → fragments → chromophore skeleton → centroid
- Non-isotropic posterior at resolution-changing steps (Lanczos algorithm)
- Architecture: Molecular Flexi-Net (hierarchical E(n)-equivariant GNN)
- Pre-training on OMol25 (140M DFT calculations, 83M molecules)
- Conditioning on UV absorption spectra, photostability, coating compatibility

## Seven Contributions
1. **Spectral graph coarsening as molecular blurring** — graph Laplacian eigenvectors define multi-resolution hierarchy (replaces BRICS/Murcko)
2. **Non-isotropic posterior** — Σ = σ²I − (σ⁴/σₜ²)MₜᵀMₜ at resolution-changing steps (domain-general, not molecule-specific)
3. **SE(3) equivariance proof** — center-of-mass aggregation preserves equivariance through entire coarsening hierarchy
4. **Information-theoretic grounding** — Info_mol(t) = (N_k/N) × (SNR(t)/SNR(0)) monotonically decreases
5. **Molecular Flexi-Net** — dynamic level activation, 3-5× compute savings at coarse levels
6. **Bridge: physical CG ↔ generative diffusion** — connects to renormalization group theory
7. **UV-absorber conditional generation** — spectrum-conditioned with classifier-free guidance

## Current Status

### Completed
- Core math modules: coarsening, diffusion, degradation, Lanczos, noise schedules
- Model architecture: EGNN blocks, embeddings, Flexi-Net
- Data pipeline: QM9 loader, preprocessing, caching (train/val/test split ready)
- 48 unit tests passing (2.23s)
- Empirical validation of contributions 1-4 on 500 real QM9 molecules (all PASS)
- Theory explainer document (2000+ lines) with diagrams, references, validation, benchmarking plan
- Paper draft structure with placeholder tables
- SBIR pitch materials (NSF, NIST, DOE)

### NOT yet done (next steps below)
- Training loop
- Actual model training
- Benchmark comparison numbers
- UV-absorber dataset
- Conditional generation
- Wet-lab experiments

## Phase A: Beat the Baselines (CURRENT PRIORITY)

### Step A1: Training Loop (NEXT IMMEDIATE TASK)
- Wire up: loss function (Min-SNR-5 weighted MSE), optimizer (AdamW), EMA (0.9999), gradient clipping (max norm 1.0)
- Learning rate: warmup + cosine decay
- Logging: W&B integration
- Checkpointing: save every N epochs + best model
- Mixed precision (bf16), multi-GPU (DDP)
- Resolution-aware batching

### Step A2: Train on QM9
- ~200-500 epochs, ~24-48h on GPU
- Generate 10,000 molecules for evaluation
- Compute: atom stability, mol stability, validity, uniqueness, novelty, FCD, wall-clock time

### Step A3: Fill the Comparison Table
Target baselines to beat or match:
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

### Step A4: Ablation Studies
- Isotropic vs non-isotropic posterior
- Spectral vs BRICS vs random coarsening
- 2-level vs 3-level vs 5-level hierarchy
- Full Flexi-Net vs fixed-width network
- Each ablation isolates one contribution's impact

### Step A5: Scale to GEOM-Drugs
- 304K drug-like molecules, up to 181 atoms
- Demonstrate speedup grows with molecule size (3-4× expected)

### Step A6: Pre-train on OMol25
- 83M molecules, 2-350 atoms
- Multi-GPU training (8-32 GPUs)
- Demonstrate sub-linear scaling (N^1.4 vs EDM's N^1.9)

## Phase B: UV-Absorber Discovery (after Phase A succeeds)

### Step B1: Curate UV-Absorber Dataset
- 50K+ molecules from ChEMBL/PubChem/ZINC
- Classes: benzotriazoles, benzophenones, triazines, cyanoacrylates, oxanilides, cinnamates
- TD-DFT spectra (CAM-B3LYP/6-311+G(d,p))

### Step B2: Conditional Training
- Spectrum conditioning: 1D CNN on discretized spectrum → cross-attention
- Property conditioning: photostability, logP, SA score, MW → adaLN
- Classifier-free guidance (dropout p=0.1-0.2, guidance w=2.5)

### Step B3: Generation Campaigns
- Campaign 1: Broad-spectrum UVA+UVB (λ_max 300-350nm, ε > 30,000)
- Campaign 2: Photostable (ESIPT mechanism, τ_S1 < 1ps)
- Campaign 3: Coating-compatible (logP 3-6, MW 250-500, HSP match)
- Campaign 4: UV-C absorbers (λ_max 200-280nm, less explored)
- Total: 50,000+ candidates across campaigns

### Step B4: Screening Pipeline
- Validity filter → uniqueness → novelty → SA score < 4.0 → structural alerts
- eSEN geometry optimization + stability (1ps MD at 300K)
- TD-DFT validation of top 500 (CAM-B3LYP)
- ADMET/toxicity screening

### Step B5: Wet-Lab Validation
- Synthesize top 10-20 candidates
- UV-Vis spectroscopy (experimental λ_max vs predicted)
- Photostability testing (500h xenon arc, target >90% retention)
- Coating formulation (1-5 wt% in epoxy/PU)
- QUV accelerated weathering (2000h)
- Compare against Tinuvin 328, Uvinul 3048, Chimassorb 81

## Key Research Gaps Addressed
1. No information-theoretic basis for multi-scale molecular generation
2. Ad-hoc molecular fragmentation (BRICS, Murcko are heuristic)
3. No formal non-isotropic posteriors in molecular coarse-graining
4. Computational inefficiency at full atomic resolution for all denoising steps
5. Lack of interpretable intermediate states
6. No unified theory connecting physical coarse-graining and generative modeling
7. No generative design framework for UV-absorbing materials

## Key External Resources
- SSD code: https://github.com/prateksha/ScaleSpaceDiffusion
- EDM code: https://github.com/ehoogeboom/e3_diffusion_for_molecules
- FairChem/eSEN: https://github.com/facebookresearch/fairchem
- OMol25 dataset: via fairchem package

## Build & Run Commands
```bash
# Run tests
python3 -m pytest molssd/tests/ -v

# Validate contributions 1-4 on QM9
python3 scripts/validate_contributions.py

# System uses python3 (not python)
```

## Environment
- Python 3.x with python3 command
- PyTorch 2.10.0+cu128 with CUDA
- PyTorch Geometric (PyG) 2.7.0
- RDKit 2025.09.5
- No LaTeX installation on system — compile .tex files locally

## Development Notes
- SE(3) equivariance must be preserved through coarsening (use center-of-mass aggregation)
- Min-SNR-5 loss weighting + ConvexDecay(0.5) resolution schedule
- eSEN is ground-state only; TD-DFT needed for excited states/absorption spectra
- The non-isotropic posterior is domain-general (images, point clouds, audio — not molecule-specific)
- Graph Laplacian on molecules is known; the novelty is using it as the blur operator inside a diffusion forward process
- Always commit AND push after completing work (user preference)
