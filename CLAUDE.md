# MolSSD Project Context

## Project Overview
- Molecular Scale Space Diffusion (MolSSD): first application of scale-space diffusion theory to molecular generation
- Application: novel UV-absorbing molecular materials for protective coatings
- Target venue: Nature

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
├── paper_draft/main.tex         # Nature-style manuscript
├── research_proposal/proposal.tex # Research proposal
└── technical_whitepaper/whitepaper.tex # Technical whitepaper
```

## Core Method
MolSSD adapts Scale Space Diffusion (SSD, arXiv:2603.08709) from images to molecular graphs:
- Forward process: x_t = M_{1:t} x_0 + sigma_t * epsilon, where M is spectral graph coarsening
- Molecular scale space: Full atoms -> functional groups -> fragments -> chromophore skeleton -> centroid
- Non-isotropic posterior at resolution-changing steps (Lanczos algorithm)
- Architecture: Molecular Flexi-Net (hierarchical E(n)-equivariant GNN)
- Pre-training on OMol25 (140M DFT calculations, 83M molecules)
- Conditioning on UV absorption spectra, photostability, coating compatibility

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
(To be populated as implementation progresses)

## Development Notes
- SE(3) equivariance must be preserved through coarsening (use center-of-mass aggregation)
- Min-SNR-5 loss weighting + ConvexDecay(0.5) resolution schedule
- eSEN is ground-state only; TD-DFT needed for excited states/absorption spectra
- Target datasets: QM9 (validation), GEOM-Drugs (scaling), OMol25 (pre-training)
