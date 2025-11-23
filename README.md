# CoTAdapter Experiment on GPT-2 Backbones

This repository contains a small-scale experiment evaluating **three adapter architectures** on frozen GPT-2 backbones:

- **BlockUniversalAdapter**
- **CoTAdapter** (weight-tied repeated refinement, CoT-style)
- **DeepTransformerAdapter** (added untied transformer layers)

Backbones tested:
- DistilGPT-2  
- GPT-2 Small  
- GPT-2 Medium  

All training and analysis were performed on a **single RTX 3050 (4 GB VRAM)** using an **adapter-based SFT strategy**: the GPT-2 backbone is frozen, and only the adapters + LM head are trained.

## Contents
- `train.py` — training script  
- `models/` — implementations of CoTAdapter, BlockUniversalAdapter, DeepTransformerAdapter  
- `results/*.csv` — metrics for each backbone  
- `conclusion.ipynb` — full analysis, plots, and written conclusions  
- `cotformer.pdf` — reference paper used (Mohtashami et al., 2024)

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
