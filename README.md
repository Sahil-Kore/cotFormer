# CoTAdapter Experiment on GPT-2 Backbones

This repository contains a small-scale experiment comparing three adapter architectures on frozen GPT-2 backbones:

- BlockUniversalAdapter  
- CoTAdapter (weight-tied repeated refinement)  
- DeepTransformerAdapter  

Backbones tested:
- DistilGPT-2  
- GPT-2 Small  
- GPT-2 Medium  

All training and analysis were performed on a **single RTX 3050 (4 GB VRAM)** using an **adapter-based SFT approach**:  
the GPT-2 backbone is kept **frozen**, and only the adapters + LM head are trained.

## Hyperparameters
All experiments use the **Wikitext-2 Raw**  with max samples set to 20000  sequence length of 128 ,
```python
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
