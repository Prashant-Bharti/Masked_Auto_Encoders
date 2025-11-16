# Masked Autoencoders for Vision (MAE)
Built a Masked Auto-Encoder (MAE) for reconstructing masked image patches using the STL-10 unlabeled dataset. Enhanced existing Auto-Encoders via curriculum‑masking technique, achieving over 5\% improvement in accuracy.

This repository implements a light-weight, Colab-friendly reproduction of
Masked Autoencoders (MAE) for self-supervised visual representation learning,
along with a compute-light set of improvements. It compares a faithful
reproduction of the MAE paper (Original MAE) with an Improved variant
that adds curriculum masking, a high-frequency reconstruction loss, and a
uniformity regularizer. The experiments use STL-10 (unlabeled for pretraining,
labeled for downstream evaluation) to enable rapid iteration in limited
compute environments.

Contents
- Overview
- What you will find
- Key ideas
- Datasets
- Model architecture
- Training & evaluation protocol
- Improvements over existing methods
- Reproducing the results
- Visualization
- How to run
- Limitations and future work
- Outputs and Results

---

## Overview

Self-supervised learning with MAE masks a large portion of image patches and trains a model to reconstruct the missing content. The encoder processes only the visible patches; a lightweight decoder handles reconstruction using mask tokens. This approach yields strong transfer to downstream tasks with minimal labeled data.

This project provides:
- A faithful, Colab-friendly MAE baseline (Original MAE).
- An Improved MAE variant with:
  - Curriculum masking
  - Masked high-frequency (Laplacian) loss
  - Uniformity regularizer on encoder embeddings
- Fair, side-by-side comparisons on the same data and evaluation protocol.
- Side-by-side reconstruction visualization using identical masks.
- Checkpointing and easy reproducibility.

---

## What you will find

- Core code for original MAE and the improved variant
  - Lightweight ViT-like encoder
  - Lightweight decoder
  - Pixel-space reconstruction loss (MSE on masked patches)
  - Optional per-patch normalization
- Improvements
  - Curriculum masking
  - Masked Laplacian high-frequency loss
  - Uniformity regularizer on encoder embeddings
- Downstream evaluation
  - Linear probing (LP)
  - End-to-end fine-tuning (FT)
- Visualization tools
  - Side-by-side recon grid (Original vs Improved)
  - Quantitative metrics for masked region quality (PSNR, Laplacian HF)
- Checkpoints, logs, and results files
  - runs/original/
  - runs/improved/
  - runs/compare/ for recon grids

---

## Key ideas

- MAE with a high masking ratio (≈75%) reduces redundancy and encourages holistic understanding.
- Encoder-only on visible patches; a lightweight decoder reconstructs the full image in pixel space.
- Improvements aim to provide early, compute-friendly gains without changing the core architecture fundamentally.

---

## Datasets

- STL-10
  - Unlabeled split used for pretraining (self-supervised MAE).
  - Labeled train/test splits used for downstream evaluation (Linear Probing and Fine-tuning).
- Image size: default 224×224 (upscaled from STL-10's native 96×96; patch size 16×16, yielding 14×14 = 196 patches).
- Optional: you can switch to 96×96 for faster ablations.

Notes:
- STL-10 is a compact, diverse dataset suitable for quick experiments and method validation on Colab.

---

## Model architecture

- Encoder: ViT-like, embed_dim = 192, depth = 8, heads = 6
- Patch size: 16×16
- Positional encoding: 2D sine-cosine
- Decoder: embed_dim = 128, depth = 4, heads = 4
- Mask tokens: learned, used to fill masked positions in the decoder input
- Reconstruction target: pixel values in masked patches; optional per-patch normalization
- Improvements (optional): curriculum masking, Laplacian HF loss, uniformity regularizer

---

## Training & evaluation protocol

- Pretraining data: STL-10 unlabeled split
- Pretraining objective: reconstruct masked patches (MSE in pixel space)
- Evaluation protocol:
  - Linear Probing (LP): encoder frozen, train BN+linear classifier on STL-10 train; test on STL-10 test
  - Fine-Tuning (FT): end-to-end training on STL-10 train; test on STL-10 test
- Hyperparameters (default)
  - Pretraining epochs: 10 (quick test); scalable to 50–200+ for stronger results
  - Batch size: 64 (pretraining); 128 (linear probing / fine-tuning)
  - Optimizer: AdamW
  - Learning rate: 1.5e-4 (base; scaled with batch size)
  - LR schedule: cosine decay with warmup
  - Mixed precision: enabled
  - Masking: Original MAE uses fixed 0.75; Improved MAE uses curriculum masking (0.5 → 0.75)
- Checkpoints and reproducibility
  - Checkpoints saved under runs/original and runs/improved
  - Encoder weights available as encoder_pretrained.pt
  - Results.json stores LP/FT and pretraining time
  - Visualization grid saved as runs/compare/recon_grid.png

---

## Improvements over existing methods

- Curriculum masking: stabilizes early training, enables faster convergence at short budgets.
- Laplacian high-frequency loss: encourages sharper edges/textures in masked regions, improving reconstruction quality.
- Uniformity regularizer: reduces representation collapse, improves linear separability.

---

## Reproducing the results

- Baseline (Original MAE) and Improved MAE share identical data, architecture, and evaluation protocols; only loss and masking schedule differ.
- Use STL-10 pretraining (unlabeled) followed by LP/FT on the labeled STL-10 splits.
- Compare LP@1 and FT@1, plus side-by-side HF/PSNR metrics and recon_grid visualization.

---

## Visualization

- recon_grid.png: side-by-side grid showing Original MAE vs Improved MAE reconstructions under identical masks.
- HF/Laplacian metric: a quantitative measure of high-frequency reconstruction accuracy.

---

## How to run 

- Prereqs: Python 3.8+, PyTorch, torchvision, tqdm
- Repo setup (clone, install, run)
  - pip install torch torchvision tqdm
  - Run original MAE baseline
  - Run improved MAE variant
  - Generate recon_grid.png with identical masks

- Example commands (adjust paths as needed)
  - Pretrain Original MAE:
    - python mae_compare.py --epochs_pretrain 10 --epochs_lp 10 --epochs_ft 10 \
      --batch_pretrain 64 --batch_cls 128 --img_size 224 --output_root ./runs \
      --resume_pretrain
  - Pretrain Improved MAE (with curriculum, Laplacian HF, uniformity):
    - python mae_compare.py --epochs_pretrain 10 --epochs_lp 10 --epochs_ft 10 \
      --batch_pretrain 64 --batch_cls 128 --img_size 224 --output_root ./runs \
      --resume_pretrain

- Visualization
  - Run the provided visualization snippet after both models are trained to produce recon_grid.png and the HF/PSNR reports.

---

## Licensing & Attribution

- This work builds on the MAE approach by He et al. (CVPR 2022). See references in the code for full citations.
- All experiments and code are provided for educational and research purposes.

---

## Outputs
### Reconstructed output images:
- Click to view:
<img width="906" height="7234" alt="recon_grid (1)" src="https://github.com/user-attachments/assets/67cb0d9f-0ca5-4d35-8037-5dbdcfc5ca0a" />

### Training:
<img width="1185" height="477" alt="image" src="https://github.com/user-attachments/assets/921b7ebb-7a57-452c-9d77-7f7552f6a6ad" />

### Results:
<img width="1463" height="867" alt="image" src="https://github.com/user-attachments/assets/e812507b-f087-4758-b0b6-d8156f911540" />
