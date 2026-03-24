# DDPM Paper Replication and Application to Laser Beam Images

Implementation of Denoising Diffusion Probabilistic Models (DDPM) applied to CIFAR-10 and OAM (Orbital Angular Momentum) laser beam images. Experiments cover pixel-space diffusion, VAE compression, and latent diffusion. CIFAR dataset should load automatically with `torchvision`; contact owen.m.omalley@gmail.com for inquiries regarding laser dataset.

---

## Setup

```bash
bash setup_env.sh && source venv/bin/activate
pip install -r requirements.txt
```

---

## Experiments

### 1. CIFAR-10 Baseline

Reproduction of [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) on CIFAR-10, used to validate the implementation. Supports training, sampling, and evaluation via FID and Inception Score.

```bash
python run_cifar.py train | sample | eval
```

---

### 2. Laser Beam Dataset — Pixel-Space DDPM

Pixel-space DDPM trained directly on 128×128 grayscale OAM beam intensity images. The dataset covers 8 OAM modes (`gauss`, `p1–p4`, `n1–n3`) at multiple turbulence strengths, stored as MATLAB `.mat` files.

```bash
python run_oam.py train_ddpm | sample_ddpm | progression_ddpm
```

---

### 3. VAE Trained on Laser Beam Data

Convolutional VAE that compresses 128×128 OAM images into an 8×8×4 latent space. Required as a first step before training the latent diffusion model. Analysis tools produce reconstruction grids, per-(mode, turbulence) quality metrics (MSE, SSIM), and PCA/t-SNE visualisations of the latent space.

```bash
python run_oam.py train_vae | visualize_vae | eval_vae_quality
```

---

### 4. Latent Diffusion Model (LDM)

DDPM trained on the frozen VAE's latent space rather than pixel space, reducing computational cost. At inference, samples are decoded back to images through the VAE. Includes slerp interpolation between images in latent space.

```bash
python run_oam.py train_ldm | sample_ldm
python analyse_interp_latent.py
```

---

### 5. Evaluation & Miscellaneous

- **FID / Inception Score** (`eval.py`) — standard image quality metrics for the CIFAR-10 model
- **CNN turbulence classifier** (`cnn_turb_classifier.py`) — small CNN trained to classify turbulence strength from beam images; applied to generated samples to test physical plausibility
- **VAE quality analysis** (`analyse_vae_quality.py`) — MSE and SSIM per (mode, turbulence) cell, exported as CSV and bar chart
- **Latent space visualisation** (`visualize_latent.py`) — PCA/t-SNE scatter plots and principal component traversals

```bash
python run_oam.py train_cnn | eval_cnn
```
