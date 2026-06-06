# Spectroscopy-Informed Dual-Stream Raman-Wavelet Fusion for Robust Bacterial Classification

This repository contains the official code for the paper: **"Spectroscopy-Informed Dual-Stream Raman-Wavelet Fusion for Robust Bacterial Classification"** (SDR-Fusion).

## Overview
Rapid and accurate identification of pathogenic bacteria is critical for clinical diagnostics and antimicrobial resistance (AMR) management. This project proposes a spectroscopy-informed dual-stream framework (SDR-Fusion) that jointly processes raw 1D Raman spectra and their 2D continuous wavelet transform (CWT) scalograms to achieve highly accurate and noise-robust bacterial identification. 

## Repository Structure
- `pinnacle/`: Core deep learning architecture, including dataset loaders, model definitions (1D branch, 2D branch, SeparationCross module), training, and evaluation utilities.
- `scripts/`: Executable scripts for generating figures, running baselines, performing ablation studies, and training the models.
- `configs/`: YAML configuration files defining model hyperparameters and training setups.
- `new figures folder/`: Generated PDFs and PNGs of the figures included in the manuscript.
- `naman AMR.ipynb`: Jupyter notebook demonstrating exploratory data analysis and the AMR classification pipeline.

## Installation

To set up the environment and install dependencies:

```bash
git clone https://github.com/Namangarg484/Biomedical-Signal-Processing-and-Control.git
cd Biomedical-Signal-Processing-and-Control
pip install -r requirements.txt
```

## Data Download & Reproducibility
> **Important Note for Reviewers:**
> Due to file size constraints on GitHub, the large raw datasets and the trained model weights are not included in this repository directly.
> 
> **Data Link for 30-Class Benchmark:** The 30-class Raman classification dataset used in this work (Bacteria-ID Benchmark Dataset) can be accessed and downloaded from:
> [https://github.com/csho33/bacteria-ID](https://github.com/csho33/bacteria-ID)

This codebase supports reproducing both the **Standard Benchmark (5-7 classes)** and the **Extended Taxonomy Benchmark (30 classes)**. Please place the downloaded dataset files into the `./data/` directory and follow the corresponding track below.

---

### Track A: Standard Benchmark (5-7 Classes)

**1. Data Preparation**
Ensure the following files are placed in the `./data/` directory:
- `X_2018_proc.npy`
- `X_2019_proc.npy`
- `y_2018clinical.npy`
- `y_2019clinical.npy`

**2. Preprocessing (Scalogram Generation)**
Generate the 2D Wavelet Scalograms required for the dual-stream model:
```bash
python scripts/generate_wavelets.py
```

**3. Training**
Run the main training pipeline:
```bash
python scripts/train.py --config config.yaml
```

---

### Track B: Extended Taxonomy Benchmark (30 Classes)

**1. Data Preparation**
Ensure the following files are placed in the `./data/` directory:
- `X_reference.npy`
- `y_reference.npy`
- `X_test.npy`
- `y_test.npy`

**2. Preprocessing (Scalogram Generation)**
Generate the 2D Wavelet Scalograms specifically for the 30-class taxonomy:
```bash
python scripts/generate_wavelets_30class.py
```

**3. Training**
Run the 30-class specific training pipeline:
```bash
python scripts/train_30class.py --config configs/30class.yaml
```

---

## Usage (Figure Generation)
To generate the manuscript figures, execute the respective files in the `scripts/` directory:
```bash
python scripts/generate_fig9_robustness.py
```

## Note on Hardware Reproducibility
While this codebase enforces strict seeding (`seed: 42`, `deterministic: true`), differences in hardware backends (e.g., Apple Silicon MPS vs NVIDIA CUDA) natively handle floating-point arithmetic differently within PyTorch's cross-attention blocks. Running this code on non-MPS hardware may result in nominal accuracy deviations ($\pm 0.1\%$ to $0.3\%$) from the exact figures reported in the manuscript.
