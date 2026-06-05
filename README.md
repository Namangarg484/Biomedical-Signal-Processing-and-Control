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
> **Data Link:** The 30-class Raman classification dataset used in this work (Bacteria-ID Benchmark Dataset) can be accessed and downloaded from:
> [https://github.com/csho33/bacteria-ID](https://github.com/csho33/bacteria-ID)

### Data Preparation
Once downloaded, place the dataset files into the `data/` directory. The codebase strictly expects the following filenames:
- `X_2018_proc.npy`
- `X_2019_proc.npy`
- `y_2018clinical.npy`
- `y_2019clinical.npy`

*(Note: The `config.yaml` has been configured to look for these in the `./data/` folder).*

### Preprocessing (Crucial Step)
Because the SDR-Fusion architecture is a dual-stream model, it requires 2D Wavelet Scalograms alongside the 1D spectra. **You must generate the scalograms before training.**
Run the wavelet generation script:
```bash
python scripts/generate_wavelets.py
```
This will generate `X_2018_wavelet.npy` and `X_2019_wavelet.npy` in your data directory.

## Usage
1. To run the main training pipeline:
```bash
python scripts/train.py --config config.yaml
```
2. To generate the manuscript figures, execute the respective files in the `scripts/` directory:
```bash
python scripts/generate_fig9_robustness.py
```

## Note on Hardware Reproducibility
While this codebase enforces strict seeding (`seed: 42`, `deterministic: true`), differences in hardware backends (e.g., Apple Silicon MPS vs NVIDIA CUDA) natively handle floating-point arithmetic differently within PyTorch's cross-attention blocks. Running this code on non-MPS hardware may result in nominal accuracy deviations ($\pm 0.1\%$ to $0.3\%$) from the exact figures reported in the manuscript.
