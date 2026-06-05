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
>
> To reproduce the results, please download the dataset from the link above and place it in the appropriate data directories as defined in `config.yaml` before running the training or evaluation scripts.

## Usage
1. Update the `config.yaml` file to point to your local data directories (e.g., `data_dir` and `image_dir`).
2. To run the main training pipeline:
```bash
python scripts/train.py --config config.yaml
```
3. To generate the manuscript figures, execute the respective files in the `scripts/` directory:
```bash
python scripts/generate_fig9_robustness.py
```
