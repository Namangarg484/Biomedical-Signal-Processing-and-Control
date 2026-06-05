#!/usr/bin/env python3
"""
PINNACLE — Full pipeline runner.

Runs the complete pipeline from data preparation through training and evaluation.

Usage:
    python scripts/run_all.py                    # Full pipeline
    python scripts/run_all.py --skip-wavelets    # Skip wavelet generation
    python scripts/run_all.py --eval-only        # Evaluate existing checkpoint
    python scripts/run_all.py --epochs 5         # Quick smoke test
    python scripts/run_all.py --figures-only     # Regenerate figures from existing data
"""

import argparse
import os
import sys
import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pinnacle.utils import set_seed, get_device, logger


def setup_data_symlinks(data_dir: str, source_dir: str):
    """Create symlinks from data/ to testing_data/ for .npy files."""
    os.makedirs(data_dir, exist_ok=True)

    files = [
        "X_2018_proc.npy",
        "X_2019_proc.npy",
        "X_2019_wavelet.npy",
        "y_2018clinical.npy",
        "y_2019clinical.npy",
        "wavenumbers.npy",
    ]

    for f in files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(data_dir, f)

        if os.path.exists(dst) or os.path.islink(dst):
            continue
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), dst)
            logger.info(f"  🔗 Linked: {f}")
        else:
            logger.warning(f"  ⚠️  Source not found: {src}")


def check_data(data_dir: str) -> dict:
    """Check which data files are available."""
    status = {}
    for f in ["X_2018_proc.npy", "X_2019_proc.npy", "y_2018clinical.npy",
              "y_2019clinical.npy", "wavenumbers.npy",
              "X_2018_wavelet.npy", "X_2019_wavelet.npy"]:
        path = os.path.join(data_dir, f)
        exists = os.path.exists(path)
        size = ""
        if exists:
            size_mb = os.path.getsize(path) / (1024**2)
            size = f" ({size_mb:.1f} MB)"
        status[f] = exists
        icon = "✅" if exists else "❌"
        logger.info(f"  {icon} {f}{size}")
    return status


def main():
    parser = argparse.ArgumentParser(description="PINNACLE Full Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip-wavelets", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-fusion", action="store_true")
    args = parser.parse_args()

    # Change to project root
    os.chdir(PROJECT_ROOT)

    # Load config
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device
    if args.no_fusion:
        config["use_fusion"] = False

    set_seed(config["seed"])
    device = get_device(config.get("device", "auto"))

    logger.info("=" * 70)
    logger.info("🧬 PINNACLE — Full Pipeline Runner")
    logger.info("=" * 70)

    # ================================================================
    # STEP 1: Data setup
    # ================================================================
    logger.info("\n📦 Step 1: Data Setup")
    logger.info("-" * 50)

    data_dir = config["data_dir"]
    source_dir = os.path.join(PROJECT_ROOT, "testing_data")
    setup_data_symlinks(data_dir, source_dir)

    logger.info("\n📊 Data inventory:")
    data_status = check_data(data_dir)

    # Check required files
    required = ["X_2018_proc.npy", "X_2019_proc.npy", "y_2018clinical.npy", "y_2019clinical.npy"]
    missing = [f for f in required if not data_status.get(f, False)]
    if missing:
        logger.error(f"❌ Missing required files: {missing}")
        logger.error("   Please ensure testing_data/ contains these files.")
        sys.exit(1)

    # ================================================================
    # STEP 2: Wavelet generation (if needed)
    # ================================================================
    if not args.skip_wavelets and not args.eval_only and not args.figures_only:
        logger.info("\n🔬 Step 2: Wavelet Generation")
        logger.info("-" * 50)

        from pinnacle.wavelet import generate_wavelet_dataset

        for dataset_name, proc_file, wav_file in [
            ("2018", "X_2018_proc.npy", "X_2018_wavelet.npy"),
            ("2019", "X_2019_proc.npy", "X_2019_wavelet.npy"),
        ]:
            wav_path = os.path.join(data_dir, wav_file)
            proc_path = os.path.join(data_dir, proc_file)

            if os.path.exists(wav_path):
                size_gb = os.path.getsize(wav_path) / (1024**3)
                logger.info(f"  ⏭️  {wav_file} exists ({size_gb:.2f} GB)")
                continue

            if not os.path.exists(proc_path):
                logger.warning(f"  ⚠️  {proc_file} not found, skipping wavelets")
                continue

            logger.info(f"  Generating {wav_file}...")
            wav_config = config.get("wavelet", {})
            X_proc = np.load(proc_path)
            generate_wavelet_dataset(
                X_proc,
                img_size=wav_config.get("img_size", 224),
                n_scales=wav_config.get("n_scales", 256),
                wavelet=wav_config.get("type", "ricker"),
                chunk_size=wav_config.get("chunk_size", 100),
                output_path=wav_path,
            )
    else:
        logger.info("\n⏭️  Step 2: Skipping wavelet generation")

    # ================================================================
    # STEP 3: Generate data figures
    # ================================================================
    logger.info("\n📊 Step 3: Data Visualization")
    logger.info("-" * 50)

    from pinnacle.visualize import plot_single_spectrum, plot_class_spectra
    from pinnacle.dataset import load_data

    data = load_data(data_dir)
    figure_dir = config.get("figure_dir", "outputs/figures")

    # Fig 1: Single spectrum
    plot_single_spectrum(
        data["X_2018"][0],
        wavenumbers=data.get("wavenumbers"),
        output_path=os.path.join(figure_dir, "fig1_single_spectrum.png"),
    )

    # Fig 2: Class spectra
    X_all = np.concatenate([data["X_2018"], data["X_2019"]])
    y_all = np.concatenate([data["y_2018"], data["y_2019"]])
    plot_class_spectra(
        X_all, y_all,
        wavenumbers=data.get("wavenumbers"),
        output_path=os.path.join(figure_dir, "fig2_class_spectra.png"),
    )

    if args.figures_only:
        logger.info("✅ Figures generated. Exiting (--figures-only mode).")
        return

    # ================================================================
    # STEP 4: Training
    # ================================================================
    if not args.eval_only:
        logger.info("\n🏋️ Step 4: Training")
        logger.info("-" * 50)

        from pinnacle.model import PINNACLE
        from pinnacle.dataset import create_dataloaders
        from pinnacle.trainer import PINNACLETrainer

        train_loader, val_loader, test_loader, y_train = create_dataloaders(
            data,
            batch_size=config["batch_size"],
            seed=config["seed"],
            num_workers=config.get("num_workers", 0),
            use_augmentation=config.get("use_augmentation", True),
        )

        model = PINNACLE(
            num_classes=config["num_classes"],
            embed_dim=config["embed_dim"],
            dropout=config.get("dropout", 0.3),
            use_fusion=config.get("use_fusion", True),
        )

        trainer = PINNACLETrainer(
            model=model,
            device=device,
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
            epochs=config["epochs"],
            log_dir=config.get("log_dir", "outputs/logs"),
            checkpoint_dir=config.get("checkpoint_dir", "outputs/checkpoints"),
            patience=config.get("early_stopping_patience", 12),
            ema_decay=config.get("ema_decay", 0.999),
        )

        results = trainer.train(train_loader, val_loader, test_loader)

        # Save training curves
        from pinnacle.visualize import plot_training_curves
        plot_training_curves(
            results["history"],
            output_path=os.path.join(figure_dir, "fig4_training_curves.png"),
        )

    # ================================================================
    # STEP 5: Evaluation
    # ================================================================
    logger.info("\n🧪 Step 5: Final Evaluation")
    logger.info("-" * 50)

    import torch
    from pinnacle.model import PINNACLE
    from pinnacle.dataset import create_dataloaders
    from pinnacle.evaluate import evaluate_model
    from pinnacle.visualize import plot_confusion_matrix, plot_perclass_metrics

    # Reload best model
    best_path = os.path.join(config.get("checkpoint_dir", "outputs/checkpoints"), "best_model.pth")
    if os.path.exists(best_path):
        model = PINNACLE(
            num_classes=config["num_classes"],
            embed_dim=config["embed_dim"],
            dropout=config.get("dropout", 0.3),
            use_fusion=config.get("use_fusion", True),
        ).to(device)

        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        logger.info(f"  ✅ Loaded best model (epoch {state['epoch']})")

        _, _, test_loader, _ = create_dataloaders(
            data,
            batch_size=config["batch_size"],
            seed=config["seed"],
            num_workers=config.get("num_workers", 0),
            use_augmentation=False,
        )

        eval_results = evaluate_model(model, test_loader, device)

        plot_confusion_matrix(
            eval_results["confusion_matrix"],
            output_path=os.path.join(figure_dir, "fig5_confusion_matrix.png"),
        )
        plot_perclass_metrics(
            eval_results["per_class"],
            output_path=os.path.join(figure_dir, "fig6_perclass_metrics.png"),
        )
    else:
        logger.warning(f"  ⚠️  No checkpoint found at {best_path}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ PINNACLE pipeline complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
