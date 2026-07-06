\
\
   
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_mllmu_bench(data_dir="data"):
\
\
\
\
\
       
                           
    os.makedirs(data_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "MLLMU-Bench")

    logger.info("=" * 80)
    logger.info("DOWNLOADING MLLMU-BENCH DATASET")
    logger.info("=" * 80)
    logger.info(f"Download location: {dataset_path}")

    try:
                                               
        logger.info("Downloading from HuggingFace: MLLMMU/MLLMU-Bench")
        logger.info("This may take a while (dataset is ~2-3 GB)...")

        snapshot_download(
            repo_id="MLLMMU/MLLMU-Bench",
            repo_type="dataset",
            local_dir=dataset_path,
            local_dir_use_symlinks=False
        )

        logger.info("✓ Dataset downloaded successfully!")

    except Exception as e:
        logger.error(f"✗ Error downloading dataset: {e}")
        logger.info("\nAlternative: You can manually download using git-lfs:")
        logger.info("  cd data")
        logger.info("  git lfs install")
        logger.info("  git clone https://huggingface.co/datasets/MLLMMU/MLLMU-Bench")
        return False

                              
    logger.info("\n" + "=" * 80)
    logger.info("VERIFYING DATASET STRUCTURE")
    logger.info("=" * 80)

    required_paths = [
        "Full_Set/train-00000-of-00001.parquet",
        "Test_Set",
        "Retain_Set/train-00000-of-00001.parquet",
        "ft_Data/train-00000-of-00001.parquet"
    ]

    missing = []
    for path in required_paths:
        full_path = os.path.join(dataset_path, path)
        if os.path.exists(full_path):
            logger.info(f"✓ Found: {path}")
        else:
            logger.warning(f"✗ Missing: {path}")
            missing.append(path)

                                    
    logger.info("\nChecking for forget/retain splits:")
    for ratio in [5, 10, 15, 20]:
        forget_path = os.path.join(dataset_path, f"forget_{ratio}")
        retain_path = os.path.join(dataset_path, f"retain_{100-ratio}")

        if os.path.exists(forget_path):
            logger.info(f"✓ Found: forget_{ratio}/")
        else:
            logger.warning(f"✗ Missing: forget_{ratio}/")

        if os.path.exists(retain_path):
            logger.info(f"✓ Found: retain_{100-ratio}/")
        else:
            logger.warning(f"✗ Missing: retain_{100-ratio}/")

    if missing:
        logger.warning("\n⚠ Some files are missing. Dataset may be incomplete.")
        logger.info("\nIf forget/retain splits are missing, you can download them separately:")
        logger.info("  https://huggingface.co/MLLMMU/baseline_train_split")
        return False

    logger.info("\n" + "=" * 80)
    logger.info("✓ DATASET READY FOR POPS ATTACK!")
    logger.info("=" * 80)
    logger.info(f"\nDataset location: {os.path.abspath(dataset_path)}")

    return True


def download_baseline_splits(data_dir="data"):
\
\
\
\
\
       
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOADING BASELINE TRAINING SPLITS")
    logger.info("=" * 80)

    splits_path = os.path.join(data_dir, "baseline_train_split")

    try:
        logger.info("Downloading from HuggingFace: MLLMMU/baseline_train_split")

        snapshot_download(
            repo_id="MLLMMU/baseline_train_split",
            repo_type="model",                               
            local_dir=splits_path,
            local_dir_use_symlinks=False
        )

        logger.info("✓ Baseline splits downloaded successfully!")

                                                   
        import shutil
        mllmu_path = os.path.join(data_dir, "MLLMU-Bench")

        logger.info("\nCopying splits to MLLMU-Bench directory...")
        for item in os.listdir(splits_path):
            if item.startswith("forget_") or item.startswith("retain_"):
                src = os.path.join(splits_path, item)
                dst = os.path.join(mllmu_path, item)
                if os.path.isdir(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    logger.info(f"✓ Copied: {item}/")

        logger.info("✓ Baseline splits integrated successfully!")
        return True

    except Exception as e:
        logger.error(f"✗ Error downloading baseline splits: {e}")
        return False


def print_dataset_summary(data_dir="data"):
                                              
    dataset_path = os.path.join(data_dir, "MLLMU-Bench")

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info("\n" + "=" * 80)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 80)

                         
    import glob
    parquet_files = glob.glob(os.path.join(dataset_path, "**/*.parquet"), recursive=True)
    logger.info(f"\nTotal parquet files: {len(parquet_files)}")

                          
    logger.info("\nMain components:")
    for item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            logger.info(f"  📁 {item}/")
        else:
            logger.info(f"  📄 {item}")

                    
    logger.info("\nAvailable forget ratios:")
    for ratio in [5, 10, 15, 20]:
        forget_path = os.path.join(dataset_path, f"forget_{ratio}")
        if os.path.exists(forget_path):
            logger.info(f"  ✓ {ratio}% forget set")

    logger.info("\n" + "=" * 80)


def main():
                                 
    import argparse

    parser = argparse.ArgumentParser(
        description="Download MLLMU-Bench dataset for POPS attack"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory to save dataset (default: data)'
    )
    parser.add_argument(
        '--download_splits',
        action='store_true',
        help='Also download pre-split baseline training data'
    )
    parser.add_argument(
        '--summary_only',
        action='store_true',
        help='Only print dataset summary (skip download)'
    )

    args = parser.parse_args()

    if args.summary_only:
        print_dataset_summary(args.data_dir)
        return

                           
    success = download_mllmu_bench(args.data_dir)

                                           
    if args.download_splits and success:
        download_baseline_splits(args.data_dir)

                   
    if success:
        print_dataset_summary(args.data_dir)

        logger.info("\n" + "=" * 80)
        logger.info("NEXT STEPS")
        logger.info("=" * 80)
        logger.info("\n1. Download a pre-trained vanilla model (optional):")
        logger.info("   python scripts/download_models.py --model vanilla")
        logger.info("\n2. Or train your own vanilla model:")
        logger.info("   python finetune.py --model_id llava-hf/llava-1.5-7b-hf \\")
        logger.info("      --data_dir data/MLLMU-Bench/ft_Data/train-00000-of-00001.parquet \\")
        logger.info("      --save_dir models/vanilla/llava-1.5-7b")
        logger.info("\n3. Run POPS attack:")
        logger.info("   See ATTACK_README.md for detailed instructions")


if __name__ == "__main__":
    main()
