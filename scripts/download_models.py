\
\
\
   
import os
import sys
import argparse
import logging
from huggingface_hub import snapshot_download
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


                      
MODELS = {
                                                  
    'llava-base': {
        'repo_id': 'llava-hf/llava-1.5-7b-hf',
        'size': '7B',
        'type': 'base',
        'description': 'LLaVA 1.5 7B base model (pre-trained, not fine-tuned on MLLMU)',
        'recommended_for': 'Starting point for vanilla model training'
    },
    'idefics2-base': {
        'repo_id': 'HuggingFaceM4/idefics2-8b',
        'size': '8B',
        'type': 'base',
        'description': 'Idefics2 8B instruction-tuned model',
        'recommended_for': 'Starting point for vanilla model training'
    },

                                                
    'llava-vanilla': {
        'repo_id': 'MLLMMU/LLaVA_Vanilla',
        'size': '7B',
        'type': 'vanilla',
        'description': 'LLaVA 1.5 7B fine-tuned on MLLMU-Bench full dataset',
        'recommended_for': 'POPS attack baseline comparison'
    },
    'idefics2-vanilla': {
        'repo_id': 'MLLMMU/Idefics2_Vanilla',
        'size': '8B',
        'type': 'vanilla',
        'description': 'Idefics2 8B fine-tuned on MLLMU-Bench full dataset',
        'recommended_for': 'POPS attack baseline comparison'
    },

                                                                           
}


def check_disk_space(required_gb=20):
\
\
\
\
\
\
\
\
       
    import shutil
    stats = shutil.disk_usage(".")
    free_gb = stats.free / (1024**3)

    logger.info(f"Available disk space: {free_gb:.2f} GB")

    if free_gb < required_gb:
        logger.warning(f"Low disk space! Required: {required_gb} GB, Available: {free_gb:.2f} GB")
        return False

    return True


def download_model(model_key, output_dir="models", use_cache=True):
\
\
\
\
\
\
\
\
\
\
       
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        logger.info(f"Available models: {list(MODELS.keys())}")
        return None

    model_info = MODELS[model_key]
    repo_id = model_info['repo_id']

    logger.info("=" * 80)
    logger.info(f"DOWNLOADING MODEL: {model_key}")
    logger.info("=" * 80)
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Size: {model_info['size']}")
    logger.info(f"Type: {model_info['type']}")
    logger.info(f"Description: {model_info['description']}")

                             
    model_type = model_info['type']
    model_dir = os.path.join(output_dir, model_type, model_key)
    os.makedirs(model_dir, exist_ok=True)

    try:
        logger.info(f"\nDownloading to: {model_dir}")
        logger.info("This may take a while depending on your internet connection...")

                        
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        logger.info(f"\n✓ Successfully downloaded {model_key}!")
        logger.info(f"Location: {os.path.abspath(model_dir)}")

                         
        import json
        info_file = os.path.join(model_dir, "model_info.json")
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        return model_dir

    except Exception as e:
        logger.error(f"\n✗ Error downloading {model_key}: {e}")
        logger.info("\nAlternative: You can manually download using:")
        logger.info(f"  from transformers import AutoModel")
        logger.info(f"  model = AutoModel.from_pretrained('{repo_id}')")
        return None


def download_with_transformers(model_key, output_dir="models"):
\
\
\
\
\
\
\
\
\
       
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        return None

    model_info = MODELS[model_key]
    repo_id = model_info['repo_id']

    logger.info("=" * 80)
    logger.info(f"DOWNLOADING MODEL WITH TRANSFORMERS: {model_key}")
    logger.info("=" * 80)

    model_type = model_info['type']
    model_dir = os.path.join(output_dir, model_type, model_key)
    os.makedirs(model_dir, exist_ok=True)

    try:
        from transformers import AutoProcessor, AutoTokenizer

        logger.info("Downloading processor...")
        processor = AutoProcessor.from_pretrained(repo_id)
        processor.save_pretrained(model_dir)

        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        tokenizer.save_pretrained(model_dir)

        logger.info("Downloading model weights...")

                                                
        if 'llava' in repo_id.lower():
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        elif 'idefics' in repo_id.lower():
            from transformers import Idefics2ForConditionalGeneration
            model = Idefics2ForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            logger.error(f"Unknown model type for {repo_id}")
            return None

        logger.info(f"Saving model to {model_dir}...")
        model.save_pretrained(model_dir)

        logger.info(f"\n✓ Successfully downloaded {model_key}!")
        logger.info(f"Location: {os.path.abspath(model_dir)}")

        return model_dir

    except Exception as e:
        logger.error(f"\n✗ Error downloading {model_key}: {e}")
        return None


def list_models():
                                    
    logger.info("=" * 80)
    logger.info("AVAILABLE MODELS")
    logger.info("=" * 80)

    for model_type in ['base', 'vanilla']:
        logger.info(f"\n{model_type.upper()} MODELS:")
        logger.info("-" * 80)

        for key, info in MODELS.items():
            if info['type'] == model_type:
                logger.info(f"\n  {key}:")
                logger.info(f"    Repository: {info['repo_id']}")
                logger.info(f"    Size: {info['size']}")
                logger.info(f"    Description: {info['description']}")
                logger.info(f"    Recommended for: {info['recommended_for']}")


def verify_model(model_dir):
\
\
\
\
\
\
\
\
       
    logger.info(f"\nVerifying model at: {model_dir}")

    required_files = ['config.json', 'model.safetensors.index.json']
    missing = []

    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            logger.info(f"  ✓ {file}")
        else:
                                           
            if 'safetensors' in file:
                                                            
                import glob
                safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
                if safetensors_files:
                    logger.info(f"  ✓ Found {len(safetensors_files)} safetensors file(s)")
                    continue

            logger.warning(f"  ✗ {file} (missing)")
            missing.append(file)

    if not missing:
        logger.info("\n✓ Model verification passed!")
        return True
    else:
        logger.warning(f"\n⚠ Model may be incomplete. Missing: {missing}")
        return False


def main():
                                 
    parser = argparse.ArgumentParser(
        description="Download models for POPS attack evaluation"
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model to download (see --list for options)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save models (default: models)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models'
    )
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify a downloaded model directory'
    )
    parser.add_argument(
        '--use_transformers',
        action='store_true',
        help='Use transformers library instead of huggingface_hub'
    )
    parser.add_argument(
        '--all_vanilla',
        action='store_true',
        help='Download all vanilla models'
    )
    parser.add_argument(
        '--all_base',
        action='store_true',
        help='Download all base models'
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.verify:
        verify_model(args.verify)
        return

    if args.all_vanilla:
        models_to_download = [k for k, v in MODELS.items() if v['type'] == 'vanilla']
    elif args.all_base:
        models_to_download = [k for k, v in MODELS.items() if v['type'] == 'base']
    elif args.model:
        models_to_download = [args.model]
    else:
        logger.error("Please specify --model, --all_vanilla, --all_base, or --list")
        parser.print_help()
        return

                      
    required_space = len(models_to_download) * 15                   
    if not check_disk_space(required_space):
        response = input(f"\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            return

                     
    for model_key in models_to_download:
        if args.use_transformers:
            model_dir = download_with_transformers(model_key, args.output_dir)
        else:
            model_dir = download_model(model_key, args.output_dir)

        if model_dir:
            verify_model(model_dir)

    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nModels saved to: {os.path.abspath(args.output_dir)}")
    logger.info("\nNext steps:")
    logger.info("  1. Use vanilla models as baseline for POPS attack")
    logger.info("  2. Or train your own vanilla model with finetune.py")
    logger.info("  3. Train unlearned baselines (see baselines/README.md)")
    logger.info("  4. Run POPS attack with attack_eval.py")


if __name__ == "__main__":
    main()
