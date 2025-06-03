#!/usr/bin/env python3
"""Download pre-trained models for NDML"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load model configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_model(model_name, model_path, auth_token=None):
    """Download a model from HuggingFace"""
    try:
        logger.info(f"Downloading {model_name}...")

        # Create model directory
        os.makedirs(model_path, exist_ok=True)

        # Download model
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            use_auth_token=auth_token,
            resume_download=True
        )

        logger.info(f"Successfully downloaded {model_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False


def verify_model(model_path):
    """Verify model was downloaded correctly"""
    required_files = ['config.json', 'pytorch_model.bin']

    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            # Check for sharded models
            if not any(f.startswith('pytorch_model-') for f in os.listdir(model_path)):
                return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Download NDML models')
    parser.add_argument('--config', default='config/models.yaml', help='Model configuration file')
    parser.add_argument('--auth-token', help='HuggingFace auth token')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    base_path = os.environ.get('NDML_MODEL_PATH', '/opt/ndml/models')

    # Download each model
    success_count = 0
    for model_config in config['models']:
        model_name = model_config['name']
        model_path = os.path.join(base_path, model_config['local_name'])

        # Skip if already downloaded
        if os.path.exists(model_path) and verify_model(model_path):
            logger.info(f"Model {model_name} already exists, skipping...")
            success_count += 1
            continue

        # Download model
        if download_model(model_name, model_path, args.auth_token):
            if verify_model(model_path):
                success_count += 1
            else:
                logger.error(f"Model {model_name} verification failed")

    logger.info(f"Downloaded {success_count}/{len(config['models'])} models successfully")


if __name__ == "__main__":
    main()