import os
import shutil
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_colab_notebook():
    """Create a Colab notebook with training setup."""
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SRRGAN Training Setup\n",
    "\n",
    "This notebook sets up and runs the SRRGAN training on Google Colab."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Mount Google Drive\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# Install required packages\n",
    "!pip install torch torchvision tqdm pillow numpy albumentations\n",
    "\n",
    "# Clone repository\n",
    "!git clone https://github.com/your-username/final_yr_project.git\n",
    "%cd final_yr_project\n",
    "\n",
    "# Add project to Python path\n",
    "import sys\n",
    "sys.path.append('/content/final_yr_project')\n",
    "\n",
    "# Verify GPU\n",
    "import torch\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create necessary directories\n",
    "!mkdir -p data/train data/val models logs\n",
    "\n",
    "# Copy data from Drive to project directory\n",
    "!cp -r /content/drive/MyDrive/final_yr_project/data/* data/\n",
    "\n",
    "# Start training\n",
    "!python scripts/train.py --data_dir data --models_dir models --log_dir logs --batch_size 16"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "name": "SRRGAN_Training.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}"""
    
    with open('SRRGAN_Training.ipynb', 'w') as f:
        f.write(notebook_content)

def prepare_upload():
    """Prepare project for upload to Google Drive."""
    setup_logging()
    
    # Create a clean directory for upload
    upload_dir = Path('upload_to_drive')
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir()
    
    # Copy necessary files and directories
    dirs_to_copy = ['src', 'scripts', 'data', 'models', 'logs']
    for dir_name in dirs_to_copy:
        if Path(dir_name).exists():
            shutil.copytree(dir_name, upload_dir / dir_name)
    
    # Copy individual files
    files_to_copy = ['requirements.txt', 'README.md']
    for file_name in files_to_copy:
        if Path(file_name).exists():
            shutil.copy2(file_name, upload_dir / file_name)
    
    # Create Colab notebook
    create_colab_notebook()
    shutil.copy2('SRRGAN_Training.ipynb', upload_dir / 'SRRGAN_Training.ipynb')
    
    logging.info(f"Project prepared for upload in {upload_dir}")
    logging.info("Next steps:")
    logging.info("1. Upload the 'upload_to_drive' folder to Google Drive")
    logging.info("2. Open SRRGAN_Training.ipynb in Google Colab")
    logging.info("3. Run the cells in order")

if __name__ == "__main__":
    prepare_upload() 