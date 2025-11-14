import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from huggingface_hub import snapshot_download


dataset = snapshot_download(
    "GSAI-ML/LLaDA-1.5",
    local_dir="./checkpoint/model_checkpoint")
