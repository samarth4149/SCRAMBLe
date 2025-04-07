CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

from pathlib import Path
# HF_CACHE_DIR = "./hf_cache/" # TODO: change this to your own cache dir
HF_CACHE_DIR = str(Path("~/.cache/huggingface/hub/").expanduser()) 

# NOTE : this is according to conv_vicuna_v1 from conversation.py
SYSTEM_MSG = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
CONTEXT_LEN = 2048