"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
"""
Implementation of the following modules is borrowed from from https://github.com/mlfoundations/open_clip
Licensed under MIT License
"""
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

# Default name for a weights file hosted on the Huggingface Hub.
HF_WEIGHTS_NAME = "open_clip_pytorch_model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "open_clip_model.safetensors"  # safetensors version
HF_CONFIG_NAME = 'open_clip_config.json'
