"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

# HW parameters (PHAZE)
FREQUENCY= 625 * (1000**2) 
PRECISION = 8
TECHNOLOGY=22


# Search params
NUM_TRIALS = 100

# Model Architecture
MODEL_ARCH= 'ViT-B-16'
# MODEL_ARCH= 'ViT-B-32'

PRETRAINED='datacomp_xl_s13b_b90k' # "Model checkpoint name to use from OpenCLIP"

# Note: FFN and hidden dimension (embedding) are divided into 8 blocks

# TEXT encoder
TEXT_MODEL_PARAMS = {
'MAX_LAYERS':12,
'MIN_LAYERS': 6,
'MAX_FFN_BLOCK':8,
'MIN_FFN_BLOCK': 4,
'MAX_EMB_BLOCK' : 8,
'MIN_EMB_BLOCK' : 4,
'MAX_ATTN_HEAD' : 8,
'MIN_ATTN_HEAD' : 4,
}

# Vision
VISION_MODEL_PARAMS = {
'MAX_VISION_LAYERS' : 12,
'MIN_VISION_LAYERS' : 6,
'MAX_VISION_FFN_BLOCK' : 8,
'MIN_VISION_FFN_BLOCK' : 4,
'MAX_VISION_EMB_BLOCK' : 8,
'MIN_VISION_EMB_BLOCK' : 4,
'MAX_VISION_ATTN_HEAD' : 12,
'MIN_VISION_ATTN_HEAD' : 6,
}

# HW Search parameters
HW_PARAMS={
'CLUSTER_NUM' : [1, 2, 4],
'WIDTH' : [32, 64, 128, 256],
'DEPTH' : [2, 4, 8, 16, 32, 64],
'L2_SRAM' : [64, 128, 256, 512, 1024], # in KB (actual SMRA is *4 this value IF OF maps)
'L2_BW' : [32, 64, 128],
'GLB_BUFFER' :[2, 4, 8], # MB
}

# constraints
LATENCY_CONSTRAINT = "latency <= 0.05" # seconds (50ms)
AREA_CONSTRAINT = "area <= 31990347" # um^2
MAX_TOPS_CONSTRAINT = "tops <= 20.48"
AREA_CONSTRAINT_VALUE = 31990347
LATENCY_CONSTRAINT_VALUE = 0.05
MAX_TOPS = 20 * (1.024)


# Source
# OPERATIONAL_CARBON_INTENSITY = 224 # california
OPERATIONAL_CARBON_INTENSITY = 68 # BC
# OPERATIONAL_CARBON_INTENSITY = 524 # Taiwan

# embodied carbon 
CARBON_INTENSITY_LOC = "loc_taiwan"
# CARBON_INTENSITY_LOC = "loc_usa"
# CARBON_INTENSITY_LOC = "loc_iceland"

