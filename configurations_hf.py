# HW parameters (PHAZE)
FREQUENCY= 625 * (1000**2) 
PRECISION = 8
TECHNOLOGY=22


# Search params
NUM_TRIALS = 100

# Model Architecture
# MODEL_ARCH= 'vit-base-patch16'
# MODEL_ARCH= 'llama3'
MODEL_ARCH= 'bertbase'

# Note: FFN and hidden dimension (embedding) are divided into 16 blocks

# TEXT encoder bert, vit
TEXT_MODEL_PARAMS = {
'MAX_LAYERS':12,
'MIN_LAYERS': 6,
'MAX_FFN_BLOCK':16,
'MIN_FFN_BLOCK': 1,
'MAX_EMB_BLOCK' : 16,
'MIN_EMB_BLOCK' : 1,
'ATTN_HEAD' : [4,6,8,12],
}

# TEXT encoder llama3

# TEXT_MODEL_PARAMS = {
# 'MAX_LAYERS':4,
# 'MIN_LAYERS': 1,
# 'MAX_FFN_BLOCK':6,
# 'MIN_FFN_BLOCK': 1,
# 'MAX_EMB_BLOCK' : 6,
# 'MIN_EMB_BLOCK' : 1,
# 'ATTN_HEAD' : [32],
# }

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


# Operational carbon source (data from Electricity Maps 2024)
OPERATIONAL_CARBON_INTENSITY = 224 # california
# OPERATIONAL_CARBON_INTENSITY = 68 # BC
# OPERATIONAL_CARBON_INTENSITY = 524 # Taiwan

# embodied carbon location (based on ACT data)
CARBON_INTENSITY_LOC = "loc_taiwan"
# CARBON_INTENSITY_LOC = "loc_usa"
# CARBON_INTENSITY_LOC = "loc_iceland"

