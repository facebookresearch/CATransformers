# HARDWARE DESCRIPTION#
# ==================================================================#
# PRECISION#
from configurations import PRECISION
PRECISION = PRECISION

# FREQUENCY#
# FREQUENCY = 700 * (1024**2)  # TPU v2 700 MHz

# DRAM#
DRAM_BW = 700 * (1024**3)  # TPU v2 700 GB/s
DRAM_BLOCK_SIZE = 4
DRAM_RD_CYCLES = 500

# DRAM BW ELEMENTS PER CYCLE#
# DRAM_BW_ELM_CYCLE = int(DRAM_BW / FREQUENCY) / (PRECISION / 8)

# BROADCAST NOC BW#
BCAST_NOC_BW = 32  # 32B/cycle
BCAST_NOC_CYCLES = 1

# GLOBAL BUFFER
GLB_BUFFER_BLOCK_SIZE = 4
GLB_BUFFER_NOC_BW = 4 * (1024) # Meta not used

# CORE#
L2_BUFFER_BLOCK_SIZE = 4
# L2_BUFFER_NOC_BW = 1 * (1024)  # 1kB/cycle  2kB/cycle # Meta not used
L1_BUFFER_BLOCK_SIZE = 1

# INTERCONNECTS#
# =================================================================#
INT_CON_BW = 1 * (1024 * 3)  # 1GB/s

# OPTIMIZATION METRIC#
# ==================================================================#
OPT_METRIC = ["delay", "energy", "edp"]

