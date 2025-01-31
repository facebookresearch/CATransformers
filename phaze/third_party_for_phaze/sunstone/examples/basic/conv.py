import time
from src.tiling import GenericTile, GenericProblem
from src.generic_optimizers import *
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src")

# architecture parameters
# sizes are in terms of total data capacity
L1_SIZE = 48
# energy (whatever unit as long as used consistently)
L1_E = 0.45

L2_SIZE = 55296
L2_E = 17.57

# PE array configuration
PEX = 16
PEY = 16
# Read / Write BW
NOC_BW = (16, 16)

start = time.time()
# First, create the problem
# This example describes conv-1 of ResNet (batch=1)
# Since we are doing single batch, "N" is not needed, and mapper could be a bit # faster (~5%)
bounds = {"P": 56, "Q": 56, "R": 3, "S": 3, "C": 64, "K": 64, "N": 1}

# We also need a problem description
# This is the one for conv (it is similar to how timeloop describes problems)
# The first entry describes how the input is indexed (inp[P+R][Q+S][C][N])
# The next entry describes how the weight is indexed (w[R][S][C][K])
# The last entry describes how the output is indexed (out[P][Q][K][N])
tens_desc = [[("P", "R"), ("Q", "S"), "C", "N"], [
    "R", "S", "C", "K"], ["P", "Q", "K", "N"]]
prob = GenericProblem(bounds, tens_desc)

# We can print out the problem to see the analysis done
# (formatting is still WIP)
print("\n\nProblem analysis")
print(prob)

# Now initialize the Mapper
# Since we are optimizing the lowest level first, we have no tiles or ordering
# to work off of so we first create a tile wtith "0-levels", and set the
# ordering to None
org_prob = [(GenericTile(prob), None)]

# Now we call the actual enumerate+ranking function to get the L1 tiles
# The first arg is the L1 size
# The second arg is a list of access energies, for each level, for each tensor
# In this case, we will have 2 levels, so we would want 2 entries in the
# access energy list, where each entry is a 3-entry tuple (one for each tensor)
# x_axis and y_axis are optional to include spatial unrolling after L1
# static arg indicates to use static bounds (Sunstone principle for tiling
# that only grows in certain dimensions depending on the order)
access_energies = ((L1_E, L1_E, L1_E),
                   (L2_E, L2_E, L2_E))

ret = tls_sptl_tmprl(
    tiles=org_prob,
    prob=prob,
    mem_size=L1_SIZE,
    access_energies=access_energies,
    x_axis=PEX,
    y_axis=PEY,
    static=True,
    bw=[None, NOC_BW],
    threads=8,
    edp=True
)

# ret[0] now contains a list of (GenericTile, ordering) tuples, where
# GenericTile now has L1+spatial tiling, and ordering is a list of strings to
# describe the best ordering at each temporal level (in this case, there is one
# ordering). ret[0] can now be passed to the next Mapping class, and each
# promising tile+ordering at the L1 level can now be used as starting point to # find the L2 tiling
# ret[1] contains the number of tiles evaluated (this is not mappings evaluated
# since for each tile, we evaluate all promising orderings)

# Now for the L2 mapper, we can pass ret[0] directly in, as well as the generic
# problem

# After initialization, we call a solve function again
# The interface is the same as the one with "with_ord", with the first arg
# being the L2 size, and second arg being a list of access energies. Since we
# will have 3 levels, we need to pass the DRAM energy as well (again, one value
# per tensor)
# prior is a optional argmuent for upper factors
access_energies = ((L1_E, L1_E, L1_E),
                   (L2_E, L2_E, L2_E),
                   (128, 128, 128))

# we also get an upper estimate on the cost for each candidate which will
# help in alpha-beta pruning
costs = ret[1]

ret = tls_tmprl_alpha_beta_mlt_thrd(
    tiles=ret[0],
    prob=prob,
    mem_size=L2_SIZE,
    access_energies=access_energies,
    costs=[cost[0] for cost in costs],
    x_axis=None,
    y_axis=None,
    static=False,
    prior=False,
    bw=[None, NOC_BW, None],
    threads=8,
    bypass=None
)

# like get_outin, solve_alpha_beta_parallel returns a list of (GenericTile, ordering) tuples
# but only return the best mapping. The ordering will now
# contain 2 strings, one for each memory interaction.
dur = time.time() - start
tile, order = ret[0], ret[1]

# we can print out tile and order
print("\n\nPrinting the resulting tiles and orders in the raw form")
print(tile, order)

# If we define a list with every memory level, indicating whether each one has
# split buffers or not, as well as a list with the names of the tensor
# (according to the definition order in the tensor description), we can
# pass those as well as the ordering to yaml function of the tile to get a
# dictionary ready to be dumped onto a yaml file that will be compatible with
# Timeloop
# Every memory level is unified in this example
arch = [('L1', False), ('L2', False), ('DRAM', False)]
tens = ["Inputs", "Weights", "Outputs"]
yaml_dict = tile.yaml(order, arch, tens)

print("\n\nPrinting the formatted result")
print(yaml_dict)

print("\n Optimization time:")
print(f"Time elapsed: {dur:.4f}")
