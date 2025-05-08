# Sunstone Examples

## Overview

In the Sunstone scheduler, optimization occurs level-by-level using the methods provided in the `generic_optimizers.py` file located in `./../src`. Typically, users will create a wrapper for the specific architecture they are optimizing for.

For example, in `eyeriss_like/cnn_optimizer_wrapper.py`, we first call the `tls_sptl_tmprl()` method to get the spatial unrollings and tile candidates. This is because in Eyeriss, the L1 buffers are distributed among several PEs, so we need both spatial unrolling and temporal tiling.

For each unrolling onto the PE array and tiling in the L1 buffers pair, we need to find the best L2 tiling candidates. To do this, we would call `tls_tmprl_alpha_beta_mlt_thrd()`, which only searches for temporal tilings and uses alpha-beta pruning to skip candidates that would lead to suboptimal mappings based on an upper-bound estimation of the cost.


## Weight Stationary

Users can fully or partially constrain the spatial unrolling at a specific level. This is useful for modeling some common accelerators, such as those with the famous `C-K` unrolling, which is also called **weight stationary** in the architecture community. Additionally, note that Sunstone is highly customizable, and users can easily customize several aspects of the optimization. Looking into the `sptl_cnstrnts()` and learning how the spatial unrolling constraints are enforced (which is trivially done by filtering a list of unrollings) can be a good starting point for learning how to customize Sunstone's optimization process.
