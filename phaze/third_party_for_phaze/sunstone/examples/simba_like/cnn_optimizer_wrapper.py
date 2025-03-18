"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from src.generic_optimizers import *
import sys
import os
sys.path.append(os.getcwd())

opt_metric_to_idx = {"edp": 0, "latency": 2, "energy": 3}

# ****************************** NN wrapper for optimizer ******************************
"""
    - Neural network wrapper as an example of how to use the optimizer
    - Args:
            1- bounds: problem dimensions and value pairs (dict: N, C, K, ...)
            2- memrs: level and memory capacity at each level pairs (dict: L1, L2, ...)
            3- acc_enrgs: access energy of each component at each level (dict: dicstL1_w, L1_ifmap, ..., L2_NoC)
            4- bws: list of pair of bw for each level. is pair is (read, write)
            5- sptl: pair of x and y spatial PE size (dict: X, Y)
"""


class CNNOptimizer:

    tens_desc = [
        [("P", "R"), ("Q", "S"), "C", "N"],
        ["R", "S", "C", "K"],
        ["P", "Q", "K", "N"],
    ]

    def __init__(self, bounds, memrs, acs_enrgs, bws, sptl, PE_sptl, bypass_info, sprt_strct, memrs_flat):
        # create a generic problem from current problem
        self.bounds = bounds
        self.problem_generic = GenericProblem(self.bounds, self.tens_desc)

        # accelerator specifics
        self.memrs = memrs
        self.memrs_flat = memrs_flat
        self.acs_enrgs = acs_enrgs
        self.bws = bws
        self.sptl = sptl
        self.PE_sptl = PE_sptl
        self.bypass_info = bypass_info
        self.sprt_strct = sprt_strct

    def uneven_tuner(self, fnl_res):
        memrs_flat = self.memrs_flat
        sprt_strct = self.sprt_strct
        gnrc_tl = fnl_res[0]
        szs = genrc_tl_sz(gnrc_tl)
        pr_bfr_tls = dvd_mpng_amng_bfrs(gnrc_tl)
        #print("\n\n\n\nPrint buffer tiles before")
        # print(pr_bfr_tls)

        pr_bfr_tls, pr_bfr_unevn = fx_unevn_L0(
            pr_bfr_tls, sprt_strct, memrs_flat)
        #print("\n\n\n\nPrint buffer tiles after")
        # for i, lvl in enumerate(pr_bfr_tls):
        #    print("\n\nlevel", i, ":")
        #    print(pr_bfr_tls[i])
        #    print(pr_bfr_unevn[i])

        pr_bfr_tls, pr_bfr_unevn = fx_unevn_L1_sm_bfr(
            pr_bfr_tls, sprt_strct, memrs_flat, pr_bfr_unevn, gnrc_tl.subtiles
        )
        #print("\n\n\n\nPrint buffer tiles after")
        # for i, lvl in enumerate(pr_bfr_tls):
        #    print("\n\nlevel", i, ":")
        #    print(pr_bfr_tls[i])
        #    print(pr_bfr_unevn[i])

        pr_bfr_tls, pr_bfr_unevn = fx_unevn_L1(
            pr_bfr_tls, sprt_strct, memrs_flat, pr_bfr_unevn, gnrc_tl.subtiles
        )
        #print("\n\n\n\nPrint buffer tiles after")
        # for i, lvl in enumerate(pr_bfr_tls):
        #    print("\n\nlevel", i, ":")
        #    print(pr_bfr_tls[i])
        #    print(pr_bfr_unevn[i])

        tmprl_i = -1
        for lvl_i, lvl in enumerate(fnl_res[0].subtiles):
            if fnl_res[0].spatial[lvl_i]:
                continue
            tmprl_i += 1
            for dim in lvl:
                fnl_res[0].subtiles[lvl_i][dim] = pr_bfr_tls[tmprl_i][0][dim]
        return [fnl_res[0], pr_bfr_unevn] + list(fnl_res[1:])

    def solve(self, threads=1, optimize_uneven_mapping=True, opt_metric="edp"):
        memrs = self.memrs
        # ******************* L1 vector MAC Optimization *******************
        access_e = [
            [
                self.acs_enrgs["reg_ifmap"],
                self.acs_enrgs["reg_w"],
                self.acs_enrgs["reg_ofmap"],
            ],
            [
                self.acs_enrgs["L1_ifmap"],
                self.acs_enrgs["L1_w"],
                self.acs_enrgs["L1_ofmap"],
            ],
        ]

        # ***** The W registers per MAC are indicated as the by-pass
        #       list below (which means only W elements are not
        #       by-passed), as well as, the memory size of 1 indicated
        #       in the tls_sptl_tmprl() method                          *****

        # Per element by-pass information of this level
        #          IFMAP    W   OFMAP
        bypass = [self.bypass_info["reg"]]
        org_prob = [(GenericTile(self.problem_generic), None)]

        ret, costs = tls_sptl_tmprl(
            tiles=org_prob,
            prob=self.problem_generic,
            # combined with the by-pass list means capacity is only 1 weight element
            mem_size=memrs["reg"],
            access_energies=access_e,
            edp=True,
            x_axis=self.PE_sptl["X"],
            y_axis=self.PE_sptl["Y"],
            static=True,  # use Sunstone principles to explore less tile
            bw=self.bws[:2],
            threads=threads,
            bypass=bypass,
            # sptl_cnstrnts=[{"K":8}, None],  # this constraints on spatial unrolling onto vector MACs is needed because
            # in Simba IFMAP is broadcasted to vector MACs
            # feel free to remove it if exact Simba architecture is not what you want
        )

        # ******************* L2 PE buffers Optimization *******************
        access_e.append(
            [
                self.acs_enrgs["L2_ifmap"],
                self.acs_enrgs["L2_w"],
                self.acs_enrgs["L2_ofmap"],
            ]
        )
        bypass.append(self.bypass_info["L1"])
        L2_cndds, costs = tls_sptl_tmprl_alpha_beta(
            tiles=ret,
            prob=self.problem_generic,
            mem_size=memrs["L1"],
            access_energies=access_e,
            bypass=bypass,
            x_axis=self.sptl["X"],
            y_axis=self.sptl["Y"],
            costs=costs,
            threads=threads
        )

        # ******************* L3 global buffers Optimization *******************
        access_e.append(
            [
                self.acs_enrgs["DR_ifmap"],
                self.acs_enrgs["DR_w"],
                self.acs_enrgs["DR_ofmap"],
            ]
        )
        bypass.append(self.bypass_info["L2"])

        bst_tl, bst_ordr, bst_cst, best_full_cst = tls_tmprl_alpha_beta_mlt_thrd(
            tiles=L2_cndds,
            prob=self.problem_generic,
            mem_size=memrs["L2"],
            access_energies=access_e,
            costs=[cost[0] for cost in costs],
            x_axis=None,
            y_axis=None,
            static=False,
            prior=False,
            bw=self.bws,
            threads=threads,
            bypass=bypass,
        )

        fnl_res = (bst_tl, bst_ordr, bst_cst)

        if optimize_uneven_mapping:
            fnl_res = self.uneven_tuner(fnl_res)
        else:
            fnl_res = [fnl_res[0], None] + list(fnl_res[1:])

        tiles, unevn_mp, orders, cost = fnl_res

        arch = [('Reg', True), ('L1', True), ('L2', False), ('DRAM', False)]
        tens = ["Inputs", "Weights", "Outputs"]

        self.optimal_mapping = tiles.yaml(orders, arch, tens, uneven=unevn_mp)

        # to pring the mapping level-by-level
        # print("\n")
        # for x in optimal_mapping['mapping']:
        #    print(x)

        return fnl_res, best_full_cst
