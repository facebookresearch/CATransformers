"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
"""
Implementation of the following modules is borrowed from from https://github.com/msr-fiddle/phaze
Licensed under MIT License
"""

import sys

from examples.simba_like.cnn_optimizer_wrapper import CNNOptimizer  # noqa

import time


def tensor_core_estimate(setupconfig, op, energy):

    start = time.time()

    memrs_flat = {
        "reg": setupconfig["L1_Buffer_TC"],
        "L1_ifmap": setupconfig["L2_Buffer"],
        "L1_w": setupconfig["L2_Buffer"],
        "L1_ofmap": setupconfig["L2_Buffer"],
        "L2": setupconfig["GLB_Buffer"],
    }

    # ******* this is the memory hierarchy format that our wrapper accepts
    #         (look into cnn_optimizer_wrapper)                            *******
    acc_memrs = {
        "reg": memrs_flat["reg"],
        "L1": (memrs_flat["L1_ifmap"], memrs_flat["L1_w"], memrs_flat["L1_ofmap"]),
        "L2": memrs_flat["L2"]
    }

    # ****** Access energies that can come from Accelergy
    #        (CACTI + ALADDIN) or any other source        ******
    acc_acs_enrgs = {
        # Register inside vector MAC
        "reg_ifmap": energy["tc_l1_read"],
        "reg_w": energy["tc_l1_read"],
        "reg_ofmap": energy["tc_l1_read"],
        # PE SRAM
        "L1_ifmap": energy["tc_l2_read"],
        "L1_w": energy["tc_l2_read"],
        "L1_ofmap": energy["tc_l2_read"],
        "L1_NoC": energy["tc_l2_read"],
        # Global SRAM
        "L2_ifmap": energy["tc_glb_read"],
        "L2_w": energy["tc_glb_read"],
        "L2_ofmap": energy["tc_glb_read"],
        # Global SRAM
        "DR_ifmap": (energy["dram_read"] + energy["dram_write"]) / 2,
        "DR_w": (energy["dram_read"] + energy["dram_write"]) / 2,
        "DR_ofmap": (energy["dram_read"] + energy["dram_write"]) / 2,
    }

    # ******* dimensions of the vector MAC *******
    pe_spatial = {"X": setupconfig["PE_x"], "Y": setupconfig["PE_y"]}

    # ******* dimensions of the PE array *******
    cores_sptl = {"X": setupconfig["Core_x"], "Y": setupconfig["Core_y"]}

    # BW (words/cycle)
    smba_bws = [None, (setupconfig["L2_BW"],setupconfig["L2_BW"]), (setupconfig["GLB_BUFFER_BW"], setupconfig["GLB_BUFFER_BW"]), (128, 128)]

    # ******* this list indicate whether each operand has
    #         a dedicated buffer at each level & will be
    #         used to optimize for uneven mapping which Timeloop
    #         also supports. If uneven mapping will not be used,
    #         set this to None                                   *******
    sprt_strct = [(False, True, False), (True, True, True),
                  (False, False, False)]

    # ******* the tensor to keep inside vector MAC registers
    #         here we by-pass both IFMAP and OFMAP and only keep weights
    mem_bypass = {
        "reg": (True, False, True),
        "L1": (False, False, False),
        "L2": (False, False, False)}

    (
        b,
        n,
        m,
        c,
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    ) = op

    '''dim = {
        "N": n, #B
        "C": c, #N
        "M": m, #C
        "R": r, #r
        "S": s, #s
        "P": p, #p
        "Q": q, #q
    }'''

    dim = {
        "N": n * b,  # batch size
        "C": c,
        "K": m,
        "R": r,  # r
        "S": s,  # s
        "P": p,  # p
        "Q": q,  # q
    }

    opt = CNNOptimizer(
        dim,
        acc_memrs,
        acc_acs_enrgs,
        smba_bws,
        cores_sptl,
        pe_spatial,
        mem_bypass,
        sprt_strct,
        memrs_flat
    )

    try:
        fnl_res, full_cost = opt.solve(threads=8, optimize_uneven_mapping=False)
        tls, unevn_mp, ords, cost = fnl_res
        end = time.time()
        return (full_cost[1], full_cost[2], end - start)

    except:
        return None
