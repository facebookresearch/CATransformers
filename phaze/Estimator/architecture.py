"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

# internal imports
from .utils import get_core_area, get_core_energy

from collections import namedtuple
from math import log2

# python imports
import os
import sys
import json
from configurations import FREQUENCY, HW_PARAMS

# Global Variables for Architecture # Meta - not used 
bandwidth = 4 * 1024 * 1024 * 1024  # 32 Gbs or 4 GBs, PCIE 3.0
num_accelerators = 1  # accelerators
bytes_per_element = 1  # 16 bits

# Meta - Turing and Edge tpu @ 500 MHz
frequency = FREQUENCY

# types of cores in the architectures
cores = ["TC", "VC"]

# config per core, for VC depth is always 1
per_core_config = namedtuple(
    "per_core_config", ["num", "width", "depth", "GLB_Buffer", "L2_Buffer", "L2_BW"])

# accelerator config tuple
acc_config = namedtuple(
    "acc_config", ["num_tc", "num_vc", "width", "depth", "width_vc", "GLB_Buffer", "L2_Buffer","L2_BW", "area"])

# The maximum accelerator config might not be able to fit the max of each of the above aspects of the core
max_acc_config_per_dim = acc_config(
    4, 4, 256, 256, 256, 8*1024*1024, 1*1024*1024, 128, -1)

# maximum accelerator config for area constraint
max_acc_config = acc_config(4, 4, 256, 64, 256, 8*1024*1024, 1*1024*1024, 128, -1)
area_constraint = -1

# potential accelerator configs to explore
all_possible_acc_configs = []
acc_configs_to_explore = []
tc_configs = []
vc_configs = []

# only largest area
only_explore_largest_area = False

# only specific configs
only_explore_specific_configs = False


def create_core_config(cc, core_type="TC"):
    if core_type not in cores:
        raise TypeError("Core type not in cores supported.", core_type)

    if core_type == "TC":
        return per_core_config(cc.num_tc, cc.width, cc.depth, cc.GLB_Buffer, cc.L2_Buffer, cc.L2_BW)
    elif core_type == "VC":
        # fix VC L2 at 4KB
        return per_core_config(cc.num_vc, cc.width_vc, 1, cc.GLB_Buffer, cc.L2_Buffer, cc.L2_BW)


def generate_area_of_acc(acc_config):
    tc_coreconfig = create_core_config(acc_config, "TC")
    area, glb_area = get_core_area(tc_coreconfig, "TC")

    vc_coreconfig = create_core_config(acc_config, "VC")
    core_area_vc, glb_area_vc = get_core_area(vc_coreconfig, "VC")

    glb_area = max(glb_area, glb_area_vc)
    area += core_area_vc + glb_area
    return area


def get_area_constraint():
    global area_constraint
    global max_acc_config

    if area_constraint == -1:
        area_constraint = generate_area_of_acc(max_acc_config)

    max_acc_config = max_acc_config._replace(area=area_constraint)
    return area_constraint


def generate_all_cores_to_explore():
    # ["num_tc", "num_vc", "width", "depth", "width_vc", "GLB_Buffer", "L2_Buffer", "L2_BW", "area"]

    global all_possible_acc_configs

    def check_if_acc_to_explore(config):
        area_factor = 0.0 if config.num_tc == 1 or config.num_vc == 1 else 0.0 # meta
        config = config._replace(area=generate_area_of_acc(config))

        max_area = get_area_constraint()

        if (area_factor * max_area) <= config.area <= max_area:
            all_possible_acc_configs.append(config)
            return True
        return False
    

# Meta - read search params from configuration file 
    width = HW_PARAMS['WIDTH']
    depth= HW_PARAMS['DEPTH']
    l2_sram_choices_KB = HW_PARAMS['L2_SRAM']
    l2_bw = HW_PARAMS['L2_BW']
    glb_buffer_MB = HW_PARAMS["GLB_BUFFER"]
    core_cluster = HW_PARAMS["CLUSTER_NUM"]

# Meta
    for x in core_cluster:
            for pe_x in width:
                for pe_y in depth:
                    print(pe_x, pe_y)
                    for glb in glb_buffer_MB:
                        for l2_sram in l2_sram_choices_KB:
                                for bw in l2_bw:
                                    config = acc_config(x, x,
                                                    pe_x, pe_y, pe_x, (glb)*1024*1024, l2_sram* 1024,bw, -1)
                                    check_if_acc_to_explore(config)
    
    # meta - grid search
    # config = acc_config(4, 4, 256, 64, 256, (8)*1024*1024, 1*1024*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(1, 1, 256, 64, 256, (8)*1024*1024, 1*1024*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(2, 2, 256, 4, 256, (8)*1024*1024, 1*1024*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(1, 1, 256, 2, 256, (4)*1024*1024, 64*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(1, 1, 128, 16, 128, (4)*1024*1024, 1*1024*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(4, 4, 64, 64, 64, (4)*1024*1024, 1*1024*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(4, 4, 32, 32, 32, (4)*1024*1024, 1*1024*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(4, 4, 256, 16, 256, (4)*1024*1024, 1*1024*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(4, 4, 128, 16, 128, (4)*1024*1024, 1*1024*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(4, 4, 256, 16, 256, (4)*1024*1024, 64*1024, 64, -1)
    # check_if_acc_to_explore(config)
    # config = acc_config(4, 4, 256, 16, 256, (4)*1024*1024, 1*1024*1024, 128, -1)
    # check_if_acc_to_explore(config)



def generate_unique_core_configs():
    global tc_configs
    global vc_configs

    # Meta: regenerate since we will be calling estimate many times for different architecture
    # if (tc_configs and vc_configs): 
    #     return

    tc_configs.clear() # Meta: regenerate configs list 
    vc_configs.clear()

    for config in acc_configs_to_explore:
        tc_config = create_core_config(config, "TC")
        vc_config = create_core_config(config, "VC")

        if tc_config not in tc_configs:
            tc_configs.append(tc_config)
        if vc_config not in vc_configs:
            vc_configs.append(vc_config)


def get_configs_to_explore():
    if acc_configs_to_explore and vc_configs and tc_configs:
        return acc_configs_to_explore
    else:
        raise ValueError("No accelerator configs or core configs set")


def set_configs_to_explore(cc_from_args=None):
    curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    config_dir = os.path.join(curr_dir, "arch_configs")

    config_filename = "core_largest.json" if only_explore_specific_configs else "cores.json"
    config_file = os.path.join(config_dir, config_filename)

    acc_config_arg = namedtuple(
    "acc_config_arg", ["num_tc", "num_vc", "width", "depth", "width_vc", "GLB_Buffer", "L2_Buffer", "L2_BW"])

    global acc_configs_to_explore
    acc_configs_to_explore.clear()

    if os.path.exists(config_file):

        if cc_from_args != None: 
            with open(config_file, "r") as f:
                config_arg = acc_config_arg(**cc_from_args)
                configs = json.load(f)
                for cc in configs:
                    config = acc_config(**cc) 
                    if all(getattr(config, attr) == getattr(config_arg, attr) for attr in config_arg._fields):
                        acc_configs_to_explore = [config]
                        break
            f.close()
        else: 
            with open(config_file, "r") as f:
                configs = json.load(f)

                acc_configs_to_explore = [acc_config(**cc) for cc in configs]
            f.close()
    else:
        if only_explore_specific_configs:
            raise ValueError("No specific config file found.")

        generate_all_cores_to_explore()
        acc_configs_to_explore = all_possible_acc_configs
        with open(config_file, "w") as f:
            f.write("[")
            for idx, config in enumerate(acc_configs_to_explore):
                json.dump(config._asdict(), f,
                          ensure_ascii=False, indent=None)
                if (idx != len(acc_configs_to_explore) - 1):
                    f.write(",\n")
            f.write("]")
        f.close()

        if cc_from_args != None: 
            with open(config_file, "r") as f:
                config_arg = acc_config_arg(**cc_from_args)
                configs = json.load(f)
                for cc in configs:
                    config = acc_config(**cc) 
                    if all(getattr(config, attr) == getattr(config_arg, attr) for attr in config_arg._fields):
                        acc_configs_to_explore = [config]
                        break
            f.close()


    if only_explore_largest_area:
        max_area = max([x.area for x in acc_configs_to_explore])
        max_area_acc_configs_to_explore = []
        for cc in acc_configs_to_explore:
            if cc.area == max_area:
                max_area_acc_configs_to_explore.append(cc)
            elif cc.num_tc == 1 or cc.num_vc == 1:
                max_area_acc_configs_to_explore.append(cc)

        acc_configs_to_explore = max_area_acc_configs_to_explore

    generate_unique_core_configs()
    return acc_configs_to_explore
