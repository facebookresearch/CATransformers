from .graph_wrapper import construct_external_scheduler, get_engine_type
from .tc_estimator import tensor_core_estimate

# Third party imports
from collections import namedtuple

# Python imports
from math import inf, log2, prod
import sys
import os
import time

from initialize import initialize as hardware_initializer
from initialize import get_core_energy as get_external_core_energy
from initialize import get_core_area as get_external_core_area
from initialize import run_area_energy_generation

from perf_wrappers.tensor_core_wrapper import get_dims_and_fused_perf_est as get_tc_dims
from perf_wrappers.vector_core_estimator import get_performance_est as external_vc_estimator


##############################################################################################
############################## Accelerator Setup and Initialization ##########################
##############################################################################################

acc_setup_status = {}
acc_setup_status["dir_setup"] = False
acc_setup_status["setup_dir_path"] = None
acc_setup_status["config_setup"] = False
acc_setup_status["curr_config"] = None

use_peak = False

phaze_coretype_mapping = {"Tensor Core": "TC", "Vector Core": "VC",
                          "Tensor Core + Vector Core": "TCandVC", "Nop": "Nop"}

e_tuple = namedtuple(
    "estimate", ["latency", "energy", "utilization", "estimation_time"])


def initialize_accelerator():
    def dir_exists_or_create(dirpath):
        isExist = os.path.exists(dirpath)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(dirpath)
            Warning("The", dirpath, "directory for architectural explorations!")

    curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    import time

    tmp_dir = os.path.join(os.path.dirname(curr_dir), "tmp")

    gemm_dir = os.path.join(tmp_dir, "GEMM")
    vc_dir = os.path.join(tmp_dir, "vector_core")
    estimates_tc_dir = os.path.join(gemm_dir, "arch_estimates")
    estimates_vc_dir = os.path.join(vc_dir, "arch_estimates")

    dir_exists_or_create(tmp_dir)
    dir_exists_or_create(vc_dir)
    dir_exists_or_create(gemm_dir)

    config_dir = os.path.join(os.path.dirname(curr_dir), "arch_configs/* ")

    os.system("cp -r " + str(config_dir) + str(tmp_dir))

    dir_exists_or_create(estimates_tc_dir)
    dir_exists_or_create(estimates_vc_dir)

    acc_setup_status["dir_setup"] = True
    acc_setup_status["setup_dir_path"] = tmp_dir
    acc_setup_status["setup_tc_dir_path"] = gemm_dir
    acc_setup_status["setup_vc_dir_path"] = vc_dir

    print("Initialized accelerator directories and files for area and energy estimation!")


def reset_accelerator():
    curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

    tmp_dir = os.path.join(os.path.dirname(curr_dir), "tmp")

    if (os.path.exists(tmp_dir)):
        os.system("rm -rf " + str(tmp_dir))

    acc_setup_status["dir_setup"] = False
    acc_setup_status["setup_dir_path"] = None
    acc_setup_status["config_setup"] = False
    acc_setup_status["curr_config"] = None


# Sets up the architecture yaml files in tmp/GEMM/arch/arch.yaml and tmp/vector_core/arch/arch.yaml
def setup_accelerator_with_config(config):
    acc_setup_status["curr_config"] = config
    acc_setup_status["config_setup"] = True

    assert acc_setup_status["dir_setup"] == True
    assert acc_setup_status["setup_dir_path"] is not None
    assert acc_setup_status["setup_tc_dir_path"] is not None
    assert acc_setup_status["setup_vc_dir_path"] is not None

    hardware_initializer(config, acc_setup_status["setup_dir_path"])

##################################### Estimator Functions #####################################

# Convert the core configuration for the estimator functions


def create_core_config_for_estimation(core_config, ):
    max_glb_size = 29360128
    max_glb_bw = 512
    buffer_scale_metric = 256

    dim1 = 2**int(log2(core_config.num)/2)
    dim2 = core_config.num // dim1

    curr_config_factor = max(
        core_config.width, core_config.depth) * max(dim1, dim2)

    GLB_BW = max_glb_bw

    L2_Buffer = core_config.L2_Buffer

    min_L2_Buffer = (2**(log2(core_config.width) +
                 log2(core_config.depth) - 6)) * 1024
    
    if (L2_Buffer < min_L2_Buffer):
        L2_Buffer = min_L2_Buffer
    
    # corner case
    if (core_config.width==256 and core_config.depth==32):
        if (L2_Buffer <256*1024):
            L2_Buffer = 256*1024
    if (core_config.width==256 and core_config.depth==16):
        if (L2_Buffer <128*1024):
            L2_Buffer = 128*1024

    acc_config = {
        # Compute Unit configuration
        "Core_x": max(dim1, dim2),
        "Core_y": min(dim1, dim2),
        "PE_x": core_config.width,
        "PE_y": core_config.depth,
        "VC_PE": core_config.width,
        # Global Buffer bandwidth
        "GLB_BUFFER_BW": GLB_BW,
        # GLB_Buffer size
        "GLB_Buffer": core_config.GLB_Buffer,
        "L2_Buffer": L2_Buffer,
        "L2_BW": core_config.L2_BW,
        # L1_Buffer size
        "L1_Buffer_TC": 32,
        "L1_Buffer_VC": 12,
        # Dataflow and Skipping
        "GLB_Buffer_skip": "Weights",
        "L2_Buffer_skip": "Inputs, Weights, Outputs",
        "L1_Buffer_skip": "None",
    }

    return acc_config


# Sets up the architecture yaml files in tmp/tensorcore/GEMM/arch
# Generates the area estimate for the setup architecture in tmp/tensorcore/GEMM/arch_estimates/ART.yaml

def get_core_area(config, core_type):
    setup_config = create_core_config_for_estimation(config)
    if acc_setup_status["curr_config"] != setup_config or acc_setup_status["config_setup"]:
        setup_accelerator_with_config(setup_config)

    if (core_type == "TC"):
        num_macs = (setup_config["Core_x"] * setup_config["Core_y"]) * \
            (setup_config["PE_x"] * setup_config["PE_y"])
        dir = acc_setup_status["setup_tc_dir_path"]
    if (core_type == "VC"):
        num_macs = (setup_config["Core_x"] *
                    setup_config["Core_y"]) * setup_config["VC_PE"]
        dir = acc_setup_status["setup_vc_dir_path"]

    return get_external_core_area(setup_config, num_macs, dir)


# Checks if the config is the current config being used
# Generates the energy estimate for the setup architecture in tmp/tensorcore/GEMM/arch_estimates/ERT_summary.yaml
def get_core_energy(coreconfig, core_type):
    setup_config = create_core_config_for_estimation(coreconfig)
    if core_type == "TC":
        dir = acc_setup_status["setup_tc_dir_path"]
    elif core_type == "VC":
        dir = acc_setup_status["setup_vc_dir_path"]
    else:
        raise ValueError("Invalid core type")

    if acc_setup_status["curr_config"] != setup_config or acc_setup_status["config_setup"]:
        setup_accelerator_with_config(setup_config)

    run_area_energy_generation(dir)
    return get_external_core_energy(dir)


def generate_fwd_bwd_dims(wgraph, wnode, setupconfig, estimates_file=""):
    schd = construct_external_scheduler(wgraph, setupconfig, estimates_file)
    return get_tc_dims(schd, wnode)


# Meta: We modify to only estimate for fwd pass
def tensor_core_estimator(wgraph, wnode, core_config, rd_wr_fnc, f=10 ** 6):
    setup_config = create_core_config_for_estimation(core_config)
    dims, fused_cyles = generate_fwd_bwd_dims(wgraph, wnode, setup_config)

    efwd = rd_wr_fnc(core_config, dims[0])
    ebwd = e_tuple(inf, 0, 0, 0,)
    # ebwd = rd_wr_fnc(core_config, dims[1])

    if efwd:
        return {"fwd": efwd._asdict(), "bwd": ebwd._asdict()}

    setup_accelerator_with_config(setup_config)

    start = time.time()
    energy = get_core_energy(core_config, "TC")
    if (use_peak):
        fwd_tuple = calc_lat_peak_flops(setup_config, dims[0])
        # bwd_tuple = calc_lat_peak_flops(setup_config, dims[0])
    else:
        fwd_tuple = tensor_core_estimate(setup_config, dims[0], energy)
        # bwd_tuple = tensor_core_estimate(setup_config, dims[1], energy)

    end = time.time()

    if fwd_tuple is None:
        efwd = e_tuple(inf, 0, 0, 0,)
    else:
        fwd_latency = (fwd_tuple[0] + fused_cyles[0]) / f
        # bwd_latency = (bwd_tuple[0] + fused_cyles[1]) / f
        efwd = e_tuple(
            fwd_latency, fwd_tuple[1] / (10**12), -1, fwd_tuple[2] + end - start,)
        # ebwd = e_tuple(
        #     bwd_latency, bwd_tuple[1] / (10**12), -1, bwd_tuple[2] + end - start,)

    rd_wr_fnc(core_config, dims[0], efwd)
    # rd_wr_fnc(core_config, dims[1], ebwd)
    ebwd = e_tuple(inf, 0, 0, 0,)

    print(wnode.node_desc, efwd, ebwd)

    return {"fwd": efwd._asdict(), "bwd": ebwd._asdict()}


# Meta: We modify to only estimate for fwd pass
def vector_core_estimator(wgraph, wnode, core_config, f=10 ** 6):
    setup_config = create_core_config_for_estimation(core_config)
    scheduler = construct_external_scheduler(wgraph, setup_config, "")

    start = time.time()
    setup_accelerator_with_config(setup_config)
    energy = get_core_energy(core_config, "VC")
    external_vc_estimator(scheduler, wnode, setup_config, energy)
    end = time.time()

    efwd = e_tuple(wnode.fwd_latency / f, wnode.fwd_energy /
                   (10**12), -1, end - start,)
    # ebwd = e_tuple(wnode.bwd_latency / f, wnode.fwd_energy /
    #                (10**12), -1, end - start,)
    ebwd = e_tuple(inf, 0, 0, 0,)

    return {"fwd": efwd._asdict(), "bwd": ebwd._asdict()}


def allreduce_estimator(phaze_node, num_devices, allreduce_type, bandwidth, bytes_per_element):
    start = time.time()

    tensorsize = phaze_node["node"].get_activation_size() * bytes_per_element
    # ring based allreduce
    size_per_size = tensorsize / num_devices * (num_devices - 1) * 2

    end = time.time()

    if allreduce_type == "AllReduceFwd" or allreduce_type == "AllReduce":
        efwd = e_tuple(size_per_size / bandwidth, 0, 0, end - start,)
        ebwd = e_tuple(0, 0, 0, end - start,)
    elif allreduce_type == "AllReduceBwd":
        efwd = e_tuple(0, 0, 0, end - start,)
        ebwd = e_tuple(size_per_size / bandwidth, 0, 0, end - start,)

    return {"fwd": efwd._asdict(), "bwd": ebwd._asdict()}


def get_flops(fusedgraph):
    VC_FLOPS = 0
    TC_FLOPS = 0

    for node in list(fusedgraph.nodes.values()):
        core_type = phaze_coretype_mapping[get_engine_type(
            node.node_desc)]

        if (core_type == "TC" or core_type == "TCandVC"):
            dims, fused_cyles = generate_fwd_bwd_dims(fusedgraph, node, {})
            TC_FLOPS += prod([i for i in dims[0] if type(i) == int and i != 0])
            TC_FLOPS += prod([i for i in dims[1] if type(i) == int and i != 0])

        if (core_type == "VC" or core_type == "TCandVC"):
            # for fwd and backward pass
            VC_FLOPS += 2 * prod(node.output_act[0])

    print("Number of nodes", len(fusedgraph.nodes.values()))
    print("VC FLOPS: ", VC_FLOPS / (10**9))
    print("TC FLOPS: ", TC_FLOPS / (10**9))


def calc_lat_peak_flops(setup_config, op):

    (b,
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
        "N": n * b,  # batch size #B
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

    numOp = 2 * n * b*c*m*r*s*p*q
    numMacs = (setup_config["Core_x"] * setup_config["Core_y"]) * \
        (setup_config["PE_x"] * setup_config["PE_y"])
    num_peak_flop_cycle = numMacs * 2

    return (numOp / (num_peak_flop_cycle / 2), 0, 0)
