import os
import inspect
import sys
import re
import shutil
import subprocess
import yaml
import definitions


def initialize(initial_config, cost_model_dir=None):

    # print("==================================================")
    # print("Initializing WHAM with the following parameters:", initial_config)
    # print("==================================================")

    if not cost_model_dir:
        this_file_path = os.path.abspath(
            inspect.getfile(inspect.currentframe()))
        this_directory = os.path.dirname(this_file_path)
        cost_model_dir = os.path.join(
            os.path.dirname(this_file_path), "../cost_model/")

    gemm_arch_config = os.path.join(cost_model_dir, "GEMM/arch/arch.yaml")
    map_const_config = os.path.join(
        cost_model_dir, "GEMM/constraints/map_constraints.yaml"
    )
    map_const_config_dw_conv = os.path.join(
        cost_model_dir, "GEMM/constraints/map_constraints_dw_conv.yaml"
    )

    mapper_config = os.path.join(cost_model_dir, "GEMM/mapper/mapper.yaml")

    vector_arch_config = os.path.join(
        cost_model_dir, 'vector_core/arch/arch.yaml')

    with open(gemm_arch_config, "r") as f:
        gemm_arch = yaml.load(f, Loader=yaml.SafeLoader)

    with open(map_const_config, "r") as f:
        gemm_map_config = yaml.load(f, Loader=yaml.SafeLoader)

    with open(map_const_config_dw_conv, "r") as f:
        gemm_map_config_dw_conv = yaml.load(f, Loader=yaml.SafeLoader)

    with open(vector_arch_config, "r") as f:
        vector_arch = yaml.load(f, Loader=yaml.SafeLoader)

    with open(mapper_config, "r") as f:
        mapper = yaml.load(f, Loader=yaml.SafeLoader)

    # DRAM
    gemm_arch["architecture"]["subtree"][0]["local"][0]["attributes"][
        "word-bits"
    ] = int(definitions.PRECISION)
    gemm_arch["architecture"]["subtree"][0]["local"][0]["attributes"][
        "block-size"
    ] = int(definitions.DRAM_BLOCK_SIZE)
    gemm_arch["architecture"]["subtree"][0]["local"][0]["attributes"]["width"] = int(
        definitions.PRECISION * definitions.DRAM_BLOCK_SIZE
    )

    # GLOBAL BUFFER
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "word-bits"
    ] = int(definitions.PRECISION)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "block-size"
    ] = int(definitions.GLB_BUFFER_BLOCK_SIZE)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "memory_width"
    ] = int(definitions.PRECISION * definitions.GLB_BUFFER_BLOCK_SIZE)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "memory_depth"
    ] = int(
        initial_config["GLB_Buffer"]
        / (definitions.GLB_BUFFER_BLOCK_SIZE * (definitions.PRECISION / 8))
    )
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "n_banks"
    ] = int(definitions.GLB_BUFFER_BLOCK_SIZE)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "read_bandwidth"
    ] = int(
        (initial_config["GLB_BUFFER_BW"] * 8)
        / (definitions.PRECISION * definitions.GLB_BUFFER_BLOCK_SIZE)
    )
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "write_bandwidth"
    ] = int(
        (initial_config["GLB_BUFFER_BW"] * 8)
        / (definitions.PRECISION * definitions.GLB_BUFFER_BLOCK_SIZE)
    )

    # Core L2 BUFFER
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["word-bits"] = int(definitions.PRECISION)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["block-size"] = int(definitions.L2_BUFFER_BLOCK_SIZE)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["memory_width"] = int(definitions.PRECISION * definitions.L2_BUFFER_BLOCK_SIZE)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["memory_depth"] = int(
        initial_config["L2_Buffer"]
        / (definitions.L2_BUFFER_BLOCK_SIZE * (definitions.PRECISION / 8))
    )
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["n_banks"] = int(definitions.L2_BUFFER_BLOCK_SIZE)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["read_bandwidth"] = int(
        (definitions.L2_BUFFER_NOC_BW * 8)
        / (definitions.PRECISION * definitions.L2_BUFFER_BLOCK_SIZE)
    )
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["write_bandwidth"] = int(
        (definitions.L2_BUFFER_NOC_BW * 8)
        / (definitions.PRECISION * definitions.L2_BUFFER_BLOCK_SIZE)
    )
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["meshX"] = int(initial_config["Core_x"])
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["meshY"] = int(initial_config["Core_y"])

    # Core Array
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["name"] = (
        "core[0.."
        + str((initial_config["Core_x"] * initial_config["Core_y"]) - 1)
        + "]"
    )

    # Core L1 Buffer
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["word-bits"] = int(definitions.PRECISION)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["block-size"] = int(definitions.L1_BUFFER_BLOCK_SIZE)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["width"] = int(definitions.PRECISION)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["depth"] = int(
        initial_config["L1_Buffer_TC"]
        / (definitions.L1_BUFFER_BLOCK_SIZE * (definitions.PRECISION / 8))
    )
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["meshX"] = int(initial_config["Core_x"] * initial_config["PE_x"])
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["meshY"] = int(initial_config["Core_y"] * initial_config["PE_y"])

    # Core MAC
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][1]["attributes"]["datawidth"] = int(definitions.PRECISION)
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][1]["attributes"]["meshX"] = int(initial_config["Core_x"] * initial_config["PE_x"])
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][1]["attributes"]["meshY"] = int(initial_config["Core_y"] * initial_config["PE_y"])

    # PE Array
    gemm_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "name"
    ] = ("PE[0.." + str((initial_config["PE_x"] * initial_config["PE_y"]) - 1) + "]")

    # DRAM
    vector_arch["architecture"]["subtree"][0]["local"][0]["attributes"][
        "word-bits"
    ] = int(definitions.PRECISION)
    vector_arch["architecture"]["subtree"][0]["local"][0]["attributes"][
        "block-size"
    ] = int(definitions.DRAM_BLOCK_SIZE)
    vector_arch["architecture"]["subtree"][0]["local"][0]["attributes"]["width"] = int(
        definitions.PRECISION * definitions.DRAM_BLOCK_SIZE
    )

    # GLOBAL BUFFER
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "word-bits"
    ] = int(definitions.PRECISION)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "block-size"
    ] = int(definitions.GLB_BUFFER_BLOCK_SIZE)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "memory_width"
    ] = int(definitions.PRECISION * definitions.GLB_BUFFER_BLOCK_SIZE)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "memory_depth"
    ] = int(
        initial_config["GLB_Buffer"]
        / (definitions.GLB_BUFFER_BLOCK_SIZE * (definitions.PRECISION / 8))
    )
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "n_banks"
    ] = int(definitions.GLB_BUFFER_BLOCK_SIZE)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "read_bandwidth"
    ] = int(
        (initial_config["GLB_BUFFER_BW"] * 8)
        / (definitions.PRECISION * definitions.GLB_BUFFER_BLOCK_SIZE)
    )
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"][
        "write_bandwidth"
    ] = int(
        (initial_config["GLB_BUFFER_BW"] * 8)
        / (definitions.PRECISION * definitions.GLB_BUFFER_BLOCK_SIZE)
    )

    # Core L2 BUFFER
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["word-bits"] = int(definitions.PRECISION)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["block-size"] = int(definitions.L2_BUFFER_BLOCK_SIZE)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["memory_width"] = int(definitions.PRECISION * definitions.L2_BUFFER_BLOCK_SIZE)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["memory_depth"] = int(
        initial_config["L2_Buffer"]
        / (definitions.L2_BUFFER_BLOCK_SIZE * (definitions.PRECISION / 8))
    )
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["n_banks"] = int(definitions.L2_BUFFER_BLOCK_SIZE)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["read_bandwidth"] = int(
        (definitions.L2_BUFFER_NOC_BW * 8)
        / (definitions.PRECISION * definitions.L2_BUFFER_BLOCK_SIZE)
    )
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["write_bandwidth"] = int(
        (definitions.L2_BUFFER_NOC_BW * 8)
        / (definitions.PRECISION * definitions.L2_BUFFER_BLOCK_SIZE)
    )
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["meshX"] = int(initial_config["Core_x"])
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"][0][
        "attributes"
    ]["meshY"] = int(initial_config["Core_y"])

    # Core Array
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["name"] = (
        "core[0.."
        + str((initial_config["Core_x"] * initial_config["Core_y"]) - 1)
        + "]"
    )

    # Core L1 Buffer
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["word-bits"] = int(definitions.PRECISION)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["block-size"] = int(definitions.L1_BUFFER_BLOCK_SIZE)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["width"] = int(definitions.PRECISION)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["depth"] = int(
        initial_config["L1_Buffer_VC"]
        / (definitions.L1_BUFFER_BLOCK_SIZE * (definitions.PRECISION / 8))
    )
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["meshX"] = int(initial_config["Core_x"] * initial_config["VC_PE"])
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][0]["attributes"]["meshY"] = int(initial_config["Core_y"])

    # Core MAC
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][1]["attributes"]["datawidth"] = int(definitions.PRECISION)
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][1]["attributes"]["meshX"] = int(initial_config["Core_x"] * initial_config["VC_PE"])
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "local"
    ][1]["attributes"]["meshY"] = int(initial_config["Core_y"])

    # PE Array
    vector_arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["subtree"][0][
        "name"
    ] = ("PE[0.." + str((initial_config["VC_PE"]) - 1) + "]")

    # Gemm Mapper Optimization Metric
    mapper["mapper"]["optimization-metrics"] = definitions.OPT_METRIC

    with open(gemm_arch_config, "w") as f:
        f.write(yaml.dump(gemm_arch))

    with open(vector_arch_config, "w") as f:
        f.write(yaml.dump(vector_arch))

    with open(mapper_config, "w") as f:
        f.write(yaml.dump(mapper))


def run_area_energy_generation(cost_model_dir=None):
    tc_est_output = os.path.join(cost_model_dir, "arch_estimates/")

    tc_arch = os.path.join(cost_model_dir, "arch/arch.yaml")
    tc_arch_comp_sram = os.path.join(
        cost_model_dir, "arch/components/smartbuffer_SRAM.yaml"
    )

    accelergy_exec = "accelergy"
    if shutil.which(accelergy_exec) is not None:
        pass
    else:
        print("Accelergy not found!!")
        sys.exit()
    status = subprocess.call(
        [accelergy_exec, "-o", tc_est_output, tc_arch, tc_arch_comp_sram], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if status != 0:
        print("Did you remember to build Accelergy and set up your environment properly?")
        sys.exit(1)


def get_core_energy(this_directory=None, scheduler=None):
    tc_energy = {}

    ERT_src = os.path.join(
        this_directory, "arch_estimates/ERT_summary.yaml")

    with open(ERT_src, "r") as f:
        ERT = yaml.load(f, Loader=yaml.SafeLoader)

    for i in range(len(ERT["ERT_summary"]["table_summary"])):
        table = ERT["ERT_summary"]["table_summary"][i]
        name = table["name"]
        name = re.split("[.]", name)
        name = name[-1]

        if name == "mac":
            mac_energy = table["actions"][0]["energy"]
            tc_energy["mac"] = mac_energy
        elif name == "DRAM":
            dram_read_energy = table["actions"][0]["energy"]
            dram_write_energy = table["actions"][1]["energy"]
            tc_energy["dram_read"] = dram_read_energy
            tc_energy["dram_write"] = dram_write_energy
        elif name == "GLB_Buffer":
            tc_glb_write_energy = table["actions"][0]["average_energy"]
            tc_glb_read_energy = table["actions"][1]["average_energy"]
            tc_energy["tc_glb_read"] = tc_glb_read_energy
            tc_energy["tc_glb_write"] = tc_glb_write_energy
        elif name == "L2_Buffer":
            tc_l2_write_energy = table["actions"][0]["average_energy"]
            tc_l2_read_energy = table["actions"][1]["average_energy"]
            tc_energy["tc_l2_read"] = tc_l2_read_energy
            tc_energy["tc_l2_write"] = tc_l2_write_energy
        elif name == "L1_Buffer":
            tc_l1_read_energy = table["actions"][0]["average_energy"]
            tc_l1_write_energy = table["actions"][1]["average_energy"]
            tc_energy["tc_l1_read"] = tc_l1_read_energy
            tc_energy["tc_l1_write"] = tc_l1_write_energy

    LUT_rd_energy = 2  # Fixed LUT energy of 2 pJ
    tc_energy["lut_read"] = LUT_rd_energy

    if scheduler:
        scheduler.set_tensor_core_energy(tc_energy)

    return tc_energy


def get_core_area(config, num_macs, dir):
    area = 0

    ART_src = os.path.join(dir, "arch_estimates/ART.yaml")

    run_area_energy_generation(dir)

    with open(ART_src, "r") as f:
        ART = yaml.load(f, Loader=yaml.SafeLoader)

    for i in range(len(ART["ART"]["tables"])):
        table = ART["ART"]["tables"][i]
        name = table["name"]
        name = re.split("[.]", name)
        name = name[-1]
        table_area = table["area"]

        if name == "mac":
            area += table_area * num_macs

        elif name == "GLB_Buffer":
            glb_area = table_area

        elif name == "L2_Buffer":
            area += table_area * (config["Core_x"] * config["Core_y"])

        elif name == "L1_Buffer":
            area += table_area * num_macs

    return area, glb_area
