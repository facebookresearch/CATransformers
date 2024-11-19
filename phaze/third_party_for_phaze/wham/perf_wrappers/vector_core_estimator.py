import sys
from math import ceil, floor, log2, sqrt, prod
from ast import literal_eval
import definitions
import op_to_compute

# ==================================== ESTIMATION ===============================================

# LUT based exponential function implementation
# http://www.acsel-lab.com/arithmetic/arith10/papers/ARITH10_Tang.pdf

# LUT Cycles = 1
# ADD Cycles = 1
# SUB Cycles = 1
# MUL Cycles = 1
# DIV Cycles = 10
# COMP Cycles = 1
# MAC Cycles = 1

# SIGMOID = S(x) = 1 / 1 + e^(-x)
# SIGMOID = LUT + ADD + DIV
# SIGMOID = 12

# DER SIGMOID = dS/dx = S(1 - S)
# DER SIGMOID = MUL + SUB
# DER SIGMOID = 2

# SWISS = S(x) = x / 1 + e^(-beta * x)
# SWISS = LUT + MUL + SIGMOID
# SWISS = 14

# DER SWISS = dS/dx = S / x + S(1 - S/x)
# DER SWISS = MUL + DIV + ADD + SUB
# DER SWISS = 15

# TANH = S(x) = e^(x) - e^(-x) / e^(x) + e^(-x)
# TANH = 2*LUT + 2*SIGMOID + SUB
# TANH = 27

# Sin = S(x) = e^(x) - e^(-x) / 2
# Sin = 2*LUT + SUB + DIV
# Sin = 13

# COS = S(x) = e^(x) + e^(-x) / 2
# COS = 2*LUT + Add + DIV
# COS = 13

# DER TANH = dS/dx = 1 - S^(2)
# DER TANH = MUL + SUB
# DER TANH = 2

# RELU = S(x) = max(0,x)
# RELU = COMP
# RELU = 1

# DER RELU = dS/dx = 0, 1
# DER RELU = CONST
# DER RELU = 1

# SOFTMAX = S_i(x) = e^(x_i) / sum(e^(x_j))
# SOFTMAX = LUT + DIV + n*ADD
# SOFTMAX = 11 + n*1

# DER SOFTMAX = dS/dx = S_i(1-S_j) , -S_i*S_j
# DER SOFTMAX = MUL + SUB
# DER SOFTMAX = 2

# ====================================================================================================


def get_optimizer_est(schd, node, optimizer, config):

    stats = get_opt_est(schd, node, optimizer, config)
    node.optimizer_latency = stats["cycles"]
    node.optimizer_energy = stats["energy_pJ"]


def get_performance_est(schd, node, config, energy={}):
    schd.set_tensor_core_energy(energy)

    # Check if fused operator
    fused = False
    fused_ops = []

    if node.node_desc.split(" ~~ ")[0] == "fused":
        fused = True
        fused_info = node.fused_operators.strip().split(" ~~ ")
        for i in range(len(fused_info)):
            if i % 3 == 0:
                name = fused_info[i]
            elif i % 3 == 1:
                weights = literal_eval(fused_info[i])
            elif i % 3 == 2:
                output = literal_eval(fused_info[i])
                op = {}
                op["name"] = name
                op["weights"] = weights
                op["output"] = output
                fused_ops.append(op)

    if fused:
        fwd_PE_cycles = 0
        bwd_PE_cycles = 0
        num_ops = 0
        num_weights = 0
        for i in range(len(fused_ops)):
            op = fused_ops[i]
            if op_to_compute.get_opr_type(op["name"]) == "transformation_opr":
                continue
            else:
                fwd_PE_cycles += op_to_compute.get_fwd_pe_cycles(op["name"])
                bwd_PE_cycles += op_to_compute.get_bwd_pe_cycles(op["name"])
                for j in range(len(op["weights"])):
                    num_weights += op["weights"][j][0]
                num_ops += 1

        output = node.output_act[0]

        # Forward Estimation
        stats = get_fused_op_est(schd, fwd_PE_cycles, output, num_ops, config)
        node.fwd_latency = stats["cycles"]
        node.fwd_energy = stats["energy_pJ"]
        node.fwd_l2_ip_tile = config["VC_PE"] * \
            config["Core_x"] * config["Core_y"]
        # Backward Estimation
        stats = get_fused_op_est(schd, bwd_PE_cycles, output, num_ops, config)
        node.bwd_latency = stats["cycles"]
        node.bwd_energy = stats["energy_pJ"]
        node.bwd_l2_ip_tile = config["VC_PE"] * \
            config["Core_x"] * config["Core_y"]
        # Weight Update Estimation
        if num_weights != 0:
            stats = get_fused_weight_update_estimation(
                schd, num_weights, config)
            node.weight_update_latency = stats["cycles"]
            node.weight_update_energy = stats["energy_pJ"]
            node.weight_update_l2_ip_tile = (
                config["VC_PE"] * config["Core_x"] * config["Core_y"]
            )

    else:
        # Forward Estimation
        stats = get_fwd_estimation(schd, node, config)
        node.fwd_latency = stats["cycles"]
        node.fwd_energy = stats["energy_pJ"]
        node.fwd_l2_ip_tile = config["VC_PE"]
        # Backward Estimation
        stats = get_bwd_estimation(schd, node, config)
        node.bwd_latency = stats["cycles"]
        node.bwd_energy = stats["energy_pJ"]
        node.bwd_l2_ip_tile = config["VC_PE"]
        # Weight Update Estimation
        num_weights = 0
        pre_dec_nodes = schd.get_predecessor_nodes(node)
        for pre_dec_node in pre_dec_nodes:
            if pre_dec_node.node_desc == "AccumulateGrad":
                num_weights += prod(pre_dec_node.parameter)

        if num_weights != 0:
            stats = get_weight_update_estimation(schd, num_weights, config)
            node.weight_update_latency = stats["cycles"]
            node.weight_update_energy = stats["energy_pJ"]
            node.weight_update_l2_ip_tile = config["VC_PE"]


def get_fused_op_est(schd, PE_cycles, output, num_ops, config):
    dram_reads = 0
    dram_writes = 0

    l2_reads = 0
    l2_writes = 0

    lut_reads = 0

    ip_reg_writes = 0
    ip_reg_reads = 0
    a_reg_writes = 0
    a_reg_reads = 0
    b_reg_writes = 0
    b_reg_reads = 0
    op_reg_writes = 0
    op_reg_reads = 0
    ir_reg_writes = 0
    ir_reg_reads = 0

    num_tiles = ceil(
        prod(output) / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )

    total_cycles = (
        ceil(prod(output) / (config["VC_PE"] *
             config["Core_x"] * config["Core_y"]))
        * PE_cycles
    )

    dram_reads = 0  # Result forwarded directly
    dram_writes = num_tiles * num_ops

    l2_writes = num_tiles * num_ops
    l2_reads = num_tiles * num_ops

    ip_reg_reads = num_tiles * PE_cycles
    ip_reg_writes = num_tiles

    op_reg_writes = num_tiles

    mac_opr = total_cycles

    access_count = (
        mac_opr,
        dram_reads,
        dram_writes,
        l2_reads,
        l2_writes,
        lut_reads,
        ip_reg_reads,
        ip_reg_writes,
        a_reg_reads,
        a_reg_writes,
        b_reg_reads,
        b_reg_writes,
        op_reg_reads,
        op_reg_writes,
        ir_reg_reads,
        ir_reg_writes,
    )
    total_energy = get_node_energy(schd, access_count, config)

    output = {"utilization": 1, "cycles": total_cycles,
              "energy_pJ": total_energy}

    return output


def get_fused_weight_update_estimation(schd, num_weights, config):
    PE_cycles = 1

    dram_reads = 0
    dram_writes = 0

    l2_reads = 0
    l2_writes = 0

    lut_reads = 0

    ip_reg_writes = 0
    ip_reg_reads = 0
    a_reg_writes = 0
    a_reg_reads = 0
    b_reg_writes = 0
    b_reg_reads = 0
    op_reg_writes = 0
    op_reg_reads = 0
    ir_reg_writes = 0
    ir_reg_reads = 0

    num_tiles = ceil(
        num_weights / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )

    total_cycles = (
        ceil(num_weights / (config["VC_PE"] *
             config["Core_x"] * config["Core_y"]))
        * PE_cycles
    )

    dram_reads = 0  # Result forwarded directly
    dram_writes = num_tiles

    l2_writes = num_tiles
    l2_reads = num_tiles

    ip_reg_reads = num_tiles
    ip_reg_writes = num_tiles

    op_reg_writes = num_tiles

    mac_opr = total_cycles
    access_count = (
        mac_opr,
        dram_reads,
        dram_writes,
        l2_reads,
        l2_writes,
        lut_reads,
        ip_reg_reads,
        ip_reg_writes,
        a_reg_reads,
        a_reg_writes,
        b_reg_reads,
        b_reg_writes,
        op_reg_reads,
        op_reg_writes,
        ir_reg_reads,
        ir_reg_writes,
    )
    total_energy = get_node_energy(schd, access_count, config)

    output = {"utilization": 1, "cycles": total_cycles,
              "energy_pJ": total_energy}

    return output


def get_node_energy(schd, access_count, config):
    (
        mac_opr,
        dram_reads,
        dram_writes,
        l2_reads,
        l2_writes,
        lut_reads,
        ip_reg_reads,
        ip_reg_writes,
        a_reg_reads,
        a_reg_writes,
        b_reg_reads,
        b_reg_writes,
        op_reg_reads,
        op_reg_writes,
        ir_reg_reads,
        ir_reg_writes,
    ) = access_count
    tc_energy = schd.get_tensor_core_energy()

    dram_read_energy = dram_reads * (
        tc_energy["dram_read"] / definitions.DRAM_BLOCK_SIZE
    )
    dram_write_energy = dram_writes * (
        tc_energy["dram_write"] / definitions.DRAM_BLOCK_SIZE
    )

    l2_read_energy = l2_reads * (
        tc_energy["tc_l2_read"] / definitions.L2_BUFFER_BLOCK_SIZE
    )
    l2_write_energy = l2_writes * (
        tc_energy["tc_l2_write"] / definitions.L2_BUFFER_BLOCK_SIZE
    )

    lut_read_energy = lut_reads * tc_energy["lut_read"]

    ip_reg_read_energy = (
        ip_reg_reads
        * tc_energy["tc_l1_read"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )
    ip_reg_write_energy = (
        ip_reg_writes
        * tc_energy["tc_l1_write"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )

    a_reg_read_energy = (
        a_reg_reads
        * tc_energy["tc_l1_read"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )
    a_reg_write_energy = (
        a_reg_writes
        * tc_energy["tc_l1_write"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )

    b_reg_read_energy = (
        b_reg_reads
        * tc_energy["tc_l1_read"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )
    b_reg_write_energy = (
        b_reg_writes
        * tc_energy["tc_l1_write"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )

    op_reg_read_energy = (
        op_reg_reads
        * tc_energy["tc_l1_read"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )
    op_reg_write_energy = (
        op_reg_writes
        * tc_energy["tc_l1_write"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )

    ir_reg_read_energy = (
        ir_reg_reads
        * tc_energy["tc_l1_read"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )
    ir_reg_write_energy = (
        ir_reg_writes
        * tc_energy["tc_l1_write"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )

    mac_energy = (
        mac_opr
        * tc_energy["mac"]
        * (config["VC_PE"] * config["Core_x"] * config["Core_y"])
    )

    total_energy = (
        dram_read_energy
        + dram_write_energy
        + l2_read_energy
        + l2_write_energy
        + lut_read_energy
        + ip_reg_read_energy
        + ip_reg_write_energy
        + a_reg_read_energy
        + a_reg_write_energy
        + b_reg_read_energy
        + b_reg_write_energy
        + op_reg_read_energy
        + op_reg_write_energy
        + ir_reg_read_energy
        + ir_reg_write_energy
        + mac_energy
    )

    return total_energy


def get_fwd_estimation(schd, node, config, total_cycles=-10):
    # ==================== CudnnBatchNorm, NativeBatchNorm =======================#
    if node.node_desc == "CudnnBatchNorm" or node.node_desc == "NativeBatchNorm":
        # y = input * gamma + beta
        # ISA
        # MUL, ADD
        # Cycles = 1
        PE_cycles = 1
        output = node.output_act[0]

        # TODO: check this for phaze, if the order is correct
        gamma = node.saved_tensors[1][0]
        beta = node.saved_tensors[2][0]

        N = output[0]
        C = output[1]
        P = output[2]
        Q = output[3]

        assert C == gamma and C == beta, "Dimensions doesn't match!!"

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        # Re-using gamma and beta across each channel and all no of batches
        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles
        a_reg_reads = total_cycles
        a_reg_writes = gamma
        b_reg_reads = total_cycles
        b_reg_writes = beta
        op_reg_writes = total_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== CudnnLayerNorm, NativeBatchNorm =======================#
    elif node.node_desc == "CudnnLayerNorm" or node.node_desc == "NativeLayerNorm":
        # y = input * gamma + beta
        # ISA
        # MUL, ADD
        # Cycles = 1
        PE_cycles = 1
        output = node.output_act[0]

        gamma = int(node.saved_tensors[0][-1])
        beta = int(node.saved_tensors[1][-1])

        if len(output) == 3:
            N = output[0]
            seq_len = output[1]
            dim = int(output[2])
        elif len(output) == 2:
            N = 1
            seq_len = output[0]
            dim = int(output[1])

        assert dim == gamma == beta, "Dimensions doesn't match!!"

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        # Re-using gamma and beta across each channel and all no of batches
        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles
        a_reg_reads = total_cycles
        a_reg_writes = gamma
        b_reg_reads = total_cycles
        b_reg_writes = beta
        op_reg_writes = total_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Add =======================#
    elif node.node_desc == "Add":
        # y = input + a_reg
        # ISA
        # ADD
        # Cycles = 1
        PE_cycles = 1
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        if len(nodes) == 2:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) != 0 and len(a_reg) != 0:
                if len(input) == len(a_reg):
                    # Example: [32, 512, 1024] + [32, 512, 1024] = [32, 512, 1024]
                    if prod(input) > prod(
                        a_reg
                    ):  # Example: [32, 512, 1024] + [32, 512, 1] = [32, 512, 1024]
                        x = input
                        y = a_reg
                    elif prod(input) < prod(
                        a_reg
                    ):  # Example: [32, 512, 1] + [32, 512, 1024] = [32, 512, 1024]
                        x = a_reg
                        y = input
                    else:  # Example: [32, 512, 1024] + [32, 512, 1024] = [32, 512, 1024]
                        x = input
                        y = a_reg

                    # tile size calculation
                    dram_tile = ceil(
                        (prod(x) * (definitions.PRECISION / 8)) /
                        config["L2_Buffer"]
                    )
                    num_dram_tiles = ceil(prod(x) / dram_tile)

                    l2_tile = ceil(
                        dram_tile
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    num_l2_tiles = num_dram_tiles * l2_tile

                    total_cycles = (
                        ceil(
                            prod(x)
                            / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                        )
                        * PE_cycles
                    )

                    ip_reg_reads = total_cycles
                    ip_reg_writes = total_cycles

                    a_reg_reads = total_cycles
                    a_reg_writes = (
                        ceil(
                            prod(y)
                            / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                        )
                        * PE_cycles
                    )

                    op_reg_writes = total_cycles

                    l2_reads = num_l2_tiles
                    l2_writes = num_l2_tiles

                    dram_reads = num_dram_tiles
                    dram_writes = num_dram_tiles

                elif len(input) > len(a_reg):
                    # Example: [32, 512, 1024] + [1024] = [32, 512, 1024]

                    # tile size calculation
                    dram_tile = ceil(
                        (prod(input) * (definitions.PRECISION / 8))
                        / config["L2_Buffer"]
                    )
                    num_dram_tiles = ceil(prod(input) / dram_tile)

                    l2_tile = ceil(
                        dram_tile
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    num_l2_tiles = num_dram_tiles * l2_tile

                    total_cycles = (
                        ceil(
                            prod(input)
                            / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                        )
                        * PE_cycles
                    )

                    ip_reg_reads = total_cycles
                    ip_reg_writes = total_cycles

                    a_reg_reads = total_cycles
                    a_reg_writes = ceil(total_cycles / a_reg[0])

                    op_reg_writes = total_cycles

                    l2_reads = num_l2_tiles
                    l2_writes = num_l2_tiles

                    dram_reads = num_dram_tiles
                    dram_writes = num_dram_tiles

                else:
                    # Example: [1024] + [32, 512, 1024]  = [32, 512, 1024]

                    # tile size calculation
                    dram_tile = ceil(
                        (prod(a_reg) * (definitions.PRECISION / 8))
                        / config["L2_Buffer"]
                    )
                    num_dram_tiles = ceil(prod(a_reg) / dram_tile)

                    l2_tile = ceil(
                        dram_tile
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    num_l2_tiles = num_dram_tiles * l2_tile

                    total_cycles = (
                        ceil(
                            prod(a_reg)
                            / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                        )
                        * PE_cycles
                    )

                    ip_reg_reads = total_cycles
                    ip_reg_writes = total_cycles

                    a_reg_reads = total_cycles
                    a_reg_writes = ceil(total_cycles / input[0])

                    op_reg_writes = total_cycles

                    l2_reads = num_l2_tiles
                    l2_writes = num_l2_tiles

                    dram_reads = num_dram_tiles
                    dram_writes = num_dram_tiles

            elif len(input) == 0 and len(a_reg) == 0:
                total_cycles = 0
        else:
            # Example: [32, 512, 1024] + scalar = [32, 512, 1024]
            # ADD_Im
            input = nodes[0].output_act[0]

            # tile size calculation
            # tile size calculation
            dram_tile = ceil(
                (prod(input) * (definitions.PRECISION / 8)) /
                config["L2_Buffer"]
            )
            num_dram_tiles = ceil(prod(input) / dram_tile)

            l2_tile = ceil(
                dram_tile / (config["VC_PE"] *
                             config["Core_x"] * config["Core_y"])
            )
            num_l2_tiles = num_dram_tiles * l2_tile

            total_cycles = (
                ceil(
                    prod(input)
                    / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                )
                * PE_cycles
            )

            ip_reg_reads = total_cycles
            ip_reg_writes = total_cycles

            op_reg_writes = total_cycles

            l2_reads = num_l2_tiles
            l2_writes = num_l2_tiles

            dram_reads = num_dram_tiles
            dram_writes = num_dram_tiles

    # ====================== Addcmul =======================#
    elif node.node_desc == "Addcmul":
        # ISA
        # ADD
        # MUL
        # Cycles = 1
        PE_cycles = 1
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        if len(nodes) == 2:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) != 0 and len(a_reg) != 0:
                if len(input) == len(a_reg):
                    # Example: [32, 512, 1024] + [32, 512, 1024] = [32, 512, 1024]
                    if prod(input) > prod(
                        a_reg
                    ):  # Example: [32, 512, 1024] + [32, 512, 1] = [32, 512, 1024]
                        x = input
                        y = a_reg
                    elif prod(input) < prod(
                        a_reg
                    ):  # Example: [32, 512, 1] + [32, 512, 1024] = [32, 512, 1024]
                        x = a_reg
                        y = input
                    else:  # Example: [32, 512, 1024] + [32, 512, 1024] = [32, 512, 1024]
                        x = input
                        y = a_reg

                    # tile size calculation
                    dram_tile = ceil(
                        (prod(x) * (definitions.PRECISION / 8)) /
                        config["L2_Buffer"]
                    )
                    num_dram_tiles = ceil(prod(x) / dram_tile)

                    l2_tile = ceil(
                        dram_tile
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    num_l2_tiles = num_dram_tiles * l2_tile

                    total_cycles = (
                        ceil(
                            prod(x)
                            / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                        )
                        * PE_cycles
                    )

                    ip_reg_reads = total_cycles
                    ip_reg_writes = total_cycles

                    a_reg_reads = total_cycles
                    a_reg_writes = (
                        ceil(
                            prod(y)
                            / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                        )
                        * PE_cycles
                    )

                    op_reg_writes = total_cycles

                    l2_reads = num_l2_tiles
                    l2_writes = num_l2_tiles

                    dram_reads = num_dram_tiles
                    dram_writes = num_dram_tiles

                elif len(input) > len(a_reg):
                    # Example: [32, 512, 1024] + [1024] = [32, 512, 1024]

                    # tile size calculation
                    dram_tile = ceil(
                        (prod(input) * (definitions.PRECISION / 8))
                        / config["L2_Buffer"]
                    )
                    num_dram_tiles = ceil(prod(input) / dram_tile)

                    l2_tile = ceil(
                        dram_tile
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    num_l2_tiles = num_dram_tiles * l2_tile

                    total_cycles = (
                        ceil(
                            prod(input)
                            / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                        )
                        * PE_cycles
                    )

                    ip_reg_reads = total_cycles
                    ip_reg_writes = total_cycles

                    a_reg_reads = total_cycles
                    a_reg_writes = ceil(total_cycles / a_reg[0])

                    op_reg_writes = total_cycles

                    l2_reads = num_l2_tiles
                    l2_writes = num_l2_tiles

                    dram_reads = num_dram_tiles
                    dram_writes = num_dram_tiles

                else:
                    # Example: [1024] + [32, 512, 1024]  = [32, 512, 1024]

                    # tile size calculation
                    dram_tile = ceil(
                        (prod(a_reg) * (definitions.PRECISION / 8))
                        / config["L2_Buffer"]
                    )
                    num_dram_tiles = ceil(prod(a_reg) / dram_tile)

                    l2_tile = ceil(
                        dram_tile
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    num_l2_tiles = num_dram_tiles * l2_tile

                    total_cycles = (
                        ceil(
                            prod(a_reg)
                            / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                        )
                        * PE_cycles
                    )

                    ip_reg_reads = total_cycles
                    ip_reg_writes = total_cycles

                    a_reg_reads = total_cycles
                    a_reg_writes = ceil(total_cycles / input[0])

                    op_reg_writes = total_cycles

                    l2_reads = num_l2_tiles
                    l2_writes = num_l2_tiles

                    dram_reads = num_dram_tiles
                    dram_writes = num_dram_tiles

            elif len(input) == 0 and len(a_reg) == 0:
                total_cycles = 0
        else:
            # Example: [32, 512, 1024] + scalar = [32, 512, 1024]
            # ADD_Im
            input = nodes[0].output_act[0]

            # tile size calculation
            # tile size calculation
            dram_tile = ceil(
                (prod(input) * (definitions.PRECISION / 8)) /
                config["L2_Buffer"]
            )
            num_dram_tiles = ceil(prod(input) / dram_tile)

            l2_tile = ceil(
                dram_tile / (config["VC_PE"] *
                             config["Core_x"] * config["Core_y"])
            )
            num_l2_tiles = num_dram_tiles * l2_tile

            total_cycles = (
                ceil(
                    prod(input)
                    / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                )
                * PE_cycles
            )

            ip_reg_reads = total_cycles
            ip_reg_writes = total_cycles

            op_reg_writes = total_cycles

            l2_reads = num_l2_tiles
            l2_writes = num_l2_tiles

            dram_reads = num_dram_tiles
            dram_writes = num_dram_tiles

    # ====================== Sub =======================#
    elif node.node_desc == "Sub":
        # y = input - a_reg
        # ISA
        # SUB
        # Cycles = 1
        PE_cycles = 1
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0
        # print(node)
        if len(nodes) == 2:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) == len(a_reg):
                # Example: [32, 512, 1024] - [32, 512, 1024] = [32, 512, 1024]
                if prod(input) > prod(
                    a_reg
                ):  # Example: [32, 512, 1024] - [32, 512, 1] = [32, 512, 1024]
                    x = input
                    y = a_reg
                elif prod(input) < prod(
                    a_reg
                ):  # Example: [32, 512, 1] - [32, 512, 1024] = [32, 512, 1024]
                    x = a_reg
                    y = a_reg
                else:  # Example: [32, 512, 1024] - [32, 512, 1024] = [32, 512, 1024]
                    x = input
                    y = a_reg

                # tile size calculation
                dram_tile = ceil(
                    (prod(x) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(x) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(x)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles
                ip_reg_writes = total_cycles

                a_reg_reads = total_cycles
                a_reg_writes = (
                    ceil(
                        prod(y)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                op_reg_writes = total_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

            elif len(input) > len(a_reg):
                # Example: [32, 512, 1024] - [1024] = [32, 512, 1024]

                # tile size calculation
                dram_tile = ceil(
                    (prod(input) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(input) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(input)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles
                ip_reg_writes = total_cycles

                a_reg_reads = total_cycles
                a_reg_writes = ceil(total_cycles / a_reg[0])

                op_reg_writes = total_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

            else:
                # Example: [1024] - [32, 512, 1024]  = [32, 512, 1024]

                # tile size calculation
                dram_tile = ceil(
                    (prod(a_reg) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(a_reg) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(a_reg)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles
                ip_reg_writes = total_cycles

                a_reg_reads = total_cycles
                a_reg_writes = ceil(total_cycles / input[0])

                op_reg_writes = total_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles
        else:
            # Example: [32, 512, 1024] - scalar = [32, 512, 1024]
            # SUB_Im

            input = nodes[0].output_act[0]

            # tile size calculation
            dram_tile = ceil(
                (prod(input) * (definitions.PRECISION / 8)) /
                config["L2_Buffer"]
            )
            num_dram_tiles = ceil(prod(input) / dram_tile)

            l2_tile = ceil(
                dram_tile / (config["VC_PE"] *
                             config["Core_x"] * config["Core_y"])
            )
            num_l2_tiles = num_dram_tiles * l2_tile

            total_cycles = (
                ceil(
                    prod(input)
                    / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                )
                * PE_cycles
            )

            ip_reg_reads = total_cycles
            ip_reg_writes = total_cycles

            op_reg_writes = total_cycles

            l2_reads = num_l2_tiles
            l2_writes = num_l2_tiles

            dram_reads = num_dram_tiles
            dram_writes = num_dram_tiles

    # ====================== Mul =======================#
    elif node.node_desc == "Mul":
        # y = input * a_reg
        # ISA
        # MUL
        # Cycles = 1
        PE_cycles = 1
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        if len(nodes) == 2 and len(nodes[1].output_act[0]) > 0:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) == len(a_reg):
                # Example: [32, 512, 1024] * [32, 512, 1024] = [32, 512, 1024]
                if prod(input) > prod(
                    a_reg
                ):  # Example: [32, 512, 1024] * [32, 512, 1] = [32, 512, 1024]
                    x = input
                    y = a_reg
                elif prod(input) < prod(
                    a_reg
                ):  # Example: [32, 512, 1] * [32, 512, 1024] = [32, 512, 1024]
                    x = a_reg
                    y = a_reg
                else:  # Example: [32, 512, 1024] * [32, 512, 1024] = [32, 512, 1024]
                    x = input
                    y = a_reg

                # tile size calculation
                dram_tile = ceil(
                    (prod(x) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(x) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(x)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles
                ip_reg_writes = total_cycles

                a_reg_reads = total_cycles
                a_reg_writes = (
                    ceil(
                        prod(y)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                op_reg_writes = total_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

            elif len(input) > len(a_reg):
                # Example: [32, 512, 1024] * [1024] = [32, 512, 1024] Element wise Multiplication along certain dimension (1024 dim)
                # tile size calculation
                dram_tile = ceil(
                    (prod(input) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(input) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(input)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles
                ip_reg_writes = total_cycles

                a_reg_reads = total_cycles
                a_reg_writes = ceil(total_cycles / a_reg[0])

                op_reg_writes = total_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

            else:
                # Example: [1024] * [32, 512, 1024]  = [32, 512, 1024] Element wise Multiplication along certain dimension (1024 dim)
                # tile size calculation
                dram_tile = ceil(
                    (prod(a_reg) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(a_reg) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(a_reg)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles
                ip_reg_writes = total_cycles

                a_reg_reads = total_cycles
                a_reg_writes = ceil(total_cycles / input[0])

                op_reg_writes = total_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles
        else:
            # Example: [32, 512, 1024] * scalar = [32, 512, 1024]
            # MUL_Im
            input = node.saved_tensors[0]

            # tile size calculation
            dram_tile = ceil(
                (prod(input) * (definitions.PRECISION / 8)) /
                config["L2_Buffer"]
            )
            num_dram_tiles = ceil(prod(input) / dram_tile)

            l2_tile = ceil(
                dram_tile / (config["VC_PE"] *
                             config["Core_x"] * config["Core_y"])
            )
            num_l2_tiles = num_dram_tiles * l2_tile

            total_cycles = (
                ceil(
                    prod(input)
                    / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                )
                * PE_cycles
            )

            ip_reg_reads = total_cycles
            ip_reg_writes = total_cycles

            op_reg_writes = total_cycles

            l2_reads = num_l2_tiles
            l2_writes = num_l2_tiles

            dram_reads = num_dram_tiles
            dram_writes = num_dram_tiles

    # ====================== Div =======================#
    elif node.node_desc == "Div":
        # y = input / a_reg
        # ISA
        # DIV
        # Cycles = 10
        PE_cycles = 10
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        if len(nodes) == 2:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) == len(a_reg):
                # Example: [32, 512, 1024] / [32, 512, 1024] = [32, 512, 1024]
                if prod(input) > prod(
                    a_reg
                ):  # Example: [32, 512, 1024] * [32, 512, 1] = [32, 512, 1024]
                    x = input
                    y = a_reg
                elif prod(input) < prod(
                    a_reg
                ):  # Example: [32, 512, 1] * [32, 512, 1024] = [32, 512, 1024]
                    x = a_reg
                    y = a_reg
                else:  # Example: [32, 512, 1024] * [32, 512, 1024] = [32, 512, 1024]
                    x = input
                    y = a_reg

                # tile size calculation
                dram_tile = ceil(
                    (prod(x) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(x) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(x)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles / PE_cycles
                ip_reg_writes = total_cycles / PE_cycles

                a_reg_reads = total_cycles / PE_cycles
                a_reg_writes = ceil(
                    prod(y) / (config["VC_PE"] *
                               config["Core_x"] * config["Core_y"])
                )

                op_reg_writes = total_cycles / PE_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

            elif len(input) > len(a_reg):
                # Example: [32, 512, 1024] * [1024] = [32, 512, 1024] Element wise Multiplication along certain dimension (1024 dim)
                # tile size calculation
                dram_tile = ceil(
                    (prod(input) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(input) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(input)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles / PE_cycles
                ip_reg_writes = total_cycles / PE_cycles

                a_reg_reads = total_cycles / PE_cycles
                if len(a_reg) == 0:
                    # Assuming scalar value
                    a_reg_writes = 1
                else:
                    a_reg_writes = ceil(total_cycles / a_reg[0]) / PE_cycles

                op_reg_writes = total_cycles / PE_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

            else:
                # Example: [1024] * [32, 512, 1024]  = [32, 512, 1024] Element wise Multiplication along certain dimension (1024 dim)
                # tile size calculation
                dram_tile = ceil(
                    (prod(a_reg) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(a_reg) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(a_reg)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles / PE_cycles
                ip_reg_writes = total_cycles / PE_cycles

                a_reg_reads = total_cycles / PE_cycles
                a_reg_writes = ceil(total_cycles / input[0]) / PE_cycles

                op_reg_writes = total_cycles / PE_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles
        else:
            # Example: [32, 512, 1024] * scalar = [32, 512, 1024]
            # MUL_Im
            input = nodes[0].output_act[0]

            if len(input) == 0:
                total_cycles = 0
                ip_reg_reads = 0
                ip_reg_write = 0
                op_reg_write = 0
                dram_reads = 0
            else:
                # tile size calculation
                dram_tile = ceil(
                    (prod(input) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(input) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles = (
                    ceil(
                        prod(input)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles / PE_cycles
                ip_reg_writes = total_cycles / PE_cycles

                op_reg_writes = total_cycles / PE_cycles

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles
    # ====================== Mean =======================#
    elif node.node_desc == "Mean":
        # y = mean(x)
        # ISA
        # ADD, DIV
        # Cycles = mean_dimension
        input = schd.get_predecessor_nodes(node)[0]
        input = input.output_act[0]
        output = node.output_act[0]
        mean_dim = input[-1]

        PE_cycles = (
            ceil(mean_dim / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            + 10
        )

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        # tile size calculation
        dram_tile = ceil(
            (prod(input) * (definitions.PRECISION / 8)) / config["L2_Buffer"]
        )
        num_dram_tiles = ceil(prod(input) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = ceil(prod(output) / mean_dim) * PE_cycles

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Neg =======================#
    elif node.node_desc == "Neg":
        # y = Neg(x)
        # ISA
        # MUL, -1
        # Cycles = 1
        PE_cycles = 1

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Pow =======================#
    elif node.node_desc == "Pow":
        # y = pow(x)
        # ISA
        # MUL
        # Cycles = 15  # TODO: Check actual number of cycles required for power
        PE_cycles = 15

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Sqrt =======================#
    elif node.node_desc == "Sqrt":
        # y = sqrat(x)
        # ISA
        # SQRT
        # Cycles = 15  # TODO: Check actual number of cycles required for sqrt
        PE_cycles = 15

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== MaskedFill =======================#
    elif node.node_desc == "MaskedFill":
        # y = mask(y, value)

        PE_cycles = 1

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        op_reg_writes = total_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Relu =======================#
    elif node.node_desc == "Relu":
        # y = max(0, x)

        PE_cycles = 1

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        op_reg_writes = total_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Tanh, Hardtanh =======================#
    elif node.node_desc == "Tanh" or node.node_desc == "Hardtanh":
        # y = tanh(x)
        # y = hardtanh(x)

        PE_cycles = 27

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        ir_reg_reads = ceil(
            prod(output) / (config["VC_PE"] *
                            config["Core_x"] * config["Core_y"])
        )
        ir_reg_writes = ceil(
            prod(output) / (config["VC_PE"] *
                            config["Core_x"] * config["Core_y"])
        )

        lut_reads = ceil(prod(output) * 2)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Cos =======================#
    elif node.node_desc == "Cos" or node.node_desc == "Sin":
        # y = tanh(x)
        # y = hardtanh(x)

        PE_cycles = 13

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        ir_reg_reads = ceil(
            prod(output) / (config["VC_PE"] *
                            config["Core_x"] * config["Core_y"])
        )
        ir_reg_writes = ceil(
            prod(output) / (config["VC_PE"] *
                            config["Core_x"] * config["Core_y"])
        )

        lut_reads = ceil(prod(output) * 2)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Hardsigmoid, Sigmoid =======================#
    elif node.node_desc == "Hardsigmoid" or node.node_desc == "Sigmoid":
        # y = Sigmoid(x)

        PE_cycles = 12

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        ir_reg_reads = ceil(
            prod(output) / (config["VC_PE"] *
                            config["Core_x"] * config["Core_y"])
        )
        ir_reg_writes = ceil(
            prod(output) / (config["VC_PE"] *
                            config["Core_x"] * config["Core_y"])
        )

        lut_reads = ceil(prod(output))

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Hardswish, Swiss =======================#
    elif node.node_desc == "Swiss" or node.node_desc == "Hardswish":
        # y = Swiss(x)

        PE_cycles = 14

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        ir_reg_reads = ceil(
            prod(output) / (config["VC_PE"] *
                            config["Core_x"] * config["Core_y"])
        )
        ir_reg_writes = ceil(
            prod(output) / (config["VC_PE"] *
                            config["Core_x"] * config["Core_y"])
        )

        lut_reads = ceil(prod(output))

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== MaxPool2DWithIndices =======================#
    elif node.node_desc == "MaxPool2DWithIndices":
        # out(N, C, h, w) = max(m=0,1,..,kH-1) max(n=0,1,..,kW-1) input(N, C, stride[0]*h+m, stride[1]*w+n)
        # ISA
        # COMP
        # Cycles = kernel[0] * kernel[1]
        kernel = node.kernel
        stride = node.stride
        if type(kernel) is int:
            PE_cycles = kernel * kernel
        else:
            PE_cycles = kernel[0] * kernel[1]

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]
        input = node.saved_tensors[0]
        N = input[0]
        C = input[1]
        H = input[2]
        W = input[3]

        x = ceil((W - kernel[0] + 1) / stride[0])
        y = ceil((H - kernel[1] + 1) / stride[1])

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(
                (N * C * x * y)
                / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
            )
            * PE_cycles
        )

        a_reg_reads = total_cycles
        a_reg_writes = total_cycles

        b_reg_reads = ceil(
            (N * C * x * y) / (config["VC_PE"] *
                               config["Core_x"] * config["Core_y"])
        )
        b_reg_writes = ceil(
            (N * C * x * y) / (config["VC_PE"] *
                               config["Core_x"] * config["Core_y"])
        )

        op_reg_writes = prod(output)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== AvgPool2D =======================#
    elif node.node_desc == "AvgPool2D":
        # out(N, C, h, w) = (1/kH*kW) max(m=0,1,..,kH-1) max(n=0,1,..,kW-1) input(N, C, stride[0]*h+m, stride[1]*w+n)
        # ISA
        # COMP
        # Cycles = kernel[0] * kernel[1]
        kernel = node.kernel
        stride = node.stride
        PE_cycles = kernel[0] * kernel[1]

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]
        input = node.saved_tensors[0]
        N = input[0]
        C = input[1]
        H = input[2]
        W = input[3]

        x = ceil((W - kernel[0] + 1) / stride[0])
        y = ceil((H - kernel[1] + 1) / stride[1])

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(
                (N * C * x * y)
                / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
            )
            * PE_cycles
        )

        a_reg_reads = total_cycles
        a_reg_writes = total_cycles

        b_reg_reads = ceil(
            (N * C * x * y) / (config["VC_PE"] *
                               config["Core_x"] * config["Core_y"])
        )
        b_reg_writes = ceil(
            (N * C * x * y) / (config["VC_PE"] *
                               config["Core_x"] * config["Core_y"])
        )

        op_reg_writes = prod(output)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== AdaptiveAveragePool2D =======================#
    elif (
        node.node_desc == "AdaptiveAveragePool2D"
        or node.node_desc == "AdaptiveAvgPool2D"
    ):
        # out(N, C, h, w) = max(m=0,1,..,kH-1) max(n=0,1,..,kW-1) input(N, C, stride[0]*h+m, stride[1]*w+n)
        # stride and kernel calculated using input and output size
        # ISA
        # COMP
        # Cycles = kernel[0] * kernel[1]

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        input = schd.get_predecessor_nodes(node)[0]
        input = input.output_act[0]
        input_size = prod(input)
        output = node.output_act[0]
        output_size = prod(output)
        stride = []
        stride_ = input_size // output_size
        stride.append(stride_)
        stride.append(stride_)
        kernel = []
        kernel_ = input_size - (output_size - 1) * stride[0]
        kernel.append(kernel_)
        kernel.append(kernel_)
        PE_cycles = kernel[0] * kernel[1]

        N = input[0]
        C = input[1]
        H = input[2]
        W = input[3]

        x = ceil((W - kernel[0] + 1) / stride[0])
        y = ceil((H - kernel[1] + 1) / stride[1])

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(
                (N * C * x * y)
                / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
            )
            * PE_cycles
        )

        a_reg_reads = total_cycles
        a_reg_writes = total_cycles

        b_reg_reads = ceil(
            (N * C * x * y) / (config["VC_PE"] *
                               config["Core_x"] * config["Core_y"])
        )
        b_reg_writes = ceil(
            (N * C * x * y) / (config["VC_PE"] *
                               config["Core_x"] * config["Core_y"])
        )

        op_reg_writes = prod(output)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== NativeDropout, FusedDropout =======================#
    elif node.node_desc == "NativeDropout" or node.node_desc == "FusedDropout":
        # y = x * (1/1-p)
        # ISA
        # RAND, COMP, ADD, MUL
        # Cycles = 1 (RAND, COMP, ADD) + 1(MUL)
        PE_cycles = 2

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        b_reg_reads = total_cycles / PE_cycles
        b_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== LogSoftmax =======================#
    elif node.node_desc == "LogSoftmax":
        # y = log(e^x/sum(e^x))

        PE_cycles = 13

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]
        n = output[-1]

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        # cycles(e^x) +  cycles(sum(e^x)) + cyles(e^x/sum(e^x))
        # sum(e^x)
        total_cycles = (
            ceil(prod(output) / n)
            * ceil(n / (config["VC_PE"] * config["Core_x"] * config["Core_y"]))
            + 10
        )
        # e^x / sum(e^x)
        total_cycles += (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = (
            total_cycles
            / ceil(n / (config["VC_PE"] * config["Core_x"] * config["Core_y"]))
            + 10
        )

        op_reg_writes = (
            total_cycles
            / ceil(n / (config["VC_PE"] * config["Core_x"] * config["Core_y"]))
            + 10
        )

        ip_reg_reads += num_l2_tiles * PE_cycles
        ip_reg_writes += num_l2_tiles

        op_reg_writes += num_l2_tiles

        lut_reads = ceil(prod(output))

        l2_reads = 2 * num_l2_tiles
        l2_writes = 2 * num_l2_tiles

        dram_reads = 2 * num_dram_tiles
        dram_writes = 2 * num_dram_tiles

    # ====================== Softmax =======================#
    elif node.node_desc == "Softmax":
        # y = e^x/sum(e^x)

        PE_cycles = 10

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]
        n = output[-1]

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        # cycles(e^x) +  cycles(sum(e^x)) + cyles(e^x/sum(e^x))
        # sum(e^x)
        total_cycles = (
            ceil(prod(output) / n)
            * ceil(n / (config["VC_PE"] * config["Core_x"] * config["Core_y"]))
            + 10
        )
        # e^x / sum(e^x)
        total_cycles += (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = (
            total_cycles
            / ceil(n / (config["VC_PE"] * config["Core_x"] * config["Core_y"]))
            + 10
        )

        op_reg_writes = (
            total_cycles
            / ceil(n / (config["VC_PE"] * config["Core_x"] * config["Core_y"]))
            + 10
        )

        ip_reg_reads += num_l2_tiles * PE_cycles
        ip_reg_writes += num_l2_tiles

        op_reg_writes += num_l2_tiles

        lut_reads = ceil(prod(output))

        l2_reads = 2 * num_l2_tiles
        l2_writes = 2 * num_l2_tiles

        dram_reads = 2 * num_dram_tiles
        dram_writes = 2 * num_dram_tiles

    # ====================== NllLoss =======================#
    elif node.node_desc == "NllLoss":
        # ln = xn, yn = yn - xn
        # l(x, y) = sum(ln)
        # ISA
        # SUB, ADD
        # Cycles = 1 + 1  = 2

        PE_cycles = 2

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        # TODO: for phaze, check if the order is correct
        x = node.saved_tensors[0]
        y = node.saved_tensors[1]

        dram_tile = ceil(
            (prod(x) * (definitions.PRECISION / 8)) / config["L2_Buffer"])
        num_dram_tiles = ceil(prod(x) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(x) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        a_reg_reads = total_cycles / PE_cycles
        a_reg_writes = total_cycles / PE_cycles

        op_reg_writes = prod(x)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Erf =======================#
    elif node.node_desc == "Erf":
        # TO DO:
        # Cycles = 15 assumption

        PE_cycles = 15

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        x = node.output_act[0]

        dram_tile = ceil(
            (prod(x) * (definitions.PRECISION / 8)) / config["L2_Buffer"])
        num_dram_tiles = ceil(prod(x) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(x) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        op_reg_writes = prod(x)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    else:
        print("Node ", node.node_desc, " not found!!")
        sys.exit()

    if(total_cycles == -10):
        raise Exception("Total cycles not calculated")

    mac_opr = total_cycles
    access_count = (
        mac_opr,
        dram_reads,
        dram_writes,
        l2_reads,
        l2_writes,
        lut_reads,
        ip_reg_reads,
        ip_reg_writes,
        a_reg_reads,
        a_reg_writes,
        b_reg_reads,
        b_reg_writes,
        op_reg_reads,
        op_reg_writes,
        ir_reg_reads,
        ir_reg_writes,
    )
    total_energy = get_node_energy(schd, access_count, config)

    output = {"utilization": 1, "cycles": total_cycles,
              "energy_pJ": total_energy}

    return output


def get_bwd_estimation(schd, node, config, total_cycles=-10):
    # ==================== CudnnBatchNorm, NativeBatchNorm =======================#
    if node.node_desc == "CudnnBatchNorm" or node.node_desc == "NativeBatchNorm":
        # dy/dx = gamma
        # Cycles = 1
        PE_cycles = 1
        output = node.output_act[0]

        # TODO: for phaze, check if the order is correct
        gamma = node.saved_tensors[1][0]
        beta = node.saved_tensors[2][0]

        N = output[0]
        C = output[1]
        P = output[2]
        Q = output[3]

        assert C == gamma and C == beta, "Dimensions doesn't match!!"

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        op_reg_writes = total_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== CudnnLayerNorm, NativeBatchNorm =======================#
    elif node.node_desc == "CudnnLayerNorm" or node.node_desc == "NativeLayerNorm":
        # dy/dx = gamma
        # Cycles = 1
        PE_cycles = 1
        output = node.output_act[0]

        gamma = int(node.saved_tensors[0][-1])
        beta = int(node.saved_tensors[1][-1])

        if len(output) == 3:
            N = output[0]
            seq_len = output[1]
            dim = int(output[2])
        elif len(output) == 2:
            N = 1
            seq_len = output[0]
            dim = int(output[1])

        assert dim == gamma == beta, "Dimensions doesn't match!!"

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles
        a_reg_reads = total_cycles
        a_reg_writes = gamma
        b_reg_reads = total_cycles
        b_reg_writes = beta
        op_reg_writes = total_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Add =======================#
    elif node.node_desc == "Add":
        # dy/dx_1 = 1 + d(x_2)/dx_1
        # dy/dx_2 = d(x_1)/dx_2 + 1
        # Cycles = 1
        PE_cycles = 1
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        if len(nodes) == 2:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) != 0 and len(a_reg) != 0:
                # Example: [32, 512, 1024] + [32, 512, 1024] = [32, 512, 1024]
                if prod(input) > prod(
                    a_reg
                ):  # Example: [32, 512, 1024] + [32, 512, 1] = [32, 512, 1024]
                    x = input
                    y = a_reg
                elif prod(input) < prod(
                    a_reg
                ):  # Example: [32, 512, 1] + [32, 512, 1024] = [32, 512, 1024]
                    x = a_reg
                    y = a_reg
                else:  # Example: [32, 512, 1024] + [32, 512, 1024] = [32, 512, 1024]
                    x = input
                    y = a_reg

                # x1 tile size calculation
                dram_tile = ceil(
                    (prod(x) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(x) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles_1 = (
                    ceil(
                        prod(x)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles_1
                ip_reg_writes = total_cycles_1

                op_reg_writes = total_cycles_1

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

                # x2 tile size calculation
                dram_tile = ceil(
                    (prod(y) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(y) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles_2 = (
                    ceil(
                        prod(y)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads += total_cycles_2
                ip_reg_writes += total_cycles_2

                op_reg_writes += total_cycles_2

                l2_reads += num_l2_tiles
                l2_writes += num_l2_tiles

                dram_reads += num_dram_tiles
                dram_writes += num_dram_tiles

                total_cycles = total_cycles_1 + total_cycles_2

            elif len(input) == 0 and len(a_reg) == 0:
                total_cycles = 0
        else:
            # Example: [32, 512, 1024] + scalar = [32, 512, 1024]
            # ADD_Im
            input = nodes[0].output_act[0]

            dram_tile = ceil(
                (prod(input) * (definitions.PRECISION / 8)) /
                config["L2_Buffer"]
            )
            num_dram_tiles = ceil(prod(input) / dram_tile)

            l2_tile = ceil(
                dram_tile / (config["VC_PE"] *
                             config["Core_x"] * config["Core_y"])
            )
            num_l2_tiles = num_dram_tiles * l2_tile

            total_cycles = (
                ceil(
                    prod(input)
                    / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                )
                * PE_cycles
            )

            ip_reg_reads = total_cycles
            ip_reg_writes = total_cycles

            op_reg_writes = total_cycles

            l2_reads = num_l2_tiles
            l2_writes = num_l2_tiles

            dram_reads = num_dram_tiles
            dram_writes = num_dram_tiles

    # ====================== Sub =======================#
    elif node.node_desc == "Sub":
        # dy/dx_1 = 1 - d(x_2)/dx_1
        # dy/dx_2 = d(x_1)/dx_2 - 1
        PE_cycles = 1
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0
        # print(node)
        if len(nodes) == 2:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) != 0 and len(a_reg) != 0:
                # Example: [32, 512, 1024] - [32, 512, 1024] = [32, 512, 1024]
                if prod(input) > prod(
                    a_reg
                ):  # Example: [32, 512, 1024] - [32, 512, 1] = [32, 512, 1024]
                    x = input
                    y = a_reg
                elif prod(input) < prod(
                    a_reg
                ):  # Example: [32, 512, 1] - [32, 512, 1024] = [32, 512, 1024]
                    x = a_reg
                    y = a_reg
                else:  # Example: [32, 512, 1024] - [32, 512, 1024] = [32, 512, 1024]
                    x = input
                    y = a_reg

                # x1 tile size calculation
                dram_tile = ceil(
                    (prod(x) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(x) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles_1 = (
                    ceil(
                        prod(x)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles_1
                ip_reg_writes = total_cycles_1

                op_reg_writes = total_cycles_1

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

                # x2 tile size calculation
                dram_tile = ceil(
                    (prod(y) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(y) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles_2 = (
                    ceil(
                        prod(y)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads += total_cycles_2
                ip_reg_writes += total_cycles_2

                op_reg_writes += total_cycles_2

                l2_reads += num_l2_tiles
                l2_writes += num_l2_tiles

                dram_reads += num_dram_tiles
                dram_writes += num_dram_tiles

                total_cycles = total_cycles_1 + total_cycles_2

            elif len(input) == 0 and len(a_reg) == 0:
                total_cycles = 0
        else:
            # Example: [32, 512, 1024] - scalar = [32, 512, 1024]
            # SUB_Im
            input = nodes[0].output_act[0]

            dram_tile = ceil(
                (prod(input) * (definitions.PRECISION / 8)) /
                config["L2_Buffer"]
            )
            num_dram_tiles = ceil(prod(input) / dram_tile)

            l2_tile = ceil(
                dram_tile / (config["VC_PE"] *
                             config["Core_x"] * config["Core_y"])
            )
            num_l2_tiles = num_dram_tiles * l2_tile

            total_cycles = (
                ceil(
                    prod(input)
                    / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                )
                * PE_cycles
            )

            ip_reg_reads = total_cycles
            ip_reg_writes = total_cycles

            op_reg_writes = total_cycles

            l2_reads = num_l2_tiles
            l2_writes = num_l2_tiles

            dram_reads = num_dram_tiles
            dram_writes = num_dram_tiles

    # ====================== Mul =======================#
    elif node.node_desc == "Mul":
        # dy/dx_1 = x_2
        # dy/dx_2 = x_1
        # Cycles = 1
        PE_cycles = 1
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        if len(nodes) == 2 and len(nodes[1].output_act[0]) > 0:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) != 0 and len(a_reg) != 0:
                # Example: [32, 512, 1024] * [32, 512, 1024] = [32, 512, 1024]
                if prod(input) > prod(
                    a_reg
                ):  # Example: [32, 512, 1024] * [32, 512, 1] = [32, 512, 1024]
                    x = input
                    y = a_reg
                elif prod(input) < prod(
                    a_reg
                ):  # Example: [32, 512, 1] * [32, 512, 1024] = [32, 512, 1024]
                    x = a_reg
                    y = a_reg
                else:  # Example: [32, 512, 1024] * [32, 512, 1024] = [32, 512, 1024]
                    x = input
                    y = a_reg

                # x1 tile size calculation
                dram_tile = ceil(
                    (prod(x) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(x) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles_1 = (
                    ceil(
                        prod(x)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads = total_cycles_1
                ip_reg_writes = total_cycles_1

                op_reg_writes = total_cycles_1

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

                # x2 tile size calculation
                dram_tile = ceil(
                    (prod(y) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(y) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles_2 = (
                    ceil(
                        prod(y)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles
                )

                ip_reg_reads += total_cycles_2
                ip_reg_writes += total_cycles_2

                op_reg_writes += total_cycles_2

                l2_reads += num_l2_tiles
                l2_writes += num_l2_tiles

                dram_reads += num_dram_tiles
                dram_writes += num_dram_tiles

                total_cycles = total_cycles_1 + total_cycles_2

            elif len(input) == 0 and len(a_reg) == 0:
                total_cycles = 0
        else:
            # Example: [32, 512, 1024] * scalar = [32, 512, 1024]
            # MUL_Im
            input = node.saved_tensors[0]

            dram_tile = ceil(
                (prod(input) * (definitions.PRECISION / 8)) /
                config["L2_Buffer"]
            )
            num_dram_tiles = ceil(prod(input) / dram_tile)

            l2_tile = ceil(
                dram_tile / (config["VC_PE"] *
                             config["Core_x"] * config["Core_y"])
            )
            num_l2_tiles = num_dram_tiles * l2_tile

            total_cycles = (
                ceil(
                    prod(input)
                    / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                )
                * PE_cycles
            )

            ip_reg_reads = total_cycles
            ip_reg_writes = total_cycles

            op_reg_writes = total_cycles

            l2_reads = num_l2_tiles
            l2_writes = num_l2_tiles

            dram_reads = num_dram_tiles
            dram_writes = num_dram_tiles

    # ====================== Div =======================#
    elif node.node_desc == "Div":
        # dy/dx_1 = 1 / x_2
        # dy_dx_2 = - x_1/x_2^(2)
        # ISA
        # DIV
        # Cycles = 10
        PE_cycles_1 = 10
        PE_cycles_2 = 11
        nodes = schd.get_predecessor_nodes(node)

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        total_c = 0

        if len(nodes) == 2:
            input = nodes[0].output_act[0]
            a_reg = nodes[1].output_act[0]

            if len(input) != 0 and len(a_reg) != 0:
                # Example: [32, 512, 1024] * [32, 512, 1024] = [32, 512, 1024]
                if prod(input) > prod(
                    a_reg
                ):  # Example: [32, 512, 1024] * [32, 512, 1] = [32, 512, 1024]
                    x = input
                    y = a_reg
                elif prod(input) < prod(
                    a_reg
                ):  # Example: [32, 512, 1] * [32, 512, 1024] = [32, 512, 1024]
                    x = a_reg
                    y = a_reg
                else:  # Example: [32, 512, 1024] * [32, 512, 1024] = [32, 512, 1024]
                    x = input
                    y = a_reg

                # x1 tile size calculation
                dram_tile = ceil(
                    (prod(x) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(x) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles_1 = (
                    ceil(
                        prod(x)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles_1
                )

                ip_reg_reads = total_cycles_1
                ip_reg_writes = total_cycles_1

                op_reg_writes = total_cycles_1

                l2_reads = num_l2_tiles
                l2_writes = num_l2_tiles

                dram_reads = num_dram_tiles
                dram_writes = num_dram_tiles

                # x2 tile size calculation
                dram_tile = ceil(
                    (prod(y) * (definitions.PRECISION / 8)) /
                    config["L2_Buffer"]
                )
                num_dram_tiles = ceil(prod(y) / dram_tile)

                l2_tile = ceil(
                    dram_tile / (config["VC_PE"] *
                                 config["Core_x"] * config["Core_y"])
                )
                num_l2_tiles = num_dram_tiles * l2_tile

                total_cycles_2 = (
                    ceil(
                        prod(y)
                        / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                    )
                    * PE_cycles_2
                )

                ip_reg_reads += total_cycles_2
                ip_reg_writes += total_cycles_2

                op_reg_writes += total_cycles_2

                l2_reads += num_l2_tiles
                l2_writes += num_l2_tiles

                dram_reads += num_dram_tiles
                dram_writes += num_dram_tiles

                total_c = total_cycles_1 + total_cycles_2

            elif len(input) == 0 and len(a_reg) == 0:
                total_c = 0
        else:
            # Example: [32, 512, 1024] * scalar = [32, 512, 1024]
            # MUL_Im
            input = node.output_act[0]

            dram_tile = ceil(
                (prod(input) * (definitions.PRECISION / 8)) /
                config["L2_Buffer"]
            )
            num_dram_tiles = ceil(prod(input) / dram_tile)

            l2_tile = ceil(
                dram_tile / (config["VC_PE"] *
                             config["Core_x"] * config["Core_y"])
            )
            num_l2_tiles = num_dram_tiles * l2_tile

            total_c = (
                ceil(
                    prod(input)
                    / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
                )
                * PE_cycles_1
            )

            ip_reg_reads = total_c
            ip_reg_writes = total_c

            op_reg_writes = total_c

            l2_reads = num_l2_tiles
            l2_writes = num_l2_tiles

            dram_reads = num_dram_tiles
            dram_writes = num_dram_tiles

        total_cycles = total_c

    # ====================== Mean =======================#
    elif node.node_desc == "Mean":
        # dy/dx = d(mean(x))/dx
        # Cycles = mean_dimension
        # TODO: Check actual derivative
        input = schd.get_predecessor_nodes(node)[0]
        input = input.output_act[0]
        output = node.output_act[0]
        mean_dim = input[-1]

        PE_cycles = (
            ceil(mean_dim / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            + 10
        )

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        # tile size calculation
        dram_tile = ceil(
            (prod(input) * (definitions.PRECISION / 8)) / config["L2_Buffer"]
        )
        num_dram_tiles = ceil(prod(input) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = ceil(prod(output) / mean_dim) * PE_cycles

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Neg =======================#
    elif node.node_desc == "Neg":
        # dy/dx = 1
        # Cycles = 1
        PE_cycles = 1

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Pow =======================#
    elif node.node_desc == "Pow":
        # dy/dx = d(x^n)/dx = n*x^(n-1)
        # Cycles = 1% + 1  # TODO: Check actual number of cycles required for power
        PE_cycles = 16

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Sqrt =======================#
    elif node.node_desc == "Sqrt":
        # dy/dx = d(x^0.5)/dx = 0.5 * x^(-0.5)
        # ISA
        # SQRT
        # Cycles = 15 + 1  # TODO: Check actual number of cycles required for sqrt
        PE_cycles = 16

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== MaskedFill =======================#
    elif node.node_desc == "MaskedFill":
        # dy/dx = 0, 1

        PE_cycles = 1

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        op_reg_writes = total_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Relu =======================#
    elif node.node_desc == "Relu":
        # dy/dx = 0, 1

        PE_cycles = 1

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        op_reg_writes = total_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Tanh, Hardtanh =======================#
    elif node.node_desc == "Tanh" or node.node_desc == "Hardtanh":
        # dy/dx = 1 - S^(2)
        # Cycles = 2

        PE_cycles = 2

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Tanh, Hardtanh =======================#
    elif node.node_desc == "Cos" or node.node_desc == "Sin":
        # dy/dx = -Sin(x)
        # Cycles = 2

        PE_cycles = 2

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Hardsigmoid, Sigmoid =======================#
    elif node.node_desc == "Hardsigmoid" or node.node_desc == "Sigmoid":
        # dy/dx = S(1 - S)
        # Cycles = 2

        PE_cycles = 22

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Hardswish, Swiss =======================#
    elif node.node_desc == "Swiss" or node.node_desc == "Hardswish":
        # dy/dx = S / x + S(1 - S/x)
        # Cycles = 15
        #
        PE_cycles = 15

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        # tile size calculation
        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== MaxPool2DWithIndices =======================#
    elif node.node_desc == "MaxPool2DWithIndices" or node.node_desc == "AvgPool2D":
        # dy/dx = x
        # Cycles = 1
        PE_cycles = 1

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        a_reg_reads = total_cycles
        a_reg_writes = total_cycles

        op_reg_writes = prod(output)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== AdaptiveAveragePool2D =======================#
    elif (
        node.node_desc == "AdaptiveAveragePool2D"
        or node.node_desc == "AdaptiveAvgPool2D"
    ):
        # dy/dx = x
        # Cycles = kernel[0] * kernel[1]

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        input = schd.get_predecessor_nodes(node)[0]
        input = input.output_act[0]
        input_size = prod(input)
        output = node.output_act[0]
        output_size = prod(output)
        stride = []
        stride_ = input_size // output_size
        stride.append(stride_)
        stride.append(stride_)
        kernel = []
        kernel_ = input_size - (output_size - 1) * stride[0]
        kernel.append(kernel_)
        kernel.append(kernel_)
        PE_cycles = 1

        dram_tile = ceil(input_size * (definitions.PRECISION / 8)
                         ) / config["L2_Buffer"]
        num_dram_tiles = ceil(input_size / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(input_size / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        a_reg_reads = total_cycles
        a_reg_writes = total_cycles

        op_reg_writes = input_size

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== NativeDropout, FusedDropout =======================#
    elif node.node_desc == "NativeDropout" or node.node_desc == "FusedDropout":
        # dy/dx = x

        PE_cycles = 1

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        b_reg_reads = total_cycles / PE_cycles
        b_reg_writes = total_cycles / PE_cycles

        op_reg_writes = total_cycles / PE_cycles

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== LogSoftmax =======================#
    elif node.node_desc == "LogSoftmax":
        # dy/dx = S_i(1-S_j) , -S_i*S_j

        PE_cycles = 2

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]
        n = output[-1]

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = ceil(total_cycles / PE_cycles)

        op_reg_writes = ceil(total_cycles / PE_cycles)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Softmax =======================#
    elif node.node_desc == "Softmax":
        # dy/dx = S_i(1-S_j) , -S_i*S_j

        PE_cycles = 2

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        output = node.output_act[0]
        n = output[-1]

        dram_tile = ceil(
            (prod(output) * (definitions.PRECISION / 8))
            / (config["L2_Buffer"] * config["Core_x"] * config["Core_y"])
        )
        num_dram_tiles = ceil(prod(output) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(output) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = ceil(total_cycles / PE_cycles)

        op_reg_writes = ceil(total_cycles / PE_cycles)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== NllLoss =======================#
    elif node.node_desc == "NllLoss":
        # ln = xn, yn = yn - xn
        # l(x, y) = sum(ln)
        # ISA
        # SUB, ADD
        # Cycles = 1 + 1  = 2

        PE_cycles = 2

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        # TODO: for phaze check if order is correct, saved_tensor [0] is input and [1] is weight
        x = node.saved_tensors[0]
        y = node.saved_tensors[1]

        dram_tile = ceil(
            (prod(x) * (definitions.PRECISION / 8)) / config["L2_Buffer"])
        num_dram_tiles = ceil(prod(x) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(x) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        a_reg_reads = total_cycles / PE_cycles
        a_reg_writes = total_cycles / PE_cycles

        op_reg_writes = prod(x)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    # ====================== Erf =======================#
    elif node.node_desc == "Erf":
        # TO DO:
        # Cycles = 15 assumption

        PE_cycles = 15

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        x = node.output_act[0]

        dram_tile = ceil(
            (prod(x) * (definitions.PRECISION / 8)) / config["L2_Buffer"])
        num_dram_tiles = ceil(prod(x) / dram_tile)

        l2_tile = ceil(
            dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"])
        )
        num_l2_tiles = num_dram_tiles * l2_tile

        total_cycles = (
            ceil(prod(x) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles
        ip_reg_writes = total_cycles

        op_reg_writes = prod(x)

        l2_reads = num_l2_tiles
        l2_writes = num_l2_tiles

        dram_reads = num_dram_tiles
        dram_writes = num_dram_tiles

    else:
        print("Node ", node.node_desc, " not found!!")
        sys.exit()

    if(total_cycles == -10):
        raise "Error in scheduling. total cycles was never set!"

    mac_opr = total_cycles
    access_count = (
        mac_opr,
        dram_reads,
        dram_writes,
        l2_reads,
        l2_writes,
        lut_reads,
        ip_reg_reads,
        ip_reg_writes,
        a_reg_reads,
        a_reg_writes,
        b_reg_reads,
        b_reg_writes,
        op_reg_reads,
        op_reg_writes,
        ir_reg_reads,
        ir_reg_writes,
    )
    total_energy = get_node_energy(schd, access_count, config)

    output = {"utilization": 1, "cycles": total_cycles,
              "energy_pJ": total_energy}

    return output


def get_weight_update_estimation(schd, num_weights, config):

    # dy/dw = x * dy/dx
    # Cycles = 1
    PE_cycles = 1

    output = num_weights

    dram_reads = 0
    dram_writes = 0

    l2_reads = 0
    l2_writes = 0

    lut_reads = 0

    ip_reg_writes = 0
    ip_reg_reads = 0
    a_reg_writes = 0
    a_reg_reads = 0
    b_reg_writes = 0
    b_reg_reads = 0
    op_reg_writes = 0
    op_reg_reads = 0
    ir_reg_writes = 0
    ir_reg_reads = 0

    # tile size calculation
    dram_tile = ceil(num_weights * (definitions.PRECISION / 8)
                     ) / config["L2_Buffer"]
    num_dram_tiles = ceil(num_weights / dram_tile)

    l2_tile = ceil(
        dram_tile / (config["VC_PE"] * config["Core_x"] * config["Core_y"]))
    num_l2_tiles = num_dram_tiles * l2_tile

    total_cycles = (
        ceil(num_weights / (config["VC_PE"] *
             config["Core_x"] * config["Core_y"]))
        * PE_cycles
    )

    ip_reg_reads = total_cycles
    ip_reg_writes = total_cycles

    op_reg_writes = total_cycles

    l2_reads = num_l2_tiles
    l2_writes = num_l2_tiles

    dram_reads = num_dram_tiles
    dram_writes = num_dram_tiles

    mac_opr = total_cycles
    access_count = (
        mac_opr,
        dram_reads,
        dram_writes,
        l2_reads,
        l2_writes,
        lut_reads,
        ip_reg_reads,
        ip_reg_writes,
        a_reg_reads,
        a_reg_writes,
        b_reg_reads,
        b_reg_writes,
        op_reg_reads,
        op_reg_writes,
        ir_reg_reads,
        ir_reg_writes,
    )
    total_energy = get_node_energy(schd, access_count, config)

    output = {"utilization": 1, "cycles": total_cycles,
              "energy_pJ": total_energy}

    return output


def get_opt_est(schd, node, optimizer, config):
    if optimizer == "SGD":
        # SGD optimizer
        # p, g, v and μ denote the parameters, gradient, velocity, and momentum respectively.
        # v(t+1) = μ * v(t) + g(t+1)
        # p(t+1) = p(t) - lr * v(t+1)
        # ISA
        # MUL, ADD
        # Cycles = 2
        PE_cycles = 2

        dram_reads = 0
        dram_writes = 0

        l2_reads = 0
        l2_writes = 0

        lut_reads = 0

        ip_reg_writes = 0
        ip_reg_reads = 0
        a_reg_writes = 0
        a_reg_reads = 0
        b_reg_writes = 0
        b_reg_reads = 0
        op_reg_writes = 0
        op_reg_reads = 0
        ir_reg_writes = 0
        ir_reg_reads = 0

        param = node.parameter
        total_cycles = (
            ceil(prod(param) / (config["VC_PE"] *
                 config["Core_x"] * config["Core_y"]))
            * PE_cycles
        )

        ip_reg_reads = total_cycles / PE_cycles
        ip_reg_writes = total_cycles / PE_cycles

        a_reg_reads = total_cycles
        a_reg_writes = total_cycles

        b_reg_reads = total_cycles
        b_reg_writes = total_cycles

        op_reg_writes = prod(param)

        dram_reads = prod(param)
        # dram_writes = prod(param)
    else:
        print("Optimizer not supported!!")
        sys.exit()

    mac_opr = total_cycles
    access_count = (
        mac_opr,
        dram_reads,
        dram_writes,
        l2_reads,
        l2_writes,
        lut_reads,
        ip_reg_reads,
        ip_reg_writes,
        a_reg_reads,
        a_reg_writes,
        b_reg_reads,
        b_reg_writes,
        op_reg_reads,
        op_reg_writes,
        ir_reg_reads,
        ir_reg_writes,
    )
    total_energy = get_node_energy(schd, access_count, config)

    output = {"utilization": 1, "cycles": total_cycles,
              "energy_pJ": total_energy}

    return output
