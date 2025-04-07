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

# internal phaze imports
from .ilp import latency_per_layer
from .baseline_configs import baseline_config

from phaze.Estimator import get_configs_to_explore
from phaze.Estimator import bandwidth, num_accelerators, bytes_per_element

from phaze.GraphExtractor import generate_out_filename

# External imports
from collections import namedtuple, OrderedDict

# External python imports
import os
import subprocess
import json
import time

# Global Variables for Model Training
pipeline_strategy = "pipedream-flush"
optimizer_type = "SGD"
global_batch_size = 4096
acc_group_range = 10000

# distributed strategy
dist_strategy = namedtuple("dist_config", ["ppdepth", "tmpcwidth", "dpwidth"])

# dictionary of explored accelerator configs
explored_acc_configs = {}

# tuple for the accelerator config and the corresponding throughput
out_per_cc_per_model = namedtuple(
    "out_per_cc_per_model", ["dist_strategy", "throughput", "fixed_strategy_throughput", "utilization"])

# dictionary of sorted accelerators by area
acc_sorted = {}
max_thpt_per_area = {}

estimation_time = 0
dp_time = 0
ilp_time = 0


def sort_acc_configs_by_area():
    global acc_sorted
    global max_thpt_per_area
    global acc_group_range

    acc_configs = get_configs_to_explore()
    for cc in acc_configs:
        area_range = int(cc.area / acc_group_range)
        if area_range not in acc_sorted.keys():
            acc_sorted[area_range] = []
            max_thpt_per_area[area_range] = -1
        acc_sorted[area_range].append(cc)

    acc_sorted = OrderedDict(sorted(acc_sorted.items(), reverse=True))


def get_next_acc_config(prev_cc, cc_iter):
    global acc_group_range

    converged = False
    if_all_archs_to_explore = False
    hysterisis_level = 5

    curr_area = prev_cc.area
    area_range = int(curr_area / acc_group_range)
    curr_area_idx = list(acc_sorted.keys()).index(area_range)

    if prev_cc in explored_acc_configs.keys():
        curr_avg_t = sum(o.throughput for m_name, o in explored_acc_configs[prev_cc].items(
        )) / len(explored_acc_configs[prev_cc].keys())
        max_thpt_per_area[area_range] = max(
            max_thpt_per_area[area_range], curr_avg_t)

    try:
        return False, next(cc_iter), cc_iter
    except:
        try:
            next_area = list(acc_sorted.keys())[curr_area_idx + 1]
            cc_iter = iter(acc_sorted[next_area])

            if if_all_archs_to_explore:
                return False, next(cc_iter), cc_iter

            larger_area_thgpt = [max_thpt_per_area[area]
                                 for area in list(acc_sorted.keys())[:curr_area_idx + 1]]

            if len(larger_area_thgpt) < hysterisis_level:
                return False, next(cc_iter), cc_iter
            larger_area_thgpt = larger_area_thgpt[-hysterisis_level:]
            converged = all(earlier >= later for earlier, later in zip(
                larger_area_thgpt, larger_area_thgpt[1:]))

            return converged, next(cc_iter), cc_iter

        except:
            print("All architectures explored")
            return True, None, None


def find_config_across_models():
    global explored_acc_configs
    highest_avg_throughput = 0

    best_acc_config = None

    for cc, out in explored_acc_configs.items():
        curr_throughput = sum(o.throughput for m_name,
                              o in out.items()) / len(out.keys())
        if curr_throughput > highest_avg_throughput:
            highest_avg_throughput = curr_throughput
            best_acc_config = cc

    if best_acc_config is None:
        print("\033[91m" + "No config found in the whole search space" + "\033[0m")
        return None, None

    return best_acc_config, explored_acc_configs[best_acc_config]


def write_explored_configs_to_file(model_name, micro_batch_size, max_tmp_width, sequence_length, activation_recomputation, hbm_size):
    global explored_acc_configs
    module_type_str = "AR_" + \
        str(activation_recomputation) + "_HBM_" + \
        str(hbm_size) + "_phaze_solved"
    out_file = generate_out_filename(
        model_name, "json", micro_batch_size, max_tmp_width, sequence_length, module_type_str)

    out_dir = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), "output")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_file = os.path.join(out_dir, out_file)

    writeable_explored_acc_configs = {}

    idx = 0
    for cc, out in explored_acc_configs.items():
        writeable_explored_acc_configs[idx] = {"acc_config": cc._asdict()}
        for m_name, o in out.items():
            writeable_explored_acc_configs[idx][m_name] = o._asdict()
        idx = idx + 1

    with open(out_file, "w") as f:
        json.dump(writeable_explored_acc_configs, f, indent=2)


def run_solver(models_info, micro_batch_size, max_tmp_width, sequence_length, activation_recomputation, hbm_size):

    global explored_acc_configs
    global acc_sorted
    global max_thpt_per_area

    global estimation_time
    global dp_time
    global ilp_time

    # reset global variables for this run
    explored_acc_configs = {}
    acc_sorted = {}
    max_thpt_per_area = {}

    estimation_time = 0
    dp_time = 0
    ilp_time = 0

    sort_acc_configs_by_area()

    converged = False

    cc_iter = iter(acc_sorted[next(iter(acc_sorted))])
    next_cc = next(cc_iter)

    while not converged:
        for model_name, model_info in models_info.items():
            print("running solver for model:", model_name,
                  "with accelerator config:", next_cc)

            tmpc_models, latency_estimates = model_info

            # per model, DAG is available for each TMPC width
            out = solve(tmpc_models, latency_estimates, next_cc,
                        micro_batch_size, max_tmp_width, sequence_length, activation_recomputation, hbm_size)

            if out is None:
                if next_cc in explored_acc_configs.keys():
                    del explored_acc_configs[next_cc]
                break

            if next_cc not in explored_acc_configs.keys():
                explored_acc_configs[next_cc] = {}

            explored_acc_configs[next_cc][model_name] = out

            print("converged for model:", model_name)

        converged, next_cc, cc_iter = get_next_acc_config(next_cc, cc_iter)

    best_cc, exec_strategies = find_config_across_models()
    write_explored_configs_to_file(model_name, micro_batch_size, max_tmp_width,
                                   sequence_length, activation_recomputation, hbm_size/(1024*1024*1024))

    return (best_cc, exec_strategies), estimation_time, ilp_time, dp_time


# executes per model
def solve(models, latency_estimates, cc, micro_batch_size, max_tmp_width, sequence_length, activation_recomputation, hbm_size):
    model_name = models[0].model_name
    layer_state_nodes = {}
    layer_state_edges = {}

    global ilp_time

    def create_layer_state_per_tmpc(m_per_tmpc):

        t = str(m_per_tmpc.tmp_width)

        layer_state_nodes[t] = []
        layer_state_edges[t] = []

        layer_graph = m_per_tmpc.get_layer_graph()
        repeat_layer_ids = m_per_tmpc.get_repeat_layer_ids()

        r_node_attr = None
        r_layer_id = None

        global estimation_time
        utilization = {"TC": {"peak": 0, "avg": 0}, "VC": {"peak": 0, "avg": 0}}
        total_aggr_latency = 0

        for layer_id in layer_graph.nodes():

            if r_layer_id is None or layer_id not in repeat_layer_ids:

                # execute the ILP solver for the optimal fwd and bwd latency
                # as many models have replicated layers, we only execute the ILP once for the replicated layers

                ret_l_attr = {'id': layer_id}
                l_attr = layer_graph.nodes[layer_id]["node"]

                ret_l_attr['activationSize'] = l_attr['activation_size'] * \
                    bytes_per_element
                ret_l_attr["parameterSize"] = l_attr["parameter_size"] * \
                    bytes_per_element
                ret_l_attr['isTensorParallelized'] = l_attr['is_tensor_parallelized']

                per_layer_op_graph = m_per_tmpc.get_op_graph(layer_id)
                fwd_latency, bwd_latency, energy, est_time, utilization_l = latency_per_layer(
                    per_layer_op_graph, cc, latency_estimates[t])

                if (fwd_latency == None or bwd_latency == None):
                    Warning("ILP solver failed for model",
                            model_name, "layer: ", layer_id)
                    del layer_state_nodes[t]
                    del layer_state_edges[t]
                    return

                aggr_val = fwd_latency + bwd_latency if layer_id not in repeat_layer_ids else fwd_latency + \
                    bwd_latency * len(repeat_layer_ids)
                total_aggr_latency += aggr_val

                for core in utilization:
                    utilization[core]['peak'] = max(
                        utilization_l[core]['peak'], utilization[core]['peak'])
                    utilization[core]['avg'] += utilization_l[core]['avg'] * aggr_val

                ret_l_attr["optimalLatencyFw"], ret_l_attr["optimalLatencyBw"] = fwd_latency, bwd_latency
                ret_l_attr["energy"] = energy

                layer_state_nodes[t].append(ret_l_attr)

                r_layer_id = layer_id if layer_id in repeat_layer_ids else r_layer_id
                r_node_attr = ret_l_attr if layer_id in repeat_layer_ids else r_node_attr

                estimation_time = estimation_time + est_time

        for layer_id in repeat_layer_ids:
            if layer_id != r_layer_id:
                new_layer_attr = r_node_attr.copy()
                new_layer_attr['id'] = layer_id
                layer_state_nodes[t].append(new_layer_attr)

        common_e_size = None
        for src, dst, data in layer_graph.edges(data=True):
            layer_state_edges[t].append(
                {"sourceId": src, "destId": dst, "communicationCost": data['tensorsize'] * bytes_per_element})

            if dst in repeat_layer_ids:
                common_e_size = data['tensorsize'] * bytes_per_element

        if m_per_tmpc.is_layer_info_extended():
            for i, layer_id in enumerate(repeat_layer_ids[:-1]):
                if not common_e_size:
                    raise "common edge size for repeat layers not found"

                layer_state_edges[t].append(
                    {"sourceId": layer_id, "destId": repeat_layer_ids[i+1], "communicationCost": common_e_size})

        for core in utilization:
            utilization[core]['avg'] = utilization[core]['avg'] / \
                total_aggr_latency

        return utilization

    start = time.time()
    for m_per_tmpc in models:
        utilization = create_layer_state_per_tmpc(m_per_tmpc)

    layer_state = (layer_state_nodes, layer_state_edges)
    end = time.time()

    ilp_time += (end - start)

    return device_placement(model_name, layer_state, micro_batch_size, max_tmp_width, sequence_length, utilization, activation_recomputation, hbm_size)


def device_placement(model_name, layer_state, micro_batch_size, max_tmp_width, sequence_length, utilization, activation_recomputation, hbm_size):

    num_micro_batches_in_batch = global_batch_size / micro_batch_size

    global dp_time

    dp_input = {"maxMemoryPerDevice": hbm_size,
                "maxDevices": num_accelerators,
                "bandwidth": bandwidth,
                "mbsInBatch": num_micro_batches_in_batch,
                "nodes": layer_state[0],
                "edges": layer_state[1],
                "activationRecomputation": activation_recomputation,
                "optimizerAlgorithm": optimizer_type,
                "fixedPP": baseline_config[model_name]["p"],
                "fixedTMPC": baseline_config[model_name]["t"],
                "fixedDP": baseline_config[model_name]["d"],
                "numTransformerLayers": baseline_config[model_name]["num_transformer_layers"],
                }

    m_dir = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), "device_placement/inputs")
    if not os.path.exists(m_dir):
        os.mkdir(m_dir)

    in_file = generate_out_filename(
        model_name, "json", micro_batch_size, max_tmp_width, sequence_length)

    out_file = generate_out_filename(
        model_name, "json", micro_batch_size, max_tmp_width, sequence_length, "out")

    in_filepath = os.path.join(m_dir, in_file)
    out_filepath = os.path.join(m_dir, out_file)

    with open(in_filepath, "w") as f:
        json.dump(dp_input, f)

    start = time.time()

    status = subprocess.call(
        ["./Solver/device_placement/device_placement", in_filepath, out_filepath], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    end = time.time()

    dp_time += end - start

    if (status != 0):
        return None

    with open(out_filepath, "r") as f:
        out = json.load(f)

    if len(out["stages"]) == 0:
        return None

    thpt = global_batch_size / out["finalTimePerBatch"]
    thptfixed = global_batch_size / out["fixedStrategyTimePerBatch"]

    dist = dist_strategy(
        len(out["stages"]), out["tensorParallelDegree"], out["dataParallelDegree"])

    return out_per_cc_per_model(dist, thpt, thptfixed, utilization)
