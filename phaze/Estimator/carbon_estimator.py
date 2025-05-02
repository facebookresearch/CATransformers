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

from .architecture import create_core_config, get_configs_to_explore
from .architecture import tc_configs, vc_configs

from .utils import convert_phaze_to_fused_graph,  get_engine_type
from .utils import embodied_carbon_estimate, operational_carbon_estimate, initialize_carbon_intensity
from .utils import phaze_coretype_mapping

# External imports
from collections import namedtuple, OrderedDict

# external imports
import networkx as nx
from math import inf
import os
import json

def estimate_carbon(models_info, micro_batch_size, max_tmp_width, sequence_length, hbm_size):
    initialize_carbon_intensity()
    print ("-------------------------")
    sort_acc_configs_by_area()
    if len(acc_sorted) <= 0:
        return None
    cc_iter = iter(acc_sorted[next(iter(acc_sorted))])
    next_cc = next(cc_iter)

    estimate_for_one_config = True

    done = False
    result_list = []
    while not done: 
        result = estimate_operational_carbon(models_info, next_cc)
        if result != None: 
            operational_carbon_for_models , sequential_latency, energy, component_carbon_per_model, component_latency_per_model = result
        else: 
            return None
        
        embodied_carbon = estimate_embodied_carbon(next_cc.area, hbm_size)

        for model_name, op_carbon in operational_carbon_for_models.items():
            total_carbon = (op_carbon * 30000000)  + embodied_carbon # scale op_carbon to 3 years continuous inference
            print("Carbon estimates for accelerator config:", next_cc)
            print("operational carbon (one inference): " + str(op_carbon)+ "kg CO2")
            print("embodied carbon: " + str(embodied_carbon) + "kg CO2")
            print("total carbon: " + str(total_carbon) + "kg CO2")
            print("total sequantial latency: " + str(sequential_latency[model_name]) + " s")
            print("total energy (lifetime):" + str(energy[model_name]* 30000000) + " kwh")
            print("Per component operational arbon and latency breakdown:")
            for name in component_carbon_per_model[model_name].keys():
                print(name + ": carbon: " + str(component_carbon_per_model[model_name][name]) + ", latency: " + str(component_latency_per_model[model_name][name]))

            print ("-------------------------")

            result = {"config": next_cc, "area": next_cc.area, "embodied":embodied_carbon, "operational":op_carbon, "total_carbon":total_carbon, "latency":sequential_latency[model_name], "breakdown":component_carbon_per_model[model_name], "energy":energy[model_name]* 30000000 }
            result_list.append(result)
            # TODO: save data into a json file
        done, next_cc, cc_iter = get_next_acc_config(next_cc, cc_iter)
        if estimate_for_one_config:
            done = True
    
    return result_list

def estimate_operational_carbon(models_info, cc):
    total_operational_carbon_per_model = {}
    total_sequential_latency_per_model = {}
    total_energy_per_model = {}
    total_operational_carbon = 0
    total_seq_lat = 0
    total_energy = 0

    total_component_carbon = {}
    total_component_latency = {}
    total_component_carbon_per_model = {}
    total_component_latency_per_model = {}
    for model_name, model_info in models_info.items():

        print("calculating carbon for model:", model_name)

        tmpc_models, latency_estimates = model_info
        total_operational_carbon = 0
        total_seq_lat = 0

        # per model, DAG is available for each TMPC width
        for m_per_tmpc in tmpc_models:

            t = str(m_per_tmpc.tmp_width)

            layer_graph = m_per_tmpc.get_layer_graph()
            repeat_layer_ids = m_per_tmpc.get_repeat_layer_ids()
            repeat_layer_first = m_per_tmpc.get_repeat_layer_first()
            repeat_layer_dict = m_per_tmpc.get_repeat_layer_dict()

            for layer_id in layer_graph.nodes():

                if layer_id not in repeat_layer_ids:

                    # calculate carbon once for replicated layers and multiply it at the end to all layers

                    per_layer_op_graph = m_per_tmpc.get_op_graph(layer_id)
                    result = op_carbon_per_layer(per_layer_op_graph, cc, latency_estimates[t])
                    if (result != None):
                        op_carbon, latency, comp_carbon, comp_lat, energy = result
                    else:
                        print("cannot find proper mapping for this configuration")
                        return None
                    t_op_carbon_component = {}
                    t_op_latency_component = {}

                    if layer_id in repeat_layer_first:
                        t_op_carbon = op_carbon * (len(repeat_layer_dict[layer_id]) + 1) # add repeated later + current layer
                        t_seq_lat = latency * (len(repeat_layer_dict[layer_id])  +1 )
                        t_energy = energy * (len(repeat_layer_dict[layer_id])+ 1)
                        print(layer_id)
                        print(latency)
                        print(t_seq_lat)
                        for name , carbon in comp_carbon.items():
                            t_op_carbon_component[name] = carbon * (len(repeat_layer_dict[layer_id]) +1)
                        for name, latency in comp_lat.items():
                            t_op_latency_component[name] = latency * (len(repeat_layer_dict[layer_id])+1)
                    else:
                        t_op_carbon = op_carbon
                        t_seq_lat = latency 
                        t_energy = energy 
                        for name , carbon in comp_carbon.items():
                            t_op_carbon_component[name] = carbon 
                        for name, latency in comp_lat.items():
                            t_op_latency_component[name] = latency 

                    total_operational_carbon += t_op_carbon
                    total_seq_lat += t_seq_lat
                    total_energy += t_energy

                    for name, carbon in t_op_carbon_component.items():
                        total_component_carbon[name] = total_component_carbon[name] + carbon if name in total_component_carbon else carbon
                    for name, latency in t_op_latency_component.items():
                        total_component_latency[name] = total_component_latency[name] + latency if name in total_component_latency else latency

        total_operational_carbon_per_model[model_name] = total_operational_carbon
        total_sequential_latency_per_model[model_name] = total_seq_lat
        total_energy_per_model[model_name] = total_energy

        total_component_carbon_per_model[model_name] = total_component_carbon
        total_component_latency_per_model[model_name] = total_component_latency

    return total_operational_carbon_per_model, total_sequential_latency_per_model, total_energy_per_model, total_component_carbon_per_model, total_component_latency_per_model


def estimate_embodied_carbon(area, hbm):

    chip, hbm = embodied_carbon_estimate(area, hbm)
    return chip + hbm 

# dictionary of sorted accelerators by area
acc_sorted = {}
acc_group_range = 1 # incase some architectures have the same area
# dictionary of explored accelerator configs
explored_acc_configs = {}

def sort_acc_configs_by_area():
    global acc_sorted
    global acc_group_range

    # Reset everytime the carbon_estimate is called since the architecture list might change 
    acc_sorted = {}

    acc_configs = get_configs_to_explore()
    for cc in acc_configs:
        area_range = int(cc.area / acc_group_range)
        if area_range not in acc_sorted.keys():
            acc_sorted[area_range] = []
        acc_sorted[area_range].append(cc)
    acc_sorted = OrderedDict(sorted(acc_sorted.items(), reverse=True))
    print(acc_sorted)


def get_next_acc_config(prev_cc, cc_iter):
    global acc_group_range

    if_all_archs_to_explore = True

    curr_area = prev_cc.area
    area_range = int(curr_area / acc_group_range)
    curr_area_idx = list(acc_sorted.keys()).index(area_range)

    try:
        return False, next(cc_iter), cc_iter
    except:
        try:
            next_area = list(acc_sorted.keys())[curr_area_idx + 1]
            cc_iter = iter(acc_sorted[next_area])

            # if if_all_archs_to_explore:
            return False, next(cc_iter), cc_iter

        except:
            print("All architectures explored")
            return True, None, None


def extract_op_graph(per_l_op_graph, cc, latency_estimates):

    def generate_digraph_from_fused_graph(wgraph, mode="fwd"):
        out_graph = nx.DiGraph()
        estimation_time = 0

        for node in wgraph.nodes.values():
            node_attr = {}

            node_attr["core_type"] = phaze_coretype_mapping[get_engine_type(
                node.node_desc)]

            if (node_attr["core_type"] == "Nop"):
                e = {"fwd": {"latency": 0, "estimation_time": 0, "energy": 0}, "bwd": {
                    "latency": 0, "estimation_time": 0, "energy": 0}}

            elif (node_attr["core_type"] == "TC" or node_attr["core_type"] == "TCandVC"):
                idx = tc_configs.index(create_core_config(cc, "TC"))
                e = latency_estimates["TC"][str(node.node_id)][idx]

            elif node_attr["core_type"] == "VC":
                # collective operator ops
                if "AllReduce" in node.node_desc:
                    e = latency_estimates["AR"][str(node.node_id)]
                else:
                    idx = vc_configs.index(create_core_config(cc, "VC"))
                    e = latency_estimates["VC"][str(node.node_id)][idx]

            estimation_time = estimation_time + e[mode]["estimation_time"]

            node_attr["intra_op_latency"] = e[mode]['latency'] # in seconds
            node_attr["intra_op_energy"] = e[mode]['energy'] # in J 
            node_attr["component"] = node.component

            if (node_attr["intra_op_latency"] == inf):
                estimation_time = 0
                return None, 0

            node_tuple = (node.node_id, {'node': node_attr})
            out_graph.add_nodes_from([node_tuple])

        for src, dstnodes in wgraph.edges.items():
            for dst in dstnodes:
                out_graph.add_edge(src, dst.node_id)

        return out_graph, estimation_time

    w_fwd_graph = convert_phaze_to_fused_graph(per_l_op_graph, "fwd")

    fwd_graph, e_time_fwd = generate_digraph_from_fused_graph(
        w_fwd_graph, "fwd")

    return fwd_graph, e_time_fwd



def op_carbon_per_layer(l_op_graph, acc_config, latency_estimates):

    fwd_graph, estimation_time = extract_op_graph(
        l_op_graph, acc_config, latency_estimates)

    if fwd_graph == None:
        return None

    carbon, latency, componentwise_carbon, componentwise_latency, energy_kwh= calculate_carbon_from_energy(
        fwd_graph)

    return carbon, latency, componentwise_carbon, componentwise_latency, energy_kwh


def calculate_carbon_from_energy(graph):
    # use the ILP x_ij values to schedule the operators
    # and then check at most how many cores are active at any given time
    non_nop_nodes = [i for i in graph.nodes() if not graph.nodes[i]
                     ['node']['core_type'] == 'Nop']

    # compute schedule_energy
    schedule_latency = 0 # in seconds
    schedule_energy = 0 # in Joules
    componentwise_latency = {}
    componentwise_energy = {} 
    componentwise_carbon = {}
    for i in non_nop_nodes:
        schedule_energy += graph.nodes[i]['node']['intra_op_energy']
        schedule_latency += graph.nodes[i]['node']['intra_op_latency']
        if graph.nodes[i]['node']['component'] in componentwise_latency:
            componentwise_latency[graph.nodes[i]['node']['component']] += graph.nodes[i]['node']['intra_op_latency']
            componentwise_energy[graph.nodes[i]['node']['component']] += graph.nodes[i]['node']['intra_op_energy']
        else:
            componentwise_latency[graph.nodes[i]['node']['component']] = graph.nodes[i]['node']['intra_op_latency']
            componentwise_energy[graph.nodes[i]['node']['component']] = graph.nodes[i]['node']['intra_op_energy']

    print(schedule_latency)
    operational_carbon = operational_carbon_estimate(schedule_energy)
    for component_name, component_energy in componentwise_energy.items():
        component_carbon = operational_carbon_estimate(component_energy)
        componentwise_carbon[component_name] = component_carbon
    
    # convert energy from Joules to kwh
    schedule_energy_kwh = schedule_energy / 3600000

    return operational_carbon, schedule_latency , componentwise_carbon, componentwise_latency, schedule_energy_kwh
