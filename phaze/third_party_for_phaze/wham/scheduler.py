"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from perf_wrappers import tensor_core_wrapper
from perf_wrappers import vector_core_estimator
from op_fusion import graph
from op_fusion import graph_bwd

import op_to_compute
import definitions

from math import prod

import os
import json
import csv

import sys


class Scheduler(object):
    def __init__(
        self, profile, config, estimates_filename="", recompute=True, optimizer="SGD", dataflow="ws"
    ):
        # Forward Graph only with the input
        if isinstance(profile, graph.Graph):
            self.graph = profile
            self.graph_bwd = graph_bwd.Graph.from_str_bwd(str(self.graph))
        elif os.path.isfile(profile):
            self.graph = graph.Graph.from_str(open(profile, "r").read())
            self.graph_bwd = graph_bwd.Graph.from_str_bwd(
                open(profile, "r").read())
        else:
            raise TypeError(
                "Input not of the type filename or Wham Graph class; in Wham scheduler")

        self.topological_nodes = []
        self.executed_nodes = []
        self.critical_nodes = []

        self.topological_nodes_bwd = []
        self.executed_nodes_bwd = []
        self.critical_nodes_bwd = []

        # Architecure Config
        self.config = config

        # Dataflow
        self.dataflow = dataflow

        # Estimation
        self.estimates_filename = estimates_filename
        self.tc_energy = {}
        if os.path.isfile(estimates_filename):
            try:
                with open(estimates_filename) as infile:
                    est_data = json.load(infile)
            except:
                est_data = {}
        else:
            est_data = {}
        self.est_dict = est_data

        # Distributed Training Details
        self.recompute = recompute
        self.optimizer = optimizer

        # Initialize graphs
        self.add_graph_source_sink()

    def print(self, node_type):
        if node_type == "all_nodes":
            nodes = self.get_topological_nodes()
        elif node_type == "critical_nodes":
            nodes = self.get_critical_nodes()
        else:
            nodes = self.get_topological_nodes()
        print("\n====== ", node_type, "======")
        print(
            "----------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "{:30s} {:20s} {:20s} {:20s}".format(
                "DESCRIPTION",
                "FWD LATENCY(Cycles)",
                "BWD LATENCY(Cycles)",
                "WU LATENCY(Cycles)",
            )
        )
        print(
            "----------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        for node in nodes:
            print(
                "{:30s} {:20s} {:20s} {:20s}".format(
                    str(node.node_desc),
                    str(node.fwd_latency),
                    str(node.bwd_latency),
                    str(node.weight_update_latency),
                )
            )
            # if node_type == 'critical_nodes':
            #    self.print_predecessors(node)

    def print_predecessors(self, node):
        pre_nodes = self.get_predecessor_nodes(node)
        if len(pre_nodes) > 0:
            print(
                "__________________________________________________________________________"
            )
            for node in pre_nodes:
                print(
                    "{:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s}".format(
                        str(node.node_desc),
                        str(node.fwd_latency),
                        str(node.asap_schd),
                        str(node.alap_schd),
                        str(node.mobility),
                        str(node.depth),
                        str(node.schd_time),
                        str(node.res_conflict),
                    )
                )
            print(
                "__________________________________________________________________________\n"
            )

    def print_node(self, node):
        print(
            "__________________________________________________________________________"
        )
        print(
            "{:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s}".format(
                str(node.node_desc),
                str(node.fwd_latency),
                str(node.asap_schd),
                str(node.alap_schd),
                str(node.mobility),
                str(node.depth),
                str(node.schd_time),
                str(node.res_conflict),
            )
        )
        print(
            "__________________________________________________________________________"
        )
        print(node.output_act, node.saved_tensors)

    def fwd_order_ops(self, order_ops_file):
        fwd_order_ops_file = order_ops_file.rstrip(".csv")
        fwd_order_ops_file = fwd_order_ops_file + "_fwd.csv"

        nodes = self.get_topological_nodes()
        ops_header = [
            "OP_ID",
            "OP",
            "SCHD_TIME (Cycles)",
            "ASAP",
            "ALAP",
            "MOBILITY",
            "FWD_TIME (Cycles)",
            "COMP_CORE",
            "CORE_ID",
            "RES_CONFLICT",
            "CONFLICT_DELAY",
        ]
        with open(fwd_order_ops_file, "w") as fwd_ops_file:
            fwd_ops_writer = csv.writer(fwd_ops_file)
            fwd_ops_writer.writerow(ops_header)
            for node in nodes:
                data = [
                    node.node_id,
                    node.node_desc,
                    node.schd_time,
                    node.asap_schd,
                    node.alap_schd,
                    node.mobility,
                    node.fwd_latency,
                    op_to_compute.get_compute_unit(node.node_desc),
                    node.core_id,
                    node.res_conflict,
                    node.res_conflict_delay,
                ]
                fwd_ops_writer.writerow(data)

    def bwd_order_ops(self, order_ops_file):
        bwd_order_ops_file = order_ops_file.rstrip(".csv")
        bwd_order_ops_file = bwd_order_ops_file + "_bwd.csv"

        nodes = self.get_topological_nodes_bwd()
        ops_header = [
            "OP_ID",
            "OP",
            "SCHD_TIME (Cycles)",
            "ASAP",
            "ALAP",
            "MOBILITY",
            "BWD_TIME (Cycles)",
            "COMP_CORE",
            "CORE_ID",
            "RES_CONFLICT",
            "CONFLICT_DELAY",
        ]
        with open(bwd_order_ops_file, "w") as bwd_ops_file:
            bwd_ops_writer = csv.writer(bwd_ops_file)
            bwd_ops_writer.writerow(ops_header)
            for node in nodes:
                data = [
                    node.node_id,
                    node.node_desc,
                    node.schd_time,
                    node.asap_schd,
                    node.alap_schd,
                    node.mobility,
                    node.bwd_latency + node.weight_update_latency,
                    op_to_compute.get_compute_unit(node.node_desc),
                    node.core_id,
                    node.res_conflict,
                    node.res_conflict_delay,
                ]
                bwd_ops_writer.writerow(data)

    def debug(self):
        nodes = self.get_topological_nodes()
        for node in nodes:
            if node.node_desc == "source":
                pass
            else:
                print("\n\n++++++++NODE+++++++")
                self.print_node(node)
                print("+++++++PREDECESSORS+++++++")
                self.print_predecessors(node)

    def add_graph_source_sink(self):

        # Adding source and sink node for fwd_graph
        source_node = graph.Node("source", node_desc="source")
        sink_node = graph.Node("sink", node_desc="sink")

        sources = self.graph.sources()
        sinks = self.graph.sinks()

        self.graph.add_node(source_node)
        for source in sources:
            self.graph.add_edge(source_node, source)

        self.graph.add_node(sink_node)
        for sink in sinks:
            self.graph.add_edge(sink, sink_node)

        node_list = [source_node, sink_node]
        self.add_executed_nodes(node_list)

        # Adding source and sink node for bwd_graph
        source_node_bwd = graph_bwd.Node("source", node_desc="source")
        sink_node_bwd = graph_bwd.Node("sink", node_desc="sink")

        sources_bwd = self.graph_bwd.sources()
        sinks_bwd = self.graph_bwd.sinks()

        self.graph_bwd.add_node(source_node_bwd)
        for source in sources_bwd:
            self.graph_bwd.add_edge(source_node_bwd, source)

        self.graph_bwd.add_node(sink_node_bwd)
        for sink in sinks_bwd:
            self.graph_bwd.add_edge(sink, sink_node_bwd)

        node_list_bwd = [source_node_bwd, sink_node_bwd]
        self.add_executed_nodes_bwd(node_list_bwd)

    def populate_latency(self):
        for node_id in self.graph.nodes:
            if node_id == "source" or node_id == "sink" or node_id == "dummy":
                continue

            node = self.graph.nodes[node_id]
            op = op_to_compute.get_compute_unit(node.node_desc)
            if op == "Tensor Core":
                tensor_core_wrapper.get_performance_est(
                    self, node, self.est_dict, self.config
                )

            elif op == "Tensor Core + Vector Core":
                tensor_core_wrapper.get_performance_est(
                    self, node, self.est_dict, self.config
                )

            elif op == "Vector Core":
                vector_core_estimator.get_performance_est(
                    self, node, self.config)

            else:
                node.fwd_latency = 0
                node.fwd_energy = 0
                node.bwd_latency = 0
                node.bwd_energy = 0
                node.weight_update_latency = 0
                node.weight_update_energy = 0

        # optimizer estimation
        for node_id in self.graph.nodes:

            node = self.graph.nodes[node_id]
            if node.node_desc == "AccumulateGrad":
                vector_core_estimator.get_optimizer_est(
                    self, node, self.optimizer, self.config
                )
            else:
                node.optimizer_latency = 0
                node.optimizer_energy = 0

    def remove_bwd_latency_first_node(self):
        nodes = self.get_topological_nodes()
        for node in nodes:
            op = op_to_compute.get_compute_unit(node.node_desc)
            if op != "Nop":
                # print(node.node_desc)
                # Get nodes with same alap schedule
                same_level_nodes = [
                    i for i in nodes if i.alap_schd == node.alap_schd]

                for same_level_node in same_level_nodes:
                    same_level_node.bwd_latency = 0
                    same_level_node.bwd_energy = 0

                break

    def save_estimates(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.est_dict, outfile)

    def copy_latency_bwd(self):
        for node_id in self.graph.nodes:
            # if node_id == 'source' or node_id == 'sink' or node_id == 'dummy':
            #    continue

            node = self.graph.nodes[node_id]
            node_bwd = self.graph_bwd.nodes[node_id]
            # node_wu = self.graph_wu.nodes[node_id]
            node_bwd.fwd_utilization = node.fwd_utilization
            node_bwd.fwd_latency = node.fwd_latency
            node_bwd.fwd_comm_latency = node.fwd_comm_latency
            node_bwd.fwd_energy = node.fwd_energy
            node_bwd.fwd_comm_energy = node.fwd_comm_energy
            node_bwd.fwd_power = node.fwd_power
            node_bwd.fwd_noc_bw = node.fwd_noc_bw
            node_bwd.fwd_dram_bw = node.fwd_dram_bw
            node_bwd.fwd_l1_size = node.fwd_l1_size
            node_bwd.fwd_l2_size = node.fwd_l2_size
            node_bwd.fwd_l2_ip_tile = node.fwd_l2_ip_tile
            node_bwd.bwd_utilization = node.bwd_utilization
            node_bwd.bwd_latency = node.bwd_latency
            node_bwd.bwd_comm_latency = node.bwd_comm_latency
            node_bwd.bwd_energy = node.bwd_energy
            node_bwd.bwd_comm_energy = node.bwd_comm_energy
            node_bwd.bwd_power = node.bwd_power
            node_bwd.bwd_noc_bw = node.bwd_noc_bw
            node_bwd.bwd_dram_bw = node.bwd_dram_bw
            node_bwd.bwd_l1_size = node.bwd_l1_size
            node_bwd.bwd_l2_size = node.bwd_l2_size
            node_bwd.bwd_l2_ip_tile = node.bwd_l2_ip_tile
            node_bwd.weight_update_utilization = node.weight_update_utilization
            node_bwd.weight_update_latency = node.weight_update_latency
            node_bwd.weight_update_energy = node.weight_update_energy
            node_bwd.weight_update_power = node.weight_update_power
            node_bwd.weight_update_noc_bw = node.weight_update_noc_bw
            node_bwd.weight_update_dram_bw = node.weight_update_dram_bw
            node_bwd.weight_update_l1_size = node.weight_update_l1_size
            node_bwd.weight_update_l2_size = node.weight_update_l2_size
            node_bwd.weight_update_l2_ip_tile = node.weight_update_l2_ip_tile
            node_bwd.optimizer_latency = node.optimizer_latency
            node_bwd.optimizer_energy = node.optimizer_energy

    def populate_sink_latency(self):
        self.graph.populate_sink_latency()
        self.graph_bwd.populate_sink_latency()
        # self.graph_wu.populate_sink_latency()

    def populate_depths(self):
        self.graph.populate_depths()
        self.graph_bwd.populate_depths()
        # self.graph_wu.populate_depths()

    def topological_sort(self):
        self.topological_nodes = self.graph.topological_sort()
        self.topological_nodes_bwd = self.graph_bwd.topological_sort()
        # self.topological_nodes_wu = self.graph_wu.topological_sort()

    def get_topological_nodes(self):
        return self.topological_nodes

    def get_topological_nodes_bwd(self):
        return self.topological_nodes_bwd

    def len_topological_nodes(self):
        return len(self.topological_nodes)

    def len_topological_nodes_bwd(self):
        return len(self.topological_nodes_bwd)

    def add_executed_nodes(self, executed_nodes_list):
        for i in range(len(executed_nodes_list)):
            self.executed_nodes.append(executed_nodes_list[i])

    def add_executed_nodes_bwd(self, executed_nodes_list):
        for i in range(len(executed_nodes_list)):
            self.executed_nodes_bwd.append(executed_nodes_list[i])

    def get_predecessor_nodes(self, node):
        return self.graph.get_predecessors(node)

    def get_predecessor_nodes_bwd(self, node):
        return self.graph_bwd.get_predecessors(node)

    def get_energy(self):
        total_energy = 0
        nodes = self.get_topological_nodes()

        for node in nodes:
            total_energy += node.fwd_energy + node.fwd_comm_energy

        return total_energy

    def get_energy_bwd(self):
        total_energy = 0
        nodes_bwd = self.get_topological_nodes_bwd()

        for node in nodes_bwd:
            total_energy += (
                node.bwd_energy + node.bwd_comm_energy + node.weight_update_energy
            )

        return total_energy

    def get_energy_opt(self):
        total_energy = 0
        nodes_opt = self.get_topological_nodes()

        for node in nodes_opt:
            total_energy += node.optimizer_energy

        return total_energy

    def set_tensor_core_energy(self, energy):
        self.tc_energy = energy

    def get_tensor_core_energy(self):
        return self.tc_energy

    def get_model_parameters_size(self):
        model_parameters = 0

        nodes = self.get_topological_nodes()
        for node in nodes:
            if node.node_desc == "AccumulateGrad":
                model_parameters += prod(node.parameter)

        # model_parameters = model_parameters * (definitions.PRECISION/8)
        # model_parameters = model_parameters / (1024 * 1024)
        print("\n")
        print("======== NUMBER OF MODEL PARAMETERS ========")
        print(model_parameters / (1024 * 1024), " Millions")
        print("=======================================")

    def get_model_size(self):
        model_size = 0
        model_parameters = 0
        model_int_act_size = 0
        model_ip_act_size = 0

        nodes = self.get_topological_nodes()
        for node in nodes:
            if node.node_desc == "AccumulateGrad":
                if self.optimizer == "SGD":
                    model_parameters += 3 * prod(node.parameter)
                elif self.optimizer == "Adam":
                    model_parameters += 4 * prod(node.parameter)
                else:
                    print("Optimizer not supported!!")
                    sys.exit()

            elif node.node_desc == "dummy":
                output_act = node.output_act
                num_output_act = len(output_act)
                if num_output_act == 0:
                    continue
                else:
                    for j in range(num_output_act):
                        model_ip_act_size += 2 * prod(output_act[j])

            else:
                op = op_to_compute.get_compute_unit(node.node_desc)
                if op == "Tensor Core" or op == "Vector Core":
                    output_act = node.output_act
                    num_output_act = len(output_act)
                    if num_output_act == 0:
                        continue
                    else:
                        for k in range(num_output_act):
                            model_int_act_size += 2 * prod(output_act[k])

        model_parameters = model_parameters * (definitions.PRECISION / 8)
        model_ip_act_size = model_ip_act_size * (definitions.PRECISION / 8)
        model_int_act_size = model_int_act_size * (definitions.PRECISION / 8)

        if self.recompute:
            model_act_size = model_ip_act_size
        else:
            model_act_size = model_ip_act_size + model_int_act_size

        model_size = model_parameters + model_act_size

        print("======== MODEL PARAMETERS SIZE ========")
        print(model_parameters / (1024**3), " GB")
        print("=======================================")

        print("======== MODEL ACTIVATIONS SIZE ========")
        print(model_act_size / (1024**3), " GB")
        print("=======================================")

        print("\n")
        print("============= MODEL SIZE ==============")
        print(model_size / (1024 * 1024 * 1024), " GB")
        print("=======================================")

        return model_parameters, model_act_size
