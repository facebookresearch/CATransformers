"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import os

from .graph import Graph, Node
import op_to_compute


class fusion(object):
    def __init__(self, profile, output_directory, acc_id):
        if isinstance(profile, Graph):
            self.graph = profile
        elif os.path.isfile(profile):
            self.graph = Graph.from_str(open(profile, "r").read())
        else:
            raise TypeError(
                "Input not of the type filename or Wham Graph class; in Op fusion")

        self.add_graph_source_sink()
        self.topological_nodes = []
        self.topological_sort()
        self.output_directory = output_directory
        self.acc_id = acc_id

    def fuse(self):
        fusion_done = False
        while not fusion_done:
            self.topological_sort()
            nodes = self.get_topological_nodes()
            all_done = True

            for node in nodes:

                if (
                    node.node_id == "SRC"
                    or node.node_id == "SNK"
                    or node.node_desc == "AccumulateGrad"
                    or node.node_desc == "T"
                    or node.node_desc == "dummy"
                ):
                    continue

                op_type = op_to_compute.get_opr_type(node.node_desc)
                succ_nodes = self.get_successor_nodes(node)

                if len(succ_nodes) > 0:
                    succ_node_0 = succ_nodes[0]
                    succ_op_type = op_to_compute.get_opr_type(
                        succ_node_0.node_desc)

                    valid_succ_op_type_1 = (
                        True
                        if succ_op_type == "activation_opr"
                        or succ_op_type == "injective_opr"
                        or succ_op_type == "norm_opr"
                        or succ_op_type == "dropout_opr"
                        or succ_op_type == "transformation_opr"
                        else False
                    )
                    valid_succ_op_type_2 = (
                        True
                        if succ_op_type == "injective_opr"
                        or succ_op_type == "reduction_opr"
                        or succ_op_type == "dropout_opr"
                        else False
                    )

                    valid_pred_succ_node_0 = True
                    succ_pre_dec_nodes = self.get_predecessor_nodes(succ_node_0)
                    # Successor pre-decessor might have more than 1 nodes with same description
                    # Using counter to identify that and making it False if counter > 1
                    counter = 0
                    for succ_pre_dec_node in succ_pre_dec_nodes:
                        if (
                            succ_pre_dec_node.node_desc != "AccumulateGrad"
                            and succ_pre_dec_node.node_desc != node.node_desc
                        ):
                            valid_pred_succ_node_0 = False
                        if succ_pre_dec_node.node_desc == node.node_desc:
                            counter += 1
                    if counter > 1:
                        valid_pred_succ_node_0 = False

                    """ ================================================================================================================== """
                    """ ===== fuse complex-out fusable ops (Conv2d) with any element-wise operator to its ouput (Reference TVM paper) ==== """
                    if (
                        op_type == "complex_out_fusable_opr"
                        and len(succ_nodes) == 1
                        and valid_succ_op_type_1
                        and valid_pred_succ_node_0
                    ):
                        # Updating the Node desc to fused node desc
                        if node.node_desc.split(" ~~ ")[0] != "fused":
                            node.node_desc = "fused ~~ " + node.node_desc

                        # Getting weights info of fused operator and output activation
                        weights_fused_opr = []
                        output_act_fused_opr = []

                        for succ_pre_dec_node in succ_pre_dec_nodes:
                            if succ_pre_dec_node.node_desc == "AccumulateGrad":
                                weights_fused_opr.append(
                                    succ_pre_dec_node.parameter)

                        for i in range(len(succ_node_0.output_act)):
                            output_act_fused_opr.append(
                                succ_node_0.output_act[i])

                        # Adding fused operator info to fused node
                        fuse_opr_info = (
                            succ_node_0.node_desc
                            + " ~~ "
                            + str(weights_fused_opr)
                            + " ~~ "
                            + str(output_act_fused_opr)
                        )
                        if node.fused_operators == "":
                            node.fused_operators = fuse_opr_info
                        else:
                            node.fused_operators = (
                                node.fused_operators + " ~~ " + fuse_opr_info
                            )

                        # Removing weight nodes of fused operator
                        len_succ_pre_dec_nodes = len(succ_pre_dec_nodes)
                        for i in range(len_succ_pre_dec_nodes):
                            if (
                                succ_pre_dec_nodes[
                                    len_succ_pre_dec_nodes - i - 1
                                ].node_desc
                                == "AccumulateGrad"
                            ):
                                self.remove_node(
                                    succ_pre_dec_nodes[len_succ_pre_dec_nodes - i - 1]
                                )

                        # Removing fused node and connecting fused operator with fused operator successors
                        succ_succ_nodes = self.get_successor_nodes(succ_node_0)
                        self.remove_node(succ_node_0)
                        for succ_succ_node in succ_succ_nodes:
                            self.add_edge(node, succ_succ_node)

                        all_done = False
                        break
                    """ ================================================================================================================== """

                    """ ================================================================================================================== """
                    """ ==== fused injective (one-to-one map e.g. add) ops (Reference TVM paper) ==== """
                    if (
                        op_type == "injective_opr"
                        and len(succ_nodes) == 1
                        and valid_succ_op_type_2
                        and valid_pred_succ_node_0
                    ):
                        # Updating the Node desc to fused node desc
                        if node.node_desc.split(" ~~ ")[0] != "fused":
                            node.node_desc = "fused ~~ " + node.node_desc

                        # Getting weights info of fused operator and output activation
                        weights_fused_opr = []
                        output_act_fused_opr = []

                        for succ_pre_dec_node in succ_pre_dec_nodes:
                            if succ_pre_dec_node.node_desc == "AccumulateGrad":
                                weights_fused_opr.append(
                                    succ_pre_dec_node.parameter)

                        for i in range(len(succ_node_0.output_act)):
                            output_act_fused_opr.append(
                                succ_node_0.output_act[i])

                        # Adding fused operator info to fused node
                        fuse_opr_info = (
                            succ_node_0.node_desc
                            + " ~~ "
                            + str(weights_fused_opr)
                            + " ~~ "
                            + str(output_act_fused_opr)
                        )
                        if node.fused_operators == "":
                            node.fused_operators = fuse_opr_info
                        else:
                            node.fused_operators = (
                                node.fused_operators + " ~~ " + fuse_opr_info
                            )

                        # Removing weight nodes of fused operator
                        len_succ_pre_dec_nodes = len(succ_pre_dec_nodes)
                        for i in range(len_succ_pre_dec_nodes):
                            if (
                                succ_pre_dec_nodes[
                                    len_succ_pre_dec_nodes - i - 1
                                ].node_desc
                                == "AccumulateGrad"
                            ):
                                self.remove_node(
                                    succ_pre_dec_nodes[len_succ_pre_dec_nodes - i - 1]
                                )

                        # Removing fused node and connecting fused operator with fused operator successors
                        succ_succ_nodes = self.get_successor_nodes(succ_node_0)
                        self.remove_node(succ_node_0)
                        for succ_succ_node in succ_succ_nodes:
                            self.add_edge(node, succ_succ_node)

                        all_done = False
                        break
                    """ ================================================================================================================== """

            if all_done == True:
                fusion_done = True

        self.remove_node(self.graph.sources()[0])
        self.remove_node(self.graph.sinks()[0])

        return self.graph
        # with open(self.output_directory + "/" + self.acc_id, "w") as f:
        #    f.write(str(self.graph))

    def topological_sort(self):
        self.topological_nodes = self.graph.topological_sort()

    def get_topological_nodes(self):
        return self.topological_nodes

    def get_predecessor_nodes(self, node):
        return self.graph.get_predecessors(node)

    def get_successor_nodes(self, node):
        return self.graph.get_successors(node)

    def add_node(self, node):
        self.graph.add_node(node)

    def remove_node(self, node):
        self.graph.remove_node(node)

    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

    def add_graph_source_sink(self):
        source_node = Node("SRC", node_desc="SRC")
        sink_node = Node("SNK", node_desc="SNK")

        sources = self.graph.sources()
        sinks = self.graph.sinks()

        self.add_node(source_node)
        for source in sources:
            self.add_edge(source_node, source)

        self.add_node(sink_node)
        for sink in sinks:
            self.add_edge(sink, sink_node)

    def print(self):
        nodes = self.get_topological_nodes()
        print(
            "----------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print("{:20s} {:20s}".format("NODE DESCRIPTION", "NODE FUSED OPERATORS"))
        print(
            "----------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        for node in nodes:
            print(
                "{:20s} {:20s}".format(
                    str(node.node_desc), str(node.fused_operators))
            )

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
