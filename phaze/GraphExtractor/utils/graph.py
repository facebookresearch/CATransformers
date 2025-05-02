# internal code
from .helpers import generate_out_filename
from .node import PhazeNode

# graph library
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

# python imports
from collections import namedtuple
from math import prod
import os
import random


memory_footprint_tuple = namedtuple(
    "memory", ["parameter_size", "input_size", "activation_size"])


class PhazeGraph:
    graph = None
    tmp_width = None
    model_name = None

    def __init__(self, tmp_width, model_name) -> None:
        self.tmp_width = tmp_width
        self.model_name = model_name

        self.graph = nx.DiGraph()
        self.layer_graph = nx.DiGraph()

        # layer information
        # remains fixed for a model, regardless of Tensor Model Parallel width
        self.layer_ids = set()
        # TODO: make the repeat layers such that there can be multiple types of repeat layers
        self.repeat_layers = []
        self.repeat_layers_dict = {}
        self.repeat_layers_first = []

        # list of graphs for each layer
        self.op_layer_graphs = []

        # contracted layer graph
        self.layer_graph = nx.DiGraph(directed=True)

        # Every dictionary: Keys are layer_ids, Values is 2D array of nodes in each layer slice
        # if tmp_width == 1, then nodes per layer contains nodes for a layer, else contains nodes in a layer slice
        self.nodes_per_layer = {}
        self.edges_across_layers = []

        # Input tensor dimensions to each layer slice or layer
        # Every dictionary: Keys are layer_ids, Values is 2D array of nodes in each layer slice
        # if tmp_width == 1, then inputs per layer, else contains inputs for a layer slice
        self.layer_inputs = {}

        self.layer_info_extended = False

    def get_graph(self):
        return self.graph

    def get_layer_graph(self):
        return self.layer_graph

    def get_op_layer_graph(self, layer_id):
        return self.op_layer_graphs[layer_id]

    def get_op_layer_graphs(self):
        return self.op_layer_graphs

    def set_layer_ids(self, layer_ids):
        self.layer_ids = layer_ids

    def get_num_nodes(self):
        # Number of nodes with single layer slice
        return len(self.graph.nodes)

    def set_repeat_layer_ids(self, language_models, unique_layers=None):
        if self.layer_ids is False:
            raise ValueError("Layer ids not set")

        # TODO: fix this for resnet as well
        if (self.model_name in ['clip'] and unique_layers !=None):
            self.repeat_layers = [
                i for i in self.layer_ids if (i >= 0 and i not in unique_layers.keys())]
            for i in unique_layers.keys():
                self.repeat_layers_first.append(i)
                self.repeat_layers_dict[i] = unique_layers[i]
        else:
            self.repeat_layers = [i for i in self.layer_ids if i > 0]
            self.repeat_layers_first.append(0)
            self.repeat_layers_dict[0] = [i for i in self.layer_ids if i > 0]
            
        print(self.repeat_layers)
        print(self.repeat_layers_first)
        print(self.repeat_layers_dict)

    def get_repeat_layer_ids(self):
        return self.repeat_layers
    def get_repeat_layer_first(self):
        return self.repeat_layers_first
    def get_repeat_layer_dict(self):
        return self.repeat_layers_dict

    def extend_layer_info_sans_graph(self, new_num_layers, repeat_layer_id):
        self.layer_info_extended = True

        if repeat_layer_id not in self.layer_ids:
            raise ValueError("Repeat layer id not in layer ids")
        if repeat_layer_id not in self.repeat_layers:
            raise ("Repeat layer id not in repeat layers")

        new_layer_ids = set(i for i in range(1, new_num_layers))

        self.layer_ids = self.layer_ids.union(new_layer_ids)
        self.repeat_layers.extend(list(new_layer_ids))

    def get_unique_op_graphs(self):
        ret_graphs = []

        # repeat_layer_added = False

        for layer_id in self.layer_ids:
            if layer_id not in self.repeat_layers:
                ret_graphs.append(self.op_layer_graphs[layer_id])
            # elif repeat_layer_added is False:
            #     ret_graphs.append(self.op_layer_graphs[layer_id])
            #     repeat_layer_added = True

        return ret_graphs

    def get_layer_ids(self):
        if self.layer_ids is False:
            raise ValueError("Layer ids not set")
        return self.layer_ids

    def print_graph(self, out_dir, micro_batch_size=1, sequence_length=1):
        colors = {layer_id: "#" + "".join([random.choice("0123456789ABCDEF")
                                          for j in range(6)]) for layer_id in self.layer_ids}

        for n in self.graph.nodes:
            node = self.graph.nodes[n]["node"]
            self.graph.nodes[node.get_id()]["shape"] = "box"
            self.graph.nodes[node.get_id()]["style"] = "filled"
            self.graph.nodes[node.get_id(
            )]["fillcolor"] = colors[node.get_layer_id()]
            self.graph.nodes[node.get_id()]["label"] = node.get_operator() + "(" +node.get_component() + ")"

        out_file_dot = str(out_dir) + generate_out_filename(self.model_name,
                                                            "dot", micro_batch_size, self.tmp_width, sequence_length,)
        out_file_png = str(out_dir) + generate_out_filename(self.model_name,
                                                            "png", micro_batch_size, self.tmp_width, sequence_length,)

        agraph = to_agraph(self.graph)
        agraph.layout("dot")
        agraph.draw(out_file_dot)
        os.system("dot -Tpng " + out_file_dot + " -o " + out_file_png)

    def print_layer_graph(self, out_dir, micro_batch_size=1, sequence_length=1):
        for n in self.layer_graph.nodes:
            self.layer_graph.nodes[n]["shape"] = "box"
            self.layer_graph.nodes[n]["style"] = "filled"
            self.layer_graph.nodes[n]["label"] = n

        out_file_dot = str(out_dir) + generate_out_filename(self.model_name, "dot",
                                                            micro_batch_size, self.tmp_width, sequence_length, "layer_graph")
        out_file_png = str(out_dir) + generate_out_filename(self.model_name, "png",
                                                            micro_batch_size, self.tmp_width, sequence_length, "layer_graph")

        agraph = to_agraph(self.layer_graph)
        agraph.layout("dot")
        agraph.draw(out_file_dot)
        os.system("dot -Tpng " + out_file_dot + " -o " + out_file_png)

    def setup_node_and_edges(self, graphmodule, node, layer_id):
        idx_node = list(graphmodule.nodes).index(node)

        # create inputs of the node
        inputs = []
        for i in node.all_input_nodes:
            inputs.extend(i.shape)

        # create phaze node in the graph
        node_attributes = PhazeNode(node, idx_node, layer_id, inputs)

        # add nodes to the graph
        node_tuple = (idx_node, {"node": node_attributes})
        self.graph.add_nodes_from([node_tuple])

        # add edges to the graph
        for user in node.users:
            self.graph.add_edge(idx_node, list(
                graphmodule.nodes).index(user), tensorsize=node.shape,)

    # Functions for layer information
    def generate_layer_info(self):
        self.nodes_per_layer = {id: [] for id in self.layer_ids}
        self.layer_inputs = {id: [] for id in self.layer_ids}

        curr_graph = self.get_graph()

        for node_id in curr_graph.nodes:
            node = curr_graph.nodes[node_id]["node"]
            layer_id = node.layer_id

            self.nodes_per_layer[layer_id].append(node)

        for src, dst in curr_graph.edges:
            src_node = curr_graph.nodes[src]["node"]
            dst_node = curr_graph.nodes[dst]["node"]

            if src_node.get_operator() == "input":
                self.layer_inputs[src_node.get_layer_id()].extend(
                    src_node.get_inputs())

            if src_node.get_layer_id() != dst_node.get_layer_id():
                edge_info = (src_node.get_layer_id(),
                             dst_node.get_layer_id(), src_node.get_activation_size())
                self.edges_across_layers.append(edge_info)
                self.layer_inputs[dst_node.get_layer_id()].extend(
                    dst_node.get_inputs())

    def return_layer_info(self):
        return (self.layer_graph, self.nodes_per_layer, self.layer_inputs)

    def set_op_layer_graphs(self):
        curr_graph = self.get_graph()

        for layer_id in self.layer_ids:
            node_ids_in_layer = [node.get_id()
                                 for node in self.nodes_per_layer[layer_id]]

            op_layer_graph = curr_graph.subgraph(node_ids_in_layer).copy()
            self.op_layer_graphs.append(op_layer_graph)

    def contract_layer_graph(self):
        self.layer_graph = nx.DiGraph()

        memory_estimates = self.get_layerwise_memory_footprint()

        # add nodes to the graph
        for layer_id in self.layer_ids:
            # add nodes to the graph
            node_attributes = {}
            node_attributes['id'] = layer_id
            node_attributes['parameter_size'] = memory_estimates[layer_id].parameter_size

            node_attributes['activation_size'] = memory_estimates[layer_id].activation_size
            node_attributes['is_tensor_parallelized'] = self.tmp_width > 1
            '''for node in self.nodes_per_layer[layer_id]:
                if node.isTensorParallelized:
                    if(self.tmp_width == 1):
                        print(node.partition_dim, node.partition_stride)
                    node_attributes['is_tensor_parallelized'] = True
                    break'''

            node_tuple = (layer_id, {"node": node_attributes})
            self.layer_graph.add_nodes_from([node_tuple])

        # add edges to the graph
        for src, dst, size in self.edges_across_layers:
            self.layer_graph.add_edge(src, dst, tensorsize=size,)

    def get_layerwise_memory_footprint(self):
        # It is either a per_layer or per_slice footprint, latter if the tmp_width > 1
        # footprint is a tuple with the three memory estimates, input, parameter (weight + bias), and output
        memory_estimates = {}

        for layer_id, nodes_per_layer in self.nodes_per_layer.items():

            input_s = 0
            for input in self.layer_inputs[layer_id]:
                input_s += prod(input)

            parameter_s = 0
            activation_s = 0
            for node in nodes_per_layer:
                per_node_estimate = node.get_memory_estimate()

                parameter_s += per_node_estimate[0]
                activation_s += per_node_estimate[1]

            m = memory_footprint_tuple(
                parameter_s, input_s, activation_s)
            memory_estimates[layer_id] = m

        return memory_estimates

    def get_memory_footprint(self):

        estimates = self.get_layerwise_memory_footprint()

        parameter_s = 0

        for layer_id, estimate in estimates.items():

            parameter_s += estimate.parameter_size

        # TODO: is the min and max here correct?
        model_memory_estimate = memory_footprint_tuple(
            parameter_s, estimates[min(self.layer_ids)].input_size,
            max(estimates[x].activation_size for x in self.repeat_layers_first)
        )

        return model_memory_estimate
