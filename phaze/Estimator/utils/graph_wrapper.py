import sys

# Third party Imports

from scheduler import Scheduler as ExternalScheduler  # noqa
from op_to_compute import get_compute_unit  # noqa
from op_fusion import node as FusedNode  # noqa
from op_fusion import graph as FusedGraph  # noqa
from op_fusion import graph_bwd as FusedBwdGraph  # noqa
from op_fusion import fusion  # noqa


phaze_op_mapping = {
    # Tensor core
    "matmul": "Mm",
    "conv2d": "MkldnnConvolution",
    "conv1d": "MkldnnConvolution",
    "linear": "Linear",
    "baddbmm": "Bmm",
    "addmm": "Addmm",
    "bmm": "Bmm",
    # Vector core
    "dropout": "NativeDropout",
    "layernorm": "NativeLayerNorm",
    "mixedfusedlayernorm": "NativeLayerNorm",
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "permute": "Permute",
    "div_": "Div",
    "truediv": "Div",
    "floordiv": "Div",
    "gelu": "Relu",
    "tanh": "Tanh",
    "cos": "Cos",
    "sin": "Sin",
    "softmax": "Softmax",
    "relu": "Relu",
    "batchnorm2d": "NativeBatchNorm",
    "maxpool2d": "MaxPool2DWithIndices",
    "adaptiveavgpool2d": "AdaptiveAvgPool2D",
    "fusedscalemasksoftmax": "Softmax",
    "bias_gelu": "Relu",
    "mean": "Mean",
    "rsqrt": "Sqrt",
    "where": "Add",
    "neg": "Neg",
    "pow": "Pow",
    "exp": "Pow",
    "silu": "Silu",
    "sigmoid": "Sigmoid",
    # Transformation
    "transpose": "Transpose",
    "t": "T",
    "size": "Index",
    "float": "Index",
    "type": "Index",
    "max": "Index",
    "argmax": "Index",
    "min": "Index",
    "dim": "Index",
    "abs": "Index",
    "getitem": "Index",
    "setitem": "Index",
    "getattr": "Index",
    "long": "Index",
    "type_as": "Index",
    "cumsum": "Sum",
    "to": "Index",
    "bool": "Index",
    "eq": "Index",
    "lt": "Index",
    "gt": "Index",
    "ge": "Index",
    "finfo": "Index",
    "ne": "Index",
    "le": "Index",
    "or_": "Index",
    "view": "View",
    "view_as": "View",
    "clone": "Clone",
    "flatten": "View",
    "contiguous": "View",
    "triu": "View",
    "expand": "Expand",
    "unsqueeze": "Unsqueeze",
    "slice": "Slice",
    "split": "Split",
    "cat": "Expand",
    "get_tensor_for_fx": "View",
    "make_viewless_tensor": "View",
    "reshape": "View",
    "expand_as": "Expand",
    "repeat": "Expand",
    "sum": "Sum",
    "log": "Pow",
    "int": "Int",
    # Create Data
    "ones": "MaskedFill",
    "zeros": "MaskedFill",
    "full": "MaskedFill",
    "tensor": "MaskedFill",
    "arange_for_fx": "MaskedFill",
    "arangeforfx": "MaskedFill",
    "arange": "MaskedFill",
    "masked_fill_": "MaskedFill",
    "masked_fill": "MaskedFill",
    "masked_fill": "MaskedFill",
    "full_like": "MaskedFill",
    "zeros_like": "MaskedFill",
    "one_hot": "MaskedFill",
    # Read data
    "embeddings": "Embedding",
    "embedding": "Embedding",
    "input": "SRC",
    "output": "SNK",
    # All Reduce is evaluated separately
    "copy_to_tensor_model_parallel_region": "AllReduceBwd",
    "all_reduce_for_fx_cross_entropy": "AllReduce",
    "reduce_from_tensor_model_parallel_region": "AllReduceFwd",
}


class FusedNodePlus(FusedNode):
    def __init__(self, id, phaze_node) -> None:
        op = (phaze_node.op).lower()

        if op not in phaze_op_mapping.keys():
            raise TypeError(
                "Type 2: Phaze to Fused operator mapping not available for op", op,
            )

        node_desc = phaze_op_mapping[op]
        activation_size = phaze_node.activation_size
        parameter_size = phaze_node.parameter_size
        stage_id = None
        output_act = phaze_node.activation_shape
        if phaze_node.saved_tensor_shape[-1]:
            saved_tensors = phaze_node.saved_tensor_shape
        else:
            saved_tensors = phaze_node.saved_tensor_shape[:-1]
        parameter = phaze_node.parameter_shape
        kernel_size = phaze_node.kernel_size
        stride = phaze_node.stride
        dilation = phaze_node.dilation
        padding = phaze_node.padding
        contiguous = phaze_node.contiguous

        super().__init__(
            id,
            node_desc,
            activation_size,
            parameter_size,
            stage_id,
            output_act,
            saved_tensors,
            parameter,
            kernel_size,
            stride,
            dilation,
            padding,
            contiguous,
        )

        self.layer_id = phaze_node.layer_id
        self.name = phaze_node.name
        self.component = phaze_node.component


def get_engine_type(node_desc):
    return get_compute_unit(node_desc)


def convert_phaze_to_fused_node(phaze_node, id):
    return FusedNodePlus(id, phaze_node)


def fuse_operators_in_graph(graph):
    f = fusion(graph, "../out", -1)
    graph = f.fuse()
    return graph


def convert_phaze_to_fused_graph(graph, mode="fwd"):
    fusedgraph = FusedGraph.Graph()

    fusednodes = {}

    for node in graph.nodes:
        fusednodes[node] = FusedNodePlus(node, graph.nodes[node]["node"])

    for src, dst in graph.edges:
        fusedgraph.add_edge(fusednodes[src], fusednodes[dst])

    fusedgraph = fuse_operators_in_graph(fusedgraph)

    if mode == "bwd":
        return FusedBwdGraph.Graph.from_str_bwd(str(fusedgraph))

    return fusedgraph


def construct_external_scheduler(fused_graph, arch_config, global_estimates_file=""):
    scheduler = ExternalScheduler(
        fused_graph, arch_config, global_estimates_file)

    return scheduler
