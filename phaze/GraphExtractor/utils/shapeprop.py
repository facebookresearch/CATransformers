"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import torch
import torch.fx
from torch.fx.node import Node
from typing import Dict


class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod):
        self.mod = mod
        self.modules = dict(self.mod.named_modules())
        # print(mod)

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}
        wrap_fn_list = [
            "get_tensor_for_fx",
            "all_reduce_for_fx_main",
            "all_reduce_for_fx_cross_entropy",
            "copy_to_tmpc_region",
            "reduce_from_tmpc_region",
        ]

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def check_for_wrap_functions(node):
            if node.target.__name__ in wrap_fn_list:
                return True
            return False

        def execute_fx_specific_all_reduce(node):
            if node.target.__name__ in [
                "all_reduce_for_fx_main",
                "all_reduce_for_fx_cross_entropy",
                "reduce_from_tensor_model_parallel_region",
                "copy_to_tensor_model_parallel_region",
            ]:
                result = load_arg(node.args)[0]
                return result

        def fetch_attr(target: str):
            target_atoms = target.split(".")
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        def extract_module_properties(module, node):
            if hasattr(module, "kernel_size"):
                node.kernel_size = module.kernel_size
            if hasattr(module, "padding"):
                node.padding = module.padding
            if hasattr(module, "dilation"):
                node.dilation = module.dilation
            if hasattr(module, "stride"):
                node.stride = module.stride
            if hasattr(module, "contiguous"):
                node.contiguous = module.contiguous
            if hasattr(module, "weight"):
                node.weights_shape = [list(module.weight.size())]
            if hasattr(module, "bias"):
                if isinstance(module.bias, torch.Tensor):
                    node.bias_shape = [list(module.bias.size())]

        def extract_attr_properties(target, result):
            type_tensor = target.split(".")
            if (["embedding"] in type_tensor):
                operator = "embedding"
            else:
                operator = "getattr"

            if isinstance(result, torch.Tensor):
                if type_tensor[-1] == "weight":
                    node.weights_shape = [list(result.shape)]
                if type_tensor[-1] == "bias":
                    node.bias_shape = [list(result.shape)]

            return operator
        
        def extract_current_component(target):
            component = None
            type_tensor = target.split(".")
            embedding_annotations = ["embeddings"]
            attention_annotations = ["attention", "self_attn"]
            ffn_annotations = ["intermediate", "output", "mlp"]
            output_annotations = ["visual_projection", "text_projection"]

            if (set(embedding_annotations) & set(type_tensor)):
                component = "embeddings"
            elif (set(attention_annotations) & set(type_tensor)):
                component = "attention"
            elif (set(ffn_annotations) & set(type_tensor)):
                component = "FFN"
            elif ("pooler" in type_tensor):
                component = "pooler"
            elif (set(output_annotations) & set(type_tensor)):
                component = "Output"

            return component
        
        cur_component = "init"

        for node in self.mod.graph.nodes:
            # print(node)
            if node.op == "placeholder":
                result = next(args_iter)
                node.operator = "input"
            elif node.op == "get_attr":
                result = fetch_attr(node.target)
                node.operator = extract_attr_properties(node.target, result)
            elif node.op == "call_function":
                result = execute_fx_specific_all_reduce(node)
                if result is None:
                    result = node.target(
                        *load_arg(node.args), **load_arg(node.kwargs))
                node.operator = node.target.__name__
            elif node.op == "call_method":
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
                node.operator = node.target
            elif node.op == "call_module":
                result = self.modules[node.target](
                    *load_arg(node.args), **load_arg(node.kwargs))
                node.operator = type(self.modules[node.target]).__name__
                extract_module_properties(self.modules[node.target], node)
                comp = extract_current_component(node.target)
                if comp != None:
                    cur_component = comp

            elif node.op == "output":
                result = 0
                node.operator = "output"
                cur_component = "output"
            node.component = cur_component

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = [list(result.shape)]
                if hasattr(result, "tensor_model_parallel"):
                    node.tensor_model_parallel = result.tensor_model_parallel
                    node.partition_dim = result.partition_dim
                    node.partition_stride = result.partition_stride
                node.dtype = result.dtype
            elif isinstance(result, torch.Size):
                node.shape = [list(result)]
                node.stride = tuple([1])
            elif (
                isinstance(result, int)
                or isinstance(result, bool)
                or isinstance(result, float)
                or isinstance(result, torch.finfo)
            ):
                node.shape = [[1]]
                node.stride = tuple([1])
            elif isinstance(result, tuple):
                node.shape = [[]]
                for r in result:
                    node.shape.append(list(r.shape))
                node.dtype = r[0].dtype
            elif isinstance(result, torch.dtype) or isinstance(result, torch.device) or isinstance(result, str):
                node.shape = [[]]
                node.dtype = result
            elif result is None:
                node.shape = [[]]
                node.dtype = None
            else:
                print("Result type not found.", node,
                                node.op, result, type(result),)
            # else:
            #     raise TypeError("Result type not found.", node,
            #                     node.op, result, type(result),)

            env[node.name] = result

        return self.mod

        # return load_arg(self.graph.result)
