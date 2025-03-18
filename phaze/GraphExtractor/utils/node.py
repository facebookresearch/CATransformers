"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from math import prod


class PhazeNode:
    def __init__(self, node_info, id, layer_id, inputs):

        # basic information
        self.name = node_info.target
        self.layer_id = layer_id
        self.node_id = id
        self.op_type = node_info.op
        self.op = node_info.operator
        self.isTensorParallelized = False

        # meta information
        self.kernel_size = None
        self.stride = None
        self.dilation = None
        self.contiguous = True
        self.padding = None

        # tensor (input and output) shapes
        # a 2D List of Tensor shapes
        self.input_shape = [[]]
        self.activation_shape = [[]]
        self.weights_shape = [[]]
        self.bias_shape = [[]]
        self.saved_tensor_shape = [[]]
        self.parameter_shape = [[]]

        # sizes
        self.parameter_size = None
        self.optimizer_size = None
        self.activation_size = None
        self.stashed_size = None

        #component in the architecture (attention, FFN, embedding...etc)
        self.component = node_info.component 

        self.set_meta_info(node_info, inputs)

    def set_meta_info(self, node_info, inputs):
        if hasattr(node_info, "kernel_size"):
            self.kernel_size = node_info.kernel_size
        if hasattr(node_info, "padding"):
            self.padding = node_info.padding
        if hasattr(node_info, "dilation"):
            self.dilation = node_info.dilation
        if hasattr(node_info, "stride"):
            self.stride = node_info.stride
        if hasattr(node_info, "contiguous"):
            self.contiguous = node_info.contiguous
        if hasattr(node_info, "weights_shape"):
            self.weights_shape = node_info.weights_shape
        if hasattr(node_info, "bias_shape"):
            self.bias_shape = node_info.bias_shape
        if hasattr(node_info, "tensor_model_parallel"):
            self.isTensorParallelized = True
            self.tensor_model_parallel = node_info.tensor_model_parallel
        if hasattr(node_info, "partition_dim"):
            self.partition_dim = node_info.partition_dim
        if hasattr(node_info, "partition_stride"):
            self.partition_stride = node_info.partition_stride

        self.activation_shape = node_info.shape

        self.input_shape = inputs

        self.saved_tensor_shape = inputs
        self.saved_tensor_shape += self.weights_shape

        self.parameter_shape = self.weights_shape + self.bias_shape

    def get_id(self):
        return self.node_id

    def get_layer_id(self):
        return self.layer_id

    def get_name(self):
        return self.name

    def get_op_type(self):
        return self.op_type

    def get_fw_time(self):
        return self.fw_time

    def get_bw_time(self):
        return self.bw_time

    def get_operator(self):
        return self.op

    def get_exec_time(self):
        ret_exec = {}
        ret_exec["fwd_time"] = self.get_fw_time()
        ret_exec["bwd_time"] = self.get_bw_time()
        return ret_exec

    def return_size(self, shapes):
        res = 0
        for shape in shapes:
            if len(shape) > 0:
                res = res + prod(shape)

        return res

    def get_activation_size(self):
        # One for fwd pass and other for the gradients during bwd pass
        self.activation_size = self.return_size(self.activation_shape)
        return self.activation_size

    def get_parameter_size(self):
        self.parameter_size = self.return_size(
            self.weights_shape) + self.return_size(self.bias_shape)

        return self.parameter_size

    def get_inputs(self):
        return self.input_shape

    def get_memory_estimate(self):
        return (self.get_parameter_size(), self.get_activation_size(),)
    def get_component(self):
        return self.component
