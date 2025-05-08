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

# internal phaze code
from .model import BaseModelIR
from ..utils import ShapeProp, PhazeGraph
from ..utils import store_obj_to_file, load_obj_from_file

# torch module loads
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, LlamaModel, AutoTokenizer
from transformers.utils import is_torch_fx_available

import os
from pathlib import Path


if is_torch_fx_available():
    from transformers.utils.fx import (
        symbolic_trace as symbolic_trace_transformers,
    )


class LlamaIR(BaseModelIR):
    def __init__(self, model_name="llama2", tmp_width=1, model_config=None):
        super().__init__(model_name, tmp_width, model_config)

        self.out_dir = None
        self.graphmodule = None

        self.out_dir = self.create_out_dir()

    def set_model(self):
        self.trace_only_model = True

        if self.model_name == "llama2":
            # Initializing a LLaMA llama-7b style configuration LlamaForCausalLM
            if(self.model_config!=None):
                config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", hidden_size=self.model_config['hidden_size'],
                                                    num_hidden_layers=self.model_config['num_hidden_layers'], 
                                                    intermediate_size=self.model_config['intermediate_size'],
                                                    num_attention_heads=self.model_config['num_attn_heads'],
                                                    attn_implementation="eager", token='')
                self.model = LlamaForCausalLM(config)
            else:
                self.model = LlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf", attn_implementation="eager", token='')

            self.tokenizer = LlamaTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf", token='', legacy=True)
        elif self.model_name == "llama3":
            # Initializing a LLaMA llama-7b style configuration LlamaForCausalLM

            if(self.model_config!=None):
                config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B", hidden_size=self.model_config['hidden_size'],
                                                    num_hidden_layers=self.model_config['num_hidden_layers'], 
                                                    intermediate_size=self.model_config['intermediate_size'],
                                                    num_attention_heads=self.model_config['num_attn_heads'],
                                                    attn_implementation="eager", token='')
                self.model = LlamaForCausalLM(config)
            else:
                self.model = LlamaForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B", attn_implementation="eager", token='')

            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B", token='', legacy=True)
        else:
            raise TypeError("Model type not found in llama2", self.model_name)

    def get_model_type(self):
        return "llama"

    def create_out_dir(self):
        curr_dir = Path(__file__).parent.absolute()
        curr_dir = os.path.join(curr_dir, "../out/Llama/")
        isExist = os.path.exists(curr_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(curr_dir)
            print("The new directory is created!")

        return curr_dir

    def get_out_dir(self):
        if not self.out_dir:
            raise ValueError("Out directory not setup for", self.model_name)

        return self.out_dir

    def print_graphmodule(self):
        self.graphmodule.print_readable()

    def obtain_symbolic_trace_model(self, micro_batch_size=1, sequence_length=1):
        input_ids = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.int32)

        self.graphmodule: torch.fx.GraphModule = symbolic_trace_transformers(
            self.model)

        model_shapeprop = ShapeProp(self.graphmodule)
        model_shapeprop.propagate(input_ids)

    def get_layer_id(self, n, curr_layer_id):
        layer_annotations = ["layer", "layers"]

        node_name = n.name
        layer_details = node_name.split("_")
        for l in range(0, len(layer_details)):
            if layer_details[l] in layer_annotations:
                if layer_details[l + 1] and layer_details[l + 1].isdigit():
                    return (True, int(layer_details[l + 1]))
        return (False, 0)

    def create_graph_from_symbolic_trace(self):
        super().create_graph_from_symbolic_trace()

    def extract_model_graph(self, micro_batch_size=1, sequence_length=64, force_reextract_model=False, model_config=None):
        self.load_language_model(
            self.out_dir, micro_batch_size, sequence_length, force_reextract_model, model_config=model_config)
