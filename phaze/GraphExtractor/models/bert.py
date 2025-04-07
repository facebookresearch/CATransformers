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
from ..utils import custom_tracer_for_megatron
from ..utils import megatron_language_models

# torch module loads
import torch
from transformers import (BertTokenizer, BertModel, BertConfig)
from transformers.utils import is_torch_fx_available

try:
    from megatron.model import BertModel as MegatronBertModel
    from megatron.model import ModelType
    from megatron.initialize import initialize_megatron
    from megatron.global_vars import get_args
except:
    Warning("Megatron is not installed. Megatron models will not be supported for full import.")
    pass

import os
import sys
from pathlib import Path

if is_torch_fx_available():
    from transformers.utils.fx import (
        symbolic_trace as symbolic_trace_transformers,
    )


class BertIR(BaseModelIR):
    def __init__(self, model_name="bertbase", tmp_width=1, model_config=None):
        super().__init__(model_name, tmp_width, model_config)

        self.out_dir = None
        self.graphmodule = None

        self.out_dir = self.create_out_dir()

    def set_model(self):
        self.trace_only_model = False

        if self.model_name == "bertbase":
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.configuration = self.model.config
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.model_name == "bertlarge":
            if(self.model_config!=None):
                config = BertConfig.from_pretrained("bert-large-uncased", hidden_size=self.model_config['hidden_size'],
                                                    num_hidden_layers=self.model_config['num_hidden_layers'], 
                                                    intermediate_size=self.model_config['intermediate_size'])
                self.model = BertModel(config)
            else:
                self.model = BertModel.from_pretrained("bert-large-uncased")

            self.configuration = self.model.config
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-large-uncased")
        
        elif self.model_name == "megatronbert":
            print(
                "MegatronBert is initialized with each tmp width. Current width ",
                self.tmp_width,
            )
            self.set_args_megatron()
            self.model = self.megatron_model_provider()
        else:
            raise TypeError("Model type not found in Bert", self.model_name)
        print(self.model.config)
    def megatron_model_provider(
        self,
        pre_process=True,
        post_process=True,
    ):
        """Build the megatron model."""

        if not torch.cuda.is_available():
            ValueError(
                "Cuda is not available on the machine, cannot extract megatron graph")

        device = torch.device("cuda")

        args = get_args()
        args.model_type = ModelType.encoder_or_decoder
        num_tokentypes = 2 if args.bert_binary_head else 0
        model = MegatronBertModel(
            num_tokentypes=num_tokentypes,
            add_binary_head=args.bert_binary_head,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        ).to(device)

        model.half()

        return model

    def create_out_dir(self):
        curr_dir = Path(__file__).parent.absolute()
        curr_dir = os.path.join(curr_dir, "../out/Bert/")
        isExist = os.path.exists(curr_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(curr_dir)
            print("The new directory is created!")

        return curr_dir

    def get_model_type(self):
        return "Bert"

    def get_out_dir(self):
        if not self.out_dir:
            raise ValueError("Out directory not setup for", self.model_name)

        return self.out_dir

    def set_args_megatron(self):
        WORLD_SIZE = self.tmp_width
        RANK = 0

        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = str(1)
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        os.environ["RANK"] = str(RANK)
        os.environ["MASTER_PORT"] = "6000"
        os.environ["MASTER_ADDR"] = "localhost"

        TENSOR_MP_SIZE = self.tmp_width
        PIPELINE_MP_SIZE = 1

        DISTRIBUTED_ARGS = {
            "--nproc_per_node": WORLD_SIZE,
            "--nnodes": 1,
            "--node_rank": RANK,
            "--master_addr": "localhost",
            "--master_port": 6000,
        }

        CHECKPOINT_PATH = self.out_dir
        VOCAB_FILE = "./vocabfiles/bert-large-uncased-vocab.txt"
        DATA_PATH = "my-bert_text_sentence"

        BERT_ARGS = {  # This is Bert-Large, with sequence length of 512
            "--num-layers": 24,
            "--hidden-size": 1024,
            "--num-attention-heads": 16,
            "--seq-length": 512,
            "--max-position-embeddings": 512,
            "--lr": 0.0001,
            "--lr-decay-iters": 990000,
            "--train-iters": 2000000,
            "--min-lr": 0.00001,
            "--lr-warmup-fraction": 0.01,
            "--micro-batch-size": 1,
            "--global-batch-size": 1,
            "--vocab-file": VOCAB_FILE,
            "--split": "949, 50, 1",
            "--fp16": True,
            "--tokenizer-type": "BertWordPieceLowerCase",
            "--no-async-tensor-model-parallel-allreduce": True,
        }

        OUTPUT_ARGS = {
            "--log-interval": 10,
            "--save-interval": 500,
            "--eval-interval": 100,
            "--eval-iters": 10,
            "--activations-checkpoint-method": "uniform",
        }

        PARALLEL_ARGS = {
            "--tensor-model-parallel-size": TENSOR_MP_SIZE,
            "--pipeline-model-parallel-size": PIPELINE_MP_SIZE,
            "--DDP-impl": "torch",
            "--no-masked-softmax-fusion": True,
        }

        #ALL_ARGS = DISTRIBUTED_ARGS | BERT_ARGS | OUTPUT_ARGS | PARALLEL_ARGS
        ALL_ARGS = {**DISTRIBUTED_ARGS, **BERT_ARGS,
                    **OUTPUT_ARGS, **PARALLEL_ARGS}

        def add_args(argdict):
            for key, val in argdict.items():
                sys.argv.append(key)
                sys.argv.append(str(val))

        add_args(ALL_ARGS)

        initialize_megatron(ignore_unknown_args=True)

    def print_graphmodule(self):
        self.graphmodule.print_readable()

    def obtain_symbolic_trace_model(self, micro_batch_size=1, sequence_length=1):
        # extracting the graphmodule
        # megatron bert symbolic trace is specific for Megatron-LM
        # remainder models use transformer huggingface's symbolic trace function
        if self.model_name in megatron_language_models:
            device = torch.device("cuda")

            input_ids = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.long,).to(device)
            attention_mask = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.float,).to(device)
            token_type_ids = torch.zeros(
                micro_batch_size, sequence_length, dtype=torch.long,).to(device)
            lm_labels = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.long,).to(device)

            graphmodule: torch.fx.GraphModule = custom_tracer_for_megatron(
                self.model)

            model_shapeprop = ShapeProp(graphmodule)
            self.graphmodule = model_shapeprop.propagate(
                input_ids, attention_mask, token_type_ids, lm_labels)

        else:
            input_ids = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.int32)

            graphmodule: torch.fx.GraphModule = symbolic_trace_transformers(
                self.model)

            model_shapeprop = ShapeProp(graphmodule)
            self.graphmodule = model_shapeprop.propagate(input_ids)

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

    def extract_model_graph(self, micro_batch_size=1, sequence_length=64, force_reextract_model=False,):
        self.load_language_model(
            self.out_dir, micro_batch_size, sequence_length, force_reextract_model)
