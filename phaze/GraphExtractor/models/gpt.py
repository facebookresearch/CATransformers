"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

# internal phaze code
from .model import BaseModelIR
from ..utils import ShapeProp, PhazeGraph
from ..utils import store_obj_to_file, load_obj_from_file, custom_tracer_for_megatron
from ..utils import megatron_language_models

# torch module loads
import torch
from transformers import GPT2Tokenizer, GPT2Model
from transformers.utils import is_torch_fx_available

try:
    from megatron.model import GPTModel as MegatronGptModel
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


class GptIR(BaseModelIR):
    def __init__(self, model_name="gpt2", tmp_width=1):
        super().__init__(model_name, tmp_width)

        self.out_dir = None
        self.graphmodule = None

        self.out_dir = self.create_out_dir()

    def set_model(self):
        self.trace_only_model = True

        # references for sizes of the models:
        # https://arxiv.org/pdf/1909.08053.pdf
        # https://huggingface.co/transformers/v2.2.0/pretrained_models.html
        if self.model_name == "gpt2":  # 1.5B
            self.model = GPT2Model.from_pretrained("gpt2-xl")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

        elif self.model_name in ["megatrongpt2-xl", "megatrongpt2-54", "megatrongpt2-72",]:
            if self.model_name == "megatrongpt2-xl":  # 1.5B -> but currently at 350M
                num_layers, hidden_size, attention_heads = 1, 1024, 16

            elif (
                self.model_name == "megatrongpt2-54"
            ):  # does not work due to some mismatch in tensor sizes for TMPC 8
                num_layers, hidden_size, attention_heads = 1, 1920, 20

            elif self.model_name == "megatrongpt2-72":  # 8.3B, changed from 72 to 1 to enable running
                num_layers, hidden_size, attention_heads = 1, 3072, 24

            print(
                "MegatronGPT is initialized with each tmp width. Current width ",
                self.tmp_width,
            )
            self.set_args_megatron_gpt2(
                num_layers, hidden_size, attention_heads)
            self.model = self.megatron_model_provider()

        elif self.model_name == "megatrongpt3":  # 175B
            print(
                "MegatronGPT is initialized with each tmp width. Current width ",
                self.tmp_width,
            )
            self.set_args_megatron_gpt3()
            self.model = self.megatron_model_provider()

        else:
            raise TypeError("Model type not found in GPT", self.model_name)

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
        model = MegatronGptModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        ).to(device)

        model.half()
        return model

    def get_model_type(self):
        return "Gpt"

    def create_out_dir(self):
        curr_dir = Path(__file__).parent.absolute()
        curr_dir = os.path.join(curr_dir, "../out/GPT/")
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

    def set_args_megatron_gpt2(self, num_layers=24, hidden_size=1024, attention_heads=16):
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
        VOCAB_FILE = "./vocabfiles/vocab.json"
        MERGE_FILE = "./vocabfiles/merges.txt"
        DATA_PATH = "my-gpt_text_sentence"

        GPT_ARGS = {
            "--num-layers": num_layers,
            "--hidden-size": hidden_size,
            "--num-attention-heads": attention_heads,
            "--seq-length": 1024,
            "--max-position-embeddings": 1024,
            "--fp16": True,
            "--micro-batch-size": 8,
            "--global-batch-size": 8,
            "--vocab-file": VOCAB_FILE,
            "--merge-file": MERGE_FILE,
            "--tokenizer-type": "GPT2BPETokenizer",
            "--no-async-tensor-model-parallel-allreduce": True,
        }

        OUTPUT_ARGS = {
            "--log-interval": 10,
            "--save-interval": 1000,
            "--eval-interval": 1000,
            "--eval-iters": 40,
            "--activations-checkpoint-method": "uniform",
        }

        PARALLEL_ARGS = {
            "--tensor-model-parallel-size": TENSOR_MP_SIZE,
            "--pipeline-model-parallel-size": PIPELINE_MP_SIZE,
            "--DDP-impl": "torch",
            "--no-masked-softmax-fusion": True,
        }

        #ALL_ARGS = DISTRIBUTED_ARGS | GPT_ARGS | OUTPUT_ARGS | PARALLEL_ARGS
        ALL_ARGS = {**DISTRIBUTED_ARGS, **GPT_ARGS,
                    **OUTPUT_ARGS, **PARALLEL_ARGS}

        def add_args(argdict):
            for key, val in argdict.items():
                sys.argv.append(key)
                sys.argv.append(str(val))

        add_args(ALL_ARGS)

        initialize_megatron(ignore_unknown_args=True)

    def set_args_megatron_gpt3(self):
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
        VOCAB_FILE = "./vocabfiles/vocab.json"
        MERGE_FILE = "./vocabfiles/merges.txt"
        DATA_PATH = "my-gpt_text_sentence"

        GPT_ARGS = {
            # For GPT3- the num-layers should be 96
            "--num-layers": 1,
            "--hidden-size": 12288,
            "--num-attention-heads": 96,
            "--seq-length": 2048,
            "--max-position-embeddings": 2048,
            "--lr": 6.0e-5,
            "--min-lr": 6.0e-6,
            "--split": "98,2,0",
            "--lr-decay-style": "cosine",
            "--fp16": True,
            "--adam-beta1": 0.9,
            "--adam-beta2": 0.95,
            "--clip-grad": 1.0,
            "--weight-decay": 0.1,
            "--lr-decay-iters": 990000,
            "--train-iters": 2000000,
            "--lr-warmup-fraction": 0.01,
            "--micro-batch-size": 1,
            "--global-batch-size": 1,
            "--vocab-file": VOCAB_FILE,
            "--merge-file": MERGE_FILE,
            "--tokenizer-type": "GPT2BPETokenizer",
            "--no-async-tensor-model-parallel-allreduce": True,
        }

        OUTPUT_ARGS = {
            "--log-interval": 10,
            "--save-interval": 1000,
            "--eval-interval": 1000,
            "--eval-iters": 40,
            "--activations-checkpoint-method": "uniform",
        }

        PARALLEL_ARGS = {
            "--tensor-model-parallel-size": TENSOR_MP_SIZE,
            "--pipeline-model-parallel-size": PIPELINE_MP_SIZE,
            "--DDP-impl": "torch",
            "--no-masked-softmax-fusion": True,
        }

        #ALL_ARGS = DISTRIBUTED_ARGS | GPT_ARGS | OUTPUT_ARGS | PARALLEL_ARGS
        ALL_ARGS = {**DISTRIBUTED_ARGS, **GPT_ARGS,
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
            position_ids = torch.zeros(
                micro_batch_size, sequence_length, dtype=torch.long,).to(device)
            lm_labels = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.long,).to(device)

            graphmodule: torch.fx.GraphModule = custom_tracer_for_megatron(
                self.model)

            model_shapeprop = ShapeProp(graphmodule)
            self.graphmodule = model_shapeprop.propagate(
                input_ids, position_ids, attention_mask, lm_labels, position_ids, None)

        else:
            input_ids = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.int32)

            self.graphmodule: torch.fx.GraphModule = symbolic_trace_transformers(
                self.model)

            model_shapeprop = ShapeProp(self.graphmodule)
            model_shapeprop.propagate(input_ids)

    def get_layer_id(self, n, curr_layer_id):
        layer_annotations = ["layer", "layers", "h"]

        node_name = n.name
        layer_details = node_name.split("_")
        for l in range(0, len(layer_details)):
            if layer_details[l] in layer_annotations:
                if layer_details[l + 1] and layer_details[l + 1].isdigit():
                    return (True, int(layer_details[l + 1]))
        return (False, 0)

    def create_graph_from_symbolic_trace(self):
        super().create_graph_from_symbolic_trace()

    # Note: this code does not add more layers to the graph, just populates the layer info
    def add_more_layer_info(self, ex_num_layers, repeat_layer_id):
        print("\033[96m" + "Extending number of layers (only) in the graph to",
              ex_num_layers, "for model", self.model_name, "\033[0m")
        self.phazegraph.extend_layer_info_sans_graph(
            ex_num_layers, repeat_layer_id)

    def extract_model_graph(self, micro_batch_size=1, sequence_length=64, force_reextract_model=False,):
        self.load_language_model(
            self.out_dir, micro_batch_size, sequence_length, force_reextract_model)

        num_layers = 0
        if self.model_name == "megatrongpt2-xl":
            num_layers = 40
        elif self.model_name == "megatrongpt2-54":
            num_layers = 54
        elif self.model_name == "megatrongpt2-72":
            num_layers = 72
        elif self.model_name == "megatrongpt3":
            num_layers = 96

        if num_layers:
            self.add_more_layer_info(num_layers, repeat_layer_id=0)
