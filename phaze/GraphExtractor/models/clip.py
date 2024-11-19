# internal phaze code
from .model import BaseModelIR
from ..utils import ShapeProp, PhazeGraph
from ..utils import store_obj_to_file, load_obj_from_file
from ..utils import language_models

# torch module loads
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPConfig,CLIPTextConfig, CLIPVisionConfig
from transformers.utils import is_torch_fx_available
from PIL import Image
import requests

import os
from pathlib import Path
from .clip_attention import CLIPAttentionCustom
from .clip_attention import CLIPModel as CLIPModelCustom


if is_torch_fx_available():
    from transformers.utils.fx import (
        symbolic_trace as symbolic_trace_transformers,
    )


class ClipIR(BaseModelIR):
    def __init__(self, model_name="clip", tmp_width=1, model_config=None):
        super().__init__(model_name, tmp_width, model_config)

        self.out_dir = None
        self.graphmodule = None

        self.out_dir = self.create_out_dir()
        self.num_text_layers = 12
        self.num_vision_layers = 12

        if(self.model_config!=None):
            self.num_text_layers = self.model_config['num_hidden_layers']
            self.num_vision_layers = self.model_config['vision_num_hidden_layers']


    def set_model(self):
        self.trace_only_model = True

        if self.model_name == "clip":
            # 
            # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
            if(self.model_config!=None):
                text_configuration = CLIPTextConfig.from_pretrained("openai/clip-vit-base-patch16",hidden_size=self.model_config['hidden_size'],
                                                        num_hidden_layers=self.model_config['num_hidden_layers'], 
                                                        intermediate_size=self.model_config['intermediate_size'])
                vision_configuration = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16",hidden_size=self.model_config['vision_hidden_size'],
                                                        num_hidden_layers=self.model_config['vision_num_hidden_layers'], 
                                                        intermediate_size=self.model_config['vision_intermediate_size'])
                configuration = CLIPConfig.from_pretrained("openai/clip-vit-base-patch16", text_config=text_configuration, vision_config=vision_configuration, attn_implementation="eager")


                # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
                self.model = CLIPModelCustom(configuration)
            else:
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", attn_implementation="eager")
            # Accessing the model configuration
            
            text_configuration.num_attention_heads = self.model_config['num_attn_heads']
            vision_configuration.num_attention_heads = self.model_config['vision_num_attn_heads']

            # new_multihead_attention = CLIPAttentionCustom(text_configuration)
            # for m in self.model.text_model.encoder.layers:
            #     m.self_attn = new_multihead_attention

            # new_vision_multihead_attention = CLIPAttentionCustom(vision_configuration)
            # for m in self.model.vision_model.encoder.layers:
            #     m.self_attn = new_vision_multihead_attention

            print(self.model)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            raise TypeError("Model type not found in clip", self.model_name)

    def get_model_type(self):
        return "clip"

    def create_out_dir(self):
        curr_dir = Path(__file__).parent.absolute()
        curr_dir = os.path.join(curr_dir, "../out/Clip/")
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
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        def generate_text(seqlen):
            return 'a ' * seqlen
        t = generate_text(sequence_length)
        inputs = self.processor(text=t, images=image, return_tensors="pt", padding=True)
        
        input_ids = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.int32)
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = input_ids
        
 
        self.graphmodule: torch.fx.GraphModule = symbolic_trace_transformers(
            self.model, inputs)
        input_ids = input_ids #torch.Size([1, 77])
        attention_mask = input_ids 
        pixel_values = inputs["pixel_values"] #torch.Size([1, 3, 224, 224])

        model_shapeprop = ShapeProp(self.graphmodule)
        model_shapeprop.propagate(input_ids, pixel_values, attention_mask)

    def get_layer_id(self, n, curr_layer_id):
        layer_annotations = ["layer", "layers"]
        text_annotations = ["text"]
        vision_annotations = ["vision"]
        text_embedding_args = 'input_ids'
        vision_embedding_args = 'pixel_values'
        final_layer_annotations = ["vision_model_post_layernorm", "text_model_final_layer_norm"]


        node_name = n.name
        layer_details = node_name.split("_")
        if (node_name in final_layer_annotations):
            return (True, -4)
        if(text_embedding_args in str(n.args)):
            return (True, -3)
        if(vision_embedding_args in str(n.args)):
            return (True, -2)
        for l in range(0, len(layer_details)):
            if layer_details[l] in layer_annotations:
                if layer_details[l + 1] and layer_details[l + 1].isdigit():
                    if layer_details[0] in text_annotations:
                        return (True, int(layer_details[l + 1]) + self.num_vision_layers)
                    else:
                        assert(layer_details[0] in vision_annotations)
                        return (True, int(layer_details[l + 1]))
        return (False, 0)

    def create_graph_from_symbolic_trace(self):
        super().create_graph_from_symbolic_trace()

    def extract_model_graph(self, micro_batch_size=1, sequence_length=64, force_reextract_model=False,model_config=None):
        self.load_language_model(
            self.out_dir, micro_batch_size, sequence_length, force_reextract_model, model_config=model_config)
    
    def generate_layer_info(self):
        g = self.phazegraph

        if g is None:
            raise ValueError(
                "Model for model name" + self.model_name,
                "and tensor model parallel width" + self.tmp_width + "does not exist",
            )

        g.set_repeat_layer_ids(language_models, {0:list(range(self.num_vision_layers)), self.num_vision_layers:list(range(self.num_vision_layers, self.num_vision_layers + self.num_text_layers))})
        g.generate_layer_info()
        g.set_op_layer_graphs()
        g.contract_layer_graph()
