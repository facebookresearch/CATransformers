# internal phaze imports
from phaze.GraphExtractor.models import BertIR
from phaze.GraphExtractor.models import GptIR
from phaze.GraphExtractor.models import OptIR
from phaze.GraphExtractor.models import LlamaIR
from phaze.GraphExtractor.models import BaseModelIR
from phaze.GraphExtractor.models import TransformerIR
from phaze.GraphExtractor.models import ClipIR
from .utils import load_obj_from_file, check_model
from .utils import (bert_models, gpt_models, opt_models, llama_models, transformer_models, clip_models, language_models,
                    )

# python imports
from math import log2

supported_models = language_models


def get_tmp_widths(max_tmp_width=1):
    tmp_widths = []

    if max_tmp_width == 1:
        return [1]
    max_log_width_iter = int(log2(max_tmp_width)) + 1
    for i in range(0, max_log_width_iter):
        tmp_widths.append(2**i)
    return tmp_widths


def extract_graph(model_name, max_tmp_width=1, micro_batch_size=1, sequence_length=64, force_reextract_model=False,model_config=None, pretrained=None):
    def extract_model_from_file(tmp_width):
        if not force_reextract_model:
            module_type_str = ""
            if (pretrained != None):
                pretrained_string = pretrained.replace("/", "_")
                pretrained_string = pretrained_string.replace(".", "_")
                module_type_str = module_type_str + "_"+ pretrained_string
            if (model_config != None):
                for key, item in model_config.items():
                    module_type_str = module_type_str + "_"+ key + str(item)
                                             
            model = load_obj_from_file(
                model_name, micro_batch_size, tmp_width, sequence_length,module_type_str)

            if check_model(model, BaseModelIR, supported_models):
                model_memory = model.phazegraph.get_memory_footprint()
                print(model_memory.parameter_size)
                return model

    model_name = model_name.lower()

    if model_name in bert_models:
        tmp_widths = get_tmp_widths(max_tmp_width)
        bertmodels = []

        for width in tmp_widths:
            bert = extract_model_from_file(width)
            if bert is None:
                bert = BertIR(model_name, width, model_config=model_config)
                bert.extract_model_graph(
                    micro_batch_size, sequence_length, force_reextract_model, model_config=model_config)
            bertmodels.append(bert)
        return bertmodels

    elif model_name in gpt_models:
        tmp_widths = get_tmp_widths(max_tmp_width)
        gptmodels = []

        for width in tmp_widths:
            if (model_name == "megatrongpt3" and width < 4):
                continue
            gpt = extract_model_from_file(width)
            if gpt is None:
                gpt = GptIR(model_name, width)
                gpt.extract_model_graph(
                    micro_batch_size, sequence_length, force_reextract_model)
            gptmodels.append(gpt)
        return gptmodels

    elif model_name in opt_models:
        opt = extract_model_from_file(max_tmp_width)

        if opt:
            return [opt]

        opt = OptIR(model_name)
        opt.extract_model_graph(
            micro_batch_size, sequence_length, force_reextract_model)

        return [opt]

    elif model_name in llama_models:
        llama = extract_model_from_file(max_tmp_width)

        if llama:
            return [llama]

        llama = LlamaIR(model_name, model_config=model_config)
        llama.extract_model_graph(
            micro_batch_size, sequence_length, force_reextract_model, model_config=model_config)

        return [llama]
    elif model_name in transformer_models:
        trans = extract_model_from_file(max_tmp_width)

        if trans:
            return [trans]

        trans = TransformerIR(model_name)
        trans.extract_model_graph(
            micro_batch_size, sequence_length, force_reextract_model)

        return [trans]

    elif model_name in clip_models:
        clip = extract_model_from_file(max_tmp_width)

        if clip:
            return [clip]
        
        if pretrained != None: # use custom pretrained model
            clip = ClipIR(model_name, model_config=model_config, pretrained=pretrained)
        else:
            clip = ClipIR(model_name, model_config=model_config) # use default openai pretrained model 
        clip.extract_model_graph(
            micro_batch_size, sequence_length, force_reextract_model, model_config=model_config, pretrained=pretrained)

        return [clip]

    else:
        raise ValueError(
            "Only following models '{}' are currently imported, but got {}".format(supported_models, model_name))
