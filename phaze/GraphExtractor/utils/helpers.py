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

# torch imports
import torch

# python imports
import os
import pickle
from pathlib import Path

# model categories
bert_models = ["bertbase", "bertlarge", "megatronbert"]
gpt_models = ["gpt2", "megatrongpt3", "megatrongpt2-xl",
              "megatrongpt2-54", "megatrongpt2-72",]
opt_models = ["opt"]
megatron_language_models = ["megatronbert", "megatrongpt3",
                            "megatrongpt2-xl", "megatrongpt2-54", "megatrongpt2-72",]
llama_models = ["llama2"]
transformer_models = ["transformer"]
clip_models = ["clip"]

language_models = bert_models + gpt_models + opt_models + llama_models + transformer_models + clip_models

assert set(megatron_language_models).issubset(set(language_models))


def generate_out_filename(model_name, ext, micro_batch_size=1, tmp_width=1, sequence_length=1, module_type="",):

    if module_type != "":
        module_type = "_" + module_type

    if model_name in megatron_language_models:
        out_file_name = (str(model_name) + "_mbs_" + str(micro_batch_size) + "_sl_" + str(
            sequence_length) + "_tmp_" + str(tmp_width) + module_type + "." + str(ext))
    elif model_name in language_models:
        out_file_name = (str(model_name) + "_mbs_" + str(micro_batch_size) +
                         "_sl_" + str(sequence_length) + module_type + "." + str(ext))
    else:
        out_file_name = str(model_name) + "_mbs_" + \
            str(micro_batch_size) + module_type + "." + str(ext)

    return out_file_name


def check_model(model_obj, model_class, supported_models):
    print("Loaded model from file")

    # Check 1
    if not isinstance(model_obj, model_class):
        print(
            "'\33[93m" + "Model load from file failed due to check 1. Exporting again." + "\033[0m")
        return False

    if model_obj.model_name not in supported_models:
        print(
            "'\33[93m" + "Model load from file failed due to check 2. Exporting again." + "\033[0m")
        return False

    # Check 2
    if type(model_obj.tmp_width) != int:
        print(
            "'\33[93m" + "Model load from file failed due to check 3. Exporting again." + "\033[0m")
        return False

    # Check 3
    if not isinstance(model_obj.model, torch.nn.Module):
        print(
            "'\33[93m" + "Model load from file failed due to check 4. Exporting again." + "\033[0m")
        return False

    print("Successfully loaded the model")

    return True


def get_out_filepath_from_modelname(model_name, micro_batch_size=1, tmp_width=1, sequence_length=1, module_type="",):
    out_dir = Path(__file__).parent.absolute()

    if model_name in bert_models:
        out_dir = os.path.join(out_dir, "../out/Bert/")
    elif model_name in gpt_models:
        out_dir = os.path.join(out_dir, "../out/GPT/")
    elif model_name in opt_models:
        out_dir = os.path.join(out_dir, "../out/OPT/")
    elif model_name in llama_models:
        out_dir = os.path.join(out_dir, "../out/Llama/")
    elif model_name in transformer_models:
        out_dir = os.path.join(out_dir, "../out/Transformer/")
    elif model_name in clip_models:
        out_dir = os.path.join(out_dir, "../out/Clip/")

    else:
        raise ValueError("Model type doesnt exist ", model_name)

    out_file_name = generate_out_filename(
        model_name, "pickle", micro_batch_size, tmp_width, sequence_length, module_type,)
    out_file_path = str(out_dir) + out_file_name

    return out_file_path


# model is stored as pickle, and trace is stored as .trace
def store_obj_to_file(model_name, obj, micro_batch_size=1, tmp_width=1, sequence_length=1, module_type="",):
    out_file_path = get_out_filepath_from_modelname(
        model_name, micro_batch_size, tmp_width, sequence_length, module_type,)

    with open(out_file_path, "wb") as file_:
        try:
            pickle.dump(obj, file_, pickle.HIGHEST_PROTOCOL)
            return True
        except:
            print("Object store was unsuccesful for", model_name,
                  "of type", module_type, "deleting the file",)
            os.remove(out_file_path)
            return False

# model should be loaded from pickle, and trace from .trace


def load_obj_from_file(model_name, micro_batch_size=1, tmp_width=1, sequence_length=1, module_type="",):
    out_file_path = get_out_filepath_from_modelname(
        model_name, micro_batch_size, tmp_width, sequence_length, module_type
    )

    if not os.path.isfile(out_file_path):
        print(
            "Object file not available to load. Extracting the model again.", out_file_path,)
        return None, False

    try:
        obj = pickle.load(open(out_file_path, "rb"))
        return obj
    except:
        print("Load unsuccesful for", model_name, "of type",
              module_type, "falling back to the basic path.",)
        return None, False
