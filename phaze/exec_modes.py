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

# internal phaze imports
from .GraphExtractor import extract_graph
from .Estimator import populate_estimates, estimate_carbon
from .Solver import run_solver
import math


def extract_only(model_names, max_tmp_width, micro_batch_size,
                 sequence_length, force_reextract_model,):
    # Every node has a corresponding estimates in a 3D matrix <TMP strategy,
    # core dimensions, and number of cores>
    for model_name in model_names:
        extract_graph(model_name, max_tmp_width, micro_batch_size,
                      sequence_length, force_reextract_model,)


def extract_and_populate(model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,force_reextract_estimates=False, model_config=None, cc_from_arg=None, pretrained=None):
    # Extract graph using Torch.fx
    print("Extracting graph for model: ", model_name, " ...")
    tmpc_models = extract_graph(
        model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,model_config, pretrained)

    print("Extracted graph for model: ", model_name, " ...")

    latency_estimates = populate_estimates(
        tmpc_models, max_tmp_width, micro_batch_size, sequence_length,force_reextract_estimates, cc_from_arg)

    print("Populated estimates for model: ", model_name, " ...")

    return tmpc_models, latency_estimates


def extract_and_prepopulate(model_names, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,):
    # Every node has a corresponding estimates in a 3D matrix <TMP strategy,
    # core dimensions, and number of cores>
    for model_name in model_names:
        extract_and_populate(model_name, max_tmp_width,
                             micro_batch_size, sequence_length, force_reextract_model,)


def extract_and_solve(model_names, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model, activation_recomputation, hbm_size):
    models_info = {}

    for model_name in model_names:
        tmpc_models, latency_estimates = extract_and_populate(
            model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,)

        models_info[model_name] = (tmpc_models, latency_estimates)

    print("Extracted graph and populated for all the models ...")

    return run_solver(models_info, micro_batch_size, max_tmp_width, sequence_length, activation_recomputation, hbm_size)


def estimate_total_carbon(model_names, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model, force_reextract_estimates, hbm_size,model_config=None, cc_from_arg=None, pretrained=None):

    models_info = {}

    for model_name in model_names:
        tmpc_models, latency_estimates = extract_and_populate(
            model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model, force_reextract_estimates, model_config, cc_from_arg, pretrained)
        
        # No valid architecture found, no estimates generated 
        if (latency_estimates) == None:
            return (math.inf,  math.inf, math.inf, math.inf)
        
        models_info[model_name] = (tmpc_models, latency_estimates)

    print("Extracted graph and populated for all the models ...")

    result_list = estimate_carbon(models_info, micro_batch_size, max_tmp_width, sequence_length, hbm_size)
    if result_list !=None:
        # currently we only search for one model and one HW configuration at a time 
        return (result_list[0]["total_carbon"], result_list[0]['latency'], result_list[0]['area'], result_list[0]['energy'])
    else:
        return (math.inf,  math.inf, math.inf, math.inf)
