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
from phaze.arguments import process_phaze_arguments
from phaze.exec_modes import extract_and_prepopulate, extract_and_solve, extract_only, estimate_total_carbon
from phaze.GraphExtractor import supported_models

import time


def main(args):

    phaze_mbs = args.phaze_micro_batch_size
    phaze_model_names = args.phaze_model_names
    phaze_exec_type = args.phaze_exec_type
    phaze_seq_len = args.phaze_sequence_length
    phaze_max_tmpc = args.phaze_max_tmp_width
    force_reextract_model = args.force_reextract_model
    force_reextract_estimates = args.force_reextract_estimates
    print(force_reextract_model)
    hbm_size_list = args.phaze_hbm_size
    model_config = None
    if args.config_model:
        model_config={}
        model_config["num_hidden_layers"] = args.num_layers
        model_config["intermediate_size"] = args.intermediate_size
        model_config["hidden_size"] = args.hidden_size
        model_config["num_attn_heads"] = args.num_attn_heads
        model_config["vision_num_hidden_layers"] = args.vision_num_layers
        model_config["vision_intermediate_size"] = args.vision_intermediate_size
        model_config["vision_hidden_size"] = args.vision_hidden_size
        model_config["vision_num_attn_heads"] = args.vision_num_attn_heads
    pretrained=None
    if args.pretrained: 
        pretrained = args.pretrained
    
    assert set(phaze_model_names).issubset(
        set(supported_models)
    ), "Model not supported by Phaze. Please check the list of supported models in GraphExtractor.py"

    print("Running for hbm: ", hbm_size_list, " mbs: ",
          phaze_mbs, " max tmp: ", phaze_max_tmpc)
    if phaze_exec_type == "extract_graph":
        # Extract graph using Torch.fx
        # Fill details about the tensor sizes - weights, activations, and
        # intermediate results
        for micro_batch_size in phaze_mbs:
            extract_only(phaze_model_names, phaze_max_tmpc, micro_batch_size,
                         phaze_seq_len, force_reextract_model,)

    elif phaze_exec_type == "prepopulate_estimates":
        # Every node has a corresponding estimates in a 3D matrix <TMP
        # strategy, core dimensions, and number of cores>
        for micro_batch_size in phaze_mbs:
            extract_and_prepopulate(phaze_model_names, phaze_max_tmpc,
                                    micro_batch_size, phaze_seq_len, force_reextract_model,)
    elif phaze_exec_type == "estimate_carbon":
        for micro_batch_size in phaze_mbs:
            for hbm_size in hbm_size_list:
                estimate_total_carbon(phaze_model_names, phaze_max_tmpc, micro_batch_size, phaze_seq_len, force_reextract_model, force_reextract_estimates, hbm_size, model_config, None, pretrained)

    elif phaze_exec_type == "run_solver":

        # initialize variables for final "best" config
        final_config = None
        final_total_time = 0
        final_ilp_time = 0
        final_dp_time = 0
        final_estimation_time = 0
        final_micro_batch_size = 0
        final_hbm_size = 0
        final_throughput = 0
        final_activation_recomputation = False

        # search both activation recomp true and false
        activation_recomputations = [False, True]

        for micro_batch_size in phaze_mbs:
            for hbm_size in hbm_size_list:
                for activation_recomputation in activation_recomputations:

                    print("mbs: ", micro_batch_size, " HBM size: " + str(hbm_size) +
                          " activation_recomputation: " + str(activation_recomputation))

                    start = time.time()

                    final_phaze_config, estimation_time, ilp_time, dp_time = extract_and_solve(
                        phaze_model_names, phaze_max_tmpc, micro_batch_size, phaze_seq_len, force_reextract_model, activation_recomputation, hbm_size*1024*1024*1024)

                    end = time.time()

                    print("Best phaze config for mbs: ", micro_batch_size, " HBM: ", hbm_size, " Activation Recomp: ", activation_recomputation, "\n",
                          "Config ", final_phaze_config, "\n",
                          "Models", phaze_model_names, "\n",
                          "total solving time, ilptime and dptime", end - start, ilp_time, dp_time, "\n",
                          "estimation time", estimation_time)

                    final_total_time += end - start
                    final_ilp_time += ilp_time
                    final_dp_time += dp_time

                    cc, strategy = final_phaze_config

                    if strategy != None:
                        if strategy[phaze_model_names[0]].throughput > final_throughput:
                            final_config = final_phaze_config
                            final_micro_batch_size = micro_batch_size
                            final_hbm_size = hbm_size
                            final_throughput = strategy[phaze_model_names[0]].throughput
                            final_activation_recomputation = activation_recomputation
            final_estimation_time += estimation_time

        if final_phaze_config != None:
            print("Best phaze config for single model comparison: mbs: ", final_micro_batch_size, " HBM: ", final_hbm_size, " Activation Recomp: ", final_activation_recomputation, "\n",
                  "Config ", final_config, "\n",
                  "Model", strategy[phaze_model_names[0]], "\n",
                  "total solving time, ilptime and dptime", final_total_time, final_ilp_time, final_dp_time, "\n",
                  "estimation time", final_estimation_time)
        else:
            print("No valid configuration found")


if __name__ == "__main__":
    args = process_phaze_arguments()
    main(args)


def estimate_carbon(phaze_model_names, phaze_seq_len, force_reextract_model, force_reextract_estimates, hbm_size, hw_config, model_config, pretrained=None):
   carbon, latency, area, energy = estimate_total_carbon(phaze_model_names, 1, 1, phaze_seq_len, force_reextract_model, force_reextract_estimates, hbm_size, model_config, hw_config, pretrained)
   return carbon, latency, area, energy
