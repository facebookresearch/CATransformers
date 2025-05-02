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

baseline_config = {
    "opt": {
        "num_transformer_layers": 24,
        "p": 12,
        "d": 85,
        "t": 1,
    },

    "bertlarge": {
        "num_transformer_layers": 24,
        "p": 8,
        "d": 128,
        "t": 1,
    },

    "gpt2": {
        "num_transformer_layers": 48,
        "p": 32,
        "d": 32,
        "t": 1,
    },

    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 8,
        "d": 128,
        "t": 1,
    },

    "megatrongpt2-xl": {
        "num_transformer_layers": 40,
        "p": 8,
        "d": 128,
        "t": 1,
    },

    "megatrongpt2-54": {
        "num_transformer_layers": 54,
        "p": 8,
        "d": 32,
        "t": 4,
    },

    "megatrongpt2-72": {
        "num_transformer_layers": 72,
        "p": 8,
        "d": 16,
        "t": 8,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 32,
        "d": 8,
        "t": 4,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 8,
        "d": 128,
        "t": 1,
    },

}
