"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from cnn_optimizer_wrapper import CNNOptimizer
from software.cnns import ResNet18
from hardware.eyeriss import acc_eyeriss_like
import sys
import os
sys.path.append(os.getcwd() + "/examples/eyeriss_like")
sys.path.append(os.getcwd() + "/commons")

eyeriss = acc_eyeriss_like()

cnn = ResNet18(n=1).get_layers()
cnn_non_duplicate = [l for l in cnn if not l.duplicate]
lyr1 = cnn_non_duplicate[0]
print("\nThe layer configuration:")
print(lyr1)

opt = CNNOptimizer(lyr1, eyeriss)
tiles, orders = opt.solve(access_energies=eyeriss.access_energies,
                          EDP_flag=True,
                          threads=8
                          )

print("\nFormatted result")
print(opt.optimal_mapping)

# comment out below to see the raw result of optimizer
# print(tiles)
# print(orders)
