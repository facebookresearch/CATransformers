from src.tiling import Problem
from examples.eyeriss_like.cnn_optimizer_wrapper import CNNOptimizer
from commons.software.cnns import ResNet18
from commons.hardware.weight_stationary import acc_weight_stationary
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/examples/eyeriss_like")
sys.path.append(os.getcwd() + "/commons")


def fully_constrained():
    acc = acc_weight_stationary()
    lyr = Problem(n=16, c=64, m=384, p=28, q=28, r=3, s=3)

    print("\nThe layer configuration:")
    print(lyr)

    opt = CNNOptimizer(lyr, acc)
    opt.solve(access_energies=acc.access_energies,
              EDP_flag=True,
              threads=8,
              sptl_cnstrnts=[{'C': 16}, {'M': 16}]
              )

    print("\nFormatted result")
    print(opt.optimal_mapping)


def partially_constrained():
    acc = acc_weight_stationary()
    lyr = Problem(n=16, c=64, m=384, p=28, q=28, r=3, s=3)

    print("\nThe layer configuration:")
    print(lyr)

    opt = CNNOptimizer(lyr, acc)
    opt.solve(access_energies=acc.access_energies,
              EDP_flag=True,
              threads=8,
              sptl_cnstrnts=[{'C': 8}, None]
              )

    print("\nFormatted result")
    print(opt.optimal_mapping)


def __main__():
    mode = input("This example deals with a 16x16 accelerator. Enter \'F\' to see an example of fully constrained unrolling (C=16, M=16) or \'P\' to see an example of partially constrained unrolling (C=8):")
    if mode.lower() == 'p':
        print("\nPartially constrained was selected")
        partially_constrained()
    elif mode.lower() == 'f':
        print("\nFully constrained was selected")
        fully_constrained()
    else:
        print("\nDidn't understand the selection. Showing fully constrained by default")
        fully_constrained()


__main__()
