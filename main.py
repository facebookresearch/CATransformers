## Console script for running carbon-NAS

import argparse
import csv
import json
import os
import random
import sys
from copy import copy
from itertools import product
from optimization import optimizer_carbon, optimizer_latency, optimizer_all

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="experiment", help="Name of the experiment, results will be save under this name")
    parser.add_argument('--metric', type=str, default="carbon", help="Metric for optimization Accuracy + [carbon, latency, all]")
    args = parser.parse_args()
    return parser, args

def main():
    parser, args = get_parser_args()
    
    if args.metric == "carbon":
        optimize_carbon(args)
    elif args.metric == "latency":
        optimize_latency(args)
    elif args.metric == "all":
        optimize_all(args)
    else:
        print("Error: invalid metric")
        parser.print_help()
        return

def optimize_latency(base):
    optimizer_latency.optimize(base.name)

def optimize_carbon(base):
    optimizer_carbon.optimize(base.name)

def optimize_all(base):
    optimizer_all.optimize(base.name)

# def post_pruning_training():
#     return None

# def evaulate_model():
#     return None

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover