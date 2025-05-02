## Console script for running carbon-NAS

import argparse
import os
import sys
from optimization import optimizer_carbon, optimizer_latency, optimizer_all, optimizer_energy, optimizer_all_hw, optimizer_energy_model
from optimization import optimizer_carbon_hf, optimizer_latency_hf, optimizer_all_hf, optimizer_energy_hf
from optimization import plot_pareto
def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="experiment", help="Name of the experiment, results will be saved under this name")
    parser.add_argument('--metric', type=str, default="carbon", help="Metric for optimization Accuracy + [carbon, latency, all, energy], or HW only optimzations (all-hw)")
    parser.add_argument('--hf', action='store_true',help="This is a hugging face model (use for all non-CLIP models)")
    args = parser.parse_args()
    return parser, args

def main():
    home_dir = os.getcwd()
    directory = f"{home_dir}/results"
    parser, args = get_parser_args()

    if args.metric == "carbon":
        optimize_carbon(args)
        plot_pareto.pareto_frontier_carbon(args.name, directory)
    elif args.metric == "latency":
        optimize_latency(args)
        plot_pareto.pareto_frontier_latency(args.name, directory)
    elif args.metric == "all":
        optimize_all(args)
        plot_pareto.pareto_frontier_all(args.name, directory)
    elif args.metric == "energy":
        optimize_energy(args)
        plot_pareto.pareto_frontier_energy(args.name, directory)
    elif args.metric == "all-hw":
        if args.hf:
            print("Error: hf models do not currently support HW only optimization")
            return
        optimize_all_hw(args)
        plot_pareto.pareto_frontier_hw(args.name, directory)
    elif args.metric == "energy-model":
        if args.hf:
            print("Error: hf models do not currently support model only optimization")
            return
        optimize_energy_model(args)
        plot_pareto.pareto_frontier_energy_model(args.name, directory)
    else:
        print("Error: invalid metric")
        parser.print_help()
        return

def optimize_latency(base):
    if base.hf:
        optimizer_latency_hf.optimize(base.name)
    else:
        optimizer_latency.optimize(base.name)

def optimize_carbon(base):
    if base.hf:
        optimizer_carbon_hf.optimize(base.name)
    else:
        optimizer_carbon.optimize(base.name)

def optimize_all(base):
    if base.hf:
        optimizer_all_hf.optimize(base.name)
    else:
        optimizer_all.optimize(base.name)

def optimize_energy(base):
    if base.hf:
        optimizer_energy_hf.optimize(base.name)
    else:
        optimizer_energy.optimize(base.name)

def optimize_all_hw(base):
    optimizer_all_hw.optimize(base.name)

def optimize_energy_model(base):
    optimizer_energy_model.optimize(base.name)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover