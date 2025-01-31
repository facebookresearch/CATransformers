from src.tiling import GenericTile, GenericProblem
from src.generic_optimizers import tls_sptl_tmprl, tls_tmprl_alpha_beta_mlt_thrd
import sys
import os
sys.path.append(os.getcwd())


class CNNOptimizer():
    def __init__(self, problem, accelerator):
        self.problem = problem
        self.bounds = {"P": problem.p, "Q": problem.q, "R": problem.r, "S": problem.s,
                       "C": problem.c, "M": problem.m, "N": problem.n}
        self.tens_desc = [[("P", "R"), ("Q", "S"), "C", "N"],
                          ["R", "S", "C", "M"], ["P", "Q", "M", "N"]]
        self.problem_generic = GenericProblem(self.bounds, self.tens_desc)

        self.dw_tens_desc = [[("P", "R"), ("Q", "S"), "D", "N"],
                             ["R", "S", "D"], ["P", "Q", "D", "N"]]
        self.dw_bounds = {"P": problem.p, "Q": problem.q, "R": problem.r, "S": problem.s,
                          "D": problem.depth, "N": problem.n}
        self.dw_problem_generic = GenericProblem(
            self.dw_bounds, self.dw_tens_desc)

        self.accelerator = accelerator
        self.levels = accelerator.levels
        self.optimal_energy = float('inf')
        self.optimal_cost = float('inf')
        self.optimal_mapping = None

    # ******************* compile with alpha-beta *******************
    def solve(self, access_energies=None, EDP_flag=False, threads=1, sptl_cnstrnts=None):
        sizes = [self.accelerator.get_memory_data_capacity(
            level) for level in range(self.levels)]
        if not access_energies:
            access_energies = self.access_energies
        assert(access_energies is not False)

        # ******************* L1 Optimization *******************
        access_e = []
        if self.accelerator.unified_L1:
            access_e.append(
                [access_energies["L1"], access_energies["L1"], access_energies["L1"]])
        else:
            access_e.append(
                [access_energies["L1"][2], access_energies["L1"][0], access_energies["L1"][1]])

        access_e.append([access_energies["L2"] + access_energies["communication"],
                         access_energies["L2"] +
                         access_energies["communication"],
                         access_energies["L2"] + access_energies["communication"]])

        NoC = (self.accelerator.memory_hierarchy[1].read_bandwidth,
               self.accelerator.memory_hierarchy[1].write_bandwidth)

        size = sizes[0]
        if type(size) is list:
            size = [size[2], size[0], size[1]]
            size = tuple(size)

        if self.problem.depthwise:
            org_prob = [(GenericTile(self.dw_problem_generic), None)]
        else:
            org_prob = [(GenericTile(self.problem_generic), None)]

        ret = tls_sptl_tmprl(
            tiles=org_prob,
            prob=self.problem_generic,
            mem_size=size,
            access_energies=access_e,
            x_axis=self.accelerator.spatial_X,
            y_axis=self.accelerator.spatial_Y,
            static=True,
            bw=[None, NoC],
            threads=8,
            sptl_cnstrnts=sptl_cnstrnts,
            edp=EDP_flag
        )

        L2_tile_order_pairs = ret[0]
        costs = ret[1]

        # ******************* L2 Optimization *******************
        access_e.append(
            [access_energies["L3"], access_energies["L3"], access_energies["L3"]])

        ret = tls_tmprl_alpha_beta_mlt_thrd(
            tiles=L2_tile_order_pairs,
            prob=self.problem_generic,
            mem_size=sizes[1],
            access_energies=access_e,
            costs=[cost[0] for cost in costs],
            x_axis=None,
            y_axis=None,
            static=False,
            prior=False,
            bw=[None, NoC, None],
            threads=8,
            bypass=None,
        )
        tiles = ret[0]
        orders = ret[1]

        # ******************* post-process the result *******************
        arch = [('L1', True), ('L2', False), ('DRAM', False)]
        tens = ["Inputs", "Weights", "Outputs"]
        self.optimal_mapping = tiles.yaml(orders, arch, tens)

        return tiles, orders
