# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from phaze.Estimator import phaze_coretype_mapping, tc_configs, vc_configs
from phaze.Estimator import create_core_config, get_flops
from phaze.Estimator import convert_phaze_to_fused_graph, get_engine_type

# external imports
import networkx as nx
try:
    import gurobipy as gurobi
except:
    print("\33[93m" + "Gurobi not installed. Please install gurobi if you want to use ILP solver." + '\033[0m')

from math import inf

# python imports
import sys
import heapq

allow_intra_op_simul_diff_types = False
# allow_intra_op_simul_diff_types means: if True, then we allow intra-op parallelism on VCs
# while TCs are also processing (either branching or also intra-op parallelism);
# if False, then intra-op parallelism of any kind blocks any other concurrent execution


def get_num_cores(config):
    num_cores_vc = config.num_vc
    num_cores_tc = config.num_tc
    return num_cores_tc, num_cores_vc


def extract_op_graph_ilp(per_l_op_graph, cc, latency_estimates):
    cc_s = cc._replace(num_tc=1, num_vc=1)

    def generate_digraph_from_fused_graph(wgraph, mode="fwd"):
        out_graph = nx.DiGraph()
        estimation_time = 0

        for node in wgraph.nodes.values():
            node_attr = {}

            node_attr["core_type"] = phaze_coretype_mapping[get_engine_type(
                node.node_desc)]

            if (node_attr["core_type"] == "Nop"):
                e = e_s = {"fwd": {"latency": 0, "estimation_time": 0, "energy": 0}, "bwd": {
                    "latency": 0, "estimation_time": 0, "energy": 0}}

            elif (node_attr["core_type"] == "TC" or node_attr["core_type"] == "TCandVC"):
                idx = tc_configs.index(create_core_config(cc, "TC"))
                idx_s = tc_configs.index(create_core_config(cc_s, "TC"))
                e = latency_estimates["TC"][str(node.node_id)][idx]
                e_s = latency_estimates["TC"][str(node.node_id)][idx_s]

            elif node_attr["core_type"] == "VC":
                # collective operator ops
                if "AllReduce" in node.node_desc:
                    e = e_s = latency_estimates["AR"][str(node.node_id)]
                else:
                    idx = vc_configs.index(create_core_config(cc, "VC"))
                    idx_s = vc_configs.index(create_core_config(cc_s, "VC"))
                    e = latency_estimates["VC"][str(node.node_id)][idx]
                    e_s = latency_estimates["VC"][str(node.node_id)][idx_s]

            estimation_time = estimation_time + e[mode]["estimation_time"]

            node_attr["latency"] = e_s[mode]['latency'] * 1000
            node_attr["intra_op_latency"] = e[mode]['latency'] * \
                1000  # converting to ms
            node_attr["energy"] = e_s[mode]['energy']
            node_attr["intra_op_energy"] = e[mode]['energy']

            if (node_attr["latency"] == inf and node_attr["intra_op_latency"] == inf):
                estimation_time = 0
                return None, 0

            if(node_attr["latency"] == inf and node_attr["intra_op_latency"] != inf):
                print(node, e, e_s)
                raise Exception(
                    "Intra-op latency is not inf but latency is inf")
            if(node_attr["latency"] != inf and node_attr["intra_op_latency"] == inf):
                print(node, e, e_s)
                raise Exception(
                    "Intra-op latency is inf but latency is not inf")

            node_tuple = (node.node_id, {'node': node_attr})
            out_graph.add_nodes_from([node_tuple])

        for src, dstnodes in wgraph.edges.items():
            for dst in dstnodes:
                out_graph.add_edge(src, dst.node_id)

        return out_graph, estimation_time

    w_fwd_graph = convert_phaze_to_fused_graph(per_l_op_graph, "fwd")
    w_bwd_graph = convert_phaze_to_fused_graph(per_l_op_graph, "bwd")

    # get_flops(w_fwd_graph)

    fwd_graph, e_time_fwd = generate_digraph_from_fused_graph(
        w_fwd_graph, "fwd")
    bwd_graph, e_time_bwd = generate_digraph_from_fused_graph(
        w_bwd_graph, "bwd")

    return fwd_graph, bwd_graph, e_time_fwd + e_time_bwd


def latency_per_layer(l_op_graph, acc_config, latency_estimates):

    fwd_graph, bwd_graph, estimation_time = extract_op_graph_ilp(
        l_op_graph, acc_config, latency_estimates)

    if fwd_graph == None or bwd_graph == None:
        return None, None, 0, 0, None

    best_latency_fwd, energy_fwd, utilization_fwd = optimize_latency(
        fwd_graph, acc_config)
    best_latency_bwd, energy_bwd, utilization_bwd = optimize_latency(
        bwd_graph, acc_config)

    # best_latency_fwd, energy_fwd, utilization_fwd = sequential_latency(
    #     fwd_graph, acc_config)
    # best_latency_bwd, energy_bwd, utilization_bwd = sequential_latency(
    #     bwd_graph, acc_config)

    if best_latency_fwd == None or best_latency_bwd == None:
        return best_latency_fwd, best_latency_bwd, 0, estimation_time, None

    utilization = {"TC": {}, "VC": {}}
    utilization["TC"]["peak"] = max(
        utilization_fwd["TC"]["peak"], utilization_bwd["TC"]["peak"])
    utilization["TC"]["avg"] = (utilization_fwd["TC"]["avg"] * best_latency_fwd +
                                utilization_bwd["TC"]["avg"] * best_latency_bwd) / (best_latency_fwd + best_latency_bwd)
    utilization["VC"]["peak"] = max(
        utilization_fwd["VC"]["peak"], utilization_bwd["VC"]["peak"])
    utilization["VC"]["avg"] = (utilization_fwd["VC"]["avg"] * best_latency_fwd +
                                utilization_bwd["VC"]["avg"] * best_latency_bwd) / (best_latency_fwd + best_latency_bwd)

    return best_latency_fwd, best_latency_bwd, energy_fwd + energy_bwd, estimation_time, utilization

# sanity check for sequential (all intra-op scheduling) 
def sequential_latency(graph, acc_config):
    num_cores_tc, num_cores_vc = get_num_cores(acc_config)

    finishing_time = {}
    latency = 0

    non_nop_nodes = [i for i in graph.nodes() if not graph.nodes[i]["node"]['core_type']=="Nop"]

    for j in non_nop_nodes:
        latency += graph.nodes[j]['node']['intra_op_latency']
    
    print("sequential: " + str(latency/1000))

    return (latency/1000, 0,0)


def optimize_latency(graph, acc_config):
    num_cores_tc, num_cores_vc = get_num_cores(acc_config)
    # first, try running ILP without any core constraints and see if not too many cores are used.
    # if so, then return. otherwise, add VC/TC core constraints as appropriate and rerun ILP
    add_VC_core_constraints = False
    add_TC_core_constraints = False
    while True:
        Warning("Running ILP with VC core constraints: %s, TC core constraints: %s" % (
            add_VC_core_constraints, add_TC_core_constraints))
        T, t, y, x = ilp(graph, num_cores_tc, num_cores_vc, add_VC_core_constraints=add_VC_core_constraints,
                         add_TC_core_constraints=add_TC_core_constraints)

        if T is None:
            return None, 0, None

        schedule_TC_cores_active, schedule_VC_cores_active, exec_energy, avg_core_utilization = schedule_using_ilp_x_values(
            graph, T, t, y, x, num_cores_tc, num_cores_vc)

        if schedule_TC_cores_active <= num_cores_tc and schedule_VC_cores_active <= num_cores_vc:
            # all done
            utilization = {"TC": {"peak": schedule_TC_cores_active / num_cores_tc, "avg": avg_core_utilization[0]},
                           "VC": {"peak": schedule_TC_cores_active / num_cores_tc, "avg": avg_core_utilization[1]},
                           }
            return T / 1000, exec_energy, utilization
        if schedule_TC_cores_active > num_cores_tc:
            if add_TC_core_constraints:
                raise "too many TC cores used even though TC core constraints were used in ILP"
            add_TC_core_constraints = True
        if schedule_VC_cores_active > num_cores_vc:
            if add_VC_core_constraints:
                raise "too many VC cores used even though VC core constraints were used in ILP"
            add_VC_core_constraints = True


def schedule_using_ilp_x_values(graph, T, t, y, x, num_cores_tc, num_cores_vc):
    # use the ILP x_ij values to schedule the operators
    # and then check at most how many cores are active at any given time

    schedule_TC_cores_active_now = 0
    schedule_TC_cores_active_max = 0
    schedule_VC_cores_active_now = 0
    schedule_VC_cores_active_max = 0

    num_tc_cores_used_by = {}
    num_vc_cores_used_by = {}
    finishing_time = {}

    non_nop_nodes = [i for i in graph.nodes() if not graph.nodes[i]
                     ['node']['core_type'] == 'Nop']

    # algorithm: we consider the DAG given by x,
    # i.e., we think there is an edge (i,j) whenever x_ij = 1.
    # to simplify code, we add a fake source node fake_source with edges to all nodes i.
    fake_source = "fake_source"
    finishing_time[fake_source] = 0
    x.update({fake_source: {}})
    for i in non_nop_nodes:
        x[fake_source][i] = True
    num_tc_cores_used_by[fake_source] = 0
    num_vc_cores_used_by[fake_source] = 0
    # we proceed in a topological order on this DAG
    # at all times, we maintain the incoming degree of each node (coming from nodes that are not done executing yet)
    incoming_degree = {
        i: 1 + sum([x[j][i] for j in non_nop_nodes if j != i]) for i in non_nop_nodes}
    # when a node gets incoming degree 0, we start executing it
    # the currently executing nodes are stored in a priority queue, ordered by finishing time

    # priority queue of (finishing_time, node) pairs
    pq = []
    heapq.heappush(pq, (0, fake_source))
    schedule_T = 0

    # calculating utilization of cores
    avg_utilization_tc = 0
    avg_utilization_vc = 0

    while len(pq) > 0:
        _, i = heapq.heappop(pq)
        # it is time finishing_time[i], and i is now done executing
        # update the active cores
        schedule_TC_cores_active_now -= num_tc_cores_used_by[i]
        schedule_VC_cores_active_now -= num_vc_cores_used_by[i]
        if schedule_TC_cores_active_now < 0 or schedule_VC_cores_active_now < 0:
            raise "bug: negative number of active cores"
        # now that i is done, its successors have one less incoming edge
        for j in non_nop_nodes:
            if j != i:
                if x[i][j]:
                    incoming_degree[j] -= 1
                    if incoming_degree[j] < 0:
                        raise "bug: negative incoming degree"
                    if incoming_degree[j] == 0:
                        # we start executing j now
                        # set the number of cores used by j
                        if graph.nodes[j]['node']['core_type'] == 'TC':
                            if y[j]:
                                num_tc_cores_used_by[j] = num_cores_tc
                            else:
                                num_tc_cores_used_by[j] = 1
                            num_vc_cores_used_by[j] = 0
                        elif graph.nodes[j]['node']['core_type'] == 'VC':
                            num_tc_cores_used_by[j] = 0
                            if y[j]:
                                num_vc_cores_used_by[j] = num_cores_vc
                            else:
                                num_vc_cores_used_by[j] = 1
                        elif graph.nodes[j]['node']['core_type'] == 'TCandVC':
                            if y[j]:
                                num_tc_cores_used_by[j] = min(
                                    num_cores_tc, num_cores_vc)
                            else:
                                num_tc_cores_used_by[j] = 1
                            num_vc_cores_used_by[j] = num_tc_cores_used_by[j]
                        elif graph.nodes[j]['node']['core_type'] == "Nop":
                            num_tc_cores_used_by[j] = 0
                            num_vc_cores_used_by[j] = 0
                        else:
                            raise "unknown core type"

                        # check if nothing else is running if intra-op parallelism
                        if y[j]:
                            if graph.nodes[j]['node']['core_type'] == 'TC':
                                if schedule_TC_cores_active_now > 0:
                                    raise "nothing should be running on TCs while TC intra-op parallelism is happening"
                                if (not allow_intra_op_simul_diff_types) and schedule_VC_cores_active_now > 0:
                                    raise "nothing should be running on VCs while TC intra-op parallelism is happening"
                            elif graph.nodes[j]['node']['core_type'] == 'VC':
                                if schedule_VC_cores_active_now > 0:
                                    raise "nothing should be running on VCs while VC intra-op parallelism is happening"
                                if (not allow_intra_op_simul_diff_types) and schedule_TC_cores_active_now > 0:
                                    raise "nothing should be running on TCs while VC intra-op parallelism is happening"
                            elif graph.nodes[j]['node']['core_type'] == 'TCandVC':
                                if schedule_TC_cores_active_now > 0:
                                    raise "nothing should be running on TCs while fused intra-op parallelism is happening"
                                if schedule_VC_cores_active_now > 0:
                                    raise "nothing should be running on VCs while fused intra-op parallelism is happening"

                        # update the active cores
                        schedule_TC_cores_active_now += num_tc_cores_used_by[j]
                        schedule_VC_cores_active_now += num_vc_cores_used_by[j]
                        schedule_TC_cores_active_max = max(
                            schedule_TC_cores_active_max, schedule_TC_cores_active_now)
                        schedule_VC_cores_active_max = max(
                            schedule_VC_cores_active_max, schedule_VC_cores_active_now)

                        # compute finishing_time[j]
                        if y[j]:
                            latency_of_j = graph.nodes[j]['node']['intra_op_latency']
                        else:
                            latency_of_j = graph.nodes[j]['node']['latency']
                        finishing_time[j] = finishing_time[i] + latency_of_j

                        avg_utilization_tc += schedule_TC_cores_active_now * latency_of_j / num_cores_tc
                        avg_utilization_vc += schedule_VC_cores_active_now * latency_of_j / num_cores_vc

                        schedule_T = max(schedule_T, finishing_time[j])

                        # put the finishing event of j on the priority queue
                        heapq.heappush(pq, (finishing_time[j], j))

    if abs(T - schedule_T) > 0.0001:
        print("max different from T:", str(abs(T - schedule_T)),
              "absolute values", T, schedule_T)

    # compute schedule_energy
    schedule_energy = 0
    for i in non_nop_nodes:
        if y[i]:
            schedule_energy += graph.nodes[i]['node']['intra_op_energy']
        else:
            schedule_energy += graph.nodes[i]['node']['energy']

    avg_utilization_tc /= schedule_T
    avg_utilization_vc /= schedule_T

    return schedule_TC_cores_active_max, schedule_VC_cores_active_max, schedule_energy, [avg_utilization_tc, avg_utilization_vc]


def ilp(graph, num_cores_tc, num_cores_vc, add_VC_core_constraints, add_TC_core_constraints):

    # we number the VC cores from 0 to num_cores_vc-1
    # and the TC cores from 0 to num_cores_tc-1
    #
    # and the first min(num_cores_tc, num_cores_vc) TC cores are
    # paired up (fused) with the first min(num_cores_tc, num_cores_vc) VC cores,
    # i.e., the pairs are: (0,0), (1,1), ..., (min(num_cores_tc, num_cores_vc)-1, min(num_cores_tc, num_cores_vc)-1)

    #print("Running ILP solver...")

    # ILP to minimize latency
    ilp_m = gurobi.Model("minimize_latency")
    ilp_m.setParam("LogToConsole", 0)
    ilp_m.setParam("LogFile", "gurobi.log")
    ilp_m.setParam("MIPGap", 0.01)
    ilp_m.setParam("TimeLimit", 3600)
    ilp_m.setParam("MIPFocus", 1)
    # if this is too large, then the reformulated
    # ex-quadratic constraints can behave funky
    ilp_m.setParam("IntFeasTol", 1e-9)
    # ilp_m.setParam("FeasibilityTol", 1e-9)
    ilp_m.setParam("IntegralityFocus", 1)

    non_nop_nodes = [i for i in graph.nodes() if graph.nodes[i]
                     ['node']["core_type"] != "Nop"]
    # create variables

    t = {}  # for i ∈ V : start time of operator i
    p = {}  # for i ∈ V : latency of operator i
    y = {}  # for i ∈ V : one if operator i is intra-operator parallelized, zero otherwise

    # for i, j ∈ V , i ̸ = j: if this is one, i finishes before j begins
    x = {j: {} for j in non_nop_nodes}

    if add_VC_core_constraints:
        # for i ∈ V, c ∈ C : one if operator i is assigned to core c, zero otherwise
        zvc = {j: {} for j in non_nop_nodes}
    if add_TC_core_constraints:
        # for i ∈ V, c ∈ C : one if operator i is assigned to core c, zero otherwise
        ztc = {j: {} for j in non_nop_nodes}

    # print("Created variables, next setting objective, and constraints...")

    # the makespan (overall latency) of the schedule
    T = ilp_m.addVar(vtype=gurobi.GRB.CONTINUOUS, lb=0.0)

    for i in non_nop_nodes:
        t[i] = ilp_m.addVar(vtype=gurobi.GRB.CONTINUOUS, lb=0.0)
        p[i] = ilp_m.addVar(vtype=gurobi.GRB.CONTINUOUS, lb=0.0)
        y[i] = ilp_m.addVar(vtype=gurobi.GRB.BINARY)
        for j in non_nop_nodes:
            if j != i:
                x[i][j] = ilp_m.addVar(vtype=gurobi.GRB.BINARY)

        if add_VC_core_constraints:
            for c in range(num_cores_vc):
                zvc[i][c] = ilp_m.addVar(vtype=gurobi.GRB.BINARY)
        if add_TC_core_constraints:
            for c in range(num_cores_tc):
                ztc[i][c] = ilp_m.addVar(vtype=gurobi.GRB.BINARY)

    # set objective
    ilp_m.setObjective(T, gurobi.GRB.MINIMIZE)

    def incomparable(i, j):
        return (not nx.has_path(graph, i, j)) and (not nx.has_path(graph, j, i))

    H_io = sum([graph.nodes[i]['node']['intra_op_latency']
               for i in non_nop_nodes])
    H_l = sum([graph.nodes[i]['node']['latency'] for i in non_nop_nodes])
    H = min(H_io, H_l)

    # add constraints
    for i in non_nop_nodes:
        # constraints for intra-operator vs non-intra-operator parallelized
        ilp_m.addConstr(T >= t[i] + p[i])  # constraint 3

        ilp_m.addConstr(p[i] == y[i]*graph.nodes[i]['node']['intra_op_latency'] +
                        (1-y[i])*graph.nodes[i]['node']['latency'])  # constraint 4

        ilp_m.addConstrs(x[i][j] + x[j][i] <=
                         1 for j in non_nop_nodes if j != i)  # constraint 5

        if graph.nodes[i]['node']['core_type'] == 'Nop':
            ilp_m.addConstr(y[i] == 0)

        for k in non_nop_nodes:
            if i != k:
                ilp_m.addConstrs(x[i][k] >= x[i][j] + x[j][k] - 1
                                 for j in non_nop_nodes if (j != i) and (j != k))  # constraint 6

        ilp_m.addConstrs(t[i] + p[i] - H * (1 - x[i][j]) <= t[j]
                         for j in non_nop_nodes if incomparable(i, j))  # constraint 10

        ilp_m.addConstrs(x[i][j] + x[j][i] >= y[i]
                         for j in non_nop_nodes
                         if incomparable(i, j)
                         and (
            graph.nodes[i]['node']["core_type"] == graph.nodes[j]['node']["core_type"]
            or graph.nodes[i]['node']["core_type"] == "TCandVC"
            or graph.nodes[j]['node']["core_type"] == "TCandVC"
            or (not allow_intra_op_simul_diff_types)
        )
        )  # constraint 11

        # constraints for precedence
        for j in nx.descendants(graph, i):
            if j != i and j in non_nop_nodes:
                ilp_m.addConstr(x[i][j] == 1)  # constraint 7
                ilp_m.addConstr(x[j][i] == 0)  # constraint 8
                ilp_m.addConstr(t[i] + p[i] <= t[j])  # constraint 9

        # constraints for resources
        if add_VC_core_constraints:
            # constraint 12: if i is VC or fused, it should be assigned to exactly one VC core (or intra-op parallelized)
            if graph.nodes[i]['node']["core_type"] == "VC" or graph.nodes[i]['node']["core_type"] == "TCandVC":
                ilp_m.addConstr(sum(zvc[i][c]
                                    for c in range(num_cores_vc)) + y[i] == 1)

            # constraint 13: if i is TC, it should not be assigned to any VC core
            if graph.nodes[i]['node']["core_type"] == "TC":
                if num_cores_vc > 1:
                    ilp_m.addConstrs(
                        zvc[i][c] == 0 for c in range(num_cores_vc))
                else:
                    ilp_m.addConstr(zvc[i][0] == 0)

            # constraint 18:
            ilp_m.addConstrs(x[i][j] + x[j][i] >= zvc[i][c] + zvc[j][c] - 1
                             for c in range(num_cores_vc) for j in non_nop_nodes if incomparable(i, j))

        if add_TC_core_constraints:
            # constraint 12: if i is TC or fused, it should be assigned to exactly one TC core (or intra-op parallelized)
            if graph.nodes[i]['node']["core_type"] == "TC" or graph.nodes[i]['node']["core_type"] == "TCandVC":
                ilp_m.addConstr(sum(ztc[i][c]
                                    for c in range(num_cores_tc)) + y[i] == 1)
            # constraint 14: if i is VC, it should not be assigned to any TC core
            if graph.nodes[i]['node']["core_type"] == "VC":
                if num_cores_tc > 1:
                    ilp_m.addConstrs(
                        ztc[i][c] == 0 for c in range(num_cores_tc))
                else:
                    ilp_m.addConstr(ztc[i][0] == 0)

            # constraint 18:
            ilp_m.addConstrs(x[i][j] + x[j][i] >= ztc[i][c] + ztc[j][c] - 1
                             for c in range(num_cores_tc) for j in non_nop_nodes if incomparable(i, j))

        if add_VC_core_constraints and add_TC_core_constraints:
            # constraint 17: fused cores should be used together by fused operators
            if graph.nodes[i]['node']["core_type"] == "TCandVC":
                if min(num_cores_vc, num_cores_tc) > 1:
                    ilp_m.addConstrs(ztc[i][c] == zvc[i][c]
                                     for c in range(min(num_cores_vc, num_cores_tc)))
                else:
                    ilp_m.addConstr(ztc[i][0] == zvc[i][0])

    # print("Set the constraints...")
    # print('Running ILP optimizer...')
    sys.stdout.flush()
    ilp_m.optimize()

    if ilp_m.Status == gurobi.GRB.Status.INFEASIBLE:
        print("Infeasible ILP")
        return None, None, None, None
    elif ilp_m.Status == gurobi.GRB.Status.OPTIMAL:
        # print("Latency value is:", T.X)
        ilp_m.update()
    else:
        raise "Wrong status code"

    # print('Runtime = ', "%.2f" % ilp_m.Runtime, 's', sep='')

    def is_one(x):
        # x: Gurobi binary variable
        return x.X > 0.99

    # get the results
    T_out = T.X
    y_out = {i: is_one(y_i) for i, y_i in y.items()}
    t_out = {i: t_i.X for i, t_i in t.items()}
    x_out = {i: {} for i in x.keys()}
    for i, vi in x.items():
        for j in vi.keys():
            x_out[i][j] = is_one(x[i][j])

    return T_out, t_out, y_out, x_out
