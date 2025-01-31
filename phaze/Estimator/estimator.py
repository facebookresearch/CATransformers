# internal phaze imports
from phaze.GraphExtractor.utils import generate_out_filename

from .architecture import set_configs_to_explore
from .architecture import tc_configs, vc_configs, bandwidth, bytes_per_element, frequency, only_explore_specific_configs

from .utils import convert_phaze_to_fused_graph, initialize_accelerator, reset_accelerator, get_engine_type
from .utils import tensor_core_estimator, vector_core_estimator, allreduce_estimator
from .utils import phaze_coretype_mapping, e_tuple

# python imports
import os
import json
import traceback
from pathlib import Path


estimates_dir = os.path.join(Path(__file__).parent.absolute(), "estimates/")
global_estimates_filepath = os.path.join(estimates_dir, "global_estimates.json")

exisiting_estimates = {}

if not os.path.exists(global_estimates_filepath):
    with open(global_estimates_filepath, "w+") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    f.close()

existing_estimates = json.load(open(global_estimates_filepath, "r"))


def setup_architecure(cc_from_args=None):
    # Setting the architecture
    reset_accelerator()
    initialize_accelerator()
    return set_configs_to_explore(cc_from_args)


def populate_estimates(tmpc_models, max_tmp_width, micro_batch_size, sequence_length, force_reextract_estimates=False, cc_from_args=None):
    architecture_list = setup_architecure(cc_from_args)
    if(len(architecture_list) == 0):
        return None

    latency_estimates = {}

    model_type = tmpc_models[0].get_model_type()
    model_name = tmpc_models[0].model_name

    model_estimates_dir = os.path.join(estimates_dir, model_type)
    estimate_file_name = generate_out_filename(
        model_name, "json", micro_batch_size, max_tmp_width, sequence_length,)

    if only_explore_specific_configs:
        for model in tmpc_models:
            latency_estimates[str(model.tmp_width)
                              ] = append_latency_estimates(model)

    else:
        estimate_filepath = os.path.join(
            model_estimates_dir, estimate_file_name)

        if (not os.path.isfile(estimate_filepath)) or force_reextract_estimates:
            if (os.path.isfile(estimate_filepath)):
                # delete exisiting file  
                try:
                    os.remove(estimate_filepath)
                    print(f"File {estimate_filepath} deleted.")
                except:
                    print(f"File {estimate_filepath} doesn't exist.")

            if not os.path.exists(model_estimates_dir):
                os.makedirs(model_estimates_dir)

            for model in tmpc_models:
                latency_estimates[str(model.tmp_width)
                                  ] = append_latency_estimates(model)

            with open(estimate_filepath, "w") as f:
                try:
                    json.dump(latency_estimates, f,
                              ensure_ascii=False, indent=4)
                except:
                    os.remove(estimate_filepath)
                    print('\33[93m' + "Error writing to file: ",
                          estimate_filepath + '\033[0m')
            f.close

        else:
            latency_estimates = json.load(open(estimate_filepath))

    return latency_estimates


def write_global_estimates():
    with open(global_estimates_filepath, "w+") as f:
        json.dump(existing_estimates, f, ensure_ascii=False, indent=2)
    f.close()


def append_latency_estimates(model):
    # generate operator graphs
    # graphs with fused operators, to ensure operators across layers are not fused
    graphs = model.get_unique_op_graphs()
    phaze_graph = model.get_phaze_graph().get_graph()
    fusedgraphs = [convert_phaze_to_fused_graph(graph) for graph in graphs]

    global existing_estimates

    # estimates dictionary
    tc_estimates, vc_estimates, ar_estimates = {}, {}, {}

    def rd_wr_estimates_global(core_config, op_dim, e=None,):
        core_key = str(core_config)
        dim_key = "dim" + str(op_dim)

        ret_e = None
        if core_key in existing_estimates.keys():
            if dim_key in existing_estimates[core_key].keys():
                ret_e = e_tuple(*existing_estimates[core_key][dim_key])

        elif ret_e is None and e is not None:
            if core_key not in existing_estimates.keys():
                existing_estimates[core_key] = {}
            existing_estimates[core_key][dim_key] = e

        return ret_e

    try:
        for fusedgraph in fusedgraphs:
            if (len(list(fusedgraph.nodes)) <= 0):
                continue
            print("Estimating for layer_id", list(
                fusedgraph.nodes.values())[0].layer_id)
            for node in list(fusedgraph.nodes.values()):
                core_type = phaze_coretype_mapping[get_engine_type(
                    node.node_desc)]

                if (node.node_desc == "AllReduceBwd" or node.node_desc == "AllReduceFwd" or node.node_desc == "AllReduce"):
                    e = allreduce_estimator(
                        phaze_graph.nodes[node.node_id], model.tmp_width, node.node_desc, bandwidth, bytes_per_element,)
                    ar_estimates[str(node.node_id)] = e

                elif (core_type == "TC" or core_type == "TCandVC"):
                    tc_estimates[str(node.node_id)] = []
                    for cc in tc_configs:
                        e = tensor_core_estimator(
                            fusedgraph, node, cc, rd_wr_estimates_global, frequency)
                        tc_estimates[str(node.node_id)].append(e)

                elif (core_type == "VC"):
                    vc_estimates[str(node.node_id)] = []
                    for cc in vc_configs:
                        e = vector_core_estimator(
                            fusedgraph, node, cc, frequency)
                        vc_estimates[str(node.node_id)].append(e)

        write_global_estimates()
    except:
        write_global_estimates()
        traceback.print_exc()
        exit("Estimates error out!")

    return {"TC": tc_estimates, "VC": vc_estimates, "AR": ar_estimates, }
