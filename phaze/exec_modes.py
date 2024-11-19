# internal phaze imports
from .GraphExtractor import extract_graph
from .Estimator import populate_estimates, estimate_carbon
from .Solver import run_solver


def extract_only(model_names, max_tmp_width, micro_batch_size,
                 sequence_length, force_reextract_model,):
    # Every node has a corresponding estimates in a 3D matrix <TMP strategy,
    # core dimensions, and number of cores>
    for model_name in model_names:
        extract_graph(model_name, max_tmp_width, micro_batch_size,
                      sequence_length, force_reextract_model,)


def extract_and_populate(model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,force_reextract_estimates=False, model_config=None, cc_from_arg=None):
    # Extract graph using Torch.fx
    print("Extracting graph for model: ", model_name, " ...")
    tmpc_models = extract_graph(
        model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,model_config)

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


def estimate_total_carbon(model_names, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model, force_reextract_estimates, hbm_size,model_config=None, cc_from_arg=None, parallel_num=1):

    models_info = {}

    for model_name in model_names:
        tmpc_models, latency_estimates = extract_and_populate(
            model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model, force_reextract_estimates, model_config, cc_from_arg)

        models_info[model_name] = (tmpc_models, latency_estimates)

    print("Extracted graph and populated for all the models ...")

    import csv
    import os
    filename = f'results_arch_search_{parallel_num}.csv'
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Text Layers', 'Text Embed Dim', 'Text FFN Dim', 'Vision Layers', 'Vision Embed Dim', 'Vision FFN Dim', 'arch config', 'embodied', 'operational', 'total_carbon', 'latency', 'breakdown'])
            result_list = estimate_carbon(models_info, micro_batch_size, max_tmp_width, sequence_length, hbm_size)
            if result_list !=None:
                for result in result_list:
                    writer.writerow([model_config["num_hidden_layers"],model_config["hidden_size"],model_config["intermediate_size"],model_config["vision_num_hidden_layers"], model_config["vision_hidden_size"], model_config["vision_intermediate_size"], result['config'],result['embodied'], result['operational'], result["total_carbon"], result['latency'],result['breakdown']]) 
    else:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            result_list = estimate_carbon(models_info, micro_batch_size, max_tmp_width, sequence_length, hbm_size)
            if result_list !=None:
                for result in result_list:
                    writer.writerow([model_config["num_hidden_layers"],model_config["hidden_size"],model_config["intermediate_size"],model_config["vision_num_hidden_layers"], model_config["vision_hidden_size"], model_config["vision_intermediate_size"], result['config'],result['embodied'], result['operational'], result["total_carbon"], result['latency'],result['breakdown']]) 
    if result_list !=None:
        return (result_list[0]["total_carbon"], result_list[0]['latency'])
    else:
        import math
        return (math.inf,  math.inf)
