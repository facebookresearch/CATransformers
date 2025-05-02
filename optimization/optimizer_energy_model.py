import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.exceptions.generation_strategy import GenerationStrategyRepeatedPoints

# Plotting imports and initialization
from ax.service.utils.report_utils import exp_to_df

from eval import model_eval, model_constants
from phaze import main
from configurations import TEXT_MODEL_PARAMS, VISION_MODEL_PARAMS, HW_PARAMS, NUM_TRIALS, AREA_CONSTRAINT, LATENCY_CONSTRAINT, AREA_CONSTRAINT_VALUE, LATENCY_CONSTRAINT_VALUE, MAX_TOPS, MAX_TOPS_CONSTRAINT, FREQUENCY, MODEL_ARCH, PRETRAINED
import csv, os, sys
import pandas as pd
import matplotlib.pyplot as plt

def model_accuracy(model_param) -> float:
        accuracy_ret, size = model_eval.train_and_eval(model_param, MODEL_ARCH, PRETRAINED)
        return float(accuracy_ret['mean_recall@1']), size


def model_carbon(model_param, hw_param) -> float:
        hf_pretrained = model_constants.orig_models[MODEL_ARCH]["hf-model"]
        model = ["CLIP"]
        phaze_seq_len = 77
        force_reextract_model = False
        force_reextract_estimates = True
        hbm_size = 1

        # latency, energy = model_eval.measure_energy(model_param, model_arch=MODEL_ARCH, pretrained=PRETRAINED)
        # carbon = 0
        # area = 0

        model, phaze_seq_len, force_reextract_model, hbm_size, hw_param, model_param
        carbon, latency, area, energy = main.estimate_carbon(model, phaze_seq_len, force_reextract_model, force_reextract_estimates, hbm_size, hw_param, model_param, hf_pretrained)
        return carbon, latency, area, energy

def calc_tops(hw_param) -> float:
    return (hw_param["num_tc"] * hw_param["width"] * hw_param["depth"] * 2 * FREQUENCY) / (1000**4)


# Evaluation Function
def evaluate(trial, parameters, csv_file_name):

    hw_config = {}

     # h100 architecture

    hw_config["num_tc"] = 528
    hw_config["num_vc"] = 16896

    hw_config["width"] = 16
    hw_config["depth"] = 4
    hw_config["width_vc"] = 16
    hw_config["GLB_Buffer"] = 50*1024 *1024
    hw_config["L2_Buffer"] = 256*1024
    hw_config["L2_BW"] = 256

    # # a100 architecture

    # hw_config["num_tc"] = 432
    # hw_config["num_vc"] = 6912

    # hw_config["width"] = 16
    # hw_config["depth"] = 4
    # hw_config["width_vc"] = 16
    # hw_config["GLB_Buffer"] = 32*1024 *1024
    # hw_config["L2_Buffer"] = 192*1024
    # hw_config["L2_BW"] = 128

# v100 architecture
    # hw_config["num_tc"] = 640
    # hw_config["num_vc"] = 5120

    # hw_config["width"] = 16
    # hw_config["depth"] = 4
    # hw_config["width_vc"] = 16
    # hw_config["GLB_Buffer"] = 6*1024 *1024
    # hw_config["L2_Buffer"] = 64*1024
    # hw_config["L2_BW"] = 128
    
    
    num_ffn_blocks = 8
    text_block_size = model_constants.orig_models[MODEL_ARCH]["text_ffn_dim"] / num_ffn_blocks
    vision_block_size = model_constants.orig_models[MODEL_ARCH]["vision_ffn_dim"] / num_ffn_blocks

    num_hidden_blocks = 8
    text_hidden_block_size = model_constants.orig_models[MODEL_ARCH]["text_embedding_dim"] / num_hidden_blocks
    vision_hidden_block_size = model_constants.orig_models[MODEL_ARCH]["vision_embedding_dim"] / num_hidden_blocks
    
    model_config = {}
    model_config["num_hidden_layers"] = parameters.get("num_hidden_layers")
    model_config["intermediate_size"] = int(parameters.get("intermediate_size") * text_block_size)
    model_config["hidden_size"] = int(parameters.get("hidden_size") * text_hidden_block_size)
    model_config["num_attn_heads"] = parameters.get("num_attn_heads")
    model_config["vision_num_hidden_layers"] = parameters.get("vision_num_hidden_layers")
    model_config["vision_intermediate_size"] = int(parameters.get("vision_intermediate_size")* vision_block_size)
    model_config["vision_hidden_size"] = int(parameters.get("vision_hidden_size") * vision_hidden_block_size)
    model_config["vision_num_attn_heads"] = parameters.get("vision_num_attn_heads")

    accuracy, size= model_accuracy(model_config)
    carbon, latency, area, energy = model_carbon(model_config, hw_config)
    tops = calc_tops(hw_config)

    with open(csv_file_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([trial, accuracy, carbon, latency, parameters, size, energy])

    return {"accuracy": (accuracy, 0.0), "carbon": (carbon, 0.0), "latency": (latency,0.0), "energy": (energy, 0.0)}

def optimize(run_name):

    if MODEL_ARCH not in model_constants.orig_models:
        print(f"ERROR: Invalid model type: {MODEL_ARCH}, exiting...")
        sys.exit(1)

    home_dir = os.getcwd()
    directory = f"{home_dir}/results/{run_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    
    ax_client = AxClient()
    # ChoiceParameterConfig.is_ordered=True 
    ax_client.create_experiment(
        name=f"{run_name}",
        parameters=[
            {
                "name": f"num_hidden_layers",
                "type": "range",
                "bounds": [TEXT_MODEL_PARAMS['MIN_LAYERS'], TEXT_MODEL_PARAMS['MAX_LAYERS']],
                "value_type": "int"
            }, 
            {
                "name": f"intermediate_size",
                "type": "range",
                "bounds": [TEXT_MODEL_PARAMS['MIN_FFN_BLOCK'], TEXT_MODEL_PARAMS['MAX_FFN_BLOCK']],
                "value_type": "int"
            }, 
            {
                "name": f"hidden_size",
                "type": "range",
                "bounds": [TEXT_MODEL_PARAMS['MIN_EMB_BLOCK'], TEXT_MODEL_PARAMS['MAX_EMB_BLOCK']],
                "value_type": "int"
            }, 
            {
                "name": f"num_attn_heads",
                "type": "range",
                "bounds": [TEXT_MODEL_PARAMS['MIN_ATTN_HEAD'], TEXT_MODEL_PARAMS['MAX_ATTN_HEAD']],
                "value_type": "int"
            }, 
            {
                "name": f"vision_num_hidden_layers",
                "type": "range",
                "bounds": [VISION_MODEL_PARAMS['MIN_VISION_LAYERS'], VISION_MODEL_PARAMS['MAX_VISION_LAYERS']],
                "value_type": "int"
            }, 
            {
                "name": f"vision_intermediate_size",
                "type": "range",
                "bounds": [VISION_MODEL_PARAMS['MIN_VISION_FFN_BLOCK'], VISION_MODEL_PARAMS['MAX_VISION_FFN_BLOCK']],
                "value_type": "int"
            }, 
            {
                "name": f"vision_hidden_size",
                "type": "range",
                "bounds": [VISION_MODEL_PARAMS['MIN_VISION_EMB_BLOCK'], VISION_MODEL_PARAMS['MAX_VISION_EMB_BLOCK']],
                "value_type": "int"
            }, 
            {
                "name": f"vision_num_attn_heads",
                "type": "range",
                "bounds": [VISION_MODEL_PARAMS['MIN_VISION_ATTN_HEAD'], VISION_MODEL_PARAMS['MAX_VISION_ATTN_HEAD']],
                "value_type": "int"
            }
        ],
        objectives={
            # `threshold` arguments are optional
            "accuracy": ObjectiveProperties(minimize=False, threshold=0.1),
            "energy": ObjectiveProperties(minimize=True,threshold=3.0),
        },
        outcome_constraints=[
            LATENCY_CONSTRAINT, 
        ],
        tracking_metric_names=[
            "latency",
            "carbon",
        ],
        overwrite_existing_experiment=True,
        choose_generation_strategy_kwargs={"should_deduplicate": True},
    )

    # ### Run Optimization

    csv_file_name = f"{directory}/{run_name}.csv"
    print(f"run: {run_name}, freq: {FREQUENCY}, num_trials: {NUM_TRIALS}, latency_constraint: {LATENCY_CONSTRAINT}")

    with open(csv_file_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Trial Number", "Accuracy", "Carbon", "Latency", "Parameters", "Model size", "Energy"])

    for i in range(NUM_TRIALS):
        try:
            parameters, trial_index = ax_client.get_next_trial()
            # Local evaluation here can be replaced with deployment to external system.
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(i,parameters, csv_file_name))
        except GenerationStrategyRepeatedPoints as e:
            ax_client.save_to_json_file(filepath=f'{directory}/{run_name}.json')
            df = exp_to_df(ax_client.experiment)
            # Save the DataFrame to a CSV file
            df.to_csv(f'{directory}/data_{run_name}.csv', index=False)
            print(f"Error occurred at iteration {i}: {e}")
            break

    # save results, and plot pareto graph  
    # Create a sample DataFrame
    df = exp_to_df(ax_client.experiment)
    # Save the DataFrame to a CSV file
    df.to_csv(f'{directory}/data_{run_name}.csv', index=False)

    ax_client.save_to_json_file(filepath=f'{directory}/{run_name}.json')