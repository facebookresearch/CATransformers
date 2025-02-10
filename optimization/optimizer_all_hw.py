import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.exceptions.generation_strategy import GenerationStrategyRepeatedPoints

# Plotting imports and initialization
from ax.service.utils.report_utils import exp_to_df

from eval import model_eval, model_constants
from phaze import main
from configurations import TEXT_MODEL_PARAMS, VISION_MODEL_PARAMS, HW_PARAMS, NUM_TRIALS, AREA_CONSTRAINT, AREA_CONSTRAINT_VALUE, MAX_TOPS, MAX_TOPS_CONSTRAINT, FREQUENCY,MODEL_ARCH
import csv, os
import pandas as pd
import matplotlib.pyplot as plt

def model_accuracy(model_param) -> float:
        accuracy_ret, size = model_eval.train_and_eval(model_param)
        return float(accuracy_ret['mean_recall@1']), size

def model_carbon(hw_param) -> float:
        hf_pretrained = model_constants.orig_models[MODEL_ARCH]["hf-model"]
        model = ["CLIP"]
        phaze_seq_len = 77
        force_reextract_model = False
        force_reextract_estimates = True
        hbm_size = 1

        # model, phaze_seq_len, force_reextract_model, hbm_size, hw_param, model_param
        carbon, latency, area, energy = main.estimate_carbon(model, phaze_seq_len, force_reextract_model, force_reextract_estimates, hbm_size, hw_param, None, hf_pretrained)
        return carbon, latency, area, energy

def calc_tops(hw_param) -> float:
    return (hw_param["num_tc"] * hw_param["width"] * hw_param["depth"] * 2 * FREQUENCY) / (1000**4)

# Evaluation Function
def evaluate(trial, parameters, csv_file_name):

    hw_config = {}
    hw_config["num_tc"] = parameters.get("cluster_num")
    hw_config["num_vc"] = parameters.get("cluster_num")

    hw_config["width"] = parameters.get("width")
    hw_config["depth"] = parameters.get("depth")
    hw_config["width_vc"] = parameters.get("width")
    hw_config["GLB_Buffer"] = parameters.get("glb_buffer_MB")*1024 *1024
    hw_config["L2_Buffer"] = parameters.get("l2_sram_choices_KB")*1024
    hw_config["L2_BW"] = parameters.get("l2_bw")

    carbon, latency, area, energy = model_carbon(hw_config)
    tops = calc_tops(hw_config)

    with open(csv_file_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([trial, carbon, latency, parameters, area, energy, tops])

    return {"carbon": (carbon, 0.0), "area": (area, 0.0), "latency": (latency,0.0), "energy": (energy, 0.0), "tops": (tops, 0.0)}

def optimize(run_name):
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
                "name": f"cluster_num",
                "type": "choice",
                "values": HW_PARAMS['CLUSTER_NUM'],
                "is_ordered": True,
            }, 
            {
                "name": f"width",
                "type": "choice",
                "values": HW_PARAMS['WIDTH'],
                "is_ordered": True,
            }, 
            {
                "name": f"depth",
                "type": "choice",
                "values": HW_PARAMS['DEPTH'],
                "is_ordered": True,
            }, 
            {
                "name": f"l2_sram_choices_KB",
                "type": "choice",
                "values": HW_PARAMS['L2_SRAM'],
                "is_ordered": True,
            }, 
            {
                "name": f"l2_bw",
                "type": "choice",
                "values": HW_PARAMS['L2_BW'],
                "is_ordered": True,
            }, 
            {
                "name": f"glb_buffer_MB",
                "type": "choice",
                "values": HW_PARAMS['GLB_BUFFER'],
                "is_ordered": True,
            }
        ],
        objectives={
            # `threshold` arguments are optional
            "latency": ObjectiveProperties(minimize=True,threshold=0.1),
            "carbon": ObjectiveProperties(minimize=True,threshold=1.0),
        },
        outcome_constraints=[
            AREA_CONSTRAINT,
            MAX_TOPS_CONSTRAINT,
        ],
        tracking_metric_names=[
            "area",
            "energy",
            "tops"
        ],
        overwrite_existing_experiment=True,
        choose_generation_strategy_kwargs={"should_deduplicate": True},
    )

    # ### Run Optimization

    csv_file_name = f"{directory}/{run_name}.csv"

    with open(csv_file_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Trial Number", "Carbon", "Latency", "Parameters", "Area", "Energy", "TOPs"])

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

