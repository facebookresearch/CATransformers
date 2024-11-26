import torch
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

# Plotting imports and initialization
# from ax.utils.notebook.plotting import init_notebook_plotting, render
from botorch.test_functions.multi_objective import BraninCurrin
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.plot.base import AxPlotConfig, AxPlotTypes

from eval import model_eval, model_constants
from phaze import main
import csv

def model_accuracy(model_param) -> float:
        text_layer, text_embedding_dim, text_ffn_dim, text_head_num, vision_layer, vision_embedding_dim, vision_ffn_dim, vision_head_num = model_param
        accuracy_ret, accuracy_zsc, size = model_eval.train_and_eval(model_param)
        return float(accuracy_ret['mean_recall@1']), size

def model_carbon(model_param, hw_param) -> float:
        model = ["CLIP"]
        phaze_seq_len = 77
        force_reextract_model = False
        force_reextract_estimates = True
        hbm_size = 1

        # model, phaze_seq_len, force_reextract_model, hbm_size, hw_param, model_param
        carbon, latency, area = main.estimate_carbon(model, phaze_seq_len, force_reextract_model, force_reextract_estimates, hbm_size, hw_param, model_param)
        return carbon, latency, area

ax_client = AxClient()
# ChoiceParameterConfig.is_ordered=True 
ax_client.create_experiment(
    name="moo_experiment",
    parameters=[
        {
            "name": f"num_hidden_layers",
            "type": "range",
            "bounds": [6, 12],
            "value_type": "int"
        }, 
        {
            "name": f"intermediate_size",
            "type": "range",
            "bounds": [4, 8],
            "value_type": "int"
        }, 
        {
            "name": f"hidden_size",
            "type": "range",
            "bounds": [256, 512],
            "value_type": "int"
        }, 
        {
            "name": f"num_attn_heads",
            "type": "range",
            "bounds": [4, 8],
            "value_type": "int"
        }, 
        {
            "name": f"vision_num_hidden_layers",
            "type": "range",
            "bounds": [6, 12],
            "value_type": "int"
        }, 
        {
            "name": f"vision_intermediate_size",
            "type": "range",
            "bounds": [4, 8],
            "value_type": "int"
        }, 
        {
            "name": f"vision_hidden_size",
            "type": "range",
            "bounds": [384, 768],
            "value_type": "int"
        }, 
        {
            "name": f"vision_num_attn_heads",
            "type": "range",
            "bounds": [6, 12],
            "value_type": "int"
        },
        {
            "name": f"cluster_num",
            "type": "choice",
            "values": [1, 2, 4],
            "is_ordered": True,
        }, 
        {
            "name": f"width",
            "type": "choice",
            "values": [32, 64, 128, 256],
            "is_ordered": True,
        }, 
        {
            "name": f"depth",
            "type": "choice",
            "values": [2, 4, 8, 16, 32, 64],
            "is_ordered": True,
        }, 
        {
            "name": f"l2_sram_choices_KB",
            "type": "choice",
            "values": [64, 128, 256, 512, 1024],
            "is_ordered": True,
        }, 
        {
            "name": f"l2_bw",
            "type": "choice",
            "values": [32, 64, 128],
            "is_ordered": True,
        }, 
        {
            "name": f"glb_buffer_MB",
            "type": "choice",
            "values": [2, 4, 8],
            "is_ordered": True,
        }
    ],
    objectives={
        # `threshold` arguments are optional
        "a": ObjectiveProperties(minimize=False),
        "b": ObjectiveProperties(minimize=True,),
    },
    outcome_constraints=[
         "area <= 38256017.69369601",
    ],
    tracking_metric_names=[
         "area",
    ],
    overwrite_existing_experiment=True,
    is_test=True,
)


# ### Create an Evaluation Function
import ast
def evaluate(trial, parameters):

    hw_config = {}
    hw_config["num_tc"] = parameters.get("cluster_num")
    hw_config["num_vc"] = parameters.get("cluster_num")

    hw_config["width"] = parameters.get("width")
    hw_config["depth"] = parameters.get("depth")
    hw_config["width_vc"] = parameters.get("width")
    hw_config["GLB_Buffer"] = parameters.get("glb_buffer_MB")*1024 *1024
    hw_config["L2_Buffer"] = parameters.get("l2_sram_choices_KB")*1024
    hw_config["L2_BW"] = parameters.get("l2_bw")
    
    num_ffn_blocks = 8
    text_block_size = model_constants.orig_models["ViT-B-16"]["text_ffn_dim"] / num_ffn_blocks
    vision_block_size = model_constants.orig_models["ViT-B-16"]["vision_ffn_dim"] / num_ffn_blocks
    
    model_config = {}
    model_config["num_hidden_layers"] = parameters.get("num_hidden_layers")
    model_config["intermediate_size"] = int(parameters.get("intermediate_size") * text_block_size)
    model_config["hidden_size"] = parameters.get("hidden_size")
    model_config["num_attn_heads"] = parameters.get("num_attn_heads")
    model_config["vision_num_hidden_layers"] = parameters.get("vision_num_hidden_layers")
    model_config["vision_intermediate_size"] = int(parameters.get("vision_intermediate_size")* vision_block_size)
    model_config["vision_hidden_size"] = parameters.get("vision_hidden_size")
    model_config["vision_num_attn_heads"] = parameters.get("vision_num_attn_heads")

    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    # accuracy = model_accuracy(model_config)
    # carbon, latency = model_carbon(model_config, hw_config)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_accuracy = executor.submit(model_accuracy, model_config)
        future_carbon = executor.submit(model_carbon, model_config, hw_config)
        accuracy, size = future_accuracy.result()
        carbon, latency, area = future_carbon.result()
    

    with open("optimize_carbon_2.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([trial, accuracy, carbon, latency, parameters, size, area])

    return {"a": (accuracy, 0.0), "b": (carbon, 0.0), "area": (area, 0.0)}


# ### Run Optimization

with open("optimize_carbon_2.csv", "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Trial Number", "Accuracy", "Carbon", "Latency", "Parameters", "Model size", "Area"])

for i in range(100):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(i,parameters))


# ### Plot Pareto Frontier

# render(plot_pareto_frontier(frontier, CI_level=0.90))
def render(plot_config: AxPlotConfig, inject_helpers: bool = False) -> None:
    # ...
    if plot_config.plot_type == AxPlotTypes.GENERIC:
        import matplotlib.pyplot as plt
        data = plot_config.data
        # Extract the x and y values
        acc_values = data['data'][0]['x']
        carbon_values = data['data'][0]['y']
        # Create the plot
        plt.plot(carbon_values, acc_values, marker='o')
        plt.xlabel('Carbon')
        plt.ylabel('Accuracy')
        plt.title('Pareto Frontier')
        plt.savefig('plot_carbon_2.png')
            
# In[6]:


objectives = ax_client.experiment.optimization_config.objective.objectives
import pandas as pd
# Create a sample DataFrame
df = exp_to_df(ax_client.experiment)
# Save the DataFrame to a CSV file
df.to_csv('data_carbon_2.csv', index=False)

frontier = compute_posterior_pareto_frontier(
    experiment=ax_client.experiment,
    data=ax_client.experiment.fetch_data(),
    primary_objective=objectives[1].metric,
    secondary_objective=objectives[0].metric,
    absolute_metrics=["a", "b"],
    num_points=20,
)
ax_client.save_to_json_file(filepath='carbon_2.json')

render(plot_pareto_frontier(frontier, CI_level=0.90))
