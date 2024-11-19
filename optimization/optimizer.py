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

from eval import model_eval
from phaze import main


# metric 1
def model_accuracy(model_param) -> float:
        # text_layer, text_embedding_dim, text_ffn_dim, text_head_num, vision_layer, vision_embedding_dim, vision_ffn_dim, vision_head_num = model_param
        accuracy_ret, accuracy_zsc, size = model_eval.train_and_eval(model_param)
        return float(accuracy_ret['mean_recall@1'])

# metric 2
def model_carbon(model_param, hw_param) -> float:
        model = ["CLIP"]
        phaze_seq_len = 77
        force_reextract_model = False
        force_reextract_estimates = True
        hbm_size = 1

        # model, phaze_seq_len, force_reextract_model, hbm_size, hw_param, model_param
        carbon, latency = main.estimate_carbon(model, phaze_seq_len, force_reextract_model, force_reextract_estimates, hbm_size, hw_param, model_param)
        return carbon

ax_client = AxClient()
ax_client.create_experiment(
    name="moo_experiment",
    parameters=[
        {
            "name": f"num_hidden_layers",
            "type": "choice",
            "values": [0.5, 0.625, 0.75, 0.875, 1.0],
        }, 
        {
            "name": f"intermediate_size",
            "type": "choice",
            "values": [0.5, 0.625, 0.75, 0.875, 1.0],
        }, 
        {
            "name": f"hidden_size",
            "type": "choice",
            "values": [0.5, 0.625, 0.75, 0.875, 1.0],
        }, 
        {
            "name": f"num_attn_heads",
            "type": "choice",
            "values": [0.5, 0.625, 0.75, 0.875, 1.0],
        }, 
        {
            "name": f"vision_num_hidden_layers",
            "type": "choice",
            "values": [0.5, 0.625, 0.75, 0.875, 1.0],
        }, 
        {
            "name": f"vision_intermediate_size",
            "type": "choice",
            "values": [0.5, 0.625, 0.75, 0.875, 1.0],
        }, 
        {
            "name": f"vision_hidden_size",
            "type": "choice",
            "values": [0.5, 0.625, 0.75, 0.875, 1.0],
        }, 
        {
            "name": f"vision_num_attn_heads",
            "type": "choice",
            "values": [0.5, 0.625, 0.75, 0.875, 1.0],
        },
        {
            "name": f"cluster_num",
            "type": "choice",
            "values": [1, 2, 4],
        }, 
        {
            "name": f"dimension",
            "type": "choice",
            "values": ["(32, 32)", "(256, 2)", "(64, 16)", "(64, 32)", "(64, 64)", "(128, 8)", "(128,16)", "(128, 32)", "(256, 4)", "(256, 8)", "(256, 16)"],
        }, 
        {
            "name": f"l2_sram_choices_KB",
            "type": "choice",
            "values": [64, 128, 512, 1024],
        }, 
        {
            "name": f"l2_bw",
            "type": "choice",
            "values": [64, 128],
        }, 
        {
            "name": f"glb_buffer_MB",
            "type": "choice",
            "values": [2, 4, 8],
        }
    ],
    objectives={
        # `threshold` arguments are optional
        "a": ObjectiveProperties(minimize=False),
        "b": ObjectiveProperties(minimize=True,),
    },
    overwrite_existing_experiment=True,
    is_test=True,
)


# ### Create an Evaluation Function
import ast
def evaluate(parameters):

    # create HW architecture configuration from parameters
    hw_config = {}
    hw_config["num_tc"] = parameters.get("cluster_num")
    hw_config["num_vc"] = parameters.get("cluster_num")
    width, depth = ast.literal_eval(parameters.get("dimension"))
    hw_config["width"] = width
    hw_config["depth"] = depth
    hw_config["width_vc"] = width
    hw_config["GLB_Buffer"] = parameters.get("glb_buffer_MB")*1024 *1024
    hw_config["L2_Buffer"] = parameters.get("l2_sram_choices_KB")*1024
    hw_config["L2_BW"] = parameters.get("l2_bw")
    
    # create model architecture configuration from parameters
    model_config = {}
    model_config["num_hidden_layers"] = parameters.get("num_hidden_layers")
    model_config["intermediate_size"] = parameters.get("intermediate_size")
    model_config["hidden_size"] = parameters.get("hidden_size")
    model_config["num_attn_heads"] = parameters.get("num_attn_heads")
    model_config["vision_num_hidden_layers"] = parameters.get("vision_num_hidden_layers")
    model_config["vision_intermediate_size"] = parameters.get("vision_intermediate_size")
    model_config["vision_hidden_size"] = parameters.get("vision_hidden_size")
    model_config["vision_num_attn_heads"] = parameters.get("vision_num_attn_heads")

    # unpruned model configurations for transforming inputs for phaze estimator
    # (scale model params from percentage to actual model dimensions)
    unpruned_config = {}
    unpruned_config["num_hidden_layers"] = 12
    unpruned_config["intermediate_size"] = 2048
    unpruned_config["hidden_size"] = 512
    unpruned_config["num_attn_heads"] = 8
    unpruned_config["vision_num_hidden_layers"] = 12
    unpruned_config["vision_intermediate_size"] = 3072
    unpruned_config["vision_hidden_size"] = 768
    unpruned_config["vision_num_attn_heads"] = 12

    # Transform model architecture for Phaze input
    model_config_estimation = {}
    for key in model_config:
        if key in unpruned_config:
            model_config_estimation[key] = int(model_config[key] * unpruned_config[key])
    print(model_config_estimation)

    accuracy = model_accuracy(model_config)
    carbon = model_carbon(model_config_estimation, hw_config)

    return {"a": (accuracy, 0.0), "b": (carbon, 0.0)}


# ### Run Optimization

for i in range(100):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


# ### Plot Pareto Frontier

# render(plot_pareto_frontier(frontier, CI_level=0.90))
def render(plot_config: AxPlotConfig, inject_helpers: bool = False) -> None:
    # ...
    if plot_config.plot_type == AxPlotTypes.GENERIC:
        import matplotlib.pyplot as plt
        data = plot_config.data
        # Extract the x and y values
        x_values = data['data'][0]['x']
        y_values = data['data'][0]['y']
        # Create the plot
        plt.plot(x_values, y_values, marker='o')
        plt.xlabel('Accyracy')
        plt.ylabel('Carbon')
        plt.title('Pareto Frontier')
        plt.savefig('plot.png')
            


objectives = ax_client.experiment.optimization_config.objective.objectives
import pandas as pd
# Export experiment results as csv
df = exp_to_df(ax_client.experiment)
df.to_csv('data.csv', index=False)

frontier = compute_posterior_pareto_frontier(
    experiment=ax_client.experiment,
    data=ax_client.experiment.fetch_data(),
    primary_objective=objectives[1].metric,
    secondary_objective=objectives[0].metric,
    absolute_metrics=["a", "b"],
    num_points=20,
)
ax_client.save_to_json_file()

render(plot_pareto_frontier(frontier, CI_level=0.90))
