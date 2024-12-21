# import torch
# from ax.plot.pareto_frontier import plot_pareto_frontier
# from ax.plot.pareto_utils import compute_posterior_pareto_frontier
# from ax.service.ax_client import AxClient
# from ax.service.utils.instantiation import ObjectiveProperties

# # Plotting imports and initialization
# # from ax.utils.notebook.plotting import init_notebook_plotting, render
# from botorch.test_functions.multi_objective import BraninCurrin
# from ax.metrics.noisy_function import NoisyFunctionMetric
# from ax.service.utils.report_utils import exp_to_df
# from ax.plot.base import AxPlotConfig, AxPlotTypes
# from ax.modelbridge.cross_validation import cross_validate
# from ax.modelbridge.registry import Models
# from ax.utils.notebook.plotting import init_notebook_plotting, render

# import csv
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# ax_client = AxClient.load_from_json_file("/private/home/irenewang/HWNAS/latency_1.json")

# experiment = ax_client.experiment
# data = experiment.fetch_data()
# model = Models.BOTORCH_MODULAR(experiment=experiment, data=data)
# cv_results = cross_validate(model)

# data = {}
# data["a_y"] = []
# data["a_yh"] = []
# data["b_y"] = []
# data["b_yh"] = []

# data["lat_y"] = []
# data["lat_yh"] = []
# data["area_y"] = []
# data["area_yh"] = []

# for rid, cv_result in enumerate(cv_results):
#     arm_name = cv_result.observed.arm_name
#     arm_data = {
#         "name": cv_result.observed.arm_name,
#         "y": {},
#         "se": {},
#         "parameters": cv_result.observed.features.parameters,
#         "y_hat": {},
#         "se_hat": {},
#         "context_stratum": None,
#     }
#     for i, mname in enumerate(cv_result.observed.data.metric_names):
#         arm_data["y"][mname] = cv_result.observed.data.means[i]
#         arm_data["se"][mname] = np.sqrt(cv_result.observed.data.covariance[i][i])
#     for i, mname in enumerate(cv_result.predicted.metric_names):
#         arm_data["y_hat"][mname] = cv_result.predicted.means[i]
#         arm_data["se_hat"][mname] = np.sqrt(cv_result.predicted.covariance[i][i])
#     data["a_y"].append(arm_data["y"]['a'])
#     data["a_yh"].append(arm_data["y_hat"]['a'])
#     data["b_y"].append(arm_data["y"]['b'])
#     data["b_yh"].append(arm_data["y_hat"]['b'])

#     data["lat_y"].append(arm_data["y"]['latency'])
#     data["lat_yh"].append(arm_data["y_hat"]['latency'])
#     data["area_y"].append(arm_data["y"]['area'])
#     data["area_yh"].append(arm_data["y_hat"]['area'])

# # Plot the observed vs predicted values
# plt.scatter(x=data["a_y"], y=data["a_yh"],label="a")
# plt.xlabel("Actual Outcome")
# plt.ylabel("Predicted Outcome")
# plt.title("Accuracy Validation")
# plt.savefig("accuracy_validation_2.png")
# plt.cla()

# # plot validation for carbon
# plt.scatter(x=data["b_y"], y=data["b_yh"],label="b")
# plt.xlabel("Actual Outcome")
# plt.ylabel("Predicted Outcome")
# plt.title("Carbon Validation")
# plt.savefig("carbon_validation_2.png")
# plt.cla()


# # plot validation for carbon
# plt.scatter(x=data["lat_y"], y=data["lat_yh"],label="latency")
# plt.xlabel("Actual Outcome")
# plt.ylabel("Predicted Outcome")
# plt.title("Latency Validation")
# plt.savefig("latency_validation_2.png")
# plt.cla()

# # plot validation for carbon
# plt.scatter(x=data["area_y"], y=data["area_yh"],label="area")
# plt.xlabel("Actual Outcome")
# plt.ylabel("Predicted Outcome")
# plt.title("Area Validation")
# plt.savefig("area_validation_2.png")
# plt.cla()
# def render(plot_config: AxPlotConfig, inject_helpers: bool = False) -> None:
#     # ...
#     if plot_config.plot_type == AxPlotTypes.GENERIC:
#         data = plot_config.data
#         # Extract the x and y values
#         acc_values = data['data'][0]['x']
#         carbon_values = data['data'][0]['y']
#         # Create the plot
#         plt.scatter(carbon_values, acc_values, marker='o')
#         plt.xlabel('Carbon')
#         plt.ylabel('Accuracy')
#         plt.title('Pareto Frontier')
#         plt.savefig('plot_pareto_lat_1.png')     


# # objectives = ax_client.experiment.optimization_config.objective.objectives

# # frontier = compute_posterior_pareto_frontier(
# #     experiment=ax_client.experiment,
# #     data=ax_client.experiment.fetch_data(),
# #     primary_objective=objectives[1].metric,
# #     secondary_objective=objectives[0].metric,
# #     absolute_metrics=["a", "b"],
# #     num_points=20,
# # )
# # render(plot_pareto_frontier(frontier, CI_level=0.90))


import torch
# from ax.plot.pareto_frontier import plot_pareto_frontier
# from ax.plot.pareto_utils import compute_posterior_pareto_frontier
# from ax.service.ax_client import AxClient
# from ax.service.utils.instantiation import ObjectiveProperties

# # Plotting imports and initialization
# # from ax.utils.notebook.plotting import init_notebook_plotting, render
# from botorch.test_functions.multi_objective import BraninCurrin
# from ax.metrics.noisy_function import NoisyFunctionMetric
# from ax.service.utils.report_utils import exp_to_df
# from ax.plot.base import AxPlotConfig, AxPlotTypes
# from ax.modelbridge.cross_validation import cross_validate
# from ax.modelbridge.registry import Models
# from ax.utils.notebook.plotting import init_notebook_plotting, render
# from ax.plot.diagnostic import interact_cross_validation

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ax_client = AxClient.load_from_json_file("/private/home/irenewang/HWNAS/latency_7.json")

# experiment = ax_client.experiment
# data = experiment.fetch_data()
# model = Models.BOTORCH_MODULAR(experiment=experiment, data=data)
# cv_results = cross_validate(model)

# render(interact_cross_validation(cv_results))

# data = {}
# data["a_y"] = []
# data["a_yh"] = []
# data["b_y"] = []
# data["b_yh"] = []

# data["lat_y"] = []
# data["lat_yh"] = []
# data["area_y"] = []
# data["area_yh"] = []

# for rid, cv_result in enumerate(cv_results):
#     arm_name = cv_result.observed.arm_name
#     arm_data = {
#         "name": cv_result.observed.arm_name,
#         "y": {},
#         "se": {},
#         "parameters": cv_result.observed.features.parameters,
#         "y_hat": {},
#         "se_hat": {},
#         "context_stratum": None,
#     }
#     for i, mname in enumerate(cv_result.observed.data.metric_names):
#         arm_data["y"][mname] = cv_result.observed.data.means[i]
#         arm_data["se"][mname] = np.sqrt(cv_result.observed.data.covariance[i][i])
#     for i, mname in enumerate(cv_result.predicted.metric_names):
#         arm_data["y_hat"][mname] = cv_result.predicted.means[i]
#         arm_data["se_hat"][mname] = np.sqrt(cv_result.predicted.covariance[i][i])
#     data["a_y"].append(arm_data["y"]['a'])
#     data["a_yh"].append(arm_data["y_hat"]['a'])
#     data["b_y"].append(arm_data["y"]['b'])
#     data["b_yh"].append(arm_data["y_hat"]['b'])

#     data["lat_y"].append(arm_data["y"]['latency'])
#     data["lat_yh"].append(arm_data["y_hat"]['latency'])
#     data["area_y"].append(arm_data["y"]['area'])
#     data["area_yh"].append(arm_data["y_hat"]['area'])

# # Plot the observed vs predicted values
# plt.scatter(x=data["a_y"], y=data["a_yh"],label="a")
# plt.xlabel("Actual Outcome")
# plt.ylabel("Predicted Outcome")
# plt.title("Accuracy Validation")
# plt.savefig("accuracy_validation_2.png")
# plt.cla()

# # plot validation for carbon
# plt.scatter(x=data["b_y"], y=data["b_yh"],label="b")
# plt.xlabel("Actual Outcome")
# plt.ylabel("Predicted Outcome")
# plt.title("Carbon Validation")
# plt.savefig("carbon_validation_2.png")
# plt.cla()


# # plot validation for carbon
# plt.scatter(x=data["lat_y"], y=data["lat_yh"],label="latency")
# plt.xlabel("Actual Outcome")
# plt.ylabel("Predicted Outcome")
# plt.title("Latency Validation")
# plt.savefig("latency_validation_2.png")
# plt.cla()

# # plot validation for carbon
# plt.scatter(x=data["area_y"], y=data["area_yh"],label="area")
# plt.xlabel("Actual Outcome")
# plt.ylabel("Predicted Outcome")
# plt.title("Area Validation")
# plt.savefig("area_validation_2.png")
# plt.cla()


# objectives = ax_client.experiment.optimization_config.objective.objectives

# frontier = compute_posterior_pareto_frontier(
#     experiment=ax_client.experiment,
#     data=ax_client.experiment.fetch_data(),
#     primary_objective=objectives[0].metric,
#     secondary_objective=objectives[1].metric,
#     absolute_metrics=["accuracy", "latency"],
#     num_points=20,
# )
# render(plot_pareto_frontier(frontier, CI_level=0.90))


# def my_own_render(plot_config: AxPlotConfig, inject_helpers: bool = False) -> None:
#     # ...
#     if plot_config.plot_type == AxPlotTypes.GENERIC:
#         import matplotlib.pyplot as plt
#         data = plot_config.data
#         # Extract the x and y values
#         acc_values = data['data'][0]['x']
#         carbon_values = data['data'][0]['y']
#         print(acc_values)
#         print(carbon_values)
#         # Create the plot
#         plt.scatter(carbon_values, acc_values, marker='o')
#         plt.xlabel('Carbon')
#         plt.ylabel('Accuracy')
#         plt.title('Pareto Frontier')
#         plt.savefig('plot_latency_7.png')

# my_own_render(plot_pareto_frontier(frontier, CI_level=0.90))



import pandas as pd
import numpy as np
def pareto_frontier_latency(area_constraint, save_name):
    directory = "/private/home/irenewang/HWNAS/results/dec_10_results"
    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    filtered_df = df[df['Area'] <= area_constraint]
    # Get accuracy and latency values
    accuracy = filtered_df['Accuracy'].values
    latency = filtered_df['Latency'].values
    carbon = filtered_df['Carbon'].values
    # Compute Pareto frontier
    indices = []
    for i in range(len(accuracy)):
        is_dominated = False
        for j in range(len(accuracy)):
            if i != j and accuracy[j] == accuracy[i] and latency[j] == latency[i] and carbon[j] ==carbon[i]:
                if i > j:
                    # repeated point, already looked at
                    is_dominated = True
                    break
            else:
                if i != j and accuracy[j] >= accuracy[i] and latency[j] <= latency[i]:
                    is_dominated = True
                    break
        if not is_dominated:
            indices.append(i)
    # Get points on the Pareto frontier
    frontier_points = filtered_df.iloc[indices]
    frontier_points.to_csv(f"{directory}/{save_name}/{save_name}_curve.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.scatter(frontier_points['Latency'], frontier_points['Accuracy'], c='red', label='Pareto frontier')
    plt.xlabel('Latency')
    plt.ylabel('Accuracy')
    plt.title(f'Pareto Frontier with Area Constraint {area_constraint}')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_lat.png")

    plt.cla()
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], c=frontier_points['Latency'], cmap='viridis', vmin=0, vmax=0.1,label='Pareto frontier')
    cbar = plt.colorbar()
    cbar.set_label('Latency')
    plt.xlabel('Carbon')
    plt.ylabel('Accuracy')
    plt.xlim(0.3, 0.8)
    area_mm = "{:.1f}".format(area_constraint/1000000)
    plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_carbon.png")

def pareto_frontier_carbon(area_constraint, latency_constraint, save_name):
# Filter data points by area constraint
    directory = "/private/home/irenewang/HWNAS/results/dec_10_results"
    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    filtered_df = df[df['Area'] <= area_constraint]
    filtered_df = filtered_df[filtered_df['Latency'] <= latency_constraint]
    # Get accuracy and carbon values
    accuracy = filtered_df['Accuracy'].values
    carbon = filtered_df['Carbon'].values
    # Compute Pareto frontier
    indices = []
    for i in range(len(accuracy)):
        is_dominated = False
        for j in range(len(accuracy)):
            if i != j and accuracy[j] >= accuracy[i] and carbon[j] <= carbon[i]:
                is_dominated = True
                break
        if not is_dominated:
            indices.append(i)
    # Get points on the Pareto frontier
    frontier_points = filtered_df.iloc[indices]
    frontier_points.to_csv(f"{directory}/{save_name}/{save_name}_curve.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], c=frontier_points['Latency'], cmap='viridis', vmin=0, vmax=0.1,label='Pareto frontier')
    cbar = plt.colorbar()
    cbar.set_label('Latency')
    plt.xlabel('Carbon')
    plt.ylabel('Accuracy')
    plt.xlim(0.3, 0.8)
    area_mm = "{:.1f}".format(area_constraint/1000000)
    plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2, latency Constraint {latency_constraint} s')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_new.png")

def pareto_frontier_all(area_constraint, save_name):
    directory = "/private/home/irenewang/HWNAS/results/dec_10_results"
    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    filtered_df = df[df['Area'] <= area_constraint]
    # Get accuracy, latency, and carbon values
    accuracy = filtered_df['Accuracy'].values
    latency = filtered_df['Latency'].values
    carbon = filtered_df['Carbon'].values
    # Compute Pareto frontier
    indices = []
    for i in range(len(accuracy)):
        is_dominated = False
        for j in range(len(accuracy)):
            if i != j and accuracy[j] >= accuracy[i] and latency[j] <= latency[i] and carbon[j] <= carbon[i]:
                is_dominated = True
                break
        if not is_dominated:
            indices.append(i)
    # Get points on the Pareto frontier
    frontier_points = filtered_df.iloc[indices]
    frontier_points.to_csv(f"{directory}/{save_name}/{save_name}_curve.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.scatter(frontier_points['Latency'], frontier_points['Accuracy'], c='red', label='Pareto frontier')
    plt.xlabel('Latency')
    plt.ylabel('Accuracy')
    plt.title(f'Pareto Frontier with Area Constraint {area_constraint}')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_lat.png")

    plt.cla()
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], c=frontier_points['Latency'], cmap='viridis',  vmin=0, vmax=0.1, label='Pareto frontier')
    cbar = plt.colorbar()
    cbar.set_label('Latency')
    plt.xlabel('Catbon')
    plt.ylabel('Accuracy')
    plt.xlim(0.3, 0.8)
    area_mm = "{:.1f}".format(area_constraint/1000000)
    plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_carbon.png")


def pareto_frontier_overlap(area_constraint, plot_names, label_names):
# Filter data points by area constraint
    directory = "/private/home/irenewang/HWNAS/results/dec_10_results"
    plt.figure(figsize=(8, 6))
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(plot_names)}
    for i, save_name in enumerate(plot_names):
        frontier_points = pd.read_csv(f"{directory}/{save_name}/{save_name}_curve.csv")
        plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], color=colors[save_name], label=label_names[i])
        plt.legend()
        plt.xlabel('Carbon')
        plt.ylabel('Accuracy')
        area_mm = "{:.1f}".format(area_constraint/1000000)
        plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/area_{area_mm}_compare.png")



# Load CSV file
# Set area constraint (e.g., 100)
area_constraint = 31990347
latency_constraint = 0.1
# latency_constraint = 
# Compute Pareto frontier
latency_constraint = 0.1
# frontier_points = pareto_frontier_carbon(area_constraint, latency_constraint, "carbon_100_comb5_7")
# latency_constraint = 0.05
# frontier_points = pareto_frontier_carbon(area_constraint, latency_constraint, "carbon_50_comb1_2")
# latency_constraint = 0.01
# frontier_points = pareto_frontier_carbon(area_constraint, latency_constraint, "carbon_10_comb1_2_3_4_5")

# frontier_points = pareto_frontier_all(area_constraint, "all_comb1_2_3")
# # frontier_points = pareto_frontier_all(area_constraint, "all_2")
# # frontier_points = pareto_frontier_all(area_constraint, "all_3")

# frontier_points = pareto_frontier_latency(area_constraint, "latency_comb3_4_9")
# frontier_points = pareto_frontier_latency(area_constraint, "latency_2")
# frontier_points = pareto_frontier_latency(area_constraint, "latency_3")
# frontier_points = pareto_frontier_latency(area_constraint, "latency_4")
# frontier_points = pareto_frontier_latency(area_constraint, "latency_5")

plot_names = [ "carbon_100_comb2_4_5_7", "carbon_50_comb1_2","carbon_10_comb1_2_3_4_5"]
label_names = ["carbon-100ms", "carbon-50ms", "carbon-10ms"]
pareto_frontier_overlap(area_constraint, plot_names, label_names)