from configurations import AREA_CONSTRAINT_VALUE, MAX_TOPS, MAX_TOPS_CONSTRAINT, FREQUENCY, LATENCY_CONSTRAINT_VALUE
import csv, os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def pareto_frontier_latency(save_name, directory):
    area_constraint = AREA_CONSTRAINT_VALUE
    tops = MAX_TOPS

    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    filtered_df = df[df['Area'] <= area_constraint]
    filtered_df = filtered_df[filtered_df['TOPs'] <= tops]
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
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Latency'], frontier_points['Accuracy'], s=100, c='red', label='Pareto frontier')
    plt.xlabel('Latency (s)')
    plt.ylabel('Accuracy (%)')
    # plt.title(f'Pareto Frontier with Area Constraint {area_constraint}')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_lat.png")

    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,8))
    plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], c=frontier_points['Latency'], s=100, cmap='viridis', vmin=0, vmax=0.05,label='Pareto frontier')
    cbar = plt.colorbar()
    cbar.set_label('Latency (s)')
    plt.xlabel('Carbon (kgCO2e)')
    plt.ylabel('Accuracy (%)')
    plt.xlim(0.3, 0.8)
    plt.ylim(0.0, 0.4)
    area_mm = "{:.1f}".format(area_constraint/1000000)
    # plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_carbon.png")

def pareto_frontier_energy(save_name, directory):
    area_constraint = AREA_CONSTRAINT_VALUE
    tops = MAX_TOPS

    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    filtered_df = df[df['Area'] <= area_constraint]
    filtered_df = filtered_df[filtered_df['TOPs'] <= tops]
    # Get accuracy and latency values
    accuracy = filtered_df['Accuracy'].values
    energy = filtered_df['Energy'].values
    carbon = filtered_df['Carbon'].values
    # Compute Pareto frontier
    indices = []
    for i in range(len(accuracy)):
        is_dominated = False
        for j in range(len(accuracy)):
            if i != j and accuracy[j] == accuracy[i] and energy[j] == energy[i] and carbon[j] ==carbon[i]:
                if i > j:
                    # repeated point, already looked at
                    is_dominated = True
                    break
            else:
                if i != j and accuracy[j] >= accuracy[i] and energy[j] <= energy[i]:
                    is_dominated = True
                    break
        if not is_dominated:
            indices.append(i)
    # Get points on the Pareto frontier
    frontier_points = filtered_df.iloc[indices]
    frontier_points.to_csv(f"{directory}/{save_name}/{save_name}_curve.csv", index=False)
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Energy'], frontier_points['Accuracy'], c='red', label='Pareto frontier')
    plt.xlabel('Energy (KWh)')
    plt.ylabel('Accuracy (%)')
    # plt.title(f'Pareto Frontier with Area Constraint {area_constraint}')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_eng.png")

    plt.cla()
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], c=frontier_points['Latency'], s=100, cmap='viridis', vmin=0, vmax=0.05,label='Pareto frontier')
    cbar = plt.colorbar()
    cbar.set_label('Latency (s)')
    plt.xlabel('Carbon (kgCO2e)')
    plt.ylabel('Accuracy (%)')
    plt.xlim(0.3, 0.8)
    plt.ylim(0.0, 0.4)
    area_mm = "{:.1f}".format(area_constraint/1000000)
    # plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_carbon.png")



def pareto_frontier_carbon(save_name, directory):
    area_constraint = AREA_CONSTRAINT_VALUE
    tops = MAX_TOPS
    latency_constraint = LATENCY_CONSTRAINT_VALUE
# Filter data points by area constraint
    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    filtered_df = df[df['Area'] <= area_constraint]
    filtered_df = filtered_df[filtered_df['Latency'] <= latency_constraint]
    filtered_df = filtered_df[filtered_df['TOPs'] <= tops]
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
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], c=frontier_points['Latency'], s=100, cmap='viridis', vmin=0, vmax=0.05,label='Pareto frontier')
    cbar = plt.colorbar()
    cbar.set_label('Latency (s)')
    plt.xlabel('Carbon (kgCO2e)')
    plt.ylabel('Accuracy (%)')
    plt.xlim(0.3, 0.8)
    plt.ylim(0.0, 0.4)
    area_mm = "{:.1f}".format(area_constraint/1000000)
    # plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2, latency Constraint {latency_constraint} s')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_new.png")

def pareto_frontier_all(save_name, directory):
    area_constraint = AREA_CONSTRAINT_VALUE
    tops = MAX_TOPS

    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    filtered_df = df[df['Area'] <= area_constraint]
    filtered_df = filtered_df[filtered_df['TOPs'] <= tops]
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
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Latency'], frontier_points['Accuracy'], c='red', label='Pareto frontier')
    plt.xlabel('Latency (s)')
    plt.ylabel('Accuracy (%)')
    # plt.title(f'Pareto Frontier with Area Constraint {area_constraint}')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_lat.png")

    plt.cla()
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], c=frontier_points['Latency'], s=100, cmap='viridis',  vmin=0, vmax=0.05, label='Pareto frontier')
    cbar = plt.colorbar()
    cbar.set_label('Latency (s)')
    plt.xlabel('Carbon (kgCO2e)')
    plt.ylabel('Accuracy (%)')
    plt.xlim(0.3, 0.8)
    plt.ylim(0.0, 0.4)
    area_mm = "{:.1f}".format(area_constraint/1000000)
    # plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_carbon.png")



def pareto_frontier_hw(save_name, directory):
    area_constraint = AREA_CONSTRAINT_VALUE
    tops = MAX_TOPS

    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    filtered_df = df[df['Area'] <= area_constraint]
    # Get accuracy, latency, and carbon values
    latency = filtered_df['Latency'].values
    carbon = filtered_df['Carbon'].values
    # Compute Pareto frontier
    indices = []
    for i in range(len(latency)):
        is_dominated = False
        for j in range(len(latency)):
            if i != j and latency[j] <= latency[i] and carbon[j] <= carbon[i]:
                is_dominated = True
                break
        if not is_dominated:
            indices.append(i)
    # Get points on the Pareto frontier
    frontier_points = filtered_df.iloc[indices]
    frontier_points.to_csv(f"{directory}/{save_name}/{save_name}_curve.csv", index=False)
    plt.rcParams['font.size'] = 24
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Carbon'], frontier_points['Latency']*1000, s=100, label='Pareto frontier')
    plt.xlabel('Carbon (kgCO2e)')
    plt.ylabel('Latency (ms)')
    area_mm = "{:.1f}".format(area_constraint/1000000)
    # plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve.png")


def pareto_frontier_overlap(plot_names, label_names):
    area_constraint = AREA_CONSTRAINT_VALUE
    tops = MAX_TOPS
    # Filter data points by area constraint
    plt.figure(figsize=(8, 6.5))
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(plot_names)}
    for i, save_name in enumerate(plot_names):
        frontier_points = pd.read_csv(f"{directory}/{save_name}/{save_name}_curve.csv")
        plt.rcParams['font.size'] = 24
        plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], color=colors[save_name], label=label_names[i])
        plt.xlim(0.3, 0.8)
        plt.ylim(0.0, 0.5)
        plt.legend()
        plt.xlabel('Carbon (kgCO2e)')
        plt.ylabel('Accuracy (%)')
        area_mm = "{:.1f}".format(area_constraint/1000000)
        # plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/area_{area_mm}_compare.png")

def pareto_frontier_energy_model(save_name, directory):
    df = pd.read_csv(f"{directory}/{save_name}/{save_name}.csv")
    # Get accuracy and latency values
    accuracy = df['Accuracy'].values
    energy = df['Energy'].values
    carbon = df['Carbon'].values
    # Compute Pareto frontier
    indices = []
    for i in range(len(accuracy)):
        is_dominated = False
        for j in range(len(accuracy)):
            if i != j and accuracy[j] == accuracy[i] and energy[j] == energy[i] and carbon[j] ==carbon[i]:
                if i > j:
                    # repeated point, already looked at
                    is_dominated = True
                    break
            else:
                if i != j and accuracy[j] >= accuracy[i] and energy[j] <= energy[i]:
                    is_dominated = True
                    break
        if not is_dominated:
            indices.append(i)
    # Get points on the Pareto frontier
    frontier_points = df.iloc[indices]
    frontier_points.to_csv(f"{directory}/{save_name}/{save_name}_curve.csv", index=False)
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Energy'], frontier_points['Accuracy'], c='red', label='Pareto frontier')
    plt.xlabel('Energy (KWh)')
    plt.ylabel('Accuracy (%)')
    # plt.title(f'Pareto Frontier with Area Constraint {area_constraint}')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_eng.png")

    plt.cla()
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(frontier_points['Carbon'], frontier_points['Accuracy'], c=frontier_points['Latency'], s=100, cmap='viridis', vmin=0, vmax=0.05,label='Pareto frontier')
    cbar = plt.colorbar()
    cbar.set_label('Latency (s)')
    plt.xlabel('Carbon (kgCO2e)')
    plt.ylabel('Accuracy (%)')
    plt.xlim(0.3, 0.8)
    plt.ylim(0.0, 0.4)
    # plt.title(f'Pareto Frontier with Area Constraint {area_mm} mm^2')
    plt.savefig(f"{directory}/{save_name}/{save_name}_curve_carbon.png")
