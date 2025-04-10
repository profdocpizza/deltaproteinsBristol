# read each file in analytical_sec folder
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FLOW_RATE_ML_PER_MIN = 0.8

params_df = pd.read_csv("/home/tadas/code/deltaproteinsBristol/experimental_results/deltaprot_designs_data_with_results.csv")[["Well Position","Name","mass"]]

raw_file_dir = "/home/tadas/code/deltaproteinsBristol/experimental_results/analytical_sec_raw"
output_dir = "/home/tadas/code/deltaproteinsBristol/experimental_results/analytical_sec_plots"

for label in params_df["Well Position"].tolist():
    path = f"{raw_file_dir}/{label}.csv"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        continue

    # Load and prepare data
    df = pd.read_csv(path)
    df['Volume_mL'] = df['min'] * FLOW_RATE_ML_PER_MIN

    # Peak-based cropping
    max_idx = df['Intensity'].idxmax()
    peak_vol = df.loc[max_idx, 'Volume_mL']
    cropped_df = df[(df['Volume_mL'] >= peak_vol - 4) & (df['Volume_mL'] <= peak_vol + 4)]

    # Plot
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=cropped_df, x='Volume_mL', y='Intensity')
    plt.title(f'{label}')
    plt.xlabel('Volume (mL)')
    plt.ylabel('Intensity')
    # use dashed grid
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save plots
    for ext in ["png", "svg"]:
        plt.savefig(os.path.join(output_dir, f"{label}.{ext}"), dpi=300)
    plt.close()

# make one ovverlapping plot for overlap for A9,A19 and B8
labels = ['A9', 'A12', 'B8']

fig = plt.figure(figsize=(5, 4))

for label in labels:
    path = f"{raw_file_dir}/{label}.csv"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        continue

    # Load and prepare data
    df = pd.read_csv(path)
    df['Volume_mL'] = df['min'] * FLOW_RATE_ML_PER_MIN

    # Peak-based cropping
    max_idx = df['Intensity'].idxmax()
    peak_vol = df.loc[max_idx, 'Volume_mL']
    cropped_df = df[(df['Volume_mL'] >= peak_vol - 4) & (df['Volume_mL'] <= peak_vol + 4)]

    # Skip if cropping removes all data
    if cropped_df.empty:
        print(f"No data in cropped region for {label}. Skipping.")
        continue

    # Plot
    sns.lineplot(data=cropped_df, x='Volume_mL', y='Intensity', label=label)

    # # Annotate the peak with mass
    # cropped_peak_idx = cropped_df['Intensity'].idxmax()
    # peak_x = cropped_df.loc[cropped_peak_idx, 'Volume_mL']
    # peak_y = cropped_df.loc[cropped_peak_idx, 'Intensity']
    # mass = params_df[params_df["Well Position"] == label]["mass_w_prefix"].iloc[0]
    # plt.annotate(f"{round(mass)}", xy=(peak_x, peak_y), fontsize=10, ha='center', va='bottom')

# use dashed grid
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.xlabel('Volume (mL)')
plt.ylabel('Intensity')
plt.legend()

# Save plots
for ext in ["png", "svg"]:
    plt.savefig(os.path.join(output_dir, f"overlap.{ext}"), dpi=300)
plt.close()
