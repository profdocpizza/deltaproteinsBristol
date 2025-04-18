# read each file in analytical_sec folder
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import line
import seaborn as sns

# set colorblind palette
sns.set_palette("colorblind")
FLOW_RATE_ML_PER_MIN = 0.8

params_df = pd.read_csv("/home/tadas/code/deltaproteinsBristol/experimental_results/deltaprot_designs_data_with_results.csv")[["Well Position","Name","mass_w_prefix"]]

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
    df['Intensity'] = df['Intensity'] / df['Intensity'].max()
    baseline_offset = df['Intensity'].min()  # Compute the minimum intensity as the baseline
    df['Intensity'] = df['Intensity'] - baseline_offset  # Subtract the baseline

    # Peak-based cropping
    max_idx = df['Intensity'].idxmax()
    peak_vol = df.loc[max_idx, 'Volume_mL']
    cropped_df = df[(df['Volume_mL'] >= peak_vol - 4) & (df['Volume_mL'] <= peak_vol + 4)]

    # Plot
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=cropped_df, x='Volume_mL', y='Intensity')
    plt.title(f'{label}')
    plt.xlabel('Volume (mL)')
    # Absorbance at 230 nm 
    plt.ylabel('Normalised Absorbance at 230 nm')
    # use dashed grid
    # plt.grid(True, linestyle='--', alpha=0.5)
    # remove y axis but keep the label
    plt.gca().yaxis.set_visible(False)
    plt.tight_layout()

    # Save plots
    for ext in ["png", "svg"]:
        plt.savefig(os.path.join(output_dir, f"{label}.{ext}"), dpi=300)
    plt.close()

# make one ovverlapping plot for overlap for A9,A19 and B8
labels = ['A9', 'A12', 'B8','A1','B12','F4','A11','B3','F4']

fig = plt.figure(figsize=(5, 3))

for label in labels:
    path = f"{raw_file_dir}/{label}.csv"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        continue

    # Load and prepare data
    df = pd.read_csv(path)
    df=df[(df['min'] >= 0.01)]
    df['Volume_mL'] = df['min'] * FLOW_RATE_ML_PER_MIN
    df['Intensity'] = df['Intensity'] / df['Intensity'].max()
    baseline_offset = df['Intensity'].min()  # Compute the minimum intensity as the baseline
    df['Intensity'] = df['Intensity'] - baseline_offset  # Subtract the baseline
    df['Intensity'] = df['Intensity'] / df['Intensity'].max()

    # Peak-based cropping
    max_idx = df['Intensity'].idxmax()
    peak_vol = df.loc[max_idx, 'Volume_mL']
    # Plot
    sns.lineplot(data=df, x='Volume_mL', y='Intensity', label=label,linewidth=1)

    # Annotate the peak with mass
    cropped_peak_idx = df['Intensity'].idxmax()
    peak_x = df.loc[cropped_peak_idx, 'Volume_mL']
    peak_y = df.loc[cropped_peak_idx, 'Intensity']
    mass = params_df[params_df["Well Position"] == label]["mass_w_prefix"].iloc[0]
    # rotate anotations 90 degrees
    plt.annotate(f"{round(mass/1000,1)}", xy=(peak_x, peak_y*1.01), fontsize=4, ha='center', va='bottom', rotation=90)

# use dashed grid
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.xlabel('Volume (mL)')
plt.ylabel('Intensity')
# set y lims
plt.ylim(0,1.1)
plt.xlim(9,18)
plt.legend()

# Save plots
for ext in ["png", "svg"]:
    plt.savefig(os.path.join(output_dir, f"overlap.{ext}"), dpi=300)
plt.close()
