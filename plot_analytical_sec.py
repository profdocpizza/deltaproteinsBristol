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

params_df = pd.read_csv(
    "/home/tadas/code/deltaproteinsBristol/experimental_results/deltaprot_designs_data_with_results.csv"
)[["Well Position", "Name", "mass_w_prefix", "orientation_code"]]

raw_file_dir = (
    "/home/tadas/code/deltaproteinsBristol/experimental_results/analytical_sec_raw"
)
output_dir = (
    "/home/tadas/code/deltaproteinsBristol/experimental_results/analytical_sec_plots"
)


def format_orientation(raw):
    return raw.upper().replace("_", ".").replace("X", "x").replace("Y", "y")


params_df["orientation_code"] = params_df["orientation_code"].apply(format_orientation)
well_to_orientation = dict(
    zip(params_df["Well Position"], params_df["orientation_code"])
)


for label in params_df["Well Position"].tolist():
    path = f"{raw_file_dir}/{label}.csv"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        continue
    orientation_code = well_to_orientation.get(label, "")
    # Load and prepare data
    df = pd.read_csv(path)
    df["Volume_mL"] = df["min"] * FLOW_RATE_ML_PER_MIN
    df["Intensity"] = df["Intensity"] / df["Intensity"].max()
    baseline_offset = df[
        "Intensity"
    ].min()  # Compute the minimum intensity as the baseline
    df["Intensity"] = df["Intensity"] - baseline_offset  # Subtract the baseline

    # Peak-based cropping
    max_idx = df["Intensity"].idxmax()
    peak_vol = df.loc[max_idx, "Volume_mL"]
    # cropped_df = df[
    #     (df["Volume_mL"] >= peak_vol - 4) & (df["Volume_mL"] <= peak_vol + 4)
    # ]

    # Plot
    plt.figure(figsize=(3, 3))
    sns.lineplot(data=df, x="Volume_mL", y="Intensity")
    plt.title(f"{orientation_code}")
    plt.xlabel("Volume (mL)")
    # Absorbance at 230 nm
    plt.ylabel("Normalised Absorbance at 230 nm")
    # use dashed grid
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.1)
    # plt.xlim(peak_vol - 4, peak_vol + 4)
    plt.xlim(9, 18)
    plt.gca().yaxis.set_visible(False)
    plt.tight_layout()

    # Save plots
    for ext in ["png", "svg"]:
        plt.savefig(
            os.path.join(
                output_dir, f"{label}_{orientation_code.replace('.', '_')}.{ext}"
            ),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()

for label in params_df["Well Position"].tolist():
    path = f"{raw_file_dir}/{label}.csv"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        continue

    # Grab this well’s orientation
    orientation_code = well_to_orientation.get(label, "")
    # If orientation_code is empty, you could skip or just plot without it.

    # Load and prepare data
    df = pd.read_csv(path)
    df["Volume_mL"] = df["min"] * FLOW_RATE_ML_PER_MIN
    df["Intensity"] = df["Intensity"] / df["Intensity"].max()
    baseline_offset = df["Intensity"].min()
    df["Intensity"] = df["Intensity"] - baseline_offset

    # Peak‐based cropping
    max_idx = df["Intensity"].idxmax()
    peak_vol = df.loc[max_idx, "Volume_mL"]
    cropped_df = df[
        (df["Volume_mL"] >= peak_vol - 4) & (df["Volume_mL"] <= peak_vol + 4)
    ]

    # Plot
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=cropped_df, x="Volume_mL", y="Intensity")
    plt.title(f"{label}   {orientation_code}")  # include orientation in title
    plt.xlabel("Volume (mL)")
    plt.ylabel("Normalised Absorbance at 230 nm")
    plt.gca().yaxis.set_visible(False)
    plt.tight_layout()

    # Save plots using “WellOrientation” as the basename
    safe_orient = orientation_code.replace(".", "_")
    basename = f"{label}_{safe_orient}"
    for ext in ["png", "svg"]:
        plt.savefig(os.path.join(output_dir, f"{basename}.{ext}"), dpi=300)
    plt.close()

# ---------------------------------------------
# 2) Overlapping plot (multiple wells together)
# ---------------------------------------------
labels = ["B12", "F4"]
fig = plt.figure(figsize=(5, 3))

# We'll build a list of "Well_Orientation" pieces for the filename:
filename_pieces = []

for label in labels:
    path = f"{raw_file_dir}/{label}.csv"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        continue

    # Grab and format orientation for legend & filename
    orientation_code = well_to_orientation.get(label, "")
    legend_label = f"{label}   {orientation_code}"

    # Also make a “safe” version of orientation for file (dots → underscores)
    safe_orient = orientation_code.replace(".", "_")
    filename_pieces.append(label)
    filename_pieces.append(safe_orient)

    # Load and prepare data
    df = pd.read_csv(path)
    df = df[df["min"] >= 0.01]
    df["Volume_mL"] = df["min"] * FLOW_RATE_ML_PER_MIN
    df["Intensity"] = df["Intensity"] / df["Intensity"].max()
    baseline_offset = df["Intensity"].min()
    df["Intensity"] = df["Intensity"] - baseline_offset
    df["Intensity"] = df["Intensity"] / df["Intensity"].max()

    # Plot the full curve
    sns.lineplot(
        data=df,
        x="Volume_mL",
        y="Intensity",
        label=legend_label,
        linewidth=1,
    )

    # Annotate peak with mass
    # cropped_peak_idx = df["Intensity"].idxmax()
    # peak_x = df.loc[cropped_peak_idx, "Volume_mL"]
    # peak_y = df.loc[cropped_peak_idx, "Intensity"]
    # mass = params_df.loc[params_df["Well Position"] == label, "mass_w_prefix"].iloc[0]
    # plt.annotate(
    #     f"{round(mass / 1000, 1)}",
    #     xy=(peak_x, peak_y * 1.01),
    #     fontsize=4,
    #     ha="center",
    #     va="bottom",
    #     rotation=90,
    # )

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.xlabel("Volume (mL)")
plt.ylabel("Intensity")
plt.ylim(0, 1.1)
plt.xlim(9, 18)
plt.legend()


combined = "_".join(filename_pieces)
for ext in ["png", "svg"]:
    plt.savefig(
        os.path.join(output_dir, f"overlap_{combined}.{ext}"),
        dpi=300,
        bbox_inches="tight",
    )
plt.close()
