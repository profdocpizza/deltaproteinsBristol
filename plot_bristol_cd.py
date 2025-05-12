# --- Plotting Dependencies ---
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import os

from CD_analysis import parse_jasco_cd_file

# --- Calculation Functions ---
def get_mean_residue_ellipticity(protein_concentration_uM,
                                 path_length_mm,
                                 num_residues):
    """
    Calculates the conversion factor from mdeg to Mean Residue Ellipticity (MRE; deg·cm²·dmol⁻¹)
    using the formula: factor = 1/(10 * (path_length_cm) * (concentration_M) * num_residues).
    """
    if not all([protein_concentration_uM, path_length_mm, num_residues]):
        raise ValueError("Missing one or more parameters for MRE calculation: concentration, path length, num residues")
    path_length_cm = path_length_mm / 10.0  # Convert mm to cm
    concentration_M = protein_concentration_uM * 1e-6  # Convert µM to M
    denominator = (10.0 * path_length_cm * concentration_M * num_residues)
    if denominator == 0:
        raise ValueError("MRE calculation denominator is zero. Check concentration, path length, and num residues.")
    return 1.0 / denominator

def interpolate_and_smooth_data(x_values, y_values, new_x_grid=None, window_length=7, polyorder=2):
    """
    Interpolates and smooths y_values based on x_values.
    Returns the new x grid and the smoothed y values.
    """
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    sort_indices = np.argsort(x_values)
    x_sorted = x_values[sort_indices]
    y_sorted = y_values[sort_indices]

    if new_x_grid is None:
        new_x_grid = np.linspace(x_sorted.min(), x_sorted.max(), num=500)
    else:
        new_x_grid = np.asarray(new_x_grid)
        new_x_grid.sort()

    try:
        interp_func = interp1d(x_sorted, y_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)
        interp_y = interp_func(new_x_grid)
    except ValueError as e:
        print(f"Interpolation failed: {e}. Using linear interpolation instead.")
        interp_func = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_y = interp_func(new_x_grid)

    valid = ~np.isnan(interp_y)
    if not np.any(valid):
        print("Warning: All interpolated values are NaN. Smoothing cannot be applied.")
        return new_x_grid, interp_y

    valid_y = interp_y[valid]
    smoothed_valid = valid_y
    if len(valid_y) > polyorder and len(valid_y) > window_length:
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(valid_y) - (1 if len(valid_y) % 2 == 0 else 0))
        if window_length > polyorder:
            try:
                smoothed_valid = savgol_filter(valid_y, window_length, polyorder)
            except ValueError as e:
                print(f"Warning: Savgol filter failed: {e}. Returning interpolated data.")
        else:
            print(f"Warning: window_length ({window_length}) must be greater than polyorder ({polyorder}). Skipping smoothing.")
    else:
        print(f"Warning: Not enough data points ({len(valid_y)}) for smoothing. Skipping smoothing.")

    smoothed_y = np.full_like(interp_y, np.nan)
    smoothed_y[valid] = smoothed_valid
    return new_x_grid, smoothed_y

# --- Plotting Functions ---
def plot_cd_spectra(cd_data_df, sample_name,
                    protein_concentration_uM,
                    path_length_mm,
                    num_residues,
                    overlay_df=None, overlay_label="Overlay",
                    title="Circular Dichroism Spectrum", save_dir=".",
                    smooth_window=7, smooth_polyorder=2,
                    x_limits=(190, 260), y_limits_mre=None):
    """
    Plots a CD spectrum (wavelength vs MRE) for a primary data set, with an optional overlay.
    """
    os.makedirs(save_dir, exist_ok=True)
    required_cols = ['Wavelength_nm', 'CD_mdeg']
    for df in [cd_data_df] + ([overlay_df] if overlay_df is not None else []):
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")

    mre_factor = get_mean_residue_ellipticity(protein_concentration_uM, path_length_mm, num_residues)
    wavelengths = cd_data_df['Wavelength_nm'].values
    cd_values = cd_data_df['CD_mdeg'].values
    new_wave, smooth_cd = interpolate_and_smooth_data(wavelengths, cd_values, window_length=smooth_window, polyorder=smooth_polyorder)
    smoothed_mre = smooth_cd * mre_factor
    smooth_df = pd.DataFrame({
        'Wavelength_nm': new_wave,
        'Smoothed_MRE_deg_cm2_dmol': smoothed_mre
    })

    overlay_smooth_df = None
    if overlay_df is not None:
        overlay_wave = overlay_df['Wavelength_nm'].values
        overlay_vals = overlay_df['CD_mdeg'].values
        new_wave_overlay, smooth_overlay = interpolate_and_smooth_data(overlay_wave, overlay_vals, new_x_grid=new_wave,
                                                                       window_length=smooth_window, polyorder=smooth_polyorder)
        overlay_smooth_df = pd.DataFrame({
            'Wavelength_nm': new_wave_overlay,
            'Smoothed_MRE_deg_cm2_dmol': smooth_overlay * mre_factor
        })
    colorblind = sns.color_palette("colorblind")
    grey_color = colorblind[7]  # Red-ish
    red_color = colorblind[3]  # Blue-ish

    plt.figure(figsize=(5, 4))
    sns.set_style("ticks")
    sns.lineplot(data=smooth_df.dropna(), x='Wavelength_nm', y='Smoothed_MRE_deg_cm2_dmol',
                 linewidth=1.5, label=f'{sample_name} Pre-Melt', color="black")
    if overlay_smooth_df is not None:
        sns.lineplot(data=overlay_smooth_df.dropna(), x='Wavelength_nm', y='Smoothed_MRE_deg_cm2_dmol',
                     color=red_color, linestyle='--', linewidth=1.5, label=f'{overlay_label}')
    full_title = f"{title}\n{sample_name}" + (f" vs {overlay_label}" if overlay_df is not None else "")
    plt.title(full_title, fontsize=14)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel("Mean Residue Ellipticity (deg·cm²·dmol⁻¹)", fontsize=12)
    plt.xlim(x_limits)
    if y_limits_mre:
        plt.ylim(y_limits_mre)
    plt.axhline(0, color='grey', linestyle='-', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='Sample', loc='best', fontsize=10)
    filename_base = f"cd_spectrum_{sample_name.replace(' ', '_')}"
    if overlay_df is not None:
        filename_base += f"_vs_{overlay_label.replace(' ', '_')}"
    for ext in ["png", "svg"]:
        plt.savefig(os.path.join(save_dir, f"{filename_base}.{ext}"), dpi=300, bbox_inches="tight")
    plt.tight_layout()

def plot_melting_curve(melt_data_df, sample_name, protein_concentration_uM,
                       path_length_mm,
                       num_residues, wavelength_nm=222,
                       reverse_melt_df=None, reverse_label="Cooling",
                       title="Thermal Melting Curve", save_dir=".",
                       y_limits_mre=None, x_limits=None):
    """
    Plots a thermal melting curve (Temperature vs MRE) for heating, and optionally cooling (reverse) data.
    """
    os.makedirs(save_dir, exist_ok=True)
    required_cols = ['Temperature_C', 'CD_mdeg']
    for df in [melt_data_df] + ([reverse_melt_df] if reverse_melt_df is not None else []):
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")

    mre_factor = get_mean_residue_ellipticity(protein_concentration_uM, path_length_mm, num_residues)
    melt_data = melt_data_df.copy()
    melt_data['MRE_deg_cm2_dmol'] = melt_data['CD_mdeg'] * mre_factor
    melt_data = melt_data.sort_values(by='Temperature_C').dropna(subset=['MRE_deg_cm2_dmol'])

    reverse_data = None
    if reverse_melt_df is not None:
        reverse_data = reverse_melt_df.copy()
        reverse_data['MRE_deg_cm2_dmol'] = reverse_data['CD_mdeg'] * mre_factor
        reverse_data = reverse_data.sort_values(by='Temperature_C').dropna(subset=['MRE_deg_cm2_dmol'])

    plt.figure(figsize=(5, 4))
    sns.set_style("ticks")
    # Use seaborn's coolwarm palette
    colorblind = sns.color_palette("colorblind")
    heat_color = colorblind[3]  # Red-ish
    cool_color = colorblind[0]  # Blue-ish

    sns.lineplot(data=melt_data, x='Temperature_C', y='MRE_deg_cm2_dmol',
                 linewidth=1.5, label=f'{sample_name} (Heating)', color=heat_color)
    if reverse_data is not None:
        sns.lineplot(data=reverse_data, x='Temperature_C', y='MRE_deg_cm2_dmol',
                     linewidth=1.5, linestyle='--',
                     label=f'{sample_name} ({reverse_label})',color = cool_color)
    full_title = f"{title} at {wavelength_nm} nm\n{sample_name}"
    plt.title(full_title, fontsize=14)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel("Mean Residue Ellipticity (deg·cm²·dmol⁻¹)", fontsize=12)
    if y_limits_mre:
        plt.ylim(y_limits_mre)
    else:
        all_mre = melt_data['MRE_deg_cm2_dmol']
        if reverse_data is not None:
            all_mre = pd.concat([all_mre, reverse_data['MRE_deg_cm2_dmol']])
        if not all_mre.empty:
            pad = (all_mre.max() - all_mre.min()) * 0.05
            plt.ylim(all_mre.min() - pad, all_mre.max() + pad)
    plt.grid(True, linestyle=':', alpha=0.6)
    # plt.legend(loc='best', fontsize=10)

    # Custom legend with arrows in the line icons
    custom_lines = [
        Line2D([0, 1], [0, 0], color=heat_color, lw=2, linestyle='-', marker='>', markersize=8, markevery=[1]),
        Line2D([0, 1], [0, 0], color=cool_color, lw=2, linestyle='--', marker='<', markersize=8, markevery=[0])
    ]

    plt.legend(custom_lines, ['Heating', 'Cooling'],
            loc='best', fontsize=10)
    plt.tight_layout()
    filename_base = f"melting_curve_{sample_name.replace(' ', '_')}_{wavelength_nm}nm"
    if reverse_data is not None:
        filename_base += "_hysteresis"
    for ext in ["png", "svg"]:
        plt.savefig(os.path.join(save_dir, f"{filename_base}.{ext}"), dpi=300, bbox_inches="tight")

# --- Helper Functions for Processing ---
def subtract_blank_from_sample(parsed_data, key, blank_df):
    """
    Subtracts the blank CD (or melt) data from the sample data if the x-axis (wavelength or temperature) matches.
    Stores the result under a new key with '_Subtracted' appended.
    """
    if key in parsed_data and 'data' in parsed_data[key]:
        sample_df = parsed_data[key]['data']
        if sample_df.columns[0] == blank_df.columns[0] and \
           len(sample_df) == len(blank_df) and \
           np.allclose(sample_df.iloc[:, 0], blank_df.iloc[:, 0]):
            y_col = sample_df.columns[1]
            subtracted = sample_df.copy()
            subtracted[y_col] = sample_df[y_col] - blank_df[y_col]
            parsed_data[f"{key}_Subtracted"] = {'meta': parsed_data[key]['meta'], 'data': subtracted}
        else:
            print(f"Warning: X-axis mismatch or unequal lengths for {key}. Skipping blank subtraction for this file.")
    else:
        print(f"Warning: {key} not found in parsed data.")

def process_sample(sample, params_df, parsed_data, params, blank_df):
    """
    Processes a sample by performing blank subtraction on its pre-melt and post-melt files,
    then plotting both the CD spectrum overlay and the 222 nm melting curve.
    """
    pre_key = sample
    post_key = f"{sample}_post_melt"
    melt_key = f"{sample}_melt"
    remelt_key = f"{sample}_remelt"
    
    # Subtract blank from pre- and post-melt files
    subtract_blank_from_sample(parsed_data, pre_key, blank_df)
    subtract_blank_from_sample(parsed_data, post_key, blank_df)
    
    try:
        conc = params_df.loc[sample, "cd_sample_molar_conc"] * 1e6
        seq_len = params_df.loc[sample, "sequence_length"]
        path_len = params_df.loc[sample, "cd_path_length_mm"]
    except KeyError:
        print(f"Parameters not found for sample {sample}. Skipping.")
        return
    
    pre_sub_key = f"{pre_key}_Subtracted"
    post_sub_key = f"{post_key}_Subtracted"
    
    if pre_sub_key in parsed_data and post_sub_key in parsed_data:
        plot_cd_spectra(parsed_data[pre_sub_key]['data'],
                        sample_name=f"Sample {sample}",
                        overlay_df=parsed_data[post_sub_key]['data'],
                        overlay_label=f"{sample} Post-Melt",
                        protein_concentration_uM=conc,
                        num_residues=seq_len,
                        path_length_mm=path_len,
                        **params)
    else:
        print(f"Missing subtracted data for pre- or post-melt for sample {sample}.")
    
    # Plot the melting curve at 222 nm (heating and optionally cooling)
    if melt_key in parsed_data:
        if remelt_key in parsed_data:
            plot_melting_curve(parsed_data[melt_key]['data'],
                               sample_name=f"Sample {sample}",
                               reverse_melt_df=parsed_data[remelt_key]['data'],
                               reverse_label="Cooling",
                               wavelength_nm=222,
                               protein_concentration_uM=conc,
                               path_length_mm=path_len,
                               num_residues=seq_len,
                               **params)
        else:
            plot_melting_curve(parsed_data[melt_key]['data'],
                               sample_name=f"Sample {sample}",
                               wavelength_nm=222,
                               protein_concentration_uM=conc,
                               path_length_mm=path_len,
                               num_residues=seq_len,
                               **params)
    else:
        print(f"Missing melt data for sample {sample}.")

# --- Main Execution ---
if __name__ == "__main__":
    # Define data directories (adjust these paths as needed)
    data_dir = "/home/tadas/code/deltaproteinsBristol/experimental_results/bristol_cd_raw"
    output_dir = "/home/tadas/code/deltaproteinsBristol/experimental_results/bristol_cd_plots"

    # Load experimental parameters
    params_df = pd.read_csv("/home/tadas/code/deltaproteinsBristol/experimental_results/deltaprot_designs_data_with_results.csv")[
        ["Well Position", "cd_path_length_mm", "cd_sample_molar_conc", "sequence_length"]
    ]
    params_df.set_index("Well Position", inplace=True)
    
    # Common plotting parameters
    params = {
        "y_limits_mre": (-40000, 0),
        "x_limits": (200, 260),
        "save_dir": output_dir
    }
    
    # Define file names for all samples
    files = {
        "Blank": "20250404_1mm_BP-Buffer_blank.txt",
        "A9": "20250404_1mm_BP-Buffer_A9.txt",
        "A9_melt": "20250404_1mm_BP-Buffer_A9_melt.txt",
        "A9_remelt": "20250404_1mm_BP-Buffer_A9_melt@.txt",
        "A9_post_melt": "20250404_1mm_BP-Buffer_A9_post_melt.txt",
        "A12": "20250404_1mm_BP-Buffer_A12.txt",
        "A12_melt": "20250404_1mm_BP-Buffer_A12_melt.txt",
        "A12_remelt": "20250404_1mm_BP-Buffer_A12_melt@.txt",
        "A12_post_melt": "20250404_1mm_BP-Buffer_A12_post_melt.txt",
        "B8": "20250404_1mm_BP-Buffer_B8.txt",
        "B8_melt": "20250404_1mm_BP-Buffer_B8_melt.txt",
        "B8_remelt": "20250404_1mm_BP-Buffer_B8_melt@.txt",
        "B8_post_melt": "20250404_1mm_BP-Buffer_B8_post_melt.txt",
        "A1": "20250404_1mm_BP-Buffer_A1.txt",
        "A1_melt": "20250404_1mm_BP-Buffer_A1_melt.txt",
        "A1_remelt": "20250404_1mm_BP-Buffer_A1_melt@.txt",
        "A1_post_melt": "20250404_1mm_BP-Buffer_A1_post_melt.txt",
        "B12": "20250404_1mm_BP-Buffer_B12.txt",
        "B12_melt": "20250404_1mm_BP-Buffer_B12_melt.txt",
        "B12_remelt": "20250404_1mm_BP-Buffer_B12_melt@.txt",
        "B12_post_melt": "20250404_1mm_BP-Buffer_B12_post_melt.txt",
        "F4": "20250404_1mm_BP-Buffer_F4.txt",
        "F4_melt": "20250404_1mm_BP-Buffer_F4_melt.txt",
        "F4_remelt": "20250404_1mm_BP-Buffer_F4_melt@.txt",
        "F4_post_melt": "20250404_1mm_BP-Buffer_F4_post_melt.txt",
        "A11": "20250416_1mm_BP-Buffer_A11.txt",
        "A11_melt": "20250416_1mm_BP-Buffer_A11_melt.txt",
        "A11_remelt": "20250416_1mm_BP-Buffer_A11_melt@.txt",
        "A11_post_melt": "20250416_1mm_BP-Buffer_A11_post_melt.txt",
        "B3": "20250416_1mm_BP-Buffer_B3.txt",
        "B3_melt": "20250416_1mm_BP-Buffer_B3_melt.txt",
        "B3_remelt": "20250416_1mm_BP-Buffer_B3_melt@.txt",
        "B3_post_melt": "20250416_1mm_BP-Buffer_B3_post_melt.txt",
        "B10": "20250416_1mm_BP-Buffer_B10.txt",
        "B10_melt": "20250416_1mm_BP-Buffer_B10_melt.txt",
        "B10_remelt": "20250416_1mm_BP-Buffer_B10_melt@.txt",
        "B10_post_melt": "20250416_1mm_BP-Buffer_B10_post_melt.txt"
    }
    
    # Parse all files
    parsed_data = {}
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        try:
            print(f"Parsing {key}: {fname}...")
            metadata, df = parse_jasco_cd_file(fpath)
            parsed_data[key] = {'meta': metadata, 'data': df}
            print(f"  -> Found {len(df)} data points. X-axis: {df.columns[0]}")
        except FileNotFoundError:
            print(f"  -> File not found: {fname}. Skipping {key}.")
        except ValueError as e:
            print(f"  -> Error parsing {fname}: {e}. Skipping {key}.")
        print("-" * 20)
    
    # Ensure blank file is available for subtraction
    if "Blank" in parsed_data and 'data' in parsed_data["Blank"]:
        blank_df = parsed_data["Blank"]['data']
    else:
        print("Blank data not found. Exiting...")
        exit(1)
    
    # Define the samples to process (all samples in the file dictionary except the Blank)
    samples = ["A9", "A12", "B8", "A1", "B12", "F4","A11", "B3", "B10"]
    for sample in samples:
        process_sample(sample, params_df, parsed_data, params, blank_df)
    
    print(f"\nProcessing complete. Plots saved to '{output_dir}'")
