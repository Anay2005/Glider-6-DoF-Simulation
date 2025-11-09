#!/usr/bin/env python3
"""
MC_plots.py

Plots for the Monte Carlo runs with turbulence.
We read the CSV file produced by monte_carlo.py,
then generate:
 1) A histogram of final distances
 2) Altitude vs time with mean ± 1 std (error bars) across runs
"""

import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def plot_mc_distance_hist_turb(csv_path="monte_carlo_turb_results.csv", bins=15):
    """
    Reads final glide distances from the CSV and plots a histogram.
    """
    distances = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist_str = row.get("distance_m", "")
            if dist_str != "":
                dist_val = float(dist_str)
                distances.append(dist_val)

    if not distances:
        print("[MC Plot] No distance data found in CSV.")
        return

    plt.figure()
    plt.hist(distances, bins=bins, alpha=0.7)
    plt.xlabel("Final Glide Distance (m)")
    plt.ylabel("Count")
    plt.title("Distribution of Final Glide Distances (with Turbulence)")
    plt.grid(True)
    plt.show()

def plot_mc_altitude_spread_turb(csv_path="monte_carlo_turb_results.csv"):
    """
    Reads altitude vs time columns from the CSV (alt_0..alt_N).
    Plots the mean altitude ± 1 std dev as a function of time.
    Assumes all runs used the same t_eval spacing.
    """
    # We'll store altitudes in a 2D list: [run_index][time_index]
    alt_data = []
    run_count = 0

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # read the column names
        # Identify which columns are alt_XXX
        alt_columns = [col for col in header if col.startswith("alt_")]
        alt_columns.sort(key=lambda x: int(x.split("_")[1]))  # sort by numeric index

        # parse times from 0..N-1 to create a time array
        # We do NOT store the times in the CSV, but we know how many alt columns exist.
        # Let's assume uniform distribution from 0..t_final. We can figure that out
        # only if we stored t_final & steps, or simply show index-based "time" on the X-axis.
        # For a robust approach, you can store "time_x" columns in the CSV as well.
        # We'll do a simplified approach using integer step index as "time".
        num_alt_cols = len(alt_columns)
        # We'll treat these as step indices for the X-axis
        t_steps = np.arange(num_alt_cols)

        # Prepare to read data rows
        alt_col_indices = [header.index(c) for c in alt_columns]

        # Fill alt_data from each row
        for row in reader:
            run_alts = []
            for idx in alt_col_indices:
                alt_str = row[idx]
                alt_val = float(alt_str)
                run_alts.append(alt_val)
            alt_data.append(run_alts)
            run_count += 1

    if run_count == 0:
        print("[MC Plot] No valid runs found in CSV for altitude spread.")
        return

    # Convert to numpy
    alt_data_np = np.array(alt_data)  # shape = (runs, num_alt_cols)

    # Mean & std at each time step
    mean_alt = np.mean(alt_data_np, axis=0)
    std_alt = np.std(alt_data_np, axis=0)

    # Plot
    plt.figure()
    plt.plot(t_steps, mean_alt, label="Mean Altitude")
    plt.fill_between(t_steps, mean_alt - std_alt, mean_alt + std_alt,
                     alpha=0.3, label="±1 Std Dev")
    plt.xlabel("Time Step Index")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude vs Time (Mean ± 1σ) Across MC Runs")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Plot the final distance histogram
    plot_mc_distance_hist_turb("monte_carlo_results.csv")

    # Plot the altitude spread (mean ± 1 std)
    plot_mc_altitude_spread_turb("monte_carlo_results.csv")

if __name__ == "__main__":
    main()
