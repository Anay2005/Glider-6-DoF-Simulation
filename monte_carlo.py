#!/usr/bin/env python3
"""
monte_carlo.py

Example: Monte Carlo simulations WITH turbulence, saving results to CSV.
 - We do NOT do any plotting here!
 - We'll collect final glide distance and altitude vs time for each run.
 - The time/altitude data are stored in columns like alt_0, alt_1, ..., alt_N in the CSV.
 - Then MC_plots.py will load and visualize these results.
"""

import os
import math
import csv
import random
import numpy as np
from scipy.integrate import solve_ivp

from dynamics import six_dof_odes
from wind_model import create_turbulence_state, get_turbulent_wind_ned

def run_monte_carlo_turbulence(n_runs=20,
                               t_final=600.0,
                               t_eval_count=200,
                               csv_filename="monte_carlo_turb_results.csv"):
    """
    Run multiple 6-DoF simulations with random turbulence.
    Save final distance and altitude-time history to CSV for later plotting.

    :param n_runs: number of Monte Carlo runs
    :param t_final: final time for each simulation (s)
    :param t_eval_count: number of points for time evaluation
    :param csv_filename: path to the output CSV file
    """
    # Aircraft parameters (example: large airliner)
    aircraft_params = {
        "m": 560000.0,                # mass [kg]
        "I": np.diag([1.2288e8, 1.5714e6, 3.8274e7]),
        "I_inv": None,
        "T": 0.0,                     # thrust [N]
        "S": 845.0,                   # wing area [m^2]
        "b": 79.75,                   # wingspan [m]
        "MAC": 11.0,                  # mean aerodynamic chord [m]
        "cd0": 0.016,
        "k": 0.05
    }
    aircraft_params["I_inv"] = np.linalg.inv(aircraft_params["I"])

    # Initial state: [N, E, D, u, v, w, phi, theta, psi, p, q, r, extra]
    X0 = np.zeros(13)
    # Start at altitude of 13,000 m (D=-13000 in NED)
    X0[2] = -13000.0
    # Body velocities
    X0[3] = 250.0  # forward speed
    X0[5] = -22.0  # negative w => slightly nose-down
    # Pitch = -5 deg
    X0[7] = math.radians(-5.0)

    # Time span and discrete times for output
    t_span = (0.0, t_final)
    t_eval = np.linspace(0.0, t_final, t_eval_count)

    # Open CSV and write the header
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # We'll store:
        #   run_idx, wind_u, wind_v, wind_w, turb_intensity, corr_time, distance_m
        # then columns alt_0,..., alt_(t_eval_count-1)
        header = [
            "run_idx", "wind_u", "wind_v", "wind_w",
            "turb_intensity", "corr_time",
            "distance_m"
        ]
        # Add columns alt_0 ... alt_(N-1)
        for i in range(t_eval_count):
            header.append(f"alt_{i}")
        writer.writerow(header)

        # Run each simulation
        for i_run in range(n_runs):
            # Randomly pick base wind or just keep zero
            wind_u = random.uniform(-5.0, 5.0)
            wind_v = random.uniform(-5.0, 5.0)
            wind_w = 0.0  # If you want vertical wind random, do similarly
            base_wind = np.array([wind_u, wind_v, wind_w])

            # Random turbulence parameters
            turb_intensity = random.uniform(3.0, 10.0)
            corr_time = random.uniform(5.0, 20.0)

            # Create new turbulence state for this run
            turb_state = create_turbulence_state(base_wind,
                                                 turbulence_intensity=turb_intensity,
                                                 correlation_time=corr_time)

            def ode_func(t, X):
                # altitude is -D in NED
                altitude_m = -X[2]
                # Evolve the turbulence
                wind_current, updated_ts = get_turbulent_wind_ned(t, altitude_m, turb_state)
                turb_state.update(updated_ts)
                # Evaluate the 6-DoF dynamics
                return six_dof_odes(t, X, aircraft_params, wind_current)

            # Solve
            sol = solve_ivp(ode_func, t_span, X0,
                            t_eval=t_eval, method='RK45',
                            rtol=1e-5, atol=1e-5)

            if not sol.success:
                print(f"[MC] Run {i_run}: Simulation failed: {sol.message}")
                # We can fill with Nones or skip
                continue

            # Final distance in horizontal plane
            xf = sol.y[0, -1]  # N
            yf = sol.y[1, -1]  # E
            distance_m = math.sqrt(xf*xf + yf*yf)

            # Collect altitude array at each time
            alt_array = -sol.y[2, :]

            # Build a row for the CSV
            row = [
                i_run,
                f"{wind_u:.3f}",
                f"{wind_v:.3f}",
                f"{wind_w:.3f}",
                f"{turb_intensity:.3f}",
                f"{corr_time:.3f}",
                f"{distance_m:.3f}"
            ]
            # Append altitudes as strings
            row += [f"{alt:.3f}" for alt in alt_array]

            writer.writerow(row)
            print(f"[MC] Run {i_run}: dist={distance_m:.1f} m, "
                  f"TI={turb_intensity:.1f}, tau={corr_time:.1f}")

    print(f"[MC] Completed {n_runs} runs with turbulence. Results in '{csv_filename}'")


def main():
    # For demonstration, run e.g. 20 Monte Carlo runs
    run_monte_carlo_turbulence(n_runs=100,
                               t_final=600.0,
                               t_eval_count=200,
                               csv_filename="monte_carlo_results.csv")

if __name__ == "__main__":
    main()
