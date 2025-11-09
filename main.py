#!/usr/bin/env python3
"""
main.py

Runs a single 6-DoF glide simulation for the Airbus A380 (zero thrust glide)
in a constant wind environment, and then produces a set of plots to validate
and analyze the simulation. The plots include:
  - Validation Plots (ground track, altitude, Euler angles, AoA, energy)
  - Analysis Plots (body velocities, sideslip, angular rates, horizontal speed,
    lift & drag, L/D ratio)
  - CL vs alpha, CD vs alpha, Fl vs airspeed, Fd vs airspeed
"""

import numpy as np
import math
from scipy.integrate import solve_ivp
import plotting
from wind_model import create_turbulence_state, get_turbulent_wind_ned
from dynamics import six_dof_odes



def main():
    # Define aircraft parameters for the A380 (typical cruise conditions)
    params = {
        "m": 560000.0,  # mass in kg
        "I": np.diag([1.2288e8, 1.5714e6, 3.8274e7]),  # inertia matrix in kg*m^2
        "I_inv": None,  # will be computed below
        "T": 0.0,       # zero thrust for gliding
        "S": 845.0,     # wing area in m^2
        "b": 79.75,     # wingspan in m
        "MAC": 11.0,    # mean aerodynamic chord in m
        "cd0": 0.016,   # zero-lift drag coefficient
        # 'k' is a constant that depends on the wing's planform shape. A higher 'k' (and thus a lower 'e') indicates that 
        # the lift distribution deviates more from the ideal elliptical shape, leading to higher induced drag
        "k": 0.05       # induced drag factor
    }
    # Compute the inverse of the inertia matrix.
    params["I_inv"] = np.linalg.inv(params["I"])

    # Define the initial state vector X0.
    # State vector layout: [N, E, D, u, v, w, φ, θ, ψ, p, q, r, dummy_fuel]
    X0 = np.zeros(13)
    X0[0:3] = [0.0, 0.0, -13000.0]              # Position: N=0, E=0, D=-13000 (i.e., altitude = 13,000 m)
    X0[3:6] = [250.0, 0.0, -22.0]                # Body velocities: u=250, v=0, w=-22 (m/s)
    X0[6:9] = [0.0, math.radians(-5.0), 0.0]      # Euler angles: roll=0, pitch=-5° (in rad), yaw=0
    X0[9:12] = [0.0, 0.0, 0.0]                   # Angular rates: p, q, r in rad/s
    X0[12] = 0.0                                # Extra state for fuel (unused, set to 0)

    # Create the initial turbulence state
    turb_state = create_turbulence_state(
        base_wind_ned=np.array([5.0, 0.0, 0.0]),
        turbulence_intensity=5.0,
        correlation_time=10.0
    )


    # Simulation time
    t_final = 600.0                # 10 minutes in seconds
    t_span = (0.0, t_final)
    t_eval = np.linspace(0.0, t_final, 1000)

    # Define the ODE function (using our six_dof_odes from the dynamics module)
    def ode_func(t, X):
        altitude_m = -X[2]  # NED: D = -alt
        wind_ned = (1,-8,2)  # no wind, but you can uncomment the next lines to use turbulence
        wind_ned, updated_state = get_turbulent_wind_ned(t, altitude_m, turb_state)
        # Make sure to store the updated state back
        turb_state.update(updated_state)
        return six_dof_odes(t, X, params, wind_ned)

    # Run the simulation using SciPy's solve_ivp
    sol = solve_ivp(ode_func, t_span, X0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)

    if sol.success:
        print("Simulation completed successfully.")
        # Validation Plots
        plotting.plot_validation_altitude(sol)
        plotting.plot_validation_euler_angles(sol)
        plotting.plot_validation_aoa(sol)
        plotting.plot_validation_energy(sol, params)

        # Analysis Plots
        plotting.plot_beta_vs_phi(sol)
        #plotting.plot_sideslip_and_roll(sol)
        plotting.plot_body_velocities(sol)
        plotting.plot_sideslip(sol)
        plotting.plot_horizontal_speed(sol)
        plotting.plot_Lift_and_Drag(sol, params)
        plotting.plot_LD_ratio(sol, params)
        #plotting.plot_CL_vs_alpha(sol, params)
        #plotting.plot_CD_vs_alpha(sol, params)
        plotting.plot_Fl_vs_airspeed(sol, params)
        plotting.plot_Fd_vs_airspeed(sol, params)

    else:
        print("Simulation failed:", sol.message)

if __name__ == "__main__":
    main()