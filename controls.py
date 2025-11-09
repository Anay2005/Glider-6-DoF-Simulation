#!/usr/bin/env python3
"""
controls.py

Simple control laws for elevator, aileron, rudder.
"""

import math
import numpy as np

# Rudder control the yaw rate and the heading angle
# Note: Rudder deflection is positive for right yaw 
# if β appears, use the rudder to yaw the nose into the wind and kill the slip.
def compute_rudder_deflection(psi, r, desired_psi=0.0):
    Kp_psi = 1.0
    Kd_r   = 0.8
    psi_error = (desired_psi - psi + math.pi) % (2*math.pi) - math.pi
    delta_r = Kp_psi*psi_error - Kd_r*r
    max_defl = math.radians(1.0)
    return np.clip(delta_r, -max_defl, max_defl)
# Aileron control the roll rate and the roll angle
# Note: Aileron deflection is positive for right roll
# This is responsible for dutch roll
def compute_aileron_deflection(phi, p, desired_phi=0.0):
    Kp_phi = 1.0
    Kd_p = 0.1
    phi_err = desired_phi - phi
    delta_a = Kp_phi*phi_err - Kd_p*p
    max_defl = math.radians(14.0)
    return np.clip(delta_a, -max_defl, max_defl)
# elevator control the pitch angle and the angle of attack
# Note: Elevator deflection is positive for nose up
# This causes the Phugoid oscillation which is a natural oscillation of the aircraft as it creates a moment
# it is used to control the aircraft in turbulent wind conditions
def compute_elevator_deflection(alpha_rad, q, alpha_trim_deg):
    
    alpha_trim_rad = math.radians(alpha_trim_deg)
    error = alpha_trim_rad - alpha_rad
    # too large k_d_alpha would cause oscillation and too low k_d_alpha would cause slow response
    # if the wind is there then you would need a bigger gain but not too big
    # if the wind is not there then you would need a smaller gain(0.1 works)
    Kp_alpha = 1.0
    Kd_alpha = 1.6
    # elevator deflection is limited to a maximum of 1 degree
    max_elev_deg = math.radians(1.0)
    delta_e = Kp_alpha * error - Kd_alpha * q
    delta_e = np.clip(delta_e, -max_elev_deg, max_elev_deg)
    print(f"alpha_rad: {alpha_rad}, q: {q}, error: {error}, delta_e: {delta_e}")
    return delta_e
"""
Without elevator control AOA settles at 5 with it it reduces to 3 degrees peak 
1. Pitch ↔ Surge/Heave (Longitudinal)
    * If you pitch up, your surge speed (u) drops (climb) and heave (w) goes positive (going up).
    * If you pitch down, you speed up and sink.
    * This back‑and‑forth is the phugoid.
2. Roll ↔ Yaw (Lateral‑Directional)
    * A roll changes where your wings point, creating a sideslip (v) that yaws the nose back 
    * A yaw creates sideslip, which rolls the airplane 
    * That dance is the Dutch‑roll.
3. Cross‑coupling via ω×v
    * The term ω×v in the equations means that any rotation (p, q, r) will “tilt” the velocity vector, feeding back into surge, sway, or heave.
    * E.g. roll rate p makes you slip (v), and yaw rate r makes you roll (p) a bit.

"""