#!/usr/bin/env python3
"""
aero.py

Aerodynamic and atmosphere modeling for the aircraft.

Provides:
- Standard atmosphere (ISA) calculation
- Computation of lift/drag coefficients
"""

import math
import numpy as np

# This function models temperature (T), pressure (p), density (\rho), and speed of sound (a) in the troposphere (up to 11 km) in SI units
# and the lower stratosphere (above 11 km) according to a simplified version of the International Standard Atmosphere (ISA)
def standard_atmosphere(altitude_m: float):
    """
    Returns T (K), p (Pa), rho (kg/m^3), and speed of sound a (m/s).
    """
    # R = specific gas constant for dry air
    R = 287.05
    # gamma = ratio of specific heats for dry air(c_p/c_v)
    # (approximately 1.4 for air)
    gamma = 1.4
    # g0 = standard gravity
    g0 = 9.80665
    # Standard temperature and pressure at sea level
    T0 = 288.15
    p0 = 101325
    # Lapse rate in K/m(approx 6.5 degrees decrease per 1000m)
    lapse_rate = -0.0065
    # Troposhere is from 0 to 11 km and temp decreases with altitude we integrate the hydrostatic equation dp/dz = -rho*g
    if altitude_m <= 11000:
        T = T0 + lapse_rate * altitude_m
        p = p0 * (T / T0) ** (-g0 / (lapse_rate * R))
    # isothermal layer(Stratosphere) above 11 km hence an exponential decay is used
    else:
        T = T0 + lapse_rate * 11000
        p_11 = p0 * (T / T0) ** (-g0 / (lapse_rate * R))
        p = p_11 * math.exp(-g0 / (R * T) * (altitude_m - 11000))
    # Ideal gas law to compute density with V = 1m^3
    rho = p / (R * T)
    # The speed of sound in a gas is related to how fast pressure disturbances travel; 
    # it depends primarily on temperature (hotter air = faster molecules = higher sound speed).
    a = math.sqrt(gamma * R * T)
    return T, p, rho, a

def compute_CL(alpha_rad, CL0=0.25, CL_alpha=6.283):
    """
    Computes the lift coefficient using a simple linear model:
    
        C_L = C_{L0} + C_{L\alpha} * alpha
    
    where:
      - alpha_rad is the angle of attack in radians.
      - CL0 is the zero-lift coefficient (dimensionless).
      - CL_alpha is the lift slope (dimensionless per radian).
    
    This model is valid for small-to-moderate angles of attack.
    """
    return CL0 + CL_alpha * alpha_rad

def compute_CD(CL: float, cd0: float, k: float) -> float:
    """
    Parabolic drag polar: CD = cd0 + k * CL^2
    """
    return cd0 + k*(CL**2)

def wind_to_body_rotation(alpha: float, beta: float):
    """
    Rotation matrix from wind axes to body axes.
    alpha = angle of attack, beta = sideslip.
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    return np.array([
        [ca*cb, -ca*sb, -sa],
        [sb,     cb,     0.0],
        [sa*cb,  sa*sb,  ca]
    ])
