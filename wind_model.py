#!/usr/bin/env python3
"""
wind_model.py

Extended to include a simplistic turbulence model using functional programming.
All wind models here rely on external (i.e., outside the function) state
to keep track of time-varying turbulence without using classes.

The user can maintain a 'turbulence_state' structure as a dictionary,
updated on each call.
"""

import numpy as np
import random
import math

def get_constant_wind_ned(u: float = -4.0, v: float = -2.0, w: float = 0.0):
    """
    Returns a constant wind vector in NED [m/s].
    """
    return np.array([u, v, w])

def sample_uniform_wind_ned(min_val: float = -10.0, max_val: float = 10.0):
    """
    Samples a single wind vector (u, v, w) in NED from a uniform distribution.
    """
    uW = random.uniform(min_val, max_val)
    vW = random.uniform(min_val, max_val)
    # For simplicity, keep vertical wind at 0.0 or also random if you like
    wW = 0.0
    return np.array([uW, vW, wW])

# turbulence_state

def create_turbulence_state(base_wind_ned: np.ndarray = np.array([0.0, 0.0, 0.0]),
                            turbulence_intensity: float = 5.0,
                            correlation_time: float = 10.0):
    """
    Creates and returns a dictionary holding the state needed
    for evolving turbulent wind with a Markov process.

    base_wind_ned: average wind vector [m/s] in NED
    turbulence_intensity: amplitude of random fluctuations (m/s)
    correlation_time: time scale (seconds) over which turbulence
                             changes significantly (first-order shaping).
    return: a dictionary holding everything needed to compute turbulent wind
             in subsequent updates.
    """
    return {
        "base_wind": base_wind_ned.copy(),
        "turbulence_intensity": turbulence_intensity,
        "correlation_time": correlation_time,
        "turb_state": np.zeros(3),  # internal random offset
        "last_time": None           # tracks last time step
    }

def get_turbulent_wind_ned(time_s: float,
                           altitude_m: float,
                           turb_state: dict):
    """
    Given the current time and altitude, returns a wind vector [m/s] in NED
    that is the sum of:
      (1) base wind from 'turb_state["base_wind"]'
      (2) a random offset that evolves using a first-order Markov process.

    We also update the 'turb_state' dictionary in-place to reflect the new
    random offset, so that each call depends on previous calls.

    time_s: current simulation time in seconds
    altitude_m: current altitude (not used in the default approach,
                       but could be used to scale turbulence)
    turb_state: dictionary from create_turbulence_state(...)
    return: (wind_ned, turb_state) -> 2-tuple of the new wind vector in NED
             and the updated turbulence state dictionary.
    """
    # If last_time is None, this is the first call
    if turb_state["last_time"] is None:
        turb_state["last_time"] = time_s
        # Return base wind only, no random offset yet
        return turb_state["base_wind"].copy(), turb_state

    dt = time_s - turb_state["last_time"]
    turb_state["last_time"] = time_s

    # If dt is invalid or zero, just return current state
    if dt <= 0:
        wind = turb_state["base_wind"] + turb_state["turb_state"]
        return wind, turb_state

    tau = turb_state["correlation_time"]
    sigma = turb_state["turbulence_intensity"]

    # White noise draw
    random_vec = np.random.normal(loc=0.0, scale=1.0, size=3)

    # The standard approach for first-order shaping:
    # dW = -(1/tau)*W * dt + sqrt(2*sigma^2/tau)* dW_random
    # We'll do an Euler step:
    offset = turb_state["turb_state"]
    offset += - (1.0/tau)*offset * dt + math.sqrt(2.0*(sigma**2)/tau)*random_vec*math.sqrt(dt)
    turb_state["turb_state"] = offset


    wind_ned = turb_state["base_wind"] + turb_state["turb_state"]
    return wind_ned, turb_state
