#!/usr/bin/env python3
"""
plotting.py

 - "Validation" plots:
    1) Ground Track
    2) Altitude vs Time
    3) Euler Angles vs Time
    4) AoA vs Time
    5) Energy vs Time

 - Analysis plots:
    1) Body velocities vs Time
    2) Sideslip vs Time
    3) p,q,r (angular rates) vs Time
    4) Horizontal speed vs Time
    5) Lift & Drag vs Time
    6) L/D ratio vs Time
    7) C_L vs alpha
    8) C_D vs alpha
    9) F_L vs airspeed
    10) F_D vs airspeed
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from aero import standard_atmosphere, compute_CL, compute_CD
from dynamics import rotation_matrix_body_to_inertial


# VALIDATION PLOTS 

def plot_validation_altitude(sol):
    """
    Altitude vs Time (NED: -z)
    """
    t = sol.t
    alt = -sol.y[2]
    plt.figure()
    plt.plot(t, alt/1000, 'r-')
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (km)")
    plt.title("Altitude vs Time")
    plt.grid(True)
    plt.show()

def plot_validation_euler_angles(sol):
    """
    Roll, Pitch, Yaw angles vs Time.
    """
    t = sol.t
    phi_deg = np.degrees(sol.y[6])
    tht_deg = np.degrees(sol.y[7])
    psi_deg = np.degrees(sol.y[8])
    plt.figure()
    plt.plot(t, phi_deg, label="Roll φ (deg)")
    plt.plot(t, tht_deg, label="Pitch θ (deg)")
    plt.plot(t, psi_deg, label="Yaw ψ (deg)")
    plt.title("Euler Angles vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_validation_aoa(sol):
    """
    Angle of Attack vs Time.
    """
    t = sol.t
    alpha_deg_array = []
    for i in range(len(t)):
        u = sol.y[3, i]
        w = sol.y[5, i]
        alpha = math.degrees(math.atan2(w, u)) if abs(u) > 1e-6 else 0.0
        alpha_deg_array.append(alpha)
    plt.figure()
    plt.plot(t, alpha_deg_array, 'g-')
    plt.xlabel("Time (s)")
    plt.ylabel("AoA (deg)")
    plt.title("Angle of Attack vs Time")
    plt.grid(True)
    plt.show()

def plot_validation_energy(sol, params):
    """
    Mechanical Energy vs Time.
    E = m*g*h + 0.5*m*V^2
    """
    t = sol.t
    m = params["m"]
    g = 9.81
    E_total = []
    for i in range(len(t)):
        alt = -sol.y[2, i]
        potE = m*g*alt
        u = sol.y[3, i]
        v = sol.y[4, i]
        w = sol.y[5, i]
        kinE = 0.5*m*(u*u + v*v + w*w)
        E_total.append(potE + kinE)

    plt.figure()
    plt.plot(t, E_total, 'm-')
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title("Total Mechanical Energy vs Time")
    plt.grid(True)
    plt.show()

# ANALYSIS PLOTS


def plot_body_velocities(sol):
    """
    Body velocities (u,v,w) vs Time
    """
    t = sol.t
    u = sol.y[3]
    v = sol.y[4]
    w = sol.y[5]
    plt.figure()
    plt.plot(t, u, label='u (fwd)')
    plt.plot(t, v, label='v (side)')
    plt.plot(t, w, label='w (down)')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Body Velocities vs Time")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_sideslip(sol):
    """
    Sideslip (beta) vs Time

    """
    t = sol.t
    beta_deg_array = []
    for i in range(len(t)):
        u = sol.y[3, i]
        v = sol.y[4, i]
        w = sol.y[5, i]
        V = math.sqrt(u*u + v*v + w*w)
        if V < 1e-6:
            beta_rad = 0.0
        else:
            beta_rad = math.asin(np.clip(v / V, -1, 1))
        beta_deg_array.append(math.degrees(beta_rad))

    plt.figure()
    plt.plot(t, beta_deg_array, 'b-')
    plt.xlabel("Time (s)")
    plt.ylabel("Beta (deg)")
    plt.title("Sideslip vs Time")
    plt.grid(True)
    plt.show()


def plot_horizontal_speed(sol):
    """
    Horizontal speed vs Time (in inertial frame)
    """
    t = sol.t
    horizontal_speed_array = []
    for i in range(len(t)):
        phi   = sol.y[6, i]
        theta = sol.y[7, i]
        psi   = sol.y[8, i]
        R_bi  = rotation_matrix_body_to_inertial(phi, theta, psi)
        u_b   = sol.y[3, i]
        v_b   = sol.y[4, i]
        w_b   = sol.y[5, i]
        vel_inertial = R_bi @ np.array([u_b, v_b, w_b])
        vx = vel_inertial[0]
        vy = vel_inertial[1]
        hz = math.sqrt(vx*vx + vy*vy)
        horizontal_speed_array.append(hz)

    plt.figure()
    plt.plot(t, horizontal_speed_array, 'g-')
    plt.xlabel("Time (s)")
    plt.ylabel("Horizontal Speed (m/s)")
    plt.title("Horizontal Speed vs Time (Inertial)")
    plt.grid(True)
    plt.show()

def plot_Lift_and_Drag(sol, params):
    """
    Lift & Drag vs Time
    """
    S    = params["S"]
    cd0  = params["cd0"]
    k    = params["k"]
    t    = sol.t
    Larr = []
    Darr = []

    for i in range(len(t)):
        alt = -sol.y[2, i]
        u   = sol.y[3, i]
        v   = sol.y[4, i]
        w   = sol.y[5, i]
        V = math.sqrt(u*u + v*v + w*w)
        _, _, rho, _ = standard_atmosphere(alt)
        alpha = math.atan2(w, u) if abs(u)>1e-6 else 0.0
        CL = compute_CL(alpha)
        CD = compute_CD(CL, cd0, k)
        qdyn = 0.5 * rho * (V**2)
        lift = qdyn * S * CL
        drag = qdyn * S * CD
        Larr.append(lift)
        Darr.append(drag)

    plt.figure()
    plt.plot(t, Larr, label="Lift (N)")
    plt.plot(t, Darr, label="Drag (N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Lift & Drag vs Time")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_LD_ratio(sol, params):
    """
    L/D ratio vs time
    """
    S    = params["S"]
    cd0  = params["cd0"]
    k    = params["k"]
    t    = sol.t
    LD   = []

    for i in range(len(t)):
        alt = -sol.y[2, i]
        u   = sol.y[3, i]
        v   = sol.y[4, i]
        w   = sol.y[5, i]
        V = math.sqrt(u*u + v*v + w*w)
        _, _, rho, _ = standard_atmosphere(alt)
        alpha = math.atan2(w, u) if abs(u)>1e-6 else 0.0
        CL = compute_CL(alpha)
        CD = compute_CD(CL, cd0, k)
        qdyn = 0.5*rho*(V**2)
        lift = qdyn*S*CL
        drag = qdyn*S*CD
        if abs(drag)>1e-6:
            LD.append(lift/drag)
        else:
            LD.append(0.0)

    plt.figure()
    plt.plot(t, LD, 'r-')
    plt.xlabel("Time (s)")
    plt.ylabel("L/D")
    plt.title("Lift-to-Drag Ratio vs Time")
    plt.grid(True)
    plt.show()


def plot_CL_vs_alpha(sol, params):
    """
    Plot C_L vs alpha over the entire trajectory.

    """
    t = sol.t
    alpha_list = []
    CL_list    = []
    for i in range(len(t)):
        # compute alpha
        u = sol.y[3, i]
        w = sol.y[5, i]
        alpha_rad = math.atan2(w, u) if abs(u)>1e-6 else 0.0
        alpha_list.append(math.degrees(alpha_rad))

        # compute CL
        CL_val = compute_CL(alpha_rad, 
                            CL0=0.25,
                            CL_alpha=6.283)
        CL_list.append(CL_val)

    plt.figure()
    plt.plot(alpha_list, CL_list, 'o', markersize=2, color='blue', alpha=0.5)
    plt.xlabel("Angle of Attack (deg)")
    plt.ylabel("C_L")
    plt.title("C_L vs alpha")
    plt.grid(True)
    # set the limits on the axes
    plt.xlim(1, 3)
    plt.ylim(-0.5, 2.5)
    plt.show()

def plot_CD_vs_alpha(sol, params):
    """
    Plot C_D vs alpha over entire trajectory.
    We'll gather alpha, compute C_D from each time step, scatter.
    """
    cd0 = params["cd0"]
    k   = params["k"]
    t = sol.t
    alpha_list = []
    CD_list    = []
    for i in range(len(t)):
        # compute alpha
        u = sol.y[3, i]
        w = sol.y[5, i]
        alpha_rad = math.atan2(w, u) if abs(u)>1e-6 else 0.0
        alpha_list.append(math.degrees(alpha_rad))

        # compute CL, then CD
        CL_val = compute_CL(alpha_rad,
                            CL0=0.25,
                            CL_alpha=6.283,
)
        CD_val = compute_CD(CL_val, cd0, k)
        CD_list.append(CD_val)

    plt.figure()
    plt.plot(alpha_list, CD_list, 'o', markersize=2, color='red', alpha=0.5)
    plt.xlabel("Angle of Attack (deg)")
    plt.ylabel("C_D")
    plt.title("C_D vs alpha")
    plt.xlim(1, 2.5)
    plt.grid(True)
   
    plt.show()

def plot_Fl_vs_airspeed(sol, params):
    """
    Plot Lift Force (F_L) vs airspeed V 

    Lift = 0.5 * rho * V^2 * S * C_L
    """
    S    = params["S"]
    cd0  = params["cd0"]
    k    = params["k"]
    t    = sol.t

    V_list  = []
    F_lift_list = []

    for i in range(len(t)):
        alt = -sol.y[2, i]
        u   = sol.y[3, i]
        v   = sol.y[4, i]
        w   = sol.y[5, i]
        V = math.sqrt(u*u + v*v + w*w)
        T_, p_, rho, _ = standard_atmosphere(alt)

        alpha_rad = math.atan2(w, u) if abs(u)>1e-6 else 0.0
        CL_val = compute_CL(alpha_rad,
                            CL0=0.25,
                            CL_alpha=6.283,)
        qdyn = 0.5 * rho * (V**2)
        lift = qdyn*S*CL_val

        V_list.append(V)
        F_lift_list.append(lift)

    # We'll sort the data by V
    data = list(zip(V_list, F_lift_list))
    data.sort(key=lambda x: x[0])
    V_sorted, L_sorted = zip(*data)

    plt.figure()
    plt.plot(V_sorted, L_sorted, 'b-')
    plt.xlabel("Airspeed V (m/s)")
    plt.ylabel("Lift (N)")
    plt.title("Lift Force vs Airspeed")
    plt.grid(True)
    plt.show()

def plot_Fd_vs_airspeed(sol, params):
    """
    Plot Drag Force (F_D) vs airspeed V across the trajectory.
    
    """
    S    = params["S"]
    cd0  = params["cd0"]
    k    = params["k"]
    t    = sol.t

    V_list  = []
    F_drag_list = []

    for i in range(len(t)):
        alt = -sol.y[2, i]
        u   = sol.y[3, i]
        v   = sol.y[4, i]
        w   = sol.y[5, i]
        V = math.sqrt(u*u + v*v + w*w)
        T_, p_, rho, _ = standard_atmosphere(alt)

        alpha_rad = math.atan2(w, u) if abs(u)>1e-6 else 0.0
        CL_val = compute_CL(alpha_rad,
                            CL0=0.25,
                            CL_alpha=6.283)
        CD_val = compute_CD(CL_val, cd0, k)
        qdyn = 0.5 * rho * (V**2)
        drag = qdyn*S*CD_val

        V_list.append(V)
        F_drag_list.append(drag)

    # sort by V
    data = list(zip(V_list, F_drag_list))
    data.sort(key=lambda x: x[0])
    V_sorted, D_sorted = zip(*data)

    plt.figure()
    plt.plot(V_sorted, D_sorted, 'm-')
    plt.xlabel("Airspeed V (m/s)")
    plt.ylabel("Drag (N)")
    plt.title("Drag Force vs Airspeed")
    plt.grid(True)
    plt.show()

# plot sideslip and roll in the same plot to see dutch roll
def plot_sideslip_and_roll(sol):
    """
    Plot Sideslip (beta) and Roll (phi) vs Time
    """
    t = sol.t
    beta_deg_array = []
    phi_deg_array = []
    for i in range(len(t)):
        u = sol.y[3, i]
        v = sol.y[4, i]
        w = sol.y[5, i]
        V = math.sqrt(u*u + v*v + w*w)
        if V < 1e-6:
            beta_rad = 0.0
        else:
            beta_rad = math.asin(np.clip(v / V, -1, 1))
        beta_deg_array.append(math.degrees(beta_rad))
        phi_deg_array.append(math.degrees(sol.y[6, i]))

    plt.figure()
    plt.plot(t, beta_deg_array, 'b-', label='Sideslip (deg)')
    plt.plot(t, phi_deg_array, 'r-', label='Roll (deg)')
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Sideslip and Roll vs Time")
    plt.grid(True)
    plt.legend()
    plt.show()

# plot beta vs phi
def plot_beta_vs_phi(sol):
    """
    Plot Sideslip (beta) vs Roll (phi)
    """
    beta_deg_array = []
    phi_deg_array = []
    for i in range(len(sol.t)):
        u = sol.y[3, i]
        v = sol.y[4, i]
        w = sol.y[5, i]
        V = math.sqrt(u*u + v*v + w*w)
        if V < 1e-6:
            beta_rad = 0.0
        else:
            beta_rad = math.asin(np.clip(v / V, -1, 1))
        beta_deg_array.append(math.degrees(beta_rad))
        phi_deg_array.append(math.degrees(sol.y[6, i]))

    plt.figure()
    plt.plot(phi_deg_array, beta_deg_array, 'b-', label='Sideslip vs Roll')
    plt.xlabel("Roll (deg)")
    plt.ylabel("Sideslip (deg)")
    plt.title("Sideslip vs Roll")
    plt.grid(True)
    plt.legend()
    plt.show()