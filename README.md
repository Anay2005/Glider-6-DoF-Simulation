# Airbus A380 Glider – 6-DoF Flight Dynamics and Monte Carlo Simulation

A complete **6-Degrees-of-Freedom (6-DoF)** flight dynamics simulation of the Airbus A380 in *engine-off* glider mode.  
This project models aerodynamic, kinematic, and control behaviour, incorporating **wind disturbance modelling**, **Monte Carlo analysis**, and **machine-learning-based flight prediction**.  

Developed as part of an advanced aerospace simulation initiative to explore large-aircraft stability, glide performance, and control robustness under uncertainty.

---

## ✈️ Key Features

- **Full 6-DoF rigid-body dynamics**  
  Simulates translational and rotational motion using realistic aircraft mass and inertia parameters.

- **Aerodynamic model (`aero.py`)**  
  Computes lift, drag, side-force, and moment coefficients as functions of angle of attack, sideslip, and control deflections.

- **Flight dynamics engine (`dynamics.py`)**  
  Integrates Newton–Euler equations of motion using adaptive time-stepping.

- **Control surface model (`controls.py`)**  
  Implements elevator, aileron, and rudder control logic for glide stabilisation.

- **Atmospheric & wind modelling (`wind_model.py`)**  
  Adds stochastic gusts, turbulence, and altitude-dependent wind profiles.

- **Monte Carlo simulation (`monte_carlo.py`)**  
  Runs hundreds of randomised flight scenarios to assess sensitivity to initial conditions and turbulence.

- **Machine-Learning performance predictor (`ml_model.py`)**  
  Trains a regression/classification model to predict glide distance and stability outcomes based on simulation data.

- **Automated plotting modules (`plotting.py`, `MC_plots.py`, `ML_plots.py`)**  
  Generates trajectory plots, attitude histories, energy curves, and statistical summaries.

---

## 🧠 Project Architecture


A380_Glider_6DoF/
├── main.py               # Main entry point for running simulations
├── aero.py               # Aerodynamic coefficient and force models
├── controls.py           # Control surface logic
├── dynamics.py           # 6-DoF motion equations and integrator
├── wind_model.py         # Atmospheric and gust models
├── monte_carlo.py        # Monte Carlo uncertainty simulations
├── ml_model.py           # Machine learning model training and evaluation
├── plotting.py           # Common plotting utilities
├── MC_plots.py           # Plots for Monte Carlo statistics
├── ML_plots.py           # Plots for ML predictions
├── requirements.txt      # Python dependencies
└── README.md             # Documentation (this file)


---

## ⚙️ Mathematical Model Overview

### Translational Dynamics
Translational Dynamics

𝑚
 
𝑣
˙
=
𝐹
aero
+
𝐹
gravity
+
𝐹
wind
m
v
˙
=F
aero
	​

+F
gravity
	​

+F
wind
	​


Rotational Dynamics

𝐼
𝜔
˙
+
𝜔
×
(
𝐼
𝜔
)
=
𝑀
aero
+
𝑀
control
I
ω
˙
+ω×(Iω)=M
aero
	​

+M
control
	​


Where:

𝑣
v: velocity vector in body frame

𝜔
ω: angular velocity vector

𝐼
I: inertia matrix

𝐹
,
𝑀
F,M: net forces and moments 

### Integration
Implemented using fixed-step or variable-step Runge–Kutta integrators.  
Attitude is propagated via quaternion updates for numerical stability.

---

## 🧩 Simulation Capabilities

| Module | Purpose |
|---------|----------|
| **`main.py`** | Sets up simulation, loads parameters, executes 6-DoF loop |
| **`monte_carlo.py`** | Randomises initial altitude, angle of attack, and wind disturbances |
| **`wind_model.py`** | Adds dry and turbulent wind components |
| **`ml_model.py`** | Learns mapping between flight inputs and performance |
| **`plotting.py`** | Generates trajectory, orientation, and energy plots |
| **`MC_plots.py`** | Plots statistical distribution of outcomes |
| **`ML_plots.py`** | Visualises model accuracy and predicted performance |

---

## 📊 Outputs

When executed, the simulation produces:

- Time-series data (`.csv` or `.npy`): position, velocity, attitude, angular rates  
- Flight plots:
  - 3D trajectory and attitude evolution  
  - Lift/drag coefficient variation  
  - Altitude and velocity vs time  
- Monte Carlo summaries:
  - Histograms of glide distance and impact time  
  - Mean ± σ envelopes for altitude decay  
- Machine learning diagnostics:
  - Predicted vs actual glide performance scatter plots  
  - Feature importance rankings

Example visualisations:

- **Glide trajectory (3D)**  
- **Monte Carlo envelope of altitude decay**  
- **ML-predicted glide ratio vs true values**

---

## 🚀 Running the Simulation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Glider-6-DoF-simulation.git
   cd Glider-6-DoF-simulation
