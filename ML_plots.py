#!/usr/bin/env python3
"""
ML.py

Contains plotting functions specifically for the Machine Learning (ML) model results.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def plot_ml_actual_vs_pred(y_true, y_pred):
    """
    ML Plot #1: Actual vs. Predicted final distance scatter + y=x line.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    plt.xlabel("Actual Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title("ML Plot #1: Actual vs. Predicted Glide Distance")
    plt.grid(True)
    plt.show()

def plot_ml_feature_importance(feature_names, importances):
    """
    ML Plot #2: Bar chart of feature importances
    """
    plt.figure()
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importances, align='center')
    plt.yticks(y_pos, feature_names)
    plt.xlabel("Importance")
    plt.title("ML Plot #2: Feature Importances")
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x')
    plt.show()

def plot_ml_partial_dependence(model, feature_index=0, feature_name="wind_u", 
                               range_min=-15, range_max=15):
    """
    ML Plot #3: Partial dependence for one wind component dimension.
    We fix other features at 0, vary the chosen feature from range_min..range_max,
    and compute the model's predictions.
    """
    x_vals = np.linspace(range_min, range_max, 50)
    X_grid = np.zeros((50, 3))  # e.g., (wind_u, wind_v, wind_w)
    for i in range(50):
        X_grid[i, feature_index] = x_vals[i]

    preds = model.predict(X_grid)

    plt.figure()
    plt.plot(x_vals, preds, 'b-')
    plt.xlabel(feature_name + " (m/s)")
    plt.ylabel("Predicted Distance (m)")
    plt.title(f"ML Plot #3: Partial Dependence on {feature_name} (Mean Only)")
    plt.grid(True)
    plt.show()

def plot_ml_prediction_error_bars(y_test, X_test, model, preds_ensemble):
    """
    ML Plot #4: Predicted vs. actual with vertical error bars from ensemble std dev.
    :param y_test: array of actual distances
    :param X_test: test features
    :param model: the random forest model
    :param preds_ensemble: shape (n_estimators, n_test) for each tree's predictions
    """
    mean_pred = np.mean(preds_ensemble, axis=0)
    std_pred  = np.std(preds_ensemble, axis=0)

    plt.figure()
    plt.errorbar(y_test, mean_pred, yerr=std_pred,
                 fmt='o', ecolor='gray', color='blue', alpha=0.7,
                 capsize=3, markersize=5)
    min_val = min(min(y_test), min(mean_pred))
    max_val = max(max(y_test), max(mean_pred))
    plt.plot([min_val, max_val],[min_val, max_val], 'k--', linewidth=1)
    plt.xlabel("Actual Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title("ML Plot #4: Actual vs Pred (Error Bars = ±1σ across forest)")
    plt.grid(True)
    plt.show()

def plot_ml_partial_dependence_with_error(model,
                                          feature_index=0,
                                          feature_name="wind_u",
                                          range_min=-15,
                                          range_max=15):
    """
    ML Plot #5: Partial dependence with ±1σ region across all trees.
    We fix other features at 0, vary the chosen feature in [range_min..range_max],
    and collect each tree's prediction => plot mean ± std.
    """
    n_points = 50
    x_vals = np.linspace(range_min, range_max, n_points)
    X_grid = np.zeros((n_points, 3))
    for i in range(n_points):
        X_grid[i, feature_index] = x_vals[i]

    # get predictions from each estimator
    preds_each_tree = []
    for est in model.estimators_:
        p = est.predict(X_grid)
        preds_each_tree.append(p)
    preds_each_tree = np.array(preds_each_tree)

    mean_pred = np.mean(preds_each_tree, axis=0)
    std_pred  = np.std(preds_each_tree, axis=0)

    plt.figure()
    plt.plot(x_vals, mean_pred, 'b-', label="Mean")
    plt.fill_between(x_vals, mean_pred - std_pred, mean_pred + std_pred,
                     color='blue', alpha=0.2, label="±1σ")
    plt.xlabel(feature_name + " (m/s)")
    plt.ylabel("Predicted Distance (m)")
    plt.title(f"ML Plot #5: Partial Dependence on {feature_name} with ±1σ")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_energy_balance_difference_heatmap(sol, params,
                                           n_time_bins=40,
                                           n_alt_bins=40):
    """
    This advanced analysis produces a 2D heatmap of the mismatch:
        Δ(t) = dE/dt(t) + dragPower(t).
    Binned by time (x-axis) and altitude (y-axis).

    This is not strictly MC or ML, but is retained here under ML.py
    for advanced flight analysis.
    """
    t = sol.t
    m = params["m"]
    S = params["S"]
    cd0 = params["cd0"]
    k   = params["k"]
    g   = 9.81

    alt_data = -sol.y[2]
    u_data   = sol.y[3]
    v_data   = sol.y[4]
    w_data   = sol.y[5]

    # 1) Mechanical energy
    E_array = []
    for i in range(len(t)):
        alt_i = alt_data[i]
        speed_i = math.sqrt(u_data[i]**2 + v_data[i]**2 + w_data[i]**2)
        PE  = m*g*alt_i
        KE  = 0.5*m*(speed_i**2)
        E_array.append(PE + KE)
    E_array = np.array(E_array)

    dE_dt = np.gradient(E_array, t)

    # 2) drag power
    drag_power_array = []
    from aero import standard_atmosphere, compute_CL, compute_CD
    for i in range(len(t)):
        alt_i = alt_data[i]
        speed_i = math.sqrt(u_data[i]**2 + v_data[i]**2 + w_data[i]**2)
        # local density
        _, _, rho, _ = standard_atmosphere(alt_i)
        alpha = 0.0
        if abs(u_data[i])>1e-6:
            alpha = math.atan2(w_data[i], u_data[i])
        CL = compute_CL(alpha)
        CD = compute_CD(CL, cd0, k)
        drag = 0.5*rho*(speed_i**2)*S*CD
        drag_power = drag * speed_i
        drag_power_array.append(drag_power)
    drag_power_array = np.array(drag_power_array)

    diff_array = dE_dt + drag_power_array

    # 3) Bin by time and altitude
    t_min, t_max = t[0], t[-1]
    alt_min, alt_max = alt_data.min(), alt_data.max()

    time_bins = np.linspace(t_min, t_max, n_time_bins+1)
    alt_bins  = np.linspace(alt_min, alt_max, n_alt_bins+1)

    aggregator       = np.zeros((n_time_bins, n_alt_bins))
    aggregator_count = np.zeros((n_time_bins, n_alt_bins))

    for i in range(len(t)):
        ti  = t[i]
        ai  = alt_data[i]
        di  = diff_array[i]

        tb = np.searchsorted(time_bins, ti) - 1
        ab = np.searchsorted(alt_bins, ai) - 1

        if 0 <= tb < n_time_bins and 0 <= ab < n_alt_bins:
            aggregator[tb, ab]       += di
            aggregator_count[tb, ab] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        difference_2d = aggregator / aggregator_count
    difference_2d[aggregator_count == 0] = np.nan

    plt.figure()
    cmap = plt.cm.RdBu
    pc = plt.pcolormesh(time_bins, alt_bins, difference_2d.T, cmap=cmap, shading='auto')
    plt.colorbar(pc, label="dE/dt + DragPower (W)")

    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("Energy Balance Heatmap: (dE/dt + DragPower) binned")
    plt.show()
