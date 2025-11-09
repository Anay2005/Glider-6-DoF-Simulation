#!/usr/bin/env python3
"""
ml_model.py

Trains a ML model (RandomForest) to predict final glide distance from wind.
Now includes an option to compute ensemble predictions for each test sample,
allowing us to plot error bars (std dev) in our new ML plots.
"""

import numpy as np
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def _collect_ensemble_predictions(model, X):
    """
    Utility to collect predictions from each estimator in the random forest
    for each sample in X. This returns an array of shape [n_estimators, n_samples].
    """
    # model.estimators_ is a list of the individual decision trees
    all_preds = []
    for estimator in model.estimators_:
        p = estimator.predict(X)
        all_preds.append(p)
    return np.array(all_preds)  # shape (n_estimators, len(X))

def train_glide_distance_model(csv_path="monte_carlo_results.csv"):
    """
    Reads the Monte Carlo CSV, trains a RandomForestRegressor, 
    and returns:
       model, data_splits, ensemble_preds_test

    data_splits = (X_train, y_train, X_test, y_test)
    ensemble_preds_test = (preds_per_estimator, ) # shape (n_estimators, len(X_test))
    """
    # Read CSV
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist_str = row["distance_m"]
            if dist_str not in [None, "None", ""]:
                wU = float(row["wind_u"])
                wV = float(row["wind_v"])
                wW = float(row["wind_w"])
                dist = float(dist_str)
                data.append([wU, wV, wW, dist])

    data = np.array(data)
    if data.shape[0] == 0:
        raise ValueError(f"No valid data found in {csv_path}.")

    X = data[:, 0:3]  # wind_u, wind_v, wind_w
    y = data[:, 3]    # final distance

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        random_state=42)

    # Train
    model = RandomForestRegressor(n_estimators=100,
                                  max_depth=None,
                                  random_state=0)
    model.fit(X_train, y_train)

    # 4. Evaluate
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    r2  = r2_score(y_test, pred_test)
    print("[ML] Regression model trained on wind -> distance")
    print("    Test MAE:", mae)
    print("    Test R^2:", r2)
    importances = model.feature_importances_
    print("    Feature importances (U, V, W):", importances)

    # 5. Gather ensemble predictions for the test set
    preds_ensemble_test = _collect_ensemble_predictions(model, X_test)
    # shape => (n_estimators, n_test)
    # We'll return that so we can do error bars.

    # 6. Return
    data_splits = (X_train, y_train, X_test, y_test)
    return model, data_splits, preds_ensemble_test
