  
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.stats import zscore


def optimize_portfolio(data, beta_target_range=(-0.01, 0.01), sector_exposure_target=(-0.01, 0.01), max_weight = 0.01, log=True):
    """
    Optimizes a portfolio to minimize CURRENT_TRR_3MO exposure while satisfying constraints:
    - Net EQY_BETA exposure between -0.05 and 0.05
    - Net GICS_SECTOR_NAME exposure between -0.05 and 0.05
    - Separate sum of long and short weights equals 1
    - Weight limits: ±3% max
    - Longs and shorts are mutually exclusive for each asset.

    Returns:
        - A DataFrame with the optimized weights for each asset.
    """

    # Preprocess data: Ensure no NaNs in required columns
    data = data.dropna(subset=["CUST_TRR_RETURN_HOLDING_PER", "BETA_RAW_OVERRIDABLE", "GICS_SECTOR_NAME", "TURNOVER"]).copy()

    if log == True:
        data["LOG_ADV"] = np.log(data["TURNOVER"].replace(0, np.nan).astype(float)).replace(0, np.nan)
    else:
        data["LOG_ADV"] = (data["TURNOVER"].replace(0, np.nan).astype(float)).replace(0, np.nan)

    # Scale data for better optimization
    data["SIGNAL"] = data["CUST_TRR_RETURN_HOLDING_PER"] / data["LOG_ADV"]

    mean_signal = data['SIGNAL'].mean()
    std_signal = data['SIGNAL'].std()

    data['Z_SCORE'] = (data['SIGNAL'] - mean_signal) / std_signal

    # Create unique sector dummies for constraints
    sector_dummies = pd.get_dummies(data["GICS_SECTOR_NAME"])

    # Variables
    n_assets = len(data)
    longs = cp.Variable(n_assets, nonneg=True)  # Long weights
    shorts = cp.Variable(n_assets, nonneg=True)  # Short weights
    z = cp.Variable(n_assets, boolean=True)  # Binary variable for long/short exclusivity
    weights = longs - shorts  # Net weights

    # Objective: Minimize exposure to CURRENT_TRR_3MO
    objective = cp.Minimize(weights @ data["Z_SCORE"].values)

    # Constraints
    constraints = []

    # Long and short sums
    constraints += [
        cp.sum(longs) == 1,  # Sum of longs equals 1
        cp.sum(shorts) == 1  # Sum of shorts equals 1
    ]

    # Long/short exclusivity
    max_weight = max_weight
    constraints += [
        longs <= max_weight * z,  # Long weight active only if z[i] = 1
        shorts <= max_weight * (1 - z),  # Short weight active only if z[i] = 0
    ]

    # Net EQY_BETA exposure between -0.05 and 0.05
    beta_exposure = data["BETA_RAW_OVERRIDABLE"].values @ weights
    constraints += [
        beta_exposure >= beta_target_range[0],
        beta_exposure <= beta_target_range[1]
    ]

    # Sector exposures within the specified range
    for sector in sector_dummies.columns:
        sector_exposure = sector_dummies[sector].values @ weights
        constraints += [
            sector_exposure >= sector_exposure_target[0],
            sector_exposure <= sector_exposure_target[1]
        ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCIPY, verbose=False)  # Use a mixed-integer solver like ECOS_BB
    solution = problem.value

    # Check if the problem is solved successfully
    if problem.status != cp.OPTIMAL:
        raise ValueError(f"Optimization failed. Status: {problem.status}")
    
    # Output results
    data["Optimized Weight"] = longs.value - shorts.value
    return data[["Optimized Weight", "CUST_TRR_RETURN_HOLDING_PER", "BETA_RAW_OVERRIDABLE", "GICS_SECTOR_NAME", "TURNOVER"]], solution