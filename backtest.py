import pandas as pd

def backtest_strategy(optimized_portfolios, schedule, daily_returns):
    """
    Backtest the strategy: Take positions on the third-to-last Friday and liquidate them on the second-to-last Friday.

    Parameters:
    - optimized_portfolios: dict of DataFrames, keyed by third-to-last Friday dates, each containing optimized weights.
    - schedule: DataFrame with schedule dates for each quarter (third-to-last Friday and second-to-last Friday).
    - daily_returns: DataFrame of daily returns for all assets.

    Returns:
    - strategy_returns: DataFrame of daily strategy returns.
    """
    # Initialize an empty DataFrame to store strategy returns
    strategy_returns = pd.DataFrame(index=daily_returns.index)
    strategy_returns["Portfolio Returns"] = 0.0
    
    for _, row in schedule.iterrows():
        third_to_last = row['Third-to-Last Friday']
        second_to_last = row['Second-to-Last Friday']

        # Ensure the third_to_last date exists in the optimized_portfolios
        if third_to_last not in optimized_portfolios:
            continue

        # Get the portfolio weights for the third-to-last Friday
        portfolio_weights = optimized_portfolios[third_to_last]['Optimized Weight']

        # Slice the daily returns for the holding period
        holding_period_returns = daily_returns.loc[third_to_last:second_to_last]

        # Calculate daily portfolio returns during the holding period
        portfolio_returns = (holding_period_returns * portfolio_weights).sum(axis=1)

        # Add the portfolio returns to the strategy returns DataFrame
        strategy_returns.loc[third_to_last:second_to_last, 'Portfolio Returns'] = portfolio_returns.astype(float)

    # Fill missing values with 0 (no position outside the holding periods)
    strategy_returns = strategy_returns.fillna(0.0)

    # Calculate cumulative strategy returns
    strategy_returns['Cumulative Returns'] = (1 + strategy_returns['Portfolio Returns']).cumprod().astype(float)

    return strategy_returns
