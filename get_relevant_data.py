import pdblp
import pandas as pd

def reformat_dataframe(df):
    """
    Reformats the DataFrame so that the index is `ticker` and each field is a separate column.

    Parameters:
        - df: Original DataFrame with `ticker`, `field`, and `value` columns.

    Returns:
        - Reformatted DataFrame.
    """
    # Pivot the DataFrame
    reformatted_df = df.pivot(index="ticker", columns="field", values="value")
    
    # Optional: Rename columns for better readability
    reformatted_df.columns = [
        col.replace(" ", "_") for col in reformatted_df.columns
    ]

    return reformatted_df

def fetch_historical_data(tickers:list, fields:list, overrides:list):
    """
    Fetches historical data for a list of tickers on a specific date using Bloomberg.

    Parameters:
        - tickers: List of Bloomberg tickers (index of the pandas Series).
        - date: Historical date in 'YYYYMMDD' format.

    Returns:
        - A pandas DataFrame with additional historical data for each ticker.
    """
    # Connect to Bloomberg
    con = pdblp.BCon(debug=False, port=8194, timeout=50000)
    con.start()

    # Fetch data for all tickers on the specific date
    data = con.ref(
        [ticker +' Equity' for ticker in tickers],
        fields,
        overrides
    )

    return reformat_dataframe(data)

    