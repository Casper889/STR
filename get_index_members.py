import pdblp
import pandas as pd

def transform_index_data(df):
    """
    Transforms the DataFrame output from Bloomberg into a clean format with 
    constituent tickers as the index and percentage weights as the values.

    Parameters:
        - df: The original DataFrame with columns 'name' and 'value'.

    Returns:
        - A pandas Series with tickers as the index and weights as values.
    """
    # Filter rows for Member Ticker and Exchange Code
    tickers = df[df["name"] == "Member Ticker and Exchange Code"]["value"].reset_index(drop=True)
    
    # Filter rows for Percentage Weight
    weights = df[df["name"] == "Percentage Weight"]["value"].astype(float).reset_index(drop=True)
    
    # Combine into a pandas Series
    series = pd.Series(data=weights.values, index=tickers.values, name="Weight (%)")
    return series

def get_index_members(index_ticker: str, override_date: str) -> pd.DataFrame:
    """
    Fetches index members and their weights for a specific date using Bloomberg's pdblp.

    Parameters:
        - index_ticker: Bloomberg ticker for the index (e.g., "RIY Index").
        - override_date: The date to override in 'YYYYMMDD' format.

    Returns:
        - A pandas DataFrame with index members and their weights.
    """
    # Connect to Bloomberg
    con = pdblp.BCon(debug=False, port=8194, timeout=50000)
    con.start()

    # Fetch index members and weights using `bds`
    members = con.bulkref(index_ticker, "INDX_MWEIGHT", [('END_DATE_OVERRIDE', override_date)])

    if members.empty:
        print(f"No data found for {index_ticker} on {override_date}")
        return pd.DataFrame()

    # Rename columns for clarity
    members.rename(
        columns={"Member Ticker and Exchange Code": "Constituent", "Weight": "Weight (%)"}, inplace=True
    )

    transform_index_data(members)

    return members


