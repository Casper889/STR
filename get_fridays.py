import pandas as pd
from datetime import timedelta

def get_last_fridays_for_years(start_year, end_year):
    """
    Calculates the second-to-last and third-to-last Fridays of each quarter for a range of years.

    Parameters:
        - start_year (int): The starting year.
        - end_year (int): The ending year (inclusive).

    Returns:
        - A DataFrame with the second-to-last and third-to-last Fridays for each quarter across all years.
    """
    results = []

    for year in range(start_year, end_year + 1):
        quarters = {
            "Q1": pd.Timestamp(f"{year}-03-31"),
            "Q2": pd.Timestamp(f"{year}-06-30"),
            "Q3": pd.Timestamp(f"{year}-09-30"),
            "Q4": pd.Timestamp(f"{year}-12-31")
        }

        for quarter, end_date in quarters.items():
            # Find the last Friday of the quarter
            last_friday = end_date - timedelta(days=(end_date.weekday() - 4) % 7)
            
            # Calculate second-to-last and third-to-last Fridays
            second_to_last_friday = last_friday - timedelta(weeks=1)
            third_to_last_friday = last_friday - timedelta(weeks=2)
            third_to_last_thursday = third_to_last_friday - timedelta(days=1)

            # Calculate the previous quarter's end date
            if quarter == "Q1":
                prev_end_date = pd.Timestamp(f"{year - 1}-12-31")
            elif quarter == "Q2":
                prev_end_date = pd.Timestamp(f"{year}-03-31")
            elif quarter == "Q3":
                prev_end_date = pd.Timestamp(f"{year}-06-30")
            else:  # Q4
                prev_end_date = pd.Timestamp(f"{year}-09-30")
            
            prev_last_friday = prev_end_date - timedelta(days=(prev_end_date.weekday() - 4) % 7)
            second_to_last_friday_previous = prev_last_friday - timedelta(weeks=1)

            # Append results
            results.append({
                "Year": year,
                "Quarter": quarter,
                "Second-to-Last Friday": second_to_last_friday,
                "Third-to-Last Friday": third_to_last_friday,
                "Third-to-Last Thursday": third_to_last_thursday,
                "Second-to-last-friday previous" :  second_to_last_friday_previous
            })

    return pd.DataFrame(results)