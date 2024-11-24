import pandas as pd
from get_fridays import get_last_fridays_for_years  # From the uploaded get_fridays.py file
from get_index_members import get_index_members, transform_index_data  # From the uploaded get_index_members.py file
from get_relevant_data import fetch_historical_data  # From the uploaded get_relevant_data.py file
from optimize_portfolio import optimize_portfolio
from backtest import backtest_strategy
from tqdm import tqdm
import os
import pickle
from datetime import timedelta
import pdblp
import glob
import itertools
import numpy as np


class STRRunner:
    def __init__(self, start_year, end_year, index_ticker):
        """
        Initializes the IndexScheduleManager with a year range and an index ticker.

        Parameters:
            - start_year (int): The starting year.
            - end_year (int): The ending year.
            - index_ticker (str): Bloomberg ticker for the index (e.g., "SPX Index").
        """
        self.start_year = start_year
        self.end_year = end_year
        self.index_ticker = index_ticker
        self.schedule = None
        self.constituents = {}
        self.constituent_data = {}
        self.prices = {}
        self.returns = {}
        self.optimized_portfolios = {}
        self.solutions = {}
        self.cumulative_strategy_returns = {}
        self.strategy_returns = {}
        print("This script pulls data from Bloomberg and saves it. Using this script often may result in hitting your data limit.\n When you are done using the script, make sure to delete any files saved. As we are not allowed to store Bloomberg data.")

        # Create a folder with the name of self.index_ticker if it doesn't exist
        if not os.path.exists(self.index_ticker):
            os.makedirs(self.index_ticker)

    def generate_schedule(self):
        """
        Generates a schedule containing third-to-last Fridays, second-to-last Fridays, 
        and third-to-last Thursdays for the given range of years.
        """
        self.schedule = get_last_fridays_for_years(self.start_year, self.end_year)

    def retrieve_index_constituents(self):
        """
        Retrieves the index constituents for each third-to-last Thursday in the schedule.
        Stores the results in the `constituent_data` attribute.
        """
        if self.schedule is None:
            raise ValueError("Schedule is not generated. Call `generate_schedule` first.")
        
        file_path = f"{self.index_ticker}/constituents.csv"
        if os.path.exists(file_path):
            print(f"Found a csv at {file_path} not pulling again as you will DEFINITELY run out of data at some point. If you want to change. Run the reset() method")
            self.constituents = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:

            for _, row in tqdm(self.schedule.iterrows()):
                date = row["Third-to-Last Thursday"].strftime('%Y%m%d')
                constituents = get_index_members(self.index_ticker, date)
                self.constituents[pd.to_datetime(date)] = transform_index_data(constituents)
            
            self.constituents = pd.DataFrame(self.constituents).T
            self.constituents.to_csv(file_path)

        

    def fetch_relevant_data(self):
        """
        Fetches relevant data for the index constituents for each third-to-last Thursday.
        Updates the `constituent_data` with the fetched data.
        """
        if type(self.constituents) != pd.DataFrame:
            raise ValueError("No constituent data found. Call `retrieve_index_constituents` first.")
        
        file_path = f"{self.index_ticker}/constituent_data.pkl"
        if os.path.exists(file_path):
            print(f"Found a pickle at {file_path} not pulling again as you will DEFINITELY run out of data at some point. If you want to change. Run the reset() method")
            with open(file_path, 'rb') as file:
                self.constituent_data = pickle.load(file)
        else:
            for date, constituents in tqdm(self.constituents.iterrows()):
                if constituents.empty:
                    continue
                tickers = constituents.index.tolist()

                # Define the fields to fetch
                fields = [
                    "BETA_RAW_OVERRIDABLE",           # 3-month total return
                ]

                # Overrides for custom start and end dates
                overrides = [
                    ("BETA_OVERRIDE_END_DT", date.strftime('%Y%m%d')),    # End date
                    ("BETA_OVERRIDE_START_DT", (date - timedelta(days=756)).strftime('%Y%m%d'))
                ]
                df = fetch_historical_data(tickers, fields, overrides)

                fields = "TURNOVER"

                # Overrides for custom start and end dates
                overrides = [
                    ("MARKET_DATA_OVERRIDE", "TURNOVER"),  # Start date
                    ("END_DATE_OVERRIDE", date.strftime('%Y%m%d')),
                    ("CALC_INTERVAL", "3M")    # End date
                ]

                df = df.merge(fetch_historical_data(tickers, fields, overrides), left_index=True, right_index=True)

                fields = "CUST_TRR_RETURN_HOLDING_PER"

                # Overrides for custom start and end dates
                overrides = [
                    ("CUST_TRR_START_DT", self.schedule[self.schedule['Third-to-Last Thursday'] == date]['Second-to-last-friday previous'].iloc[0].strftime('%Y%m%d')),  # Start date
                    ("CUST_TRR_END_DT", date.strftime('%Y%m%d'))    # End date
                ]

                df = df.merge(fetch_historical_data(tickers, fields, overrides), left_index=True, right_index=True)

                fields = "GICS_SECTOR_NAME"

                # Overrides for custom start and end dates
                overrides = [
                    ("END_DATE_OVERRIDE", date.strftime('%Y%m%d')),  # Start date
                ]

                df = df.merge(fetch_historical_data(tickers, fields, overrides), left_index=True, right_index=True)

                self.constituent_data[pd.to_datetime(date)] = df

            with open(f'{self.index_ticker}/constituent_data.pkl', 'wb') as file:
                pickle.dump(self.constituent_data, file)

    def fetch_prices(self):

        if type(self.constituents) != pd.DataFrame:
            raise ValueError("No constituent data found. Call `retrieve_index_constituents` first.")

        file_path = f"{self.index_ticker}/prices.csv"
        if os.path.exists(file_path):
            print(f"Found a csv at {file_path} not pulling again as you will DEFINITELY run out of data at some point. If you want to change. Run the reset() method")
            self.prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.returns = self.prices.pct_change()
        else:
            # Initialize Bloomberg connection
            con = pdblp.BCon(debug=False, port=8194, timeout=5000)
            con.start()

            # Prepare tickers and fields
            tickers = self.constituents.columns
            field = "PX_LAST"  # Fetch adjusted closing prices (gross total return)
            
            # Define the date range
            start_date = f"{self.start_year}0101"  # Start of self.start_year
            end_date = f"{self.end_year}1231"      # End of self.end_year

            # Fetch data
            try:
                price_data = con.bdh([ticker + ' Equity' for ticker in tickers], field, start_date, end_date)
                price_data.columns = price_data.columns.levels[0]
            except:
                price_data = pd.DataFrame()
                for ticker in tickers:
                    ticker = ticker + ' Equity'
                    try:
                        price_data_temp = con.bdh([ticker + ' Equity' for ticker in tickers], field, start_date, end_date)
                        price_data.merge(price_data_temp, left_index=True, right_index=True, how="outer")
                    except Exception as e:
                        print(e)
                        continue

            # Process fetched data (optional: align to your constituents DataFrame)
            self.prices = price_data.ffill()  # Drop rows where all values are NaN
            print("Price data fetched successfully.")

            self.prices.to_csv(file_path)
            self.returns = self.prices.pct_change()

    def create_portfolios(self, beta_range=(-0.01, 0.01), sector_range=(-0.01, 0.01), max_weight = 0.01, log=True):

        if not self.constituent_data:
            raise ValueError("No constituent data found. Call `retrieve_index_constituents` first.")
        
        for key in tqdm(self.constituent_data.keys()):
            date = self.schedule[self.schedule['Third-to-Last Thursday'] == key]['Third-to-Last Friday'].iloc[0]
            try:
                self.optimized_portfolios[date], self.solutions[date] = optimize_portfolio(self.constituent_data[key], beta_range, sector_range, max_weight, log)
            except Exception as e:
                print(f"Failed for {key}. Potentially this is a future date {e}")

        with open(f'{self.index_ticker}/optimized_portfolios.pkl', 'wb') as file:
            pickle.dump(self.optimized_portfolios, file)

    def run_backtest(self):

        if not self.optimized_portfolios:
            raise ValueError("No portfolios found. Call `create_portfolios` first.")
        
        strategy_returns = backtest_strategy(
            self.optimized_portfolios, self.schedule, self.returns
        )

        self.cumulative_strategy_returns = strategy_returns['Cumulative Returns']
        self.strategy_returns = strategy_returns['Portfolio Returns']

    def reset(self):
        """
        Removes all files from the folder associated with self.index_ticker.
        """
        folder_path = f"{self.index_ticker}"  # Path to the folder

        # Check if the folder exists
        if os.path.exists(folder_path):
            # Use glob to find all files in the folder
            files = glob.glob(os.path.join(folder_path, "*"))
            for file in files:
                try:
                    os.remove(file)  # Remove each file
                    print(f"Removed: {file}")
                except Exception as e:
                    print(f"Error removing file {file}: {e}")
            print(f"All files in {folder_path} have been removed.")
        else:
            print(f"The folder {folder_path} does not exist.")

    def grid_search(self, beta_limits, sector_limits, max_weights, log, start):
        """
        Perform a grid search to find the best combination of beta_limit, sector_limit, and max_weight.

        Parameters:
        - spx: The STRRunner instance.
        - beta_limits: List of beta limit values to test.
        - sector_limits: List of sector limit values to test.
        - max_weights: List of max weight values to test.

        Returns:
        - results: A DataFrame containing all combinations and their corresponding final cumulative returns.
        """
        # Store results
        results = []

        # Generate all combinations of the parameters
        param_combinations = itertools.product(beta_limits, sector_limits, max_weights, log)

        for beta_limit, sector_limit, max_weight, log in param_combinations:
            print(f"Testing: beta_limit={beta_limit}, sector_limit={sector_limit}, max_weight={max_weight}, log={log}")

            # Run the portfolio creation and backtest
            self.create_portfolios(
                beta_range=(-beta_limit, beta_limit),
                sector_range=(-sector_limit, sector_limit),
                max_weight=max_weight,
                log=log
            )
            self.run_backtest()

            # Get the final cumulative return
            final_cumulative_return = self.strategy_returns.loc[start:].add(1).cumprod().iloc[-1]
            sharpe = self.strategy_returns[start:].mean()*252 / (self.strategy_returns[start:].std() * np.sqrt(252))
            roll_max = self.strategy_returns[start:].add(1).cumprod().cummax()
            daily_drawdown = self.strategy_returns[start:].add(1).cumprod() / roll_max - 1
            max_dd = daily_drawdown.cummin().iloc[-1]
            print(f"Result: {sharpe}")

            # Store the results
            results.append({
                "beta_limit": beta_limit,
                "sector_limit": sector_limit,
                "max_weight": max_weight,
                "log" : log,
                "final_cumulative_return": final_cumulative_return,
                "sharpe" : sharpe,
                "max_dd" : max_dd
            })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        return results_df

  

        


