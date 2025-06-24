import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):
    """
    Load stock data from Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL' for Apple)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing stock data
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None