import time
import random
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import traceback

# In stlite we don't use caching to disk as browser storage is limited
# but we define the structure to maintain compatibility

def exponential_backoff(max_retries=3, base_delay=1.0):
    """Decorator for exponential backoff retry logic"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    
                    # Calculate delay with jitter
                    delay = base_delay * (2 ** (retries - 1)) + random.uniform(0, 0.5)
                    time.sleep(delay)
            return func(*args, **kwargs)  # One last try
        return wrapper
    return decorator

@exponential_backoff()
def get_price_history(ticker, period="1y", interval="1d", force_refresh=False):
    """Get historical price data - in stlite version we always fetch fresh data"""
    # Fetch data from yfinance
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    
    # Check if data is empty
    if data.empty:
        raise ValueError(f"No price history data available for {ticker}")
    
    return data

@exponential_backoff(max_retries=3)
def get_options_chain(ticker, force_refresh=False):
    """Get options chain data"""
    # Try multiple tickers - some stocks may have different ticker symbols for options
    alternative_tickers = [ticker]
    
    # Add common variations for indices
    if ticker.startswith('^'):
        alternative_tickers.append(ticker[1:])  # Try without the ^
        alternative_tickers.append(f"{ticker[1:]}X")  # Some indices use X suffix
    
    # For SPY, QQQ, try the related index
    if ticker == 'SPY':
        alternative_tickers.append('^SPX')
        alternative_tickers.append('^GSPC')
    elif ticker == 'QQQ':
        alternative_tickers.append('^NDX')
        alternative_tickers.append('^IXIC')
    
    # For common symbols, try alternative forms
    special_mappings = {
        'AAPL': ['AAPL'],  # Apple often has good options data
        'MSFT': ['MSFT'],  # Microsoft often has good options data
        'GOOGL': ['GOOG', 'GOOGL'],  # Try both Google tickers
        'AMZN': ['AMZN'],  # Amazon often has good options data
    }
    
    if ticker in special_mappings:
        alternative_tickers = special_mappings[ticker]
    
    for attempt_ticker in alternative_tickers:
        try:
            # Fetch data from yfinance
            stock = yf.Ticker(attempt_ticker)
            expirations = stock.options
            
            if not expirations:
                continue
            
            # Prepare container for all options data
            options_data = {
                "expiry_dates": [],
                "options": {}
            }
            
            # Get data for each expiration date
            for expiry in expirations:
                try:
                    option_chain = stock.option_chain(expiry)
                    
                    # Skip if no calls or puts
                    if option_chain.calls.empty and option_chain.puts.empty:
                        continue
                    
                    # Add this expiry date to our list
                    options_data["expiry_dates"].append(expiry)
                    
                    # Process calls
                    calls_records = []
                    for record in option_chain.calls.to_dict("records"):
                        for key, value in list(record.items()):
                            if isinstance(value, pd.Timestamp):
                                record[key] = value.strftime('%Y-%m-%d')
                        calls_records.append(record)
                    
                    # Process puts
                    puts_records = []
                    for record in option_chain.puts.to_dict("records"):
                        for key, value in list(record.items()):
                            if isinstance(value, pd.Timestamp):
                                record[key] = value.strftime('%Y-%m-%d')
                        puts_records.append(record)
                    
                    # Add both calls and puts for this expiry date
                    options_data["options"][expiry] = {
                        "calls": calls_records,
                        "puts": puts_records
                    }
                    
                except Exception:
                    continue
            
            # If we found any options, return
            if options_data["expiry_dates"]:
                return options_data
        
        except Exception:
            pass
    
    # Return empty structure
    return {"expiry_dates": [], "options": {}}

@exponential_backoff()
def get_fundamentals(ticker, force_refresh=False):
    """Get company fundamentals - in stlite we always fetch fresh data"""
    # Fetch data from yfinance
    stock = yf.Ticker(ticker)
    
    try:
        # Get relevant info fields
        info = stock.info
        
        # Extract only the fields we need
        fundamentals = {
            "symbol": ticker,
            "longName": info.get("longName", ticker),
            "shortName": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "marketCap": info.get("marketCap", 0),
            "trailingPE": info.get("trailingPE", None),
            "forwardPE": info.get("forwardPE", None),
            "dividendYield": info.get("dividendYield", 0),
            "beta": info.get("beta", 1),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0),
            "targetHighPrice": info.get("targetHighPrice", None),
            "targetLowPrice": info.get("targetLowPrice", None),
            "targetMeanPrice": info.get("targetMeanPrice", None),
            "recommendationMean": info.get("recommendationMean", None),
            "recommendationKey": info.get("recommendationKey", "none"),
            "lastDividendValue": info.get("lastDividendValue", 0),
            "lastDividendDate": info.get("lastDividendDate", None)
        }
        
        return fundamentals
    
    except Exception as e:
        # Return minimum data to avoid errors downstream
        return {
            "symbol": ticker,
            "longName": ticker,
            "shortName": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "marketCap": 0,
        }

def get_historical_volatility(price_history, window=20):
    """Calculate historical volatility from price data"""
    # Calculate daily returns
    returns = price_history['Close'].pct_change().dropna()
    
    # Calculate rolling standard deviation and annualize
    hist_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    return hist_vol