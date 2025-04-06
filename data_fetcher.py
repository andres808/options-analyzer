import os
import time
import random
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import traceback

# Define cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache expiration times (in seconds)
CACHE_EXPIRY = {
    "price_history": 3600,  # 1 hour
    "options_chain": 3600,  # 1 hour
    "fundamentals": 86400,  # 24 hours
}

def get_cache_path(ticker, data_type, ext="parquet"):
    """Generate standardized cache file path"""
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.{ext}")

def is_cache_valid(cache_path, expiry_seconds=3600):
    """Check if cache file exists and is not expired"""
    if not os.path.exists(cache_path):
        return False
    
    file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    time_since_modified = (datetime.now() - file_modified_time).total_seconds()
    
    return time_since_modified < expiry_seconds

def exponential_backoff(max_retries=5, base_delay=1.0):
    """Decorator for exponential backoff retry logic"""
    max_retries = int(os.getenv("YF_MAX_RETRIES", max_retries))
    
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
    """Get historical price data with caching"""
    cache_path = get_cache_path(ticker, f"history_{period}_{interval}")
    
    if not force_refresh and is_cache_valid(cache_path, CACHE_EXPIRY["price_history"]):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            # If there's an error reading the cache, proceed to fetch new data
            pass
    
    # Fetch data from yfinance
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    
    # Check if data is empty
    if data.empty:
        raise ValueError(f"No price history data available for {ticker}")
    
    # Save to cache
    data.to_parquet(cache_path)
    
    return data

@exponential_backoff()
def get_options_chain(ticker, force_refresh=False):
    """Get options chain data with caching"""
    cache_path = get_cache_path(ticker, "options_chain", ext="json")
    
    if not force_refresh and is_cache_valid(cache_path, CACHE_EXPIRY["options_chain"]):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            # If there's an error reading the cache, proceed to fetch new data
            pass
    
    # Fetch data from yfinance
    stock = yf.Ticker(ticker)
    
    try:
        expirations = stock.options
        
        if not expirations:
            return {"expiry_dates": [], "options": {}}
        
        # Prepare container for all options data
        options_data = {
            "expiry_dates": [],
            "options": {}
        }
        
        # Get data for each expiration date
        for expiry in expirations:
            options_data["expiry_dates"].append(expiry)
            
            # Get calls and puts
            calls = stock.option_chain(expiry).calls
            puts = stock.option_chain(expiry).puts
            
            # Convert DataFrames to dictionaries for JSON serialization
            options_data["options"][expiry] = {
                "calls": calls.to_dict("records"),
                "puts": puts.to_dict("records")
            }
        
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(options_data, f)
        
        return options_data
    
    except Exception as e:
        print(f"Error fetching options data for {ticker}: {str(e)}")
        traceback.print_exc()
        return {"expiry_dates": [], "options": {}}

@exponential_backoff()
def get_fundamentals(ticker, force_refresh=False):
    """Get company fundamentals with caching"""
    cache_path = get_cache_path(ticker, "fundamentals", ext="json")
    
    if not force_refresh and is_cache_valid(cache_path, CACHE_EXPIRY["fundamentals"]):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            # If there's an error reading the cache, proceed to fetch new data
            pass
    
    # Fetch data from yfinance
    stock = yf.Ticker(ticker)
    
    try:
        # Get relevant info fields
        info = stock.info
        
        # Extract only the fields we need to reduce cache size
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
        
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(fundamentals, f)
        
        return fundamentals
    
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {str(e)}")
        traceback.print_exc()
        
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
