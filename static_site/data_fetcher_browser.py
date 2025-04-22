import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import traceback

# In browser version, we use demo data instead of fetching from yfinance
# This allows the application to work in GitHub Pages without API dependencies

# Demo data for popular tickers
DEMO_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'SPY', 'QQQ']

def generate_demo_price_history(ticker, period="1y", interval="1d"):
    """Generate realistic demo price data for a ticker"""
    # Set seed based on ticker for consistent results
    seed = sum(ord(c) for c in ticker)
    np.random.seed(seed)
    
    # Calculate number of days based on period
    days_map = {
        "1mo": 22,  # trading days in a month
        "3mo": 66,
        "6mo": 126,
        "1y": 252,
        "2y": 504,
        "5y": 1260
    }
    days = days_map.get(period, 252)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # Base price varies by ticker (ensures different scales)
    ticker_base_prices = {
        'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 120.0, 'AMZN': 130.0,
        'META': 200.0, 'TSLA': 250.0, 'NVDA': 400.0, 'SPY': 450.0, 'QQQ': 350.0
    }
    base_price = ticker_base_prices.get(ticker, 100.0)
    
    # Generate prices with realistic patterns
    # Start with random walk
    random_walk = np.random.normal(0, 0.015, days).cumsum()
    
    # Add trend
    ticker_trends = {
        'AAPL': 0.2, 'MSFT': 0.25, 'GOOGL': 0.1, 'AMZN': 0.15,
        'META': 0.3, 'TSLA': -0.1, 'NVDA': 0.4, 'SPY': 0.1, 'QQQ': 0.15
    }
    trend = ticker_trends.get(ticker, 0.1)
    trend_component = np.linspace(0, trend, days)
    
    # Add seasonality (periodic pattern)
    seasonality = 0.05 * np.sin(np.linspace(0, 4*np.pi, days))
    
    # Combine components
    price_series = base_price * (1 + random_walk + trend_component + seasonality)
    
    # Generate daily components
    prices = pd.DataFrame(index=date_range)
    
    # Close price
    prices['Close'] = price_series
    
    # Open price (slight variation from previous close)
    open_noise = np.random.normal(0, 0.005, days)
    prices['Open'] = prices['Close'].shift(1) * (1 + open_noise)
    prices.loc[prices.index[0], 'Open'] = prices['Close'].iloc[0] * (1 - 0.002)
    
    # High and Low prices
    daily_volatility = np.random.uniform(0.01, 0.03, days)
    prices['High'] = prices.apply(lambda x: max(x['Open'], x['Close']) * (1 + np.random.uniform(0.001, daily_volatility[prices.index.get_loc(x.name)])), axis=1)
    prices['Low'] = prices.apply(lambda x: min(x['Open'], x['Close']) * (1 - np.random.uniform(0.001, daily_volatility[prices.index.get_loc(x.name)])), axis=1)
    
    # Volume (higher on volatile days)
    base_volume = {
        'AAPL': 80e6, 'MSFT': 30e6, 'GOOGL': 15e6, 'AMZN': 10e6,
        'META': 20e6, 'TSLA': 100e6, 'NVDA': 40e6, 'SPY': 120e6, 'QQQ': 80e6
    }.get(ticker, 10e6)
    
    vol_factor = 1 + np.abs(prices['Close'].pct_change())
    prices['Volume'] = base_volume * vol_factor.fillna(1) * np.random.uniform(0.5, 1.5, days)
    prices['Volume'] = prices['Volume'].astype(int)
    
    return prices

def generate_demo_options_chain(ticker, price_history):
    """Generate realistic demo options data"""
    if ticker not in DEMO_TICKERS:
        # Return empty data for non-demo tickers
        return {"expiry_dates": [], "options": {}}
    
    # Current stock price
    current_price = price_history['Close'].iloc[-1]
    
    # Generate expiration dates (every 30 days for 6 months)
    today = datetime.now()
    expiry_dates = []
    options_data = {"expiry_dates": [], "options": {}}
    
    for month in range(1, 7):
        expiry = (today + timedelta(days=30*month)).strftime('%Y-%m-%d')
        expiry_dates.append(expiry)
        options_data["expiry_dates"].append(expiry)
        
        # Generate options for this expiry
        calls = []
        puts = []
        
        # Days to expiry
        dte = 30 * month
        
        # Time value factor (longer expiry = higher time value)
        time_factor = (dte/365) ** 0.5
        
        # Generate strikes (±30% around current price)
        min_strike = current_price * 0.7
        max_strike = current_price * 1.3
        num_strikes = 15
        strikes = np.linspace(min_strike, max_strike, num_strikes)
        
        # Implied volatility based on ticker
        base_iv = {
            'AAPL': 0.25, 'MSFT': 0.22, 'GOOGL': 0.30, 'AMZN': 0.35,
            'META': 0.40, 'TSLA': 0.60, 'NVDA': 0.45, 'SPY': 0.15, 'QQQ': 0.18
        }.get(ticker, 0.3)
        
        # Generate options for each strike
        for strike in strikes:
            strike = round(strike, 2)
            
            # Adjust IV based on strike distance from current price (smile effect)
            moneyness = strike / current_price
            iv_adjustment = 0.1 * ((moneyness - 1) ** 2)  # Smile curve
            implied_vol = base_iv + iv_adjustment
            
            # Calculate option prices (simplified Black-Scholes approximation)
            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            
            # Time value component
            time_value_call = current_price * implied_vol * time_factor
            time_value_put = strike * implied_vol * time_factor
            
            call_price = max(0.01, intrinsic_call + time_value_call)
            put_price = max(0.01, intrinsic_put + time_value_put)
            
            # In the money flag
            call_itm = current_price > strike
            put_itm = current_price < strike
            
            # Generate some volume and open interest
            volume_base = np.random.randint(10, 1000)
            volume_factor = 2 if abs(moneyness - 1) < 0.05 else 1  # More volume near ATM
            volume = int(volume_base * volume_factor)
            open_interest = int(volume * (2 + np.random.random()))
            
            # Generate bid-ask spread
            bid_call = call_price * 0.95
            ask_call = call_price * 1.05
            bid_put = put_price * 0.95
            ask_put = put_price * 1.05
            
            # Add call option
            call = {
                "contractSymbol": f"{ticker}{expiry.replace('-','')}C{int(strike*100)}",
                "strike": strike,
                "lastPrice": round(call_price, 2),
                "bid": round(bid_call, 2),
                "ask": round(ask_call, 2),
                "volume": volume,
                "openInterest": open_interest,
                "impliedVolatility": implied_vol,
                "inTheMoney": call_itm
            }
            calls.append(call)
            
            # Add put option
            put = {
                "contractSymbol": f"{ticker}{expiry.replace('-','')}P{int(strike*100)}",
                "strike": strike,
                "lastPrice": round(put_price, 2),
                "bid": round(bid_put, 2),
                "ask": round(ask_put, 2),
                "volume": volume,
                "openInterest": open_interest,
                "impliedVolatility": implied_vol,
                "inTheMoney": put_itm
            }
            puts.append(put)
        
        # Add options to data
        options_data["options"][expiry] = {
            "calls": calls,
            "puts": puts
        }
    
    return options_data

def generate_demo_fundamentals(ticker):
    """Generate realistic demo fundamentals for a ticker"""
    # Default fundamentals
    fundamentals = {
        "symbol": ticker,
        "longName": ticker,
        "shortName": ticker,
        "sector": "Unknown",
        "industry": "Unknown",
        "marketCap": 0,
        "trailingPE": None,
        "forwardPE": None,
        "dividendYield": 0,
        "beta": 1,
        "fiftyTwoWeekHigh": 0,
        "fiftyTwoWeekLow": 0,
        "targetHighPrice": None,
        "targetLowPrice": None,
        "targetMeanPrice": None,
        "recommendationMean": None,
        "recommendationKey": "none",
        "lastDividendValue": 0,
        "lastDividendDate": None
    }
    
    # Realistic data for demo tickers
    ticker_fundamentals = {
        'AAPL': {
            "longName": "Apple Inc.",
            "shortName": "Apple",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 2_500_000_000_000,
            "trailingPE": 28.5,
            "forwardPE": 25.2,
            "dividendYield": 0.0055,
            "beta": 1.2,
            "recommendationKey": "buy"
        },
        'MSFT': {
            "longName": "Microsoft Corporation",
            "shortName": "Microsoft",
            "sector": "Technology",
            "industry": "Software—Infrastructure",
            "marketCap": 2_700_000_000_000,
            "trailingPE": 32.1,
            "forwardPE": 29.8,
            "dividendYield": 0.0082,
            "beta": 0.95,
            "recommendationKey": "buy"
        },
        'GOOGL': {
            "longName": "Alphabet Inc.",
            "shortName": "Alphabet",
            "sector": "Technology",
            "industry": "Internet Content & Information",
            "marketCap": 1_700_000_000_000,
            "trailingPE": 25.4,
            "forwardPE": 21.2,
            "dividendYield": 0.0,
            "beta": 1.1,
            "recommendationKey": "buy"
        },
        'AMZN': {
            "longName": "Amazon.com, Inc.",
            "shortName": "Amazon",
            "sector": "Consumer Cyclical",
            "industry": "Internet Retail",
            "marketCap": 1_400_000_000_000,
            "trailingPE": 42.3,
            "forwardPE": 35.7,
            "dividendYield": 0.0,
            "beta": 1.25,
            "recommendationKey": "buy"
        },
        'META': {
            "longName": "Meta Platforms, Inc.",
            "shortName": "Meta",
            "sector": "Technology",
            "industry": "Internet Content & Information",
            "marketCap": 800_000_000_000,
            "trailingPE": 28.7,
            "forwardPE": 22.5,
            "dividendYield": 0.005,
            "beta": 1.4,
            "recommendationKey": "buy"
        },
        'TSLA': {
            "longName": "Tesla, Inc.",
            "shortName": "Tesla",
            "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers",
            "marketCap": 700_000_000_000,
            "trailingPE": 60.2,
            "forwardPE": 48.5,
            "dividendYield": 0.0,
            "beta": 2.0,
            "recommendationKey": "hold"
        },
        'NVDA': {
            "longName": "NVIDIA Corporation",
            "shortName": "NVIDIA",
            "sector": "Technology",
            "industry": "Semiconductors",
            "marketCap": 1_100_000_000_000,
            "trailingPE": 75.3,
            "forwardPE": 40.2,
            "dividendYield": 0.0025,
            "beta": 1.7,
            "recommendationKey": "buy"
        },
        'SPY': {
            "longName": "SPDR S&P 500 ETF Trust",
            "shortName": "S&P 500 ETF",
            "sector": "Financial Services",
            "industry": "Exchange Traded Fund",
            "marketCap": None,
            "trailingPE": None,
            "forwardPE": None,
            "dividendYield": 0.015,
            "beta": 1.0,
            "recommendationKey": "none"
        },
        'QQQ': {
            "longName": "Invesco QQQ Trust",
            "shortName": "Nasdaq 100 ETF",
            "sector": "Financial Services",
            "industry": "Exchange Traded Fund",
            "marketCap": None,
            "trailingPE": None,
            "forwardPE": None,
            "dividendYield": 0.005,
            "beta": 1.1,
            "recommendationKey": "none"
        }
    }
    
    # Update with known fundamentals if available
    if ticker in ticker_fundamentals:
        fundamentals.update(ticker_fundamentals[ticker])
    
    return fundamentals

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
    """Get historical price data using demo data for browser compatibility"""
    try:
        # Generate demo data
        data = generate_demo_price_history(ticker, period=period, interval=interval)
        
        # Check if data is empty
        if data.empty:
            raise ValueError(f"No price history data available for {ticker}")
        
        return data
    except Exception as e:
        # Create a warning message for the user
        print(f"Warning: Using demo data for {ticker} due to browser limitations")
        
        # Use default ticker if the requested one fails
        fallback_ticker = 'SPY'
        return generate_demo_price_history(fallback_ticker, period=period, interval=interval)

@exponential_backoff(max_retries=3)
def get_options_chain(ticker, force_refresh=False):
    """Get options chain data using demo data for browser compatibility"""
    try:
        # Get price history first (needed for generating options)
        price_history = get_price_history(ticker)
        
        # Generate demo options data
        options_data = generate_demo_options_chain(ticker, price_history)
        
        # If we found any options, return
        if options_data["expiry_dates"]:
            return options_data
        
        # If ticker not in demo list, try to normalize it
        upper_ticker = ticker.upper()
        for demo_ticker in DEMO_TICKERS:
            if demo_ticker in upper_ticker or upper_ticker in demo_ticker:
                # Use a similar demo ticker
                return generate_demo_options_chain(demo_ticker, price_history)
        
        # Return empty structure if no match found
        return {"expiry_dates": [], "options": {}}
    except Exception as e:
        print(f"Warning: Unable to get options data for {ticker}: {str(e)}")
        # Return empty structure
        return {"expiry_dates": [], "options": {}}

@exponential_backoff()
def get_fundamentals(ticker, force_refresh=False):
    """Get company fundamentals using demo data for browser compatibility"""
    try:
        # Generate demo fundamentals
        fundamentals = generate_demo_fundamentals(ticker)
        return fundamentals
    except Exception as e:
        print(f"Warning: Using generic fundamentals for {ticker} due to browser limitations")
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