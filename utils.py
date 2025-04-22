import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import time
import random
import traceback
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Import custom modules
# Using relative imports to avoid circular dependencies
# These imports are executed when the function is called
scheduler = None

def initialize_scheduler():
    """Initialize the background scheduler for cache refresh and model retraining"""
    global scheduler
    
    # Check if scheduler already exists and is running
    if scheduler is not None and scheduler.running:
        logger.info("Scheduler already running, not initializing again")
        return
        
    # Create new scheduler or reuse existing one
    if scheduler is None:
        scheduler = BackgroundScheduler()
    
    # Try to remove existing jobs first to avoid conflicts
    try:
        scheduler.remove_job('cache_refresh')
        logger.info("Removed existing cache_refresh job")
    except:
        pass
        
    try:
        scheduler.remove_job('model_check')
        logger.info("Removed existing model_check job")
    except:
        pass
    
    # Add scheduled jobs with new IDs to avoid conflicts
    job_id_suffix = str(int(time.time()))
    
    # Add job for cache refresh
    scheduler.add_job(
        refresh_cache_for_popular_tickers,
        'interval',
        hours=1,
        id=f'cache_refresh_{job_id_suffix}'
    )
    logger.info(f"Added cache refresh job with ID: cache_refresh_{job_id_suffix}")
    
    # Add job for model checking
    scheduler.add_job(
        check_model_drift,
        'interval',
        days=1,
        id=f'model_check_{job_id_suffix}'
    )
    logger.info(f"Added model check job with ID: model_check_{job_id_suffix}")
    
    # Start the scheduler if not already running
    if not scheduler.running:
        logger.info("Starting scheduler")
        scheduler.start()
    
    logger.info("Scheduler initialization complete")

def refresh_cache_for_popular_tickers():
    """Refresh cache for popular tickers on a schedule"""
    try:
        # Import here to avoid circular imports
        import data_fetcher as df
        
        # List of popular tickers to keep fresh
        popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ"]
        
        # Randomize order to distribute API calls
        random.shuffle(popular_tickers)
        
        for ticker in popular_tickers:
            try:
                # Add jitter to avoid rate limits
                time.sleep(random.uniform(1, 3))
                
                # Refresh price history
                df.get_price_history(ticker, force_refresh=True)
                
                # Refresh options chain
                df.get_options_chain(ticker, force_refresh=True)
                
                # Refresh fundamentals
                df.get_fundamentals(ticker, force_refresh=True)
                
                print(f"Successfully refreshed cache for {ticker}")
            except Exception as e:
                print(f"Error refreshing cache for {ticker}: {str(e)}")
    except Exception as e:
        print(f"Error in cache refresh job: {str(e)}")
        traceback.print_exc()

def check_model_drift():
    """Check for model drift and retrain if necessary"""
    try:
        # Import here to avoid circular imports
        import analysis as al
        
        # Check if model should be retrained
        model, accuracy = al.load_or_train_model(force_retrain=False)
        
        print(f"Model check complete. Current accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error in model drift check: {str(e)}")
        traceback.print_exc()

def plot_price_history(price_history, ticker):
    """Create an interactive price chart with volume"""
    # Create subplot grid with shared x-axis
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Stock Price", "Volume")
    )
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=price_history.index,
            open=price_history['Open'],
            high=price_history['High'],
            low=price_history['Low'],
            close=price_history['Close'],
            name=f"{ticker} Price"
        ),
        row=1, col=1
    )
    
    # Add volume as bar chart
    fig.add_trace(
        go.Bar(
            x=price_history.index,
            y=price_history['Volume'],
            name='Volume',
            marker=dict(color='rgba(100, 100, 255, 0.3)')
        ),
        row=2, col=1
    )
    
    # Add moving averages
    price_history['SMA20'] = price_history['Close'].rolling(window=20).mean()
    price_history['SMA50'] = price_history['Close'].rolling(window=50).mean()
    
    fig.add_trace(
        go.Scatter(
            x=price_history.index,
            y=price_history['SMA20'],
            name='20-day SMA',
            line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_history.index,
            y=price_history['SMA50'],
            name='50-day SMA',
            line=dict(color='rgba(255, 0, 0, 0.8)', width=1.5)
        ),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    
    # Update y-axis ranges
    fig.update_yaxes(
        range=[price_history['Low'].min() * 0.95, price_history['High'].max() * 1.05],
        row=1, col=1
    )
    
    fig.update_yaxes(
        range=[0, price_history['Volume'].max() * 5],
        row=2, col=1
    )
    
    return fig

def plot_volatility(iv_hv_data):
    """Create a volatility comparison chart using subplots"""
    from plotly.subplots import make_subplots
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        if iv_hv_data is None:
            # Create an empty chart with a message
            logger.warning("Volatility data is None, creating empty chart")
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="Volatility data not available",
                showarrow=False,
                font=dict(size=20)
            )
            
            fig.update_layout(
                title="Volatility Analysis",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            return fig
        
        # Log the structure of the data for debugging
        logger.info(f"IV-HV data shape: {iv_hv_data.shape}, columns: {list(iv_hv_data.columns)}")
        logger.info(f"IV-HV data index type: {type(iv_hv_data.index)}")
        
        # Check if we have IV data - determine if we need one or two plots
        has_iv_data = 'IV' in iv_hv_data.columns and not iv_hv_data['IV'].isna().all()
        has_hv_data = 'HV' in iv_hv_data.columns and not iv_hv_data['HV'].isna().all()
        has_spread_data = 'IV_HV_Spread' in iv_hv_data.columns and not iv_hv_data['IV_HV_Spread'].isna().all()
        
        logger.info(f"Data availability - IV: {has_iv_data}, HV: {has_hv_data}, Spread: {has_spread_data}")
        
        if has_iv_data and has_hv_data:
            # We have both IV and HV data - create a comparison chart with both subplots
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=("Implied vs Historical Volatility", "IV-HV Spread")
            )
            
            # Clean HV data
            hv_values = iv_hv_data['HV'].copy()
            hv_values = hv_values.replace([np.inf, -np.inf], np.nan)
            iv_hv_data['HV_clean'] = hv_values.fillna(method='ffill').fillna(method='bfill')
            
            # Add historical volatility
            fig.add_trace(
                go.Scatter(
                    x=iv_hv_data.index,
                    y=iv_hv_data['HV_clean'] * 100,  # Convert to percentage
                    mode='lines',
                    name='Historical Volatility',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Clean IV data
            iv_values = iv_hv_data['IV'].copy()
            iv_values = iv_values.replace([np.inf, -np.inf], np.nan)
            iv_hv_data['IV_clean'] = iv_values.fillna(method='ffill').fillna(method='bfill')
            
            # Add implied volatility
            fig.add_trace(
                go.Scatter(
                    x=iv_hv_data.index,
                    y=iv_hv_data['IV_clean'] * 100,  # Convert to percentage
                    mode='lines',
                    name='Implied Volatility',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # Recalculate spread using clean data if needed
            if has_spread_data:
                spread_values = iv_hv_data['IV_HV_Spread'].copy()
                spread_values = spread_values.replace([np.inf, -np.inf], np.nan)
                iv_hv_data['IV_HV_Spread_clean'] = spread_values.fillna(method='ffill').fillna(method='bfill')
            else:
                iv_hv_data['IV_HV_Spread_clean'] = iv_hv_data['IV_clean'] - iv_hv_data['HV_clean']
            
            # Add IV-HV spread as a bar chart
            fig.add_trace(
                go.Bar(
                    x=iv_hv_data.index,
                    y=iv_hv_data['IV_HV_Spread_clean'] * 100,  # Convert to percentage
                    name='IV-HV Spread',
                    marker=dict(
                        color=iv_hv_data['IV_HV_Spread_clean'].apply(
                            lambda x: 'rgba(0, 255, 0, 0.5)' if x > 0 else 'rgba(255, 0, 0, 0.5)'
                        )
                    )
                ),
                row=2, col=1
            )
            
            # Add zero line for the spread
            fig.add_hline(
                y=0, 
                line=dict(color="black", width=1, dash="dash"),
                row=2, col=1
            )
            
            # Set y-axis ranges to reasonable values for volatility (top plot)
            fig.update_yaxes(
                title="Volatility (%)",
                range=[0, max(iv_hv_data['IV_clean'].max(), iv_hv_data['HV_clean'].max()) * 100 * 1.1],
                row=1, col=1
            )
            
            # Set y-axis ranges for spread (bottom plot)
            fig.update_yaxes(
                title="Spread (%)",
                range=[min(iv_hv_data['IV_HV_Spread_clean'].min() * 100 * 1.5, -2), 
                      max(iv_hv_data['IV_HV_Spread_clean'].max() * 100 * 1.5, 2)],
                row=2, col=1
            )
            
        elif has_hv_data:
            # We only have historical volatility data
            fig = go.Figure()
            
            # Clean HV data
            hv_values = iv_hv_data['HV'].copy()
            hv_values = hv_values.replace([np.inf, -np.inf], np.nan)
            clean_hv = hv_values.fillna(method='ffill').fillna(method='bfill')
            
            # Add historical volatility
            fig.add_trace(
                go.Scatter(
                    x=iv_hv_data.index,
                    y=clean_hv * 100,  # Convert to percentage
                    mode='lines',
                    name='Historical Volatility',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add note about missing IV data
            fig.add_annotation(
                x=0.5,
                y=0.9,
                xref="paper",
                yref="paper",
                text="Implied Volatility data not available - showing Historical Volatility only",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Set y-axis range
            fig.update_yaxes(
                title="Historical Volatility (%)",
                range=[0, clean_hv.max() * 100 * 1.1]
            )
            
            # Set title
            fig.update_layout(title="Historical Volatility")
            
        else:
            # No volatility data at all
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="No volatility data available for this ticker",
                showarrow=False,
                font=dict(size=16)
            )
            
            fig.update_layout(title="Volatility Analysis")
        
        # Common layout settings
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"  # Shows all data at the same x-coordinate on hover
        )
        
        logger.info("Successfully created volatility chart")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating volatility chart: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a simple error chart with more details for debugging
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Error creating volatility chart: {str(e)}",
            showarrow=False,
            font=dict(size=16)
        )
        
        if iv_hv_data is not None:
            # Add additional info about the data that caused the error
            try:
                data_info = f"Columns: {list(iv_hv_data.columns)}, Shape: {iv_hv_data.shape}"
                fig.add_annotation(
                    x=0.5,
                    y=0.4,
                    text=data_info,
                    showarrow=False,
                    font=dict(size=12)
                )
            except:
                pass
        
        fig.update_layout(
            title="Implied vs Historical Volatility (Error)",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig

def calculate_option_greeks(option, stock_price, risk_free_rate=0.05):
    """
    Calculate option greeks (simplified approach).
    In production, you would use specialized libraries or APIs for this.
    """
    # This is a placeholder for demonstration
    # In a real application, you would use a proper option pricing model
    
    strike = option['strike']
    premium = option['lastPrice']
    implied_vol = option['impliedVolatility']
    days_to_expiry = 30  # Approximation
    
    # Simplified delta calculation
    if option['type'] == 'call':
        delta = 0.5 + 0.5 * (stock_price - strike) / (stock_price * implied_vol * np.sqrt(days_to_expiry/365))
    else:  # put
        delta = -0.5 - 0.5 * (stock_price - strike) / (stock_price * implied_vol * np.sqrt(days_to_expiry/365))
    
    # Clip delta to reasonable range
    delta = max(min(delta, 1.0), -1.0)
    
    # Other greeks - these are very simplified approximations
    gamma = 0.1  # Placeholder
    theta = -premium * 0.01  # Rough approximation of daily time decay
    vega = premium * 0.1  # Rough approximation
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

## file_path: cache/.gitkeep
