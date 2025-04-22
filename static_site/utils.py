import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# In the stlite version, we don't need the scheduler functionality

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
    
    try:
        if iv_hv_data is None:
            # Create an empty chart with a message
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
        
        # Check if we have IV data - determine if we need one or two plots
        has_iv_data = 'IV' in iv_hv_data.columns and not iv_hv_data['IV'].isna().all()
        has_hv_data = 'HV' in iv_hv_data.columns and not iv_hv_data['HV'].isna().all()
        has_spread_data = 'IV_HV_Spread' in iv_hv_data.columns and not iv_hv_data['IV_HV_Spread'].isna().all()
        
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
            iv_hv_data['HV_clean'] = hv_values.ffill().bfill()
            
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
            iv_hv_data['IV_clean'] = iv_values.ffill().bfill()
            
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
                iv_hv_data['IV_HV_Spread_clean'] = spread_values.ffill().bfill()
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
            clean_hv = hv_values.ffill().bfill()
            
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
            
            # Update layout
            fig.update_layout(
                title="Historical Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # Set y-axis range
            fig.update_yaxes(
                range=[0, clean_hv.max() * 100 * 1.1]
            )
            
        else:
            # No volatility data available
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
        
    except Exception as e:
        # Create a fallback figure in case of any error
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="Error generating volatility chart",
            showarrow=False,
            font=dict(size=20)
        )
        
        fig.update_layout(
            title="Volatility Analysis",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig

def calculate_option_greeks(option, stock_price, risk_free_rate=0.05):
    """
    Calculate option greeks (simplified approach).
    In production, you would use specialized libraries or APIs for this.
    """
    # For stlite, we'll just return simplified values
    # In a real application, you'd use proper models like Black-Scholes
    
    # Extract option details
    strike = option['strike']
    implied_vol = option['impliedVolatility']
    
    # Calculate simple delta
    delta = 0.5  # Default at-the-money value
    
    # Adjust delta based on moneyness
    moneyness = stock_price / strike
    if moneyness > 1.05:  # Deep in-the-money
        delta = 0.8
    elif moneyness < 0.95:  # Deep out-of-the-money
        delta = 0.2
    
    # Simple approximations for other greeks
    gamma = 0.05  # Higher for ATM options
    theta = -0.01 * stock_price  # Negative, higher for higher prices
    vega = 0.1 * stock_price  # Higher for higher prices
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }