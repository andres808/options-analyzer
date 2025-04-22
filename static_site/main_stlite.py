import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime
import sys
import traceback

# Page configuration
st.set_page_config(
    page_title="Options Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
try:
    import data_fetcher as df
    import analysis as al
    import strategist as strat
    import utils
except Exception as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# App title and description
st.title("Options Analysis & Strategy Recommender")
st.write("""
Analyze options data with machine learning insights and get personalized strategy recommendations
based on market conditions and your risk tolerance.
""")

# Add a note about stlite version
st.warning("""
⚠️ **DEMO MODE** - This is a static web version running in your browser.

This version uses simulated stock data to demonstrate functionality without requiring server-side API calls.
For real data analysis, please use the full server version of the application.

The interface and features are identical to the server version, but all data is simulated.
""")

# Sidebar inputs
with st.sidebar:
    st.header("Stock Selection")
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
    
    # Validate ticker input for security
    if not re.match(r'^[A-Z]{1,5}$', ticker_input):
        st.error("Invalid ticker format. Please use 1-5 capital letters only.")
        ticker = "AAPL"
    else:
        ticker = ticker_input
    
    # Risk profile selection
    st.header("Risk Profile")
    risk = st.select_slider(
        "Risk Tolerance", 
        ["Conservative", "Moderate", "Aggressive"], 
        value="Moderate"
    )
    
    # Period selection
    st.header("Data Settings")
    period = st.selectbox(
        "Historical Data Period",
        ["1mo", "3mo", "6mo", "1y"],
        index=2  # Default to 6mo for faster loading in browser
    )
    
    # Add refresh button
    if st.button("Refresh Data"):
        st.experimental_rerun()

# Main content
try:
    # Display loading message
    with st.spinner(f"Fetching data for {ticker}..."):
        # Get stock data
        price_history = df.get_price_history(
            ticker, 
            period=period, 
            interval="1d", 
            force_refresh=True  # Always fetch fresh data in stlite
        )
        
        # Get options data
        options_data = df.get_options_chain(ticker, force_refresh=True)
        
        # Get company fundamentals
        fundamentals = df.get_fundamentals(ticker, force_refresh=True)
    
    # Display stock information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        fig = utils.plot_price_history(price_history, ticker)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Company info card
        st.subheader(f"{fundamentals.get('longName', ticker)}")
        st.metric("Current Price", f"${price_history['Close'].iloc[-1]:.2f}", 
                 f"{(price_history['Close'].iloc[-1] / price_history['Close'].iloc[-2] - 1) * 100:.2f}%")
        
        st.write(f"**Sector:** {fundamentals.get('sector', 'N/A')}")
        st.write(f"**Industry:** {fundamentals.get('industry', 'N/A')}")
        st.write(f"**Market Cap:** ${fundamentals.get('marketCap', 0) / 1e9:.2f}B")
        
        # Market outlook based on analysis
        outlook = al.determine_market_outlook(price_history, fundamentals)
        
        st.subheader("Market Outlook")
        outlook_color = {
            "bullish": "green",
            "bearish": "red",
            "neutral": "blue",
            "volatile": "orange"
        }
        st.markdown(f"<h3 style='color: {outlook_color.get(outlook, 'gray')};'>{outlook.title()}</h3>", 
                   unsafe_allow_html=True)
    
    # Options Analysis Section
    st.header("Options Analysis")
    
    # Check if options data is available
    if not options_data or not options_data.get('expiry_dates'):
        st.warning(f"No options data available for {ticker}. Try popular tickers like AAPL, MSFT, AMZN, or SPY which typically have active options markets.")
        st.info("Many stocks don't have options trading or Yahoo Finance may not provide options data for them.")
    else:
        # Display available expiration dates
        expiry_dates = options_data.get('expiry_dates', [])
        
        if not expiry_dates:
            st.warning("No expiration dates found.")
        else:
            # Convert dates to more readable format
            formatted_dates = [datetime.strptime(date, '%Y-%m-%d').strftime('%b %d, %Y') for date in expiry_dates]
            expiry_dict = dict(zip(formatted_dates, expiry_dates))
            
            # Let user select expiration date
            selected_date_formatted = st.selectbox("Select Expiration Date", formatted_dates)
            selected_date = expiry_dict[selected_date_formatted]
            
            # Get options for selected date
            calls = options_data['options'][selected_date]['calls']
            puts = options_data['options'][selected_date]['puts']
            
            # Option type tabs
            option_tab1, option_tab2 = st.tabs(["Call Options", "Put Options"])
            
            with option_tab1:
                if len(calls) > 0:
                    # Add ML predictions to calls
                    calls_with_predictions = al.add_predictions(calls, price_history, fundamentals, 'call')
                    
                    # Display calls dataframe with columns selection
                    st.dataframe(
                        calls_with_predictions[['strike', 'lastPrice', 'bid', 'ask', 'volume', 
                                               'openInterest', 'impliedVolatility', 'inTheMoney', 
                                               'win_probability', 'expected_return']],
                        use_container_width=True,
                        height=300
                    )
                    
                    # Let user select a specific call option
                    call_strikes = calls_with_predictions['strike'].tolist()
                    selected_call_strike = st.selectbox("Select Call Strike for Detailed Analysis", call_strikes)
                    
                    # Get selected option data
                    selected_call = calls_with_predictions[calls_with_predictions['strike'] == selected_call_strike].iloc[0]
                    
                    # Display detailed analysis and strategy recommendation
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader(f"${selected_call_strike} Call Analysis")
                        st.write(f"**Last Price:** ${selected_call['lastPrice']:.2f}")
                        st.write(f"**Implied Volatility:** {selected_call['impliedVolatility'] * 100:.2f}%")
                        st.write(f"**Win Probability:** {selected_call['win_probability'] * 100:.2f}%")
                        st.write(f"**Expected Return:** {selected_call['expected_return'] * 100:.2f}%")
                        
                        # Greeks if available
                        if 'delta' in selected_call:
                            st.write("**Greeks:**")
                            st.write(f"Delta: {selected_call.get('delta', 'N/A'):.4f} | "
                                    f"Gamma: {selected_call.get('gamma', 'N/A'):.4f} | "
                                    f"Theta: {selected_call.get('theta', 'N/A'):.4f} | "
                                    f"Vega: {selected_call.get('vega', 'N/A'):.4f}")
                    
                    with col2:
                        # Get and display strategy recommendation for selected call
                        strategy = strat.recommend_strategy(
                            price_history, 
                            fundamentals, 
                            selected_call, 
                            'call', 
                            outlook, 
                            risk
                        )
                        
                        strat.render_strategy_card(strategy, selected_call, 'call', ticker, price_history['Close'].iloc[-1])
                else:
                    st.info("No call options available for this expiration date.")
            
            with option_tab2:
                if len(puts) > 0:
                    # Add ML predictions to puts
                    puts_with_predictions = al.add_predictions(puts, price_history, fundamentals, 'put')
                    
                    # Display puts dataframe
                    st.dataframe(
                        puts_with_predictions[['strike', 'lastPrice', 'bid', 'ask', 'volume', 
                                              'openInterest', 'impliedVolatility', 'inTheMoney',
                                              'win_probability', 'expected_return']],
                        use_container_width=True,
                        height=300
                    )
                    
                    # Let user select a specific put option
                    put_strikes = puts_with_predictions['strike'].tolist()
                    selected_put_strike = st.selectbox("Select Put Strike for Detailed Analysis", put_strikes)
                    
                    # Get selected option data
                    selected_put = puts_with_predictions[puts_with_predictions['strike'] == selected_put_strike].iloc[0]
                    
                    # Display detailed analysis and strategy recommendation
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader(f"${selected_put_strike} Put Analysis")
                        st.write(f"**Last Price:** ${selected_put['lastPrice']:.2f}")
                        st.write(f"**Implied Volatility:** {selected_put['impliedVolatility'] * 100:.2f}%")
                        st.write(f"**Win Probability:** {selected_put['win_probability'] * 100:.2f}%")
                        st.write(f"**Expected Return:** {selected_put['expected_return'] * 100:.2f}%")
                        
                        # Greeks if available
                        if 'delta' in selected_put:
                            st.write("**Greeks:**")
                            st.write(f"Delta: {selected_put.get('delta', 'N/A'):.4f} | "
                                    f"Gamma: {selected_put.get('gamma', 'N/A'):.4f} | "
                                    f"Theta: {selected_put.get('theta', 'N/A'):.4f} | "
                                    f"Vega: {selected_put.get('vega', 'N/A'):.4f}")
                    
                    with col2:
                        # Get and display strategy recommendation for selected put
                        strategy = strat.recommend_strategy(
                            price_history, 
                            fundamentals, 
                            selected_put, 
                            'put', 
                            outlook, 
                            risk
                        )
                        
                        strat.render_strategy_card(strategy, selected_put, 'put', ticker, price_history['Close'].iloc[-1])
                else:
                    st.info("No put options available for this expiration date.")
    
    # Volatility analysis section
    st.header("Volatility Analysis")
    
    # Check if options data is available for volatility analysis
    if not options_data or not options_data.get('expiry_dates'):
        st.info("Implied volatility analysis requires options data. Try popular tickers like AAPL, MSFT, AMZN, or SPY.")
        
        # Display only historical volatility
        hist_vol = df.get_historical_volatility(price_history)
        if not hist_vol.empty:
            st.subheader("Historical Volatility")
            hist_vol_df = pd.DataFrame({'Date': hist_vol.index, 'HV': hist_vol.values * 100})
            st.line_chart(hist_vol_df.set_index('Date'))
            st.caption("Historical volatility based on 20-day rolling standard deviation of returns, annualized.")
    else:
        # Calculate and display IV-HV spread
        try:
            iv_hv_data = al.calculate_iv_hv_spread(ticker, price_history, options_data)
            fig = utils.plot_volatility(iv_hv_data)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Implied volatility from options pricing vs historical volatility from price movements.")
        except Exception as e:
            st.error(f"Error calculating volatility data: {str(e)}")
            st.info("Try another ticker with more active options trading.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please try another ticker or check your internet connection.")

# Footer
st.markdown("---")
st.caption("Demo version for GitHub Pages deployment. Not financial advice.")
st.caption("All data shown is simulated for demonstration purposes only.")
st.caption("Static website version powered by [Stlite](https://github.com/whitphx/stlite) | [GitHub Repo](https://github.com/yourusername/options-analysis)")