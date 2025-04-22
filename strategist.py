import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import traceback

# Define strategy templates
STRATEGIES = {
    "bullish_low_risk": {
        "name": "Bull Call Spread",
        "description": "Buy a call option at a lower strike price and sell a call option at a higher strike price with the same expiration date.",
        "legs": ["Long Call (ATM)", "Short Call (OTM)"],
        "max_loss": "Net Premium Paid",
        "max_gain": "Spread Width - Net Premium Paid",
        "breakeven": "Lower Strike + Net Premium Paid",
        "risk_profile": "Limited risk, limited reward strategy",
        "best_when": "Moderate rise in the underlying stock"
    },
    "bullish_moderate_risk": {
        "name": "Long Call",
        "description": "Buy a call option to profit from a rise in the stock price.",
        "legs": ["Long Call (slightly OTM)"],
        "max_loss": "Premium Paid",
        "max_gain": "Unlimited",
        "breakeven": "Strike + Premium Paid",
        "risk_profile": "Limited risk, unlimited reward strategy",
        "best_when": "Strong rise in the underlying stock"
    },
    "bullish_high_risk": {
        "name": "Call Ratio Backspread",
        "description": "Sell one ATM call and buy multiple OTM calls with the same expiration date.",
        "legs": ["Short Call (ATM)", "2-3x Long Calls (OTM)"],
        "max_loss": "Difference between strikes - Net Credit",
        "max_gain": "Unlimited",
        "breakeven": "Complex - depends on ratio",
        "risk_profile": "Limited risk, unlimited reward strategy with higher leverage",
        "best_when": "Sharp rise in the underlying stock"
    },
    "bearish_low_risk": {
        "name": "Bear Put Spread",
        "description": "Buy a put option at a higher strike price and sell a put option at a lower strike price with the same expiration date.",
        "legs": ["Long Put (ATM)", "Short Put (OTM)"],
        "max_loss": "Net Premium Paid",
        "max_gain": "Spread Width - Net Premium Paid",
        "breakeven": "Higher Strike - Net Premium Paid",
        "risk_profile": "Limited risk, limited reward strategy",
        "best_when": "Moderate decline in the underlying stock"
    },
    "bearish_moderate_risk": {
        "name": "Long Put",
        "description": "Buy a put option to profit from a decline in the stock price.",
        "legs": ["Long Put (slightly OTM)"],
        "max_loss": "Premium Paid",
        "max_gain": "Strike - Premium Paid (stock can't go below zero)",
        "breakeven": "Strike - Premium Paid",
        "risk_profile": "Limited risk, high reward strategy",
        "best_when": "Strong decline in the underlying stock"
    },
    "bearish_high_risk": {
        "name": "Put Ratio Backspread",
        "description": "Sell one ATM put and buy multiple OTM puts with the same expiration date.",
        "legs": ["Short Put (ATM)", "2-3x Long Puts (OTM)"],
        "max_loss": "Difference between strikes - Net Credit",
        "max_gain": "High but limited (stock can't go below zero)",
        "breakeven": "Complex - depends on ratio",
        "risk_profile": "Limited risk, high reward strategy with higher leverage",
        "best_when": "Sharp decline in the underlying stock"
    },
    "neutral_low_risk": {
        "name": "Iron Condor",
        "description": "A combination of a bull put spread and a bear call spread.",
        "legs": ["Short Call (OTM)", "Long Call (further OTM)", "Short Put (OTM)", "Long Put (further OTM)"],
        "max_loss": "Width of either spread - Net Premium Received",
        "max_gain": "Net Premium Received",
        "breakeven": "Two points: Lower short strike + Net Premium, Upper short strike - Net Premium",
        "risk_profile": "Limited risk, limited reward strategy",
        "best_when": "Stock trades in a range between short strikes"
    },
    "neutral_moderate_risk": {
        "name": "Short Straddle",
        "description": "Sell a call and a put at the same strike price and expiration date.",
        "legs": ["Short Call (ATM)", "Short Put (ATM)"],
        "max_loss": "Unlimited",
        "max_gain": "Premium Received",
        "breakeven": "Two points: Strike - Premium, Strike + Premium",
        "risk_profile": "Unlimited risk, limited reward strategy",
        "best_when": "Low volatility, stock trades near strike price"
    },
    "neutral_high_risk": {
        "name": "Butterfly Spread",
        "description": "Buy one option at a lower strike, sell two at a middle strike, and buy one at a higher strike.",
        "legs": ["Long Call/Put (Lower Strike)", "2x Short Call/Put (Middle Strike)", "Long Call/Put (Higher Strike)"],
        "max_loss": "Net Premium Paid",
        "max_gain": "Difference between adjacent strikes - Net Premium Paid",
        "breakeven": "Two points: Lower strike + Premium, Higher strike - Premium",
        "risk_profile": "Limited risk, limited reward strategy with high leverage",
        "best_when": "Stock price ends exactly at middle strike at expiration"
    },
    "volatile_low_risk": {
        "name": "Long Straddle",
        "description": "Buy a call and a put at the same strike price and expiration date.",
        "legs": ["Long Call (ATM)", "Long Put (ATM)"],
        "max_loss": "Premium Paid",
        "max_gain": "Unlimited",
        "breakeven": "Two points: Strike - Premium, Strike + Premium",
        "risk_profile": "Limited risk, unlimited reward strategy",
        "best_when": "Large move in either direction"
    },
    "volatile_moderate_risk": {
        "name": "Long Strangle",
        "description": "Buy an OTM call and an OTM put with the same expiration date.",
        "legs": ["Long Call (OTM)", "Long Put (OTM)"],
        "max_loss": "Premium Paid",
        "max_gain": "Unlimited",
        "breakeven": "Two points: Call Strike + Premium, Put Strike - Premium",
        "risk_profile": "Limited risk, unlimited reward strategy with lower cost",
        "best_when": "Very large move in either direction"
    },
    "volatile_high_risk": {
        "name": "Long Guts",
        "description": "Buy an ITM call and an ITM put with the same expiration date.",
        "legs": ["Long Call (ITM)", "Long Put (ITM)"],
        "max_loss": "Premium Paid - Intrinsic Value",
        "max_gain": "Unlimited",
        "breakeven": "Two points: Call Strike + Premium - Put Strike, Put Strike - Premium + Call Strike",
        "risk_profile": "Limited risk, unlimited reward strategy with higher cost but lower breakeven range",
        "best_when": "Extreme move in either direction"
    },
    "income_conservative": {
        "name": "Covered Call",
        "description": "Own the stock and sell a call option against it.",
        "legs": ["Long 100 Shares", "Short Call (OTM)"],
        "max_loss": "Stock Price - Premium Received",
        "max_gain": "Premium Received + (Strike - Stock Price) if called",
        "breakeven": "Stock Price - Premium Received",
        "risk_profile": "Same downside risk as stock ownership, limited upside",
        "best_when": "Stock remains flat or rises slightly"
    },
    "income_moderate": {
        "name": "Cash-Secured Put",
        "description": "Sell a put option while having cash reserved to buy the stock if assigned.",
        "legs": ["Short Put (OTM)"],
        "max_loss": "Strike - Premium Received",
        "max_gain": "Premium Received",
        "breakeven": "Strike - Premium Received",
        "risk_profile": "Limited but potentially substantial risk, limited reward",
        "best_when": "Stock remains flat or rises slightly"
    }
}

def recommend_strategy(price_history, fundamentals, selected_option, option_type, outlook, risk_profile):
    """Recommend an options strategy based on market outlook and risk profile"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Log inputs for debugging
    logger.info(f"Strategy inputs - outlook: {outlook}, risk_profile: {risk_profile}, option_type: {option_type}")
    
    # Validate option_type and provide a default if needed
    if option_type not in ['call', 'put']:
        logger.warning(f"Invalid option_type: {option_type}, defaulting to 'call'")
        option_type = 'call'
    
    # Default strategy for fallback
    default_strategy = STRATEGIES["neutral_low_risk"]
    
    # Map risk profiles to risk levels
    risk_level_map = {
        "Conservative": "low_risk",
        "Moderate": "moderate_risk",
        "Aggressive": "high_risk"
    }
    
    risk_level = risk_level_map.get(risk_profile, "moderate_risk")
    
    # Build strategy key
    strategy_key = f"{outlook}_{risk_level}"
    logger.info(f"Strategy key: {strategy_key}")
    
    # Handle income strategies separately for more conservative outlooks
    if risk_profile == "Conservative" and outlook in ["bullish", "neutral"]:
        if option_type == "call":
            return STRATEGIES["income_conservative"]  # Covered call
        else:
            return STRATEGIES["income_moderate"]  # Cash-secured put
    
    # Get strategy or default to neutral
    strategy = STRATEGIES.get(strategy_key, default_strategy)
    logger.info(f"Selected strategy: {strategy['name']}")
    
    # Add additional context based on the selected option
    strategy_with_context = strategy.copy()
    
    # Safely extract values from selected_option with defaults
    try:
        strike = selected_option.get("strike", 0)
        last_price = selected_option.get("lastPrice", 0)
    except (AttributeError, TypeError) as e:
        logger.error(f"Error extracting option values: {e}")
        # Provide default values if selected_option is None or not a dict
        strike = 0
        last_price = 0
    
    # Add specific option details
    strategy_with_context["selected_option"] = {
        "strike": strike,
        "expiry": "Selected expiration date",  # Could be more specific with actual date
        "premium": last_price,
        "type": option_type
    }
    
    # Add specific breakeven calculation if possible
    if strategy["name"] == "Long Call":
        strategy_with_context["specific_breakeven"] = strike + last_price
    elif strategy["name"] == "Long Put":
        strategy_with_context["specific_breakeven"] = strike - last_price
    
    return strategy_with_context

def generate_payoff_chart(strategy, selected_option, option_type, current_price):
    """Generate a payoff chart for the recommended strategy"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Safely get strike and lastPrice from selected_option
    try:
        strike = selected_option.get("strike", current_price)
        last_price = selected_option.get("lastPrice", current_price * 0.05)  # Default to 5% of current price
    except (AttributeError, TypeError) as e:
        logger.error(f"Error extracting option values for payoff chart: {e}")
        # Provide default values if selected_option is None or not a dict
        strike = current_price
        last_price = current_price * 0.05
    
    # Validate option_type
    if option_type not in ['call', 'put']:
        logger.warning(f"Invalid option_type in payoff chart: {option_type}, defaulting to 'call'")
        option_type = 'call'
    
    # Define price range for x-axis (30% below and above current price)
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
    
    # Calculate payoff based on strategy
    strategy_name = strategy["name"]
    payoff = []
    
    if strategy_name == "Long Call":
        # For long call: max(0, stock_price - strike) - premium
        for price in price_range:
            payoff.append(max(0, price - strike) - last_price)
    
    elif strategy_name == "Long Put":
        # For long put: max(0, strike - stock_price) - premium
        for price in price_range:
            payoff.append(max(0, strike - price) - last_price)
    
    elif strategy_name == "Bull Call Spread":
        # Simplified bull call spread (assuming strikes are 5% apart)
        higher_strike = strike * 1.05
        spread_cost = last_price * 0.6  # Approximate net cost
        
        for price in price_range:
            if price <= strike:
                payoff.append(-spread_cost)
            elif price >= higher_strike:
                payoff.append(higher_strike - strike - spread_cost)
            else:
                payoff.append(price - strike - spread_cost)
    
    elif strategy_name == "Bear Put Spread":
        # Simplified bear put spread (assuming strikes are 5% apart)
        lower_strike = strike * 0.95
        spread_cost = last_price * 0.6  # Approximate net cost
        
        for price in price_range:
            if price >= strike:
                payoff.append(-spread_cost)
            elif price <= lower_strike:
                payoff.append(strike - lower_strike - spread_cost)
            else:
                payoff.append(strike - price - spread_cost)
    
    elif strategy_name == "Covered Call":
        # For covered call: (stock_price - current_price) + premium - max(0, stock_price - strike)
        premium = last_price
        
        for price in price_range:
            stock_pl = price - current_price
            option_pl = premium - max(0, price - strike)
            payoff.append(stock_pl + option_pl)
    
    elif strategy_name == "Cash-Secured Put":
        # For cash-secured put: premium - max(0, strike - stock_price)
        premium = last_price
        
        for price in price_range:
            payoff.append(premium - max(0, strike - price))
    
    elif strategy_name == "Long Straddle":
        # For long straddle (using selected option as one leg and estimating the other)
        call_premium = last_price
        put_premium = call_premium * 0.9  # Estimate put cost
        total_premium = call_premium + put_premium
        
        for price in price_range:
            call_payoff = max(0, price - strike)
            put_payoff = max(0, strike - price)
            payoff.append(call_payoff + put_payoff - total_premium)
    
    else:
        # Default to single option payoff if strategy not specifically handled
        if option_type == "call":
            for price in price_range:
                payoff.append(max(0, price - strike) - last_price)
        else:  # put
            for price in price_range:
                payoff.append(max(0, strike - price) - last_price)
    
    # Create the chart
    fig = go.Figure()
    
    # Add payoff line
    fig.add_trace(go.Scatter(
        x=price_range, 
        y=payoff,
        mode='lines',
        name='Payoff at Expiration',
        line=dict(color='blue', width=2)
    ))
    
    # Add breakeven lines
    breakeven_points = []
    for i in range(1, len(payoff)):
        if (payoff[i-1] <= 0 and payoff[i] >= 0) or (payoff[i-1] >= 0 and payoff[i] <= 0):
            # Linear interpolation to find more precise breakeven
            x1, x2 = price_range[i-1], price_range[i]
            y1, y2 = payoff[i-1], payoff[i]
            breakeven = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
            breakeven_points.append(breakeven)
    
    for point in breakeven_points:
        fig.add_vline(x=point, line_dash="dash", line_color="green", annotation_text="Breakeven")
    
    # Add current price line
    fig.add_vline(x=current_price, line_dash="dash", line_color="red", annotation_text="Current Price")
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Layout
    fig.update_layout(
        title=f"Payoff Diagram: {strategy_name}",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss ($)",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def render_strategy_card(strategy, selected_option, option_type, ticker, current_price):
    """Render a card with strategy details and payoff chart"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Safely get strategy name with fallback
        strategy_name = strategy.get('name', 'Unspecified Strategy')
        st.subheader(f"Recommended Strategy: {strategy_name}")
        
        # Basic strategy information
        description = strategy.get('description', 'No description available')
        st.write(f"**Description:** {description}")
        
        # Strategy legs with safety check
        st.write("**Strategy Legs:**")
        legs = strategy.get('legs', ['Strategy components not specified'])
        for leg in legs:
            st.write(f"- {leg}")
        
        # Risk-reward profile
        col1, col2 = st.columns(2)
        with col1:
            max_loss = strategy.get('max_loss', 'Unspecified')
            max_gain = strategy.get('max_gain', 'Unspecified')
            st.write(f"**Maximum Loss:** {max_loss}")
            st.write(f"**Maximum Gain:** {max_gain}")
        
        with col2:
            breakeven = strategy.get('breakeven', 'Unspecified')
            st.write(f"**Breakeven:** {breakeven}")
            if "specific_breakeven" in strategy:
                try:
                    specific_be = float(strategy['specific_breakeven'])
                    st.write(f"**Specific Breakeven:** ${specific_be:.2f}")
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error formatting specific_breakeven: {e}")
        
        # Handle missing option_type
        if option_type not in ['call', 'put']:
            logger.warning(f"Invalid option_type in render_strategy_card: {option_type}, defaulting to 'call'")
            option_type = 'call'
            
        # Payoff chart
        payoff_chart = generate_payoff_chart(strategy, selected_option, option_type, current_price)
        st.plotly_chart(payoff_chart, use_container_width=True)
        
        # When to use this strategy
        best_when = strategy.get('best_when', 'Appropriate market conditions')
        risk_profile = strategy.get('risk_profile', 'Risk profile not specified')
        st.write(f"**Best When:** {best_when}")
        st.write(f"**Risk Profile:** {risk_profile}")
        
    except Exception as e:
        logger.error(f"Error rendering strategy card: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying strategy recommendations. Please try a different ticker or option selection.")
