import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

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
        "max_gain": "Strike - Premium Paid (stock goes to zero)",
        "breakeven": "Strike - Premium Paid",
        "risk_profile": "Limited risk, high reward strategy",
        "best_when": "Strong decline in the underlying stock"
    },
    "bearish_high_risk": {
        "name": "Put Ratio Backspread",
        "description": "Sell one ATM put and buy multiple OTM puts with the same expiration date.",
        "legs": ["Short Put (ATM)", "2-3x Long Puts (OTM)"],
        "max_loss": "Difference between strikes - Net Credit",
        "max_gain": "High, limited by stock falling to zero",
        "breakeven": "Complex - depends on ratio",
        "risk_profile": "Limited risk, high reward strategy with leverage",
        "best_when": "Sharp decline in the underlying stock"
    },
    "neutral_low_risk": {
        "name": "Calendar Spread",
        "description": "Sell a near-term option and buy a longer-term option at the same strike price.",
        "legs": ["Short Option (near-term)", "Long Option (longer-term)"],
        "max_loss": "Net Premium Paid",
        "max_gain": "Limited, depends on volatility",
        "breakeven": "Complex - depends on time decay",
        "risk_profile": "Limited risk, limited reward strategy",
        "best_when": "Stable stock price with time decay"
    },
    "neutral_moderate_risk": {
        "name": "Iron Condor",
        "description": "Sell an OTM put spread and an OTM call spread with the same expiration date.",
        "legs": ["Short Call (OTM)", "Long Call (further OTM)", "Short Put (OTM)", "Long Put (further OTM)"],
        "max_loss": "Greater of two spread widths - Net Credit",
        "max_gain": "Net Credit Received",
        "breakeven": "Lower Call Strike - Net Credit, Higher Put Strike + Net Credit",
        "risk_profile": "Limited risk, limited reward strategy",
        "best_when": "Stock price remains between short strikes"
    },
    "neutral_high_risk": {
        "name": "Short Straddle",
        "description": "Sell a put and a call at the same strike price and expiration date.",
        "legs": ["Short Call (ATM)", "Short Put (ATM)"],
        "max_loss": "Unlimited (on the upside)",
        "max_gain": "Net Premium Received",
        "breakeven": "Strike ± Net Premium Received",
        "risk_profile": "High risk, limited reward strategy",
        "best_when": "Very stable stock price with low volatility"
    },
    "volatile_low_risk": {
        "name": "Long Straddle",
        "description": "Buy a put and a call at the same strike price and expiration date.",
        "legs": ["Long Call (ATM)", "Long Put (ATM)"],
        "max_loss": "Net Premium Paid",
        "max_gain": "Unlimited (up) or Strike - Premium (down)",
        "breakeven": "Strike ± Net Premium Paid",
        "risk_profile": "Limited risk, unlimited reward strategy",
        "best_when": "Big move in either direction"
    },
    "volatile_moderate_risk": {
        "name": "Long Strangle",
        "description": "Buy an OTM put and an OTM call with the same expiration date.",
        "legs": ["Long Call (OTM)", "Long Put (OTM)"],
        "max_loss": "Net Premium Paid",
        "max_gain": "Unlimited (up) or Put Strike - Premium (down)",
        "breakeven": "Call Strike + Premium or Put Strike - Premium",
        "risk_profile": "Limited risk, unlimited reward strategy (cheaper than straddle)",
        "best_when": "Big move in either direction"
    },
    "volatile_high_risk": {
        "name": "Butterfly Spread",
        "description": "Buy one option at a lower strike, sell two at a middle strike, and buy one at a higher strike.",
        "legs": ["Long Option (lower)", "2x Short Options (middle)", "Long Option (higher)"],
        "max_loss": "Net Premium Paid",
        "max_gain": "Higher Strike - Lower Strike - Net Premium",
        "breakeven": "Lower Strike + Premium or Higher Strike - Premium",
        "risk_profile": "Limited risk, higher reward than other volatile strategies",
        "best_when": "Big move in either direction past breakeven points"
    }
}

def recommend_strategy(price_history, fundamentals, selected_option, option_type, outlook, risk_profile):
    """Recommend an options strategy based on market outlook and risk profile"""
    # Map risk profile to risk level
    risk_level = {
        "Conservative": "low_risk",
        "Moderate": "moderate_risk",
        "Aggressive": "high_risk"
    }.get(risk_profile, "moderate_risk")
    
    # Define strategy key
    strategy_key = f"{outlook}_{risk_level}"
    
    # Get strategy or fall back to neutral strategy
    if strategy_key in STRATEGIES:
        strategy = STRATEGIES[strategy_key]
    else:
        # Default to a neutral strategy
        strategy = STRATEGIES[f"neutral_{risk_level}"]
    
    return strategy

def generate_payoff_chart(strategy, selected_option, option_type, current_price):
    """Generate a payoff chart for the recommended strategy"""
    # Get option strike and price
    strike = selected_option['strike']
    option_price = selected_option['lastPrice']
    
    # Generate price points for x-axis
    price_range = 0.3  # Show prices ±30% of current
    min_price = current_price * (1 - price_range)
    max_price = current_price * (1 + price_range)
    
    price_points = np.linspace(min_price, max_price, 100)
    
    # Simple payoff calculations based on strategy
    if strategy['name'] == "Long Call":
        payoffs = [max(price - strike - option_price, -option_price) for price in price_points]
        
    elif strategy['name'] == "Long Put":
        payoffs = [max(strike - price - option_price, -option_price) for price in price_points]
        
    elif strategy['name'] == "Bull Call Spread":
        # Simplified: assume higher strike is 10% above lower strike
        higher_strike = strike * 1.1
        # Assume higher strike option price is 40% of the selected option
        higher_option_price = option_price * 0.4
        net_premium = option_price - higher_option_price
        
        payoffs = []
        for price in price_points:
            if price <= strike:
                payoff = -net_premium
            elif price >= higher_strike:
                payoff = higher_strike - strike - net_premium
            else:
                payoff = price - strike - net_premium
            payoffs.append(payoff)
            
    elif strategy['name'] == "Bear Put Spread":
        # Simplified: assume lower strike is 10% below higher strike
        lower_strike = strike * 0.9
        # Assume lower strike option price is 40% of the selected option
        lower_option_price = option_price * 0.4
        net_premium = option_price - lower_option_price
        
        payoffs = []
        for price in price_points:
            if price >= strike:
                payoff = -net_premium
            elif price <= lower_strike:
                payoff = strike - lower_strike - net_premium
            else:
                payoff = strike - price - net_premium
            payoffs.append(payoff)
            
    elif strategy['name'] == "Long Straddle":
        # Simplified: assume put price is 80% of call price at same strike
        put_price = option_price * 0.8
        net_premium = option_price + put_price
        
        payoffs = []
        for price in price_points:
            call_payoff = max(price - strike, 0) - option_price
            put_payoff = max(strike - price, 0) - put_price
            payoffs.append(call_payoff + put_payoff)
            
    elif strategy['name'] == "Iron Condor":
        # Simplified assumptions for the other legs
        call_spread = 0.1 * current_price  # 10% spread
        put_spread = 0.1 * current_price   # 10% spread
        
        short_call_strike = current_price * 1.05  # 5% OTM
        long_call_strike = short_call_strike + call_spread
        
        short_put_strike = current_price * 0.95   # 5% OTM
        long_put_strike = short_put_strike - put_spread
        
        # Simplified credit calculation
        net_credit = option_price * 0.3  # Typical net credit
        
        payoffs = []
        for price in price_points:
            if price <= long_put_strike:
                payoff = -put_spread + net_credit
            elif price < short_put_strike:
                payoff = -(short_put_strike - price) + net_credit
            elif price <= short_call_strike:
                payoff = net_credit  # Max profit zone
            elif price < long_call_strike:
                payoff = -(price - short_call_strike) + net_credit
            else:
                payoff = -call_spread + net_credit
            payoffs.append(payoff)
    
    else:
        # Default linear payoff curve for unimplemented strategies
        # Determine if the strategy name suggests bullish or bearish direction
        strategy_name = strategy['name'].lower()
        is_bullish = any(term in strategy_name for term in ['bull', 'call', 'long'])
        slope = 1 if is_bullish else -1
        payoffs = [slope * (price - current_price) for price in price_points]
    
    # Create the payoff chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=price_points,
            y=payoffs,
            mode='lines',
            name=f'{strategy["name"]} Payoff',
            line=dict(color='green' if any(p > 0 for p in payoffs) else 'red', width=2)
        )
    )
    
    # Add break-even line
    fig.add_hline(
        y=0, 
        line=dict(color="black", width=1, dash="dash"),
        annotation_text="Break-even",
        annotation_position="bottom right"
    )
    
    # Add current price line
    fig.add_vline(
        x=current_price, 
        line=dict(color="blue", width=1, dash="dash"),
        annotation_text="Current Price",
        annotation_position="top right"
    )
    
    # Add option strike line
    fig.add_vline(
        x=strike, 
        line=dict(color="red", width=1, dash="dash"),
        annotation_text=f"{option_type.capitalize()} Strike",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        title=f"{strategy['name']} Payoff Diagram",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss ($)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x"
    )
    
    return fig

def render_strategy_card(strategy, selected_option, option_type, ticker, current_price):
    """Render a card with strategy details and payoff chart"""
    st.subheader(f"Recommended Strategy: {strategy['name']}")
    
    st.write(f"**Description:** {strategy['description']}")
    
    # Display strategy details
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Strategy Legs:**")
        for leg in strategy['legs']:
            st.write(f"- {leg}")
        
        st.write(f"**Best When:** {strategy['best_when']}")
        
    with col2:
        st.write(f"**Max Loss:** {strategy['max_loss']}")
        st.write(f"**Max Gain:** {strategy['max_gain']}")
        st.write(f"**Breakeven:** {strategy['breakeven']}")
        
    # Generate and display payoff chart
    fig = generate_payoff_chart(strategy, selected_option, option_type, current_price)
    st.plotly_chart(fig, use_container_width=True)