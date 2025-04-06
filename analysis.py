import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import hashlib
import warnings
import traceback
import logging

# Import custom modules
import data_fetcher as df

# Constants
MODEL_PATH = "model.pkl"
MODEL_METADATA_PATH = "model_metadata.json"
RETRAIN_THRESHOLD_DAYS = 30  # Retrain model if older than 30 days
MIN_ACCURACY_THRESHOLD = 0.7  # Retrain if accuracy drops below 70%

def calculate_iv_hv_spread(ticker, price_history, options_data):
    """Calculate the spread between implied volatility and historical volatility"""
    logger = logging.getLogger(__name__)
    
    try:
        # Get historical volatility
        historical_volatility = df.get_historical_volatility(price_history)
        
        # Calculate average implied volatility from ATM options
        if not options_data or not options_data.get('expiry_dates'):
            logger.warning(f"No expiry dates found for {ticker}")
            return None
        
        # Get the nearest expiration date
        expiry = options_data['expiry_dates'][0]
        logger.info(f"Using nearest expiry date: {expiry}")
        
        # Get options for this expiry
        if not options_data.get('options') or not options_data['options'].get(expiry):
            logger.warning(f"No options data found for expiry date {expiry}")
            return None
            
        if not options_data['options'][expiry].get('calls'):
            logger.warning(f"No call options found for expiry date {expiry}")
            return None
            
        calls = pd.DataFrame(options_data['options'][expiry]['calls'])
        
        if calls.empty:
            logger.warning(f"Call options dataframe is empty for {ticker} with expiry {expiry}")
            return None
        
        # Get current price
        current_price = price_history['Close'].iloc[-1]
        
        # Find at-the-money options (closest strikes to current price)
        calls['distance'] = abs(calls['strike'] - current_price)
        atm_calls = calls.nsmallest(3, 'distance')
        
        # Calculate average IV
        avg_iv = atm_calls['impliedVolatility'].mean()
        logger.info(f"Average IV for ATM calls: {avg_iv:.4f}")
        
        # Get the last 30 trading days
        if len(price_history) < 30:
            logger.warning(f"Price history has less than 30 data points ({len(price_history)})")
            last_n = min(30, len(price_history))
            last_dates = price_history.index[-last_n:]
        else:
            last_dates = price_history.index[-30:]
            
        # Calculate IV-HV spread for recent dates
        iv_hv_data = pd.DataFrame(index=last_dates)
        iv_hv_data['Date'] = iv_hv_data.index
        
        # Check that historical_volatility has enough data points
        if len(historical_volatility) < 30:
            logger.warning(f"Historical volatility has less than 30 data points ({len(historical_volatility)})")
            last_n = min(30, len(historical_volatility))
            hv_values = historical_volatility.iloc[-last_n:].values
            # Pad with zeros if necessary
            if len(hv_values) < len(last_dates):
                padding = [0] * (len(last_dates) - len(hv_values))
                hv_values = np.concatenate([padding, hv_values])
        else:
            hv_values = historical_volatility.iloc[-30:].values
            
        iv_hv_data['HV'] = hv_values
        iv_hv_data['IV'] = avg_iv  # Same IV for all dates (snapshot)
        iv_hv_data['IV_HV_Spread'] = iv_hv_data['IV'] - iv_hv_data['HV']
        
        logger.info(f"Successfully calculated IV-HV spread for {ticker}")
        return iv_hv_data
        
    except Exception as e:
        logger.error(f"Error calculating IV-HV spread for {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def iv_hv_spread(iv, hv):
    """Helper function to calculate the spread between IV and HV"""
    return iv - hv

def determine_market_outlook(price_history, fundamentals):
    """Determine market outlook based on technical indicators and fundamentals"""
    # Calculate simple moving averages
    price_history['SMA20'] = price_history['Close'].rolling(window=20).mean()
    price_history['SMA50'] = price_history['Close'].rolling(window=50).mean()
    
    # Get last values
    current_price = price_history['Close'].iloc[-1]
    sma20 = price_history['SMA20'].iloc[-1]
    sma50 = price_history['SMA50'].iloc[-1]
    
    # Calculate historical volatility
    hist_vol = df.get_historical_volatility(price_history).iloc[-1]
    
    # Calculate IV-HV spread if we have options data
    try:
        options_data = df.get_options_chain(fundamentals['symbol'])
        iv_hv_data = calculate_iv_hv_spread(fundamentals['symbol'], price_history, options_data)
        if iv_hv_data is not None:
            iv_hv_spread_value = iv_hv_data['IV_HV_Spread'].iloc[-1]
        else:
            iv_hv_spread_value = 0
    except:
        iv_hv_spread_value = 0
    
    # Determine if market is volatile (IV-HV spread > 10%)
    is_volatile = iv_hv_spread_value > 0.1
    
    # Determine trend direction
    if current_price > sma20 > sma50:
        # Strong uptrend
        outlook = "bullish"
    elif current_price < sma20 < sma50:
        # Strong downtrend
        outlook = "bearish"
    elif abs(sma20 - sma50) / sma50 < 0.02:
        # SMA20 and SMA50 are very close (within 2%)
        outlook = "neutral"
    else:
        # Mixed signals
        outlook = "neutral"
    
    # Override with volatile if applicable
    if is_volatile:
        outlook = "volatile"
    
    return outlook

def generate_training_features(ticker="SPY", period="5y"):
    """Generate features for model training from historical data"""
    # Get historical data
    price_history = df.get_price_history(ticker, period=period)
    
    # Calculate features
    data = pd.DataFrame()
    
    # Price-based features
    data['returns_1d'] = price_history['Close'].pct_change()
    data['returns_5d'] = price_history['Close'].pct_change(5)
    data['returns_20d'] = price_history['Close'].pct_change(20)
    
    # Volatility
    data['volatility_20d'] = data['returns_1d'].rolling(window=20).std() * np.sqrt(252)
    
    # Moving averages
    data['sma_20'] = price_history['Close'].rolling(window=20).mean()
    data['sma_50'] = price_history['Close'].rolling(window=50).mean()
    data['sma_ratio'] = data['sma_20'] / data['sma_50']
    
    # Volume features
    data['volume_change'] = price_history['Volume'].pct_change()
    data['volume_ma_ratio'] = price_history['Volume'] / price_history['Volume'].rolling(window=20).mean()
    
    # Higher-order moments
    data['skewness'] = data['returns_1d'].rolling(window=20).skew()
    data['kurtosis'] = data['returns_1d'].rolling(window=20).kurt()
    
    # Create target variable (whether price goes up in next 30 days)
    data['target'] = (price_history['Close'].shift(-30) > price_history['Close']).astype(int)
    
    # Drop NaN values
    data = data.dropna()
    
    return data

def train_model():
    """Train or retrain the machine learning model"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Training new model...")
    
    try:
        # Generate training data
        logger.info("Generating training features...")
        train_data = generate_training_features("SPY", "5y")
        
        # Split features and target
        X = train_data.drop('target', axis=1)
        y = train_data['target']
        
        # Log data shape
        logger.info(f"Training data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        logger.info("Fitting RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Save model and metadata
        logger.info(f"Saving model to {MODEL_PATH}")
        joblib.dump(model, MODEL_PATH)
        
        # Generate model hash based on shape of training data
        model_hash = hashlib.md5(str(X.shape).encode()).hexdigest()
        
        # Save metadata
        metadata = pd.DataFrame({
            'timestamp': [datetime.now().isoformat()],
            'accuracy': [accuracy],
            'model_hash': [model_hash],
            'features': [list(X.columns)]
        })
        metadata.to_json(MODEL_METADATA_PATH)
        logger.info(f"Model metadata saved to {MODEL_METADATA_PATH}")
        
        return model, accuracy
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a simple fallback model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), np.array([0]))
        accuracy = 0.5
        
        # Set the feature names explicitly for the untrained model
        feature_names = [
            'moneyness', 'iv', 'hv', 'iv_hv_spread', 'days_to_expiry',
            'volume_ma_ratio', 'returns_5d', 'returns_20d', 'volatility_20d',
            'sma_ratio', 'skewness', 'kurtosis'
        ]
        model.feature_names_in_ = np.array(feature_names)
        
        return model, accuracy

def load_or_train_model(force_retrain=False):
    """Load existing model or train new one if needed"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Checking model status, force_retrain={force_retrain}")
    should_retrain = force_retrain
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_METADATA_PATH):
        logger.info(f"Model files missing - MODEL_PATH exists: {os.path.exists(MODEL_PATH)}, MODEL_METADATA_PATH exists: {os.path.exists(MODEL_METADATA_PATH)}")
        should_retrain = True
    else:
        # Check model age and accuracy
        try:
            metadata = pd.read_json(MODEL_METADATA_PATH)
            last_trained = datetime.fromisoformat(metadata['timestamp'].iloc[0])
            accuracy = metadata['accuracy'].iloc[0]
            
            days_since_trained = (datetime.now() - last_trained).days
            logger.info(f"Model was last trained {days_since_trained} days ago with accuracy {accuracy}")
            
            if days_since_trained > RETRAIN_THRESHOLD_DAYS or accuracy < MIN_ACCURACY_THRESHOLD:
                logger.info(f"Model needs retraining: age threshold={days_since_trained > RETRAIN_THRESHOLD_DAYS}, accuracy threshold={accuracy < MIN_ACCURACY_THRESHOLD}")
                should_retrain = True
        except Exception as e:
            logger.error(f"Error checking model metadata: {str(e)}")
            should_retrain = True
    
    if should_retrain:
        logger.info("Retraining model...")
        try:
            return train_model()
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            # Create a simple fallback model if training fails
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Just return a default model with a placeholder accuracy
            return model, 0.5
    else:
        # Load existing model
        try:
            logger.info(f"Loading existing model from {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
            metadata = pd.read_json(MODEL_METADATA_PATH)
            accuracy = metadata['accuracy'].iloc[0]
            logger.info(f"Model loaded successfully with accuracy {accuracy}")
            return model, accuracy
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Create a simple fallback model if loading fails
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Just return a default model with a placeholder accuracy
            return model, 0.5

def prepare_option_features(option, price_history, option_type):
    """Prepare features for a single option contract"""
    current_price = price_history['Close'].iloc[-1]
    
    # Basic option properties
    strike = option['strike']
    iv = option['impliedVolatility']
    days_to_expiry = 30  # Estimate if not available
    
    # Calculate moneyness
    if option_type == 'call':
        moneyness = current_price / strike - 1
    else:  # put
        moneyness = strike / current_price - 1
    
    # Calculate historical volatility
    hv = df.get_historical_volatility(price_history).iloc[-1]
    
    # Create feature vector
    features = pd.DataFrame({
        'moneyness': [moneyness],
        'iv': [iv],
        'hv': [hv],
        'iv_hv_spread': [iv_hv_spread(iv, hv)],
        'days_to_expiry': [days_to_expiry],
        'volume_ma_ratio': [price_history['Volume'].iloc[-1] / price_history['Volume'].iloc[-20:].mean()],
        'returns_5d': [price_history['Close'].pct_change(5).iloc[-1]],
        'returns_20d': [price_history['Close'].pct_change(20).iloc[-1]],
        'volatility_20d': [price_history['Close'].pct_change().iloc[-20:].std() * np.sqrt(252)],
        'sma_ratio': [price_history['Close'].rolling(window=20).mean().iloc[-1] / 
                     price_history['Close'].rolling(window=50).mean().iloc[-1]],
        'skewness': [price_history['Close'].pct_change().iloc[-20:].skew()],
        'kurtosis': [price_history['Close'].pct_change().iloc[-20:].kurt()]
    })
    
    return features

def predict_option_success(option, price_history, model, option_type):
    """Predict probability of success for an option contract"""
    # Prepare features
    features = prepare_option_features(option, price_history, option_type)
    
    # Handle missing features
    missing_cols = set(model.feature_names_in_) - set(features.columns)
    for col in missing_cols:
        features[col] = 0
    
    # Reorder columns to match model's expected order
    features = features[model.feature_names_in_]
    
    # Make prediction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probability = model.predict_proba(features)[0][1]  # Probability of class 1 (success)
    
    return probability

def calculate_expected_return(option, win_probability, option_type):
    """Calculate expected return based on option price and win probability"""
    price = option['lastPrice']
    strike = option['strike']
    
    # Simple expected return calculation
    if option_type == 'call':
        # For calls: probability of profit * (2*premium) - (1-probability) * premium
        expected_return = win_probability * (2 * price) - (1 - win_probability) * price
    else:  # put
        # For puts: similar calculation
        expected_return = win_probability * (2 * price) - (1 - win_probability) * price
    
    # Normalize by price
    expected_return_pct = expected_return / price
    
    return expected_return_pct

def add_predictions(options_df, price_history, fundamentals, option_type):
    """Add prediction columns to the options dataframe"""
    # Convert to DataFrame if it's a list of dicts
    if isinstance(options_df, list):
        options_df = pd.DataFrame(options_df)
    
    # Load or train model
    model, _ = load_or_train_model()
    
    # Add win probability column
    options_df['win_probability'] = options_df.apply(
        lambda x: predict_option_success(x, price_history, model, option_type), axis=1
    )
    
    # Add expected return column
    options_df['expected_return'] = options_df.apply(
        lambda x: calculate_expected_return(x, x['win_probability'], option_type), axis=1
    )
    
    return options_df
