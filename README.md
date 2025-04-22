# Options Analysis & Strategy Recommender

A sophisticated Streamlit-based options analysis application that leverages machine learning to provide intelligent strategy recommendations and insights for traders.

## Features

- Historical price visualization with candlestick charts and volume indicators
- Options chain analysis with success probability predictions
- Strategy recommendations based on market outlook and risk tolerance
- Volatility analysis comparing implied vs historical volatility
- Interactive payoff diagrams for recommended strategies
- Background cache refreshing for popular tickers

## Technical Components

- **Streamlit**: Web framework for the user interface
- **Python**: Core programming language
- **yfinance**: For fetching stock and options data
- **Pandas/NumPy**: For data manipulation
- **Plotly**: For interactive visualizations
- **scikit-learn**: For machine learning predictions

## Running the App

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install streamlit pandas numpy plotly yfinance scikit-learn joblib apscheduler
   ```
3. Run the app:
   ```
   streamlit run main.py
   ```

## Static Website Version

A static version of this app has been created using Stlite, which allows deployment to GitHub Pages. The static version is located in the `static_site` directory.

### Deploying to GitHub Pages

See the [README.md](static_site/README.md) in the static_site directory for detailed instructions on deploying the app to GitHub Pages.

## Project Structure

- `main.py`: Main application entry point
- `data_fetcher.py`: Handles stock and options data retrieval
- `analysis.py`: Performs options analysis with ML predictions
- `strategist.py`: Generates strategy recommendations
- `utils.py`: Contains visualization functions and scheduler
- `static_site/`: Static website version for GitHub Pages

## Limitations

The app relies on Yahoo Finance API, which has some limitations:
- Data may not be real-time
- Some stocks may not have options data available
- API rate limits may apply

## Future Improvements

- Add more sophisticated ML models for predictions
- Implement portfolio analysis features
- Add option spread strategies visualization
- Include fundamental analysis metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.