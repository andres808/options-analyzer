# Options Analysis App - Vercel Deployment

This directory contains a version of the Options Analysis application optimized for Vercel deployment.

## What is this?

This is a deployment-ready version of the Options Analysis app that uses FastAPI to serve a complete client-side application. The application uses demo data generated on the server side to demonstrate options analysis and strategy recommendation features.

## How to Deploy to Vercel

1. Sign up for a [Vercel account](https://vercel.com/signup) if you don't have one already.

2. Install the Vercel CLI (optional):
   ```
   npm install -g vercel
   ```

3. Deploy using the Vercel CLI:
   ```
   cd static_site
   vercel
   ```
   
   Or simply connect your GitHub repository to Vercel:
   - Push this code to a GitHub repository
   - Import the repository in the Vercel dashboard
   - Vercel will automatically detect the project configuration

4. After a few minutes, your app will be available at your Vercel URL (`https://your-project-name.vercel.app`).

## File Structure

- `api/` - FastAPI backend code for serving data
  - `index.py` - Main API entry point for Vercel Serverless Functions
- `index.html` - The main HTML file for the client-side application
- `data_fetcher_browser.py` - Module that provides simulated stock and options data
- `analysis.py` - Module for analyzing options data
- `strategist.py` - Module for generating strategy recommendations
- `utils.py` - Utility functions for plotting and data manipulation
- `model.pkl` - Machine learning model file for options prediction
- `requirements.txt` - Python dependencies for Vercel
- `vercel.json` - Configuration file for Vercel deployment

## Key Features

- **Client-Side Application**: The HTML/JavaScript frontend provides a responsive user interface.
- **FastAPI Backend**: The API endpoints deliver simulated market data and options analysis.
- **Demo Data Generation**: Realistic stock price history, options chains, and fundamentals are generated for demonstration purposes.
- **Interactive Charts**: Uses Plotly.js to create interactive price charts, payoff diagrams, and volatility analysis.
- **Strategy Recommendations**: Provides options trading strategy recommendations based on market outlook and risk tolerance.

## Limitations

This demo version has some limitations compared to the full server version:

1. **Uses simulated data** - not connected to real market data sources
2. No data caching - fresh data is generated on each request
3. Limited ticker support - only popular tickers are included in the demo
4. No user authentication or persisted settings

## Customizing

You can customize the app by modifying the HTML, JavaScript, and Python files. Key areas for customization:

- Update the stock tickers and simulated data in `data_fetcher_browser.py`
- Modify the UI by editing the HTML and CSS in `index.html`
- Adjust strategy recommendations and analysis algorithms
- Add new features to the FastAPI backend

After making changes, deploy to Vercel again to update your live application.