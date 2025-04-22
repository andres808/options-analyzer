from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import sys

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the data fetcher and other dependencies
from data_fetcher_browser import get_price_history, get_options_chain, get_fundamentals

app = FastAPI()

@app.get("/")
async def root():
    """Redirect to the index.html file"""
    return RedirectResponse(url="/index.html")

@app.get("/api/price/{ticker}")
async def get_price_data(ticker: str, period: str = "1y"):
    """Get price history data for a ticker"""
    data = get_price_history(ticker, period=period)
    return data.to_dict(orient="records")

@app.get("/api/options/{ticker}")
async def get_options_data(ticker: str):
    """Get options data for a ticker"""
    return get_options_chain(ticker)

@app.get("/api/fundamentals/{ticker}")
async def get_fundamentals_data(ticker: str):
    """Get fundamentals data for a ticker"""
    return get_fundamentals(ticker)

# Mount the static files (HTML, CSS, JS, etc.)
app.mount("/", StaticFiles(directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), name="static")