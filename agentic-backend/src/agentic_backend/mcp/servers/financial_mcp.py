from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd
import numpy as np

mcp = FastMCP("Financial MCP Server")

# --- Utility: Ensure JSON safe types ---
def safe_dict(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.datetime64, pd.Timestamp)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: safe_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_dict(v) for v in obj]
    return obj


# --- Tools ---
@mcp.tool()
def get_stock_profile(ticker_symbol: str) -> dict:
    """Fetches basic company profile, stock summary, and metadata."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = safe_dict(ticker.info)
        return {
            "symbol": info.get("symbol"),
            "short_name": info.get("shortName"),
            "long_name": info.get("longName"),
            "industry": info.get("industry"),
            "sector": info.get("sector"),
            "exchange": info.get("exchange"),
            "currency": info.get("currency"),
            "website": info.get("website"),
            "phone": info.get("phone"),
            "employees": info.get("fullTimeEmployees"),
            "summary": info.get("longBusinessSummary")
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_stock_price_data(ticker_symbol: str) -> dict:
    """Returns current and recent price data of a stock."""
    try:
        info = safe_dict(yf.Ticker(ticker_symbol).info)
        return {
            "current_price": info.get("currentPrice"),
            "previous_close": info.get("previousClose"),
            "open_price": info.get("open"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap")
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_financial_statements(ticker_symbol: str) -> dict:
    """Retrieves key financial statements."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        return {
            "income_statement": safe_dict(ticker.financials),
            "balance_sheet": safe_dict(ticker.balance_sheet),
            "cashflow_statement": safe_dict(ticker.cashflow)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_earnings_data(ticker_symbol: str) -> dict:
    """Get the latest income statement data."""
    try:
        stock = yf.Ticker(ticker_symbol)
        income_stmt = stock.financials

        if income_stmt.empty:
            return {"warning": f"No income statement data found for {ticker_symbol}"}

        latest = safe_dict(income_stmt.iloc[:, 0])

        return {
            "ticker": ticker_symbol,
            "total_revenue": latest.get('Total Revenue', 0),
            "gross_profit": latest.get('Gross Profit', 0),
            "operating_income": latest.get('Operating Income', 0),
            "net_income": latest.get('Net Income', 0),
            "earnings_per_share": latest.get('Basic EPS', 0)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_dividends_and_splits(ticker_symbol: str) -> dict:
    """Returns dividend and stock split history."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        return {
            "dividends": safe_dict(ticker.dividends),
            "splits": safe_dict(ticker.splits),
            "actions": safe_dict(ticker.actions)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_analyst_recommendations(ticker_symbol: str) -> dict:
    """Returns analyst price targets and recommendations."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = safe_dict(ticker.info)
        return {
            "recommendation": info.get("recommendationKey"),
            "target_low_price": info.get("targetLowPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_mean_price": info.get("targetMeanPrice"),
            "analyst_ratings": safe_dict(ticker.recommendations.tail(5))
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_institutional_holders(ticker_symbol: str) -> dict:
    """Returns major and institutional stockholders."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        return {
            "institutional_holders": safe_dict(ticker.institutional_holders),
            "major_holders": safe_dict(ticker.major_holders)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_options_chain(ticker_symbol: str, expiry_date: str) -> dict:
    """Returns options chain for a given expiration date (YYYY-MM-DD)."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        chain = ticker.option_chain(expiry_date)
        return {
            "calls": safe_dict(chain.calls),
            "puts": safe_dict(chain.puts)
        }
    except Exception as e:
        return {"error": str(e)}

# --- Run MCP Server ---
if __name__ == "__main__":
    print("[SERVER] Starting Financial MCP...", flush=True)
    mcp.run()
