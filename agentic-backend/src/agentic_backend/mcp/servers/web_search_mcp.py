from mcp.server.fastmcp import FastMCP
import yfinance as yf
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

mcp = FastMCP("Web Search MCP Server")
# Use stderr for logging to avoid breaking JSON-RPC on stdout
print("loaded environment variables", file=sys.stderr, flush=True)

@mcp.tool()
def web_search(query: str) -> dict:
    """
    use this when you whant to search some information on web only if you do not have tools and knowledge to answer the question.
    Uses the Serper API to perform a Google-like web search.
    Returns the top search results including title, link, and snippet.
    """

    
    import requests
    import os

    try:
        # Get API key from environment variable
        SERPER_API_KEY = os.getenv("SERPAPI_API_KEY")
        if not SERPER_API_KEY:
            return {"error": "SERPAPI_API_KEY is not set in environment variables."}

        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "gl": "us",  # optional: country
            "hl": "en"   # optional: language
        }

        response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        results = []
        for result in data.get("organic", [])[:5]:  # top 5 results
            results.append({
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet")
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    mcp.run()