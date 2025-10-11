import os
# import dotenv
import requests
import json
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import sys
import importlib.util
# Pass the absolute path of the prompt module
prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompt.py")
spec = importlib.util.spec_from_file_location("prompt", prompt_path)
prompt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_module)
sentiment_prompt = prompt_module.sentiment_prompt

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
SERPER_API_KEY = os.getenv("SERPAPI_API_KEY", "your_api_key_here")
# print(f"Using SERPER_API_KEY: {OPENAI_API_KEY}")

app = FastMCP("market-sentiment-mcp")


def scrape_article(url: str) -> str:
    """Scrape text content from a news article."""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:3000]  # trim to avoid overload
    except Exception as e:
        return f"Error scraping {url}: {e}"

@app.tool(
    name="fetch_news",
    description="Fetch latest market news and provide indepth research by scraping article content",
    tags={"finance", "crypto news", "tocken news", "market sentiment", "sentiment analysis"},
)
def fetch_news_sentiment(query: str, gl: str = "us") -> dict:
    """
    Fetch latest market news and give detailed news for sentiment analysis .
    """
    summerise_model= ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    url = "https://google.serper.dev/news"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = json.dumps({"q": query, "gl": gl,"tbs": "qdr:d"})

    resp = requests.post(url, headers=headers, data=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    enriched_articles = []
    content=""
    for article in data.get("news", [])[:5]:
        title = article.get("title")
        link = article.get("link")
        snippet = article.get("snippet")
        date = article.get("date")
        source = article.get("source")

        indepth_text = scrape_article(link)

        enriched_articles.append({
            "title": title,
            "link": link,
            "snippet": snippet,
            "date": date,
            "source": source,
        })
        
    #  append all indepth research to summary
        
        content+=indepth_text+"\n"
    # want to  sumerise this using open ai  call 
    
    analysis_prompt=sentiment_prompt(content)
    analysis= summerise_model.invoke([{"role":"user","content":analysis_prompt}])
    
    return {
        "query": query,
        "results": enriched_articles,
        "count": len(enriched_articles),
        "summary":analysis.content
    }



if __name__ == "__main__":
    # Use stderr for logging to avoid breaking JSON-RPC on stdout
    import sys
    print("[SERVER] Starting market sentiment MCP...", file=sys.stderr, flush=True)
    app.run()
