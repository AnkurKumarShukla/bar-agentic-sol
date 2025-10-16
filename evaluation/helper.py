from openai import OpenAI
import os, json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# For generating reference for evaluation
def infer_reference(agent_name, user_query):
    prompt = f"""
    You are a reference generator for an investment research assistant. Your task is to generate a concise, factual reference answer and metadata for automated evaluation.

    User Query: {user_query}
    agent_name: {agent_name}

    Instructions:
    1. **Reference**: Search on web and use available information and generate answer for the user_query.Write an ideal answer that covers the key points a correct agent response should contain. Use 3-6 bullet points or a short paragraph. Do not hallucinate specific numbers or events; . Keep the tone neutral and factual.
    2. **Reference Topic**: Summarize the main theme of the query in a few words (e.g., stock analysis, investment recommendation, risk profile). If the query covers multiple topics, combine them with '&'.
    3. **Agent & Tools Selection**:
    - If agent_name is "news_sentiment_agent", focus on news sentiment analysis and use its tools.
    - If agent_name is "finance_agent", focus only on financial analysis and use finance tools.
    - If agent_name is "websearch_agent", focus on internet/news queries and use web search tools.
    - If agent_name is "rag_agent", focus on document retrieval/analysis and use semantic search tools.
    -If agent_name is "supervisor", only include the most relevant routed agent(s) from [finance_agent, web_search_agent, news_sentiment_agent] in expected_tools.
    Do not list any individual function or tool other than these agent names in expected_tools for supervisor.- Do not include agents/tools unrelated to the query.
    - Use only the tools listed below.
    4. **Tool Lists**:
    - Finance Agent: [get_stock_profile, get_stock_price_data, get_financial_statements, get_earnings_data, get_dividends_and_splits, get_analyst_recommendations, get_institutional_holders, get_options_chain]
    - Web Search Agent: [web_search]
    - RAG Agent: [semantic_search]
    - Supervisor: [finance_agent, web_search_agent, news_sentiment_agent]
    - News Sentiment Agent: [fetch_news]
    5  ***Very Important***: If expected_agent == "supervisor" then its tools will be: [finance_agent, web_search_agent, news_sentiment_agent]. dont add any other tools.
    Output Format (JSON):
    {{
    "reference": "...",
    "reference_topic": "...",
    "agent_name": {agent_name},
    "expected_agent": "...",
    "expected_tools": ["..."]
    }}

    Respond only with the JSON object.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an evaluator that extracts user intent and gives responseas reference , the correct agent, and expected tools."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {
            "reference": user_query,
            "reference_topic": "",
            "agent_name": {agent_name},
            "expected_agent": "...",
            "expected_tools": []
        }


# # helper.py
# import os
# import json
# import datetime
# import requests
# import yfinance as yf
# from bs4 import BeautifulSoup
# from openai import OpenAI
# from dotenv import load_dotenv

# # ---------------------- LOAD ENV ----------------------
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found. Add it to your .env file or environment.")

# client = OpenAI(api_key=OPENAI_API_KEY)

# # ---------------------- LIVE DATA TOOLS ----------------------

# def fetch_stock_data(ticker: str):
#     """Fetch real-time stock data using yfinance."""
#     try:
#         stock = yf.Ticker(ticker)
#         data = getattr(stock, "fast_info", {})
#         last_price = data.get("last_price")
#         market_cap = data.get("market_cap")
#         volume = data.get("volume")
#         currency = data.get("currency", "USD")
#         return {
#             "ticker": ticker,
#             "last_price": last_price,
#             "market_cap": market_cap,
#             "volume": volume,
#             "currency": currency
#         }
#     except Exception as e:
#         return {"error": f"Stock data fetch failed: {str(e)}"}

# def fetch_news(query: str, max_articles: int = 3):
#     """Fetch latest financial news using Google News RSS."""
#     try:
#         url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+stock"
#         r = requests.get(url, timeout=10)
#         soup = BeautifulSoup(r.text, "xml")
#         items = soup.find_all("item")[:max_articles]
#         if not items:
#             return [{"error": "No news found"}]
#         return [
#             {"title": item.title.text, "link": item.link.text, "published": item.pubDate.text}
#             for item in items
#         ]
#     except Exception as e:
#         return [{"error": f"News fetch failed: {str(e)}"}]

# # ---------------------- REFERENCE GENERATION ----------------------

# def infer_reference(agent_name: str, user_query: str, ticker: str = None):
#     """
#     Generate a factual reference output (JSON) using real-time stock & news data.
#     If ticker is provided, fetch stock data; otherwise just news.
#     """
#     today = datetime.date.today().isoformat()
#     stock_data = fetch_stock_data(ticker) if ticker else None
#     news_data = fetch_news(ticker or user_query)

#     factual_context = {
#         "date": today,
#         "stock_data": stock_data,
#         "news_data": news_data
#     }

#     prompt = f"""
#     You are a reference generator for an investment research assistant.
#     Produce a concise, factual reference answer based on the real-time data below.

#     User Query: {user_query}
#     Agent Name: {agent_name}

#     Real-Time Context:
#     {json.dumps(factual_context, indent=2)}

#     Instructions:
#     1. Write an ideal answer (3–6 bullet points) combining financial + news info.
#     2. Keep all numbers factual from the context above.
#     3. Identify the reference topic (e.g., "Stock performance & sentiment").
#     4. Select the expected agent and expected tools:
#         - Finance Agent → [get_stock_profile, get_stock_price_data, get_financial_statements, get_analyst_recommendations]
#         - News Sentiment Agent → [fetch_news]
#         - Web Search Agent → [web_search]
#         - Supervisor → [finance_agent, web_search_agent, news_sentiment_agent]
#     Output strictly in JSON:
#     {{
#         "reference": "...",
#         "reference_topic": "...",
#         "agent_name": "{agent_name}",
#         "expected_agent": "...",
#         "expected_tools": ["..."]
#     }}
#     """

#     response = client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[
#             {"role": "system", "content": "You generate factual investment reference outputs using provided real data."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0
#     )

#     try:
#         return json.loads(response.choices[0].message.content)
#     except Exception as e:
#         print("⚠️ JSON parse error:", e)
#         return {
#             "reference": f"Factual summary based on current data for {ticker or user_query}.",
#             "reference_topic": "investment analysis",
#             "agent_name": agent_name,
#             "expected_agent": "finance_agent",
#             "expected_tools": ["get_stock_profile", "get_stock_price_data"]
#         }

# # ---------------------- USAGE EXAMPLE ----------------------
# # reference = infer_reference("finance_agent", "How is Lloyds Bank performing this week?", "LLOY.L")
# # print(json.dumps(reference, indent=2))
