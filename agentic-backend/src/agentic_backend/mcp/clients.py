import os
from langchain_mcp_adapters.client import MultiServerMCPClient

# Base directory → src/agentic_backend/mcp
BASE_DIR = os.path.dirname(__file__)
SERVERS_DIR = os.path.join(BASE_DIR, "servers")

async def init_clients():
    financial_mcp_client = MultiServerMCPClient(
        {
            "Client Portfolio Server": {
                "command": "python",
                "args": [os.path.join(SERVERS_DIR, "financial_mcp.py")],
                "transport": "stdio",
            }
        }
    )
    financial_tools = await financial_mcp_client.get_tools()

    web_search_mcp_client = MultiServerMCPClient(
        {
            "Web Search Server": {
                "command": "python",
                "args": [os.path.join(SERVERS_DIR, "web_search_mcp.py")],
                "transport": "stdio",
            }
        }
    )
    web_search_tools = await web_search_mcp_client.get_tools()

    # rag_mcp_client = MultiServerMCPClient(
    #     {
    #         "RAG Server": {
    #             "command": "python",
    #             "args": [os.path.join(SERVERS_DIR, "rag_mcp.py")],
    #             "transport": "stdio",
    #         }
    #     }
    # )
    # rag_tools = await rag_mcp_client.get_tools()

    news_sentiment_client = MultiServerMCPClient(
        {
            "Sentiment Analysis Server": {
                "command": "python",
                "args": [os.path.join(SERVERS_DIR, "news_sentiment_mcp.py")],
                "transport": "stdio",
            }
        }
    )
    sentiment_tools = await news_sentiment_client.get_tools()

    return {
        "financial_tools": financial_tools,
        "web_search_tools": web_search_tools,
        # "rag_tools": rag_tools,
        "sentiment_tools": sentiment_tools,
    }
