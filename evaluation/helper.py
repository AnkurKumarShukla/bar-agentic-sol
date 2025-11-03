import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Add MCP path
mcp_dir = r"C:\Users\Divya\TCS\AI\bar-agentic-sol\agentic-backend\src\agentic_backend\mcp"
sys.path.insert(0, mcp_dir)

from clients import init_clients

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Global variable to cache MCP tools
_mcp_tools_cache = None

async def _load_mcp_tools():
    """Load MCP tools once and cache them"""
    global _mcp_tools_cache
    if _mcp_tools_cache is None:
        print("Loading MCP tools...")
        tools_dict = await init_clients()
        _mcp_tools_cache = {
            "financial": tools_dict["financial_tools"],
            "web_search": tools_dict["web_search_tools"],
            "sentiment": tools_dict["sentiment_tools"],
        }
        print(f"✓ Loaded MCP tools successfully")
    return _mcp_tools_cache

def _get_tools_for_agent(agent_name: str, mcp_tools: dict) -> list:
    """Get tools based on agent type"""
    if agent_name == "finance_agent":
        return mcp_tools["financial"]
    elif agent_name == "web_search_agent":
        return mcp_tools["web_search"]
    elif agent_name == "news_sentiment_agent":
        return mcp_tools["sentiment"]
    elif agent_name == "supervisor":
        return (
            mcp_tools["financial"] + 
            mcp_tools["web_search"] + 
            mcp_tools["sentiment"]
        )
    else:
        return mcp_tools["financial"]

def _get_expected_tools_list(agent_name: str) -> list:
    """Get the expected tool names for each agent type"""
    tool_mapping = {
        "finance_agent": ["get_stock_profile", "get_stock_price_data", "get_financial_statements", 
                         "get_earnings_data", "get_dividends_and_splits", "get_analyst_recommendations", 
                         "get_institutional_holders", "get_options_chain"],
        "web_search_agent": ["web_search"],
        "news_sentiment_agent": ["fetch_news"],
        "rag_agent": ["semantic_search"],
        "supervisor": ["finance_agent", "web_search_agent", "news_sentiment_agent"]
    }
    return tool_mapping.get(agent_name, [])
async def safe_tool_call(tool, args):
    try:
        # Prefer async invocation if available
        if hasattr(tool, "ainvoke"):
            return await asyncio.wait_for(tool.ainvoke(args), timeout=60)
        else:
            return await asyncio.wait_for(asyncio.to_thread(tool.invoke, args), timeout=60)
    except asyncio.TimeoutError:
        return {"error": "Tool execution timeout after 60s"}

async def infer_reference_simple(agent_name: str, user_query: str, verbose: bool = True) -> dict:
    """
    SIMPLER APPROACH: Directly call tools without ReAct agent loop
    This avoids recursion issues by manually controlling tool execution
    
    Args:
        agent_name: Type of agent
        user_query: The user's question/query
        verbose: If True, print execution details
        
    Returns:
        dict: Reference answer with metadata
    """
    
    # Load MCP tools
    mcp_tools = await _load_mcp_tools()
    tools = _get_tools_for_agent(agent_name, mcp_tools)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}")
        print(f"Available Tools: {[t.name for t in tools]}")
        print(f"Query: {user_query}")
        print(f"{'='*60}\n")
    
    try:
        # STEP 1: Use LLM with function calling to determine which tool to use
        print("🤔 STEP 1: Determining which tool to use...")
        
        llm_with_tools = llm.bind_tools(tools)
        
        planning_prompt = f"""Analyze this query and determine which tool to call with what parameters.

Query: {user_query}
Agent Type: {agent_name}

Available tools: {[t.name for t in tools]}

Notes:
3.**Agent & Tools Selection Criteria**:
    - If agent_name is "news_sentiment_agent", focus on news sentiment analysis and summarise as it is positive , negative or mixed etc.
    - If agent_name is "finance_agent", focus on financial analysis
    - If agent_name is "web_search_agent", focus on internet/news queries
    - If agent_name is "supervisor", include all relevant agents
4.**Tool Lists You Have**:
    - Finance Agent: [get_stock_profile, get_stock_price_data, get_financial_statements, get_earnings_data, get_dividends_and_splits, get_analyst_recommendations, get_institutional_holders, get_options_chain]
    - Web Search Agent: [web_search]
    - RAG Agent: [semantic_search]
    - Supervisor: [finance_agent, web_search_agent, news_sentiment_agent]
    - News Sentiment Agent: [fetch_news]

For stocks in UK symbols for stock on LSE are appended with ".L". If not defined consider LSE as primary stock exchange
Call the appropriate tool now with correct parameters."""

        response = llm_with_tools.invoke(planning_prompt)
        
        if verbose:
            print(f"LLM Response type: {type(response)}")
            print(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}")
        
        # STEP 2: Execute the tool calls
        if not response.tool_calls:
            print("⚠️  No tool calls generated by LLM")
            return {
                "reference": "Error: LLM did not generate tool calls",
                "reference_topic": "",
                "agent_name": agent_name,
                "expected_agent": agent_name,
                "expected_tools": _get_expected_tools_list(agent_name),
                "error": "No tool calls generated"
            }
        
        print(f"\n🔧 STEP 2: Executing {len(response.tool_calls)} tool call(s)...")
        
        tool_results = []
        tools_dict = {tool.name: tool for tool in tools}
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n📞 Calling: {tool_name}")
            print(f"   Args: {json.dumps(tool_args, indent=2)}")
            
            if tool_name in tools_dict:
                tool = tools_dict[tool_name]
                
                # Execute the tool
                try:
                    result = await safe_tool_call(tool, tool_args)

                    print(f"✅ Result received: {str(result)[:200]}...")
                    
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result
                    })
                except Exception as e:
                    print(f"❌ Tool execution error: {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "error": str(e)
                    })
            else:
                print(f"⚠️  Tool {tool_name} not found in available tools")
        
        # STEP 3: Generate structured response using tool results
        print("\n📝 STEP 3: Generating structured reference answer...")
        
        structuring_prompt = f"""Based on the tool results below, create a comprehensive reference answer.
        do not hallucinate any data and use only the tool results provided.

        Original Query: {user_query}
        Agent Type: {agent_name}

        Tool Results:
        {json.dumps(tool_results, indent=2)}
        IMPORTANT:
        - ***Very Important***: If expected_agent is "supervisor", only include the most relevant routed agent(s) from [finance_agent, web_search_agent, news_sentiment_agent] in expected_tools and use this only for generating the expected_tools in the final JSON.Do not add any other tools.
        - Base your answer ONLY on the actual tool results above
        - Act like a financial advisor and repond based on the facts and results from tools.
        - Include specific numbers, dates, and facts from the results
        - If sentiment analysis, be clear about whether outlook is positive/negative/mixed
        - Do NOT hallucinate or make up information
        - Respond ONLY with valid JSON, no extra text

        Create a JSON response with this EXACT format:
        {{
            "reference": "Detailed answer based on tool results (3-6 bullet points or detailed paragraph) as a summary. Include specific data points and do not miss any points from the tool results.",
            "reference_topic": "Brief topic summary (e.g., 'news sentiment analysis', 'stock performance')",
            "expected_agent": "{agent_name}",
            "expected_tools": ["list", "of", "expected", "tools"] If expected_agent is "supervisor", only include the most relevant routed agent(s) from [finance_agent, web_search_agent, news_sentiment_agent] in expected_tools and use this only for generating the expected_tools in the final JSON.Do not add any other tools.,
            
        }}
        """

        

        structure_response = llm.invoke(structuring_prompt)
        response_text = structure_response.content
        
        # Parse JSON
        try:
            # Clean up markdown code blocks if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            result_json = json.loads(response_text)
            
            # Add execution metadata
           
            
            if verbose:
                print("\n✅ Successfully generated reference answer")
            
            return result_json
            
        except json.JSONDecodeError as je:
            print(f"⚠️  JSON Parse Error: {je}")
            print(f"Raw response: {response_text[:500]}...")
            
            return {
                "reference": response_text,
                "reference_topic": "",
                "agent_name": agent_name,
                "expected_agent": agent_name,
                "expected_tools": _get_expected_tools_list(agent_name),
                "tools_used": [tr["tool"] for tr in tool_results],
                "raw_response": response_text,
                "parse_error": str(je)
            }
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "reference": user_query,
            "reference_topic": "",
            "agent_name": agent_name,
            "expected_agent": agent_name,
            "expected_tools": _get_expected_tools_list(agent_name),
            "error": str(e)
        }


def generate_reference(agent_name: str, user_query: str, verbose: bool = True) -> dict:
    """Synchronous wrapper"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    infer_reference_simple(agent_name, user_query, verbose)
                )
                return future.result()
        else:
            return asyncio.run(infer_reference_simple(agent_name, user_query, verbose))
    except RuntimeError:
        return asyncio.run(infer_reference_simple(agent_name, user_query, verbose))