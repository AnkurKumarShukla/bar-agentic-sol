from ..models.state_models import SupervisorState
from ..agents.base import build_agent_state
# from ..tools.tool_wrappers import invoke_agent
from ..mcp.clients import init_clients

from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


from langchain.chat_models import init_chat_model

llm=init_chat_model("openai:gpt-4o-mini")

async def financial_agent_node(state: SupervisorState) -> SupervisorState:
    """Run the financial agent with the current task and update state."""
    sysprompt_fin_agent = f"""
You are an advanced financial analysis AI assistant equipped with specialized tools
to access and analyze financial data. Your primary function is to help users with
financial analysis
You are a financial assistant. You can call tools to answer user questions.

If the user asks for stock data, institutional holders, or financial metrics, use the appropriate tool.

Call a tool with the correct arguments. Do not just answer without trying to use tools.

tools you have :
    get_stock_profile,
    get_stock_price_data,
    get_financial_statements,
    get_earnings_data,
    get_dividends_and_splits,
    get_analyst_recommendations,
    get_institutional_holders,
    get_options_chain, 

context so far is : {str(state.context)}

- before making decision go through context thoroughly
- analyse context and past decisions to avoid redundant calls.
Remember, your goal is to provide accurate, insightful financial analysis to
help users make informed decisions. Always maintain a professional and objective tone in your responses.
For stocks in India as suffics to ticker symbol  as '.NS' for NSE   and '.BO' for BSE

give respose what tool to call and get result from that tool. Just do what you can do .
"""
    tools = await init_clients()
    finance_tools=  tools["financial_tools"]
    finance_agent = create_react_agent(
        llm.bind_tools(finance_tools ,parallel_tool_calls=False),
        tools=finance_tools,
        prompt=sysprompt_fin_agent,
        name="finance_agent",
    )
    if not state.current_task:
        return state  # no task assigned, nothing to do

    # Run the financial agent on the supervisor's current task
    # Ensure your agent is the ReAct agent you created earlier
    input = {"messages": [{"role": "user", "content": state.current_task}]}
    print("this is context ",state.context)
    result = await finance_agent.ainvoke(input=input)
    
    # Convert the messages from the agent into AgentState
    agent_state = build_agent_state(result["messages"], agent_name="finance_agent")

    # Include agent name
    # agent_state.agent_name = "FinanceAgent"

    # Add agent state to SupervisorState (flat array in sequential order)
    state.agent_states.append(agent_state)

    # Update context for downstream use
    agent_count = sum(1 for s in state.agent_states if s.agent_name == "finance_agent")
    state.context[f"finance_agent_step{agent_count}"] = agent_state.agent_output

    # Clear current_task (supervisor will decide next)
    state.current_task = None

    return state


# state = SupervisorState(user_query="Check stock info")
# state.current_task = "what is TCS stock price"

# # Run financial agent node
# updated_state = financial_agent_node(state)

# # Show result
# # print(updated_state.dump_json(indent=2))