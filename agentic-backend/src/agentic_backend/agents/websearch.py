from ..models.state_models import SupervisorState
from ..agents.base import build_agent_state
# from ..tools.tool_wrappers import invoke_agent
from ..mcp.clients import init_clients

from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# finance_tools= tools["financial_tools"]
from langchain.chat_models import init_chat_model

llm=init_chat_model("openai:gpt-4o-mini")

async def websearch_agent_node(state: SupervisorState) -> SupervisorState:
    """Run the web search agent with the current task and update state."""
    sysprompt_web_search = f"""
You are a Web Search AI Assistant.
You can access the internet via the web search tool to gather the most relevant and 
comprehensive information.
context so far is : {str(state.context)}

- before making decision go through context thoroughly
- analyse context and past decisions to avoid redundant calls.
Guidelines:
1. Always perform a web search if the user asks for real-time, general knowledge, or 
   non-financial information.
2. Fetch detailed results without omitting important context.
3. Summarize findings clearly and objectively, avoiding unnecessary fluff.
4. Always ensure sources are reliable and up-to-date.

Your goal: Deliver accurate, detailed, and useful search results from the web.
"""
    tools = await init_clients()
    web_search_tools=  tools["web_search_tools"]
    websearch_agent = create_react_agent(
        llm.bind_tools(web_search_tools, parallel_tool_calls=False),
        tools=web_search_tools,
        prompt=sysprompt_web_search,
        name="websearch_agent",
    )
    if not state.current_task:
        return state  # no task assigned

    # Run the web search agent
    # Ensure your agent is the ReAct agent created earlier
    input= {
        "messages": [{"role": "user", "content": state.current_task}]
    }
    result = await websearch_agent.ainvoke(input=input, context=state.context)

    # Convert messages to structured AgentState
    agent_state = build_agent_state(result["messages"],agent_name="websearch_agent")
    # agent_state.agent_name = "websearchAgent"

    # Add agent state to SupervisorState (flat array in sequential order)
    state.agent_states.append(agent_state)

    # Update context with latest output
    agent_count = sum(1 for s in state.agent_states if s.agent_name == "websearch_agent")
    state.context[f"websearch_agent_step{agent_count}"] = agent_state.agent_output

    # Clear current task (Supervisor will assign next)
    state.current_task = None

    return state
