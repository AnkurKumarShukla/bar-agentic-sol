from ..models.state_models import SupervisorState
from ..agents.base import build_agent_state
# from ..tools.tool_wrappers import invoke_agent
from ..mcp.clients import init_clients
from ..prompt import news_sentiment_prompt
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
llm=init_chat_model("openai:gpt-4o-mini")



async def news_sentiment_agent_node(state: SupervisorState) -> SupervisorState:
    """Run the news sentiment agent with the current task and update state."""
    news_sentiment_prompt= f"""You are a expert market news sentiment analysis agent. 
        context so far is : {str(state.context)}

- before making decision go through context thoroughly
- analyse context and past decisions to avoid redundant calls.
        Analyze the sentiment of the news article and provide a summary of its sentiment impact on the market."""
    tools = await init_clients()
    news_sentiment_tools= tools["sentiment_tools"]
    sentiment_agent = create_react_agent(
        llm.bind_tools(news_sentiment_tools ,parallel_tool_calls=False),
        tools=news_sentiment_tools,
        prompt=news_sentiment_prompt,
        name="news_sentiment_agent",
    )
    if not state.current_task:
        return state  # no task assigned, nothing to do

    # Run the news sentiment agent on the supervisor's current task
    # Ensure your agent is the ReAct agent you created earlier
    input = {"messages": [{"role": "user", "content": state.current_task}]}
    result = await sentiment_agent.ainvoke(input=input, context=state.context)
    
    # Convert the messages from the agent into AgentState
    agent_state = build_agent_state(result["messages"], agent_name="news_sentiment_agent")

    # Include agent name
    agent_state.agent_name = "news_sentiment_agent"

    # Add agent state to SupervisorState (flat array in sequential order)
    state.agent_states.append(agent_state)

    # Update context for downstream use
    agent_count = sum(1 for s in state.agent_states if s.agent_name == "news_sentiment_agent")
    state.context[f"news_sentiment_agent_step{agent_count}"] = agent_state.agent_output

    # Clear current_task (supervisor will decide next)
    state.current_task = None

    return state