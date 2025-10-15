import asyncio
from langgraph.graph import StateGraph, END
from ..models.state_models import SupervisorState
from ..agents.supervisor import supervisor_node
from ..agents.finance import financial_agent_node
from ..agents.websearch import websearch_agent_node
from ..agents.news_sentiment import news_sentiment_agent_node
from langgraph.checkpoint.memory import InMemorySaver
# Global singletons
checkpointer = InMemorySaver()

_graph: StateGraph = None
_app = None

# Router function
def router(state: SupervisorState):
    if state.current_task is None:
        return "END"
    sel = state.decisions[-1].selected_agent.lower().replace(" ", "_")
    if sel in ("finance_agent", "financeagent", "finance"):
        return "finance_agent"
    if sel in ("websearch_agent", "websearch", "websearchagent"):
        return "websearch_agent"
    if sel in ("sentiment_agent", "news_sentiment_agent", "sentimentagent", "newsentiment"):
        return "sentiment_agent"
    return "END"


def build_graph():
    """
    Build and cache the graph once, using the persistent checkpointer.
    """
    global  _graph, _app
    if _app is None:
        g = StateGraph(SupervisorState)
        g.add_node("supervisor", supervisor_node)
        g.add_node("finance_agent", financial_agent_node)
        g.add_node("websearch_agent", websearch_agent_node)
        g.add_node("sentiment_agent", news_sentiment_agent_node)
        g.set_entry_point("supervisor")

        # Conditional edges
        g.add_conditional_edges("supervisor", router, {
            "finance_agent": "finance_agent",
            "websearch_agent": "websearch_agent",
            "sentiment_agent": "sentiment_agent",
            "END": END
        })

        # Return edges to supervisor
        g.add_edge("finance_agent", "supervisor")
        g.add_edge("websearch_agent", "supervisor")
        g.add_edge("sentiment_agent", "supervisor")

        _graph = g
        _app = g.compile(checkpointer=checkpointer)

    return _app


async def run_sync(state: SupervisorState, thread_id: str,**kwargs):
    """
    Execute the graph step-by-step and yield intermediate states.
    To fetch user_id : 
    ``` user_id=kwargs.get("user_id")```
    """
    print(kwargs.get("user_id"))

    config = {"configurable": {"thread_id": thread_id}}
    # print("=============================")
    # print(checkpointer.get_tuple(config))
    # print("==============================")
    app = build_graph()
    final_state = None
    

    async for s in app.astream(state, config):
        yield s
        final_state = s




