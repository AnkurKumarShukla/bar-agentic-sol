import os
import asyncio
from langgraph.graph import StateGraph, END
from ..models.state_models import SupervisorState
from ..agents.supervisor import supervisor_node
from ..agents.finance import financial_agent_node
from ..agents.websearch import websearch_agent_node
from ..agents.news_sentiment import news_sentiment_agent_node
from ..services.persistence import save_state

# Build the graph once (synchronous graph)
_graph = None
_app = None
from graphviz import Digraph

from graphviz import Digraph

from IPython.display import Image, display
import tempfile
from graphviz import Digraph



def router(state: SupervisorState):
    if state.current_task is None:
        return "END"
# Use a standardized lower-case, underscore format for comparison.
    sel = state.decisions[-1].selected_agent.lower().replace(" ", "_")
    if sel in ("finance_agent", "financeagent", "finance"):
        return "finance_agent"
    if sel in ("websearch_agent", "websearch", "websearchagent"):
        return "websearch_agent"
    if sel in ("sentiment_agent", "news_sentiment_agent", "sentimentagent", "newsentiment"):
        return "sentiment_agent"
    return "END"
def build_graph():
    global _graph, _app
    print("build graph wala ", _graph)
    if _graph is None:
        g = StateGraph(SupervisorState)
        g.add_node("supervisor", supervisor_node)
        g.add_node("finance_agent", financial_agent_node)
        g.add_node("websearch_agent", websearch_agent_node)
        g.add_node("sentiment_agent", news_sentiment_agent_node)
        g.set_entry_point("supervisor")
        # map routing labels to node names
        g.add_conditional_edges("supervisor", router, {"finance_agent": "finance_agent", "websearch_agent": "websearch_agent","sentiment_agent":"sentiment_agent" ,"END": END})

        # each agent returns to supervisor
        g.add_edge("finance_agent", "supervisor")
        g.add_edge("websearch_agent", "supervisor")
        g.add_edge("sentiment_agent", "supervisor")

        _graph = g
        
        _app =  g.compile()
        
    
    return _app


async def run_sync(state: SupervisorState) -> SupervisorState:
    app = build_graph()
    print("running graph...")

    # This will execute the graph step by step until "END"
    final_state = None
    async for s in app.astream(state):
        yield s
        final_state = s

    # persist final result
    print("Final state:", final_state)
    try:
        save_state(final_state)
    except Exception:
        pass

    



# async def run_background(state: SupervisorState):
#     loop = asyncio.get_running_loop()
#     result = await loop.run_in_executor(None, run_sync, state)
#     return result
