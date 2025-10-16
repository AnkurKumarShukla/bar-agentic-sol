import asyncio
from langgraph.graph import StateGraph, END
from ..models.state_models import SupervisorState
from ..agents.supervisor import supervisor_node
from ..agents.finance import financial_agent_node
from ..agents.websearch import websearch_agent_node
from ..agents.news_sentiment import news_sentiment_agent_node
from langgraph.checkpoint.memory import InMemorySaver
from langfuse.langchain import CallbackHandler
from langfuse import get_client
import sys
import os
from dotenv import load_dotenv
# Global singletons
checkpointer = InMemorySaver()
load_dotenv()
langfuse = get_client()
langfuse_handler = CallbackHandler()

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

    """
    user_id = kwargs.get("user_id")
    dataset = kwargs.get("dataset")

    config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler], "metadata": {
            "langfuse_user_id": user_id,
    },}
    # print("=============================")
    # print(checkpointer.get_tuple(config))
    # print("==============================")
    app = build_graph()
    final_state = None
    

    async for s in app.astream(state, config):
        yield s
        final_state = s
    try:
        trace_id = langfuse_handler.last_trace_id
        langfuse_handler.client.flush()
        print(f"Flushed trace {trace_id} to Langfuse.")
    except Exception as e:
        print(f"Warning: could not flush Langfuse data: {e}")
    # Get absolute path to repo/evaluation
    evaluation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../evaluation"))
    # Add to sys.path if not already present
    if evaluation_path not in sys.path:
        sys.path.append(evaluation_path)

    # Now import the module
    from enqueue_job import enqueue_evaluation
    enqueue_evaluation(trace_id,dataset)




