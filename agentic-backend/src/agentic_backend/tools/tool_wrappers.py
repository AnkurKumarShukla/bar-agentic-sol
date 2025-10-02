import os
import json
from typing import List, Dict, Any
from ..config import settings

# If you use OpenAI, import here; otherwise the file uses mocks
try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
except Exception:
    openai = None

from langchain.schema import HumanMessage, AIMessage, ToolMessage


def get_llm_response(prompt: str) -> str:
    """
    Simple LLM wrapper: if OPENAI_API_KEY present we call OpenAI, otherwise return a safe mock.
    Returns raw text.
    """
    if openai:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            print("OpenAI call failed, falling back to mock:", e)

    # Mock deterministic policy for testing
    # For first step return websearch decision; next step FINISH
    # You can replace with smarter logic.
    mock = {
        "selected_agent": "websearch_agent",
        "task": prompt[:200] if len(prompt) > 0 else "do a websearch",
        "reasoning": "Mock supervisor picks websearch first to gather current info."
    }
    return json.dumps(mock)


def run_mock_agent_return_messages(task: str, agent_name: str):
    """
    Return a fake message stream similar to ReAct agent messages.
    """
    # Example: human -> ai (tool call) -> tool -> ai final
    human = HumanMessage(content=task)
    # AIMessage with a tool call
    ai_call = AIMessage(content="", tool_calls=[{"id": "tool_1", "name": "search_tool", "args": {"q": task}}])
    tool_response = ToolMessage(content=f"Found top news: TCS stock price is ₹3,650", tool_call_id="tool_1")
    ai_final = AIMessage(content=f"Top result: TCS current price ~ ₹3,650")
    return [human, ai_call, tool_response, ai_final]


def invoke_agent(agent_name: str, task: str) -> List[Any]:
    """
    If USE_REAL_AGENTS is True, call your real agent builder; otherwise return mock messages list.
    """
    if settings.USE_REAL_AGENTS:
        # Example placeholder - integrate your create_react_agent object here.
        raise NotImplementedError("Hook your real create_react_agent.invoke(...) here")
    else:
        return run_mock_agent_return_messages(task, agent_name)
