from typing import List, Union
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from ..models.state_models import AgentState


def build_agent_state(messages: List[Union[HumanMessage, AIMessage, ToolMessage]], agent_name: str) -> AgentState:
    agent_input = ""
    tool_call_response_pair = []
    agent_output = ""

    pending_tool_calls = {}
    for msg in messages:
        # HumanMessage
        if hasattr(msg, "content") and msg.__class__.__name__ == "HumanMessage":
            agent_input = msg.content
        elif msg.__class__.__name__ == "AIMessage":
            # final text output
            if getattr(msg, "content", "").strip():
                agent_output = msg.content
            # tool_calls might be present
            for call in getattr(msg, "tool_calls", []) or []:
                pending_tool_calls[call.get("id")] = {
                    "tool_name": call.get("name"),
                    "arguments": call.get("args", {}),
                }
        elif msg.__class__.__name__ == "ToolMessage":
            # attach tool response to pending call
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id and tool_call_id in pending_tool_calls:
                entry = pending_tool_calls[tool_call_id]
                entry["response"] = getattr(msg, "content", "")
                tool_call_response_pair.append(entry)

    return AgentState(
        agent_name=agent_name,
        agent_input=agent_input,
        tool_call_response_pair=tool_call_response_pair,
        agent_output=agent_output,
    )
