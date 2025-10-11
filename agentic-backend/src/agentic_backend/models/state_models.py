from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class AgentState(BaseModel):
    agent_name: str
    agent_input: str = ""
    tool_call_response_pair: List[Dict[str, Any]] = Field(default_factory=list)
    agent_output: str = ""


class SupervisorDecision(BaseModel):
    step: int
    selected_agent: str
    reasoning: str
    task: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SupervisorState(BaseModel):
    run_id: Optional[str] = None
    user_query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    request_summary: Optional[str] = None
    response_summary: Optional[str] = None
    current_task: Optional[str] = None
    decisions: List[SupervisorDecision] = Field(default_factory=list)
    agent_states: List[AgentState] = Field(default_factory=list)  # Changed to flat array
    final_output: Optional[str] = None

    def add_decision(self, step: int, agent: str, reasoning: str, task: str):
        d = SupervisorDecision(step=step, selected_agent=agent, reasoning=reasoning, task=task)
        self.decisions.append(d)
        self.current_task = task

    def add_agent_state(self, agent_name: str, agent_state: AgentState):
        # Append to flat array in sequential order
        self.agent_states.append(agent_state)
        # add agent output to context for downstream steps
        agent_count = sum(1 for s in self.agent_states if s.agent_name == agent_name)
        self.context[f"{agent_name}_step{agent_count}"] = agent_state.agent_output

    def set_final_output(self, out: str):
        self.final_output = out
        self.current_task = None

    def get_trace(self) -> str:
        parts = [f"User query: {self.user_query}\n"]
        for i, d in enumerate(self.decisions):
            # Find corresponding agent state
            agent_output = ""
            if i < len(self.agent_states):
                agent_output = self.agent_states[i].agent_output
            parts.append(f"Step {d.step}: {d.selected_agent} -> {d.task}\n Reasoning: {d.reasoning}\n Output: {agent_output}\n")
        parts.append(f"Final: {self.final_output}")
        return "\n".join(parts)
    def dump_json(self, **kwargs) -> str:
        """Dump the entire state as a JSON string."""
        return self.model_dump_json(**kwargs)
    
    # def merge(self, other: "SupervisorState") -> "SupervisorState":
    #     """Merge previous + new states, maintaining rolling summaries."""
    
    # # Combine request summaries
    #     new_request_summary = (
    #         f"{self.request_summary or ''}\n"
    #         f"Previous request: {self.user_query}\n"
    #         f"New request: {other.user_query}"
    #     ).strip()

    # # Combine response summaries
    #     new_response_summary = (
    #         f"{self.response_summary or ''}\n"
    #         f"Previous response summary: {self.final_output or ''}"
    #     ).strip()

    #     return SupervisorState(
    #         user_query=other.user_query,
    #         request_summary=new_request_summary,
    #         response_summary=new_response_summary,
    #         context=self.context,  # optionally keep as-is
    #         current_task=other.current_task,
    #         decisions=self.decisions + other.decisions,
    #         agent_states=self.agent_states,
    #         final_output=other.final_output or self.final_output,
    #     )
