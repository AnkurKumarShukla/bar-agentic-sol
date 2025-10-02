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
    current_task: Optional[str] = None
    decisions: List[SupervisorDecision] = Field(default_factory=list)
    agent_states: Dict[str, List[AgentState]] = Field(default_factory=dict)
    final_output: Optional[str] = None

    def add_decision(self, step: int, agent: str, reasoning: str, task: str):
        d = SupervisorDecision(step=step, selected_agent=agent, reasoning=reasoning, task=task)
        self.decisions.append(d)
        self.current_task = task

    def add_agent_state(self, agent_name: str, agent_state: AgentState):
        self.agent_states.setdefault(agent_name, []).append(agent_state)
        # add agent output to context for downstream steps
        self.context[f"{agent_name}_step{len(self.agent_states[agent_name])}"] = agent_state.agent_output

    def set_final_output(self, out: str):
        self.final_output = out
        self.current_task = None

    def get_trace(self) -> str:
        parts = [f"User query: {self.user_query}\n"]
        for d in self.decisions:
            out = self.context.get(f"{d.selected_agent}_step{d.step}", "")
            parts.append(f"Step {d.step}: {d.selected_agent} -> {d.task}\n Reasoning: {d.reasoning}\n Output: {out}\n")
        parts.append(f"Final: {self.final_output}")
        return "\n".join(parts)
    def dump_json(self, **kwargs) -> str:
        """Dump the entire state as a JSON string."""
        return self.model_dump_json(**kwargs)