from dotenv import load_dotenv
import json
import os
from datetime import datetime, timedelta, timezone
from langfuse_client import client
from langfuse.openai import openai
import time
import random
load_dotenv()

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj
def get_trace_with_retry(trace_id, retries=5, base_delay=1):
    for attempt in range(retries):
        try:
            trace = client.api.trace.get(trace_id)
            if trace:  # if data exists
                return trace
        except Exception:
            pass
        # wait before retry
        sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(sleep_time)
    raise RuntimeError(f"Trace {trace_id} not found after {retries} retries")

# To get data from langfuse dashboard
def get_langfuse_response(trace_id):
    output_file = "traces.json"
    all_traces = []

    trace = get_trace_with_retry(trace_id)
    if trace is None:
        raise ValueError(f"❌ No trace found in Langfuse for trace_id={trace_id}")

    # Convert to dict
    if hasattr(trace, "dict"):
        trace_data = trace.dict()
    elif hasattr(trace, "model_dump"):
        trace_data = trace.model_dump()
    else:
        raise TypeError("Unsupported trace object type from Langfuse API")

    trace_data_clean = clean_for_json(trace_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_log_file = f"langfuse_raw_trace_{trace_id}.json"
    with open(raw_log_file, "w", encoding="utf-8") as f:
        json.dump(trace_data_clean, f, indent=4, ensure_ascii=False)

    print(f"🟢 Raw Langfuse trace saved to: {os.path.abspath(raw_log_file)}")

    transformed = {
        "trace_id": trace.id,
        "trace_name": trace.name,
        "overall_input": trace.output.get("user_query"),
        "overall_output": trace.output.get("final_output"),
        "agents": []
    }

    agent_states = trace.output.get("agent_states", {})
    # print("agent_states =", type(agent_states), agent_states)

    # ✅ Normalize agent_states (dict or list)
    if isinstance(agent_states, dict):
        all_agent_names = list(agent_states.keys())
    elif isinstance(agent_states, list):
        all_agent_names = [
            state.get("agent_name", f"agent_{i}")
            for i, state in enumerate(agent_states)
            if isinstance(state, dict)
        ]
        # Convert list to dict for consistency
        agent_states = {
            state.get("agent_name", f"agent_{i}"): [state]
            for i, state in enumerate(agent_states)
            if isinstance(state, dict)
        }
    else:
        all_agent_names = []

    # 1️⃣ Supervisor agent
    transformed["agents"].append({
        "agent_name": "supervisor",
        "tool_calls": all_agent_names,
        "tool_call_response_pair": [],
        "input": trace.output.get("user_query"),
        "output": trace.output.get("final_output", ""),
        "decisions": trace.output.get("decisions", [])
    })

    # 2️⃣ Other agents
    if isinstance(agent_states, dict) and agent_states:
        for agent_name, states in agent_states.items():
            if agent_name == "supervisor":
                continue
            for state in states:
                tool_calls = [tool["tool_name"] for tool in state.get("tool_call_response_pair", [])]
                transformed["agents"].append({
                    "agent_name": state.get("agent_name", agent_name),
                    "tool_calls": tool_calls,
                    "tool_call_response_pair": state.get("tool_call_response_pair", []),
                    "input": state.get("agent_input", ""),
                    "output": state.get("agent_output", ""),
                })

    all_traces.append(transformed)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_traces, f, indent=4, ensure_ascii=False)

    print(f"✅ Clean trace data written to {os.path.abspath(output_file)}")
    return all_traces




# Set evaluation score to Langfuse Dashboard
def set_score_langfuse(ragas_scores, trace_id, meta_reference, testcase, score_reasons=None):
    ragas_scores = {
        k: float(v[0]) if isinstance(v, list) and len(v) == 1 else float(v)
        for k, v in ragas_scores.items()
    }
    for name, value in ragas_scores.items():
        # Use reason if available, else default comment
        comment = score_reasons.get(name) if score_reasons and name in score_reasons else f"{name} evaluation metric score"
        client.create_score(
            name=name,
            value=float(value),
            trace_id=trace_id,
            data_type="NUMERIC",
            comment=comment
        )
    status = "pass" if all(
        v >= 0.7 for k, v in ragas_scores.items() if "hallucination" not in k.lower()
    ) else "fail"

    client.create_score(
        name="test_status",
        value=status,
        trace_id=trace_id,
        data_type="CATEGORICAL",
        comment=testcase
    )

    with client.start_as_current_span(
        name="evaluation-metadata",
        trace_context={"trace_id": trace_id}
    ) as span:
        span.update_trace(metadata={"evaluation_reference": json.dumps(meta_reference),"langfuse_tags": ["golden_data"]})
        
    client.flush()
    with open("reference.json", "w", encoding="utf-8") as f:
        json.dump(meta_reference, f, indent=4, ensure_ascii=False)
