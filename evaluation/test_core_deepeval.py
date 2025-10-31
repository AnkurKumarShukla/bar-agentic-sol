from deepeval.metrics import (
    ToolCorrectnessMetric,
    TaskCompletionMetric,
    HallucinationMetric,
    AnswerRelevancyMetric,
    GEval,
    PIILeakageMetric
)
import deepeval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate
from deepeval.test_case import ToolCall as DeepToolCall
from langfuse_import_export import get_langfuse_response, set_score_langfuse
import os
from helper import generate_reference
from dotenv import load_dotenv
from langfuse.openai import openai
# openai.default_timeout = 240

load_dotenv()

# Evaluation function for DeepEval
def evaluate_trace_deepeval(trace_id, opik_trace_id=None):

    predictions = get_langfuse_response(trace_id)

    # Shared metrics
    metrics = [
        ToolCorrectnessMetric(threshold=0.7,  include_reason=True),
        TaskCompletionMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True, async_mode=False),
        HallucinationMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True, async_mode=False),
        PIILeakageMetric(threshold=0.5, model="gpt-4o-mini", include_reason=True, async_mode=False),
        GEval(
            model="gpt-4o-mini",
            name="Correctness",
            criteria="""
                1. Fidelity: Does the answer correctly address the user’s query without hallucination?
                2. Domain Soundness: Is the reasoning consistent with general investment research practices ?
                3. Evidence: Are statements supported by plausible financial knowledge, retrieved data, or standard industry sources (e.g., Bloomberg, Barclays research, Yahoo Finance)?
                4. Completeness: Does the answer fully address all parts of the query?""",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.6,
        ),
    ]

    for prediction in predictions:
        reference = {}
        agent_references = {}
        all_scores = {}
        score_reasons = {}

        # ---------------- Per-agent evaluation ----------------
        for agent in prediction["agents"]:
            agent_name = agent["agent_name"]
            agent_input = agent["input"]
            agent_output = agent["output"]
            agent_tools = agent["tool_calls"]

            print(f"⚡ Evaluating {agent_name}...")

            # Infer reference for this agent
            agent_reference = generate_reference(agent_name, agent_input)
            agent_references[agent_name] = agent_reference
            test_case = LLMTestCase(
                input=agent_input,
                actual_output=agent_output,
                tools_called=[DeepToolCall(name=t, args={}) for t in agent_tools],
                expected_output=agent_reference["reference"],
                expected_tools=[DeepToolCall(name=t, args={}) for t in agent_reference.get("expected_tools", [])],
                context=[agent_reference["reference_topic"]]
            )

            results = evaluate(test_cases=[test_case], metrics=metrics)
            for test_result in results.test_results:
                for metric in test_result.metrics_data:
                    metric_key = f"{agent_name}_{metric.name.lower().replace(' ', '_').replace('[geval]', 'geval')}"
                    all_scores[metric_key] = float(metric.score)
                    reason = getattr(metric, "reason", None) or getattr(metric, "explanation", None)
                    if reason:
                        score_reasons[metric_key] = reason

            print(f"📊 {agent_name} Scores:", all_scores)
            print(f"📊 {agent_name} Reasons:", score_reasons)

        # Attach agent references to main reference dict
        reference["agent_references"] = agent_references
        print("all scores:", all_scores)
        add_evaluation_scores_to_opik(opik_trace_id, all_scores, score_reasons, reference)
        set_score_langfuse(all_scores, prediction["trace_id"], reference, "deepeval", score_reasons=score_reasons)



import opik
from opik import opik_context
import json

def add_evaluation_scores_to_opik(opik_trace_id: str, evaluation_results: dict, score_reasons: dict = None, reference: dict = None):
    """
    Add evaluation scores to an existing Opik trace
    
    Args:
        opik_trace_id: The trace ID from Opik
        evaluation_results: Dict with score names and values
    """
    try:
        opik_client = opik.Opik()
        # Method 1: Using log_traces_feedback_scores (batch)
        feedback_scores = [
            {
                "id": opik_trace_id,
                "name": name,
                "value": value,
                "reason": score_reasons.get(name) if score_reasons and name in score_reasons else f"{name} evaluation metric score"
            }
            for name, value in evaluation_results.items()
        ]
        opik_client.log_traces_feedback_scores(scores=feedback_scores)
        evaluation_results = {
            k: float(v[0]) if isinstance(v, list) and len(v) == 1 else float(v)
            for k, v in evaluation_results.items()
        }
        
           
        status = 1 if all(
            v >= 0.7 for k, v in evaluation_results.items() if "hallucination" not in k.lower()
        ) else 0

        scores = [{
            "id": opik_trace_id,
            "name": "test_status",
            "value": status
        }]

        opik_client.log_traces_feedback_scores(
            scores=scores
            
        )
        trace = opik_client.trace(id=opik_trace_id)
        trace_content = opik_client.get_trace_content(id=opik_trace_id)
        print("trace_content type ========================",type(trace_content))
        print("trace content ============",trace_content)
        # existing_metadata = trace_content.metadata or {}
        graph_metadata = build_mermaid_graph_from_trace(trace_content)
        graph_metadata["reference"] = json.dumps(reference)
        trace.update(metadata=graph_metadata)
        print("existing metadata",graph_metadata)
        # trace.update(metadata={"evaluation_reference": json.dumps(reference)})
        print(f"✓ Added {len(feedback_scores)} evaluation scores to Opik trace")
    except Exception as e:
        print(f"✗ Failed to add scores to Opik: {e}")
        import traceback
        traceback.print_exc()



def build_mermaid_graph_from_trace(trace_content):
    """
    Dynamically builds an Opik-compatible Mermaid graph definition
    from agent + tool execution info in a trace.
    """
    agents = []
    connections = []

    # Always include start and end
    agents.append("__start__")
    agents.append("__end__")

    # Add supervisor if exists
    agents.append("supervisor")

    # Extract agents actually used
    for agent_state in trace_content.output.get("agent_states", []):
        agent_name = agent_state.get("agent_name")
        if agent_name and agent_name not in agents:
            agents.append(agent_name)
            connections.append(f"supervisor --> {agent_name}")
            connections.append(f"{agent_name} --> supervisor")

        # Add tools used
        for tool_pair in agent_state.get("tool_call_response_pair", []):
            tool_name = tool_pair.get("tool_name")
            if tool_name:
                agents.append(tool_name)
                connections.append(f"{agent_name} --> {tool_name}")
                connections.append(f"{tool_name} --> {agent_name}")

      # 👇 Ensure start and end are connected
    connections.append("__start__ --> supervisor")
    connections.append("supervisor --> __end__")
    # Build mermaid definition
    mermaid = [
        "---",
        "config:",
        "  flowchart:",
        "    curve: linear",
        "---",
        "graph TD;",
        "\t__start__([<p>__start__</p>]):::first",
    ]

    for agent in agents:
        if agent not in ["__start__", "__end__"]:
            mermaid.append(f"\t{agent}({agent})")

    mermaid.append("\t__end__([<p>__end__</p>]):::last")

    # Add connections
    for conn in connections:
        mermaid.append(f"\t{conn};")

    # Style definitions
    mermaid += [
        "\tclassDef default fill:#f2f0ff,line-height:1.2",
        "\tclassDef first fill-opacity:0",
        "\tclassDef last fill:#bfb6fc"
    ]

    return {
        "_opik_graph_definition": {
            "format": "mermaid",
            "data": "\n".join(mermaid)
        }
    }

# # Example usage:
# opik_client = OpikClient()
# trace_content = opik_client.get_trace_content(id="019a30b1-7f6d-7378-9cb3-586dd1272a84")

# graph_metadata = build_mermaid_graph_from_trace(trace_content)
# print(graph_metadata["_opik_graph_definition"]["data"])

