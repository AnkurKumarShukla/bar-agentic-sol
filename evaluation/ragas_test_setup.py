from ragas.metrics import ToolCallAccuracy,TopicAdherenceScore, AgentGoalAccuracyWithReference, AgentGoalAccuracyWithoutReference
from ragas import evaluate
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from langfuse_import_export import get_langfuse_response, set_score_langfuse
from openai import OpenAI
import os, json
from ragas.messages import HumanMessage as RHMessage, AIMessage as RAMessage
from ragas.messages import ToolCall
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic
from helper import infer_reference

# Evaluation function for RAGAS
def evaluate_trace_ragas(trace_id):
# 🔹 Step 2: Fetch multiple traces from Langfuse
    predictions = get_langfuse_response(trace_id)  # <-- this returns a LIST of trace dicts
    samples = []
    topic_adherence_metric = TopicAdherenceScore(mode="recall")
    multi_turn_metrics = [ToolCallAccuracy(), AgentGoalAccuracyWithoutReference(), topic_adherence_metric]
   
    for trace in predictions:
        agent_references = {}
        reference = {}
        all_scores = {}
        # ---------------- Per-agent evaluation ----------------
        for agent in trace.get("agents", []):
            agent_name = agent["agent_name"]
            agent_input = agent["input"]
            agent_output = agent["output"]
            agent_tools = agent["tool_calls"]

            print(f"⚡ Evaluating {agent_name}...")

            # Infer reference for this agent using its input and output
            agent_reference = infer_reference(agent_name, agent_input)
            print(f"Agent {agent_name} reference:", agent_reference)

            # Store agent reference for later use
            agent_references[agent_name] = agent_reference

            agent_tool_calls = sorted(
                [ToolCall(name=tool, args={}) for tool in set(agent_tools)],
                key=lambda x: x.name
            )

            agent_sample = MultiTurnSample(
                user_input=[
                    RHMessage(content=agent_input),
                    RAMessage(content=agent_output, tool_calls=agent_tool_calls),
                ],
                reference_tool_calls=[
                    ToolCall(name=tool, args={}) for tool in agent_reference.get("expected_tools", [])
                ],
                reference=agent_reference["reference"],
                reference_topics=[agent_reference["reference_topic"]]
            )

            dataset_agent = EvaluationDataset(samples=[agent_sample])
            agent_results = evaluate(dataset=dataset_agent, metrics=multi_turn_metrics)
           

            all_scores.update({
                f"{agent_name}_tool_call_accuracy": float(agent_results["tool_call_accuracy"][0]),
                f"{agent_name}_agent_goal_accuracy": float(agent_results["agent_goal_accuracy"][0]),
                f"{agent_name}_topic_adherence": float(agent_results["topic_adherence(mode=recall)"][0])
            })

            print(f"📊 {agent_name} Scores:", all_scores)
        # Add agent references to main reference dict
        reference["agent_references"] = agent_references
        print("all scores:", all_scores)
        set_score_langfuse(all_scores, trace["trace_id"], reference, "ragas")






