from deepeval.metrics import (
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
from helper import infer_reference
from dotenv import load_dotenv
from langfuse.openai import openai
# openai.default_timeout = 240

load_dotenv()

# Evaluation function for DeepEval
def evaluate_trace_deepeval(trace_id):

    predictions = get_langfuse_response(trace_id)

    # Shared metrics
    metrics = [
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
            agent_reference = infer_reference(agent_name, agent_input)
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
        set_score_langfuse(all_scores, prediction["trace_id"], reference, "deepeval", score_reasons=score_reasons)
