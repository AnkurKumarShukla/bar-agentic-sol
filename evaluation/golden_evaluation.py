# from deepeval.metrics import (
#     TaskCompletionMetric,
#     HallucinationMetric,
#     AnswerRelevancyMetric,
#     GEval,
#     PIILeakageMetric
# )
# import deepeval
# from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# from deepeval import evaluate
# from deepeval.test_case import ToolCall as DeepToolCall
# from langfuse_import_export import get_langfuse_response, set_score_langfuse
# import os
# from helper import infer_reference
# from dotenv import load_dotenv
# from langfuse.openai import openai
# # openai.default_timeout = 240

# load_dotenv()

# from deepeval.dataset import Golden
# from deepeval.test_case import LLMTestCase
# from deepeval.dataset import EvaluationDataset

# def evaluate_combined(trace_id,dataset):
#     goldens = [Golden(
#         input="Fetch the latest news sentiment for Lloyds Bank and summarise if the market outlook is positive or negative?",
#         context=["news sentiment analysis & market outlook"],
#         expected_tools=["fetch_news"],
#         expected_output="The latest news sentiment for Lloyds Bank indicates a predominantly positive outlook in the market. Here's a comprehensive summary of the current situation and market outlook:\n\n### Summary of Sentiment Impact\n- **Overall Sentiment**: Positive (Sentiment Score: 0.6)\n- **Key Highlights**:\n  - Lloyds' share price has surged approximately **50% year-to-date**, reaching levels not seen since 2015.\n  - This increase is driven by **high UK interest rates**, enhanced net interest income, a significant **share buyback program**, and an increased dividend.\n  - Analysts project further growth, with a target share price of **93p**, suggesting potential returns of around **15%**.\n  - However, there are concerns regarding the **fragility of the UK economy**, which could pose risks moving forward.\n\n### Emotional Indicators\n- **Emotions Present**: Optimism, Uncertainty\n\n### Event Sentiment Breakdown\n- **Macroeconomics**: Positive\n- **Market Sentiment**: Positive\n- **Regulatory, Adoption, Technology, Security Stance**: Neutral\n\n### Market Outlook\n- **Risk Level**: Medium\n- **Trading Signals**:\n  - **Momentum**: Bullish\n  - **Volatility Outlook**: Medium\n  - **Liquidity Outlook**: Positive\n  - **Whale Activity**: Supportive\n\n### Investment Recommendation\n- **Recommendation**: Hold, considering the medium risk and uncertainty in the broader economic context.\n- **Confidence Level**: Medium\n\nIn conclusion, while the sentiment surrounding Lloyds Bank shows strong growth potential, investors should exercise caution due to mixed signals from the broader economic landscape."
#     )]
#     metrics = [
#         TaskCompletionMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True, async_mode=False),
#         HallucinationMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True, async_mode=False),
#         AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True, async_mode=False),
#         PIILeakageMetric(threshold=0.5, model="gpt-4o-mini", include_reason=True, async_mode=False),
#         GEval(
#             model="gpt-4o-mini",
#             name="Correctness",
#             criteria="""
#                 1. Fidelity: Does the answer correctly address the user’s query without hallucination?
#                 2. Domain Soundness: Is the reasoning consistent with general investment research practices (e.g., portfolio analysis, market outlook, risk factors)?
#                 3. Evidence: Are statements supported by plausible financial knowledge, retrieved data, or standard industry sources (e.g., Bloomberg, Barclays research, Yahoo Finance)?
#                 4. Completeness: Does the answer fully address all parts of the query?""",
#             evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
#             threshold=0.6,
#         ),
#     ]
#     predictions = get_langfuse_response(trace_id)
#     test_cases = []
#     for golden in goldens:
#         reference = {}
#         agent_references = {}
#         all_scores = {}
#         score_reasons = {}
#         # ---------------- Per-agent evaluation ----------------
#         for agent in predictions[0]["agents"]:
#             agent_name = agent["agent_name"]
#             agent_input = agent["input"]
#             agent_output = agent["output"]
#             agent_tools = agent["tool_calls"]
#             print(f"⚡ Evaluating {agent_name}...")
#             test_case = LLMTestCase(
#                 input=agent_input,
#                 actual_output=agent_output,
#                 tools_called=[DeepToolCall(name=t, args={}) for t in agent_tools],
#                 expected_output=golden.expected_output,
#                 expected_tools=[DeepToolCall(name=t, args={}) for t in golden.expected_tools],
#                 context=golden.context
#             )

#             results = evaluate(test_cases=[test_case], metrics=metrics)
#             for test_result in results.test_results:
#                 for metric in test_result.metrics_data:
#                     metric_key = f"{agent_name}_{metric.name.lower().replace(' ', '_').replace('[geval]', 'geval')}"
#                     all_scores[metric_key] = float(metric.score)
#                     reason = getattr(metric, "reason", None) or getattr(metric, "explanation", None)
#                     if reason:
#                         score_reasons[metric_key] = reason

#             print(f"📊 {agent_name} Scores:", all_scores)
#             print(f"📊 {agent_name} Reasons:", score_reasons)

#         # Attach agent references to main reference dict
#         reference["agent_references"] = golden
#         print("all scores:", all_scores)
#         set_score_langfuse(all_scores, predictions[0]["trace_id"], reference, "deepeval", score_reasons=score_reasons)
#     print("test cases:",test_cases)
    

from deepeval.metrics import (
    TaskCompletionMetric,
    HallucinationMetric,
    AnswerRelevancyMetric,
    GEval,
    PIILeakageMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall as DeepToolCall
from deepeval import evaluate
from deepeval.dataset import Golden
from dotenv import load_dotenv
from langfuse_import_export import get_langfuse_response, set_score_langfuse

import os

load_dotenv()


def evaluate_combined(trace_id, dataset=None):
    # -------------------------------------------------
    # 🟩 Define multiple Goldens (one per agent)
    # -------------------------------------------------
    goldens = {
        "supervisor": Golden(
            input="Fetch the latest news sentiment for Lloyds Bank and summarise if the market outlook is positive or negative.",
            context=["news sentiment analysis & market outlook"],
            expected_tools=[DeepToolCall(name="news_sentiment_agent", args={})],
            expected_output=(
                "The latest news sentiment for Lloyds Bank indicates a predominantly positive outlook in the market. Here's a comprehensive summary of the current situation and market outlook:\n\n### Summary of Sentiment Impact\n- **Overall Sentiment**: Positive (Sentiment Score: 0.6)\n- **Key Highlights**:\n  - Lloyds' share price has surged approximately **50% year-to-date**, reaching levels not seen since 2015.\n  - This increase is driven by **high UK interest rates**, enhanced net interest income, a significant **share buyback program**, and an increased dividend.\n  - Analysts project further growth, with a target share price of **93p**, suggesting potential returns of around **15%**.\n  - However, there are concerns regarding the **fragility of the UK economy**, which could pose risks moving forward.\n\n### Emotional Indicators\n- **Emotions Present**: Optimism, Uncertainty\n\n### Event Sentiment Breakdown\n- **Macroeconomics**: Positive\n- **Market Sentiment**: Positive\n- **Regulatory, Adoption, Technology, Security Stance**: Neutral\n\n### Market Outlook\n- **Risk Level**: Medium\n- **Trading Signals**:\n  - **Momentum**: Bullish\n  - **Volatility Outlook**: Medium\n  - **Liquidity Outlook**: Positive\n  - **Whale Activity**: Supportive\n\n### Investment Recommendation\n- **Recommendation**: Hold, considering the medium risk and uncertainty in the broader economic context.\n- **Confidence Level**: Medium\n\nIn conclusion, while the sentiment surrounding Lloyds Bank shows strong growth potential, investors should exercise caution due to mixed signals from the broader economic landscape."
            )
        ),
        "news_sentiment_agent": Golden(
            input="Fetch the latest news sentiment for Lloyds Bank.",
            context=["news sentiment analysis"],
            expected_tools=[DeepToolCall(name="fetch_news", args={})],
            expected_output=(
               "The latest news sentiment regarding Lloyds Bank indicates a predominantly positive outlook in the market. Here’s a summary of the sentiment analysis:\n\n### Summary of Sentiment Impact\n- **Overall Sentiment**: Positive (Sentiment Score: 0.6)\n- **Key Highlights**:\n  - Lloyds’ share price has surged approximately 50% year-to-date, reaching levels not seen since 2015.\n  - The increase is attributed to high UK interest rates, greater net interest income, a substantial share buyback program, and an increased dividend.\n  - Analysts predict further growth, with a target share price of 93p, forecasting potential returns of around 15%.\n  - Despite these positive developments, there is an underlying concern regarding the fragility of the UK economy, which may present risks moving forward.\n\n### Emotional Indicators\n- **Emotions Present**: Optimism, Uncertainty\n\n### Event Sentiment Breakdown\n- **Macroeconomics**: Positive\n- **Market Sentiment**: Positive\n- **Regulatory, Adoption, Technology, Security Stance**: Neutral\n\n### Market Outlook\n- **Risk Level**: Medium\n- **Trading Signals**:\n  - **Momentum**: Bullish\n  - **Volatility Outlook**: Medium\n  - **Liquidity Outlook**: Positive\n  - **Whale Activity**: Supportive\n\n### Investment Recommendation\n- **Recommendation**: Hold (given the medium risk and uncertainty in the broader economic context)\n- **Confidence Level**: Medium\n\nIn conclusion, while the sentiment around Lloyds Bank shows strong potential for growth, investors should remain cautious due to the mixed signals from the broader economic landscape."
            )
        )
    }

    # -------------------------------------------------
    # 🟨 Define metrics
    # -------------------------------------------------
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
                2. Domain Soundness: Is the reasoning consistent with investment research practices?
                3. Evidence: Are statements supported by plausible financial knowledge or data sources?
                4. Completeness: Does the answer fully address all parts of the query?
            """,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.6,
        ),
    ]

    # -------------------------------------------------
    # 🟦 Load Langfuse trace
    # -------------------------------------------------
    predictions = get_langfuse_response(trace_id)
    test_cases = []

    # -------------------------------------------------
    # 🧠 Helper to select golden dynamically
    # -------------------------------------------------
    def select_golden(agent_name):
        name = agent_name.lower()
        if "news" in name:
            return goldens.get("news_sentiment_agent")
        elif "supervisor" in name:
            return goldens.get("supervisor")
        else:
            return None

    # -------------------------------------------------
    # 🧩 Evaluate each agent separately
    # -------------------------------------------------
    for agent in predictions[0]["agents"]:
        agent_name = agent["agent_name"]
        golden = select_golden(agent_name)

        if not golden:
            print(f"⚠️ No golden reference found for agent: {agent_name}")
            continue

        print(f"\n⚡ Evaluating {agent_name} ...")

        test_case = LLMTestCase(
            input=agent["input"],
            actual_output=agent["output"],
            tools_called=[DeepToolCall(name=t, args={}) for t in agent["tool_calls"]],
            expected_output=golden.expected_output,
            expected_tools=golden.expected_tools,
            context=golden.context
        )
        test_cases.append(test_case)

        # Run evaluation
        results = evaluate(test_cases=[test_case], metrics=metrics)

        # Collect results
        all_scores = {}
        score_reasons = {}
        for test_result in results.test_results:
            for metric in test_result.metrics_data:
                metric_key = f"{agent_name}_{metric.name.lower().replace(' ', '_').replace('[geval]', 'geval')}"
                all_scores[metric_key] = float(metric.score)
                reason = getattr(metric, "reason", None) or getattr(metric, "explanation", None)
                if reason:
                    score_reasons[metric_key] = reason

        print(f"📊 {agent_name} Scores: {all_scores}")
        print(f"📘 {agent_name} Reasons: {score_reasons}")

        # Push to Langfuse
        reference = {"agent_references": golden.model_dump(mode="json")}
        set_score_langfuse(all_scores, predictions[0]["trace_id"], reference, "deepeval_golden", score_reasons=score_reasons)

    print(f"\n✅ Evaluation complete for {len(test_cases)} test cases.")


