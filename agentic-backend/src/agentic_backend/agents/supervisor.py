import json
from ..models.state_models import SupervisorState
from ..models.state_models import *
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain.chat_models import init_chat_model
llm=init_chat_model("openai:gpt-4o-mini")

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor decides the next agent + task based on state so far."""

    step = len(state.decisions) + 1

    system_prompt = f"""
You are the Supervisor AI, orchestrating agents for investment research.

Rules:
- Never call multiple agents in parallel.
- always analyse context first thoroughly
- Do not answer queries yourself; your sole role is routing and summarizing.
- Break down the user query into minimal, clear sub-queries for other agents.
- Forward only the relevant rewritten sub-query, never the full original query.

Agent Selection:
1. **finance_agent**: Handles financial data requests (stock prices, earnings, analyst ratings, financial statements, institutional holders, options, etc.).
2. **websearch_agent**: Handles general internet knowledge or real-time (apart from news and  events not covered by financial tools.)
3. **news_sentiment_agent**: handles news to fetch news and return sentiment analysis of news

Guidelines:
- Be decisive: choose only one agent per step.
- Ensure the agent receives the exact, minimal input needed.
- After all agents respond, integrate the results into a clear, final user-facing message.
- Avoid merely relaying an agent's output; synthesize and summarize.
- before making decision go through context thoroughly
- analyse context and past decisions to avoid redundant calls.

User query: {state.user_query}
Context so far: {state.context}

Decision Task:
1. Select the next agent (choose from: 'finance_agent', 'websearch_agent', 'news_sentiment_agent', or FINISH).
2. Specify the exact task they should perform.
3. Provide reasoning for your choice.

Respond strictly in JSON with keys: selected_agent, task, reasoning.
"""


    response = llm.invoke(system_prompt)

    # --- normalize response to string ---
    if hasattr(response, "content"):
        raw_text = response.content
    elif isinstance(response, str):
        raw_text = response
    else:
        raw_text = str(response)

    # --- try parse as JSON ---
    import json
    try:
        parsed = json.loads(raw_text)
    except Exception:
        # fallback: strip to nearest JSON object
        json_str = raw_text[raw_text.find("{"): raw_text.rfind("}") + 1]
        parsed = json.loads(json_str)

    selected_agent = parsed.get("selected_agent")
    task = parsed.get("task")
    reasoning = parsed.get("reasoning")
    print(system_prompt)
    # handle FINISH
    if selected_agent == "FINISH":
        state.final_output = f"Final decision: {task}"
        state.current_task = None
        return state

    # record decision
    decision = SupervisorDecision(
        step=step,
        selected_agent=selected_agent,
        reasoning=reasoning,
        task=task
    )
    state.decisions.append(decision)
    state.current_task = task

    return state

