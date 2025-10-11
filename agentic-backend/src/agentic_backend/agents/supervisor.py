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
Previous conversation context user request- {state.request_summary}
Previous conversation context user request- {state.response_summary}


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
context so far: {state.context}

Decision Task:
0. If previous conversation have sufficient context then no need to call agent . Answer directly from there . 
1. Decide is agent is required to answer the querry . If yes then Select the next agent (choose from: 'finance_agent', 'websearch_agent', 'news_sentiment_agent', or FINISH). Else answer form previous context if its sufficient.
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
        # Generate comprehensive final response based on all collected context
        final_response_prompt = f"""
Act like experience financial advisor, response to the user.You have agents response who worked as per your guidence. 

User's original query: {state.user_query}

All collected information from agents:
{json.dumps(state.context, indent=2)}

Agent execution history:
{chr(10).join([f"Step {d.step}: {d.selected_agent} - {d.task}" for d in state.decisions])}

Agent outputs:
{chr(10).join([f"{s.agent_name}: {s.agent_output}" for s in state.agent_states])}

Previous conversation context:
Request summary: {state.request_summary}
Response summary: {state.response_summary}

Task: Synthesize all the above information into a clear, comprehensive, and well-structured response that directly answers the user's query.

Guidelines:
- Provide specific data points (stock prices, percentages, sentiment scores, etc.)
- Structure the response logically with proper formatting
- Be concise but complete
- If multiple items were requested, address each one
- Include relevant context and insights
- Do not mention agent names or internal processes
- Present the information as if you gathered it yourself

Provide your final response as plain text (not JSON).
"""

        final_response = llm.invoke(final_response_prompt)

        # Extract content from response
        if hasattr(final_response, "content"):
            state.final_output = final_response.content
        elif isinstance(final_response, str):
            state.final_output = final_response
        else:
            state.final_output = str(final_response)

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

