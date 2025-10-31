import json

def collect_citations(agentic_state_json: dict) -> list:
    """
    Extract all citation URLs from web_search and fetch_news tool responses
    in the agent_states section of the given agentic system state.
    """
    citations = []

    # Iterate through all agent states
    for agent_state in agentic_state_json.get("supervisor", {}).get("agent_states", []):
        for pair in agent_state.get("tool_call_response_pair", []):
            tool_name = pair.get("tool_name", "")
            # Only look at relevant tools
            if tool_name in ["web_search", "fetch_news"]:
                try:
                    response_data = json.loads(pair.get("response", "{}"))
                    # News or web search results may be under 'results'
                    results = response_data.get("results", [])
                    for item in results:
                        link = item.get("link")
                        if link:
                            citations.append(link)
                except json.JSONDecodeError:
                    continue

    return citations
