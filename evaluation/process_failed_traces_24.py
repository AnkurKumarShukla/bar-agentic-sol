from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from langfuse import get_client
from enqueue_job import enqueue_evaluation

load_dotenv()

langfuse = get_client()

def fetch_trace_ids_last_24h(limit: int = 100, max_pages: int = 1000, name: str = None):
    all_ids = []
    page = 1

    # Compute time window
    now = datetime.utcnow()
    yesterday = now - timedelta(hours=1)

    while True:
        traces_resp = langfuse.api.trace.list(
            limit=limit,
            page=page,
            name=name,
            from_timestamp=yesterday,
            to_timestamp=now
        )
        print(traces_resp)
        traces = traces_resp.data or []
        if not traces:
            print("No more traces found.")
            break

        # Add trace IDs
        for t in traces:
            all_ids.append(t.id)

        if len(traces) < limit:
            break
        page += 1
        if page > max_pages:
            break

    return all_ids

if __name__ == "__main__":
    trace_ids = fetch_trace_ids_last_24h(name="LangGraph", limit=100)
    print("Trace IDs in last 24h:", trace_ids)
    for tid in trace_ids:
        enqueue_evaluation(tid)
