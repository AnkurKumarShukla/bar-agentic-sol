import redis
from rq import Queue
from ragas_test_setup import evaluate_trace_ragas  # your evaluation function
from test_core_deepeval import evaluate_trace_deepeval
from golden_evaluation import evaluate_combined
import os
import time

# Make sure host, port, and password match your Redis container
redis_password = os.getenv("REDIS_PASSWORD")
redis_host = os.getenv("REDIS_HOST")
redis_port = os.getenv("REDIS_PORT")
redis_conn = redis.Redis(
    host=redis_host,  # or container name if using docker-compose network
    port=redis_port,
    password=redis_password,  # this must match REDIS_AUTH
    decode_responses=False
)

q = Queue("evaluation", connection=redis_conn, default_timeout=600) #evaluation - queue name

failed_jobs = []

# For pushing jobs to evaluation queue
def enqueue_evaluation(trace_id: str,dataset: str, opik_trace_id: str = None):
    jobs = []
    print(f"Enqueuing evaluation for trace_id: {trace_id} with dataset: {dataset}")
    if dataset=="golden_data":
        jobs.append(q.enqueue(evaluate_combined, trace_id,dataset))
    else:
        try:
            time.sleep(15)
            # First evaluation function ragas
            # jobs.append(q.enqueue(evaluate_trace_ragas, trace_id))
            # Second evaluation function deepeval
            jobs.append(q.enqueue(evaluate_trace_deepeval, trace_id, opik_trace_id))
        except Exception as e:
            print(f"Skipping trace_id {trace_id} due to error: {e}")
            failed_jobs.append(trace_id)
        print("queue count :", q.count)
        return jobs

# Retry failed jobs every X seconds
def retry_failed_jobs(interval=60):
    while True:
        for trace_id in failed_jobs[:]:
            print(f"Retrying {trace_id}")
            result = enqueue_evaluation(trace_id)
            if result:
                failed_jobs.remove(trace_id)
        time.sleep(interval)

# To start retrying in the background:
import threading
threading.Thread(target=retry_failed_jobs, daemon=True).start()
