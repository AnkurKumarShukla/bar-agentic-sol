# worker.py - file for listening to queue
import redis
import os
from rq import Queue
from rq.worker import SimpleWorker   # ✅ use SimpleWorker, not Worker
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    redis_password = os.getenv("REDIS_PASSWORD")
    redis_host = os.getenv("REDIS_HOST")
    
    
    redis_port = os.getenv("REDIS_PORT")
    print(redis_host,redis_password,redis_port)
    redis_conn = redis.Redis(
        host=redis_host,  # or container name if using docker-compose network
        port=redis_port,
        password=redis_password,  # this must match REDIS_AUTH
        decode_responses=False
    )

    q = Queue("evaluation", connection=redis_conn)
    worker = SimpleWorker([q], connection=redis_conn)   # ✅ SimpleWorker instead of Worker

    print(">>> Worker started in SimpleWorker mode (Windows compatible)")
    worker.work(burst=False)   # works fine now
