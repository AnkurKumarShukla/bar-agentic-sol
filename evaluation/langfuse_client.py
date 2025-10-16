# evaluation/langfuse_client.py
from langfuse import Langfuse
from dotenv import load_dotenv
load_dotenv()
import os

# Create Langfuse client ONCE per process
client = Langfuse(
    timeout=60.0,
    tracing_enabled=True  # or False if you don’t need OTEL spans
)
