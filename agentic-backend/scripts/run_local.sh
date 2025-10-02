#!/usr/bin/env bash
export PYTHONPATH=./src
uvicorn agentic_backend.main:app --reload --host 0.0.0.0 --port 8000
