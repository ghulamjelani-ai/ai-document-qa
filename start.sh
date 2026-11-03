#!/usr/bin/env bash
# Start the backend (internal) and the Streamlit UI (public) in one container.
set -e

# Backend on 127.0.0.1:8000 (not exposed; only the UI talks to it).
( cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 ) &

# Streamlit UI on the port the host gives us (Render/Railway/Fly set $PORT).
cd frontend
exec streamlit run app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
