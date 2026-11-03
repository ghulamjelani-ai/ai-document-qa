# Single-container image for a free cloud host (Render / Railway / Fly).
# Runs the FastAPI backend (internal :8000) and the Streamlit UI (public $PORT)
# together. For local two-service development use docker-compose.yml instead.
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -r frontend/requirements.txt

COPY backend/app backend/app
COPY frontend/app.py frontend/app.py
COPY start.sh start.sh
RUN chmod +x start.sh

# The UI talks to the backend inside the same container.
ENV API_BASE_URL=http://127.0.0.1:8000
# Cloud uses API providers (Ollama is local-only).
ENV EMBEDDINGS_PROVIDER=openai
ENV LLM_PROVIDER=openai

EXPOSE 8501
CMD ["./start.sh"]
