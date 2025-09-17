FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# --- Install Ollama inside the container ---
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Pull the Ollama model at build time
RUN ollama pull gemma3:270m

EXPOSE 9000 11434

# Start Ollama server + your FastAPI app
CMD ["/bin/bash", "-c", "ollama serve & sleep 3 && uvicorn api_rag:app --host 0.0.0.0 --port 9000"]
