# Lightweight Python base
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bash build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Setup Python environment - use cache for faster builds!
ENV PIP_CACHE_DIR=/app/.pip-cache
ENV TMPDIR=/app/tmp
RUN mkdir -p /app/tmp /app/.pip-cache

# Copy requirements first for better caching
COPY requirements.txt .
# REMOVED --no-cache-dir and added retry logic
RUN pip install --default-timeout=1000 --retries=3 --no-deps -r requirements.txt

# Extra deps - also removed --no-cache-dir
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastapi uvicorn[standard] requests chromadb sentence-transformers

# Clean up pip cache (optional, for smaller final image)
RUN rm -rf /app/.pip-cache

# Copy application code
COPY . .

# Copy start.sh explicitly and ensure permissions
COPY start.sh /app/start.sh
# Normalize line endings (CRLF -> LF) to avoid "no such file or directory" from shebangs
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 9000 11434

# Entrypoint
CMD ["/app/start.sh"]