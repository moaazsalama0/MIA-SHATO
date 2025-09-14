FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for soundfile + audio ops)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8004

CMD ["python", "main.py"]

