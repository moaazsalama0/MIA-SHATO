![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Microservices](https://img.shields.io/badge/Architecture-Microservices-success)

# SHATO — Speech-to-Action Orchestrator

Compact summary
- SHATO is a microservice stack that converts user audio into actions and spoken responses: STT -> LLM (RAG) -> Validator -> TTS. It includes a Gradio UI for quick demos and Docker Compose for local runs.

Repository layout (key files)
- docker-compose.yml — Compose orchestration for all services and network.
- orchestrator_recons/
  - orchestrator/orchestrator.py — central FastAPI orchestrator (POST /process_audio, GET /tts/{text}, health).
- stt-service/
  - main.py — STT FastAPI service (POST /transcribe).
  - stt_processor.py — audio transcription logic.
- llm-service/
  - app/api_rag.py — LLM + retrieval endpoints and parsing utilities.
  - data_expansion/ — data converters & splitters for training examples.
  - Dockerfile, start.sh — image for LLM service (installs Ollama).
- robot-validator-api/
  - main.py — validation API (POST /execute_command).
  - validator.py — command schema & validation logic.
- tts-service/
  - main.py — TTS FastAPI service (POST /speak).
  - tts_processor.py — TTS backend implementation.
- ui-service/
  - ui.py — Gradio-based UI that calls orchestrator and plays TTS output.

Quick start (Docker Compose)
1. Build and run:
   ```bash
   docker-compose up --build
   ```
2. Services will expose ports defined in docker-compose. Health endpoints:
   - Orchestrator: GET /health
   - STT: GET /health
   - LLM: GET /health
   - Validator: GET /health
   - TTS: GET /health

Primary endpoints (examples)
- Orchestrator pipeline (audio file):
  ```bash
  curl -X POST "http://localhost:8000/process_audio/" -F "file=@/path/to/utterance.wav"
  ```
  Response includes: transcription, llm_output, validation_output, tts_audio_base64, response_text, response_type, error.

- Direct TTS (wav stream):
  ```bash
  curl "http://localhost:8000/tts/Hello%20world" --output hello.wav
  ```

Configuration notes
- Orchestrator uses environment variables for downstream service URLs (STT_URL, LLM_URL, VALIDATOR_URL, TTS_URL).
- The LLM image installs Ollama (network access and appropriate permissions required). If you run into Ollama install issues, check Docker logs and host compatibility.

Development workflow
- Services are independent — run or test them individually:
  - Start a service: run its Dockerfile or use uvicorn to launch the FastAPI app.
  - Tests: run pytest in service directories (if present).
- Data expansion tools (llm-service/app/data_expansion) help convert/augment training examples used by the RAG pipeline.

Troubleshooting & tips
- If start.sh shows "bad interpreter" or permissions errors, ensure CRLF -> LF normalization is applied (Dockerfile already includes a normalization step).
- For large models, running LLM & embeddings may require additional memory/CPU. Use smaller test models for development.

Contributing
- Open issues for bugs/features.
- Create PRs against main; include tests where applicable and keep changes per-service.
- Follow consistent formatting and add docs for new endpoints.

