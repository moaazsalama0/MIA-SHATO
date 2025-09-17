import uuid
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

# ----------------- CONFIG -----------------
LLM_URL = "http://llm:8002/generate"
VALIDATOR_URL = "http://validator:8001/execute_command"
STT_URL = "http://stt:8003/transcribe"
TTS_URL = "http://tts:8004/speak"

# ----------------- APP INIT -----------------
app = FastAPI(title="Orchestrator Service")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("orchestrator")

# ----------------- MODELS -----------------
class OrchestratorResponse(BaseModel):
    request_id: str
    status: str
    transcription: str | None
    llm_output: dict | None
    validation_output: dict | None
    tts_audio_url: str | None
    error: str | None

# ----------------- HELPERS -----------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def call_service(url: str, payload: dict = None, files: dict = None):
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, files=files, timeout=20)
        resp.raise_for_status()
        return resp.json()

# ----------------- ENDPOINTS -----------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "orchestrator"}

@app.post("/process_audio/", response_model=OrchestratorResponse)
async def process_audio(audio: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] New request received.")

    try:
        # Step 1: Send audio to STT
        logger.info(f"[{request_id}] Sending audio to STT service...")
        stt_data = await call_service(STT_URL, files={"audio": (audio.filename, await audio.read(), audio.content_type)})
        transcription = stt_data.get("text")
        if not transcription:
            raise HTTPException(status_code=502, detail="STT did not return text")
        logger.info(f"[{request_id}] STT transcription: {transcription}")

        # Step 2: Send text to LLM
        logger.info(f"[{request_id}] Sending text to LLM...")
        llm_data = await call_service(LLM_URL, payload={"input_text": transcription})
        cmd_json = llm_data.get("command_json")
        verbal_response = llm_data.get("verbal_response", "Command processed.")
        if not cmd_json:
            raise HTTPException(status_code=502, detail="LLM did not return command_json")
        logger.info(f"[{request_id}] LLM output received.")

        # Step 3: Validate command
        logger.info(f"[{request_id}] Sending command to Validator...")
        val_data = await call_service(VALIDATOR_URL, payload=cmd_json)
        logger.info(f"[{request_id}] Command validated successfully.")

        # Step 4: Convert verbal_response to speech via TTS
        logger.info(f"[{request_id}] Sending verbal response to TTS...")
        tts_data = await call_service(TTS_URL, payload={"text": verbal_response})
        tts_audio_url = tts_data.get("audio_url")
        logger.info(f"[{request_id}] TTS audio generated: {tts_audio_url}")

        return OrchestratorResponse(
            request_id=request_id,
            status="success",
            transcription=transcription,
            llm_output=llm_data,
            validation_output=val_data,
            tts_audio_url=tts_audio_url,
            error=None,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error occurred: {e}")
        return OrchestratorResponse(
            request_id=request_id,
            status="failed",
            transcription=None,
            llm_output=None,
            validation_output=None,
            tts_audio_url=None,
            error=str(e),
        )
