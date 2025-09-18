from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import os

app = FastAPI(title="SHATO Orchestrator")

STT_URL = "http://stt-service:8003/transcribe"
LLM_URL = os.getenv("LLM_URL", "http://llm-service:9000")

def call_llm(transcription_text: str):
    if not transcription_text:
        return {"llm_output": None, "model_raw": None, "retrieved": None, "error": "Empty transcription"}

    try:
        payload = {"message": transcription_text}
        print(f"[DEBUG] Sending to LLM-service at {LLM_URL}/chat: {payload}")
        r = requests.post(f"{LLM_URL}/chat", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        print(f"[DEBUG] LLM response: {data}")
        return {
            "llm_output": data.get("model_json") or data.get("llm_output") or data,
            "model_raw": data.get("model_raw"),
            "retrieved": data.get("retrieved"),
            "error": data.get("error")
        }
    except Exception as e:
        print(f"[ERROR] LLM request failed: {e}")
        return {"llm_output": None, "model_raw": None, "retrieved": None, "error": f"LLM error: {e}"}

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio provided")

    audio_bytes = await file.read()
    files = {"file": (file.filename, audio_bytes, "audio/wav")}

    # Step 1: Call STT
    try:
        print(f"[DEBUG] Sending audio to STT-service: {file.filename}")
        stt_response = requests.post(STT_URL, files=files, timeout=60)
        stt_response.raise_for_status()
        stt_json = stt_response.json()
        print(f"[DEBUG] STT response: {stt_json}")
    except Exception as e:
        print(f"[ERROR] STT request failed: {e}")
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

    transcription = stt_json.get("transcribed_text", "")

    # Step 2: Call LLM
    llm_result = call_llm(transcription)

    # Step 3: Return combined response
    return {
        "transcription": transcription,
        "llm_output": llm_result.get("llm_output"),
        "validation_output": {},
        "tts_audio_url": None,
        "error": llm_result.get("error")
    }
