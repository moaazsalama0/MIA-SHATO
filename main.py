from fastapi import FastAPI, HTTPException, Response, Form
from fastapi.responses import HTMLResponse
import uvicorn
import logging

from tts_processor import create_tts_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTSService")

app = FastAPI(title="TTS")

tts_processor = create_tts_processor()

@app.post("/tts")
async def tts_get(text: str, voice: str = "af_heart"):
    if not tts_processor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    result = tts_processor.text_to_speech(text, voice)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return Response(
        content=result["audio_data"],
        media_type="audio/wav"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=False)