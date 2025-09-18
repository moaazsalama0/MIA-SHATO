"""
Enhanced orchestrator with better error handling and debugging.
Replace your orchestrator/orchestrator.py with this version.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import os
import traceback

app = FastAPI(title="SHATO Orchestrator")

STT_URL = "http://stt-service:8003/transcribe"
LLM_URL = os.getenv("LLM_URL", "http://llm-service:9000")

def call_llm(transcription_text: str):
    if not transcription_text:
        print("[DEBUG] Empty transcription provided to LLM")
        return {"llm_output": None, "model_raw": None, "retrieved": None, "error": "Empty transcription"}

    try:
        payload = {"message": transcription_text}
        llm_endpoint = f"{LLM_URL}/chat"
        
        print(f"[DEBUG] Sending to LLM-service at {llm_endpoint}")
        print(f"[DEBUG] LLM payload: {payload}")
        
        # Add more detailed request logging
        r = requests.post(llm_endpoint, json=payload, timeout=60)
        
        print(f"[DEBUG] LLM response status: {r.status_code}")
        print(f"[DEBUG] LLM response headers: {dict(r.headers)}")
        
        if r.status_code != 200:
            error_detail = f"LLM service returned {r.status_code}"
            try:
                error_body = r.text
                print(f"[ERROR] LLM error body: {error_body}")
                error_detail += f": {error_body}"
            except:
                pass
            return {"llm_output": None, "model_raw": None, "retrieved": None, "error": error_detail}
        
        r.raise_for_status()
        data = r.json()
        print(f"[DEBUG] LLM response data: {data}")
        
        return {
            "llm_output": data.get("llm_output"),
            "model_raw": data.get("model_raw"),
            "retrieved": data.get("retrieved"),
            "error": data.get("error")
        }
        
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Cannot connect to LLM service at {LLM_URL}: {e}"
        print(f"[ERROR] {error_msg}")
        return {"llm_output": None, "model_raw": None, "retrieved": None, "error": error_msg}
        
    except requests.exceptions.Timeout as e:
        error_msg = f"LLM request timed out: {e}"
        print(f"[ERROR] {error_msg}")
        return {"llm_output": None, "model_raw": None, "retrieved": None, "error": error_msg}
        
    except requests.exceptions.RequestException as e:
        error_msg = f"LLM request failed: {e}"
        print(f"[ERROR] {error_msg}")
        return {"llm_output": None, "model_raw": None, "retrieved": None, "error": error_msg}
        
    except Exception as e:
        error_msg = f"Unexpected LLM error: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return {"llm_output": None, "model_raw": None, "retrieved": None, "error": error_msg}

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio provided")

    print(f"[DEBUG] Processing audio file: {file.filename}")
    
    try:
        audio_bytes = await file.read()
        print(f"[DEBUG] Audio file size: {len(audio_bytes)} bytes")
        
        files = {"file": (file.filename, audio_bytes, "audio/wav")}

        # Step 1: Call STT with better error handling
        try:
            print(f"[DEBUG] Sending audio to STT-service: {file.filename}")
            stt_response = requests.post(STT_URL, files=files, timeout=60)
            
            print(f"[DEBUG] STT response status: {stt_response.status_code}")
            
            if stt_response.status_code != 200:
                error_detail = f"STT service returned {stt_response.status_code}"
                try:
                    error_body = stt_response.text
                    print(f"[ERROR] STT error body: {error_body}")
                    error_detail += f": {error_body}"
                except:
                    pass
                raise HTTPException(status_code=500, detail=error_detail)
            
            stt_response.raise_for_status()
            stt_json = stt_response.json()
            print(f"[DEBUG] STT response: {stt_json}")
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to STT service: {e}"
            print(f"[ERROR] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"STT request failed: {e}"
            print(f"[ERROR] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
            
        except Exception as e:
            error_msg = f"STT processing failed: {e}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_msg)

        # Extract transcription with fallback
        transcription = stt_json.get("text", "") or stt_json.get("transcribed_text", "")
        
        if not transcription:
            print("[WARNING] No transcription text found in STT response")
            transcription = ""

        print(f"[DEBUG] Transcription: '{transcription}'")

        # Step 2: Call LLM
        llm_result = call_llm(transcription)
        print(f"[DEBUG] LLM result: {llm_result}")

        # Step 3: Return combined response
        response = {
            "transcription": transcription,
            "llm_output": llm_result.get("llm_output"),
            "validation_output": {},
            "tts_audio_url": None,
            "error": llm_result.get("error")
        }
        
        print(f"[DEBUG] Final response: {response}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_msg = f"Orchestrator processing failed: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "shato-orchestrator",
        "stt_url": STT_URL,
        "llm_url": LLM_URL
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "orchestrator"}
