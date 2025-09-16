import gradio as gr
import requests
import uuid
import os

ORCHESTRATOR_URL = "http://orchestrator:8000/process_audio/"

def process_audio_ui(audio_file):
    if not audio_file:
        return "No audio provided", None, None, None, None

    # Prepare file
    try:
        with open(audio_file, "rb") as f:
            files = {"audio": (str(uuid.uuid4()) + ".wav", f, "audio/wav")}
            response = requests.post(ORCHESTRATOR_URL, files=files, timeout=60)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return f"Error contacting orchestrator: {e}", None, None, None, None

    # extract fields from orchestrator response
    transcription = data.get("transcription", "")
    llm_output = data.get("llm_output", {})
    validation_output = data.get("validation_output", {})
    tts_audio_url = data.get("tts_audio_url", None)
    error = data.get("error", None)

    # Convert tts_audio_url to a filepath Gradio can play (if local file)
    tts_audio_path = None
    if tts_audio_url and os.path.isfile(tts_audio_url):
        tts_audio_path = tts_audio_url  # Local path works
    elif tts_audio_url and tts_audio_url.startswith("http"):
        tts_audio_path = tts_audio_url  # Remote file (if hosted somewhere)

    return transcription, llm_output, validation_output, tts_audio_path, error

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Robot Voice Assistant UI")
    gr.Markdown("Record audio to control SHATO.")

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
            submit_btn = gr.Button("Send to Orchestrator")

        with gr.Column():
            transcription = gr.Textbox(label="STT Transcription")
            llm_output = gr.JSON(label="LLM Output (Full JSON)")
            validation_output = gr.JSON(label="Validation Result")
            tts_audio = gr.Audio(label="TTS Response (Robot Reply)", type="filepath")
            error_box = gr.Textbox(label="Errors", interactive=False)

    submit_btn.click(
        process_audio_ui,
        inputs=audio_in,
        outputs=[transcription, llm_output, validation_output, tts_audio, error_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
