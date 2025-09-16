from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import random

app = FastAPI()

class LLMRequest(BaseModel):
    text: str

class LLMResponse(BaseModel):
    command: Dict[str, Any]
    verbal_response: str

VALID_COMMANDS = {
    "move_to": {"x": int, "y": int},
    "rotate": {"angle": int, "direction":str},
    "start_patrol": {"route_id":str}
}

# Example command responses
COMMAND_EXAMPLES = [
    {"command": {"name": "move_to", "params": {"x": 5, "y": 10}}, "verbal_response": "I'll move to position x:5, y:10"},
    {"command": {"name": "rotate", "params": {"angle": 90},"direction":"left"}, "verbal_response": "Rotating 90 degrees left"},
]

@app.post("/generate", response_model=LLMResponse)
async def generate_command(request: LLMRequest):
    """
    Mock LLM that returns predefined robot commands based on input text.
    """
    text = request.text.lower()
    
    # Simple text matching to return appropriate commands
    if "move" in text or "go" in text or "position" in text:
        response = {"command": {"name": "move_to", "params": {"x": random.randint(1, 10), "y": random.randint(1, 10)}}, "verbal_response": f"Moving to position x:{random.randint(1, 10)}, y:{random.randint(1, 10)}"}
    elif "rotate" in text or "turn" in text:
        response = {"command": {"name": "rotate", "params": {"angle": random.choice([45, 90, 180])}}, "verbal_response": f"Rotating {random.choice([45, 90, 180])} degrees"}
    else:
        # Return a random command if no match
        response = random.choice(COMMAND_EXAMPLES)
        response["verbal_response"] = "I understood your command and will execute it."
    
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mock-llm-service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)