from fastapi import FastAPI, HTTPException
from validator import Command, validate_command
import uvicorn  # ADD THIS

app = FastAPI(title="Robot Validator & Control Service")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "robot-validator"}

@app.post("/execute_command")
async def execute_command(cmd: Command):  # ADD async
    try:
        params = validate_command(cmd)
        print(f"[ROBOT-VALIDATOR-SUCCESS] Received and validated command: '{cmd.command}' with params {params}")
        return {"status": "success", "command": cmd.command, "params": params}

    except Exception as e:
        detail_msg = f"[ROBOT-VALIDATOR-ERROR] Invalid command or parameters: {str(e)}"
        print(detail_msg)
        raise HTTPException(status_code=400, detail=detail_msg)  # REMOVE the nested dict

if __name__ == "__main__":  # ADD THIS
    uvicorn.run(app, host="0.0.0.0", port=8001)  # ADD THIS