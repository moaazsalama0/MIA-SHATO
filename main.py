from fastapi import FastAPI, HTTPException
from validator import Command, validate_command

app = FastAPI(title="Robot Validator & Control Service")

@app.post("/execute_command")
def execute_command(cmd: Command):
    try:
        params = validate_command(cmd)
        print(f"[ROBOT-VALIDATOR-SUCCESS] Received and validated command: '{cmd.command}' with params {params}")
        return {"status": "success", "command": cmd.command, "params": params}

    except Exception as e:
        detail_msg = f"[ROBOT-VALIDATOR-ERROR] Invalid command or parameters: {str(e)}"
        print(detail_msg)
        raise HTTPException(status_code=400, detail={"status": "error", "reason": detail_msg})
