import json
import traceback
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import threading

app = FastAPI(
    title="PACE AI Inference Service",
    description="Hugging Face Deployment for PACE Actor-Critic Ensemble",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    text: str
    mode: Optional[str] = "coding"
    speed_mode: Optional[str] = "pro"

_generate_lock = threading.Lock()

@app.get("/")
def root():
    return {"status": "PACE AI Service Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/generate_stream")
def generate_stream(request: GenerateRequest):
    def event_stream():
        if not _generate_lock.acquire(blocking=False):
            yield json.dumps({"type": "status", "content": "Server busy"}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return
            
        try:
            mode = request.mode
            speed = request.speed_mode
            
            if mode == "literacy":
                from masteries.literacy.inference.v4_orchestrator import literacy_pipeline as active_pipeline
            elif mode == "research":
                from masteries.research.inference.v4_orchestrator import research_pipeline as active_pipeline
            else:
                from masteries.coding.inference.v4_orchestrator import v4_pipeline as active_pipeline

            for event in active_pipeline(request.text, speed_mode=speed):
                # The event is already a dictionary.
                yield json.dumps(event) + "\n"
                
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"
        finally:
            _generate_lock.release()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
