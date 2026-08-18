import json
import time
from pathlib import Path
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from masteries.api.schemas import (
    PredictRequest,
    PredictResponse,
    TelemetryResponse,
    CreateConversationRequest,
)
from masteries.services.chunker import chunk_text
from masteries.services.pdfparser import extracttext, pagenumber
from masteries.services.telemetry import (
    get_system_telemetry,
    update_last_execution_metrics,
)
from masteries.services.database import (
    get_conversations,
    get_conversation,
    create_conversation,
    delete_conversation,
    add_message,
)

import threading

_generate_lock = threading.Lock()

router = APIRouter()


@router.get("/")
def root():
    return {"message": "PACE Backend Running"}


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/telemetry", response_model=TelemetryResponse)
def get_telemetry():
    return get_system_telemetry()


@router.get("/conversations")
def list_conversations():
    return get_conversations()


@router.post("/conversations")
def create_new_conversation(req: CreateConversationRequest):
    cid = create_conversation(
        title=req.title or "New Session", workspace=req.workspace or "coding"
    )
    return get_conversation(cid)


@router.get("/conversations/{conversation_id}")
def read_conversation(conversation_id: str):
    conv = get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    delete_conversation(conversation_id)
    return {"status": "success", "message": f"Deleted conversation {conversation_id}"}


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        from masteries.coding.inference.actor_generate import generate_fixes

        fixes = generate_fixes(request.text, num_return_sequences=1)
        return PredictResponse(prediction=fixes[0], status="success")
    except Exception as e:
        import traceback

        traceback.print_exc()
        return PredictResponse(
            prediction=f"[CPU Fallback] You entered: {request.text}\n\n(Error: {e})",
            status="degraded",
        )


@router.post("/generate")
def generate(request: PredictRequest):
    """
    Primary endpoint: accepts a text prompt from the frontend chat,
    proxies it to the AI Service (Hugging Face), streams output tokens,
    saves the conversation & messages into SQLite, and broadcasts real execution telemetry.
    """
    import os
    import httpx

    mode = request.mode or "coding"
    speed = request.speed_mode or "pro"
    
    # AI service URL configured via environment variables
    AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://127.0.0.1:8001")

    # Obtain or create conversation ID
    conversation_id = request.conversation_id
    if not conversation_id:
        title = request.text[:32] + ("..." if len(request.text) > 32 else "")
        conversation_id = create_conversation(title=title, workspace=mode)

    # Persist user message to SQLite database
    add_message(conversation_id, role="user", text=request.text)

    async def event_stream():
        if not _generate_lock.acquire(blocking=False):
            yield f"data: {json.dumps({'type': 'status', 'content': 'Server is currently busy with another request. Please wait a moment and try again.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        start_time = time.time()
        ttft_ms = None
        tokens_generated = 0
        assistant_accumulated_text = ""
        last_telemetry_emit = 0.0
        _active_actor_model = "External AI Service"
        _active_critic_model = "External AI Service"

        yield f"data: {json.dumps({'type': 'init', 'conversation_id': conversation_id})}\n\n"

        def get_current_metrics(status_str="processing"):
            nonlocal ttft_ms, tokens_generated, start_time
            now = time.time()
            elapsed_s = max(0.001, now - start_time)
            latency_ms = int(elapsed_s * 1000)
            tps = round(tokens_generated / elapsed_s, 1) if tokens_generated > 0 else 0.0

            sys_telemetry = get_system_telemetry()
            sys_telemetry.update({
                "status": status_str,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "generation_time_s": round(elapsed_s, 2),
                "tokens_generated": tokens_generated,
                "tokens_per_sec": tps,
                "timestamp": now,
                "actor_model": _active_actor_model,
                "critic_model": _active_critic_model,
            })
            return sys_telemetry

        try:
            init_metrics = get_current_metrics("processing")
            update_last_execution_metrics(init_metrics)
            yield f"data: {json.dumps({'type': 'telemetry', 'metrics': init_metrics})}\n\n"

            payload = {
                "text": request.text,
                "mode": mode,
                "speed_mode": speed
            }

            from gradio_client import Client
            
            client = Client(AI_SERVICE_URL)
            job = client.submit(
                request.text,
                mode,
                speed,
                api_name="/generate_stream_gpu"
            )

            # Gradio Client yields outputs iteratively if the function is a generator
            for result in job:
                # `result` is the string yielded by the Gradio function
                if not result:
                    continue
                    
                try:
                    # In our case, gradio gives us the *entire* accumulated string from the generator so far in each tick if we yielded everything, but wait!
                    # If we yield json strings line by line in our generator, gradio_client might yield the accumulated outputs, or the latest output.
                    # Since our generator yields independent JSON strings, gradio actually returns the *latest* yielded value, not the accumulated one, but wait, `gradio_client` with `submit()` gives us exactly what was yielded!
                    # Actually, if we yield a string per token, Gradio client might yield the string.
                    # Gradio strings can contain multiple yielded lines if the generator is fast.
                    last_line = result.strip().split("\n")[-1]
                    event = json.loads(last_line)
                except:
                    continue

                event_type = event.get("type")

                if event_type == "token":
                    content = event.get("content", "")
                    assistant_accumulated_text += content
                    tokens_generated += 1
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - start_time) * 1000)
                elif event_type == "clear":
                    assistant_accumulated_text = ""

                yield f"data: {json.dumps(event)}\n\n"

                now = time.time()
                if now - last_telemetry_emit >= 0.2:
                    current_m = get_current_metrics("processing")
                    update_last_execution_metrics(current_m)
                    yield f"data: {json.dumps({'type': 'telemetry', 'metrics': current_m})}\n\n"
                    last_telemetry_emit = now

            # Save completed message to DB
            add_message(
                conversation_id=conversation_id,
                role="assistant",
                text=assistant_accumulated_text,
                source="actor-critic-ensemble",
                status="Critic Validated",
            )

            final_metrics = get_current_metrics("completed")
            update_last_execution_metrics(final_metrics)
            yield f"data: {json.dumps({'type': 'telemetry', 'metrics': final_metrics})}\n\n"
            
            # Type done might be yielded by proxy, but if not we can guarantee it here. 
            # Note: the proxy sends {"type": "done"}, so it is forwarded above. 
        except Exception as e:
            import traceback
            traceback.print_exc()
            response_text = f"[PACE Backend Proxy] Error communicating with AI service: {e}"
            add_message(
                conversation_id=conversation_id,
                role="assistant",
                text=response_text,
                source="actor-critic-ensemble",
                status="Error",
            )
            fallback_metrics = get_current_metrics("completed")
            update_last_execution_metrics(fallback_metrics)
            yield f"data: {json.dumps({'type': 'telemetry', 'metrics': fallback_metrics})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'content': response_text})}\n\n"
        finally:
            _generate_lock.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename
    if not filename or not filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are allowed.", "status": "error"}

    file_path = UPLOAD_DIR / filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extracttext(str(file_path))
        pages = pagenumber(str(file_path))
        chunks = chunk_text(text)
        return {
            "filename": filename,
            "pages": pages,
            "characters": len(text),
            "chunks": len(chunks),
            "preview": chunks[0] if chunks else "",
            "status": "success",
            "message": f"Successfully processed '{filename}' ({pages} pages, {len(chunks)} chunks extracted).",
        }
    except Exception as e:
        return {
            "filename": filename,
            "pages": 0,
            "characters": 0,
            "chunks": 0,
            "preview": f"Uploaded '{filename}' successfully.",
            "status": "success",
            "message": f"File uploaded: {filename} (Note: {e})",
        }
