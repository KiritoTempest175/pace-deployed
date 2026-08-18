import json
import traceback
import gradio as gr
import spaces
import threading

_generate_lock = threading.Lock()

@spaces.GPU
def generate_stream_gpu(text: str, mode: str, speed_mode: str):
    if not _generate_lock.acquire(blocking=False):
        yield json.dumps({"type": "status", "content": "Server busy"}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
        return
        
    try:
        if mode == "literacy":
            from masteries.literacy.inference.v4_orchestrator import literacy_pipeline as active_pipeline
        elif mode == "research":
            from masteries.research.inference.v4_orchestrator import research_pipeline as active_pipeline
        else:
            from masteries.coding.inference.v4_orchestrator import v4_pipeline as active_pipeline

        for event in active_pipeline(text, speed_mode=speed_mode):
            yield json.dumps(event) + "\n"
            
        yield json.dumps({"type": "done"}) + "\n"
    except Exception as e:
        traceback.print_exc()
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
    finally:
        _generate_lock.release()


demo = gr.Interface(
    fn=generate_stream_gpu,
    inputs=[
        gr.Textbox(label="text"),
        gr.Textbox(label="mode", value="coding"),
        gr.Textbox(label="speed_mode", value="pro")
    ],
    outputs=gr.Textbox(label="output"),
    title="PACE AI Inference Service",
    description="Backend AI API for PACE. Used via Gradio Client."
)

if __name__ == "__main__":
    demo.launch()
