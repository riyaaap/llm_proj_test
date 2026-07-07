import os
import sys
import time
import subprocess
import requests
import json
import gradio as gr

# HARDWARE/ENVIRONMENT CONFIG stuff
# force system to use only GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# enable pytorch memory management
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force backend to TORCH/FLASH_ATTN to bypass flashinfer JIT errors
# os.environ["VLLM_USE_V1"] = "0"
# os.environ["VLLM_DISABLE_FLASHINFER"] = "1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
# os.environ["FLASHINFER_DISABLE_JIT"] = "1"

# Pipeline Path Configurations
ENV_PYTHON_PATH = "/mnt/data/riya/llm_proj/venv/bin/python"
MODEL_NAME = "./Qwen2.5-7B-Summarizer"
VLLM_URL = "http://127.0.0"

# automated vLLM backend service initiation
def launch_vllm():

    if not os.path.exists(ENV_PYTHON_PATH):
        print(f"Error: Could not find virtual environment at: {ENV_PYTHON_PATH}", file=sys.stderr)
        sys.exit(1)

    print("Initializing automated vLLM bg engine on GPU 1..")
    print(f"Routing execution through environment: {ENV_PYTHON_PATH}")
    #define exact execution args
    vllm_cmd = [
        ENV_PYTHON_PATH, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", "8000",
        "--host", "127.0.0.1",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "2048",
        "--disable-custom-all-reduce",  ## Prevents single-GPU process hanging
    ]

    # launch vLLM as independent process to not block python execution
    process = subprocess.Popen(
        vllm_cmd,
        stdout=None,
        stderr=None,
        env=os.environ.copy()
    )

    # wait until backend server port is fully responsive
    print("Waiting for model weights to load into VRAM (takes around 1-2 minutes)..")
    retries = 0
    while retries < 40:
        try:
            # Ping the vLLM internal health model API endpoint
            response = requests.get("http://127.0.0", timeout=2)
            if response.status_code == 200:
                print("vLLM Engine fully stabilized and active on port 8000!")
                return process
        except requests.exceptions.ConnectionError:
            time.sleep(5)
            retries += 1

    print("Timeout: vLLM backend failed to boot up within 2.5 min")
    process.terminate()
    sys.exit(1)

# Gradio frontend engine functionality

def generate_summary(article_text, max_tokens, temperature):
    if not article_text.strip():
        return "Please input an article first!"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert assistant. Summarize the following article accurately, preserving technical terms, methods, and key focuses, in an easily readable manner that lay audiences can understand."
            },
            {
                "role": "user",
                "content": article_text.strip()
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True  # Streams token generation progressively for responsive UI
    }

    try:
        response = requests.post(VLLM_URL, json=payload, stream=True)
        if response.status_code != 200:
            return f"Backend Server Error: {response.text}"

        partial_summary = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        delta = data_json["choices"]["delta"]
                        if "content" in delta:
                            partial_summary += delta["content"]
                            yield partial_summary
                    except Exception:
                        continue

    except Exception as e:
        yield f"Could not connect to vLLM server: {str(e)}"

# main interface layout, execution
def run_pipeline():
    # Start the vLLM process first
    vllm_process = launch_vllm()

    # Configure and build the interface dashboard layout
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("Automated Qwen2.5-7B Summarization Dashboard")
        gr.Markdown("Fully self-contained serving infrastructure using unquantized FP16 parameters on a single GPU")

        with gr.Row():
            with gr.Column(scale=2):
                input_box = gr.Textbox(label="Source Article Text", placeholder="Paste text here..", lines=15)
                submit_btn = gr.Button("Generate Summary", variant="primary")
            with gr.Column(scale=1):
                output_box = gr.Textbox(label="AI Summary Output", lines=12, interactive=False)
                with gr.Accordion("Advanced Parameters", open=False):
                    max_tokens = gr.Slider(minimum=64, maximum=512, value=256, step=16, label="Max Summary Length")
                    temp = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.05, label="Creativity/Temperature")

        submit_btn.click(fn=generate_summary, inputs=[input_box, max_tokens, temp], outputs=output_box)

    try:
        print("\nStarting UI Dashboard on local port 7860...")
        demo.queue()
        demo.launch(server_name="127.0.0.1", server_port=7860)
    finally:
        # IMPORTANT..: Ensure the background vLLM process is safely terminated when the UI is closed
        print("\nShutting down backend vLLM processes..")
        vllm_process.terminate()
        vllm_process.wait()
        print("Pipeline offline.")

if __name__ == "__main__":
    run_pipeline()
