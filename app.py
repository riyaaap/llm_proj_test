import os
import requests
import json
import gradio as gr

# Backend vLLM server endpoint
VLLM_URL = "http://127.0.0"
MODEL_NAME = "./Qwen2.5-7B-Summarizer"

def generate_summary(article_text, max_tokens, temperature):
    if not article_text.strip():
        return "Please input an article first!"

    # Format prompt structure exactly like training data schema
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
        "stream": True # Enable text streaming for better UI experience
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(VLLM_URL, json=payload, headers=headers, stream=True)
        if response.status_code != 200:
            return f"Backend Error: {response.text}"

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
                            yield partial_summary # Streams text word-by-word into the textbox
                    except Exception:
                        continue

    except Exception as e:
        yield f"Could not connect to vLLM server: {str(e)}\nMake sure your vLLM server is running on port 8000!"

# building Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Qwen2.5-7B Custom Article Summarizer")
    gr.Markdown("Fine-tuned using LoRA on the CNN/DailyMail dataset from hugging face, and optimized with vLLM acceleration.")

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="Source Article Text",
                placeholder="Paste your long article or text here..",
                lines=15

            submit_btn = gr.Button("Generate Summary", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="AI Summary Output",
                lines=10,
                interactive=False
            )
            with gr.Accordion("Advanced Generation Parameters", open=False):
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=512, value=256, step=16,
                    label="Max Summary Length"
                )
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                    label="Temperature (Creativity)"
                )

    # Start execution trigger
    submit_btn.click(
        fn=generate_summary,
        inputs=[input_box, max_tokens_slider, temp_slider],
        outputs=output_box
    )

if __name__ == "__main__":
    # Launch on localhost port 7860
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860)

## If working on remote server via SSH, forward ports to local machine as follows:
## ssh -L 7860:127.0.0.1:7860 -L 8000:127.0.0.1:8000 user@your-server-ip

