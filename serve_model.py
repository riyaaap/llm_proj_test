import os
import subprocess
import sys

# clean corrupted compiler artifact directory automatically
CACHE_DIR = "/home/riya/.cache/flashinfer"
if os.path.exists(CACHE_DIR):
    print(f"Clearing corrupted compiler cache at: {CACHE_DIR}")
    subprocess.run(["rm", "-rf", CACHE_DIR])

# Expose compilation environment paths to Ninja/C++ tools
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

# turn off V1 engine architecture to force fallback execution rules
#os.environ["VLLM_USE_V1"] = "0"
# tell older engine versions to explicitly avoid FlashInfer JIT 
#os.environ["VLLM_DISABLE_FLASHINFER"] = "1"
# Force backend to TORCH to bypass flashinfer JIT errors
#os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
#os.environ["FLASHINFER_DISABLE_JIT"] = "1"

# target the exact virtual environment python binary containing the vLLM install
ENV_PYTHON_PATH = "/mnt/data/riya/llm_proj/.env/bin/python"

# define exact serving command and arguments for memory usage, context length, etc.
COMMAND = [
	ENV_PYTHON_PATH, "-m", "vllm.entrypoints.openai.api_server",
	"--model", "Qwen/Qwen2.5-7B",
	"--host", "0.0.0.0",
	"--port", "8000",
    	"--gpu-memory-utilization", "0.30", # restricts vLLM to ~7.2 GB of VRAM from nvidia L4 gpu chip
    	"--max-model-len", "2048" # caps cpmtext window to free up allocation blocks 
]

def main():
	#first verify env binary act exists 
	if not os.path.exists(ENV_PYTHON_PATH):
		print(f"error could not find virtual environment at: {ENV_PYTHON_PATH}", file=sys.stderr)
		print("please check file path and try again", file=sys.stderr)
		sys.exit(1)

	print("Disabling vLLM V1 engine arch and FlashInfer compilation to Sett VLLM_ATTENTION_BACKEND to TORCH")
	print(f"routing execution through environment: {ENV_PYTHON_PATH}")
	print(f"Launching server command...\n")

	try:
		# run the server directly using target environment context
		#and launching with explicitly isolated env configurations 
		subprocess.run(COMMAND, check=True, env=os.environ.copy())
	except subprocess.CalledProcessError as e:
		print(f"\n Server exited with an error code: {e.returncode}", file=sys.stderr)
	except KeyboardInterrupt:
		print("\n Server stopped manually by user.")

if __name__ == "__main__":
	main()
