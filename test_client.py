import json
import urllib.request

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "Qwen/Qwen2.5-7B",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning in one sentence."}
    ],
    "temperature": 0.7
}

try:
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        print("Model Response:\n", result["choices"][0]["message"]["content"])
except Exception as e:
    print(f"Failed to reach server: {e}")
