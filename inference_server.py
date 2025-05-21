from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./opt-350m")
tokenizer = AutoTokenizer.from_pretrained("./opt-350m")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    prompt = messages[-1]["content"] if messages else ""

    # Simulate a numerical rating (0-1) for now
    response_text = "0.9"

    # Return in OpenAI-compatible format
    return {
        "choices": [{"message": {"content": response_text}}]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9001)