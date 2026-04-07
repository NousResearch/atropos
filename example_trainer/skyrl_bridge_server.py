import logging
import os
from typing import Any, Dict, List

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from transformers import AutoTokenizer

app = FastAPI()

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] SkyRL-Bridge: %(message)s"
)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model = None
tokenizer = None


@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
    logger.info(f"Loading SkyRL-Native Bridge | model: {model_path} | device: cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = HFModelWrapper(
        model_path,
        use_flash_attention_2=False,  # Stable SDPA for RTX 3090/CUDA 13
        bf16=True,
        device_map="cuda:0",
    )
    model.eval()
    logger.info("SkyRL-Native Bridge is ready.")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()

    # Handle vLLM prompt format: {"prompt": {"prompt_token_ids": [...]}} OR {"prompt": "..."}
    prompt_data = data.get("prompt")
    if isinstance(prompt_data, dict):
        prompt_token_ids = prompt_data.get("prompt_token_ids")
        input_ids = torch.tensor([prompt_token_ids]).to("cuda:0")
    else:
        # Fallback to text prompt
        inputs = tokenizer(prompt_data, return_tensors="pt").to("cuda:0")
        input_ids = inputs.input_ids
        prompt_token_ids = input_ids[0].tolist()

    max_new_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 1.0)
    n = data.get("n", 1)  # Number of completions

    responses = []
    # vLLM-style logprobs (first token of response)
    # Atropos expects logprobs: [[{token_id: logprob}, ...]] for each position

    # Simple generation loop for 'n' completions
    for _ in range(n):
        with torch.no_grad():
            output = model.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0),
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = output.sequences[0][len(input_ids[0]) :].tolist()

        # Calculate logprobs for generated tokens
        # scores is a tuple of (max_new_tokens,) tensors of shape (batch, vocab_size)
        logprobs_list = []
        for i, score in enumerate(output.scores):
            # score is (1, vocab_size)
            probs = torch.log_softmax(score, dim=-1)
            token_id = gen_tokens[i]
            token_logprob = probs[0, token_id].item()
            # Format: [{token_id: logprob}] as expected by vllm_server.py:215
            logprobs_list.append([{str(token_id): token_logprob}])

        responses.append(
            {
                "token_ids": gen_tokens,
                "logprobs": logprobs_list,
                "finish_reason": (
                    "stop" if gen_tokens[-1] == tokenizer.eos_token_id else "length"
                ),
            }
        )

    # Mimic vLLM response format
    # results["logprobs"] is a list of logprobs_list for each 'n' completion
    result = {
        "logprobs": [resp["logprobs"] for resp in responses],
        "finish_reasons": [resp["finish_reason"] for resp in responses],
    }
    return JSONResponse(content=result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
