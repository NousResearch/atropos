#!/usr/bin/env python
import asyncio
import argparse
from datasets import load_dataset
import openai
import os

# Reuse the prompts from our goofy math server
system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \\boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \\boxed{your answer here}"""

# Define the goofiness preference string
goofiness_preference = (
    "be the GOOFIEST math solver ever! Use wild exaggerations, silly sound effects, "
    "dramatic reactions to calculations, personify numbers, and be totally over-the-top "
    "enthusiastic! Don't just solve the problem - make it a PERFORMANCE! Give your solution "
    "with maximum silliness - include dramatic gasps, unexpected tangents, and random sound effects. "
    "But still get the answer right, you absolute mathematical goofball! Your answers should "
    "feel like they're coming from an extremely enthusiastic but chaotic math genius."
)

async def query_model(question, api_key, add_goofiness=False):
    """Make a request to OpenAI to simulate the environment behavior"""
    client = openai.AsyncOpenAI(api_key=api_key)
    
    # Prepare the prompt with or without goofiness
    if add_goofiness:
        full_prompt = f"{system_prompt}\n\n{goofiness_preference}"
    else:
        full_prompt = system_prompt
    
    try:
        # Make the request - using gpt-3.5-turbo to keep costs down
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

async def main():
    parser = argparse.ArgumentParser(description="Demo the goofy math solver")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--num_examples", type=int, default=2, help="Number of examples to generate")
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Provide it with --api_key or set OPENAI_API_KEY environment variable.")
        return
    
    # Load examples from GSM8K test set
    dataset = load_dataset("gsm8k", "main", split="test")
    
    for i in range(min(args.num_examples, len(dataset))):
        question = dataset[i]["question"]
        gold_answer = dataset[i]["answer"].split("#")[-1].strip().replace(",", "")
        
        print(f"\n{'='*80}\n")
        print(f"QUESTION {i+1}: {question}")
        print(f"GOLD ANSWER: {gold_answer}")
        
        print(f"\n{'-'*40}\nSTANDARD RESPONSE:\n{'-'*40}")
        standard_response = await query_model(question, api_key, add_goofiness=False)
        print(standard_response)
        
        print(f"\n{'-'*40}\nGOOFY RESPONSE:\n{'-'*40}")
        goofy_response = await query_model(question, api_key, add_goofiness=True)
        print(goofy_response)
        
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main()) 