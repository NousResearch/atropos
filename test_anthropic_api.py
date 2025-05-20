from anthropic import Anthropic
import os

# Initialize Anthropic client
api_key = os.getenv('ANTHROPIC_API_KEY')
anthropic = Anthropic(api_key=api_key)

# Define a simple prompt for testing
prompt = """You are an AI research assistant. Provide a brief summary of the importance of testing API keys."""

try:
    # Make a request to the Anthropic API
    response = anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=100,
        temperature=0,
        system="You are an AI research assistant.",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    # Print the response
    print("API Response:", response.content)
except Exception as e:
    print("Error testing Anthropic API:", str(e)) 