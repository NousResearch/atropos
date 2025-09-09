#!/usr/bin/env python3
"""
Add thinking blocks to Diplomacy data with memory.

For each turn, generates strategic thinking that leads to the action,
without revealing future knowledge.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
import time


class ThinkingGenerator:
    """Generates thinking blocks for Diplomacy turns."""
    
    def __init__(self, api_key: str):
        """Initialize the thinking generator."""
        self.api_key = api_key
        
        # API configuration
        self.api_url = "https://inference-api.nousresearch.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0
    
    async def generate_thinking(
        self,
        memory: str,
        user_prompt: str,
        assistant_response: str,
        phase: str,
        retry_count: int = 3
    ) -> str:
        """
        Generate thinking that leads to the action taken.
        
        Args:
            memory: Current memory block
            user_prompt: The user's prompt/game state
            assistant_response: The action taken (tool calls)
            phase: Current game phase
            retry_count: Number of retries
            
        Returns:
            Generated thinking text
        """
        # Extract key info from the assistant response
        action_summary = self._summarize_action(assistant_response)
        
        # Create thinking generation prompt
        thinking_prompt = self._create_thinking_prompt(
            memory, user_prompt, action_summary, phase
        )
        
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        for attempt in range(retry_count):
            try:
                self.last_request_time = time.time()
                
                payload = {
                    "model": "Hermes-4-405B",
                    "messages": [
                        {"role": "system", "content": "You are Italy in a Diplomacy game, reasoning through your next action."},
                        {"role": "user", "content": thinking_prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.8,
                    "top_p": 0.9,
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            thinking = result["choices"][0]["message"]["content"]
                            
                            # Clean up thinking
                            if "<think>" in thinking:
                                thinking = thinking.split("<think>")[1].split("</think>")[0].strip()
                            
                            return thinking
                        else:
                            error_text = await resp.text()
                            print(f"  API error (attempt {attempt+1}): {resp.status}")
                            
                            if resp.status == 429:
                                await asyncio.sleep(5 * (attempt + 1))
                            
            except Exception as e:
                print(f"  Error generating thinking (attempt {attempt+1}): {e}")
                await asyncio.sleep(2)
        
        # Fallback thinking
        return f"Analyzing the situation in {phase}. Considering diplomatic and strategic options."
    
    def _summarize_action(self, assistant_response: str) -> str:
        """Extract a summary of actions from the assistant response."""
        actions = []
        
        if "send_message" in assistant_response:
            # Count messages
            msg_count = assistant_response.count('"name": "send_message"')
            actions.append(f"sending {msg_count} diplomatic messages")
        
        if "analyze_phase" in assistant_response:
            actions.append("analyzing the game state")
        
        if "submit_orders" in assistant_response:
            actions.append("submitting unit orders")
        
        if "summarize_turn" in assistant_response:
            actions.append("summarizing the turn")
        
        if not actions:
            return "taking action"
        
        return ", ".join(actions)
    
    def _create_thinking_prompt(
        self,
        memory: str,
        user_prompt: str,
        action_summary: str,
        phase: str
    ) -> str:
        """Create the prompt for thinking generation."""
        # Extract key info from user prompt
        prompt_type = "unknown"
        if "TASK\nAnalyze the phase" in user_prompt:
            prompt_type = "phase analysis"
        elif "messaging phase" in user_prompt.lower():
            prompt_type = "diplomatic messaging"
        elif "submit your orders" in user_prompt.lower():
            prompt_type = "order submission"
        
        prompt = f"""Current game phase: {phase}

Memory: {memory}

You need to respond to a {prompt_type} request by {action_summary}.

Generate strategic thinking (3-5 sentences) that:
1. Assesses the current situation
2. Considers options and risks
3. Explains why your chosen action makes sense
4. Shows strategic foresight WITHOUT revealing specific future events

Focus on:
- Current board position and power dynamics
- Diplomatic relationships and trust levels
- Strategic objectives and priorities
- Risks and opportunities

Write in first person as Italy, showing your reasoning process."""
        
        return prompt


def format_with_thinking_and_memory(
    interaction: Dict[str, Any],
    thinking: str,
    memory: str
) -> Dict[str, Any]:
    """
    Format the interaction with thinking and memory blocks.
    
    The assistant response becomes:
    <memory>...</memory>
    <think>...</think>
    <tool_call>...</tool_call>
    """
    # Parse the assistant content to get tool calls
    tool_calls = interaction['assistant']
    
    # Build the new assistant content
    new_content = f"""<memory>
{memory}
</memory>

<think>
{thinking}
</think>

{tool_calls}"""
    
    # Create the formatted entry
    formatted = {
        'interaction': interaction.get('interaction', 0),
        'year': interaction.get('year', 'unknown'),
        'phase': interaction.get('phase', 'unknown'),
        'user': interaction['user'],
        'assistant': new_content
    }
    
    return formatted


async def process_with_thinking(
    input_path: Path,
    output_path: Path,
    api_key: str,
    start_from: int = 0,
    limit: Optional[int] = None
):
    """Process interactions and add thinking blocks."""
    generator = ThinkingGenerator(api_key)
    
    # Load interactions with memory
    print("Loading interactions with memory...")
    interactions = []
    with open(input_path, 'r') as f:
        for line in f:
            interactions.append(json.loads(line))
    
    print(f"Loaded {len(interactions)} interactions")
    
    # Process interactions
    results = []
    end_at = min(start_from + limit, len(interactions)) if limit else len(interactions)
    
    for i in range(start_from, end_at):
        interaction = interactions[i]
        interaction_num = interaction.get('interaction', i+1)
        phase = f"{interaction.get('year', 'unknown')}_{interaction.get('phase', 'unknown')}"
        memory = interaction.get('memory', 'No memory available.')
        
        print(f"\nProcessing interaction {interaction_num} ({i+1}/{end_at})")
        
        # Generate thinking
        thinking = await generator.generate_thinking(
            memory,
            interaction['user'],
            interaction['assistant'],
            phase
        )
        
        print(f"  Generated thinking: {thinking[:100]}...")
        
        # Format with thinking and memory
        formatted = format_with_thinking_and_memory(
            interaction,
            thinking,
            memory
        )
        
        results.append(formatted)
        
        # Save periodically
        if (i + 1) % 10 == 0:
            print(f"  Saving checkpoint...")
            checkpoint_path = output_path.with_suffix(f'.checkpoint_{interaction_num}.jsonl')
            with open(checkpoint_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Save final results
    print(f"\nSaving final results to {output_path}...")
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"Successfully added thinking to {len(results)} interactions")


def create_sample_transformed(input_path: Path, output_path: Path, num_samples: int = 5):
    """
    Create sample transformed data without API calls for testing.
    
    This creates a few samples with placeholder thinking and memory.
    """
    print(f"Creating {num_samples} sample transformed entries...")
    
    # Load original data
    interactions = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            interactions.append(json.loads(line))
    
    # Transform each interaction
    results = []
    for i, interaction in enumerate(interactions):
        # Create placeholder memory
        phase = interaction.get('phase', 'unknown')
        year = interaction.get('year', 'unknown')
        memory = f"Italy in {year} {phase}. Maintaining diplomatic relationships and pursuing strategic objectives. Recent actions have advanced our position toward victory."
        
        # Create placeholder thinking
        thinking = f"The current situation requires careful consideration. Our position is strong but we must maintain momentum. The diplomatic landscape offers opportunities that align with our long-term strategy. This action will advance our interests while managing risks."
        
        # Format the entry
        formatted = format_with_thinking_and_memory(
            interaction,
            thinking,
            memory
        )
        
        results.append(formatted)
    
    # Save results
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"Created {len(results)} sample entries at {output_path}")
    
    # Show a sample
    if results:
        print("\nSample transformed entry:")
        print(f"Year/Phase: {results[0]['year']}/{results[0]['phase']}")
        print(f"Assistant response preview:")
        print(results[0]['assistant'][:500] + "...")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add thinking blocks to Diplomacy data')
    parser.add_argument('--start', type=int, default=0, help='Starting interaction')
    parser.add_argument('--limit', type=int, default=None, help='Max interactions to process')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 5')
    parser.add_argument('--sample', action='store_true', help='Create sample output without API')
    args = parser.parse_args()
    
    if args.sample:
        # Create sample transformed data without API calls
        input_file = Path('/home/maxpaperclips/atropos/data/diplomacy_full_trajectory_simple.jsonl')
        output_file = Path('/home/maxpaperclips/atropos/data/diplomacy_sample_transformed.jsonl')
        create_sample_transformed(input_file, output_file, num_samples=10)
        return
    
    # Get API key
    api_key = os.getenv("HERMES_API_KEY") or os.getenv("NOUS_API_KEY")
    if not api_key:
        print("Error: No API key found. Use --sample for testing without API.")
        return
    
    input_file = Path('/home/maxpaperclips/atropos/data/diplomacy_with_memory.jsonl')
    output_file = Path('/home/maxpaperclips/atropos/data/diplomacy_final_transformed.jsonl')
    
    if args.test:
        print("Running in TEST mode")
        args.limit = 5
    
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Run async processing
    asyncio.run(process_with_thinking(
        input_file,
        output_file,
        api_key,
        start_from=args.start,
        limit=args.limit
    ))


if __name__ == '__main__':
    main()