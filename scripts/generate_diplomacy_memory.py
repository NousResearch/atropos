#!/usr/bin/env python3
"""
Generate memory blocks for Diplomacy trajectory using Hermes-405B.

Implements sliding window to manage 120k context limit and generates
a memory block for each turn that summarizes game state and strategy.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
import tiktoken
import time


class MemoryGenerator:
    """Handles memory generation with sliding window context management."""
    
    def __init__(self, api_key: str, max_context_tokens: int = 100000):
        """
        Initialize the memory generator.
        
        Args:
            api_key: Hermes API key
            max_context_tokens: Maximum tokens for context (default 100k)
        """
        self.api_key = api_key
        self.max_context_tokens = max_context_tokens
        
        # Initialize tokenizer for accurate token counting
        print("Loading tokenizer...")
        # Use cl100k_base which is close enough for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # API configuration
        self.api_url = "https://inference-api.nousresearch.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        return len(self.tokenizer.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens in a list of messages."""
        # Simple approximation: concatenate all message content
        total_text = ""
        for msg in messages:
            total_text += f"{msg['role']}: {msg['content']}\n\n"
        return self.count_tokens(total_text)
    
    def manage_sliding_window(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Manage sliding window to keep messages within token limit.
        
        Strategy:
        1. Always keep system message
        2. Always keep last 10 turns (20 messages)
        3. Drop oldest messages when exceeding limit
        """
        if not messages:
            return messages
        
        # Calculate current token count
        current_tokens = self.count_messages_tokens(messages)
        
        if current_tokens <= self.max_context_tokens:
            return messages
        
        print(f"  Context exceeds limit ({current_tokens} > {self.max_context_tokens}), applying sliding window...")
        
        # Keep system message and minimum recent context
        system_msg = messages[0] if messages[0]['role'] == 'system' else None
        min_keep = 20  # Keep at least 10 turns (20 messages)
        
        # Start with system + recent messages
        if system_msg:
            windowed = [system_msg] + messages[-min_keep:]
        else:
            windowed = messages[-min_keep:]
        
        # Add older messages if there's room
        remaining_msgs = messages[1:-min_keep] if system_msg else messages[:-min_keep]
        
        for msg in reversed(remaining_msgs):
            test_msgs = [windowed[0]] + [msg] + windowed[1:] if system_msg else [msg] + windowed
            test_tokens = self.count_messages_tokens(test_msgs)
            
            if test_tokens <= self.max_context_tokens:
                if system_msg:
                    windowed = [windowed[0]] + [msg] + windowed[1:]
                else:
                    windowed = [msg] + windowed
            else:
                break
        
        final_tokens = self.count_messages_tokens(windowed)
        print(f"  Windowed context: {len(windowed)} messages, {final_tokens} tokens")
        
        return windowed
    
    async def generate_memory(
        self, 
        messages: List[Dict[str, str]], 
        previous_memory: Optional[str] = None,
        interaction_num: int = 0,
        phase: str = "",
        retry_count: int = 3
    ) -> str:
        """
        Generate a memory block for the current turn.
        
        Args:
            messages: Conversation history
            previous_memory: Previous turn's memory
            interaction_num: Current interaction number
            phase: Current game phase
            retry_count: Number of retries on failure
            
        Returns:
            Generated memory text
        """
        # Prepare memory generation prompt
        memory_prompt = self._create_memory_prompt(messages, previous_memory, phase)
        
        # Apply sliding window if needed
        context_messages = self.manage_sliding_window(messages)
        
        # Add memory generation instruction
        generation_messages = context_messages + [
            {"role": "user", "content": memory_prompt}
        ]
        
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        # Try to generate memory
        for attempt in range(retry_count):
            try:
                self.last_request_time = time.time()
                
                payload = {
                    "model": "Hermes-4-405B",
                    "messages": generation_messages,
                    "max_tokens": 300,
                    "temperature": 0.7,
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
                            memory = result["choices"][0]["message"]["content"]
                            
                            # Extract memory if it's wrapped in tags
                            if "<memory>" in memory:
                                memory = memory.split("<memory>")[1].split("</memory>")[0].strip()
                            
                            # If it's JSON, try to extract reasoning or make a summary
                            if memory.strip().startswith('{'):
                                try:
                                    data = json.loads(memory)
                                    if 'reasoning' in data:
                                        # Extract key points from reasoning
                                        reasoning = data['reasoning']
                                        # Take first 2 sentences
                                        sentences = reasoning.split('. ')[:2]
                                        memory = '. '.join(sentences) + '.'
                                except:
                                    pass
                            
                            # Ensure memory is not too long
                            if len(memory) > 500:
                                sentences = memory.split('. ')[:3]
                                memory = '. '.join(sentences) + '.'
                            
                            return memory
                        else:
                            error_text = await resp.text()
                            print(f"  API error (attempt {attempt+1}): {resp.status} - {error_text[:200]}")
                            
                            if resp.status == 429:  # Rate limited
                                await asyncio.sleep(5 * (attempt + 1))
                            
            except Exception as e:
                print(f"  Error generating memory (attempt {attempt+1}): {e}")
                await asyncio.sleep(2)
        
        # Fallback memory if generation fails
        return f"Turn {interaction_num} in phase {phase}. Continuing game as Italy."
    
    def _create_memory_prompt(
        self, 
        messages: List[Dict[str, str]], 
        previous_memory: Optional[str],
        phase: str
    ) -> str:
        """Create the prompt for memory generation."""
        # Extract recent events from messages
        recent_events = []
        for msg in messages[-6:]:  # Last 3 turns
            if msg['role'] == 'user':
                # Extract key info from user message
                content = msg['content']
                if 'Phase:' in content:
                    phase_info = content.split('Phase:')[1].split('\n')[0].strip()
                    recent_events.append(f"Phase update: {phase_info}")
            elif msg['role'] == 'assistant':
                # Extract key actions from assistant
                if 'send_message' in msg['content']:
                    recent_events.append("Sent diplomatic messages")
                elif 'analyze_phase' in msg['content']:
                    recent_events.append("Analyzed game state")
                elif 'submit_orders' in msg['content']:
                    recent_events.append("Submitted unit orders")
        
        recent_str = "\n".join(recent_events[-3:]) if recent_events else "No recent events"
        
        prompt = f"""You are tracking the game state for Italy in a Diplomacy game.

Current phase: {phase}

Previous memory:
{previous_memory if previous_memory else "Starting fresh - no previous memory"}

Recent events:
{recent_str}

Generate a 2-3 sentence memory summary that captures:
1. Italy's current position (supply centers, key units)
2. Active alliances and enemies
3. Immediate strategic objectives
4. Critical context from recent diplomacy

Be concise but comprehensive. Focus on actionable information for decision-making.
Format: Plain text, 2-3 sentences maximum."""
        
        return prompt
    
    def extract_phase_info(self, user_content: str) -> tuple[str, str]:
        """Extract year and phase from user message."""
        year = "unknown"
        phase = "unknown"
        
        if 'Year:' in user_content:
            year = user_content.split('Year:')[1].split('\n')[0].strip()
        if 'Phase:' in user_content:
            phase = user_content.split('Phase:')[1].split('\n')[0].strip()
        
        return year, phase


async def process_trajectory(
    input_path: Path,
    output_path: Path,
    api_key: str,
    start_from: int = 0,
    limit: Optional[int] = None
):
    """
    Process the trajectory and generate memory for each turn.
    
    Args:
        input_path: Path to simplified trajectory
        output_path: Path to save output with memories
        api_key: Hermes API key
        start_from: Starting interaction number (for resuming)
        limit: Maximum number of interactions to process
    """
    generator = MemoryGenerator(api_key)
    
    # Load the simplified trajectory
    print("Loading trajectory...")
    interactions = []
    with open(input_path, 'r') as f:
        for line in f:
            interactions.append(json.loads(line))
    
    print(f"Loaded {len(interactions)} interactions")
    
    # Check for existing checkpoint files
    checkpoint_files = sorted(output_path.parent.glob(f"{output_path.stem}.checkpoint_*.jsonl"))
    
    # Initialize variables
    results = []
    previous_memory = None
    messages_history = []
    actual_start = start_from
    checkpoint_data = []  # Store checkpoint data for later
    
    # Add system prompt
    system_prompt = """You are playing Diplomacy as ITALY. You are demonstrating optimal play that leads to victory.
Your goal is to achieve a solo win by controlling 18 supply centers.
Maintain strategic thinking and adapt to changing game dynamics."""
    
    messages_history.append({"role": "system", "content": system_prompt})
    
    # If checkpoint exists, load from it
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_num = int(latest_checkpoint.stem.split('_')[-1])
        
        print(f"Found checkpoint at interaction {checkpoint_num}, loading state...")
        
        # Load checkpoint data
        checkpoint_data = []
        with open(latest_checkpoint, 'r') as f:
            for line in f:
                checkpoint_data.append(json.loads(line))
        
        if checkpoint_data:
            # Get the last processed interaction
            last_interaction = checkpoint_data[-1]['interaction']
            
            # Resume from the next interaction
            actual_start = last_interaction  # Start from the next one (0-indexed)
            
            # Don't load checkpoint data into results - we'll handle it at save time
            # Just get previous memory from last entry
            previous_memory = checkpoint_data[-1]['memory']
            
            # Rebuild message history from all interactions up to checkpoint
            print(f"Rebuilding message history from {last_interaction} interactions...")
            for i in range(last_interaction):
                interaction = interactions[i]
                messages_history.append({"role": "user", "content": interaction['user']})
                messages_history.append({"role": "assistant", "content": interaction['assistant']})
            
            print(f"Resuming from interaction {actual_start + 1} with {len(messages_history)} messages in history")
    
    end_at = min(actual_start + limit, len(interactions)) if limit else len(interactions)
    
    for i in range(actual_start, end_at):
        interaction = interactions[i]
        interaction_num = interaction['interaction']
        year = interaction['year']
        phase = interaction['phase']
        
        print(f"\nProcessing interaction {interaction_num} ({i+1}/{end_at}): {year} {phase}")
        
        # Add user message to history
        messages_history.append({"role": "user", "content": interaction['user']})
        
        # Generate memory for this turn
        memory = await generator.generate_memory(
            messages_history,
            previous_memory,
            interaction_num,
            f"{year}_{phase}"
        )
        
        print(f"  Generated memory: {memory[:100]}...")
        
        # Add assistant message to history
        messages_history.append({"role": "assistant", "content": interaction['assistant']})
        
        # Store result
        result = {
            'interaction': interaction_num,
            'year': year,
            'phase': phase,
            'memory': memory,
            'user': interaction['user'],
            'assistant': interaction['assistant']
        }
        results.append(result)
        
        # Update previous memory
        previous_memory = memory
        
        # Save periodically
        if (i + 1) % 10 == 0:
            print(f"  Saving checkpoint at interaction {interaction_num}...")
            save_path = output_path.with_suffix(f'.checkpoint_{interaction_num}.jsonl')
            with open(save_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Save final results
    print(f"\nSaving final results to {output_path}...")
    
    # If we resumed from a checkpoint, combine checkpoint data with new results
    all_results = []
    if checkpoint_files and checkpoint_data:
        all_results = checkpoint_data + results
    else:
        all_results = results
    
    with open(output_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"Successfully generated memory for {len(all_results)} interactions")
    
    # Clean up checkpoint files after successful completion
    if all_results:
        checkpoint_files = sorted(output_path.parent.glob(f"{output_path.stem}.checkpoint_*.jsonl"))
        for checkpoint_file in checkpoint_files:
            print(f"Removing checkpoint file: {checkpoint_file}")
            checkpoint_file.unlink()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate memory blocks for Diplomacy trajectory')
    parser.add_argument('--start', type=int, default=0, help='Starting interaction (for resuming)')
    parser.add_argument('--limit', type=int, default=None, help='Max interactions to process')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 5 interactions')
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("HERMES_API_KEY") or os.getenv("NOUS_API_KEY")
    if not api_key:
        print("Error: No HERMES_API_KEY or NOUS_API_KEY found in environment")
        return
    
    input_file = Path('/home/maxpaperclips/atropos/data/diplomacy_full_trajectory_simple.jsonl')
    output_file = Path('/home/maxpaperclips/atropos/data/diplomacy_with_memory.jsonl')
    
    if args.test:
        print("Running in TEST mode - processing only 5 interactions")
        args.limit = 5
    
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Starting from interaction: {args.start}")
    if args.limit:
        print(f"Processing limit: {args.limit}")
    
    # Run async processing
    asyncio.run(process_trajectory(
        input_file,
        output_file,
        api_key,
        start_from=args.start,
        limit=args.limit
    ))


if __name__ == '__main__':
    main()