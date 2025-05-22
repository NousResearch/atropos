"""
Trajectory utils

Utils for managing trajectory sizing, formatting, compression, etc.
"""

import logging
from typing import List, Optional, Dict, Any

from transformers import PreTrainedTokenizer

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.type_definitions import Message

logger = logging.getLogger(__name__)


def strip_thinking(response_text: str) -> str:
    """Helper to strip the <think> block of a response entirely.

    Args:
        response_text: The response text to strip.

    Returns:
        The stripped response text.
    """
    think_start_tag = "<think>"
    think_end_tag = "</think>"

    think_start_idx = response_text.find(think_start_tag)
    think_end_idx = response_text.find(think_end_tag)

    if think_start_idx != -1 and think_end_idx != -1:
        return (
            response_text[:think_start_idx]
            + response_text[think_end_idx + len(think_end_tag) :]
        )
    else:
        return response_text


def truncate_thinking(
    response_text: str, tokenizer: PreTrainedTokenizer, max_think_tokens: int
) -> str:
    """Helper to truncate the <think> block of a response for message history based on token count.

    Args:
        response_text: The response text to truncate.
        tokenizer: The tokenizer to use for counting tokens.
        max_think_tokens: The maximum number of tokens to keep in the <think> block.

    Returns:
        The truncated response text.
    """
    try:
        think_start_tag = "<think>"
        think_end_tag = "</think>"

        think_start_idx = response_text.find(think_start_tag)
        think_end_idx = response_text.find(think_end_tag)

        if not (
            think_start_idx != -1
            and think_end_idx != -1
            and think_start_idx < think_end_idx
        ):
            return response_text

        part_before_content = response_text[: think_start_idx + len(think_start_tag)]
        original_think_content_raw = response_text[
            think_start_idx + len(think_start_tag) : think_end_idx
        ]
        part_after_content = response_text[think_end_idx:]

        original_think_content_stripped = original_think_content_raw.strip()

        if not original_think_content_stripped:
            # Normalize empty or whitespace-only think blocks to <think></think>
            return f"{part_before_content.rstrip()}{part_after_content.lstrip()}"

        all_think_tokens = tokenizer.encode(
            original_think_content_stripped, add_special_tokens=False
        )

        is_truncated_internally = False
        final_think_tokens: List[int]

        if len(all_think_tokens) <= max_think_tokens:
            final_think_tokens = all_think_tokens
            is_truncated_internally = False
        else:
            is_truncated_internally = (
                True  # Mark as truncated if len(all_think_tokens) > max_think_tokens
            )
            paragraphs = [
                p.strip()
                for p in original_think_content_stripped.split("\n\n")
                if p.strip()
            ]

            attempted_paragraph_truncation = False
            if paragraphs:
                last_paragraph_text = paragraphs[-1]
                # Check if last paragraph is genuinely shorter than the whole content
                # (i.e., there was content before it)
                if len(last_paragraph_text) < len(original_think_content_stripped):
                    last_paragraph_tokens = tokenizer.encode(
                        last_paragraph_text, add_special_tokens=False
                    )
                    if len(last_paragraph_tokens) <= max_think_tokens:
                        final_think_tokens = last_paragraph_tokens
                        attempted_paragraph_truncation = True

            if (
                not attempted_paragraph_truncation
            ):  # Default to truncating the whole content from the end
                # Ensure max_think_tokens is not negative, though practically it shouldn't be.
                slice_start = max(0, len(all_think_tokens) - max_think_tokens)
                final_think_tokens = all_think_tokens[slice_start:]

        # Decode the tokens to string
        decoded_think_content = tokenizer.decode(
            final_think_tokens, skip_special_tokens=True
        )

        # Add "..." prefix if truncated and content remains
        final_internal_content_str = decoded_think_content
        if is_truncated_internally and decoded_think_content.strip():
            final_internal_content_str = "... " + decoded_think_content.lstrip()

        # Determine the final block content (empty or with newlines)
        final_internal_content_str_stripped = final_internal_content_str.strip()
        final_content_for_block: str
        if (
            not final_internal_content_str_stripped
            or final_internal_content_str_stripped == "..."
        ):
            final_content_for_block = ""
        else:
            final_content_for_block = f"\n{final_internal_content_str_stripped}\n"

        return f"{part_before_content.rstrip()}{final_content_for_block}{part_after_content.lstrip()}"

    except Exception as e:
        logger.error(
            f"Error in truncate_thinking for text '{response_text[:200]}...': {e}",
            exc_info=True,
        )
        return response_text


async def summarize_conversation_history(
    messages: List[Message], 
    server_client,
    tokenizer: PreTrainedTokenizer,
    max_summary_tokens: int = 512,
    preserve_recent_turns: int = 2
) -> List[Message]:
    """
    Summarize conversation history using an LLM call, preserving recent turns.
    
    Args:
        messages: List of messages to potentially summarize
        server_client: Server client for making LLM calls
        tokenizer: Tokenizer for counting tokens
        max_summary_tokens: Maximum tokens for the summary
        preserve_recent_turns: Number of recent turns to keep unsummarized
        
    Returns:
        List of messages with older messages summarized
    """
    if len(messages) <= preserve_recent_turns * 2 + 1:  # system + preserve_recent_turns * (user + assistant)
        return messages
    
    try:
        # Separate system message, messages to summarize, and recent messages to preserve
        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        start_idx = 1 if system_msg else 0
        
        # Calculate how many messages to preserve (recent turns)
        # Each turn is typically user + assistant, so preserve_recent_turns * 2 messages
        preserve_count = preserve_recent_turns * 2
        split_idx = max(start_idx, len(messages) - preserve_count)
        
        messages_to_summarize = messages[start_idx:split_idx]
        recent_messages = messages[split_idx:]
        
        if not messages_to_summarize:
            return messages
        
        # Create summarization prompt
        conversation_text = ""
        for msg in messages_to_summarize:
            role = msg["role"]
            content = msg["content"]
            
            # Strip thinking blocks from assistant messages for summarization
            if role == "assistant":
                content = strip_thinking(content)
            
            conversation_text += f"{role.upper()}: {content}\n\n"
        
        summarization_prompt = f"""Please provide a concise summary of the following conversation history. Focus on:
1. Key game state information and observations
2. Important actions taken and their outcomes
3. Strategic decisions and reasoning (without full chains of thought)
4. Current objectives and progress

Conversation to summarize:
{conversation_text.strip()}

Provide a summary in 2-3 sentences that captures the essential context needed to continue the conversation effectively."""

        # Make LLM call for summarization
        summary_messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations concisely while preserving important context."},
            {"role": "user", "content": summarization_prompt}
        ]
        
        response = await server_client.chat_completion(
            messages=summary_messages,
            max_tokens=max_summary_tokens,
            temperature=0.1  # Low temperature for consistent summaries
        )
        
        summary_content = response.choices[0].message.content.strip()
        
        # Create summarized message list
        result_messages = []
        if system_msg:
            result_messages.append(system_msg)
        
        # Add summary as a special message
        result_messages.append({
            "role": "user", 
            "content": f"[CONVERSATION SUMMARY]: {summary_content}"
        })
        
        # Add recent messages
        result_messages.extend(recent_messages)
        
        logger.info(f"Summarized {len(messages_to_summarize)} messages into summary. "
                   f"Original tokens: ~{sum(len(tokenizer.encode(msg['content'])) for msg in messages_to_summarize)}, "
                   f"Summary tokens: ~{len(tokenizer.encode(summary_content))}")
        
        return result_messages
        
    except Exception as e:
        logger.error(f"Error in summarize_conversation_history: {e}", exc_info=True)
        return messages  # Return original messages if summarization fails


def prepare_reward_model_input(
    scored_data_group: ScoredDataGroup,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 4096,
    strip_thinking_from_history: bool = True,
    summarize_thinking_with_llm: bool = False,
    server_client = None,
    max_thinking_summary_tokens: int = 128
) -> ScoredDataGroup:
    """
    Prepare ScoredDataGroup for reward model by filtering and truncating appropriately.
    
    Args:
        scored_data_group: Original ScoredDataGroup with full conversation history
        tokenizer: Tokenizer for counting tokens
        max_tokens: Maximum tokens for reward model input
        strip_thinking_from_history: Whether to strip <think> blocks from previous messages
        summarize_thinking_with_llm: Whether to use LLM to summarize thinking blocks instead of stripping
        server_client: Server client for LLM calls (required if summarize_thinking_with_llm=True)
        max_thinking_summary_tokens: Maximum tokens for thinking block summaries
        
    Returns:
        Filtered and truncated ScoredDataGroup suitable for reward model
    """
    try:
        import asyncio
        
        async def process_messages_async():
            filtered_messages = []
            filtered_tokens = []
            filtered_masks = []
            
            for alt_idx, alt_messages in enumerate(scored_data_group["messages"]):
                # Process messages for reward model
                processed_messages = []
                
                for msg_idx, msg in enumerate(alt_messages):
                    processed_msg = msg.copy()
                    
                    # Process thinking blocks in assistant messages (except the last one)
                    if (msg["role"] == "assistant" and msg_idx < len(alt_messages) - 1):
                        if summarize_thinking_with_llm and server_client:
                            # Use LLM-based summarization
                            processed_msg["content"] = await summarize_thinking_block(
                                msg["content"], 
                                server_client, 
                                tokenizer, 
                                max_thinking_summary_tokens
                            )
                        elif strip_thinking_from_history:
                            # Strip thinking blocks entirely
                            processed_msg["content"] = strip_thinking(msg["content"])
                    
                    processed_messages.append(processed_msg)
                
                # Apply token limit truncation
                truncated_messages = ensure_message_token_limit(
                    processed_messages, tokenizer, max_tokens
                )
                
                # Re-tokenize the processed messages
                try:
                    tokenized_output = tokenize_for_trainer(tokenizer, truncated_messages)
                    filtered_messages.append(truncated_messages)
                    filtered_tokens.append(tokenized_output["tokens"])
                    filtered_masks.append(tokenized_output["masks"])
                except Exception as e:
                    logger.error(f"Error re-tokenizing messages for reward model alt {alt_idx}: {e}")
                    # Keep original if re-tokenization fails
                    filtered_messages.append(alt_messages)
                    filtered_tokens.append(scored_data_group["tokens"][alt_idx] if alt_idx < len(scored_data_group["tokens"]) else [])
                    filtered_masks.append(scored_data_group["masks"][alt_idx] if alt_idx < len(scored_data_group["masks"]) else [])
            
            return filtered_messages, filtered_tokens, filtered_masks
        
        # Check if we're in an async context
        if summarize_thinking_with_llm and server_client:
            try:
                # Try to run the async processing
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, process_messages_async())
                        filtered_messages, filtered_tokens, filtered_masks = future.result()
                else:
                    # We're not in an async context, run directly
                    filtered_messages, filtered_tokens, filtered_masks = asyncio.run(process_messages_async())
            except Exception as e:
                logger.error(f"Error in async processing, falling back to sync: {e}")
                # Fall back to synchronous processing without LLM summarization
                summarize_thinking_with_llm = False
        
        # Synchronous fallback processing
        if not (summarize_thinking_with_llm and server_client):
            filtered_messages = []
            filtered_tokens = []
            filtered_masks = []
            
            for alt_idx, alt_messages in enumerate(scored_data_group["messages"]):
                processed_messages = []
                
                for msg_idx, msg in enumerate(alt_messages):
                    processed_msg = msg.copy()
                    
                    # Strip thinking blocks from all but the last assistant message
                    if (msg["role"] == "assistant" and 
                        strip_thinking_from_history and 
                        msg_idx < len(alt_messages) - 1):
                        processed_msg["content"] = strip_thinking(msg["content"])
                    
                    processed_messages.append(processed_msg)
                
                # Apply token limit truncation
                truncated_messages = ensure_message_token_limit(
                    processed_messages, tokenizer, max_tokens
                )
                
                # Re-tokenize the processed messages
                try:
                    tokenized_output = tokenize_for_trainer(tokenizer, truncated_messages)
                    filtered_messages.append(truncated_messages)
                    filtered_tokens.append(tokenized_output["tokens"])
                    filtered_masks.append(tokenized_output["masks"])
                except Exception as e:
                    logger.error(f"Error re-tokenizing messages for reward model alt {alt_idx}: {e}")
                    # Keep original if re-tokenization fails
                    filtered_messages.append(alt_messages)
                    filtered_tokens.append(scored_data_group["tokens"][alt_idx] if alt_idx < len(scored_data_group["tokens"]) else [])
                    filtered_masks.append(scored_data_group["masks"][alt_idx] if alt_idx < len(scored_data_group["masks"]) else [])
        
        # Create new ScoredDataGroup with filtered data
        filtered_sdg = ScoredDataGroup(
            tokens=filtered_tokens,
            masks=filtered_masks,
            scores=scored_data_group["scores"],
            messages=filtered_messages,
            metadata={
                **(scored_data_group.get("metadata", {})),
                "reward_model_processed": True,
                "thinking_summarized_with_llm": summarize_thinking_with_llm,
                "original_message_count": len(scored_data_group["messages"][0]) if scored_data_group["messages"] else 0,
                "filtered_message_count": len(filtered_messages[0]) if filtered_messages else 0
            }
        )
        
        return filtered_sdg
        
    except Exception as e:
        logger.error(f"Error in prepare_reward_model_input: {e}", exc_info=True)
        return scored_data_group  # Return original if processing fails


def ensure_message_token_limit(
    messages: List[Message],
    tokenizer: PreTrainedTokenizer,
    max_tokens: int
) -> List[Message]:
    """
    Ensure message list doesn't exceed token limit by truncating older messages.
    Preserves system message and recent messages.
    
    Args:
        messages: List of messages to check and potentially truncate
        tokenizer: Tokenizer for counting tokens
        max_tokens: Maximum allowed tokens
        
    Returns:
        Truncated message list within token limit
    """
    if not messages:
        return messages
    
    try:
        # Count current tokens
        total_tokens = sum(len(tokenizer.encode(msg["content"], add_special_tokens=False)) for msg in messages)
        
        if total_tokens <= max_tokens:
            return messages
        
        logger.info(f"Message history ({total_tokens} tokens) exceeds limit ({max_tokens}). Truncating...")
        
        # Preserve system message and last few messages
        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        start_idx = 1 if system_msg else 0
        
        # Keep last 4 messages (2 turns) minimum
        preserve_count = min(4, len(messages) - start_idx)
        
        # Binary search to find optimal truncation point
        left, right = start_idx, len(messages) - preserve_count
        best_split = start_idx
        
        while left <= right:
            mid = (left + right) // 2
            
            # Test messages from mid to end
            test_messages = []
            if system_msg:
                test_messages.append(system_msg)
            test_messages.extend(messages[mid:])
            
            test_tokens = sum(len(tokenizer.encode(msg["content"], add_special_tokens=False)) for msg in test_messages)
            
            if test_tokens <= max_tokens:
                best_split = mid
                right = mid - 1
            else:
                left = mid + 1
        
        # Construct final message list
        result_messages = []
        if system_msg:
            result_messages.append(system_msg)
        
        if best_split > start_idx:
            # Add truncation indicator
            result_messages.append({
                "role": "user",
                "content": "[CONVERSATION TRUNCATED - Earlier messages removed due to length]"
            })
        
        result_messages.extend(messages[best_split:])
        
        final_tokens = sum(len(tokenizer.encode(msg["content"], add_special_tokens=False)) for msg in result_messages)
        logger.info(f"Truncated to {len(result_messages)} messages ({final_tokens} tokens)")
        
        return result_messages
        
    except Exception as e:
        logger.error(f"Error in ensure_message_token_limit: {e}", exc_info=True)
        return messages


def manage_token_budget(
    messages: List[Message],
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    summarization_trigger_ratio: float = 0.8,
    server_client = None
) -> tuple[List[Message], bool]:
    """
    Manage token budget by triggering summarization or truncation as needed.
    
    Args:
        messages: Current message history
        tokenizer: Tokenizer for counting tokens
        max_tokens: Maximum allowed tokens
        summarization_trigger_ratio: Ratio of max_tokens that triggers summarization
        server_client: Server client for LLM calls (required for summarization)
        
    Returns:
        Tuple of (processed_messages, was_summarized)
    """
    try:
        current_tokens = sum(len(tokenizer.encode(msg["content"], add_special_tokens=False)) for msg in messages)
        trigger_threshold = int(max_tokens * summarization_trigger_ratio)
        
        if current_tokens <= trigger_threshold:
            return messages, False
        
        logger.info(f"Token budget management triggered. Current: {current_tokens}, Threshold: {trigger_threshold}")
        
        # Try summarization first if server_client is available
        if server_client and current_tokens > trigger_threshold:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(summarize_conversation_history):
                    # If we're in an async context, we need to handle this properly
                    # For now, fall back to truncation if we can't await
                    logger.warning("Cannot await summarization in sync context, falling back to truncation")
                    processed_messages = ensure_message_token_limit(messages, tokenizer, max_tokens)
                    return processed_messages, False
                else:
                    summarized_messages = summarize_conversation_history(
                        messages, server_client, tokenizer
                    )
                    return summarized_messages, True
            except Exception as e:
                logger.error(f"Summarization failed, falling back to truncation: {e}")
        
        # Fall back to truncation
        processed_messages = ensure_message_token_limit(messages, tokenizer, max_tokens)
        return processed_messages, False
        
    except Exception as e:
        logger.error(f"Error in manage_token_budget: {e}", exc_info=True)
        return messages, False


def ensure_trajectory_token_limit(
    trajectory: List[ScoredDataGroup],
    tokenizer: PreTrainedTokenizer,
    max_trajectory_tokens: int,
) -> List[ScoredDataGroup]:
    """
    Ensure token sequences in a trajectory don't exceed max_trajectory_tokens.
    Attempts to uniformly truncate older messages (preferably paired turns) from all alternatives within a step.
    The system prompt, last environment observation, and last agent response are preserved as a minimum.
    If a step still exceeds the limit after maximum possible truncation, it is discarded.

    Args:
        trajectory: List of ScoredDataGroup from an episode

    Returns:
        The trajectory with potentially truncated messages/tokens/masks or filtered steps
    """
    if not trajectory:
        return trajectory

    filtered_trajectory: List[ScoredDataGroup] = []

    for step_idx, original_step_data in enumerate(trajectory):
        if not (
            original_step_data.get("messages")
            and original_step_data.get("tokens")
            and original_step_data.get("masks")
            and original_step_data.get("seed") is not None
            and original_step_data.get("parsed_actions") is not None
        ):
            logger.warning(
                f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env "
                f"is missing critical data. Skipping."
            )
            continue

        max_initial_tokens = 0
        if original_step_data["tokens"]:
            max_initial_tokens = (
                max(
                    len(alt_tokens)
                    for alt_tokens in original_step_data["tokens"]
                    if isinstance(alt_tokens, list)
                )
                if any(
                    isinstance(alt_tokens, list)
                    for alt_tokens in original_step_data["tokens"]
                )
                else 0
            )

        if max_initial_tokens <= max_trajectory_tokens:
            filtered_trajectory.append(original_step_data)
            logger.info(
                f"[_ensure_trajectory_token_limit] Step {step_idx} compliant in MC env. "
                f"Max tokens: {max_initial_tokens}"
            )
            continue

        logger.info(
            f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env (max tokens: {max_initial_tokens}) "
            f"exceeds limit ({max_trajectory_tokens}). Attempting truncation."
        )

        working_messages = [
            msgs_list.copy() for msgs_list in original_step_data["messages"] or []
        ]
        working_tokens = [
            tkns_list.copy() for tkns_list in original_step_data["tokens"] or []
        ]
        working_masks = [
            msks_list.copy() for msks_list in original_step_data["masks"] or []
        ]
        max_current_tokens = max_initial_tokens
        num_alternatives = len(working_messages)

        if num_alternatives == 0:
            logger.warning(
                f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env has no alternatives"
                " after copying. Skipping."
            )
            continue

        retokenization_error_this_step = False
        while max_current_tokens > max_trajectory_tokens:
            target_pop_counts_per_alt = []
            for alt_idx in range(num_alternatives):
                alt_msg_list = working_messages[alt_idx]
                num_preserved_at_end = 0
                if len(alt_msg_list) > 1 and alt_msg_list[-1]["role"] in [
                    "agent",
                    "assistant",
                ]:
                    num_preserved_at_end = 1
                    if (
                        len(alt_msg_list) > 2
                        and alt_msg_list[-2]["role"] == "environment"
                    ):
                        num_preserved_at_end = 2

                available_to_pop = len(alt_msg_list) - 1 - num_preserved_at_end

                if available_to_pop <= 0:
                    target_pop_counts_per_alt.append(0)
                else:
                    can_pop_pair = (
                        available_to_pop >= 2
                        and len(alt_msg_list) > 2
                        and alt_msg_list[1]["role"] == "environment"
                        and alt_msg_list[2]["role"] in ["agent", "assistant"]
                    )
                    if can_pop_pair:
                        target_pop_counts_per_alt.append(2)
                    else:
                        target_pop_counts_per_alt.append(1)

            positive_pop_counts = [c for c in target_pop_counts_per_alt if c > 0]
            if not positive_pop_counts:
                break

            min_pop_this_round = min(positive_pop_counts)
            temp_new_alt_tokens = []
            temp_new_alt_masks = []
            max_tokens_after_this_trunc = 0

            for alt_idx in range(num_alternatives):
                for _ in range(min_pop_this_round):
                    if len(working_messages[alt_idx]) > 1:
                        working_messages[alt_idx].pop(1)
                    else:
                        logger.error(
                            f"[_ensure_trajectory_token_limit] MC env: Critical error during pop for "
                            f"alt {alt_idx}, step {step_idx}. List too short."
                        )
                        retokenization_error_this_step = True
                        break
                if retokenization_error_this_step:
                    break

                try:
                    tokenized_alt = tokenize_for_trainer(
                        tokenizer, working_messages[alt_idx]
                    )
                    temp_new_alt_tokens.append(tokenized_alt["tokens"])
                    temp_new_alt_masks.append(tokenized_alt["masks"])
                    max_tokens_after_this_trunc = max(
                        max_tokens_after_this_trunc, len(tokenized_alt["tokens"])
                    )
                except Exception as e:
                    logger.error(
                        f"[_ensure_trajectory_token_limit] MC env: Error re-tokenizing alt {alt_idx} "
                        f"in step {step_idx} after truncation: {e}"
                    )
                    retokenization_error_this_step = True
                    break

            if retokenization_error_this_step:
                break

            working_tokens = temp_new_alt_tokens
            working_masks = temp_new_alt_masks
            max_current_tokens = max_tokens_after_this_trunc
            logger.debug(
                f"[_ensure_trajectory_token_limit] MC env: Step {step_idx}, "
                f"after uniform pop of {min_pop_this_round}, "
                f"max tokens: {max_current_tokens}"
            )

        if (
            not retokenization_error_this_step
            and max_current_tokens <= max_trajectory_tokens
        ):
            updated_step_data: ScoredDataGroup = {
                "seed": original_step_data["seed"],
                "messages": working_messages,
                "tokens": working_tokens,
                "masks": working_masks,
                "scores": original_step_data.get("scores"),
                "parsed_actions": original_step_data.get("parsed_actions"),
            }
            filtered_trajectory.append(updated_step_data)
            logger.info(
                f"[_ensure_trajectory_token_limit] MC env: Step {step_idx} successfully processed. "
                f"Final max tokens: {max_current_tokens}"
            )
        else:
            logger.warning(
                f"[_ensure_trajectory_token_limit] MC env: Discarding step {step_idx}. "
                f"Max tokens ({max_current_tokens}) still exceed limit ({max_trajectory_tokens}) "
                f"or retokenization error occurred ({retokenization_error_this_step})."
            )

    if len(filtered_trajectory) < len(trajectory):
        logger.warning(
            f"[_ensure_trajectory_token_limit] MC env: Filtered out "
            f"{len(trajectory) - len(filtered_trajectory)} steps "
            f"due to token limit constraints. Original: {len(trajectory)}, Filtered: {len(filtered_trajectory)}"
        )
    return filtered_trajectory


async def summarize_thinking_block(
    response_text: str,
    server_client,
    tokenizer: PreTrainedTokenizer,
    max_summary_tokens: int = 256
) -> str:
    """
    Summarize the thinking block in a response using an LLM call.
    Preserves key reasoning, strategic planning, and conclusions while making it more concise.
    
    Args:
        response_text: The response text containing a thinking block to summarize
        server_client: Server client for making LLM calls
        tokenizer: Tokenizer for counting tokens
        max_summary_tokens: Maximum tokens for the summarized thinking block
        
    Returns:
        Response text with summarized thinking block, or original if no thinking block found
    """
    try:
        think_start_tag = "<think>"
        think_end_tag = "</think>"

        think_start_idx = response_text.find(think_start_tag)
        think_end_idx = response_text.find(think_end_tag)

        # If no thinking block found, return original
        if not (think_start_idx != -1 and think_end_idx != -1 and think_start_idx < think_end_idx):
            return response_text

        # Extract parts
        part_before_thinking = response_text[:think_start_idx]
        original_thinking_content = response_text[think_start_idx + len(think_start_tag):think_end_idx]
        part_after_thinking = response_text[think_end_idx + len(think_end_tag):]

        # Skip summarization if thinking block is already short or empty
        original_thinking_stripped = original_thinking_content.strip()
        if not original_thinking_stripped:
            return response_text
        
        original_tokens = tokenizer.encode(original_thinking_stripped, add_special_tokens=False)
        if len(original_tokens) <= max_summary_tokens:
            return response_text

        # Create summarization prompt
        summarization_prompt = f"""Please summarize the following thinking block concisely while preserving:
1. Key reasoning steps that led to the final decision/conclusion
2. Strategic planning information needed for subsequent actions
3. Important observations or insights about the current situation
4. The logical flow from analysis to decision

Keep the summary brief but comprehensive enough that someone could understand the essential reasoning process.

Original thinking block:
{original_thinking_stripped}

Provide only the summarized thinking content (without the <think> tags)."""

        # Make LLM call for summarization
        summary_messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that summarizes reasoning processes concisely while preserving essential strategic information and logical flow."
            },
            {"role": "user", "content": summarization_prompt}
        ]
        
        response = await server_client.chat_completion(
            messages=summary_messages,
            max_tokens=max_summary_tokens,
            temperature=0.1  # Low temperature for consistent summaries
        )
        
        if not (response and response.choices and response.choices[0].message):
            logger.warning("LLM summarization call failed or returned empty response")
            return response_text
            
        summary_content = response.choices[0].message.content.strip()
        
        if not summary_content:
            logger.warning("LLM returned empty summary for thinking block")
            return response_text
        
        # Reconstruct the response with summarized thinking block
        summarized_response = f"{part_before_thinking}{think_start_tag}\n{summary_content}\n{think_end_tag}{part_after_thinking}"
        
        # Log the summarization
        original_token_count = len(original_tokens)
        summary_token_count = len(tokenizer.encode(summary_content, add_special_tokens=False))
        logger.info(f"Summarized thinking block: {original_token_count} -> {summary_token_count} tokens "
                   f"({summary_token_count/original_token_count*100:.1f}% of original)")
        
        return summarized_response
        
    except Exception as e:
        logger.error(f"Error in summarize_thinking_block: {e}", exc_info=True)
        return response_text  # Return original if summarization fails
