import torch
from transformers import PreTrainedTokenizer

from atroposlib.type_definitions import Message

# Roles that should be masked in the loss calculation (not used for training)
UNMASKED_ROLES = ["assistant", "agent"]


def tokenize_for_trainer_multistep(
    tokenizer: PreTrainedTokenizer,
    chat: list[Message],
    include_messages: bool = False,
    finish_reason: str = "",
) -> dict:
    """
    Tokenizes a list of chat messages for training in a multistep RL environment.

    This function is specifically designed for scenarios where previous assistant messages
    in the chat history might have been truncated or summarized to manage context length.
    To ensure the model learns from high-quality, complete responses, this function
    implements a specific masking strategy:
    - Only the content of the *last* message with a role in UNMASKED_ROLES (e.g., 'assistant', 'agent')
      is unmasked for loss calculation (i.e., its labels will be the token IDs).
    - All other parts of the chat, including system prompts, user messages, tool calls/responses,
      and any prior assistant/agent messages (which might be modified), are masked out (labels set to -100).

    This approach prevents the model from being trained on potentially noisy or incomplete
    data from summarized/truncated turns, focusing the learning signal on the final,
    presumably complete, assistant generation in the sequence.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        chat (list[Message]): A list of chat messages. Previous assistant messages may
                              be truncated or summarized.
        include_messages (bool): Whether to include the original `chat` messages in the output dict.
                                 Defaults to False.
        finish_reason (str): Optional string indicating the reason generation finished.
                             If "length", the last token might be truncated if it's an EOS token,
                             as this can be an artifact of hitting max length rather than a deliberate stop.
                             Defaults to "".

    Returns:
        dict: A dictionary containing:
              - "tokens" (list[int]): The tokenized IDs for the entire chat.
              - "masks" (list[int]): The labels for training. Token IDs for the last assistant
                                     message's content, -100 otherwise.
              - "messages" (list[Message], optional): The input chat, if `include_messages` is True.
    """
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True)
    if not isinstance(input_ids, list):  # Ensure it's a list for consistency
        input_ids = input_ids.tolist()

    labels = torch.ones(len(input_ids), dtype=torch.long) * -100

    last_unmasked_message_idx = -1
    for i in range(len(chat) - 1, -1, -1):
        if chat[i]["role"] in UNMASKED_ROLES:
            last_unmasked_message_idx = i
            break

    if last_unmasked_message_idx != -1:
        # Determine the token span for the content of chat[last_unmasked_message_idx]

        # Tokens of all messages *before* chat[last_unmasked_message_idx],
        # plus the role prompt for it.
        # `add_generation_prompt=True` prepares the template for chat[last_unmasked_message_idx] to start.
        tokens_before_target_message_content_starts = tokenizer.apply_chat_template(
            chat[:last_unmasked_message_idx], tokenize=True, add_generation_prompt=True
        )
        if not isinstance(tokens_before_target_message_content_starts, list):
            tokens_before_target_message_content_starts = (
                tokens_before_target_message_content_starts.tolist()
            )

        # Tokens of all messages *up to and including* chat[last_unmasked_message_idx].
        # `add_generation_prompt=False` (default) ensures no *extra* prompt after it.
        tokens_up_to_target_message_content_ends = tokenizer.apply_chat_template(
            chat[: last_unmasked_message_idx + 1], tokenize=True
        )
        if not isinstance(tokens_up_to_target_message_content_ends, list):
            tokens_up_to_target_message_content_ends = (
                tokens_up_to_target_message_content_ends.tolist()
            )

        start_idx = len(tokens_before_target_message_content_starts)
        end_idx = len(tokens_up_to_target_message_content_ends)

        if 0 <= start_idx < end_idx <= len(input_ids):
            actual_token_ids_for_label = torch.tensor(
                input_ids[start_idx:end_idx], dtype=torch.long
            )
            labels[start_idx:end_idx] = actual_token_ids_for_label
        else:
            pass

    final_labels = labels.tolist()

    if finish_reason == "length":
        if input_ids and input_ids[-1] == tokenizer.eos_token_id:
            input_ids = input_ids[:-1]
            final_labels = final_labels[:-1]

    return {
        "tokens": input_ids,
        "masks": final_labels,  # "masks" is used for labels in this context
    } | ({"messages": chat} if include_messages else {})


def tokenize_for_trainer(
    tokenizer: PreTrainedTokenizer,
    chat: list[Message],
    include_messages: bool = False,
    train_on_all_assistant_turns: bool = False,
    finish_reason: str = "",
) -> dict:
    """
    Tokenize a list of chat messages for the trainer.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        chat (list): A list of chat messages.
        include_messages (bool): Whether to include the messages in the output.
        train_on_all_assistant_turns (bool): If True, mask out system/user/tool roles.
                                            If False, use the original prefix masking.
    Returns:
        dict: A dictionary containing the tokenized chat messages.
    """

    tokens = tokenizer.apply_chat_template(chat)

    if not train_on_all_assistant_turns:
        prefix_len = len(
            tokenizer.apply_chat_template(chat[:-1], add_generation_prompt=True)
        )
        masks = [-100] * prefix_len + tokens[prefix_len:]
    else:
        # NOTE: This implementation will break if the default system prompt is used and depends on world state
        # (e.g. current date). e.g. consider a system prompt that depends on the current date and a run that crosses
        # midnight from 3/9 to 3/10 under a tokenizer that tokenizes 3/9 and 3/10 with a different number of tokens.

        masks = torch.ones(len(tokens), dtype=torch.long) * -100

        for i, msg in enumerate(chat):
            if msg["role"] in UNMASKED_ROLES:
                prefix_tokens = tokenizer.apply_chat_template(
                    chat[:i], tokenize=True, add_generation_prompt=True
                )
                unmasked_tokens = tokenizer.apply_chat_template(
                    chat[: i + 1], tokenize=True
                )
                start_idx = len(prefix_tokens)
                end_idx = len(unmasked_tokens)
                masks[start_idx:end_idx] = torch.tensor(unmasked_tokens[start_idx:])

        masks = masks.tolist()
    if finish_reason == "length":
        if tokens[-1] == tokenizer.eos_token_id:
            print("bad token\n")
            # truncate the last token
            tokens = tokens[:-1]
            masks = masks[:-1]

    return {
        "tokens": tokens,
        "masks": masks,
    } | ({"messages": chat} if include_messages else {})


if __name__ == "__main__":

    # Inspired by `preprocess --debug`` of https://github.com/axolotl-ai-cloud/axolotl
    def decode_token_ids(
        token_ids: list, mask, tokenizer, use_rich: bool = False
    ) -> str:
        """Convert a list of token IDs to a formatted string using tokenizer.decode,
        with an option to highlight masked tokens in red using rich markup.

        Each token is represented as decoded(tokenid, mask). If decoding a token returns an empty string
        and the token is a known special token, it is replaced with a descriptive placeholder.
        When use_rich is True, any token whose corresponding mask is -100 is wrapped with red highlighting.

        Args:
            token_ids (list[int]): A list of integer token IDs,
                e.g. [50256, 329].
            mask (list[int]): A list of masks corresponding to token_ids.
                A mask value of -100 indicates the token is masked.
            tokenizer: The Hugging Face tokenizer.
            use_rich (bool): If True, wrap tokens with a mask of -100 in red highlighting.
                Defaults to False.

        Returns:
            str: A space-separated string where each token is represented as decoded(tokenid, mask).

        Raises:
            ValueError: If any element in token_ids is not an integer.

        Example:
            >>> decode_token_ids([50256, 329], mask=[-100, 329], tokenizer=tokenizer, use_rich=True)
            '[red]<|eos|>(50256, -100)[/red] '
            'tokenX(329, 329)'  # (actual output will vary based on the model's tokenizer)
        """
        # Validate that all token_ids are integers.
        if not all(isinstance(t, int) for t in token_ids):
            raise ValueError("All token IDs must be integers.")

        tokens_str_list = []
        for tid, mid in zip(token_ids, mask):
            # Use decode with flags to include special tokens.
            decoded = tokenizer.decode(
                [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ).strip()
            # If the decoded string is empty and it's a special token, replace with a placeholder.
            if not decoded and tid in tokenizer.all_special_ids:
                if tid == tokenizer.eos_token_id:
                    decoded = "<|eos|>"
                else:
                    decoded = f"<SPECIAL_{tid}>"

            # Highlight token in red if use_rich is True and the token is masked (mid == -100)
            if use_rich:
                if mid == -100:
                    token_str = f"[pink3][bold]{decoded}[/bold][/pink3][steel_blue]({tid}, {mid})[/steel_blue]"
                else:
                    token_str = (
                        f"[pale_green3][bold]{decoded}[/bold][/pale_green3]"
                        f"[steel_blue]({tid}, {mid})[/steel_blue]"
                    )
            else:
                token_str = f"{decoded}({tid}, {mid})"

            tokens_str_list.append(token_str)

        return " ".join(tokens_str_list)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that provides accurate information.",
        },
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Can you tell me more about Paris?"},
        {
            "role": "assistant",
            "content": "<tool_call>{'tool_name': 'web_search', 'args': {'query': 'Paris'}}</tool_call>",
        },
        {
            "role": "tool",
            "content": (
                "Paris is the capital and most populous city of France. "
                "It has an estimated population of 2,165,423 residents in 2019 "
                "in an area of more than 105 km²."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Paris is indeed the capital of France and its most populous city with over 2 million residents. "
                "It's known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. "
                "The city is a global center for art, fashion, gastronomy, and culture."
            ),
        },
    ]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    last_turn_only = tokenize_for_trainer(
        tokenizer, messages, train_on_all_assistant_turns=False
    )
    last_turn_only["repr"] = decode_token_ids(
        last_turn_only["tokens"], last_turn_only["masks"], tokenizer, use_rich=True
    )
    all_assistant_turns = tokenize_for_trainer(
        tokenizer, messages, train_on_all_assistant_turns=True
    )
    all_assistant_turns["repr"] = decode_token_ids(
        all_assistant_turns["tokens"],
        all_assistant_turns["masks"],
        tokenizer,
        use_rich=True,
    )

    from rich import print

    print("[bold cyan]last turn only[/]")
    print(last_turn_only["repr"])
    print()
    print("[bold cyan]all assistant turns[/]")
    print(all_assistant_turns["repr"])
