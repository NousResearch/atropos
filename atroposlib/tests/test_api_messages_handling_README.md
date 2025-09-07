# API Messages Handling Tests

This test suite validates the API server's handling of messages in various formats, particularly for SFT (Supervised Fine-Tuning) scenarios.

## Test Coverage

### Basic API Functionality
- **test_register_trainer**: Tests trainer registration with the API server
- **test_scored_data_with_messages**: Tests posting scored data with OpenAI-format messages
- **test_scored_data_list_with_messages**: Tests batch posting of multiple scored data items
- **test_empty_messages_handling**: Tests handling of optional/empty messages field

### Message Format Tests
- **test_sft_style_messages**: Tests ShareGPT format messages with SFT overrides
- **test_multimodal_messages_with_images**: Tests multimodal messages with image content
- **test_complex_message_structures**: Tests messages with tool role interactions
- **test_message_reward_field**: Tests messages with reward fields

### Data Retrieval Tests
- **test_batch_retrieval_with_messages**: Tests retrieving batches containing messages
- **test_latest_example_with_messages**: Tests the latest example endpoint preserves messages

### SFT Integration Tests
- **test_sft_completion_format**: Tests simple completion format (without messages)
- **test_sft_prefixed_completion**: Tests prefixed completion with masked tokens
- **test_sft_batch_processing**: Tests batch processing of SFT data

## Key Findings

1. **Message Type Requirements**: The API expects messages in the format `List[List[Message]]` where `Message` is a TypedDict with required fields:
   - `role`: Literal["system", "user", "assistant", "tool"]
   - `content`: str or list of content parts
   - `reward`: Optional[float] (but must be present, can be None)

2. **SFT Format Handling**: For completion-style SFT data (raw text without conversation structure), the messages field should be omitted rather than trying to pass strings.

3. **Advantages Field**: Must be a list of lists matching the token structure, not a single value.

## Running the Tests

```bash
# Run all message handling tests
python -m pytest atroposlib/tests/test_api_messages_handling.py -v

# Run a specific test
python -m pytest atroposlib/tests/test_api_messages_handling.py::TestAPIMessagesHandling::test_scored_data_with_messages -v

# Run with output for debugging
python -m pytest atroposlib/tests/test_api_messages_handling.py -v -s
```

## Test Infrastructure

The tests use:
- A fixture to launch the API server as a subprocess
- Automatic cleanup and state reset between tests
- Proper process group handling to ensure all child processes are terminated

## Future Considerations

The current API type definition for messages (`List[List[Message]]`) doesn't fully align with how the SFT loader sends data for completion formats (plain strings). This test suite works around this by omitting the messages field for completion-style data, but a future improvement might be to make the API more flexible with Union types.
