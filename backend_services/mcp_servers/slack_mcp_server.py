# backend_services/mcp_servers/slack_mcp_server.py

import os
import json # For example usage block
from .base_mcp_server import BaseMCPServer
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import Optional, Dict, List, Any # For type hinting

class SlackMCPServer(BaseMCPServer):
    """
    MCP Server implementation for interacting with Slack API using slack_sdk.
    """

    def __init__(self, server_name: str, config: dict):
        """
        Initializes the SlackMCPServer.
        Expected config keys:
            - "bot_token" (optional if SLACK_BOT_TOKEN env var set): Your Slack Bot User OAuth Token (starts with xoxb-).
            - "default_channel" (optional): Default Slack channel ID or name to post to.
        """
        self.client: Optional[WebClient] = None
        self.configured_bot_token_source: Optional[str] = None
        super().__init__(server_name, config)

    def _validate_config(self):
        """Validates bot_token from config, ENV_ var in config, or SLACK_BOT_TOKEN env var."""
        config_bot_token = self.config.get("bot_token")
        env_var_name_in_config = None

        if isinstance(config_bot_token, str) and config_bot_token.startswith("ENV_"):
            env_var_name_in_config = config_bot_token[4:]
            print(f"  SlackMCPServer '{self.server_name}': Config requests bot token from env var '{env_var_name_in_config}'.")
            loaded_env_token = os.environ.get(env_var_name_in_config)
            if loaded_env_token:
                self.config["bot_token"] = loaded_env_token
                self.configured_bot_token_source = f"environment ({env_var_name_in_config}) via config"
            else:
                generic_env_token = os.environ.get("SLACK_BOT_TOKEN")
                if generic_env_token:
                    self.config["bot_token"] = generic_env_token
                    self.configured_bot_token_source = "environment (SLACK_BOT_TOKEN fallback)"
                    print(f"  SlackMCPServer '{self.server_name}': '{env_var_name_in_config}' not set, using SLACK_BOT_TOKEN fallback.")
                else:
                    raise ValueError(f"SlackMCPServer '{self.server_name}': Bot token '{env_var_name_in_config}' (from config) not set, and SLACK_BOT_TOKEN env var also not set.")
        elif config_bot_token: 
            self.configured_bot_token_source = "config_direct"
        elif os.environ.get("SLACK_BOT_TOKEN"):
            self.config["bot_token"] = os.environ.get("SLACK_BOT_TOKEN")
            self.configured_bot_token_source = "environment (SLACK_BOT_TOKEN)"
        else:
            raise ValueError(f"Missing 'bot_token' in config for SlackMCPServer '{self.server_name}' and SLACK_BOT_TOKEN environment variable not set.")
        
        if not self.config.get("bot_token", "").startswith("xoxb-"):
             print(f"Warning: Slack bot token for '{self.server_name}' (from {self.configured_bot_token_source or 'unknown'}) does not look like a valid xoxb- token.")

    def _initialize_client(self):
        """
        Initializes the Slack WebClient.
        (Assumes _validate_config has ensured self.config["bot_token"] exists)
        """
        bot_token = self.config.get("bot_token")
        if not bot_token: # Should have been caught by _validate_config
            print(f"CRITICAL: No bot_token available for SlackMCPServer '{self.server_name}' during client init. Client not created.")
            self.client = None
            return
        try:
            self.client = WebClient(token=bot_token)
            auth_test_response = self.client.auth_test()
            if auth_test_response.get("ok"):
                print(f"Slack WebClient initialized and auth successful for bot '{auth_test_response.get('user')}' (team: '{auth_test_response.get('team')}') on server '{self.server_name}' using token from {self.configured_bot_token_source or 'unknown source'}. (Actual SDK TBI)")
            else:
                # This case might happen if token is invalid but doesn't raise an exception on WebClient() itself.
                print(f"Warning: Slack WebClient initialized for {self.server_name}, but auth.test failed: {auth_test_response.get('error')} (token from {self.configured_bot_token_source or 'unknown source'}). Client might be unusable.")
                # Depending on strictness, we might set self.client = None here.
        except SlackApiError as e:
            print(f"CRITICAL: Error initializing Slack WebClient or auth.test for {self.server_name}: {e.response['error']}")
            self.client = None
        except Exception as e:
            print(f"CRITICAL: General error initializing Slack WebClient for {self.server_name}: {e}")
            self.client = None

    def _send_message(self, channel_id: Optional[str], text: Optional[str] = None, blocks: Optional[List[Dict[str, Any]]] = None, thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends a message to a Slack channel using slack_sdk.
        Returns a dictionary with status and response details.
        """
        if not self.client:
            return {"error": f"Slack client not initialized for {self.server_name}.", "status": "failure"}

        target_channel = channel_id or self.config.get("default_channel")
        if not target_channel:
            return {"error": "No channel_id specified and no default_channel configured for sending message.", "status": "failure"}
        
        if not text and not blocks:
            return {"error": "Either text or blocks must be provided to send a message.", "status": "failure"}

        print(f"  {self.server_name}._send_message to channel '{target_channel}'. Text: '{str(text)[:50]}...' Blocks: {len(blocks) if blocks else 0}")
        try:
            response = self.client.chat_postMessage(
                channel=target_channel,
                text=text, # Fallback text for notifications, required even if blocks are used
                blocks=blocks,
                thread_ts=thread_ts
            )
            # slack_sdk response is a SlackResponse object, access data via .data property
            response_data = response.data
            if response_data and response_data.get("ok"):
                return {
                    "status": "success",
                    "channel": response_data.get("channel"),
                    "ts": response_data.get("ts"),
                    "message_text_sent": text or "(blocks_only)" # Simplified, actual sent text might differ if blocks are primary
                }
            else:
                return {"status": "failure", "error": response_data.get("error") if response_data else "unknown_slack_error", "details": response_data}
        except SlackApiError as e:
            print(f"SlackApiError sending message to {target_channel} for {self.server_name}: {e.response['error']}")
            return {"status": "failure", "error": e.response['error'], "details": e.response.data}
        except Exception as e:
            print(f"General error sending Slack message for {self.server_name}: {type(e).__name__} - {e}")
            return {"status": "failure", "error": str(e)}

    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Executes a tool supported by the SlackMCPServer.
        Supported tools:
            - "send_message": Sends a message to a channel.
                Params: "channel_id" (Optional[str]), "text" (Optional[str]), "blocks" (Optional[List[Dict]]), "thread_ts" (Optional[str])
        """
        if tool_name == "send_message":
            channel = parameters.get("channel_id")
            text = parameters.get("text")
            blocks = parameters.get("blocks")
            
            # Ensure either channel_id is provided or a default is configured at server level
            if not channel and not self.config.get("default_channel"):
                 raise ValueError(f"Missing 'channel_id' and no default_channel set for 'send_message' tool in {self.server_name}.")
            if not text and not blocks:
                raise ValueError(f"Either 'text' or 'blocks' must be provided for 'send_message' tool in {self.server_name}.")

            return self._send_message(
                channel_id=channel,
                text=text,
                blocks=blocks,
                thread_ts=parameters.get("thread_ts")
            )
        else:
            raise NotImplementedError(f"Tool '{tool_name}' is not supported by {self.server_name} ({self.__class__.__name__}).")

# Example Usage (conceptual)
if __name__ == "__main__":
    print("Testing SlackMCPServer with slack_sdk... (Requires SLACK_BOT_TOKEN env var and a test channel)")
    test_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    test_channel_id = os.environ.get("SLACK_TEST_CHANNEL_ID") # e.g., C12345678 or name #test-channel

    if not test_bot_token or not test_channel_id:
        print("  SKIPPING SlackMCPServer real API call test: SLACK_BOT_TOKEN and/or SLACK_TEST_CHANNEL_ID environment variables not set.")
        print("    Please set them to a valid bot token and channel ID to test real Slack message posting.")
    else:
        print(f"  Using SLACK_BOT_TOKEN: {test_bot_token[:9]}... and SLACK_TEST_CHANNEL_ID: {test_channel_id}")
        sample_slack_config = {
            "bot_token": test_bot_token, # Loaded from env for test
            "default_channel": test_channel_id # Using test channel as default for this run
        }
        slack_server = None
        try:
            slack_server = SlackMCPServer(server_name="slack_sdk_test", config=sample_slack_config)

            if slack_server and slack_server.client: # Check if client initialized successfully
                print(f"\n--- Test 1: Send simple text message ---")
                msg_params_text = {
                    "channel_id": test_channel_id,
                    "text": f"Hello from SlackMCPServer test! Timestamp: {datetime.datetime.now().isoformat()}"
                }
                send_result_text = slack_server.call_tool("send_message", msg_params_text)
                print(f"Send Text Result:\n{json.dumps(send_result_text, indent=2)}")

                print(f"\n--- Test 2: Send message with Block Kit ---")
                msg_params_blocks = {
                    "channel_id": test_channel_id,
                    "text": "Fallback text for notification: Pipeline Report Available!",
                    "blocks": [
                        {"type": "header", "text": {"type": "plain_text", "text": ":wave: Pipeline Report Ready!"}},
                        {"type": "section", "text": {"type": "mrkdwn", "text": "The weekly experiment summary report is now available."}},
                        {"type": "actions", "elements": [
                            {"type": "button", "text": {"type": "plain_text", "text": "View Report (Dummy Link)"}, "url": "https://example.com/dummy_report"}
                        ]}
                    ]
                }
                send_result_blocks = slack_server.call_tool("send_message", msg_params_blocks)
                print(f"Send Blocks Result:\n{json.dumps(send_result_blocks, indent=2)}")
            else:
                print("Slack client did not initialize successfully. Skipping API call tests.")
        except ValueError as ve:
            print(f"ValueError during Slack test setup: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during SlackMCPServer test: {type(e).__name__} - {e}")
            import traceback; traceback.print_exc()
        finally:
            print("\nSlackMCPServer test attempt finished.")

# For test timestamp
import datetime

# Example Usage (conceptual)
if __name__ == "__main__":
    print("Testing SlackMCPServer...")
    # Test cases should now rely on SLACK_BOT_TOKEN env var primarily, or a direct key in test_config.
    test_token_value = os.environ.get("SLACK_BOT_TOKEN") or "xoxb-dummy-slack-token-for-testing" 
    token_source_for_test = "environment (SLACK_BOT_TOKEN)" if os.environ.get("SLACK_BOT_TOKEN") else "config_direct (dummy)" 

    if not os.environ.get("SLACK_BOT_TOKEN"):
        print(f"  WARNING: SLACK_BOT_TOKEN environment variable not set. Mock tests will use dummy token: {test_token_value}.")
    else:
        print(f"  Using SLACK_BOT_TOKEN from environment for testing.")

    sample_slack_config = {
        "bot_token": test_token_value, # Server will use this if direct, or validated from env if it was ENV_ pref.
        "default_channel": "#test-automation-alerts"
    }
    slack_server = None
    try:
        slack_server = SlackMCPServer(server_name="slack_main_test", config=sample_slack_config)

        # Test send_message with text
        msg_params_text = {
            "channel_id": "#test-specific",
            "text": "Hello from Slack MCP! This is a test notification."
        }
        print(f"\nSimulating send_message (text) to channel '{msg_params_text['channel_id']}'")
        send_result_text = slack_server.call_tool("send_message", msg_params_text)
        print(f"Send Result (text, mock):\n{json.dumps(send_result_text, indent=2)}")

        # Test send_message with blocks (and using default channel from config)
        msg_params_blocks = {
            # No channel_id, so default_channel from config should be used
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "*Test Update*"}},
                {"type": "divider"},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": "*Batch Name:*\nWeekly_Spatial_Reasoning_Run"},
                    {"type": "mrkdwn", "text": "*Status:*\n:large_green_circle: Success"}
                ]}
            ],
            "text": "Test Update Fallback" # For notifications
        }
        print(f"\nSimulating send_message (blocks) to default channel")
        send_result_blocks = slack_server.call_tool("send_message", msg_params_blocks)
        print(f"Send Result (blocks, mock to default channel):\n{json.dumps(send_result_blocks, indent=2)}")
        
        status = slack_server.get_status()
        print(f"\nServer Status: {status}")

    except ValueError as ve:
        print(f"ValueError during Slack test setup: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {type(e).__name__} - {e}")
    finally:
        print("\nSlackMCPServer test attempt finished.") 