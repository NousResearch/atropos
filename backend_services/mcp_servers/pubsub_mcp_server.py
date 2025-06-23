# backend_services/mcp_servers/pubsub_mcp_server.py

import json
import os
from typing import Any, Dict, List, Optional  # For type hinting

from google.api_core.exceptions import NotFound  # Explicit import for NotFound
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.publisher.client import (
    Client as PublisherClient,  # For type hinting
)

from .base_mcp_server import BaseMCPServer


class PubSubMCPServer(BaseMCPServer):
    """
    MCP Server implementation for interacting with Google Cloud Pub/Sub.
    Currently focuses on publishing messages.
    """

    def __init__(self, server_name: str, config: dict):
        """
        Initializes the PubSubMCPServer.
        Expected config keys:
            - "project_id": Your Google Cloud Project ID.
        """
        self.publisher_client: Optional[PublisherClient] = None
        # self.subscriber_client = None # If supporting subscriptions directly in the future
        super().__init__(server_name, config)

    def _validate_config(self):
        """Validates that 'project_id' is in the config."""
        if "project_id" not in self.config:
            raise ValueError(
                "Missing "project_id' in config for PubSubMCPServer: {self.server_name}"
            )
        self.project_id = self.config["project_id"]

    def _initialize_client(self):
        """Initializes the Pub/Sub Publisher client."""
        try:
            self.publisher_client = pubsub_v1.PublisherClient()
            print(
                "Pub/Sub Publisher client initialized for project "{self.project_id}' for server '{self.server_name}'."
            )
            # Test connectivity by trying to list topics (lightweight call, requires list permission)
            # This helps confirm that authentication is working.
            topics = list(
                self.publisher_client.list_topics(
                    project=f"projects/{self.project_id}", page_size=1
                )
            )
            if topics:
                print(
                    f"  Successfully listed at least one Pub/Sub topic: {topics[0].name}"
                )
            else:
                print(
                    "  No Pub/Sub topics found in project or not permitted to list, but client initialized."
                )
        except Exception as e:
            print(
                f"CRITICAL: Error initializing Pub/Sub Publisher client for {self.server_name}: {e}"
            )
            self.publisher_client = None  # Ensure client is None if init fails
            # raise # Uncomment to make Pub/Sub client initialization critical

    def _publish_message(
        self,
        topic_id: str,
        message_data: Dict[str, Any],
        attributes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Publishes a message to a Pub/Sub topic.
        `message_data` should be a JSON-serializable dictionary.
        `attributes` should be a dictionary of string key-value pairs.
        Returns a status dictionary including the message_id on success.
        """
        if not self.publisher_client:
            return {
                "status": "failure",
                "error": f"Pub/Sub Publisher client not initialized for {self.server_name}.",
            }

        topic_path = self.publisher_client.topic_path(self.project_id, topic_id)
        print(f"  {self.server_name}._publish_message called for topic '{topic_path}'.")

        try:
            data_bytes = json.dumps(message_data).encode("utf-8")
        except TypeError as e:
            print(f"Error encoding message data to JSON for Pub/Sub: {e}")
            return {
                "status": "failure",
                "error": "Message data not JSON serializable",
                "details": str(e),
            }

        print(
            f"    Message Data (first 100 bytes as string): {data_bytes.decode('utf-8', errors='ignore')[:100]}..."
        )
        str_attributes = (
            {k: str(v) for k, v in attributes.items()} if attributes else None
        )
        if str_attributes:
            print(f"    Attributes: {str_attributes}")

        try:
            future = self.publisher_client.publish(
                topic_path, data_bytes, **(str_attributes if str_attributes else {})
            )
            message_id = future.result()  # Wait for publish to complete (blocks)
            print(
                f"    Message published successfully to {topic_path}. Message ID: {message_id}"
            )
            return {
                "status": "success",
                "topic_path": topic_path,
                "message_id": message_id,
            }
        except NotFound:
            error_msg = "Pub/Sub topic "{topic_path}' not found."
            print(f"Error for {self.server_name}: {error_msg}")
            return {"status": "failure", "error": error_msg, "topic_path": topic_path}
        except Exception as e:
            error_msg = f"Publishing to {topic_path} failed for {self.server_name}."
            print(f"Error: {error_msg} Details: {e}")
            return {"status": "failure", "error": error_msg, "details": str(e)}

    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Executes a tool supported by the PubSubMCPServer.
        Supported tools:
            - "publish_message": Publishes a message to a topic. Returns status dict.
        """
        if tool_name == "publish_message":
            topic_id = parameters.get("topic_id")
            message_data = parameters.get("message_data")  # Should be a dict

            if not topic_id:
                raise ValueError(
                    "Missing "topic_id' for 'publish_message' tool in {self.server_name}."
                )
            if not isinstance(message_data, dict):
                raise ValueError(
                    ""message_data' must be a dictionary for 'publish_message' tool in {self.server_name}."
                )

            return self._publish_message(
                topic_id=topic_id,
                message_data=message_data,
                attributes=parameters.get("attributes"),
            )
        else:
            raise NotImplementedError(
                "Tool "{tool_name}' is not supported by {self.server_name} ({self.__class__.__name__})."
            )


# Example Usage (conceptual)
if __name__ == "__main__":
    print("Testing PubSubMCPServer with google-cloud-pubsub SDK...")
    # THIS TEST WILL FAIL IF YOU DON'T HAVE GCP AUTHENTICATION SET UP
    # AND A VALID PROJECT ID + AN EXISTING TOPIC FOR TESTING.
    gcp_project_id_env = os.environ.get("GOOGLE_CLOUD_PROJECT")
    test_topic_id = "mcp_pubsub_test_topic"  # IMPORTANT: This topic must exist in your project for the test to fully succeed.

    if not gcp_project_id_env:
        print(
            "SKIPPING real PubSubMCPServer test: GOOGLE_CLOUD_PROJECT env var not set."
        )
    else:
        print(f"Using GCP Project: {gcp_project_id_env} for Pub/Sub tests.")
        sample_ps_config_real = {"project_id": gcp_project_id_env}
        ps_server_real = PubSubMCPServer(
            server_name="pubsub_sdk_test", config=sample_ps_config_real
        )

        if ps_server_real.publisher_client:
            print(
                f"Attempting to publish to topic: {test_topic_id} (ensure it exists or auto-create is enabled if supported by test setup)"
            )
            publish_params = {
                "topic_id": test_topic_id,
                "message_data": {
                    "experiment_id": "exp_sdk_test_123",
                    "status": "completed",
                    "score": 0.99,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                },
                "attributes": {
                    "source_system": "mcp_server_test",
                    "priority": "medium",
                    "attempt_num": "1",
                },
            }
            print(
                "\nSimulating publish_message to topic "{publish_params['topic_id']}'"
            )
            publish_result = ps_server_real.call_tool("publish_message", publish_params)
            print(
                f"Publish Result (real client):\n{json.dumps(publish_result, indent=2)}"
            )
            assert publish_result.get(
                "status"
            ) == "success" or "Topic not found" in publish_result.get(
                "error", ""
            )  # Topic might not exist for test

            # Test with non-serializable data (should fail gracefully in _publish_message)
            print(
                "\n--- Test: Publish Non-Serializable Data (should fail gracefully) ---"
            )
            bad_data_params = {
                "topic_id": test_topic_id,
                "message_data": {set([1, 2, 3])},  # Set is not JSON serializable
            }
            bad_data_result = ps_server_real.call_tool(
                "publish_message", bad_data_params
            )
            print(
                f"Publish Result (bad data):\n{json.dumps(bad_data_result, indent=2)}"
            )
            assert bad_data_result.get(
                "status"
            ) == "failure" and "not JSON serializable" in bad_data_result.get(
                "error", ""
            )

            # Test publish to a non-existent topic (should fail gracefully if topic doesn't exist)
            print("\n--- Test: Publish to Non-Existent Topic ---")
            non_existent_topic_params = {
                "topic_id": "this_topic_should_not_exist_hopefully_123xyz",
                "message_data": {"test": "data"},
            }
            nx_topic_result = ps_server_real.call_tool(
                "publish_message", non_existent_topic_params
            )
            print(
                f"Publish Result (non-existent topic):\n{json.dumps(nx_topic_result, indent=2)}"
            )
            assert nx_topic_result.get(
                "status"
            ) == "failure" and "Topic not found" in nx_topic_result.get("error", "")

        else:
            print("Skipping real Pub/Sub client tests as client did not initialize.")

    print("\nPubSubMCPServer test complete.")

# Need for test data with timestamp
import datetime
from typing import Any, Dict, List, Optional  # For type hinting
