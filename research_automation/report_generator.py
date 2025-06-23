# research_automation/report_generator.py

import asyncio  # For main test block example
import datetime
import json

# from ..backend_services.mcp_manager import MCPManager # Adjust import for actual structure
# For GDocs: from googleapiclient.discovery import build; from google.oauth2.service_account import Credentials
# For Slack: from slack_sdk import WebClient
# from google.cloud import bigquery # Used by BQ MCP, not directly here
import logging
import uuid  # Added for generating unique IDs
from pathlib import Path  # Added for loading config in main
from typing import Any, Dict, List, Optional

import pandas as pd  # For mock BQ data in test

# Ensure MCPManager can be imported correctly
try:
    from backend_services.mcp_manager import MCPManager
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from backend_services.mcp_manager import MCPManager


class ReportGenerator:
    """
    Creates reports based on analysis results and shares them through
    various channels like Google Docs, Slack, etc.
    """

    def __init__(self, mcp_manager, config=None):
        """
        Initialize the ReportGenerator.

        Args:
            mcp_manager: An instance of MCPManager
            config (dict, optional): Configuration for the ReportGenerator
        """
        self.mcp_manager = mcp_manager
        self.config = config or {}
        # self.mock_mode = self.config.get("mock_mode", False) # Mock mode is now per-MCP-server

        self.default_gdocs_mcp = self.config.get(
            "default_gdocs_mcp_server_name", "gdocs_main"
        )  # Assumes a gdocs_main server
        self.default_slack_mcp = self.config.get(
            "default_slack_mcp_server_name", "slack_main"
        )  # Assumes a slack_main server
        self.default_llm_mcp = self.config.get(
            "default_llm_mcp_server_name", "gemini_pro_main"
        )  # For formatting
        self.default_data_source_mcp_map = {
            "bigquery": self.config.get("default_bigquery_mcp_server_name", "bq_main"),
            "firestore": self.config.get(
                "default_firestore_mcp_server_name", "firestore_main"
            ),
            # Add other data source types and their default MCPs here
        }
        logging.info(
            f"ReportGenerator initialized. GDocs: {self.default_gdocs_mcp}, Slack: {self.default_slack_mcp}, LLM Formatter: {self.default_llm_mcp}"
        )

    async def generate_report(self, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a report based on the provided configuration.

        Args:
            report_config: Configuration for the report

        Returns:
            Dict containing report generation results
        """
        report_name = report_config.get("name", "Unnamed Report")
        logging.info(f"Generating report: {report_name}")

        # Top-level mock_mode removed; behavior depends on underlying MCP servers.

        try:
            report_data_collection = await self._fetch_report_data(report_config)
            if not report_data_collection or report_data_collection.get("error"):
                logging.error(
                    "Failed to fetch report data for "{report_name}': {report_data_collection.get('error')}"
                )
                return {
                    "status": "error",
                    "report_name": report_name,
                    "error": f"Failed to fetch report data: {report_data_collection.get('error', 'Unknown data fetching error')}",
                }

            report_content_parts = await self._format_report_content(
                report_data_collection.get("data", {}), report_config
            )
            if not report_content_parts or report_content_parts.get("error"):
                logging.error(
                    "Failed to format report content for "{report_name}': {report_content_parts.get('error')}"
                )
                return {
                    "status": "error",
                    "report_name": report_name,
                    "error": f"Failed to format report content: {report_content_parts.get('error', 'Unknown formatting error')}",
                }

            gdoc_info = {}
            if report_config.get(
                "output_gdoc", True
            ):  # Default to creating GDoc unless specified false
                gdoc_info = await self._create_or_update_gdoc(
                    report_content_parts.get("formatted_content", {}), report_config
                )
                if not gdoc_info or gdoc_info.get("error"):
                    logging.error(
                        "Failed to create/update GDoc for "{report_name}': {gdoc_info.get('error')}"
                    )
                    # Non-fatal, can still send notifications if content exists

            notification_info = {}
            if report_config.get(
                "send_notifications", True
            ):  # Default to sending notifications
                notification_info = await self._send_notifications(
                    gdoc_info,
                    report_content_parts.get("formatted_content", {}),
                    report_config,
                )
                if not notification_info or notification_info.get("error"):
                    logging.error(
                        "Failed to send notifications for "{report_name}': {notification_info.get('error')}"
                    )
                    # Non-fatal

            return {
                "status": "success",
                "report_name": report_name,
                "message": "Report "{report_name}' generated.",
                "gdoc_info": gdoc_info,
                "notification_info": notification_info,
                "data_summary": report_data_collection.get(
                    "summary", "No data summary."
                ),
                "content_summary": report_content_parts.get(
                    "summary", "No content summary."
                ),
            }
        except Exception as e:
            logging.exception(
                "Critical error in generate_report for "{report_name}': {e}"
            )
            return {"status": "error", "report_name": report_name, "error": str(e)}

    async def _fetch_report_data(self, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data for the report from specified sources via MCP."""
        report_name = report_config.get("name", "Unnamed Report")
        data_sources_config = report_config.get("data_sources", [])

        if not data_sources_config:
            logging.warning(
                f"No data sources specified for report: {report_name}. Report will be based on static content or template only."
            )
            return {"data": {}, "summary": "No dynamic data sources specified."}

        logging.info(
            f"Fetching data for report: {report_name} from {len(data_sources_config)} sources"
        )

        aggregated_data = {}
        source_summaries = []

        for i, source_conf in enumerate(data_sources_config):
            source_name = source_conf.get("name", f"source_{i}")
            source_type = source_conf.get("type")
            mcp_server_name = source_conf.get(
                "mcp_server_name"
            )  # Specific MCP server for this source
            tool_name = source_conf.get("tool_name")
            parameters = source_conf.get("parameters", {})

            if not source_type:
                logging.warning(
                    "Skipping data source "{source_name}' due to missing 'type'."
                )
                source_summaries.append(f"{source_name}: skipped (missing type)")
                continue

            # Determine default MCP server if not specified
            if not mcp_server_name:
                mcp_server_name = self.default_data_source_mcp_map.get(source_type)

            if not mcp_server_name:
                logging.error(
                    "Cannot determine MCP server for source "{source_name}' of type '{source_type}'. Skipping."
                )
                source_summaries.append(
                    f"{source_name}: skipped (no MCP server for type '{source_type}')"
                )
                aggregated_data[source_name] = {
                    "error": "No MCP server configured for type "{source_type}'"
                }
                continue

            if not tool_name:
                logging.error(
                    "Skipping data source "{source_name}': 'tool_name' is required for MCP interaction."
                )
                source_summaries.append(f"{source_name}: skipped (missing tool_name)")
                aggregated_data[source_name] = {"error": "Missing tool_name"}
                continue

            logging.info(
                "  Fetching data from "{source_name}' (type: {source_type}) using {mcp_server_name}.{tool_name}"
            )
            try:
                data = await self.mcp_manager.call_tool(
                    server_name=mcp_server_name,
                    tool_name=tool_name,
                    parameters=parameters,
                )
                if isinstance(data, dict) and data.get("error"):
                    logging.error(
                        "Error fetching data from "{source_name}': {data.get('error')}"
                    )
                    aggregated_data[source_name] = data  # Store error response
                    source_summaries.append(
                        f"{source_name}: error ({data.get('error')})"
                    )
                else:
                    aggregated_data[source_name] = data
                    source_summaries.append(f"{source_name}: success ({type(data)})")
            except Exception as e:
                logging.exception("Exception fetching data from "{source_name}': {e}")
                aggregated_data[source_name] = {"error": str(e)}
                source_summaries.append(f"{source_name}: exception ({str(e)})")

        return {
            "data": aggregated_data,
            "summary": f"Fetched data from {len(data_sources_config)} sources. Status: {source_summaries}",
        }

    async def _format_report_content(
        self, report_data: Dict[str, Any], report_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the report data into structured content, potentially using an LLM via MCP."""
        report_name = report_config.get("name", "Unnamed Report")
        template_id = report_config.get(
            "template_id", "default_summary"
        )  # e.g., "detailed_analysis_v1", "weekly_brie"
        static_content = report_config.get(
            "static_content", {}
        )  # Predefined sections, text
        use_llm_formatter = report_config.get("use_llm_formatter", True)

        logging.info(
            "Formatting report content for: "{report_name}' using template/strategy: '{template_id}'. LLM Formatter: {use_llm_formatter}"
        )

        if not report_data and not static_content:
            return {"error": "No data or static content provided for formatting."}

        # For this version, we'll assume an LLM is primarily used for formatting dynamic data.
        # A more complex version could have specific template rendering logic.
        if use_llm_formatter and report_data:
            prompt = """
            You are a report formatting assistant. Based on the following data and configuration, generate the content for a report named '{report_name}'.
            Report Template/Goal: {template_id}
            Data collected (by source name):
            {json.dumps(report_data, indent=2, default=str)}

            Static content to incorporate (if any):
            {json.dumps(static_content, indent=2, default=str)}

            Please structure your output as a JSON object with a main "title" and a list of "sections". Each section should have a "heading" and "content" (which can be text or structured data like lists/tables if appropriate for the template goal).
            Example JSON output format:
            {{
                "title": "Formatted Report Title",
                "summary_introduction": "A brief intro generated from the data.",
                "sections": [
                    {{ "heading": "Section 1 Heading", "content": "Formatted text for section 1..." }},
                    {{ "heading": "Data Highlights", "content": {{ "key_metric_1": "value", "key_finding": "description" }} }}
                ],
                "conclusion": "A concluding remark."
            }}
            Adapt the output based on the template goal '{template_id}'. For example, if it's a 'weekly_brie", keep it concise.
            If errors are present in the input data, mention them appropriately.
            """
            try:
                llm_response = await self.mcp_manager.call_tool(
                    server_name=self.default_llm_mcp,
                    tool_name="generate_text",
                    parameters={
                        "prompt": prompt,
                        "max_tokens": 2048,
                    },  # Adjust as needed
                )

                if llm_response and not (
                    isinstance(llm_response, dict) and llm_response.get("error")
                ):
                    content_str = (
                        llm_response.get("text")
                        if isinstance(llm_response, dict)
                        else str(llm_response)
                    )
                    if content_str.strip().startswith("```json"):
                        content_str = content_str.strip()[7:-3].strip()
                    elif content_str.strip().startswith("```"):
                        content_str = content_str.strip()[3:-3].strip()
                    try:
                        formatted_content = json.loads(content_str)
                        return {
                            "formatted_content": formatted_content,
                            "summary": f"Content formatted using LLM ({self.default_llm_mcp}).",
                        }
                    except json.JSONDecodeError as jde:
                        logging.error(
                            "LLM formatter returned invalid JSON for "{report_name}': {jde}. Raw: {content_str[:500]}"
                        )
                        # Fallback: use raw LLM output as a single content block
                        return {
                            "formatted_content": {
                                "title": report_name,
                                "sections": [
                                    {
                                        "heading": "LLM Output (Raw)",
                                        "content": content_str,
                                    }
                                ],
                            },
                            "summary": "LLM content formatting failed (JSON parse error), used raw output.",
                            "warning": "LLM response was not valid JSON.",
                        }
                else:
                    err_msg = (
                        llm_response.get("error", "Unknown LLM error")
                        if isinstance(llm_response, dict)
                        else "Unknown LLM error"
                    )
                    logging.error(
                        "Error from LLM formatter for "{report_name}': {err_msg}"
                    )
                    return {"error": f"LLM formatter MCP error: {err_msg}"}
            except Exception as e:
                logging.exception(
                    "Exception using LLM formatter for "{report_name}': {e}"
                )
                return {"error": f"LLM formatter exception: {str(e)}"}
        else:
            # Fallback if LLM formatter is not used or no dynamic data to format
            # Just use static content or a very basic structure
            logging.info(
                "Using basic formatting (no LLM or no dynamic data) for "{report_name}'."
            )
            return {
                "formatted_content": {
                    "title": static_content.get("title", report_name),
                    "sections": static_content.get(
                        "sections",
                        [
                            {
                                "heading": "Data Overview",
                                "content": (
                                    json.dumps(report_data, indent=2, default=str)
                                    if report_data
                                    else "No dynamic data available."
                                ),
                            }
                        ],
                    ),
                },
                "summary": "Content formatted using basic/static structure.",
            }

    async def _create_or_update_gdoc(
        self, report_content: Dict[str, Any], report_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create or update a Google Doc with the report content via MCP."""
        report_name = report_config.get("name", "Unnamed Report")
        gdoc_title = report_content.get(
            "title", report_name
        )  # Use title from formatted content
        gdoc_template_id = report_config.get(
            "gdoc_template_id"
        )  # Optional: for using a GDoc template
        target_folder_id = report_config.get(
            "gdoc_folder_id"
        )  # Optional: to save in specific GDrive folder

        logging.info(
            "Creating/updating Google Doc for report: "{report_name}' with title '{gdoc_title}'."
        )

        if not self.default_gdocs_mcp:
            return {"error": "Google Docs MCP server name not configured."}

        # For a real GDocs MCP, you'd translate `report_content` (title, sections)
        # into the calls needed to build the document (e.g., batchUpdate requests).
        # The mock GDocs server will just simulate this.
        parameters = {
            "title": gdoc_title,
            "body_content": report_content,  # Mock server might just take the whole dict
            "template_id": gdoc_template_id,
            "folder_id": target_folder_id,
        }
        tool_to_call = (
            "create_doc_from_content"  # Assuming a tool that takes structured content
        )
        if report_config.get("existing_gdoc_id"):  # Logic for updating existing doc
            parameters["document_id"] = report_config["existing_gdoc_id"]
            tool_to_call = "update_doc_content"

        try:
            gdoc_response = await self.mcp_manager.call_tool(
                server_name=self.default_gdocs_mcp,
                tool_name=tool_to_call,
                parameters=parameters,
            )

            if (
                gdoc_response
                and gdoc_response.get("status", "").startswith("success")
                and gdoc_response.get("document_id")
            ):
                logging.info(
                    "Successfully created/updated GDoc for "{report_name}': ID {gdoc_response.get('document_id')}"
                )
                return {
                    "status": "success",
                    "doc_id": gdoc_response.get("document_id"),
                    "doc_url": gdoc_response.get(
                        "url",
                        f"https://docs.google.com/document/d/{gdoc_response.get('document_id')}/edit",
                    ),
                    "operation": (
                        "created"
                        if tool_to_call == "create_doc_from_content"
                        else "updated"
                    ),
                }
            else:
                error_msg = (
                    gdoc_response.get("error", "Unknown GDocs MCP error")
                    if isinstance(gdoc_response, dict)
                    else "Unknown GDocs MCP error"
                )
                logging.error("Error from GDocs MCP for "{report_name}': {error_msg}")
                return {"error": f"GDocs MCP error: {error_msg}"}
        except Exception as e:
            logging.exception(
                "Exception creating/updating GDoc for "{report_name}': {e}"
            )
            return {"error": f"GDocs MCP exception: {str(e)}"}

    async def _send_notifications(
        self,
        gdoc_info: Dict[str, Any],
        report_content: Dict[str, Any],
        report_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send notifications about the report via MCP (e.g., Slack)."""
        report_name = report_config.get("name", "Unnamed Report")
        doc_url = gdoc_info.get("doc_url", "N/A")
        report_title = report_content.get("title", report_name)

        notification_channels = report_config.get(
            "notification_channels", ["slack"]
        )  # Default to slack
        slack_channel_id = report_config.get(
            "slack_channel_id", "#general"
        )  # Example channel

        if not notification_channels:
            logging.info(
                "No notification channels configured for report "{report_name}'. Skipping notifications."
            )
            return {
                "status": "skipped",
                "message": "No notification channels configured.",
            }

        logging.info(
            "Sending notifications for report: "{report_name}' via {notification_channels}"
        )

        results = {}

        if "slack" in notification_channels and self.default_slack_mcp:
            # Construct a concise summary for Slack
            summary_points = []
            if report_content.get("summary_introduction"):
                summary_points.append(report_content["summary_introduction"])
            for section in report_content.get("sections", [])[:2]:  # First 2 sections
                heading = section.get("heading")
                content_preview = (
                    str(section.get("content"))[:100] + "..."
                    if len(str(section.get("content"))) > 100
                    else str(section.get("content"))
                )
                if heading:
                    summary_points.append(f"*{heading}*: {content_preview}")

            message_text = (
                f"New Report Generated: *{report_title}*\n"
                f"GDoc Link: {doc_url}\n\n"
                "Summary:\n" + "\n".join([f"- {s}" for s in summary_points])
            )
            if not summary_points:
                message_text = (
                    f"New Report Generated: *{report_title}*\nLink: {doc_url}"
                )

            try:
                slack_response = await self.mcp_manager.call_tool(
                    server_name=self.default_slack_mcp,
                    tool_name="send_message",
                    parameters={
                        "channel_id": slack_channel_id,
                        "text_content": message_text,
                    },
                )
                if slack_response and slack_response.get("status", "").startswith(
                    "success"
                ):
                    logging.info(
                        "Successfully sent Slack notification for "{report_name}' to {slack_channel_id}"
                    )
                    results["slack"] = {
                        "status": "success",
                        "channel": slack_channel_id,
                        "ts": slack_response.get("ts"),
                    }
                else:
                    err_msg = (
                        slack_response.get("error", "Unknown Slack MCP error")
                        if isinstance(slack_response, dict)
                        else "Unknown Slack MCP error"
                    )
                    logging.error(
                        "Error from Slack MCP for "{report_name}': {err_msg}"
                    )
                    results["slack"] = {"status": "error", "error": err_msg}
            except Exception as e:
                logging.exception(
                    "Exception sending Slack notification for "{report_name}': {e}"
                )
                results["slack"] = {
                    "status": "error",
                    "error": f"Slack MCP exception: {str(e)}",
                }
        elif "slack" in notification_channels:
            results["slack"] = {
                "status": "skipped",
                "error": "Slack MCP server not configured.",
            }
            logging.warning(
                "Slack notification requested but Slack MCP server not configured."
            )

        # Add other notification channels (e.g., email) similarly
        # ...

        return results if results else {"status": "no_channels_processed"}


async def run_rg_test_with_mcp():
    print("\nTesting ReportGenerator with live MCPManager (async)..")

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "pipeline_config.json"

    if not config_path.exists():
        print(f"ERROR: pipeline_config.json not found at {config_path}")
        return

    with open(config_path, "r") as f:
        pipeline_config = json.load(f)
    print(f"Loaded pipeline config from {config_path}")
    # Log mock status of relevant MCPs
    for mcp_name in [
        "gdocs_main",
        "slack_main",
        "bq_main",
        "firestore_main",
        "gemini_pro_main",
    ]:
        mock_status = (
            pipeline_config.get("mcp_server_configurations", {})
            .get(mcp_name, {})
            .get("config", {})
            .get("mock_mode")
        )
        print(f"  {mcp_name} mock_mode: {mock_status}")

    mcp_manager = MCPManager(config=pipeline_config)

    rg_config_from_pipeline = pipeline_config.get("report_generator_config", {})
    # Augment rg_config with default MCP names if not present, for the test
    rg_config_from_pipeline.setdefault("default_gdocs_mcp_server_name", "gdocs_main")
    rg_config_from_pipeline.setdefault("default_slack_mcp_server_name", "slack_main")
    rg_config_from_pipeline.setdefault(
        "default_llm_mcp_server_name", "gemini_pro_main"
    )  # Or claude_main
    rg_config_from_pipeline.setdefault("default_bigquery_mcp_server_name", "bq_main")
    rg_config_from_pipeline.setdefault(
        "default_firestore_mcp_server_name", "firestore_main"
    )

    reporter = ReportGenerator(mcp_manager=mcp_manager, config=rg_config_from_pipeline)

    # Test 1: Report from BigQuery data, output to GDoc and Slack
    report_task1 = {
        "name": "Weekly Performance Overview (via MCP)",
        "data_sources": [
            {
                "name": "weekly_bq_summary",
                "type": "bigquery",  # used by RG to find default_bigquery_mcp
                # "mcp_server_name": "bq_main", # Can be omitted if default is fine
                "tool_name": "run_query",
                "parameters": {
                    "query": "SELECT status, COUNT(*) as count FROM mock_dataset.trial_results WHERE week = CURRENT_WEEK() GROUP BY status"
                },
            }
        ],
        "use_llm_formatter": True,
        "template_id": "concise_weekly_summary",
        "output_gdoc": True,
        "send_notifications": True,
        "slack_channel_id": "#test-reports-mcp",
    }
    print(f"\nGenerating Report Task 1: {report_task1['name']}")
    report_output1 = await reporter.generate_report(report_task1)
    print(f"Report Task 1 Output:\n{json.dumps(report_output1, indent=2, default=str)}")

    # Test 2: Report from Firestore data, LLM formatted, Slack only, no GDoc
    report_task2 = {
        "name": "Specific Experiment Insight (via MCP)",
        "data_sources": [
            {
                "name": "firestore_exp_insight",
                "type": "firestore",
                "tool_name": "get_document",
                "parameters": {
                    "collection_id": "assistant_outputs_test",
                    "document_id": "exp_001_completed_insights_example",
                },
                # ^ This doc_id would need to exist in Firestore mock data for FS MCP to return something
            }
        ],
        "use_llm_formatter": True,
        "template_id": "single_insight_focus",
        "output_gdoc": False,  # Do not create GDoc
        "send_notifications": True,
        "slack_channel_id": "#test-alerts-mcp",
    }
    print(f"\nGenerating Report Task 2: {report_task2['name']}")
    report_output2 = await reporter.generate_report(report_task2)
    print(f"Report Task 2 Output:\n{json.dumps(report_output2, indent=2, default=str)}")

    # Test 3: Report with no dynamic data (static content), GDoc only
    report_task3 = {
        "name": "Static Announcement Document (via MCP)",
        "static_content": {
            "title": "Important Announcement: Q3 Planning",
            "sections": [
                {
                    "heading": "Meeting Schedule",
                    "content": "Q3 planning meetings will be held next week...",
                },
                {
                    "heading": "Key Topics",
                    "content": "- Review Q2 results\n- Define Q3 OKRs\n- Resource allocation",
                },
            ],
        },
        "use_llm_formatter": False,  # Use static content directly
        "output_gdoc": True,
        "gdoc_folder_id": "mock_folder_id_for_announcements",
        "send_notifications": False,
    }
    print(f"\nGenerating Report Task 3: {report_task3['name']}")
    report_output3 = await reporter.generate_report(report_task3)
    print(f"Report Task 3 Output:\n{json.dumps(report_output3, indent=2, default=str)}")

    # Clean up MCP client resources
    for server_instance in mcp_manager.mcp_servers.values():
        if hasattr(server_instance, "close_client") and asyncio.iscoroutinefunction(
            server_instance.close_client
        ):
            await server_instance.close_client()

    print("\nReportGenerator with MCPManager (async) conceptual test complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )
    asyncio.run(run_rg_test_with_mcp())
