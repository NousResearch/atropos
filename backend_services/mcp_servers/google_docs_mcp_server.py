# backend_services/mcp_servers/google_docs_mcp_server.py

import json  # For __main__ example printing
import os
from typing import Any, Dict, List, Optional  # For type hinting

from google.auth import default as get_application_default_credentials
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import Resource, build

from .base_mcp_server import BaseMCPServer


class GoogleDocsMCPServer(BaseMCPServer):
    """
    MCP Server implementation for interacting with Google Docs API & Google Drive API (for templates).
    Requires Drive API for copying templates, Docs API for content manipulation.
    Scopes needed:
        'https://www.googleapis.com/auth/documents' (for Docs API)
        'https://www.googleapis.com/auth/drive.file' (for copying templates from Drive, or general Drive access)
        'https://www.googleapis.com/auth/drive' (if broader Drive access is needed, e.g., folder creation/management)
    """

    DRIVE_API_VERSION = "v3"
    DOCS_API_VERSION = "v1"
    REQUIRED_SCOPES = [
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive.file",
        # Add 'https://www.googleapis.com/auth/drive' if more extensive Drive operations are needed beyond file copy
    ]

    def __init__(self, server_name: str, config: dict):
        """
        Initializes the GoogleDocsMCPServer.
        Expected config keys:
            - "credentials_path" (optional): Path to Google service account credentials JSON file.
                                            (Other auth methods like ADC might be used too)
            - "default_folder_id" (optional): Default Google Drive folder ID for new docs.
        """
        self.docs_service: Optional[Resource] = None
        self.drive_service: Optional[Resource] = None
        self.credentials = None
        self.auth_method_used: Optional[str] = None
        super().__init__(server_name, config)

    def _validate_config(self):
        """Validates necessary configuration for Google Docs API interaction."""
        # Credentials path is optional if ADC is intended.
        # If provided, it should exist.
        creds_path = self.config.get("credentials_path")
        if creds_path and not os.path.exists(creds_path):
            raise ValueError(
                "Invalid "credentials_path' for GoogleDocsMCPServer '{self.server_name}': File not found at {creds_path}"
            )
        print(
            "  GoogleDocsMCPServer "{self.server_name}' config validation passed. Creds path: {creds_path or 'Not specified (will try ADC)'}"
        )

    def _initialize_client(self):
        """Initializes the Google Docs API service client."""
        creds_path = self.config.get("credentials_path")
        try:
            if creds_path:
                self.credentials = ServiceAccountCredentials.from_service_account_file(
                    creds_path, scopes=self.REQUIRED_SCOPES
                )
                self.auth_method_used = f"ServiceAccount ({creds_path})"
            else:
                print(
                    "  GoogleDocsMCPServer "{self.server_name}': No credentials_path provided. Attempting Application Default Credentials (ADC)."
                )
                self.credentials, project_id = get_application_default_credentials(
                    scopes=self.REQUIRED_SCOPES
                )
                self.auth_method_used = f"ApplicationDefaultCredentials (Project: {project_id or 'Unknown'})"

            if not self.credentials or not self.credentials.valid:
                if (
                    self.credentials
                    and self.credentials.expired
                    and self.credentials.refresh_token
                ):
                    self.credentials.refresh(httpx.Request())
                else:
                    raise DefaultCredentialsError("Could not create valid credentials.")

            self.docs_service = build(
                "docs",
                self.DOCS_API_VERSION,
                credentials=self.credentials,
                static_discovery=False,
            )
            self.drive_service = build(
                "drive",
                self.DRIVE_API_VERSION,
                credentials=self.credentials,
                static_discovery=False,
            )
            print(
                "Google Docs & Drive API clients initialized for "{self.server_name}' using {self.auth_method_used}."
            )
            # Test call (optional, e.g., drive.about.get())
            # about = self.drive_service.about().get(fields="user").execute()
            # print(f"    Drive API user: {about['user']['displayName']}")
        except DefaultCredentialsError as dce:
            print(
                "CRITICAL: DefaultCredentialsError for GoogleDocsMCPServer "{self.server_name}': {dce}. Ensure ADC are configured or provide a service account key."
            )
            self.docs_service = None
            self.drive_service = None
        except Exception as e:
            print(
                f"CRITICAL: Error initializing Google API clients for {self.server_name}: {type(e).__name__} - {e}"
            )
            self.docs_service = None
            self.drive_service = None

    def _create_document_from_template(
        self,
        template_id: str,
        title: str,
        folder_id: str = None,
        replacements: dict = None,
    ) -> dict:
        """
        Creates a new Google Doc from a template by copying it and replacing placeholders.
        (Placeholder: Actual Google Drive and Docs API calls.)
        Requires Drive API for copying, Docs API for replacements.
        """
        target_folder_id = folder_id or self.config.get("default_folder_id")
        print(f"  {self.server_name}._create_document_from_template (mock) called.")
        print(
            f"    Template ID: {template_id}, Title: '{title}', Folder ID: {target_folder_id}"
        )
        print(f"    Replacements (sample): {str(replacements)[:100]}...")

        # Placeholder logic:
        # 1. Use Drive API to copy the template_id. (drive_service.files().copy(...).execute())
        #    - This gives a new_document_id.
        #    - Might need to move the new doc to target_folder_id if copy doesn't support it directly.
        # 2. Use Docs API with new_document_id to perform batchUpdate for replacements.
        #    - requests = []
        #    - for placeholder, value in replacements.items():
        #    -   requests.append({
        #    -       'replaceAllText': {
        #    -           'containsText': {'text': f'{{{{{placeholder}}}}}', 'matchCase': True},
        #    -           'replaceText': value
        #    -       }
        #    -   })
        #    - if requests:
        #    -   self.service.documents().batchUpdate(documentId=new_document_id, body={'requests': requests}).execute()
        # return {"status": "success_mock", "document_id": f"mock_new_doc_id_{hash(title)}", "url": f"https://docs.google.com/document/d/mock_new_doc_id_{hash(title)}/edit"}
        mock_doc_id = f"mock_gdoc_{hash(title) % 100000}"
        return {
            "status": "success_mock",
            "document_id": mock_doc_id,
            "url": f"https://docs.google.com/document/d/{mock_doc_id}/edit",
        }

    def _update_document_content(self, document_id: str, requests: list) -> dict:
        """
        Updates an existing Google Doc using a batchUpdate request.
        (Placeholder: Actual Google Docs API batchUpdate call.)
        """
        print(
            f"  {self.server_name}._update_document_content (mock) called for doc '{document_id}'. Num requests: {len(requests)}."
        )
        # try:
        #     result = self.service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()
        #     return {"status": "success_mock", "document_id": document_id, "result": result}
        # except Exception as e:
        #     print(f"Error updating Google Doc {document_id} for {self.server_name}: {e}")
        #     raise
        return {
            "status": "success_mock",
            "document_id": document_id,
            "updated_elements": len(requests),
        }

    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Executes a tool supported by the GoogleDocsMCPServer.
        Supported tools:
            - "create_doc_from_template": Creates a new doc from a template.
                Required params: "template_id" (str), "title" (str)
                Optional params: "folder_id" (str), "replacements" (dict of placeholder:value)
            - "update_document_content": Updates an existing document.
                Required params: "document_id" (str), "requests" (list of Docs API request objects)
        """
        if tool_name == "create_doc_from_template":
            template_id = parameters.get("template_id")
            title = parameters.get("title")
            if not template_id or not title:
                raise ValueError(
                    "Missing "template_id' or 'title' for 'create_doc_from_template' tool in {self.server_name}."
                )
            return self._create_document_from_template(
                template_id=template_id,
                title=title,
                folder_id=parameters.get("folder_id"),
                replacements=parameters.get("replacements"),
            )
        elif tool_name == "update_document_content":
            document_id = parameters.get("document_id")
            requests = parameters.get("requests")
            if not document_id or not requests:
                raise ValueError(
                    "Missing "document_id' or 'requests' for 'update_document_content' tool in {self.server_name}."
                )
            return self._update_document_content(document_id, requests)
        else:
            raise NotImplementedError(
                "Tool "{tool_name}' is not supported by {self.server_name} ({self.__class__.__name__})."
            )


# Example Usage (conceptual)
if __name__ == "__main__":
    print(
        "Testing GoogleDocsMCPServer... (Requires ADC or service_account.json and relevant Drive/Docs API enabled)"
    )
    # For ADC: ensure `gcloud auth application-default login` has been run with user creds that have access.
    # For Service Account: Create/download key, put path in config.
    gcp_project_for_adc = os.environ.get(
        "GOOGLE_CLOUD_PROJECT"
    )  # ADC often associated with a project
    creds_file_path = "./config/gcp_service_account_key.json"  # Placeholder - REPLACE if using service account

    test_config = {}
    if os.path.exists(creds_file_path):
        test_config["credentials_path"] = creds_file_path
        print(f"  Using service account key: {creds_file_path}")
    elif gcp_project_for_adc:
        # No explicit creds_path, will attempt ADC. Project for context if ADC shows it.
        print(
            f"  Attempting ADC (associated project if shown by SDK: {gcp_project_for_adc}). Ensure you are authenticated."
        )
    else:
        print(
            "  WARNING: No explicit credentials_path and GOOGLE_CLOUD_PROJECT env var not set. ADC might fail or use unexpected project."
        )
        # test_config["credentials_path"] = "path/to/your/service_account.json" # Or skip if only testing ADC fail

    # IMPORTANT: Replace with a REAL Google Docs template ID you own and can copy.
    # This template should have placeholders like {{REPORT_TITLE}} and {{MAIN_CONTENT_AREA}}
    test_template_id = "YOUR_GOOGLE_DOCS_TEMPLATE_ID_HERE"
    test_folder_id = None  # Optional: A Drive Folder ID to create the new doc in

    if test_template_id == "YOUR_GOOGLE_DOCS_TEMPLATE_ID_HERE":
        print(
            "\nSKIPPING Google Docs test: Please replace 'YOUR_GOOGLE_DOCS_TEMPLATE_ID_HERE' with a real template ID."
        )
    else:
        gdocs_server = GoogleDocsMCPServer(
            server_name="gdocs_sdk_test", config=test_config
        )
        if gdocs_server.docs_service and gdocs_server.drive_service:
            print("\n--- Test 1: Create Document from Template ---")
            create_params = {
                "template_id": test_template_id,
                "title": f"MCP Test Report - {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "folder_id": test_folder_id,  # Optional
                "replacements": {
                    "REPORT_TITLE": "Automated Test Report",
                    "MAIN_CONTENT_AREA": "This is the main content.\n- Point 1\n- Point 2",
                },
            }
            create_result = gdocs_server.call_tool(
                "create_doc_from_template", create_params
            )
            print(f"Create Doc Result:\n{json.dumps(create_result, indent=2)}")

            if create_result.get("status") == "success" and create_result.get(
                "document_id"
            ):
                new_doc_id = create_result["document_id"]
                print("\n--- Test 2: Update Document Content (Append text) ---")
                update_requests = [
                    {
                        "insertText": {
                            "endOfSegmentLocation": {
                                "segmentId": ""
                            },  # Appends to the end of the body
                            "text": "\n\nAdditional appended section via MCP.",
                        }
                    }
                ]
                update_result = gdocs_server.call_tool(
                    "update_document_content",
                    {"document_id": new_doc_id, "requests": update_requests},
                )
                print(f"Update Doc Result:\n{json.dumps(update_result, indent=2)}")
        else:
            print(
                "Google Docs/Drive services not initialized. Skipping API call tests."
            )

    print("\nGoogleDocsMCPServer test completed.")

# Required for SERVER_TIMESTAMP and test main datetime
import datetime
