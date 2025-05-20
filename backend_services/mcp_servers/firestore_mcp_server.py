import os
from .base_mcp_server import BaseMCPServer
from google.cloud import firestore
from google.cloud.firestore import Client as FirestoreClient # Explicit import for type hinting
from google.cloud.exceptions import NotFound
from typing import Optional, Dict, Any

class FirestoreMCPServer(BaseMCPServer):
    """
    MCP Server implementation for interacting with Google Cloud Firestore.
    """

    def __init__(self, server_name: str, config: dict):
        """
        Initializes the FirestoreMCPServer.
        Expected config keys:
            - "project_id": Your Google Cloud Project ID.
            - "database_id" (optional): The Firestore database ID (default is "(default)").
        """
        self.db: Optional[FirestoreClient] = None
        super().__init__(server_name, config)

    def _validate_config(self):
        """Validates that 'project_id' is in the config."""
        if "project_id" not in self.config:
            raise ValueError(f"Missing 'project_id' in config for FirestoreMCPServer: {self.server_name}")
        self.project_id = self.config["project_id"]
        self.database_id = self.config.get("database_id", "(default)")

    def _initialize_client(self):
        """Initializes the Firestore client."""
        try:
            self.db = firestore.Client(
                project=self.project_id,
                database=self.database_id
            )
            print(f"Firestore client initialized for project '{self.project_id}' (database: '{self.database_id}') for server '{self.server_name}'.")
            # Test connection by trying to get a non-existent document in a test collection.
            # This is a lightweight way to confirm basic connectivity and auth.
            test_doc_ref = self.db.collection("__mcp_server_connectivity_test__").document("__check__")
            test_doc_ref.get() # This operation itself is the test.
            print(f"  Firestore connectivity test successful for {self.server_name}.")
        except Exception as e:
            print(f"CRITICAL: Error initializing Firestore client for {self.server_name}: {e}")
            self.db = None # Ensure db is None if init fails
            # raise # Uncomment to make Firestore client initialization critical

    def _get_document(self, collection_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a document from Firestore.
        Returns the document data as a dict, or None if not found, or an error dict on failure.
        """
        if not self.db:
            print(f"Error: Firestore client not initialized for {self.server_name}. Cannot get document.")
            return {"status": "failure", "error": f"Firestore client not initialized for {self.server_name}"}
        
        print(f"  {self.server_name}._get_document called for collection '{collection_id}', document '{document_id}'.")
        try:
            doc_ref = self.db.collection(collection_id).document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict() # Success, raw document data
            else:
                return None # Successfully determined not found
        except NotFound:
            return None # Successfully determined not found (explicitly caught)
        except Exception as e:
            print(f"Error getting document {collection_id}/{document_id} from Firestore for {self.server_name}: {e}")
            return {"status": "failure", "error": f"Failed to get document {collection_id}/{document_id}", "details": str(e)}

    def _set_document(self, collection_id: str, document_id: str, data: Dict[str, Any], merge: bool = False) -> Dict[str, Any]:
        """
        Sets/creates or merges a document in Firestore.
        Returns a status dictionary.
        """
        if not self.db:
            return {"status": "failure", "error": f"Firestore client not initialized for {self.server_name}"}
        
        print(f"  {self.server_name}._set_document called for '{collection_id}/{document_id}'. Merge: {merge}. Data (sample): {str(data)[:100]}...")
        try:
            doc_ref = self.db.collection(collection_id).document(document_id)
            doc_ref.set(data, merge=merge)
            return {"status": "success", "collection_id": collection_id, "document_id": document_id, "operation": "set", "merged": merge}
        except Exception as e:
            print(f"Error setting document {collection_id}/{document_id} in Firestore for {self.server_name}: {e}")
            return {"status": "failure", "error": f"Failed to set document {collection_id}/{document_id}", "details": str(e)}

    def _add_document(self, collection_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds a new document to a Firestore collection with an auto-generated ID.
        Returns a status dictionary including the new document ID.
        """
        if not self.db:
            return {"status": "failure", "error": f"Firestore client not initialized for {self.server_name}"}
        
        print(f"  {self.server_name}._add_document called for collection '{collection_id}'. Data (sample): {str(data)[:100]}...")
        try:
            update_time, doc_ref = self.db.collection(collection_id).add(data)
            return {"status": "success", "collection_id": collection_id, "document_id": doc_ref.id, "update_time": update_time.isoformat(), "operation": "add"}
        except Exception as e:
            print(f"Error adding document to collection {collection_id} in Firestore for {self.server_name}: {e}")
            return {"status": "failure", "error": f"Failed to add document to {collection_id}", "details": str(e)}

    def _delete_document(self, collection_id: str, document_id: str) -> Dict[str, Any]:
        """
        Deletes a document from Firestore.
        Returns a status dictionary.
        """
        if not self.db:
            return {"status": "failure", "error": f"Firestore client not initialized for {self.server_name}"}

        print(f"  {self.server_name}._delete_document called for '{collection_id}/{document_id}'.")
        try:
            doc_ref = self.db.collection(collection_id).document(document_id)
            doc_ref.delete()
            return {"status": "success", "collection_id": collection_id, "document_id": document_id, "operation": "delete"}
        except Exception as e:
            print(f"Error deleting document {collection_id}/{document_id} from Firestore for {self.server_name}: {e}")
            return {"status": "failure", "error": f"Failed to delete document {collection_id}/{document_id}", "details": str(e)}

    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Executes a tool supported by the FirestoreMCPServer.
        Supported tools:
            - "get_document": Retrieves a document. Returns doc data dict, None if not found, or error dict.
            - "set_document": Creates or overwrites/merges a document. Returns status dict.
            - "add_document": Adds a new document with an auto-generated ID. Returns status dict.
            - "delete_document": Deletes a document. Returns status dict.
        """
        if tool_name == "get_document":
            collection_id = parameters.get("collection_id")
            document_id = parameters.get("document_id")
            if not collection_id or not document_id:
                raise ValueError(f"Missing 'collection_id' or 'document_id' for 'get_document' tool in {self.server_name}.")
            return self._get_document(collection_id, document_id)
        
        elif tool_name == "set_document":
            collection_id = parameters.get("collection_id")
            document_id = parameters.get("document_id")
            data = parameters.get("data")
            if not collection_id or not document_id or data is None:
                raise ValueError(f"Missing 'collection_id', 'document_id', or 'data' for 'set_document' tool in {self.server_name}.")
            return self._set_document(collection_id, document_id, data, merge=parameters.get("merge", False))
        
        elif tool_name == "add_document":
            collection_id = parameters.get("collection_id")
            data = parameters.get("data")
            if not collection_id or data is None:
                raise ValueError(f"Missing 'collection_id' or 'data' for 'add_document' tool in {self.server_name}.")
            return self._add_document(collection_id, data)

        elif tool_name == "delete_document":
            collection_id = parameters.get("collection_id")
            document_id = parameters.get("document_id")
            if not collection_id or not document_id:
                raise ValueError(f"Missing 'collection_id' or 'document_id' for 'delete_document' tool in {self.server_name}.")
            return self._delete_document(collection_id, document_id)
        else:
            raise NotImplementedError(f"Tool '{tool_name}' is not supported by {self.server_name} ({self.__class__.__name__}).")

# Example Usage (conceptual)
if __name__ == "__main__":
    print("Testing FirestoreMCPServer with google-cloud-firestore SDK...")
    # THIS TEST WILL FAIL IF YOU DON'T HAVE GCP AUTHENTICATION SET UP (e.g., gcloud auth application-default login)
    # AND A VALID PROJECT ID IN THE CONFIG or GOOGLE_CLOUD_PROJECT env var.
    
    gcp_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not gcp_project_id:
        print("SKIPPING real FirestoreMCPServer test: GOOGLE_CLOUD_PROJECT env var not set.")
        print("Set it to your GCP project ID to run this test against a real Firestore instance.")
    else:
        print(f"Using GCP Project: {gcp_project_id} for Firestore tests.")
        sample_fs_config_real = {
            "project_id": gcp_project_id
            # "database_id": "(default)" # Optional, defaults in client
        }
        fs_server_real = FirestoreMCPServer(server_name="firestore_sdk_test", config=sample_fs_config_real)

        if fs_server_real.db: # Proceed only if client initialized
            test_collection = "mcp_server_tests"
            test_doc_id1 = "test_doc_001"
            test_doc_id2 = "test_doc_002_autoid_target"

            print(f"\n--- Test: Set Document ({test_doc_id1}) ---")
            set_data = {"name": "Test Document Alpha", "value": 123, "tags": ["test", "alpha"], "timestamp": firestore.SERVER_TIMESTAMP}
            set_result = fs_server_real.call_tool("set_document", {"collection_id": test_collection, "document_id": test_doc_id1, "data": set_data})
            print(f"Set Result: {set_result}")
            assert set_result["status"] == "success"

            print(f"\n--- Test: Get Document ({test_doc_id1}) ---")
            get_result = fs_server_real.call_tool("get_document", {"collection_id": test_collection, "document_id": test_doc_id1})
            print(f"Get Result: {get_result}")
            assert get_result is not None and get_result.get("name") == "Test Document Alpha"

            print(f"\n--- Test: Add Document (auto-ID) to {test_collection} ---")
            add_data = {"name": "Test Document Beta (AutoID)", "value": 456, "status": "new"}
            add_result = fs_server_real.call_tool("add_document", {"collection_id": test_collection, "data": add_data})
            print(f"Add Result: {add_result}")
            assert add_result["status"] == "success" and add_result.get("document_id")
            auto_id_to_delete = add_result.get("document_id")

            if auto_id_to_delete:
                print(f"\n--- Test: Get Document (auto-ID: {auto_id_to_delete}) ---")
                get_auto_result = fs_server_real.call_tool("get_document", {"collection_id": test_collection, "document_id": auto_id_to_delete})
                print(f"Get Auto-ID Result: {get_auto_result}")
                assert get_auto_result is not None and get_auto_result.get("name") == "Test Document Beta (AutoID)"

            print(f"\n--- Test: Set Document with Merge ({test_doc_id1}) ---")
            merge_data = {"value": 789, "status": "updated", "new_field": "merged_in"}
            merge_result = fs_server_real.call_tool("set_document", {"collection_id": test_collection, "document_id": test_doc_id1, "data": merge_data, "merge": True})
            print(f"Merge Result: {merge_result}")
            assert merge_result["status"] == "success"
            get_merged_result = fs_server_real.call_tool("get_document", {"collection_id": test_collection, "document_id": test_doc_id1})
            print(f"Get After Merge Result: {get_merged_result}")
            assert get_merged_result and get_merged_result.get("new_field") == "merged_in" and get_merged_result.get("value") == 789
            
            print(f"\n--- Test: Delete Document ({test_doc_id1}) ---")
            delete_result = fs_server_real.call_tool("delete_document", {"collection_id": test_collection, "document_id": test_doc_id1})
            print(f"Delete Result ({test_doc_id1}): {delete_result}")
            get_after_delete = fs_server_real.call_tool("get_document", {"collection_id": test_collection, "document_id": test_doc_id1})
            print(f"Get After Delete ({test_doc_id1}): {get_after_delete} (should be None or error if get_document returns None for not found)")

            if auto_id_to_delete:
                print(f"\n--- Test: Delete Document (auto-ID: {auto_id_to_delete}) ---")
                delete_auto_result = fs_server_real.call_tool("delete_document", {"collection_id": test_collection, "document_id": auto_id_to_delete})
                print(f"Delete Result ({auto_id_to_delete}): {delete_auto_result}")
        else:
            print("Skipping real Firestore client tests as client did not initialize.")

    print("\nFirestoreMCPServer test completed.") 