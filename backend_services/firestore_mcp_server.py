from google.cloud import firestore
from backend_services.mcp_servers.base_mcp_server import BaseMCPServer
import os

class FirestoreMCPServer(BaseMCPServer):
    """
    MCP Server for interacting with Google Firestore.
    """
    
    def __init__(self, server_name: str, config: dict = None):
        """
        Initializes the FirestoreMCPServer.
        
        Args:
            server_name (str): The name of this MCP server instance.
            config (dict, optional): Configuration for this server, including:
                - project_id (str): GCP project ID.
                - mock_mode (bool): Whether to run in mock mode.
                - mock_data (dict): Mock data for document operations in mock mode.
        """
        super().__init__(server_name, config)
        self.config = config or {}
        self.project_id = config.get("project_id", os.environ.get("GOOGLE_CLOUD_PROJECT"))
        self.mock_mode = config.get("mock_mode", False)
        self.client = None
        
        # Mock data storage (for mock mode)
        self.mock_store = config.get("mock_data", {})
        
        if self.mock_mode:
            print(f"FirestoreMCPServer '{server_name}' initialized in MOCK mode with {len(self.mock_store)} mock documents.")
        else:
            self._initialize_client()
    
    def _initialize_client(self):
        """
        Initializes the Firestore client.
        """
        try:
            self.client = firestore.Client(project=self.project_id)
            print(f"Firestore client initialized for {self.server_name}.")
        except Exception as e:
            print(f"CRITICAL: Error initializing Firestore client for {self.server_name}: {e}")
            print(f"MCP Server '{self.server_name}' initialized of type FirestoreMCPServer.")
    
    def _log_error(self, message: str):
        """
        Logs an error message.
        """
        print(f"FirestoreMCPServer '{self.server_name}' error: {message}")
    
    def get_status(self) -> dict:
        """
        Gets the status of the FirestoreMCPServer.
        
        Returns:
            dict: Status information.
        """
        if self.mock_mode:
            return {
                "status": "available",
                "mode": "mock",
                "documents_available": len(self.mock_store)
            }
        else:
            return {
                "status": "available" if self.client else "error",
                "mode": "live"
            }
    
    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Calls a tool on the FirestoreMCPServer.
        
        Args:
            tool_name (str): The name of the tool to call.
            parameters (dict): The parameters for the tool call.
            
        Returns:
            any: The result of the tool call.
            
        Raises:
            NotImplementedError: If the tool is not supported.
        """
        if tool_name == "get_document":
            return self._get_document(parameters)
        elif tool_name == "set_document":
            return self._set_document(parameters)
        elif tool_name == "update_document":
            return self._update_document(parameters)
        elif tool_name == "delete_document":
            return self._delete_document(parameters)
        elif tool_name == "query_documents":
            return self._query_documents(parameters)
        else:
            raise NotImplementedError(f"Tool '{tool_name}' is not supported by FirestoreMCPServer.")
    
    def _get_document(self, parameters: dict) -> dict:
        """
        Gets a document from a Firestore collection.
        
        Args:
            parameters (dict): The parameters for the tool call, including:
                - collection_id (str): The ID of the collection.
                - document_id (str): The ID of the document.
                
        Returns:
            dict: The document data.
        """
        collection_id = parameters.get("collection_id", "")
        document_id = parameters.get("document_id", "")
        
        if not collection_id or not document_id:
            return {"error": "Both collection_id and document_id are required parameters."}
        
        # Handle mock mode
        if self.mock_mode:
            mock_doc_key = f"{collection_id}/{document_id}"
            if mock_doc_key in self.mock_store:
                print(f"FirestoreMCPServer (MOCK): Returning mock data for {mock_doc_key}")
                return self.mock_store[mock_doc_key]
            else:
                print(f"FirestoreMCPServer (MOCK): No mock data found for {mock_doc_key}")
                return {"error": f"No mock data found for {mock_doc_key}"}
        
        # Actual Firestore implementation
        if not self.client:
            return {"error": f"Firestore client not initialized for {self.server_name}. Cannot get document."}
        
        try:
            doc_ref = self.client.collection(collection_id).document(document_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                return {"error": f"Document '{document_id}' not found in collection '{collection_id}'."}
        except Exception as e:
            self._log_error(f"Error getting document: {e}")
            return {"error": f"Error getting document: {e}"}
    
    def _set_document(self, parameters: dict) -> dict:
        """
        Sets a document in a Firestore collection.
        
        Args:
            parameters (dict): The parameters for the tool call, including:
                - collection_id (str): The ID of the collection.
                - document_id (str): The ID of the document.
                - data (dict): The document data.
                
        Returns:
            dict: Result of the operation.
        """
        collection_id = parameters.get("collection_id")
        document_id = parameters.get("document_id")
        data = parameters.get("data")
        
        if not collection_id or not document_id or not data:
            return {"error": "collection_id, document_id, and data are required parameters."}
        
        # Handle mock mode
        if self.mock_mode:
            mock_doc_key = f"{collection_id}/{document_id}"
            self.mock_store[mock_doc_key] = data
            print(f"FirestoreMCPServer (MOCK): Set mock data for {mock_doc_key}")
            return {"status": "success_mock", "document_id": document_id}
        
        # Actual Firestore implementation
        if not self.client:
            return {"error": f"Firestore client not initialized for {self.server_name}. Cannot set document."}
        
        try:
            doc_ref = self.client.collection(collection_id).document(document_id)
            doc_ref.set(data)
            return {"status": "success", "document_id": document_id}
        except Exception as e:
            self._log_error(f"Error setting document: {e}")
            return {"error": f"Error setting document: {e}"}
    
    def _update_document(self, parameters: dict) -> dict:
        """
        Updates a document in a Firestore collection.
        
        Args:
            parameters (dict): The parameters for the tool call, including:
                - collection_id (str): The ID of the collection.
                - document_id (str): The ID of the document.
                - data (dict): The fields to update.
                
        Returns:
            dict: Result of the operation.
        """
        collection_id = parameters.get("collection_id")
        document_id = parameters.get("document_id")
        data = parameters.get("data")
        
        if not collection_id or not document_id or not data:
            return {"error": "collection_id, document_id, and data are required parameters."}
        
        # Handle mock mode
        if self.mock_mode:
            mock_doc_key = f"{collection_id}/{document_id}"
            if mock_doc_key in self.mock_store:
                self.mock_store[mock_doc_key].update(data)
                print(f"FirestoreMCPServer (MOCK): Updated mock data for {mock_doc_key}")
                return {"status": "success_mock", "document_id": document_id}
            else:
                print(f"FirestoreMCPServer (MOCK): Document not found for update: {mock_doc_key}")
                return {"error": f"Document not found for update: {mock_doc_key}"}
        
        # Actual Firestore implementation
        if not self.client:
            return {"error": f"Firestore client not initialized for {self.server_name}. Cannot update document."}
        
        try:
            doc_ref = self.client.collection(collection_id).document(document_id)
            doc_ref.update(data)
            return {"status": "success", "document_id": document_id}
        except Exception as e:
            self._log_error(f"Error updating document: {e}")
            return {"error": f"Error updating document: {e}"}
    
    def _delete_document(self, parameters: dict) -> dict:
        """
        Deletes a document from a Firestore collection.
        
        Args:
            parameters (dict): The parameters for the tool call, including:
                - collection_id (str): The ID of the collection.
                - document_id (str): The ID of the document.
                
        Returns:
            dict: Result of the operation.
        """
        collection_id = parameters.get("collection_id")
        document_id = parameters.get("document_id")
        
        if not collection_id or not document_id:
            return {"error": "Both collection_id and document_id are required parameters."}
        
        # Handle mock mode
        if self.mock_mode:
            mock_doc_key = f"{collection_id}/{document_id}"
            if mock_doc_key in self.mock_store:
                del self.mock_store[mock_doc_key]
                print(f"FirestoreMCPServer (MOCK): Deleted mock data for {mock_doc_key}")
                return {"status": "success_mock", "document_id": document_id}
            else:
                print(f"FirestoreMCPServer (MOCK): Document not found for deletion: {mock_doc_key}")
                return {"error": f"Document not found for deletion: {mock_doc_key}"}
        
        # Actual Firestore implementation
        if not self.client:
            return {"error": f"Firestore client not initialized for {self.server_name}. Cannot delete document."}
        
        try:
            doc_ref = self.client.collection(collection_id).document(document_id)
            doc_ref.delete()
            return {"status": "success", "document_id": document_id}
        except Exception as e:
            self._log_error(f"Error deleting document: {e}")
            return {"error": f"Error deleting document: {e}"}
    
    def _query_documents(self, parameters: dict) -> dict:
        """
        Queries documents from a Firestore collection.
        
        Args:
            parameters (dict): The parameters for the tool call, including:
                - collection_id (str): The ID of the collection.
                - query_filters (list): List of query filter tuples (field, op, value).
                - limit (int, optional): Maximum number of documents to return.
                
        Returns:
            dict: Query results with documents.
        """
        collection_id = parameters.get("collection_id")
        query_filters = parameters.get("query_filters", [])
        limit = parameters.get("limit")
        
        if not collection_id:
            return {"error": "collection_id is a required parameter."}
        
        # Handle mock mode
        if self.mock_mode:
            results = []
            for doc_key, doc_data in self.mock_store.items():
                if doc_key.startswith(f"{collection_id}/"):
                    # Apply filters if any
                    matches = True
                    for field, op, value in query_filters:
                        if field not in doc_data:
                            matches = False
                            break
                        if op == "==":
                            if doc_data[field] != value:
                                matches = False
                                break
                        # Add more operators as needed
                    
                    if matches:
                        results.append(doc_data)
            
            # Apply limit if specified
            if limit and isinstance(limit, int):
                results = results[:limit]
                
            print(f"FirestoreMCPServer (MOCK): Query returned {len(results)} results from collection {collection_id}")
            return {"documents": results}
        
        # Actual Firestore implementation
        if not self.client:
            return {"error": f"Firestore client not initialized for {self.server_name}. Cannot query documents."}
        
        try:
            query = self.client.collection(collection_id)
            
            # Apply filters
            for field, op, value in query_filters:
                query = query.where(field, op, value)
            
            # Apply limit if specified
            if limit and isinstance(limit, int):
                query = query.limit(limit)
                
            # Execute query
            docs = query.stream()
            results = [doc.to_dict() for doc in docs]
            
            return {"documents": results}
        except Exception as e:
            self._log_error(f"Error querying documents: {e}")
            return {"error": f"Error querying documents: {e}"} 