from mcp_servers.google_docs_mcp import GoogleDocsMCP

class AutomatedDocumentationSystem:
    def __init__(self):
        self.docs_mcp = GoogleDocsMCP()
    
    def create_research_report(self, report_content, title):
        """Create a Google Doc with the research report"""
        
        # Create document
        doc_id = self.docs_mcp.create_document(
            title=f"Padres Research Report - {title}",
            content=report_content
        )
        
        # Add formatting, images, etc.
        self.docs_mcp.format_document(doc_id)
        
        return doc_id
    
    def update_research_log(self, experiment_summary):
        """Maintain a running log of all experiments"""
        
        # Append to master research log
        self.docs_mcp.append_to_document(
            document_id="master_research_log_id",
            content=experiment_summary
        )

# Example usage
if __name__ == "__main__":
    documentation_system = AutomatedDocumentationSystem()
    
    # Example report content and title
    report_content = "This is a sample research report content."
    title = "Sample Report"
    
    # Create research report
    doc_id = documentation_system.create_research_report(report_content, title)
    print(f"Research report created with ID: {doc_id}")
    
    # Example experiment summary
    experiment_summary = "This is a sample experiment summary."
    
    # Update research log
    documentation_system.update_research_log(experiment_summary)
    print("Research log updated.") 