"""
Process-based runner for SmolaGents CodeAgent.
"""

import logging
import multiprocessing
import os
import time
import traceback
from typing import Any, Dict

from smolagents import CodeAgent

# Import tools directly
from environments.smolagents_integration.server_proxy import ServerProxy
from environments.smolagents_integration.smolagents_model import ProcessSafeAtroposServerModel
from environments.smolagents_integration.tools.file_tools import (
    append_to_file, read_file, write_file
)

# Conditionally import Tavily tools if API key is available
tavily_tools = []
if os.environ.get("TAVILY_API_KEY"):
    try:
        from environments.smolagents_integration.tools.tavily_tools import (
            TavilyExtractTool, TavilySearchTool
        )
        tavily_tools = [
            TavilySearchTool(api_key=os.environ.get("TAVILY_API_KEY")),
            TavilyExtractTool(api_key=os.environ.get("TAVILY_API_KEY"))
        ]
    except ImportError:
        pass

# Configure logging for the subprocess
logging.basicConfig(
    level=logging.INFO, format="Process-%(process)d: %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent propagation to root logger

# Reduce verbosity of HTTP/networking related loggers
for logger_name in ["httpx", "httpcore", "openai", "requests", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def run_agent_process(
    prompt: str,
    task_metadata: Dict[str, Any],
    server_proxy: ServerProxy,
    agent_config: Dict[str, Any],
    result_queue: multiprocessing.Queue,
):
    """
    Run the CodeAgent in a separate process.

    Args:
        prompt: The prompt to send to the agent
        task_metadata: Metadata about the task
        server_proxy: Proxy for communicating with the Atropos server
        agent_config: Configuration for the agent
        result_queue: Queue to put the result in
    """
    try:
        start_time = time.time()
        process_id = os.getpid()
        logger.info(f"Process {process_id} starting for task {task_metadata.get('task_id', 'unknown')}")

        # Create a model using the server proxy
        model = ProcessSafeAtroposServerModel(
            server_proxy=server_proxy,
            use_chat_completion=agent_config.get("use_chat_completion", True),
            model_id=agent_config.get("model_name", "atropos-smolagents"),
        )

        # Combine all tools
        tools = [read_file, write_file, append_to_file] + tavily_tools

        # Initialize the CodeAgent
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=agent_config.get("max_steps", 12),
            additional_authorized_imports=["*"],
            verbosity_level=agent_config.get("verbosity", 2),
        )

        logger.info(f"Process {process_id}: Running agent on prompt with {len(prompt)} chars")

        # Run the agent and get response
        agent_response = agent.run(prompt)
        
        # Ensure the response is properly formatted (convert sets, etc. to strings)
        if not isinstance(agent_response, str):
            logger.info(f"Converting non-string response of type {type(agent_response)} to string")
            try:
                if isinstance(agent_response, set):
                    # Convert sets to comma-separated strings
                    agent_response = ", ".join(str(item) for item in agent_response)
                else:
                    # Try to convert other types to string
                    agent_response = str(agent_response)
            except Exception as e:
                logger.error(f"Failed to convert agent_response to string: {e}")
                agent_response = str(agent_response)
        
        # Extract agent memory
        agent_memory = getattr(agent, "memory", None)
        if hasattr(agent, "write_memory_to_messages"):
            agent_memory = agent.write_memory_to_messages()

        # Calculate execution time
        execution_time = time.time() - start_time

        # Prepare and send result
        result_queue.put({
            "status": "success",
            "response": agent_response,
            "task_id": task_metadata.get("task_id"),
            "execution_time": execution_time,
            "agent_memory": agent_memory,
            "task_metadata": task_metadata,
        })

        logger.info(f"Process {process_id}: Agent completed in {execution_time:.2f}s")

    except Exception as e:
        # Log the exception and put error result in queue
        logger.error(f"Process {os.getpid()}: Error in agent execution: {e}")
        logger.error(traceback.format_exc())
        
        result_queue.put({
            "status": "error",
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "task_id": task_metadata.get("task_id"),
            "task_metadata": task_metadata,
        })

    finally:
        logger.info(f"Process {os.getpid()}: Cleanup complete")
