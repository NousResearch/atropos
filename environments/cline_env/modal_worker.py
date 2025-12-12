"""
Modal Worker for Cline Containers

This module provides Modal-based execution for Cline coding tasks using the official CLI.
Each task runs in an isolated container with the appropriate language toolchain.

Usage:
    # Set up Modal secrets first:
    modal secret create dockerhub-creds REGISTRY_USERNAME=xxx REGISTRY_PASSWORD=xxx
    modal secret create anthropic-key ANTHROPIC_API_KEY=sk-ant-xxx
    
    # Deploy the app:
    modal deploy modal_worker.py
    
    # Test locally:
    modal run modal_worker.py
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# Modal app definition
app = modal.App("cline-workers")

# Secrets
dockerhub_secret = modal.Secret.from_name("dockerhub-creds")
anthropic_secret = modal.Secret.from_name("anthropic-key")

# Registry configuration
REGISTRY = os.getenv("DOCKER_REGISTRY", "nousresearch")
TAG = os.getenv("DOCKER_TAG", "latest")

# All supported language profiles
SUPPORTED_PROFILES = [
    "base",
    "python",
    "rust",
    "node",
    "go",
    "cpp",
    "c",
    "java",
    "csharp",
    "kotlin",
    "php",
    "scala",
    "ruby",
    "dart",
    "lua",
    "elixir",
    "jupyter",
    "haskell",
    "swift",
    "shell",
]


def get_image_for_profile(profile_key: str) -> modal.Image:
    """Get Modal Image for a language profile.
    
    Uses the CLI-based Docker images that have `cline` installed globally.
    """
    image_name = f"{REGISTRY}/cline-{profile_key}:{TAG}"
    return (
        modal.Image.from_registry(
            image_name,
            secret=dockerhub_secret,
            force_build=True,  # Force pull latest image (bypass cache)
        )
        .entrypoint([])  # Clear entrypoint so Modal can run our function
    )


@dataclass
class ClineTaskResult:
    """Result from a Cline task execution."""
    task_id: str
    success: bool
    assistant_content: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    workspace_path: Optional[str] = None
    files_created: List[str] = field(default_factory=list)
    execution_time_s: float = 0.0


def run_cline_task_cli(
    issue_text: str,
    language: str = "Python",
    repo_url: Optional[str] = None,
    repo_branch: Optional[str] = None,
    task_timeout_s: float = 300.0,
) -> ClineTaskResult:
    """
    Run a Cline coding task using the CLI.
    
    This function:
    1. Configures the Cline CLI with API credentials
    2. Runs the task in YOLO mode (-y -o)
    3. Collects the trajectory from ~/.cline/data/tasks/
    4. Returns the results
    
    Args:
        issue_text: The coding task/issue to solve
        language: Programming language context
        repo_url: Optional git repo to clone
        repo_branch: Optional branch to checkout
        task_timeout_s: Overall timeout for the task
        
    Returns:
        ClineTaskResult with the task outcome
    """
    start_time = time.time()
    
    # Log for Modal visibility (print goes to stdout which Modal captures)
    print(f"[CLINE] Starting task for language={language}, timeout={task_timeout_s}s")
    print(f"[CLINE] Issue text (first 200 chars): {issue_text[:200]}...")
    if repo_url:
        print(f"[CLINE] Repo: {repo_url} branch={repo_branch}")
    workspace_dir = Path(tempfile.mkdtemp(prefix=f"cline-{language.lower()}-"))
    cline_data_dir = Path.home() / ".cline" / "data" / "tasks"
    
    # Get API credentials from environment
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    anthropic_model = os.environ.get("CLINE_MODEL", "claude-sonnet-4-5-20250929")
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    
    env = os.environ.copy()
    env["HOME"] = str(Path.home())
    
    # npm global bin locations - covers various Node.js installation methods
    npm_bin_paths = [
        "/usr/local/bin",
        "/usr/bin",
        "/usr/lib/node_modules/.bin",
        "/opt/homebrew/bin",
        str(Path.home() / ".npm" / "bin"),
        str(Path.home() / ".npm-global" / "bin"),
    ]
    env["PATH"] = ":".join(npm_bin_paths) + ":" + env.get("PATH", "")
    
    # Find cline binary - first check common locations directly
    cline_bin = None
    candidate_paths = [
        "/usr/bin/cline",
        "/usr/local/bin/cline",
        "/usr/lib/node_modules/.bin/cline",
        "/usr/lib/node_modules/cline/dist/cli.js",
    ]
    for path in candidate_paths:
        if Path(path).exists():
            cline_bin = path
            break
    
    # Fall back to shutil.which
    if not cline_bin:
        cline_bin = shutil.which("cline", path=env["PATH"])
    
    if not cline_bin:
        # Debug: list what's in /usr/bin
        usr_bin_files = []
        try:
            usr_bin_files = sorted([f.name for f in Path("/usr/bin").iterdir() if "cline" in f.name.lower() or "node" in f.name.lower()])[:20]
        except Exception:
            pass
        return ClineTaskResult(
            task_id="",
            success=False,
            assistant_content="",
            error=f"Could not find 'cline' binary. Candidates: {candidate_paths}. PATH: {env['PATH'][:200]}... Relevant /usr/bin files: {usr_bin_files}",
        )
    
    try:
        # Configure API provider
        if anthropic_key:
            auth_cmd = [
                cline_bin, "auth",
                "-p", "anthropic",
                "-k", anthropic_key,
                "-m", anthropic_model,
                "--output-format", "plain"
            ]
        elif openai_key and openai_base_url:
            # Custom OpenAI-compatible endpoint (e.g., vLLM)
            auth_cmd = [
                cline_bin, "auth",
                "-p", "openai-compatible",
                "-k", openai_key,
                "-m", os.environ.get("CLINE_MODEL", "gpt-4"),
                "-b", openai_base_url,
                "--output-format", "plain"
            ]
        elif openai_key:
            auth_cmd = [
                cline_bin, "auth",
                "-p", "openai-native",
                "-k", openai_key,
                "-m", os.environ.get("CLINE_MODEL", "gpt-4o"),
                "--output-format", "plain"
            ]
        else:
            return ClineTaskResult(
                task_id="",
                success=False,
                assistant_content="",
                error="No API credentials provided (ANTHROPIC_API_KEY or OPENAI_API_KEY)",
            )
        
        # Run auth command
        print(f"[CLINE] Running auth with provider: {'anthropic' if anthropic_key else 'openai'}")
        auth_result = subprocess.run(
            auth_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        
        if auth_result.returncode != 0:
            print(f"[CLINE] Auth failed: {auth_result.stderr}")
            return ClineTaskResult(
                task_id="",
                success=False,
                assistant_content="",
                error=f"Auth failed: {auth_result.stderr}",
            )
        print("[CLINE] Auth successful")
        
        # Clone repo if provided
        if repo_url:
            clone_cmd = ["git", "clone", "--depth", "1"]
            if repo_branch:
                clone_cmd.extend(["-b", repo_branch])
            clone_cmd.extend([repo_url, str(workspace_dir)])
            subprocess.run(clone_cmd, capture_output=True, timeout=120)
        
        # Get list of existing task dirs to find new one after
        existing_tasks = set()
        if cline_data_dir.exists():
            existing_tasks = set(d.name for d in cline_data_dir.iterdir() if d.is_dir())
        
        # Run the Cline task in YOLO mode
        task_cmd = [
            cline_bin,
            issue_text,
            "-y",  # YOLO mode (auto-approve)
            "-o",  # Oneshot mode (full autonomy)
            "--output-format", "plain",
        ]
        
        task_result = subprocess.run(
            task_cmd,
            capture_output=True,
            text=True,
            timeout=task_timeout_s,
            env=env,
            cwd=str(workspace_dir),
        )
        
        execution_time = time.time() - start_time
        
        # Find the new task directory
        task_id = ""
        conversation_history: List[Dict[str, Any]] = []
        
        if cline_data_dir.exists():
            current_tasks = set(d.name for d in cline_data_dir.iterdir() if d.is_dir())
            new_tasks = current_tasks - existing_tasks
            if new_tasks:
                # Get the most recent new task
                task_id = max(new_tasks)  # Task IDs are timestamps
                task_dir = cline_data_dir / task_id
                
                # Read conversation history
                conv_file = task_dir / "api_conversation_history.json"
                if conv_file.exists():
                    try:
                        # Use errors='replace' to handle any non-UTF-8 bytes
                        content = conv_file.read_text(encoding='utf-8', errors='replace')
                        conversation_history = json.loads(content)
                        print(f"[CLINE] Loaded conversation history: {len(conversation_history)} messages")
                    except json.JSONDecodeError as e:
                        print(f"[CLINE] Failed to parse conversation history JSON: {e}")
                    except Exception as e:
                        print(f"[CLINE] Error reading conversation history: {e}")
        
        # Build assistant content from conversation, including tool calls and results
        assistant_parts: List[str] = []
        for msg in conversation_history:
            role = msg.get("role")
            content = msg.get("content", [])
            
            # Handle both list and string content formats
            if isinstance(content, str):
                if role == "assistant":
                    assistant_parts.append(content)
                continue
            
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    
                    if item_type == "text":
                        text = item.get("text", "")
                        if role == "assistant" and text:
                            assistant_parts.append(text)
                    elif item_type == "thinking":
                        thinking = item.get("thinking", "")
                        if thinking:
                            assistant_parts.append(f"<thinking>{thinking}</thinking>")
                    elif item_type == "tool_use":
                        # Format tool call for training data (assistant turn)
                        tool_name = item.get("name", "")
                        tool_input = item.get("input", {})
                        tool_id = item.get("id", "")
                        tool_input_str = json.dumps(tool_input, indent=2) if isinstance(tool_input, dict) else str(tool_input)
                        tool_call_xml = f"<tool_use id=\"{tool_id}\" name=\"{tool_name}\">\n{tool_input_str}\n</tool_use>"
                        assistant_parts.append(tool_call_xml)
                    elif item_type == "tool_result":
                        # Format tool result for training data (user turn containing result)
                        tool_use_id = item.get("tool_use_id", "")
                        result_content = item.get("content", "")
                        # Handle content that might be a list of text blocks
                        if isinstance(result_content, list):
                            result_parts = []
                            for rc in result_content:
                                if isinstance(rc, dict) and rc.get("type") == "text":
                                    result_parts.append(rc.get("text", ""))
                                elif isinstance(rc, str):
                                    result_parts.append(rc)
                            result_content = "\n".join(result_parts)
                        tool_result_xml = f"<tool_result id=\"{tool_use_id}\">\n{result_content}\n</tool_result>"
                        assistant_parts.append(tool_result_xml)
        
        assistant_content = "\n\n".join(assistant_parts)
        
        # List files created in workspace (check both the cwd and /workspace)
        files_created = []
        
        # Check temp workspace directory
        for f in workspace_dir.rglob("*"):
            if f.is_file() and not str(f.relative_to(workspace_dir)).startswith(".git"):
                files_created.append(str(f.relative_to(workspace_dir)))
        
        # Also check Docker's /workspace directory (if different from cwd)
        docker_workspace = Path("/workspace")
        if docker_workspace.exists() and docker_workspace != workspace_dir:
            for f in docker_workspace.rglob("*"):
                if f.is_file() and not str(f.relative_to(docker_workspace)).startswith(".git"):
                    rel_path = str(f.relative_to(docker_workspace))
                    if rel_path not in files_created:
                        files_created.append(f"/workspace/{rel_path}")
        
        print(f"[CLINE] Task completed: success={task_result.returncode == 0}, time={execution_time:.1f}s")
        print(f"[CLINE] Task ID: {task_id}, files_created: {len(files_created)}, conv_messages: {len(conversation_history)}")
        
        return ClineTaskResult(
            task_id=task_id,
            success=task_result.returncode == 0,
            assistant_content=assistant_content,
            conversation_history=conversation_history,
            workspace_path=str(workspace_dir),
            files_created=files_created,
            execution_time_s=execution_time,
            error=task_result.stderr if task_result.returncode != 0 else None,
        )
        
    except subprocess.TimeoutExpired:
        print(f"[CLINE] Task TIMEOUT after {task_timeout_s}s")
        return ClineTaskResult(
            task_id="",
            success=False,
            assistant_content="",
            error=f"Task timed out after {task_timeout_s}s",
            execution_time_s=time.time() - start_time,
        )
    except Exception as e:
        return ClineTaskResult(
            task_id="",
            success=False,
            assistant_content="",
            error=str(e),
            execution_time_s=time.time() - start_time,
        )


def _make_task_handler(profile_key: str):
    """Create the inner task handler function for a profile."""
    def task_handler(
        issue_text: str,
        repo_url: Optional[str] = None,
        repo_branch: Optional[str] = None,
        task_timeout_s: float = 300.0,
    ) -> Dict[str, Any]:
        result = run_cline_task_cli(
            issue_text=issue_text,
            language=profile_key.capitalize(),
            repo_url=repo_url,
            repo_branch=repo_branch,
            task_timeout_s=task_timeout_s,
        )
        return {
            "task_id": result.task_id,
            "success": result.success,
            "assistant_content": result.assistant_content,
            "conversation_history": result.conversation_history,
            "error": result.error,
            "workspace_path": result.workspace_path,
            "files_created": result.files_created,
            "execution_time_s": result.execution_time_s,
        }
    return task_handler


# Pre-create images for all profiles
_PROFILE_IMAGES = {profile: get_image_for_profile(profile) for profile in SUPPORTED_PROFILES}

# Define Modal functions explicitly at module level for each profile
# Using explicit definitions to avoid serialization issues


@app.function(
    image=_PROFILE_IMAGES["base"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_base_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("base")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["python"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_python_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("python")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["rust"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_rust_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("rust")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["node"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_node_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("node")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["go"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_go_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("go")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["cpp"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_cpp_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("cpp")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["c"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_c_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("c")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["java"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_java_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("java")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["csharp"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_csharp_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("csharp")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["kotlin"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_kotlin_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("kotlin")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["php"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_php_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("php")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["scala"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_scala_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("scala")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["ruby"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_ruby_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("ruby")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["dart"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_dart_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("dart")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["lua"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_lua_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("lua")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["elixir"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_elixir_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("elixir")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["jupyter"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_jupyter_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("jupyter")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["haskell"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_haskell_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("haskell")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["swift"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_swift_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("swift")(issue_text, repo_url, repo_branch, task_timeout_s)


@app.function(
    image=_PROFILE_IMAGES["shell"],
    secrets=[dockerhub_secret, anthropic_secret],
    timeout=600,
    memory=4096,
    cpu=2.0,
)
def run_shell_task(issue_text: str, repo_url: Optional[str] = None, repo_branch: Optional[str] = None, task_timeout_s: float = 300.0) -> Dict[str, Any]:
    return _make_task_handler("shell")(issue_text, repo_url, repo_branch, task_timeout_s)


# Map profile keys to Modal functions
LANGUAGE_FUNCTIONS: Dict[str, Any] = {
    "base": run_base_task,
    "python": run_python_task,
    "rust": run_rust_task,
    "node": run_node_task,
    "go": run_go_task,
    "cpp": run_cpp_task,
    "c": run_c_task,
    "java": run_java_task,
    "csharp": run_csharp_task,
    "kotlin": run_kotlin_task,
    "php": run_php_task,
    "scala": run_scala_task,
    "ruby": run_ruby_task,
    "dart": run_dart_task,
    "lua": run_lua_task,
    "elixir": run_elixir_task,
    "jupyter": run_jupyter_task,
    "haskell": run_haskell_task,
    "swift": run_swift_task,
    "shell": run_shell_task,
}


def get_function_for_language(language: str):
    """Get the Modal function for a specific language.
    
    Args:
        language: Language name (e.g., "Python", "rust", "TypeScript")
        
    Returns:
        Modal function for the language, or base function as fallback
    """
    lang_key = language.lower()
    
    # Handle common aliases
    if lang_key in ("typescript", "javascript"):
        lang_key = "node"
    elif lang_key == "c++":
        lang_key = "cpp"
    elif lang_key == "c#":
        lang_key = "csharp"
    elif lang_key == "jupyter notebook":
        lang_key = "jupyter"
    
    if lang_key in LANGUAGE_FUNCTIONS:
        return LANGUAGE_FUNCTIONS[lang_key]
    
    # Fall back to base image for unknown languages
    return LANGUAGE_FUNCTIONS["base"]


def list_available_profiles() -> List[str]:
    """Return list of all available language profiles."""
    return SUPPORTED_PROFILES.copy()


# Local test entrypoint
@app.local_entrypoint()
def main():
    """Test Modal Cline worker."""
    print("Testing Modal Cline Worker (CLI-based)...")
    print(f"Available profiles: {', '.join(SUPPORTED_PROFILES)}")
    
    # Simple infra test - just create a hello world file in the current directory
    issue_text = "Create a file named hello.py in the current directory with: print('Hello, World!'). Just create the file, nothing else."
    
    result = run_python_task.remote(
        issue_text=issue_text,
        task_timeout_s=120.0,
    )
    
    print(f"Task ID: {result['task_id']}")
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")
    print(f"Files created: {result['files_created']}")
    print(f"Execution time: {result['execution_time_s']:.1f}s")
    print(f"Conversation messages: {len(result['conversation_history'])}")
    
    if result['assistant_content']:
        print(f"\nAssistant content (first 500 chars):")
        print(result['assistant_content'][:500])
