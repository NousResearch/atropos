"""
Scoring module for Cline agent trajectories.

Provides multiple scoring strategies:
1. Execution (syntax check): Does the code compile/parse?
2. LLM-as-judge: Claude evaluates against the original issue
3. Complexity bonus: Simple heuristics on trajectory quality

Combined score formula:
    final_score = 0.3 * execution + 0.6 * llm_judge + 0.1 * complexity
"""

import asyncio
import glob
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Weight configuration for combined scoring
WEIGHTS = {
    "execution": 0.3,
    "llm_judge": 0.6,
    "complexity": 0.1,
}

# Syntax check commands by language
# These should NOT execute code, just validate syntax
SYNTAX_CHECK_COMMANDS: Dict[str, List[List[str]]] = {
    "Python": [["python", "-m", "py_compile"]],  # Takes file as arg
    "JavaScript": [["node", "--check"]],  # Takes file as arg
    "TypeScript": [["npx", "tsc", "--noEmit", "--allowJs"]],
    "Rust": [["cargo", "check", "--message-format=short"]],
    "Go": [["go", "build", "-n", "."]],  # -n = dry run
    "Java": [["javac", "-d", "/tmp", "-Xlint:none"]],  # Takes files as args
    "C": [["gcc", "-fsyntax-only"]],  # Takes files as args
    "C++": [["g++", "-fsyntax-only"]],  # Takes files as args
    "C#": [["dotnet", "build", "--no-restore"]],
    "Kotlin": [["kotlinc", "-Werror"]],  # Takes files as args
    "PHP": [["php", "-l"]],  # Takes file as arg
    "Ruby": [["ruby", "-c"]],  # Takes file as arg
    "Scala": [["scalac", "-Werror"]],  # Takes files as args
    "Dart": [["dart", "analyze"]],
    "Lua": [["luac", "-p"]],  # Takes file as arg
    "Swift": [["swiftc", "-typecheck"]],  # Takes files as args
    "Shell": [["bash", "-n"]],  # Takes file as arg
    "Haskell": [["ghc", "-fno-code"]],  # Takes files as args
    "Elixir": [["mix", "compile", "--warnings-as-errors"]],
    "HTML": [],  # No syntax check for HTML
    "Jupyter Notebook": [],  # No syntax check for notebooks
}

# File extensions by language
FILE_EXTENSIONS: Dict[str, List[str]] = {
    "Python": [".py"],
    "JavaScript": [".js", ".mjs", ".cjs"],
    "TypeScript": [".ts", ".tsx"],
    "Rust": [".rs"],
    "Go": [".go"],
    "Java": [".java"],
    "C": [".c", ".h"],
    "C++": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
    "C#": [".cs"],
    "Kotlin": [".kt", ".kts"],
    "PHP": [".php"],
    "Ruby": [".rb"],
    "Scala": [".scala"],
    "Dart": [".dart"],
    "Lua": [".lua"],
    "Swift": [".swift"],
    "Shell": [".sh", ".bash"],
    "Haskell": [".hs"],
    "Elixir": [".ex", ".exs"],
    "HTML": [".html", ".htm"],
    "Jupyter Notebook": [".ipynb"],
}

# LLM Judge rubric prompt
LLM_JUDGE_PROMPT = """You are evaluating a coding agent's work on a GitHub issue.

## Original Issue/Task
{issue_text}

## Agent's Work Summary
The agent performed the following actions:
{trajectory_summary}

## Files Modified
{files_modified}

## Evaluation Criteria
Score the agent's work on a scale of 0-10:

- **0-2**: Did not address the issue at all, or made things worse
- **3-4**: Attempted but made significant errors or incomplete implementation
- **5-6**: Partially addressed the issue, some functionality works
- **7-8**: Good solution with minor issues or improvements possible
- **9-10**: Excellent, complete solution that fully addresses the requirements

Consider:
1. Did the agent understand the issue correctly?
2. Is the solution logically sound?
3. Is the code well-structured and maintainable?
4. Are there any obvious bugs or issues?

Respond with ONLY valid JSON in this exact format:
{{"score": <number 0-10>, "reasoning": "<brief explanation>"}}
"""


async def score_execution(
    workspace_path: Path,
    language: str,
    timeout_seconds: float = 30.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check syntax of code files in the workspace.
    
    Returns:
        Tuple of (score 0.0-1.0, metadata dict with details)
    """
    metadata: Dict[str, Any] = {
        "language": language,
        "workspace": str(workspace_path),
        "checks_run": [],
        "errors": [],
    }
    
    if not workspace_path.exists():
        metadata["errors"].append("Workspace does not exist")
        return 0.0, metadata
    
    commands = SYNTAX_CHECK_COMMANDS.get(language, [])
    extensions = FILE_EXTENSIONS.get(language, [])
    
    if not commands:
        # No syntax check available for this language
        metadata["checks_run"].append(f"No syntax check available for {language}")
        return 0.5, metadata  # Neutral score when we can't check
    
    # Find files to check
    files_to_check: List[Path] = []
    for ext in extensions:
        pattern = f"**/*{ext}"
        files_to_check.extend(workspace_path.glob(pattern))
    
    if not files_to_check:
        metadata["checks_run"].append(f"No {language} files found")
        return 0.5, metadata  # Neutral if no files
    
    metadata["files_found"] = [str(f.relative_to(workspace_path)) for f in files_to_check[:20]]
    
    # Run syntax checks
    errors_found = 0
    files_checked = 0
    
    for cmd_template in commands:
        try:
            # For commands that take individual files
            if cmd_template[-1] in ["python", "node", "php", "ruby", "bash", "luac"]:
                # These take file as argument
                for file_path in files_to_check[:10]:  # Limit to avoid long runs
                    cmd = cmd_template + [str(file_path)]
                    result = subprocess.run(
                        cmd,
                        cwd=workspace_path,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                    )
                    files_checked += 1
                    if result.returncode != 0:
                        errors_found += 1
                        metadata["errors"].append(f"{file_path.name}: {result.stderr[:200]}")
            else:
                # Project-level commands (cargo, go, npm, etc.)
                cmd = cmd_template
                result = subprocess.run(
                    cmd,
                    cwd=workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
                files_checked = len(files_to_check)
                if result.returncode != 0:
                    errors_found = len(files_to_check)
                    metadata["errors"].append(result.stderr[:500])
                    
            metadata["checks_run"].append(" ".join(cmd_template))
            
        except subprocess.TimeoutExpired:
            metadata["errors"].append(f"Syntax check timed out after {timeout_seconds}s")
            return 0.3, metadata
        except FileNotFoundError as e:
            metadata["errors"].append(f"Command not found: {e}")
            return 0.5, metadata  # Neutral if tooling missing
        except Exception as e:
            metadata["errors"].append(f"Error running syntax check: {e}")
            return 0.3, metadata
    
    if files_checked == 0:
        return 0.5, metadata
    
    # Calculate score based on error rate
    error_rate = errors_found / files_checked
    score = 1.0 - error_rate
    
    metadata["files_checked"] = files_checked
    metadata["errors_found"] = errors_found
    metadata["score"] = score
    
    return score, metadata


async def score_llm_judge(
    issue_text: str,
    trajectory_summary: str,
    files_modified: List[str],
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5-20250929",
    base_url: str = "https://api.anthropic.com",
    timeout_seconds: float = 60.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Use Claude to evaluate the trajectory against the original issue.
    
    Returns:
        Tuple of (score 0.0-1.0, metadata dict with reasoning)
    """
    metadata: Dict[str, Any] = {
        "model": model,
        "raw_score": None,
        "reasoning": None,
        "error": None,
    }
    
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        metadata["error"] = "No API key available"
        return 0.5, metadata  # Neutral when we can't judge
    
    # Format the prompt
    files_str = "\n".join(f"- {f}" for f in files_modified[:20]) or "No files modified"
    prompt = LLM_JUDGE_PROMPT.format(
        issue_text=issue_text[:4000],  # Truncate to avoid token limits
        trajectory_summary=trajectory_summary[:8000],
        files_modified=files_str,
    )
    
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("content", [{}])[0].get("text", "")
            
            # Parse JSON response
            try:
                # Handle potential markdown code blocks
                if "```" in content:
                    content = content.split("```json")[-1].split("```")[0].strip()
                    if not content:
                        content = data.get("content", [{}])[0].get("text", "").split("```")[-2].strip()
                
                result = json.loads(content)
                raw_score = float(result.get("score", 5))
                reasoning = result.get("reasoning", "No reasoning provided")
                
                # Clamp and normalize to 0-1
                raw_score = max(0, min(10, raw_score))
                normalized_score = raw_score / 10.0
                
                metadata["raw_score"] = raw_score
                metadata["reasoning"] = reasoning
                
                return normalized_score, metadata
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                metadata["error"] = f"Failed to parse LLM response: {e}"
                metadata["raw_response"] = content[:500]
                return 0.5, metadata
                
    except httpx.HTTPStatusError as e:
        metadata["error"] = f"API error: {e.response.status_code}"
        return 0.5, metadata
    except Exception as e:
        metadata["error"] = f"LLM judge error: {e}"
        return 0.5, metadata


def score_complexity(
    cline_metadata: Optional[Dict[str, Any]],
    trajectory_messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score based on trajectory complexity heuristics.
    
    Factors:
    - Did it make file changes?
    - Reasonable trajectory length?
    - Number of tool uses?
    
    Returns:
        Tuple of (score 0.0-1.0, metadata dict)
    """
    metadata: Dict[str, Any] = {
        "factors": {},
    }
    
    if not cline_metadata:
        return 0.5, metadata
    
    ui_messages = cline_metadata.get("ui_messages", [])
    
    # Factor 1: Made file changes (0.3)
    file_change_score = 0.0
    tool_uses = 0
    for msg in ui_messages:
        msg_type = msg.get("type") or msg.get("say")
        if msg_type in ["tool", "tool_use"]:
            tool_uses += 1
            tool_name = msg.get("text", "").lower()
            if any(t in tool_name for t in ["write", "replace", "create", "edit"]):
                file_change_score = 1.0
    metadata["factors"]["made_file_changes"] = file_change_score > 0
    
    # Factor 2: Reasonable trajectory length (0.4)
    # Too short (<3 messages) = probably didn't do much
    # Too long (>100 messages) = probably stuck in a loop
    msg_count = len(ui_messages)
    if msg_count < 3:
        length_score = 0.2
    elif msg_count > 100:
        length_score = 0.5
    elif msg_count > 50:
        length_score = 0.7
    else:
        length_score = 1.0
    metadata["factors"]["message_count"] = msg_count
    metadata["factors"]["length_score"] = length_score
    
    # Factor 3: Tool usage (0.3)
    # Some tool use is good, too much might indicate struggles
    if tool_uses == 0:
        tool_score = 0.3
    elif tool_uses > 30:
        tool_score = 0.6
    else:
        tool_score = 1.0
    metadata["factors"]["tool_uses"] = tool_uses
    metadata["factors"]["tool_score"] = tool_score
    
    # Combine factors
    combined = (
        0.3 * file_change_score +
        0.4 * length_score +
        0.3 * tool_score
    )
    
    metadata["combined_score"] = combined
    return combined, metadata


async def score_trajectory(
    issue_text: str,
    trajectory_summary: str,
    workspace_path: Optional[Path],
    language: str,
    cline_metadata: Optional[Dict[str, Any]] = None,
    files_modified: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Combined scoring using all three methods.
    
    Default weights: execution=0.3, llm_judge=0.6, complexity=0.1
    
    Returns:
        Tuple of (final_score 0.0-1.0, detailed metadata)
    """
    weights = weights or WEIGHTS
    metadata: Dict[str, Any] = {
        "weights": weights,
        "component_scores": {},
    }
    
    # 1. Execution score (syntax check)
    if workspace_path:
        exec_score, exec_meta = await score_execution(workspace_path, language)
    else:
        exec_score = 0.5
        exec_meta = {"skipped": "No workspace path provided"}
    metadata["component_scores"]["execution"] = {
        "score": exec_score,
        "details": exec_meta,
    }
    
    # 2. LLM judge score
    llm_score, llm_meta = await score_llm_judge(
        issue_text=issue_text,
        trajectory_summary=trajectory_summary,
        files_modified=files_modified or [],
        api_key=api_key,
    )
    metadata["component_scores"]["llm_judge"] = {
        "score": llm_score,
        "details": llm_meta,
    }
    
    # 3. Complexity score
    complexity_score, complexity_meta = score_complexity(cline_metadata)
    metadata["component_scores"]["complexity"] = {
        "score": complexity_score,
        "details": complexity_meta,
    }
    
    # Combine scores
    final_score = (
        weights["execution"] * exec_score +
        weights["llm_judge"] * llm_score +
        weights["complexity"] * complexity_score
    )
    
    metadata["final_score"] = final_score
    metadata["component_scores"]["execution"]["weighted"] = weights["execution"] * exec_score
    metadata["component_scores"]["llm_judge"]["weighted"] = weights["llm_judge"] * llm_score
    metadata["component_scores"]["complexity"]["weighted"] = weights["complexity"] * complexity_score
    
    logger.info(
        "Scored trajectory: exec=%.2f (%.2f), llm=%.2f (%.2f), complexity=%.2f (%.2f) -> final=%.2f",
        exec_score, weights["execution"] * exec_score,
        llm_score, weights["llm_judge"] * llm_score,
        complexity_score, weights["complexity"] * complexity_score,
        final_score,
    )
    
    return final_score, metadata


def extract_trajectory_summary(cline_metadata: Optional[Dict[str, Any]]) -> str:
    """Extract a text summary of the trajectory for LLM judging."""
    if not cline_metadata:
        return "No trajectory data available."
    
    ui_messages = cline_metadata.get("ui_messages", [])
    if not ui_messages:
        return "No UI messages recorded."
    
    parts = []
    for i, msg in enumerate(ui_messages[:50]):  # Limit to first 50 messages
        msg_type = msg.get("type") or msg.get("say") or "unknown"
        text = msg.get("text", "")[:500]  # Truncate long messages
        reasoning = msg.get("reasoning", "")[:300]
        
        if reasoning:
            parts.append(f"[{msg_type}] {reasoning}")
        if text:
            parts.append(f"  -> {text}")
    
    if len(ui_messages) > 50:
        parts.append(f"... and {len(ui_messages) - 50} more messages")
    
    return "\n".join(parts)


def extract_files_modified(cline_metadata: Optional[Dict[str, Any]]) -> List[str]:
    """Extract list of files that were modified from trajectory."""
    if not cline_metadata:
        return []
    
    files = set()
    ui_messages = cline_metadata.get("ui_messages", [])
    
    for msg in ui_messages:
        text = msg.get("text", "")
        # Look for file paths in tool use messages
        if any(pattern in text.lower() for pattern in ["write_to_file", "replace_in_file", "created", "modified"]):
            # Extract paths - this is a simple heuristic
            words = text.split()
            for word in words:
                if "/" in word and "." in word.split("/")[-1]:
                    # Looks like a file path
                    clean = word.strip("'\",:()[]{}")
                    if len(clean) < 200:  # Sanity check
                        files.add(clean)
    
    return list(files)[:30]  # Limit to 30 files
