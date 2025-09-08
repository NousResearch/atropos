#!/usr/bin/env python3
"""
Fixed Factorio agent using proper Gym environment initialization.
"""

import os
import sys

import gym

# Add FLE to path
fle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle")
sys.path.insert(0, fle_path)

import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fle  # Registers the environments
import requests
from fle.commons.models.game_state import GameState
from fle.env.gym_env.action import Action
from fle.env.gym_env.registry import get_environment_info, list_available_environments
from fle.env.tools import get_agent_tools


class LlamaFactorioAgent:
    """Agent that uses llama.cpp server to play Factorio"""

    def __init__(self, server_url="http://localhost:8080", task_goal=""):
        self.server_url = server_url
        self.api_url = f"{server_url}/v1/chat/completions"
        self.last_code = ""  # Track last executed code
        self.current_goals = []  # LLM's self-managed goal list
        self.task_goal = task_goal  # Main task objective
        self.tools_prompt = self._build_tools_prompt()

        # Check server connection
        try:
            requests.get(f"{server_url}/health", timeout=1)
            print(f"✅ Connected to llama.cpp server")
        except:
            print(f"⚠️ Could not connect to llama.cpp server")

        self.system_prompt = (
            f"You are an autonomous Factorio agent.\n\n"
            f"MAIN OBJECTIVE: {self.task_goal if self.task_goal else 'Create an AUTOMATIC factory that produces iron ore at 16/minute.'}\n\n"
            "MINING & CRAFTING SPEEDS:\n"
            "- Burner mining drill: 15 resources/minute\n"
            "- Electric mining drill: 30 resources/minute\n"
            "- Stone furnace: 18.75 plates/minute (1x speed)\n"
            "- Electric furnace: 37.5 plates/minute (2x speed)\n"
            "- Transport belt: moves 15 items/second\n\n"
            "GOAL MANAGEMENT:\n"
            "1. First, use update_goals to create your plan - break the objective into specific steps\n"
            "2. Update your goals as you complete them (remove finished ones)\n"
            "3. Revise goals when you discover new information\n"
            "4. Keep goals specific and actionable (e.g., 'Place drill at (15,70)' not 'Place drills')\n\n"
            "Starting inventory includes: 50 burner-mining-drills, 50 electric-mining-drills, "
            "500 transport-belts, 50 inserters, 10 chests, 500 coal, 10 furnaces, 500 electric poles.\n"
            "\nYou have access to tools. When you want to use one, emit exactly one XML block of the form:"
            "\n<tool_call>{'name': '<tool_name>', 'arguments': { ... }}</tool_call>\n"
            "\n\nIMPORTANT DISTINCTIONS:\n"
            "- RESOURCES are natural materials you harvest: Coal, IronOre, CopperOre, Stone, Wood (trees), Water, CrudeOil, UraniumOre\n"
            "- ENTITIES are built structures: TransportBelt, Inserter, AssemblingMachine1, StoneFurnace, IronChest, etc.\n"
            "- Use nearest(Resource.X) to find resources, get_entities(Prototype.Y) to find built structures\n"
            "- Tool results are always shown in the response (e.g., nearest returns 'Position(x=10, y=20)')\n"
            "\n\nAvailable Resources (use with nearest, harvest_resource, get_resource_patch):\n"
            "- Resource.Coal, Resource.IronOre, Resource.CopperOre, Resource.Stone\n"
            "- Resource.Wood (trees), Resource.Water, Resource.CrudeOil, Resource.UraniumOre\n"
            "\n\nCommon Prototypes (use with get_entities, place_entity):\n"
            "- Miners: Prototype.BurnerMiningDrill, Prototype.ElectricMiningDrill\n"
            "- Furnaces: Prototype.StoneFurnace, Prototype.SteelFurnace, Prototype.ElectricFurnace\n"
            "- Belts: Prototype.TransportBelt, Prototype.Splitter, Prototype.UndergroundBelt\n"
            "- Inserters: Prototype.Inserter, Prototype.LongHandedInserter, Prototype.FastInserter\n"
            "- Machines: Prototype.AssemblingMachine1, Prototype.Lab, Prototype.ChemicalPlant\n"
            "- Storage: Prototype.WoodenChest, Prototype.IronChest, Prototype.SteelChest\n"
            "- Power: Prototype.SmallElectricPole, Prototype.SteamEngine, Prototype.Boiler\n"
            "- Items: Prototype.IronPlate, Prototype.CopperPlate, Prototype.IronGearWheel, Prototype.ElectronicCircuit\n"
            "\n\nConventions:\n"
            "- Positions are objects: {'x': float, 'y': float} - use exact coordinates from nearest()\n"
            "- Resources/Prototypes can be bare names ('Wood', 'IronOre') or fully qualified ('Resource.Wood')\n"
            "- Parameters marked (optional) can be omitted - don't pass None for them\n"
            "- Keep each step to one tool call. If an error occurs, fix inputs and try again\n"
            "\nTools:\n" + self.tools_prompt
        )

    def _build_tools_prompt(self) -> str:
        """Auto-discover all agent tools and produce a schema + examples list."""
        # Start with meta-tools
        specs_lines = [
            "- {'name': 'update_goals', 'description': 'Update your goal list. Remove completed goals, add new ones.', 'arguments': {'goals': 'list of strings'}}"
        ]
        example_lines = [
            "<tool_call>{'name': 'update_goals', 'arguments': {'goals': ['Find iron ore', 'Place mining drills', 'Set up transport belts']}}</tool_call>"
        ]

        tools = get_agent_tools() or []
        base = Path(__file__).parent / "fle" / "fle" / "env" / "tools" / "agent"

        def parse_signature(
            tool: str,
        ) -> Tuple[List[Tuple[str, Optional[str], bool]], Optional[str]]:
            """Return list of (arg, annotation, optional) and a short signature string if extractable."""
            client = base / tool / "client.py"
            md = base / tool / "agent.md"
            args: List[Tuple[str, Optional[str], bool]] = []
            sig = None
            try:
                if client.exists():
                    tree = ast.parse(
                        client.read_text(encoding="utf-8", errors="ignore")
                    )
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            # find __call__
                            for fn in node.body:
                                if (
                                    isinstance(fn, ast.FunctionDef)
                                    and fn.name == "__call__"
                                ):
                                    # Collect args excluding self
                                    arg_nodes = fn.args.args[1:]  # skip self
                                    defaults = list(fn.args.defaults)
                                    # Align defaults to args (pad left)
                                    if defaults:
                                        pad = [None] * (len(arg_nodes) - len(defaults))
                                        defaults = pad + defaults
                                    else:
                                        defaults = [None] * len(arg_nodes)
                                    for a, d in zip(arg_nodes, defaults):
                                        ann = None
                                        if a.annotation is not None:
                                            try:
                                                ann = ast.unparse(a.annotation)
                                            except Exception:
                                                ann = None
                                        is_optional = (
                                            isinstance(d, ast.Constant)
                                            and d.value is None
                                        )
                                        args.append((a.arg, ann, is_optional))
                                    break
                            break
                if md.exists():
                    text = md.read_text(encoding="utf-8", errors="ignore")
                    start = text.find("```python")
                    if start == -1:
                        start = text.find("```")
                    if start != -1:
                        start = text.find("\n", start) + 1
                        end = text.find("```", start)
                        if end != -1:
                            line = text[start:end].strip().splitlines()
                            if line:
                                sig = line[0].strip()
            except Exception:
                pass
            return args, sig

        def arg_type_hint(ann: Optional[str]) -> str:
            if not ann:
                return "any"
            # simplify long annotations
            return re.sub(r"\s+", " ", ann)

        def extract_description(tool: str) -> str:
            md = base / tool / "agent.md"
            if not md.exists():
                return ""
            try:
                text = md.read_text(encoding="utf-8", errors="ignore").strip()
                # Trim to before first code fence
                fence = text.find("```")
                if fence != -1:
                    text = text[:fence]
                # Drop leading title line starting with '#'
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if lines and lines[0].startswith("#"):
                    lines = lines[1:]
                # Take the first non-heading paragraph
                para = []
                for ln in lines:
                    if ln.startswith("#"):
                        continue
                    para.append(ln)
                    if ln.endswith(".") and len(" ".join(para)) > 80:
                        break
                desc = " ".join(para).strip()
                # Shorten and sanitize quotes
                desc = re.sub(r"\s+", " ", desc)[:180]
                return desc.replace("'", "\u2019")
            except Exception:
                return ""

        for t in sorted(tools):
            params, sig = parse_signature(t)
            desc = extract_description(t)
            # Build arguments schema
            if params:
                arg_entries = []
                for name, ann, is_opt in params:
                    # Don't add ? to the key - just use the name
                    # We can indicate optional in the type hint instead
                    type_hint = arg_type_hint(ann)
                    if is_opt:
                        type_hint = f"{type_hint} (optional)"
                    arg_entries.append(f"'{name}': '{type_hint}'")
                arg_schema = "{" + ", ".join(arg_entries) + "}"
            else:
                arg_schema = "{}"
            if desc:
                specs_lines.append(
                    f"- {{'name': '{t}', 'description': '{desc}', 'arguments': {arg_schema}}}"
                )
            else:
                specs_lines.append(f"- {{'name': '{t}', 'arguments': {arg_schema}}}")

            # Example with real Resource/Prototype values
            example_args = {}
            for name, ann, is_opt in params[:3]:  # keep short
                lname = name.lower()
                if "pos" in lname:
                    example_args[name] = {"x": 5, "y": 5}
                elif "radius" in lname:
                    example_args[name] = 10
                elif "resource" in lname:
                    # Use real Resource examples
                    example_args[name] = (
                        "Coal" if t == "harvest_resource" else "IronOre"
                    )
                elif "prototype" in lname or "entity" in lname:
                    # Use real Prototype examples
                    example_args[name] = (
                        "TransportBelt" if "place" in t else "StoneFurnace"
                    )
                elif "type" in lname:
                    # Could be Resource or Prototype depending on tool
                    if t == "nearest":
                        example_args[name] = "Wood"  # Resource for nearest
                    else:
                        example_args[name] = "Inserter"  # Prototype for others
                elif any(k in lname for k in ["count", "quantity", "amount", "level"]):
                    example_args[name] = 1
                else:
                    # generic placeholder
                    example_args[name] = 0
            example_lines.append(
                "<tool_call>{'name': '"
                + t
                + "', 'arguments': "
                + json.dumps(example_args)
                + "}</tool_call>"
            )

        return "\n".join(specs_lines) + "\nExamples:\n" + "\n".join(example_lines[:8])

    def parse_observation(self, obs: Dict) -> str:
        """Summarize observation into a compact prompt for the LLM."""
        raw = (obs or {}).get("raw_text", "") or ""
        inv = obs.get("inventory", []) if obs else []
        entities = obs.get("entities", []) if obs else []
        score = obs.get("score", 0.0) if obs else 0.0
        game_info = obs.get("game_info", {}) if obs else {}
        research = obs.get("research", {}) if obs else {}
        task_verification = obs.get("task_verification", {}) if obs else {}

        # Start with current goals if any
        parts: List[str] = []
        if self.current_goals:
            parts.append("📋 Current goals:")
            for i, goal in enumerate(self.current_goals[:5]):  # Show first 5 goals
                parts.append(f"  {i+1}. {goal}")
            if len(self.current_goals) > 5:
                parts.append(f"  ... and {len(self.current_goals)-5} more")
        else:
            parts.append("📋 No goals set yet. Use update_goals to create your plan!")

        # Detect errors or syntax issues from raw_text
        error_markers = ("Traceback", "SyntaxError", "Exception:", "error:")
        had_error = any(m.lower() in raw.lower() for m in error_markers)
        if had_error:
            excerpt = raw.strip().splitlines()[-5:]
            parts.append("ERROR in last action:")
            parts.extend(excerpt)
            if self.last_code:
                parts.append(f"Last code: {self.last_code[:160]}")
        else:
            if raw.strip():
                parts.append("Last action result:")
                parts.append(raw.strip()[:400])

        # Inventory with better formatting
        inv_preview = []
        try:
            for item in inv[:8]:
                t = item.get("type")
                q = item.get("quantity")
                if t is not None and q is not None:
                    inv_preview.append(f"{t}:{q}")
        except Exception:
            pass

        if inv_preview:
            parts.append(f"Inventory: {', '.join(inv_preview)}")
        else:
            parts.append(
                "Inventory: Empty (use nearest(Resource.Wood) to find trees, or craft_item to make tools)"
            )

        # Entities with type information
        if entities:
            entity_types = {}
            try:
                for e in entities[:20]:  # Sample first 20
                    e_type = e.get("name", "unknown") if isinstance(e, dict) else str(e)
                    entity_types[e_type] = entity_types.get(e_type, 0) + 1
                entity_summary = ", ".join(
                    [f"{k}:{v}" for k, v in entity_types.items()]
                )
                parts.append(
                    f"Entities nearby ({len(entities)} total): {entity_summary}"
                )
            except:
                parts.append(f"Entities nearby: {len(entities)}")
        else:
            parts.append(
                "Entities nearby: None (this is normal at game start - use nearest to find resources)"
            )

        # Add player position if available
        if game_info and "player_position" in game_info:
            pos = game_info["player_position"]
            parts.append(f"Player position: x={pos.get('x', 0)}, y={pos.get('y', 0)}")

        # Add task progress/throughput if available
        if "current throughput" in raw.lower():
            # Extract throughput from raw text (e.g., "current throughput of your factory: 10 created per 60 seconds")
            import re

            match = re.search(r"current throughput.*?(\d+)\s+created", raw.lower())
            if match:
                parts.append(f"🏭 Current throughput: {match.group(1)} per minute")

        # Add research info if relevant
        if research and research.get("current_research"):
            current = research.get("current_research", "None")
            progress = research.get("research_progress", 0)
            if current != "None":
                parts.append(f"Research: {current} ({progress:.0%} complete)")

        parts.append(f"Score: {score}")

        return "\n".join(parts)

    def generate_tool_call(
        self, messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Ask the model for a <tool_call> and parse it into a dict."""
        # prefill a <think>\n token to the messages as an assistant message
        messages.append({"role": "assistant", "content": "<tool_call>"})
        data = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        try:
            resp = requests.post(self.api_url, json=data, timeout=60)
            result = resp.json()
            content = "<tool_call>" + result["choices"][0]["message"]["content"].strip()
            print(f"Agent response: {content}")
            # Extract between <tool_call>...</tool_call>
            start_tag, end_tag = "<tool_call>", "</tool_call>"
            if start_tag in content and end_tag in content:
                inner = content.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
            else:
                return None
            # pop the <tool_call> token prefill
            messages.pop()
            messages.append({"role": "assistant", "content": content})

            # Handle both JSON and Python dict-like
            import ast as _ast
            import json as _json

            parsed = None
            try:
                parsed = _json.loads(inner)
            except Exception:
                try:
                    parsed = _ast.literal_eval(inner)
                except Exception:
                    return None

            if not isinstance(parsed, dict) or "name" not in parsed:
                return None
            # Normalize
            call = {
                "name": str(parsed.get("name")),
                "arguments": parsed.get("arguments") or {},
            }
            if not isinstance(call["arguments"], dict):
                call["arguments"] = {}
            return call
        except Exception as e:
            print(f"Error generating tool call: {e}")
            input("Press Enter to continue...")
            return None

    def _enum_expr(self, enum_type: str, value: Union[str, int, None]) -> Optional[str]:
        if value is None:
            return None
        v = str(value)
        if "." in v:
            return v  # Assume already namespaced e.g., Resource.Wood
        return f"{enum_type}.{v}"

    def _pos_expr(self, val: Any) -> Optional[str]:
        try:
            if isinstance(val, dict) and "x" in val and "y" in val:
                return f"Position(x={int(val['x'])}, y={int(val['y'])})"
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return f"Position(x={int(val[0])}, y={int(val[1])})"
        except Exception:
            return None
        return None

    def update_goals(self, goals: List[str]) -> str:
        """Meta-tool for the LLM to manage its own goal list"""
        self.current_goals = goals
        if goals:
            return f"Goals updated successfully. Current goals:\n" + "\n".join(
                [f"{i+1}. {g}" for i, g in enumerate(goals)]
            )
        else:
            return "Goals cleared. Ready for new objectives."

    def tool_call_to_code(self, call: Dict[str, Any]) -> str:
        """Translate a tool_call dict to executable Python code string (generic)."""
        name = call.get("name", "")
        raw_args = call.get("arguments", {}) or {}

        # Special handling for meta-tools (not Factorio tools)
        if name == "update_goals":
            # This is handled directly, not as Python code
            goals = raw_args.get("goals", [])
            return f"__META__:update_goals:{json.dumps(goals)}"

        # Tools that expect Resources vs Prototypes
        resource_tools = {"nearest", "harvest_resource", "get_resource_patch"}
        prototype_tools = {
            "get_entities",
            "place_entity",
            "place_entity_next_to",
            "pickup_entity",
            "craft_item",
            "insert_item",
            "extract_item",
        }

        def render_value(key: str, val: Any) -> str:
            # Skip None values completely
            if val is None:
                return None  # Will be filtered out later

            # Position-like - keep float precision
            if isinstance(val, dict) and set(val.keys()) >= {"x", "y"}:
                try:
                    x = val["x"]
                    y = val["y"]
                    return f"Position(x={x}, y={y})"
                except Exception:
                    return "Position(x=0, y=0)"
            if isinstance(val, (list, tuple)) and len(val) == 2:
                try:
                    return f"Position(x={val[0]}, y={val[1]})"
                except Exception:
                    return "Position(x=0, y=0)"

            # Enum-like strings
            if isinstance(val, str):
                if re.match(r"^[A-Za-z_]+\.[A-Za-z_]+$", val):
                    return val  # already qualified

                # Smart detection based on tool name and parameter
                if "resource" in key.lower() or name in resource_tools:
                    # This is a Resource
                    return f"Resource.{val}"
                elif (
                    "prototype" in key.lower()
                    or "entity" in key.lower()
                    or name in prototype_tools
                ):
                    # This is a Prototype
                    return f"Prototype.{val}"
                elif "direction" in key.lower():
                    return f"Direction.{val}"
                elif "type" in key.lower():
                    # Type could be either - check the tool
                    if name in resource_tools:
                        return f"Resource.{val}"
                    elif name in prototype_tools:
                        return f"Prototype.{val}"
                    else:
                        # Default to Prototype for entities
                        return f"Prototype.{val}"

                # generic string
                return repr(val)

            # Numbers and booleans
            if isinstance(val, (int, float)):
                return str(val)
            if isinstance(val, bool):
                return "True" if val else "False"

            # Fallback
            return repr(val)

        # Build the base function call
        if not raw_args:
            func_call = f"{name}()"
        else:
            parts = []
            for k, v in raw_args.items():
                rendered = render_value(k, v)
                if rendered is not None:  # Skip None values
                    parts.append(f"{k}={rendered}")
            func_call = f"{name}({', '.join(parts)})"

        # Always wrap with print() to capture any return values
        # Tools that don't return anything will just print None or empty
        return f"print({func_call})"


def main():
    print("🎮 Starting Factorio Gym Agent\n")

    # List environments
    env_ids = list_available_environments()
    print(f"Found {len(env_ids)} environments")

    # Use iron_ore_throughput for structured task training
    # Other good starter tasks: iron_plate_throughput, copper_ore_throughput
    env_id = None
    for eid in env_ids:
        if eid == "iron_ore_throughput":  # Start with simplest mining task
            env_id = eid
            break

    if not env_id:
        # Fallback to any throughput task
        for eid in env_ids:
            if "throughput" in eid.lower() and "unbounded" not in eid.lower():
                env_id = eid
                break

    if not env_id and env_ids:
        env_id = env_ids[0]

    if not env_id:
        print("❌ No environments available!")
        return

    print(f"Using environment: {env_id}\n")

    # Create environment directly without gym.make()
    # The gym.make() is hanging because it tries to create a new instance
    # We need to create the env directly with the existing connection
    from fle.env import FactorioInstance
    from fle.env.gym_env.environment import FactorioGymEnv
    from fle.eval.tasks import TaskFactory

    try:
        # Get task info
        info = get_environment_info(env_id)
        task_path = info["task_config_path"]

        # Create task
        task = TaskFactory.create_task(task_path)

        # Use existing connection
        instance = FactorioInstance(
            address="localhost", tcp_port=27000, fast=True, num_agents=1
        )

        # Setup task
        task.setup(instance)

        # Create gym environment
        env = FactorioGymEnv(instance=instance, task=task)
        print("✅ Environment created\n")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback

        traceback.print_exc()
        return

    # Get task goal from environment info
    task_goal = (
        "Create an automatic iron ore factory that produces 16 iron ore per 60 seconds"
    )

    # Create agent with task goal
    agent = LlamaFactorioAgent(task_goal=task_goal)

    try:
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset(options={"game_state": None})
        print("✅ Environment reset\n")

        # Initialize chat history with system prompt
        history: List[Dict[str, str]] = [
            {"role": "system", "content": agent.system_prompt}
        ]

        # Run for more steps to allow factory building
        for step in range(15):  # More steps for planning and building
            print(f"{'='*50}\nStep {step + 1}\n")

            # Print just the important parts
            if step == 0:
                print(f"Raw text: {obs.get('raw_text', 'None')[:100]}")
                print(f"Entities: {len(obs.get('entities', []))}")
                print(f"Inventory: {obs.get('inventory', [])}")

            # Parse observation and ask for a tool_call
            obs_text = agent.parse_observation(obs)
            print(f"📊 {obs_text}")

            # Different prompt for first step - encourage planning
            if step == 0:
                user_msg = (
                    f"Observation:\n{obs_text}\n"
                    f"Main objective: {task_goal}\n"
                    "Start by creating your goals with update_goals, then work on the first goal.\n"
                    "Respond with a single <tool_call>{{...}}</tool_call> only."
                )
            else:
                user_msg = (
                    f"Observation:\n{obs_text}\n"
                    "Continue working on your goals. Update them as needed.\n"
                    "Respond with a single <tool_call>{{...}}</tool_call> only."
                )
            history.append({"role": "user", "content": user_msg})

            tool_call = agent.generate_tool_call(history)
            if not tool_call:
                print(
                    "⚠️ Model did not return a tool_call; defaulting to inspect_inventory."
                )
                tool_call = {"name": "inspect_inventory", "arguments": {}}

            print(f"🛠️ Tool call: {tool_call}")
            code = agent.tool_call_to_code(tool_call)
            agent.last_code = code
            print(f"💻 Translated code: {code}\n")

            # Handle meta-tools separately
            if code.startswith("__META__:"):
                # Parse meta-tool call
                parts = code.split(":", 2)
                if len(parts) >= 3 and parts[1] == "update_goals":
                    goals = json.loads(parts[2])
                    result = agent.update_goals(goals)
                    print(f"📝 {result}")
                    # Don't execute as Factorio code, just update observation
                    raw_text = result
                    reward = 0.0
                    info = {}
                else:
                    print(f"❌ Unknown meta-tool: {parts[1]}")
                    raw_text = "Error: Unknown meta-tool"
                    reward = 0.0
                    info = {"error_occurred": True}
            else:
                # Execute as gym code
                current_game_state = GameState.from_instance(env.instance)
                action = Action(agent_idx=0, game_state=current_game_state, code=code)

                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"✅ Reward: {reward:.2f}")

                    # Debug: Show what the tool returned
                    raw_result = (
                        obs.get("raw_text", "") if isinstance(obs, dict) else ""
                    )

                    # Special handling for nearest - it returns Position but raw_text is empty
                    if (
                        tool_call.get("name") == "nearest"
                        and not raw_result
                        and reward >= 0
                    ):
                        # Try to extract position from the last action's result
                        # The Position is likely returned but not in raw_text
                        # For now, we'll need to work around this
                        raw_result = (
                            f"Position found (check game state for coordinates)"
                        )

                    if raw_result:
                        print(f"📝 Tool returned: {raw_result[:200]}")

                    raw_text = raw_result

                except Exception as e:
                    print(f"❌ Error executing code: {e}")
                    info = {"error_occurred": True, "result": str(e)}
                    obs = {}
                    raw_text = str(e)
                    reward = -10.0

            # Provide tool_response back to the model
            if not code.startswith("__META__:"):
                raw_text = (info.get("result") if isinstance(info, dict) else None) or (
                    obs.get("raw_text") if isinstance(obs, dict) else ""
                )
            try:
                inv_preview = []
                for item in (obs.get("inventory") or [])[:5]:
                    t = item.get("type")
                    q = item.get("quantity")
                    if t is not None and q is not None:
                        # Convert numpy types to regular Python types
                        if hasattr(q, "item"):  # numpy type
                            q = q.item()
                        inv_preview.append({"type": t, "quantity": q})
            except Exception:
                inv_preview = []

            tool_response_payload = {
                "name": tool_call.get("name"),
                "content": {
                    "raw_text": raw_text,
                    "error": (
                        bool(info.get("error_occurred"))
                        if isinstance(info, dict)
                        else False
                    ),
                    "inventory": inv_preview,
                },
            }

            # Debug: Show what we're sending back to the model (skip if JSON fails)
            try:
                print(f"🔄 Sending to model: {json.dumps(tool_response_payload)[:300]}")
            except (TypeError, ValueError):
                # Skip debug print if numpy types present
                pass

            history.append(
                {
                    "role": "tool",
                    "content": "<tool_response>"
                    + json.dumps(tool_response_payload)
                    + "</tool_response>",
                }
            )

            if terminated or truncated:
                print("🏁 Done!")
                break

            time.sleep(0.5)

    finally:
        env.close()
        print("\n🛑 Environment closed")


if __name__ == "__main__":
    main()
