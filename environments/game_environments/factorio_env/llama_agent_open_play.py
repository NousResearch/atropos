#!/usr/bin/env python3
"""
Demo script for use with Factorio client
Open play mode, no task goals beyond "build giant factory"
"""

import os
import sys

# import gym  # Unused import

# Add FLE to path
fle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle")
sys.path.insert(0, fle_path)

import ast  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: E402

import fle  # noqa: F401,E402  # Registers the environments
import requests  # noqa: E402
from fle.commons.models.game_state import GameState  # noqa: E402
from fle.env.gym_env.action import Action  # noqa: E402
from fle.env.gym_env.registry import (  # noqa: E402
    get_environment_info,
    list_available_environments,
)
from fle.env.tools import get_agent_tools  # noqa: E402


class LlamaFactorioAgent:
    """Agent that uses llama.cpp server to play Factorio"""

    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.api_url = f"{server_url}/v1/chat/completions"
        self.last_code = ""  # Track last executed code
        self.tools_prompt = self._build_tools_prompt()

        # Check server connection
        try:
            requests.get(f"{server_url}/health", timeout=1)
            print("✅ Connected to llama.cpp server")
        except Exception:
            print("⚠️ Could not connect to llama.cpp server")

        self.system_prompt = (
            "You are an autonomous Factorio agent in open play."
            # "You can think through your moves, but MUST contain all thinking inside a <think>...</think> block."
            "\nYou have access to tools. When you want to use one, emit exactly one XML block of the form:"
            "\n<tool_call>{'name': '<tool_name>', 'arguments': { ... }}</tool_call>\n"
            # "\nThe <tool_call> block MUST follow the <think>...</think> block."
            "\n\nIMPORTANT DISTINCTIONS:\n"
            "- RESOURCES are natural materials you harvest: Coal, IronOre, CopperOre, Stone, "
            "Wood (trees), Water, CrudeOil, UraniumOre\n"
            "- ENTITIES are built structures: TransportBelt, Inserter, AssemblingMachine1, "
            "StoneFurnace, IronChest, etc.\n"
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
            "- Items: Prototype.IronPlate, Prototype.CopperPlate, Prototype.IronGearWheel, "
            "Prototype.ElectronicCircuit\n"
            "\n\nConventions:\n"
            "- Positions are objects: {'x': float, 'y': float} - use exact coordinates from nearest()\n"
            "- Resources/Prototypes can be bare names ('Wood', 'IronOre') or fully qualified ('Resource.Wood')\n"
            "- Parameters marked (optional) can be omitted - don't pass None for them\n"
            "- Keep each step to one tool call. If an error occurs, fix inputs and try again\n"
            "\nTools:\n" + self.tools_prompt
        )

    def _build_tools_prompt(self) -> str:
        """Auto-discover all agent tools and produce a schema + examples list."""
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

        specs_lines: List[str] = []
        example_lines: List[str] = []
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

        # Detect errors or syntax issues from raw_text
        error_markers = ("Traceback", "SyntaxError", "Exception:", "error:")
        had_error = any(m.lower() in raw.lower() for m in error_markers)

        parts: List[str] = []
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
            except Exception:
                parts.append(f"Entities nearby: {len(entities)}")
        else:
            parts.append(
                "Entities nearby: None (this is normal at game start - use nearest to find resources)"
            )

        # Add player position if available
        if game_info and "player_position" in game_info:
            pos = game_info["player_position"]
            parts.append(f"Player position: x={pos.get('x', 0)}, y={pos.get('y', 0)}")

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

    def tool_call_to_code(self, call: Dict[str, Any]) -> str:
        """Translate a tool_call dict to executable Python code string (generic)."""
        name = call.get("name", "")
        raw_args = call.get("arguments", {}) or {}

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
        return f"print({func_call})"


def main():
    print("🎮 Starting Factorio Gym Agent\n")

    # List environments
    env_ids = list_available_environments()
    print(f"Found {len(env_ids)} environments")

    # Use the open_play environment which is simplest
    env_id = None
    for eid in env_ids:
        if "open_play" in eid.lower():
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

    # Create agent
    agent = LlamaFactorioAgent()

    try:
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset(options={"game_state": None})
        print("✅ Environment reset\n")

        # Simple goals - change em to whatever you want
        goals = [
            "Find the nearest trees (Wood resource) and tell me the position",
            "Move to the position of the trees you just found",
            "Harvest 10 wood from the trees",
            "Check your inventory to see the wood",
        ]

        # Initialize chat history with system prompt
        history: List[Dict[str, str]] = [
            {"role": "system", "content": agent.system_prompt}
        ]

        # Run for a few steps
        for step in range(4):
            print(f"{'='*50}\nStep {step + 1}\n")

            goal = goals[step % len(goals)]
            print(f"🎯 {goal}")

            # Print just the important parts
            if step == 0:
                print(f"Raw text: {obs.get('raw_text', 'None')[:100]}")
                print(f"Entities: {len(obs.get('entities', []))}")
                print(f"Inventory: {obs.get('inventory', [])}")

            # Parse observation and ask for a tool_call
            obs_text = agent.parse_observation(obs)
            print(f"📊 {obs_text}")

            user_msg = (
                f"Observation:\n{obs_text}\nGoal: {goal}\n"
                "Respond with a single <tool_call>{...}</tool_call> only."
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

            # Execute as gym code
            current_game_state = GameState.from_instance(env.instance)
            action = Action(agent_idx=0, game_state=current_game_state, code=code)

            try:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"✅ Reward: {reward:.2f}")

                # Debug: Show what the tool returned
                raw_result = obs.get("raw_text", "") if isinstance(obs, dict) else ""

                # Special handling for nearest - it returns Position
                if (
                    tool_call.get("name") == "nearest"
                    and not raw_result
                    and reward >= 0
                ):
                    raw_result = "Position found (check game state for coordinates)"

                if raw_result:
                    print(f"📝 Tool returned: {raw_result[:200]}")

            except Exception as e:
                print(f"❌ Error executing code: {e}")
                info = {"error_occurred": True, "result": str(e)}
                # Keep obs as-is

            # Provide tool_response back to the model
            raw_text = (info.get("result") if isinstance(info, dict) else None) or (
                obs.get("raw_text") if isinstance(obs, dict) else ""
            )
            try:
                inv_preview = []
                for item in (obs.get("inventory") or [])[:5]:
                    t = item.get("type")
                    q = item.get("quantity")
                    if t is not None and q is not None:
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

            # Debug: Show what we're sending back to the model
            print(f"🔄 Sending to model: {json.dumps(tool_response_payload)[:300]}")

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
