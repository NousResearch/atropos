#!/usr/bin/env python3
"""
Minimal Factorio Environment for Atropos

A simplified integration of Factorio Learning Environment (FLE) with Atropos RL trainer.
Focuses on:
- Single Docker instance (group_size=1 initially)
- Throughput tasks with clear objectives
- LLM-based agent with self-planning capabilities
- Simple reward based on task completion
"""

import ast
import asyncio
import contextlib
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Add FLE to path
fle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle")
sys.path.insert(0, fle_path)


import fle  # noqa: F401,E402  # Registers the environments
from fle.commons.models.game_state import GameState  # noqa: E402
from fle.env import FactorioInstance  # noqa: E402
from fle.env.gym_env.action import Action  # noqa: E402
from fle.env.gym_env.environment import FactorioGymEnv  # noqa: E402
from fle.env.gym_env.registry import (  # noqa: E402
    get_environment_info,
    list_available_environments,
)
from fle.env.tools import get_agent_tools  # noqa: E402
from fle.eval.tasks import TaskFactory  # noqa: E402

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
try:
    from factorio_rcon import (
        RCONClient,
    )  # lightweight RCON ping without resetting worlds
except Exception:  # pragma: no cover
    RCONClient = None  # type: ignore

logger = logging.getLogger(__name__)


class FactorioEnvConfig(BaseEnvConfig):
    """Configuration for the minimal Factorio environment."""

    env_name: str = "factorio_minimal"
    wandb_name: str = "factorio-trainer-minimal"

    # Task settings
    task_names: List[str] = [
        "iron_ore_throughput",
        "copper_ore_throughput",
        "iron_plate_throughput",
        "automation_science_pack_throughput",
    ]
    randomize_task_selection: bool = True
    max_steps_per_episode: int = 50  # Limit episode length

    # Factorio server settings
    factorio_host: str = "localhost"
    # Base RCON port for a locally started cluster (factorio_0 -> 27000, factorio_1 -> 27001, ...)
    factorio_tcp_port_base: int = 27000
    # For legacy single-instance runs; ignored when group_size > 1
    factorio_tcp_port: int = 27000
    factorio_fast_mode: bool = True  # Run game faster for training

    # Agent settings
    enable_self_planning: bool = True  # Allow agent to use update_goals
    max_goals: int = 10  # Maximum number of self-managed goals

    # Training settings
    group_size: int = 1  # Start with single trajectory (scale later)
    max_num_workers: int = 1  # Single Docker instance for now
    total_steps: int = 100
    max_token_length: int = 32768

    # Scoring weights
    task_completion_weight: float = 10.0
    throughput_weight: float = 1.0
    efficiency_weight: float = 0.1  # Reward for fewer steps

    # Monitoring and preflight
    factorio_total_servers: int = 1  # Number of locally running server containers
    resource_log_interval_seconds: int = 15
    enable_resource_logging: bool = True
    preflight_timeout_seconds: int = 60


class FactorioEnv(BaseEnv):
    """Minimal Factorio environment for training LLMs."""

    name = "factorio_minimal"
    env_config_cls = FactorioEnvConfig

    def __init__(
        self,
        config: FactorioEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: FactorioEnvConfig = config

        # Agent components (adapted from llama_agent.py)
        self.current_goals: List[str] = []

        # Metrics tracking
        self.episode_outcomes_buffer = []
        self.episode_rewards_buffer = []
        self.episode_steps_buffer = []
        self.episode_task_types = []
        self.eval_metrics_custom = []

        self.tools_prompt = self._build_tools_prompt()
        # System prompt will be built per-trajectory with specific task goal

        # Port allocation for multi-worker scaling
        self._port_lock = asyncio.Lock()
        self._ports_in_use: set[int] = set()

        # Resource logger task
        self._resource_logger_task: Optional[asyncio.Task] = None
        # Per-group aggregate metrics buffer
        self._group_metrics_buffer: List[Dict[str, float]] = []

    def _build_tools_prompt(self) -> str:
        """Auto-discover all agent tools and produce a schema + examples list."""
        # Start with meta-tools
        specs_lines = [
            (
                "- {'name': 'update_goals', "
                "'description': 'Update your goal list. Remove completed goals, add new ones.', "
                "'arguments': {'goals': 'list of strings'}}"
            )
        ]
        example_lines = [
            (
                "<tool_call>{'name': 'update_goals', 'arguments': {'goals': "
                "['Find iron ore', 'Place mining drills', 'Set up transport belts']}}</tool_call>"
            )
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

    def _build_system_prompt(self, task_goal: str) -> str:
        """Build the system prompt for the LLM agent."""
        system_prompt = (
            f"You are an autonomous Factorio agent.\n\n"
            f"MAIN OBJECTIVE: {task_goal}\n\n"
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
            "500 transport-belts, 50 inserters, 10 chests, 500 coal, 10 furnaces, "
            "500 electric poles.\n"
            "\nYou have access to tools. When you want to use one, emit exactly one XML block of the form:"
            "\n<tool_call>{'name': '<tool_name>', 'arguments': { ... }}</tool_call>\n"
            "\n\nIMPORTANT DISTINCTIONS:\n"
            "- RESOURCES are natural materials you harvest: Coal, IronOre, CopperOre, Stone, Wood (trees), "
            "Water, CrudeOil, UraniumOre\n"
            "- ENTITIES are built structures: TransportBelt, Inserter, AssemblingMachine1, StoneFurnace, "
            "IronChest, etc.\n"
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

        return system_prompt

    async def setup(self):
        """Initialize the environment (but don't connect to server yet)."""
        logger.info(
            (
                f"setup: env={self.name} base_port={self.config.factorio_tcp_port_base} "
                f"total_servers={self.config.factorio_total_servers} group_size={self.config.group_size} "
                f"max_workers={self.config.max_num_workers}"
            )
        )

        try:
            # Just verify environments are available, don't connect yet
            env_ids = list_available_environments()
            logger.info(f"Found {len(env_ids)} FLE environments available")
            if self.config.group_size > self.config.factorio_total_servers:
                logger.warning(
                    (
                        f"setup: group_size ({self.config.group_size}) exceeds "
                        f"factorio_total_servers ({self.config.factorio_total_servers}). "
                        "Episodes will block waiting for free servers. Consider increasing the cluster size."
                    )
                )
            # Preflight RCON connectivity and capacity vs demand
            await self._preflight_check()

            # Start resource logger
            if (
                self.config.enable_resource_logging
                and self._resource_logger_task is None
            ):
                self._resource_logger_task = asyncio.create_task(
                    self._resource_logger()
                )

        except Exception as e:
            logger.error(f"Failed to setup Factorio environment: {e}")
            logger.error(traceback.format_exc())
            raise

    async def get_next_item(self) -> Item:
        """Get the next task configuration."""
        # Select task
        if self.config.randomize_task_selection:
            task_name = random.choice(self.config.task_names)
        else:
            task_name = self.config.task_names[0]
        item = {
            "task_name": task_name,
            "seed": random.randint(0, 1_000_000),
            "episode_id": f"ep-{int(time.time())}-{random.randint(1000, 9999)}",
        }
        logger.info(
            f"get_next_item: prepared episode_id={item['episode_id']} task={item['task_name']}"
        )
        return item

    async def _preflight_check(self):
        """Check RCON connectivity across the configured port range and log capacity vs demand."""
        base = int(self.config.factorio_tcp_port_base)
        total = int(self.config.factorio_total_servers)
        required = max(1, int(self.config.group_size)) * max(
            1, int(self.config.max_num_workers)
        )
        ports = [base + i for i in range(total)]
        reachable: List[int] = []
        unreachable: List[int] = []
        start_time = time.time()
        if RCONClient is None:
            logger.warning(
                "preflight: factorio_rcon not available; skipping RCON connectivity test"
            )
        else:
            for p in ports:
                try:
                    client = RCONClient(self.config.factorio_host, p, "factorio")
                    client.connect()
                    with contextlib.suppress(Exception):
                        client.send_command("/sc rcon.print('ok')")
                    client.close()
                    reachable.append(p)
                except Exception:
                    unreachable.append(p)
        elapsed = time.time() - start_time
        logger.info(
            f"preflight: reachable={reachable} unreachable={unreachable} elapsed={elapsed:.2f}s"
        )
        if total < required:
            logger.warning(
                (
                    f"preflight: capacity shortfall. servers={total}, required={required} "
                    "(group_size*max_workers). Increase factorio_total_servers or reduce concurrency."
                )
            )
        elif len(reachable) < min(total, required):
            logger.warning(
                (
                    f"preflight: only {len(reachable)} ports reachable out of "
                    f"{min(total, required)} needed for peak. Episodes may queue."
                )
            )

    async def _resource_logger(self):
        """Periodically log system resource usage for tuning."""
        interval = max(5, int(self.config.resource_log_interval_seconds))
        while True:
            try:
                # Ports in use snapshot
                async with self._port_lock:
                    in_use = sorted(list(self._ports_in_use))
                    num_in_use = len(in_use)

                # CPU/Mem usage (best-effort)
                cpu_pct = mem_pct = rss_mb = -1.0
                if psutil is not None:
                    try:
                        cpu_pct = psutil.cpu_percent(interval=0.0)
                        vm = psutil.virtual_memory()
                        mem_pct = float(vm.percent)
                        proc = psutil.Process(os.getpid())
                        rss_mb = float(proc.memory_info().rss) / (1024 * 1024)
                    except Exception:
                        pass

                logger.info(
                    (
                        f"resource: ports_in_use={in_use} n_in_use={num_in_use} cpu%={cpu_pct:.1f} "
                        f"mem%={mem_pct:.1f} rss_mb={rss_mb:.1f}"
                    )
                )
                # Optional wandb logging
                try:
                    if self.config.use_wandb:
                        import wandb  # local import to avoid hard dep when disabled

                        wandb.log(
                            {
                                "resource/ports_in_use": num_in_use,
                                "resource/cpu_percent": cpu_pct,
                                "resource/mem_percent": mem_pct,
                                "resource/rss_mb": rss_mb,
                            }
                        )
                except Exception:
                    pass
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"resource logger error: {e}")
                await asyncio.sleep(interval)

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """Collect trajectories from parallel games (currently just one)."""
        task_name = item["task_name"]
        # seed = item["seed"]  # Unused variable
        episode_id = item["episode_id"]

        logger.info(
            f"collect_trajectories: episode_id={episode_id} task={task_name} group_size={self.config.group_size}"
        )

        # Parallel rollouts: clone identical initial GameState across instances
        group_size = max(1, int(self.config.group_size))
        ports = await self._reserve_ports(group_size)
        logger.info(
            (
                f"collect_trajectories: episode_id={episode_id} reserved_ports={ports} "
                f"base={self.config.factorio_tcp_port_base}"
            )
        )
        scored_items: List[ScoredDataItem] = []

        # 1) Build the task once
        env_info = get_environment_info(task_name)
        task_path = env_info["task_config_path"]
        task = TaskFactory.create_task(task_path)

        # 2) Prepare a template starting state using the first available server
        template_state = await self._prepare_template_state(
            task=task,
            tcp_port=ports[0],
        )

        # 3) Launch group_size identical rollouts in parallel on distinct servers
        async def run_one(i: int):
            try:
                # Use the same seed for true clones; if you want diversity, alter it externally
                return await self._run_single_trajectory(
                    task=task,
                    task_name=task_name,
                    trajectory_id=f"{episode_id}-{i}",
                    tcp_port=ports[i],
                    initial_state=template_state,
                )
            except Exception as e:
                logger.error(f"Error collecting trajectory {i}: {e}")
                logger.error(traceback.format_exc())
                return None

        try:
            results = await asyncio.gather(*[run_one(i) for i in range(group_size)])
            scored_items = [r for r in results if r]
        finally:
            # Release ports for this episode
            await self._release_ports(ports)
            logger.info(
                f"collect_trajectories: episode_id={episode_id} releasing_ports={ports}"
            )

        # Build ScoredDataGroup
        if not scored_items:
            return (
                ScoredDataGroup(
                    tokens=[],
                    masks=[],
                    scores=[],
                    messages=[],
                    advantages=None,
                    ref_logprobs=None,
                    group_overrides={},
                    overrides=None,
                    images=None,
                ),
                [],
            )

        sdg = ScoredDataGroup(
            tokens=[],
            masks=[],
            scores=[],
            messages=[],
            advantages=None,
            ref_logprobs=None,
            group_overrides={},
            overrides=None,
            images=None,
        )

        for scored_item in scored_items:
            sdg["tokens"].append(scored_item["tokens"])
            sdg["masks"].append(scored_item["masks"])
            sdg["scores"].append(scored_item["scores"])
            if self.config.include_messages and scored_item.get("messages"):
                sdg["messages"].append(scored_item["messages"])

            # Track metrics
            metadata = scored_item.get("metadata", {})
            self.episode_outcomes_buffer.append(metadata.get("task_completed", False))
            self.episode_rewards_buffer.append(metadata.get("total_reward", 0))
            self.episode_steps_buffer.append(metadata.get("steps", 0))
            self.episode_task_types.append(task_name)
        # Per-group aggregates (since all scored_items are one group)
        try:
            rewards = [
                si.get("metadata", {}).get("total_reward", 0.0) for si in scored_items
            ]
            steps = [si.get("metadata", {}).get("steps", 0) for si in scored_items]
            completes = [
                1.0 if si.get("metadata", {}).get("task_completed", False) else 0.0
                for si in scored_items
            ]
            if rewards:
                self._group_metrics_buffer.append(
                    {
                        "avg_reward": float(sum(rewards) / len(rewards)),
                        "avg_steps": float(sum(steps) / len(steps)) if steps else 0.0,
                        "completion_rate": (
                            float(sum(completes) / len(completes)) if completes else 0.0
                        ),
                        "group_size": float(len(scored_items)),
                    }
                )
        except Exception:
            logger.warning("Failed to compute per-group metrics for wandb logging")

        logger.warning(f"Collected {len(scored_items)} trajectories")

        return sdg, []

    async def _reserve_ports(self, n: int) -> List[int]:
        """Reserve n distinct RCON ports from the configured cluster pool.
        Waits until enough free ports are available. Logs allocations to aid debugging.
        """
        if n <= 1 and self.config.factorio_total_servers <= 1:
            return [self.config.factorio_tcp_port]

        base = int(self.config.factorio_tcp_port_base)
        total = int(self.config.factorio_total_servers)
        deadline = time.time() + 300  # 5 minute safety timeout
        while True:
            async with self._port_lock:
                free_ports = [
                    p for p in range(base, base + total) if p not in self._ports_in_use
                ]
                if len(free_ports) >= n:
                    chosen = free_ports[:n]
                    self._ports_in_use.update(chosen)
                    logger.info(
                        (
                            f"port_allocator: reserved ports={chosen} "
                            f"in_use={sorted(list(self._ports_in_use))}"
                        )
                    )
                    return chosen
                else:
                    logger.warning(
                        (
                            f"port_allocator: waiting for {n} ports; free={free_ports}, "
                            f"in_use={sorted(list(self._ports_in_use))}"
                        )
                    )
            if time.time() > deadline:
                raise TimeoutError(
                    f"Timed out reserving {n} ports from pool base={base} total={total}"
                )
            await asyncio.sleep(1.0)

    async def _release_ports(self, ports: List[int]):
        async with self._port_lock:
            for p in ports:
                self._ports_in_use.discard(p)
            logger.info(
                (
                    f"port_allocator: released ports={ports} "
                    f"now_in_use={sorted(list(self._ports_in_use))}"
                )
            )

    async def _prepare_template_state(self, task: Any, tcp_port: int) -> GameState:
        """Create a Factorio instance, run task.setup() to produce a canonical starting GameState, then close it."""
        loop = asyncio.get_event_loop()
        instance = await loop.run_in_executor(
            None,
            lambda: FactorioInstance(
                address=self.config.factorio_host,
                tcp_port=tcp_port,
                fast=self.config.factorio_fast_mode,
                num_agents=1,
            ),
        )
        try:
            # Let the task configure inventory/tech and capture its starting state
            task.setup(instance)
            # TaskABC.setup stores starting_game_state; prefer that if present
            if getattr(task, "starting_game_state", None) is not None:
                template = task.starting_game_state
            else:
                template = GameState.from_instance(instance)
            return template
        finally:
            try:
                instance.cleanup()
            except Exception:
                pass

    async def _run_single_trajectory(
        self,
        task: Any,
        task_name: str,
        trajectory_id: str,
        tcp_port: int,
        initial_state: Optional[GameState] = None,
    ) -> Optional[ScoredDataItem]:
        """Run a single trajectory with the LLM agent."""
        messages: List[Message] = []

        try:
            # Create NEW instance for this trajectory (like llama_agent.py does)
            logger.warning(f"Creating FactorioInstance for trajectory {trajectory_id}")

            # Run blocking FactorioInstance creation in executor to avoid blocking async loop
            import asyncio

            loop = asyncio.get_event_loop()
            logger.info(
                (
                    f"_run_single_trajectory: trajectory_id={trajectory_id} connecting "
                    f"rcon={self.config.factorio_host}:{tcp_port}"
                )
            )
            instance = await loop.run_in_executor(
                None,
                lambda: FactorioInstance(
                    address=self.config.factorio_host,
                    tcp_port=tcp_port,
                    fast=self.config.factorio_fast_mode,
                    num_agents=1,
                ),
            )
            logger.warning(
                f"_run_single_trajectory: trajectory_id={trajectory_id} FactorioInstance created"
            )

            # Create gym environment
            env = FactorioGymEnv(instance=instance, task=task)

            # Reset environment
            # If provided, clone the exact starting GameState; otherwise default reset
            opts = (
                {"game_state": initial_state}
                if initial_state is not None
                else {"game_state": None}
            )
            obs, info = env.reset(options=opts)
            logger.warning(
                (
                    f"_run_single_trajectory: trajectory_id={trajectory_id} reset ok, "
                    f"info_keys={list(info.keys()) if info else []}"
                )
            )

            # Initialize conversation with task-specific system prompt
            task_goal = (
                task.goal_description
                if hasattr(task, "goal_description")
                else "Complete the task"
            )
            system_prompt = self._build_system_prompt(task_goal)
            messages.append({"role": "system", "content": system_prompt})
            logger.warning(
                (
                    f"_run_single_trajectory: trajectory_id={trajectory_id} start task "
                    f"goal='{task_goal[:80]}...' port={tcp_port}"
                )
            )

            # Reset agent goals
            self.current_goals = []

            done = False
            total_reward = 0.0
            steps = 0

            async with self.server.dedicated_server() as server:
                while not done and steps < self.config.max_steps_per_episode:
                    # Check token limit
                    # Format observation
                    obs_text = self._format_observation(obs, info)
                    messages.append({"role": "user", "content": obs_text})
                    logger.debug(f"Step {steps+1} - Observation: {obs_text[:200]}...")
                    current_tokens = len(
                        self.tokenizer.apply_chat_template(messages, tokenize=True)
                    )
                    if current_tokens > self.config.max_token_length - 500:
                        logger.warning(
                            f"Trajectory {trajectory_id}: Approaching token limit"
                        )
                        break

                    messages.append({"role": "assistant", "content": "<tool_call>"})
                    # Get LLM action
                    try:
                        response = await server.chat_completion(
                            messages=messages,
                            n=1,
                            max_tokens=300,
                            temperature=0.7,
                        )
                        action_text = (
                            "<tool_call>" + response.choices[0].message.content.strip()
                        )
                        logger.info(
                            f"Step {steps+1} - LLM response: {action_text[:150]}..."
                        )
                    except Exception as e:
                        logger.error(f"LLM error: {e}")
                        break

                    messages.pop()
                    messages.append({"role": "assistant", "content": action_text})

                    # Parse and execute action
                    code = self._parse_action_to_code(action_text)
                    logger.debug(f"Step {steps+1} - Parsed code: {code}")

                    if code.startswith("__META__:"):
                        # Handle meta-tools (like update_goals)
                        result = self._handle_meta_tool(code)
                        obs_text = result
                        reward = 0
                        logger.info(f"Step {steps+1} - Meta-tool result: {result}")
                    else:
                        # Execute game action
                        current_game_state = GameState.from_instance(env.instance)
                        action = Action(
                            agent_idx=0, game_state=current_game_state, code=code
                        )

                        try:
                            logger.debug(f"Step {steps+1} - Executing action...")
                            obs, reward, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                            total_reward += reward
                            logger.info(
                                f"Step {steps+1} - Result: reward={reward:.2f}, done={done}, info={info}"
                            )
                        except Exception as e:
                            logger.error(f"Environment error: {e}", exc_info=True)
                            obs = {"error": str(e)}
                            reward = -1

                    steps += 1

            env.close()

            # Calculate final score
            task_completed = info.get("task_completed", False) if info else False
            throughput_achieved = info.get("throughput", 0) if info else 0

            score = self._calculate_score(
                task_completed, throughput_achieved, steps, total_reward
            )

            # Tokenize trajectory
            tokenization_result = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=messages,
                train_on_all_assistant_turns=True,
            )

            return ScoredDataItem(
                messages=messages if self.config.include_messages else None,
                tokens=tokenization_result["tokens"],
                masks=tokenization_result["masks"],
                scores=score,
                metadata={
                    "trajectory_id": trajectory_id,
                    "task_name": task_name,
                    "task_completed": task_completed,
                    "throughput": throughput_achieved,
                    "steps": steps,
                    "total_reward": total_reward,
                },
            )

        except Exception as e:
            logger.error(f"Fatal error in trajectory {trajectory_id}: {e}")
            logger.error(traceback.format_exc())
            return None

    def _format_observation(self, obs: Dict, info: Dict) -> str:
        """Format game observation for the LLM."""
        parts = []

        # Current goals if any
        if self.current_goals:
            parts.append("Current goals:")
            for i, goal in enumerate(self.current_goals[:5]):
                parts.append(f"  {i+1}. {goal}")

        # Game state from observation
        if isinstance(obs, dict):
            if "inventory" in obs:
                inv = obs["inventory"]
                if inv:
                    inv_str = ", ".join(
                        [f"{item['type']}:{item['quantity']}" for item in inv[:8]]
                    )
                    parts.append(f"Inventory: {inv_str}")

            if "entities" in obs:
                parts.append(f"Entities nearby: {len(obs['entities'])}")

            if "raw_text" in obs and obs["raw_text"]:
                parts.append(f"Last action result: {obs['raw_text'][:200]}")

        # Task progress if available
        if info:
            if "throughput" in info:
                parts.append(f"Current throughput: {info['throughput']}")
            if "score" in info:
                parts.append(f"Score: {info['score']}")

        return "\n".join(parts) if parts else "Observing game state..."

    def _parse_action_to_code(self, action_text: str) -> str:
        """Parse LLM response to executable code."""
        # Look for tool_call blocks
        if "<tool_call>" in action_text and "</tool_call>" in action_text:
            start = action_text.find("<tool_call>") + len("<tool_call>")
            end = action_text.find("</tool_call>")
            tool_call_str = action_text[start:end].strip()

            try:
                import ast

                tool_call = ast.literal_eval(tool_call_str)

                # Handle update_goals specially
                if tool_call.get("name") == "update_goals":
                    goals = tool_call.get("arguments", {}).get("goals", [])
                    return f"__META__:update_goals:{json.dumps(goals)}"

                # Convert to Python code for FLE
                name = tool_call.get("name", "inspect_inventory")
                args = tool_call.get("arguments", {})

                # Build function call
                arg_strs = []
                for k, v in args.items():
                    if isinstance(v, dict) and "x" in v and "y" in v:
                        arg_strs.append(f"{k}=Position(x={v['x']}, y={v['y']})")
                    elif isinstance(v, str):
                        # Handle Resource/Prototype enums
                        if any(
                            v.startswith(prefix)
                            for prefix in ["Resource.", "Prototype.", "Direction."]
                        ):
                            arg_strs.append(f"{k}={v}")
                        else:
                            arg_strs.append(f"{k}={repr(v)}")
                    else:
                        arg_strs.append(f"{k}={v}")

                return f"print({name}({', '.join(arg_strs)}))"

            except Exception as e:
                logger.debug(f"Failed to parse tool_call: {e}")

        # Default to inspect_inventory if no valid action
        return "print(inspect_inventory())"

    def _handle_meta_tool(self, code: str) -> str:
        """Handle meta-tools like update_goals."""
        if code.startswith("__META__:update_goals:"):
            goals_json = code.split(":", 2)[2]
            try:
                goals = json.loads(goals_json)
                self.current_goals = goals[: self.config.max_goals]
                return f"Goals updated: {', '.join(self.current_goals[:3])}"
            except Exception:
                return "Failed to update goals"
        return "Unknown meta-tool"

    def _calculate_score(
        self, task_completed: bool, throughput: float, steps: int, reward: float
    ) -> float:
        """Calculate score for a trajectory."""
        score = 0.0

        # Task completion bonus
        if task_completed:
            score += self.config.task_completion_weight

        # Throughput score
        score += throughput * self.config.throughput_weight

        # Efficiency bonus (fewer steps is better)
        if steps > 0:
            score += (
                self.config.max_steps_per_episode / steps
            ) * self.config.efficiency_weight

        # Add environment reward
        score += reward

        return score

    @classmethod
    def config_init(cls) -> Tuple[FactorioEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = FactorioEnvConfig(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
            group_size=1,  # Single trajectory for now
            use_wandb=True,
            wandb_name=cls.name,
            max_token_length=32768,
            total_steps=100,
            task_names=["iron_ore_throughput"],  # Start simple
            enable_self_planning=True,
        )

        # Single server config for testing
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                base_url="http://localhost:8080/v1",  # Assuming local LLM server
                api_key="x",
                num_requests_for_eval=10,
            ),
        ]

        return env_config, server_configs

    async def evaluate(self, num_items: int = 10) -> Dict[str, Any]:
        """Run evaluation episodes."""
        logger.info(f"Starting evaluation with {num_items} episodes")

        eval_scores = []
        eval_completions = []

        for i in range(num_items):
            item = await self.get_next_item()
            item["is_eval"] = True

            scored_data_group, _ = await self.collect_trajectories(item)
            if scored_data_group and scored_data_group["scores"]:
                avg_score = sum(scored_data_group["scores"]) / len(
                    scored_data_group["scores"]
                )
                eval_scores.append(avg_score)

                # Check completion from last episode
                if self.episode_outcomes_buffer:
                    eval_completions.append(self.episode_outcomes_buffer[-1])

        if eval_scores:
            avg_score = sum(eval_scores) / len(eval_scores)
            completion_rate = (
                sum(eval_completions) / len(eval_completions) if eval_completions else 0
            )

            self.eval_metrics_custom = [
                (f"{self.name}_eval/avg_score", avg_score),
                (f"{self.name}_eval/completion_rate", completion_rate),
                (f"{self.name}_eval/num_episodes", len(eval_scores)),
            ]

            logger.info(
                f"Evaluation: avg_score={avg_score:.2f}, completion_rate={completion_rate:.2%}"
            )

            return {
                "avg_score": avg_score,
                "completion_rate": completion_rate,
                "num_episodes": len(eval_scores),
            }

        return {"message": "No evaluation episodes completed"}

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log training metrics
        if self.episode_rewards_buffer:
            total_episodes = len(self.episode_rewards_buffer)
            avg_reward = sum(self.episode_rewards_buffer) / total_episodes
            avg_steps = (
                sum(self.episode_steps_buffer) / total_episodes
                if self.episode_steps_buffer
                else 0
            )
            completion_rate = (
                sum(self.episode_outcomes_buffer) / total_episodes
                if self.episode_outcomes_buffer
                else 0
            )

            wandb_metrics.update(
                {
                    f"{self.name}/train/total_episodes": total_episodes,
                    f"{self.name}/train/avg_reward": avg_reward,
                    f"{self.name}/train/avg_steps": avg_steps,
                    f"{self.name}/train/completion_rate": completion_rate,
                }
            )

            # Task breakdown
            task_counts = {}
            for task in self.episode_task_types:
                task_counts[task] = task_counts.get(task, 0) + 1
            for task, count in task_counts.items():
                wandb_metrics[f"{self.name}/train/task_{task}_count"] = count
        # Per-group aggregates across groups since last log
        if self._group_metrics_buffer:
            try:
                num_groups = len(self._group_metrics_buffer)
                avg_reward = (
                    sum(g["avg_reward"] for g in self._group_metrics_buffer)
                    / num_groups
                )
                avg_steps = (
                    sum(g["avg_steps"] for g in self._group_metrics_buffer) / num_groups
                )
                completion_rate = (
                    sum(g["completion_rate"] for g in self._group_metrics_buffer)
                    / num_groups
                )
                wandb_metrics.update(
                    {
                        f"{self.name}/train/group/num_groups": num_groups,
                        f"{self.name}/train/group/avg_reward": avg_reward,
                        f"{self.name}/train/group/avg_steps": avg_steps,
                        f"{self.name}/train/group/completion_rate": completion_rate,
                    }
                )
            finally:
                self._group_metrics_buffer = []

        # Clear buffers
        self.episode_rewards_buffer = []
        self.episode_steps_buffer = []
        self.episode_outcomes_buffer = []
        self.episode_task_types = []

        # Add eval metrics
        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    # Allow running directly for testing
    FactorioEnv.cli()
