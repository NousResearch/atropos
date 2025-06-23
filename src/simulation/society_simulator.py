import asyncio
import json  # Added for saving/loading state to/from JSON files
import logging
import os  # Added os for path operations in autosave
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mesa

from src.agents.llm_agent import (  # Assuming Position is in llm_agent or define it here
    LLMAgent,
    Position,
)
from src.asset_generation.asset_manager import Asset, AssetManager  # Added Asset
from src.database.database_handler import (  # Added DatabaseHandler import
    DatabaseHandler,
)
from src.economics.banking_system import (  # Might need AccountType, LoanType etc. for LLMAgent
    BankingSystem,
)
from src.economics.market_system import (  # Import ResourceType for market processing
    MarketSystem,
    ResourceType,
)
from src.flame_gpu.flame_gpu_simulation import (
    AgentType,
    CulturalGroup,
    FlameGPUSimulation,
)
from src.flame_gpu.flame_gpu_simulation import (
    SimulationConfig as FlameGPUSimConfig,  # Import AgentType and CulturalGroup
)

# Placeholder for LLMCoordinator if needed by LLMAgent directly
from src.llm.coordinator import LLMCoordinator  # Added import
from src.social.family_system import FamilySystem
from src.utils.config import Config  # Assuming this path is correct

logger = logging.getLogger(__name__)

# PointPosition alias is no longer needed as we use llm_agent.Position

# Placeholder for current simulation version
# This should ideally come from a centralized version definition (e.g., __init__.py of the project)
SIMULATION_VERSION = "0.1.0-alpha+state_v1"


@dataclass
class WorldObject:
    obj_id: str  # Will be set in add_world_object
    asset_id: str
    description: str
    file_path: str
    position: Position  # Use the Position dataclass from LLMAgent
    creator_id: Optional[str] = None
    created_at_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Position is already a dataclass, asdict handles it if it doesn't have its own to_dict
        # However, LLMAgent.Position does not have to_dict, so we do it manually here.
        if isinstance(self.position, Position):
            data["position"] = {
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z,
            }
        elif isinstance(
            self.position, dict
        ):  # Should already be a dict if from_dict was used correctly
            data["position"] = self.position
        # else: position might be something else, log warning or error if necessary
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldObject":
        pos_data = data.get("position", {})
        # Ensure position is reconstructed as a Position object
        position_obj = Position(
            x=pos_data.get("x", 0.0), y=pos_data.get("y", 0.0), z=pos_data.get("z", 0.0)
        )

        # Create a new dict for cls construction to avoid modifying input `data` directly if it's reused
        init_data = data.copy()
        init_data["position"] = position_obj

        # Ensure all required fields for WorldObject are present
        # obj_id, asset_id, description, file_path are required by dataclass implicitly.
        # creator_id, created_at_step have defaults or are Optional.
        return cls(**init_data)


class SocietySimulator(mesa.Model):
    """
    Main simulation model for the LLM Society.
    Manages agents, the environment, and the simulation loop.
    """

    def __init__(self, config: Config, running_from_load: bool = False):
        super().__init__()
        self.config = config
        self.current_step = 0
        self.next_id_counter = 0
        self.schedule = mesa.time.RandomActivation(self)

        self.database_handler: Optional[DatabaseHandler] = None
        db_url = getattr(self.config.output, "database_url", None)
        if db_url:
            try:
                self.database_handler = DatabaseHandler(config=self.config)
                if not self.database_handler.engine:
                    logger.warning(
                        "DatabaseHandler engine not initialized. DB operations will be skipped."
                    )
                    self.database_handler = None
            except Exception as e:
                logger.error(
                    f"Failed to initialize DatabaseHandler: {e}", exc_info=True
                )
                self.database_handler = None
        else:
            logger.info(
                "No database_url configured. Database operations will be skipped."
            )

        self.llm_coordinator = LLMCoordinator(config)
        self.asset_manager = AssetManager(
            output_directory=config.output.generated_assets_dir or "generated_assets"
        )
        self.world_objects: List[WorldObject] = []
        flame_gpu_config = FlameGPUSimConfig(
            max_agents=self.config.agents.count,
            world_width=self.config.simulation.world_size[0],
            world_height=self.config.simulation.world_size[1],
        )
        self.flame_gpu_sim = FlameGPUSimulation(flame_gpu_config)

        # Pass db_handler and current_step_getter to FamilySystem
        self.family_system = FamilySystem(
            database_handler=self.database_handler,
            current_step_getter=lambda: self.current_step,
        )
        self.market_system = (
            MarketSystem()
        )  # MarketSystem might need these too if it starts logging events
        self.banking_system = BankingSystem(
            database_handler=self.database_handler,
            current_step_getter=lambda: self.current_step,
        )

        self.agents_to_initialize_in_flame: Optional[List[Dict[str, Any]]] = None
        self._is_loaded_run = running_from_load

        if not running_from_load:
            logger.info(
                "SocietySimulator: New run. Agent creation deferred to async setup in run()."
            )
        else:
            logger.info(
                "SocietySimulator: Loaded run. Agent & system restoration via from_dict. Async setup in run()."
            )

    def get_next_id_str(self, prefix="item") -> str:
        self.next_id_counter += 1
        return f"{prefix}_{self.next_id_counter}"

    async def _async_initial_setup(self):
        """Handles asynchronous parts of initial setup like starting DB run and creating/initializing agents."""
        if self.database_handler:
            config_snapshot = (
                self.config.to_dict() if hasattr(self.config, "to_dict") else None
            )
            await self.database_handler.start_simulation_run(
                config_snapshot=config_snapshot
            )

        if self.llm_coordinator and not self.llm_coordinator._running:
            asyncio.create_task(self.llm_coordinator.start())

        if not self._is_loaded_run:
            await self._create_agents_and_init_flamegpu()  # This will create agents and init FlameGPU
        elif (
            self.agents_to_initialize_in_flame
        ):  # If loaded, and there are states prepared by from_dict
            logger.info(
                f"Re-initializing FlameGPUSimulation with {len(self.agents_to_initialize_in_flame)} loaded agent states..."
            )
            if hasattr(self, "flame_gpu_sim") and self.flame_gpu_sim:
                success = await self.flame_gpu_sim.initialize_agents(
                    self.agents_to_initialize_in_flame
                )
                if success:
                    logger.info(
                        "FlameGPUSimulation successfully re-initialized with loaded agent states."
                    )
                else:
                    logger.error(
                        "Failed to re-initialize FlameGPUSimulation with loaded agent states."
                    )
            self.agents_to_initialize_in_flame = None  # Clear after use

    async def _create_agents_and_init_flamegpu(self):
        """Creates Python LLMAgents and then initializes them in FlameGPU asynchronously."""
        agent_initial_states_for_flame_gpu = []
        logger.debug("Creating initial agents and preparing FlameGPU states...")
        for i in range(self.config.agents.count):
            agent_id_int = i
            agent_unique_id_str = f"agent_{agent_id_int}"
            pos = Position(
                self.random.uniform(0, self.config.simulation.world_size[0]),
                self.random.uniform(0, self.config.simulation.world_size[1]),
                0.0,
            )
            agent = LLMAgent(
                self,
                agent_unique_id_str,
                self.llm_coordinator,
                self.config,
                pos,
                initial_age=float(
                    getattr(self.config.agents, "initial_min_age", 20)
                    + random.random()
                    * (
                        getattr(self.config.agents, "initial_max_age", 50)
                        - getattr(self.config.agents, "initial_min_age", 20)
                    )
                ),
                initial_health=float(
                    getattr(self.config.agents, "initial_health", 1.0)
                ),
                initial_employed_status=int(
                    getattr(self.config.agents, "initial_employed_prob", 0.5)
                    > random.random()
                ),
            )
            self.schedule.add(agent)
            # ... (flame_gpu_state prep as before using agent.* attributes) ...
            cash = float(agent.resources.get("currency", 100.0))
            bank = 0.0
            flame_gpu_wealth_init = cash + bank
            cultural_group_val = (
                agent.cultural_group_id
                if agent.cultural_group_id is not None
                else CulturalGroup.HARMONISTS.value
            )
            initial_flame_gpu_state = {
                "agent_id": agent_id_int,
                "x": agent.position.x,
                "y": agent.position.y,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "agent_type": agent.agent_type.value,
                "age": agent.age,
                "energy": agent.energy,
                "happiness": agent.happiness,
                "health": agent.health,
                "employed": agent.employed,
                "family_id": agent.family_id if agent.family_id is not None else -1,
                "cultural_group": cultural_group_val,
                "social_reputation": agent.social_reputation,
                "num_connections": 0,
                "wealth": flame_gpu_wealth_init,
                "currency": cash,
                "food_resources": float(agent.resources.get("food", 10.0)),
                "material_resources": float(agent.resources.get("materials", 5.0)),
                "energy_resources": float(agent.resources.get("energy_item", 5.0)),
                "luxury_resources": float(agent.resources.get("luxury", 1.0)),
                "knowledge_resources": float(agent.resources.get("knowledge", 0.0)),
                "tools_resources": float(agent.resources.get("tools", 1.0)),
                "services_resources": float(agent.resources.get("services", 0.0)),
                "credit_score": agent.credit_score,
                "total_debt": agent.total_debt,
                "monthly_income": agent.monthly_income,
                **{
                    f"cultural_affinity_{cg.name.lower()}": agent.cultural_affinities.get(
                        cg.name.lower(), 0.2
                    )
                    for cg in CulturalGroup
                },
            }
            if (
                agent.cultural_group_id is not None
                and 0 <= agent.cultural_group_id < len(CulturalGroup)
            ):
                affinity_names = [cg.name.lower() for cg in CulturalGroup]
                [
                    initial_flame_gpu_state.update(
                        {f"cultural_affinity_{name_part}": 0.1}
                    )
                    for name_part in affinity_names
                ]
                initial_flame_gpu_state[
                    f"cultural_affinity_{CulturalGroup(agent.cultural_group_id).name.lower()}"
                ] = 0.6
            agent_initial_states_for_flame_gpu.append(initial_flame_gpu_state)

        logger.info(f"Created {len(self.schedule.agents)} LLMAgents.")
        if hasattr(self, "flame_gpu_sim") and self.flame_gpu_sim:
            success = await self.flame_gpu_sim.initialize_agents(
                agent_initial_states_for_flame_gpu
            )
            logger.info(
                f"Async FlameGPU agent initialization {'succeeded' if success else 'failed'}."
            )
        else:
            logger.warning(
                "FlameGPUSimulation not available, skipping agent initialization in Flame GPU."
            )

    def add_world_object(
        self, asset: Asset, position: Position, creator_id: Optional[str] = None
    ) -> WorldObject:
        """
        Adds a newly created asset as a WorldObject to the simulation environment.
        Uses the Position dataclass.
        """
        obj_id = self.get_next_id_str("obj")
        world_obj = WorldObject(
            obj_id=obj_id,
            asset_id=asset.asset_id,
            description=asset.description,
            file_path=asset.file_path,
            position=position,  # Directly use the Position object
            creator_id=creator_id,
            created_at_step=self.current_step,
        )
        self.world_objects.append(world_obj)
        logger.info(
            f"Added WorldObject: {world_obj.obj_id} ({world_obj.description}) at {world_obj.position} created by {creator_id or 'N/A'}"
        )
        return world_obj

    def get_objects_near(self, position: Position, radius: float) -> List[WorldObject]:
        """
        Returns a list of WorldObjects within a given radius of a position.
        Uses the Position dataclass and its distance_to method.
        """
        nearby_objects = []
        for obj in self.world_objects:
            if position.distance_to(obj.position) <= radius:
                nearby_objects.append(obj)
        return nearby_objects

    async def step(self):
        """
        Advance the model by one step.
        This will handle asynchronous agent steps.
        """
        logger.info(f"--- Simulation Step {self.current_step} Starting ---")

        logger.debug("Phase 1: LLMAgent steps...")
        await asyncio.gather(*[agent.step() for agent in self.schedule.agents])
        logger.debug("Phase 1: LLMAgent steps completed.")

        logger.debug(
            "Phase 2: Python systems processing (Market, Family, Banking periodic)..."
        )
        if hasattr(self, "market_system") and self.market_system:
            all_transactions_by_resource = self.market_system.process_all_markets()
            for (
                resource_type_val,
                transactions_list,
            ) in all_transactions_by_resource.items():
                for transaction_obj in transactions_list:
                    buyer = next(
                        (
                            a
                            for a in self.schedule.agents
                            if a.unique_id == transaction_obj.buyer_id
                        ),
                        None,
                    )
                    seller = next(
                        (
                            a
                            for a in self.schedule.agents
                            if a.unique_id == transaction_obj.seller_id
                        ),
                        None,
                    )
                    if buyer:
                        buyer.resources["currency"] = (
                            buyer.resources.get("currency", 0)
                            - transaction_obj.total_cost
                        )
                        buyer.resources[transaction_obj.resource_type.value] = (
                            buyer.resources.get(transaction_obj.resource_type.value, 0)
                            + transaction_obj.quantity
                        )
                        await buyer._add_memory(
                            f"Bought {transaction_obj.quantity} {transaction_obj.resource_type.value} for {transaction_obj.total_cost}. Order: {transaction_obj.buy_order_id}",
                            0.8,
                        )
                    if seller:
                        seller.resources["currency"] = (
                            seller.resources.get("currency", 0)
                            + transaction_obj.total_cost
                        )
                        seller.resources[transaction_obj.resource_type.value] = (
                            seller.resources.get(transaction_obj.resource_type.value, 0)
                            - transaction_obj.quantity
                        )
                        await seller._add_memory(
                            f"Sold {transaction_obj.quantity} {transaction_obj.resource_type.value} for {transaction_obj.total_cost}. Order: {transaction_obj.sell_order_id}",
                            0.8,
                        )
                    if self.database_handler:
                        try:
                            await self.database_handler.save_market_transaction(
                                transaction_obj.to_dict(), self.current_step
                            )
                        except Exception as e:
                            logger.error(
                                f"DB Error saving market txn {transaction_obj.transaction_id}: {e}",
                                exc_info=True,
                            )
            logger.debug("MarketSystem processing and transaction saving completed.")

        if hasattr(self, "family_system") and self.family_system:
            logger.debug("Processing family dynamics...")
            agent_states_for_family = {ag.unique_id: ag for ag in self.schedule.agents}
            family_events = await self.family_system.process_family_dynamics(
                agent_states=agent_states_for_family
            )
            if family_events:
                logger.debug(f"Family system generated events: {family_events}")
                for event_key, event_data in family_events.items():
                    if event_data.get("type") == "reproduction":
                        logger.info(
                            f"Reproduction event: {event_data}. New agent creation logic TBD in SocietySimulator."
                        )
                        if self.database_handler:
                            await self.database_handler.save_simulation_event(
                                event_type="SIM_REPRODUCTION_EVENT_HANDLING_NEEDED",
                                step=self.current_step,
                                details=event_data,
                                description=f"Reproduction identified for parents {event_data.get('parents')}",
                            )
            logger.debug("FamilySystem dynamics processed.")

        if (
            hasattr(self, "banking_system")
            and self.banking_system
            and hasattr(self.banking_system, "process_monthly_interest")
        ):
            if self.current_step > 0 and self.current_step % 30 == 0:
                logger.debug("Processing monthly banking interest...")
                await self.banking_system.process_monthly_interest(
                    current_step=self.current_step
                )
                logger.debug("Monthly banking interest processed.")
        logger.debug("Phase 2: Python systems processing completed.")

        # 3. Prime FlameGPU with current LLMAgent states
        logger.debug("Phase 3: Priming FlameGPU with current agent states...")
        if hasattr(self, "flame_gpu_sim") and self.flame_gpu_sim:
            current_agent_states_for_flame_gpu = []
            for llm_agent in self.schedule.agents:
                try:
                    agent_id_int = int(llm_agent.unique_id.split("_")[-1])
                except (ValueError, IndexError):
                    logger.error(
                        f"Could not parse int ID from LLMAgent unique_id: {llm_agent.unique_id}. Skipping for FlameGPU prime."
                    )
                    continue

                cash_on_hand_prime = float(llm_agent.resources.get("currency", 0.0))
                bank_balance_prime = 0.0
                if hasattr(self.banking_system, "get_agent_financial_summary"):
                    financial_summary = self.banking_system.get_agent_financial_summary(
                        llm_agent.unique_id
                    )
                    if financial_summary and "error" not in financial_summary:
                        bank_balance_prime = financial_summary.get("total_balance", 0.0)
                flame_gpu_wealth_prime = cash_on_hand_prime + bank_balance_prime

                flame_gpu_agent_prime_state = {
                    "agent_id": agent_id_int,
                    "x": llm_agent.position.x,
                    "y": llm_agent.position.y,
                    "velocity_x": 0.0,
                    "velocity_y": 0.0,
                    "agent_type": llm_agent.agent_type.value,
                    "age": llm_agent.age,
                    "energy": llm_agent.energy,
                    "happiness": llm_agent.happiness,
                    "health": llm_agent.health,
                    "employed": llm_agent.employed,
                    "family_id": (
                        llm_agent.family_id if llm_agent.family_id is not None else -1
                    ),
                    "cultural_group": (
                        llm_agent.cultural_group_id
                        if llm_agent.cultural_group_id is not None
                        else CulturalGroup.HARMONISTS.value
                    ),
                    "social_reputation": llm_agent.social_reputation,
                    "num_connections": len(llm_agent.social_connections),
                    "wealth": flame_gpu_wealth_prime,
                    "currency": cash_on_hand_prime,
                    "food_resources": float(llm_agent.resources.get("food", 0.0)),
                    "material_resources": float(
                        llm_agent.resources.get("materials", 0.0)
                    ),
                    "energy_resources": float(
                        llm_agent.resources.get("energy_item", 0.0)
                    ),
                    "luxury_resources": float(llm_agent.resources.get("luxury", 0.0)),
                    "knowledge_resources": float(
                        llm_agent.resources.get("knowledge", 0.0)
                    ),
                    "tools_resources": float(llm_agent.resources.get("tools", 0.0)),
                    "services_resources": float(
                        llm_agent.resources.get("services", 0.0)
                    ),
                    "credit_score": llm_agent.credit_score,
                    "total_debt": llm_agent.total_debt,
                    "monthly_income": llm_agent.monthly_income,
                    "cultural_affinity_harmonists": llm_agent.cultural_affinities.get(
                        "harmonists", 0.2
                    ),
                    "cultural_affinity_builders": llm_agent.cultural_affinities.get(
                        "builders", 0.2
                    ),
                    "cultural_affinity_guardians": llm_agent.cultural_affinities.get(
                        "guardians", 0.2
                    ),
                    "cultural_affinity_scholars": llm_agent.cultural_affinities.get(
                        "scholars", 0.2
                    ),
                    "cultural_affinity_wanderers": llm_agent.cultural_affinities.get(
                        "wanderers", 0.2
                    ),
                }
                current_agent_states_for_flame_gpu.append(flame_gpu_agent_prime_state)

            if (
                current_agent_states_for_flame_gpu
            ):  # Log first agent's prime state if any
                logger.debug(
                    f"Priming FlameGPU with {len(current_agent_states_for_flame_gpu)} states. First agent prime data: {current_agent_states_for_flame_gpu[0] if current_agent_states_for_flame_gpu else 'N/A'}"
                )
            await self.flame_gpu_sim.prime_agent_states_for_step(
                current_agent_states_for_flame_gpu
            )
            logger.debug("Phase 3: FlameGPU priming completed.")

            # 4. Run FlameGPU simulation step
            logger.debug("Phase 4: Running FlameGPU simulation step...")
            await self.flame_gpu_sim.run_simulation_step()
            logger.debug("Phase 4: FlameGPU step completed.")

            # 5. Update LLMAgents with states from FlameGPUSimulation
            logger.debug(
                "Phase 5: Updating LLMAgents from FlameGPU states with validation..."
            )
            flame_gpu_agent_states = await self.flame_gpu_sim.get_agent_states()
            if flame_gpu_agent_states:
                for i, fg_state in enumerate(flame_gpu_agent_states):
                    agent_id_int_from_gpu = fg_state.get("agent_id")
                    llm_agent_id = f"agent_{agent_id_int_from_gpu}"
                    llm_agent = next(
                        (
                            a
                            for a in self.schedule.agents
                            if a.unique_id == llm_agent_id
                        ),
                        None,
                    )

                    if not llm_agent:
                        logger.warning(
                            f"LLMAgent {llm_agent_id} not found for FlameGPU update."
                        )
                        continue

                    try:
                        llm_agent.position.x = float(
                            fg_state.get("x", llm_agent.position.x)
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid position.x from FlameGPU for {llm_agent_id}: {fg_state.get('x')}"
                        )
                    try:
                        llm_agent.position.y = float(
                            fg_state.get("y", llm_agent.position.y)
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid position.y from FlameGPU for {llm_agent_id}: {fg_state.get('y')}"
                        )
                    # Z position can be handled similarly if used by FlameGPU

                    try:
                        llm_agent.energy = max(
                            0.0,
                            min(1.0, float(fg_state.get("energy", llm_agent.energy))),
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid energy from FlameGPU for {llm_agent_id}: {fg_state.get('energy')}"
                        )
                    try:
                        llm_agent.happiness = max(
                            0.0,
                            min(
                                1.0,
                                float(fg_state.get("happiness", llm_agent.happiness)),
                            ),
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid happiness from FlameGPU for {llm_agent_id}: {fg_state.get('happiness')}"
                        )

                    llm_agent.family_id = fg_state.get("family_id", llm_agent.family_id)
                    try:
                        llm_agent.cultural_group_id = CulturalGroup(
                            int(
                                fg_state.get(
                                    "cultural_group",
                                    (
                                        llm_agent.cultural_group_id
                                        if llm_agent.cultural_group_id is not None
                                        else 0
                                    ),
                                )
                            )
                        ).value
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid cultural_group from FlameGPU for {llm_agent_id_to_find}: {flame_agent_state.get('cultural_group')}"
                        )
                    try:
                        llm_agent.social_reputation = float(
                            flame_agent_state.get(
                                "social_reputation", llm_agent.social_reputation
                            )
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid social_reputation from FlameGPU for {llm_agent_id_to_find}: {flame_agent_state.get('social_reputation')}"
                        )
                    try:
                        llm_agent.credit_score = float(
                            flame_agent_state.get(
                                "credit_score", llm_agent.credit_score
                            )
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid credit_score from FlameGPU for {llm_agent_id_to_find}: {flame_agent_state.get('credit_score')}"
                        )
                    try:
                        llm_agent.total_debt = float(
                            flame_agent_state.get("total_debt", llm_agent.total_debt)
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid total_debt from FlameGPU for {llm_agent_id_to_find}: {flame_agent_state.get('total_debt')}"
                        )
                    try:
                        llm_agent.monthly_income = float(
                            flame_agent_state.get(
                                "monthly_income", llm_agent.monthly_income
                            )
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid monthly_income from FlameGPU for {llm_agent_id_to_find}: {flame_agent_state.get('monthly_income')}"
                        )

                    gpu_agent_type_val = flame_agent_state.get("agent_type")
                    if gpu_agent_type_val is not None:
                        try:
                            llm_agent.agent_type = AgentType(int(gpu_agent_type_val))
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid AgentType value {gpu_agent_type_val} from FlameGPU for {llm_agent_id_to_find}."
                            )

                    flame_affinities = flame_agent_state.get("cultural_affinities", {})
                    for aff_name in llm_agent.cultural_affinities.keys():
                        flame_key = aff_name.capitalize()
                        if flame_key in flame_affinities:
                            try:
                                llm_agent.cultural_affinities[aff_name] = float(
                                    flame_affinities[flame_key]
                                )
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Invalid cultural_affinity {flame_key} from FlameGPU for {llm_agent_id_to_find}: {flame_affinities[flame_key]}"
                                )

                    # Resource updates with validation
                    resource_map = {
                        "food": "food_resources",
                        "materials": "material_resources",
                        "currency": "currency",
                        "energy_item": "energy_resources",
                        "luxury": "luxury_resources",
                        "knowledge": "knowledge_resources",
                        "tools": "tools_resources",
                        "services": "services_resources",
                    }
                    for py_res_name, fg_res_name in resource_map.items():
                        try:
                            llm_agent.resources[py_res_name] = int(
                                float(
                                    flame_agent_state.get(
                                        fg_res_name,
                                        llm_agent.resources.get(py_res_name, 0),
                                    )
                                )
                            )
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid {fg_res_name} from FlameGPU for {llm_agent_id_to_find}: {flame_agent_state.get(fg_res_name)}"
                            )

                    # Sync age, health, employed from FlameGPU to LLMAgent
                    try:
                        llm_agent.age = float(fg_state.get("age", llm_agent.age))
                    except (ValueError, TypeError):
                        logger.warning(
                            "Invalid age "{fg_state.get('age')}' from FlameGPU for {llm_agent_id}."
                        )
                    try:
                        llm_agent.health = max(
                            0.0,
                            min(1.0, float(fg_state.get("health", llm_agent.health))),
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            "Invalid health "{fg_state.get('health')}' from FlameGPU for {llm_agent_id}."
                        )
                    try:
                        llm_agent.employed = int(
                            fg_state.get("employed", llm_agent.employed)
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            "Invalid employed status "{fg_state.get('employed')}' from FlameGPU for {llm_agent_id}."
                        )

                    if i == 0:  # Log first agent's state change
                        logger.debug(
                            f"Agent {llm_agent_id_to_find} energy: Python_pre_prime={llm_agent.energy:.2f} -> FlameGPU_final={flame_agent_state.get('energy', llm_agent.energy):.2f}. Currency: {llm_agent.resources.get('currency')}"
                        )
                logger.debug("Phase 5: LLMAgent updates from FlameGPU completed.")
            else:
                logger.warning(
                    "No agent states received from FlameGPUSimulation for update."
                )
        else:
            logger.warning(
                "FlameGPUSimulation not available, skipping FlameGPU step and agent updates from FlameGPU."
            )

        self.schedule.steps += 1
        self.current_step += 1
        logger.info(f"--- Simulation Step {self.current_step -1} Ended ---")

    async def run(self):
        # Perform async initial setup (DB run record, LLM Coordinator, initial agent creation & FlameGPU init)
        await self._async_initial_setup()

        logger.info(
            f"Starting simulation run for {self.config.simulation.max_steps} steps. Current run ID: {getattr(self.database_handler, '_current_run_id', 'N/A')}"
        )
        run_status = "completed"
        try:
            for i in range(self.config.simulation.max_steps):
                if i % 10 == 0 and i > 0:
                    logger.info(
                        f"Progress: Step {i}/{self.config.simulation.max_steps}"
                    )
                await self.step()

                if (
                    self.config.simulation.autosave_enabled
                    and self.config.simulation.autosave_interval_steps
                    and self.current_step > 0
                    and self.current_step
                    % self.config.simulation.autosave_interval_steps
                    == 0
                ):
                    autosave_base_dir = self.config.output.directory
                    autosave_subdir = self.config.simulation.autosave_directory
                    filename_pattern = self.config.simulation.autosave_file_pattern
                    full_autosave_dir = os.path.join(autosave_base_dir, autosave_subdir)
                    try:
                        save_filename = filename_pattern.format(step=self.current_step)
                    except KeyError:
                        logger.warning(
                            "Autosave pattern "{filename_pattern}' invalid."
                        )
                        save_filename = f"autosave_step_{self.current_step}.json"
                    full_save_path = os.path.join(full_autosave_dir, save_filename)
                    self.save_state(full_save_path)

                if self.running is False:  # Mesa model `running` flag for early stop
                    logger.info(
                        f"Simulation stopped early by self.running=False at step {i}."
                    )
                    run_status = "stopped_early"
                    break
        except KeyboardInterrupt:
            logger.warning("Simulation run interrupted by user (KeyboardInterrupt).")
            run_status = "interrupted"
        except Exception as e_run:
            logger.error(
                f"Critical error during simulation run: {e_run}", exc_info=True
            )
            run_status = "failed"
        finally:
            if self.llm_coordinator:
                try:
                    if asyncio.iscoroutinefunction(self.llm_coordinator.stop):
                        await self.llm_coordinator.stop()
                    else:
                        self.llm_coordinator.stop()
                    logger.info("LLM Coordinator stopped.")
                except Exception as e_llm_stop:
                    logger.error(f"Error stopping LLMCoordinator: {e_llm_stop}")

            if self.database_handler:
                self.database_handler.end_simulation_run(status=run_status)

            logger.info(f"Simulation run finished with status: {run_status}.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the complete state of the SocietySimulator."""
        logger.info("Serializing SocietySimulator state...")
        return {
            "current_step": self.current_step,
            "next_id_counter": self.next_id_counter,
            "agents": [agent.to_dict() for agent in self.schedule.agents],
            "world_objects": [obj.to_dict() for obj in self.world_objects],
            "llm_coordinator_state": self.llm_coordinator.to_dict(),
            "asset_manager_state": self.asset_manager.to_dict(),
            "family_system_state": self.family_system.to_dict(),
            "market_system_state": self.market_system.to_dict(),
            "banking_system_state": self.banking_system.to_dict(),
            # self.config is not saved; it's passed during reconstruction.
            # self.flame_gpu_sim object itself is not serialized.
            # Its state is effectively captured by the agent states, which are used to
            # re-initialize the FlameGPU simulation upon loading via flame_gpu_sim.initialize_agents().
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Config) -> "SocietySimulator":
        logger.info("Attempting to restore SocietySimulator from_dict...")
        simulator = cls(config, running_from_load=True)
        saved_version = data.get("simulation_version", "UNKNOWN")
        if saved_version != SIMULATION_VERSION and saved_version != "UNKNOWN":
            logger.warning(
                f"Loading state v{saved_version} with sim v{SIMULATION_VERSION}. Compatibility issues may arise."
            )

        simulator.current_step = data.get("current_step", 0)
        simulator.next_id_counter = data.get("next_id_counter", 0)

        if "llm_coordinator_state" in data and simulator.llm_coordinator:
            simulator.llm_coordinator = LLMCoordinator.from_dict(
                data["llm_coordinator_state"], config
            )
        if "asset_manager_state" in data and simulator.asset_manager:
            simulator.asset_manager = AssetManager.from_dict(
                data["asset_manager_state"],
                config.output.generated_assets_dir or "generated_assets",
            )
        if "family_system_state" in data and simulator.family_system:
            simulator.family_system = FamilySystem.from_dict(
                data["family_system_state"],
                database_handler=simulator.database_handler,
                current_step_getter=lambda: simulator.current_step,
            )
        if "market_system_state" in data and simulator.market_system:
            simulator.market_system = MarketSystem.from_dict(
                data["market_system_state"]
            )
        if "banking_system_state" in data and simulator.banking_system:
            simulator.banking_system = BankingSystem.from_dict(
                data["banking_system_state"],
                database_handler=simulator.database_handler,
                current_step_getter=lambda: simulator.current_step,
            )

        simulator.schedule = mesa.time.RandomActivation(simulator)
        agent_data_list = data.get("agents", [])
        flame_gpu_states_load = []
        for i, agent_data in enumerate(agent_data_list):
            try:
                agent = LLMAgent.from_dict(
                    agent_data, simulator, simulator.llm_coordinator, config
                )
                simulator.schedule.add(agent)
                agent_id_int = int(agent.unique_id.split("_")[-1])
                cash = float(agent.resources.get("currency", 0.0))
                bank = 0.0
                if hasattr(simulator.banking_system, "get_agent_financial_summary"):
                    summary = simulator.banking_system.get_agent_financial_summary(
                        agent.unique_id
                    )
                    if summary and "error" not in summary:
                        bank = summary.get("total_balance", 0.0)
                wealth = cash + bank
                cultural_group_val = (
                    agent.cultural_group_id
                    if agent.cultural_group_id is not None
                    else CulturalGroup.HARMONISTS.value
                )
                fg_state = {
                    "agent_id": agent_id_int,
                    "x": agent.position.x,
                    "y": agent.position.y,
                    "velocity_x": 0.0,
                    "velocity_y": 0.0,
                    "agent_type": agent.agent_type.value,
                    "age": agent.age,
                    "energy": agent.energy,
                    "happiness": agent.happiness,
                    "health": agent.health,
                    "employed": agent.employed,
                    "family_id": agent.family_id if agent.family_id is not None else -1,
                    "cultural_group": cultural_group_val,
                    "social_reputation": agent.social_reputation,
                    "num_connections": len(agent.social_connections),
                    "wealth": wealth,
                    "currency": cash,
                    "food_resources": float(agent.resources.get("food", 0.0)),
                    "material_resources": float(agent.resources.get("materials", 0.0)),
                    "energy_resources": float(agent.resources.get("energy_item", 0.0)),
                    "luxury_resources": float(agent.resources.get("luxury", 0.0)),
                    "knowledge_resources": float(agent.resources.get("knowledge", 0.0)),
                    "tools_resources": float(agent.resources.get("tools", 0.0)),
                    "services_resources": float(agent.resources.get("services", 0.0)),
                    "credit_score": agent.credit_score,
                    "total_debt": agent.total_debt,
                    "monthly_income": agent.monthly_income,
                    **{
                        f"cultural_affinity_{cg.name.lower()}": agent.cultural_affinities.get(
                            cg.name.lower(), 0.2
                        )
                        for cg in CulturalGroup
                    },
                }
                flame_gpu_states_load.append(fg_state)
            except Exception as e_agent_load:
                logger.error(
                    f"Error loading agent from data: {agent_data.get('unique_id','UNKNOWN_ID')}. Error: {e_agent_load}",
                    exc_info=True,
                )

        simulator.agents_to_initialize_in_flame = flame_gpu_states_load
        simulator.world_objects = [
            WorldObject.from_dict(obj_data)
            for obj_data in data.get("world_objects", [])
        ]
        logger.info(
            f"SocietySimulator state components restored. {len(simulator.schedule.agents)} agents, {len(simulator.world_objects)} objects. FlameGPU init deferred to run()."
        )
        return simulator

    def save_state(self, file_path: str):
        """Saves the current simulation state to a JSON file."""
        logger.info(f"Attempting to save simulation state to: {file_path}")
        try:
            state_dict = self.to_dict()
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(state_dict, f, indent=4, sort_keys=True)
            logger.info(f"Simulation state successfully saved to: {file_path}")
        except Exception as e:
            logger.error(
                f"Error saving simulation state to {file_path}: {e}", exc_info=True
            )
            # Optionally, re-raise or handle more gracefully depending on desired behavior

    @classmethod
    def load_from_file(
        cls, file_path: str, config: Config
    ) -> Optional["SocietySimulator"]:
        """Loads simulation state from a JSON file."""
        logger.info(f"Attempting to load simulation state from: {file_path}")
        try:
            if not os.path.exists(file_path):
                logger.error(f"Save file not found: {file_path}")
                return None
            with open(file_path, "r") as f:
                state_dict = json.load(f)

            # Basic validation of loaded data (optional, but good practice)
            if (
                not isinstance(state_dict, dict) or "agents" not in state_dict
            ):  # Check for a key field
                logger.error(f"Invalid or corrupt save file format: {file_path}")
                return None

            return cls.from_dict(state_dict, config)
        except json.JSONDecodeError as e_json:
            logger.error(
                f"Error decoding JSON from save file {file_path}: {e_json}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Error loading simulation state from {file_path}: {e}", exc_info=True
            )
            return None


if __name__ == "__main__":
    # This is a basic test, real execution would be from src/main.py
    async def main_test_simulator():
        print("Testing SocietySimulator...")

        # Create a mock config for testing
        class MockLLMConfig:
            model_name = "mock_gemini"

        class MockAgentConfig:
            count = 1  # Single agent for this test
            movement_speed = 1.0
            social_radius = 5.0
            memory_size = 10
            interaction_radius = 15.0  # Agent's perception radius for objects

        class MockSimulationConfig:
            world_size = (100, 100, 10)  # Made it 3D for position
            max_steps = 3  # Run a few steps to test agent interaction

        class MockOutputConfig:
            directory = "./test_sim_output"
            generated_assets_dir = "./test_sim_output/generated_assets"

        class MockAssetsConfig:  # Added for consistency
            enable_generation = True

        class MockConfig(Config):
            def __init__(self):
                self.llm = MockLLMConfig()
                self.agents = MockAgentConfig()
                self.simulation = MockSimulationConfig()
                self.output = MockOutputConfig()
                self.assets = MockAssetsConfig()  # Added

            @staticmethod
            def default():
                return MockConfig()

        test_config = MockConfig.default()
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        import os

        simulator = SocietySimulator(config=test_config)

        # Manually add a test asset to the world for the agent to find
        test_asset_desc = "a shiny red apple"
        # Simulate asset creation by AssetManager
        mock_asset_file_dir = simulator.asset_manager.output_directory
        if not os.path.exists(mock_asset_file_dir):
            os.makedirs(mock_asset_file_dir)
        mock_asset_file = os.path.join(mock_asset_file_dir, "test_apple.ply")
        # Create dummy file for test
        with open(mock_asset_file, "w") as f:
            f.write("dummy ply data for apple")

        mock_created_asset = Asset(
            asset_id="asset_apple_001",
            agent_id="sim_test_setup",
            description=test_asset_desc,
            file_path=mock_asset_file,
        )
        apple_position = Position(x=10.0, y=10.0, z=0.0)
        simulator.add_world_object(
            mock_created_asset, apple_position, creator_id="simulator_setup"
        )
        print(f"Manually added test object: {test_asset_desc} at {apple_position}")
        print(f"Total world objects: {len(simulator.world_objects)}")

        # Set agent's initial position to be near the apple for testing perception
        if simulator.schedule.agents:
            agent = simulator.schedule.agents[0]
            agent.position = Position(x=5.0, y=5.0, z=0.0)  # Place agent near the apple
            print(
                f"Agent {agent.unique_id} placed at {agent.position} for testing object perception."
            )

        await simulator.run()  # This will call agent.step() which might call _get_nearby_objects
        print(f"Test simulation run completed. Current step: {simulator.current_step}")
        print(
            f"Assets created by AssetManager during run: {len(simulator.asset_manager.assets_created)}"
        )
        if simulator.asset_manager.assets_created:
            for asset_item in simulator.asset_manager.assets_created:
                print(
                    f"  - Asset: {asset_item.asset_id}, Path: {asset_item.file_path}, Desc: {asset_item.description}"
                )

        print(f"Total world objects at end of sim: {len(simulator.world_objects)}")
        for wo in simulator.world_objects:
            print(
                f"  - World Object: {wo.obj_id}, Desc: {wo.description}, Pos: {wo.position}"
            )

        # Clean up dummy file
        if os.path.exists(mock_asset_file):
            os.remove(mock_asset_file)

    asyncio.run(main_test_simulator())
