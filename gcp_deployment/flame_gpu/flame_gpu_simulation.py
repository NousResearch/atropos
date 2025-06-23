"""
FLAME GPU 2 Simulation Engine for LLM Society Phase β

This module provides the main simulation engine that coordinates all GPU kernels
and manages the complex agent state for social, economic, and cultural systems.
"""

import asyncio  # Added asyncio
import logging
import sys  # For self.simulation.initialise
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Removed Mock FLAME GPU 2 imports
import pyflamegpu  # Direct import

from .agent_kernels import (
    output_cultural_influence_pyfgpu,  # Import new cultural output function
)
from .agent_kernels import output_family_signals_pyfgpu  # Import new family stub
from .agent_kernels import (
    output_social_signal_pyfgpu,  # Import new social output function
)
from .agent_kernels import output_trade_offers_pyfgpu  # Import new economic stub
from .agent_kernels import (
    process_cultural_influence_pyfgpu,  # Import new cultural process function
)
from .agent_kernels import process_family_interactions_pyfgpu  # Import new family stub
from .agent_kernels import (
    process_social_interactions_pyfgpu,  # Import new social process function
)
from .agent_kernels import process_trade_offers_pyfgpu  # Import new economic stub
from .agent_kernels import (
    update_agent_core_state_pyfgpu,  # Import the new Python agent function
)
from .agent_kernels import (
    CulturalInfluenceKernel,
    EconomicTradeKernel,
    FamilyInteractionKernel,
    MovementKernel,
    ResourceManagementKernel,
    SocialInteractionKernel,
    move_agent_pyfgpu,
)

logger = logging.getLogger(__name__)

# Removed MockMessageInstanceWrapper as it was for the mock


class AgentType(IntEnum):
    """Agent profession types for GPU kernels"""

    FARMER = 0
    CRAFTSMAN = 1
    TRADER = 2
    SCHOLAR = 3
    LEADER = 4
    UNEMPLOYED = 5


class CulturalGroup(IntEnum):
    """Cultural group identifiers"""

    HARMONISTS = 0
    BUILDERS = 1
    GUARDIANS = 2
    SCHOLARS = 3
    WANDERERS = 4


@dataclass
class SimulationConfig:
    """Configuration for FLAME GPU simulation"""

    max_agents: int = 2500
    world_width: float = 1000.0
    world_height: float = 1000.0
    interaction_radius: float = 50.0
    trade_radius: float = 75.0
    cultural_influence_radius: float = 100.0
    enable_families: bool = True
    enable_economics: bool = True
    enable_culture: bool = True
    gpu_device_id: int = 0
    steps_per_epoch: int = 100


class FlameGPUSimulation:
    """
    Main FLAME GPU simulation engine for LLM Society Phase β

    Coordinates all GPU kernels for social, economic, and cultural interactions
    while maintaining compatibility with the existing LLM agent system.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.model_description = pyflamegpu.ModelDescription("LLMSocietySimulation")
        self.simulation: Optional[pyflamegpu.CUDASimulation] = None
        self.step_count = 0
        self.loop = asyncio.get_event_loop()  # Store the loop

        # Environment properties (can be set on model_description)
        env = self.model_description.Environment()
        env.newPropertyFloat("world_width", self.config.world_width)
        env.newPropertyFloat("world_height", self.config.world_height)
        env.newPropertyFloat("interaction_radius", self.config.interaction_radius)
        env.newPropertyFloat("trade_radius", self.config.trade_radius)
        env.newPropertyFloat(
            "cultural_influence_radius", self.config.cultural_influence_radius
        )
        env.newPropertyInt("MAX_INTERACTIONS_PER_STEP", MAX_INTERACTIONS_PER_STEP)
        env.newPropertyFloat("CULTURAL_SHIFT_FACTOR", 0.02)
        env.newPropertyFloat("GROUP_CHANGE_THRESHOLD", 0.4)
        env.newPropertyInt("MAX_TRADE_OFFERS_PER_STEP", MAX_TRADE_OFFERS_PER_STEP)
        env.newPropertyFloat("STEPS_PER_YEAR", 365.0)
        env.newPropertyFloat("INFLUENCE_STRENGTH_FACTOR", 0.5)

        # Performance tracking
        self.kernel_times = {}
        self.total_simulation_time = 0.0

        # Initialize FLAME GPU model structure
        self._initialize_model_structure()  # Renamed from _initialize_model
        self._define_model_layers_and_functions()  # Define layers after basic model structure

        logger.info(
            f"FlameGPU model description and layers defined for {config.max_agents} agents"
        )

    def _blocking_init_simulation_object(self):
        if not self.simulation:  # Create CUDASimulation object if it doesn't exist
            logger.info("Creating CUDASimulation object for FlameGPU.")
            self.simulation = pyflamegpu.CUDASimulation(self.model_description)
            logger.debug(
                f"Setting CUDASimulation device ID to: {self.config.gpu_device_id}"
            )
            self.simulation.setDeviceID(self.config.gpu_device_id)
            logger.debug(
                "Calling CUDASimulation.initialise(sys.argv)..._blocking_init_simulation_object"
            )
            self.simulation.initialise(sys.argv)
            logger.info(
                "CUDASimulation object initialised. _blocking_init_simulation_object"
            )

    def _blocking_set_population_data(self, agent_data_list: List[Dict]):
        num_agents = len(agent_data_list)
        population_data_obj = pyflamegpu.AgentPopulation(
            self.model_description.Agent("SocietyAgent"), num_agents
        )
        for i, agent_state_dict in enumerate(agent_data_list):
            agent_instance = population_data_obj.Agent(i)
            for key, value in agent_state_dict.items():
                try:
                    if isinstance(value, float):
                        agent_instance.setVariableFloat(key, value)
                    elif isinstance(value, int):
                        agent_instance.setVariableInt(key, value)
                    else:
                        logger.warning(
                            f"Skipping unknown type for FlameGPU agent variable {key}: {type(value)}"
                        )
                except Exception as e_set_var:
                    logger.error(
                        "Error setting FlameGPU agent variable "{key}' for agent {agent_state_dict.get('agent_id', 'UNKNOWN')}: {e_set_var}"
                    )
        logger.debug(
            f"Attempting to set agent population data in FlameGPU for {num_agents} agents. _blocking_set_population_data"
        )
        if self.simulation:
            self.simulation.setPopulationData(population_data_obj)
        else:
            raise RuntimeError(
                "FlameGPU simulation object not initialized before setting population data."
            )

    async def initialize_agents(self, agent_states: List[Dict]) -> bool:
        try:
            num_agents = min(len(agent_states), self.config.max_agents)
            logger.info(
                f"Async Initializing {num_agents} agents for FLAME GPU simulation."
            )
            if not self.simulation:
                await self.loop.run_in_executor(
                    None, self._blocking_init_simulation_object
                )
            await self.loop.run_in_executor(
                None, self._blocking_set_population_data, agent_states[:num_agents]
            )
            logger.info(
                f"Successfully initialized population for {num_agents} agents in FLAME GPU (async)."
            )
            return True
        except pyflamegpu.FLAMEGPUException as fgpu_e:
            logger.error(
                f"FLAMEGPUException during async agent initialization: {fgpu_e}",
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to async initialize FLAME GPU agents: {e}", exc_info=True
            )
            return False

    async def prime_agent_states_for_step(self, agent_states_from_python: List[Dict]):
        if not self.simulation:
            logger.error("FLAME GPU sim not initialized. Cannot prime states.")
            return
        try:
            logger.debug(
                f"Async priming {len(agent_states_from_python)} agent states into FLAME GPU."
            )
            await self.loop.run_in_executor(
                None, self._blocking_set_population_data, agent_states_from_python
            )
            logger.debug("FlameGPU agent states primed (async).")
        except pyflamegpu.FLAMEGPUException as fgpu_e:
            logger.error(
                f"FLAMEGPUException during async agent state priming: {fgpu_e}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Error async priming agent states for FlameGPU: {e}", exc_info=True
            )

    async def run_simulation_step(self) -> Dict:
        step_start_time = time.time()
        try:
            if not self.simulation:
                return {
                    "error": "Simulation not initialized",
                    "step_number": self.step_count,
                }
            logger.debug(
                f"Async executing FLAME GPU step {self.step_count}. Calling simulation.step()..."
            )
            await self.loop.run_in_executor(
                None, self.simulation.step
            )  # self.simulation.step() is blocking
            logger.debug(
                f"FLAME GPU step {self.step_count} execution finished (async)."
            )
            self.step_count += 1
            total_step_time = time.time() - step_start_time
            return {
                "total_step_time": total_step_time,
                "step_number": self.step_count - 1,
            }
        except pyflamegpu.FLAMEGPUException as fgpu_e:
            logger.error(
                f"FLAMEGPUException in async simulation step {self.step_count}: {fgpu_e}",
                exc_info=True,
            )
            return {
                "error": f"FLAMEGPUException: {fgpu_e}",
                "step_number": self.step_count,
            }
        except Exception as e:
            logger.error(
                f"Error in async FLAME GPU simulation step {self.step_count}: {e}",
                exc_info=True,
            )
            return {"error": str(e), "step_number": self.step_count}

    async def get_agent_states(self) -> List[Dict]:
        if not self.simulation:
            logger.warning("FLAME GPU Sim not init. Cannot get states.")
            return []

        def _blocking_get_population_data(
            sim: pyflamegpu.CUDASimulation, model_desc: pyflamegpu.ModelDescription
        ) -> pyflamegpu.AgentPopulation:
            return sim.getAgentPopulation(model_desc.Agent("SocietyAgent"))

        try:
            logger.debug("Async getting agent population data from FlameGPU...")
            agent_vector = await self.loop.run_in_executor(
                None,
                _blocking_get_population_data,
                self.simulation,
                self.model_description,
            )
            logger.debug(
                f"Retrieved {len(agent_vector)} agent states from FlameGPU (async)."
            )
            agent_states = []
            for i in range(len(agent_vector)):
                reader = agent_vector.Agent(i)
                state = {
                    v.name: reader.getVariable(v.name)
                    for v in self.model_description.Agent(
                        "SocietyAgent"
                    ).variables.values()
                }
                state["cultural_affinities"] = {
                    "Harmonists": reader.getVariableFloat(
                        "cultural_affinity_harmonists"
                    ),
                    "Builders": reader.getVariableFloat("cultural_affinity_builders"),
                    "Guardians": reader.getVariableFloat("cultural_affinity_guardians"),
                    "Scholars": reader.getVariableFloat("cultural_affinity_scholars"),
                    "Wanderers": reader.getVariableFloat("cultural_affinity_wanderers"),
                }
                agent_states.append(state)
            return agent_states
        except pyflamegpu.FLAMEGPUException as fgpu_e:
            logger.error(
                f"FLAMEGPUException async getting agent states: {fgpu_e}", exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"Error async extracting agent states from FLAME GPU: {e}",
                exc_info=True,
            )
            return []

    def _initialize_model_structure(self):  # Renamed
        """Initialize FLAME GPU 2 model structure (agents, messages, functions)"""
        # self.model_description is used here (was self.model)

        # Define agent type with all necessary properties
        agent = self.model_description.newAgent("SocietyAgent")

        # Spatial properties
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
        agent.newVariableFloat("velocity_x")
        agent.newVariableFloat("velocity_y")

        # Basic agent properties
        agent.newVariableInt("agent_id")  # Keep as int for GPU
        agent.newVariableInt("agent_type")  # AgentType enum
        agent.newVariableFloat("age")
        agent.newVariableFloat("energy")
        agent.newVariableFloat("happiness")
        agent.newVariableFloat("health")

        # Employment status
        agent.newVariableInt("employed")  # 0 = unemployed, 1 = employed

        # Social properties
        agent.newVariableInt("family_id")
        agent.newVariableInt("cultural_group")  # CulturalGroup enum
        agent.newVariableFloat("social_reputation")
        agent.newVariableInt("num_connections")

        # Economic properties
        agent.newVariableFloat("wealth")
        agent.newVariableFloat("food_resources")
        agent.newVariableFloat("material_resources")
        agent.newVariableFloat("energy_resources")
        agent.newVariableFloat("luxury_resources")
        agent.newVariableFloat("knowledge_resources")
        agent.newVariableFloat("tools_resources")
        agent.newVariableFloat("services_resources")
        agent.newVariableFloat("currency")

        # Banking and finance
        agent.newVariableFloat("credit_score")
        agent.newVariableFloat("total_debt")
        agent.newVariableFloat("monthly_income")

        # Cultural properties
        agent.newVariableFloat("cultural_affinity_harmonists")
        agent.newVariableFloat("cultural_affinity_builders")
        agent.newVariableFloat("cultural_affinity_guardians")
        agent.newVariableFloat("cultural_affinity_scholars")
        agent.newVariableFloat("cultural_affinity_wanderers")

        # Message types for agent communication
        self._define_message_types()

        # Agent functions (kernels) - these are now declarations for FLAME GPU
        self._define_agent_function_declarations()  # Renamed

        logger.info("FLAME GPU agent and message structure defined.")

    def _define_message_types(self):
        """Define message types for agent communication"""
        # Using self.model_description
        social_msg = self.model_description.newMessageBruteForce("social_interaction")
        social_msg.newVariableInt("sender_id")  # Changed to Int to match agent_id type
        social_msg.newVariableFloat("sender_x")
        social_msg.newVariableFloat("sender_y")
        social_msg.newVariableInt("cultural_group")  # Changed to Int
        social_msg.newVariableFloat("interaction_strength")

        trade_msg = self.model_description.newMessageBruteForce("trade_offer")
        trade_msg.newVariableInt("trader_id")  # Changed to Int
        trade_msg.newVariableFloat("trader_x")
        trade_msg.newVariableFloat("trader_y")
        trade_msg.newVariableInt(
            "resource_type"
        )  # Assuming ResourceType enum maps to int
        trade_msg.newVariableFloat("quantity")
        trade_msg.newVariableFloat("price")
        trade_msg.newVariableInt("is_buy_order")  # 0 for sell, 1 for buy

        cultural_msg = self.model_description.newMessageBruteForce("cultural_influence")
        cultural_msg.newVariableInt("influencer_id")  # Changed to Int
        cultural_msg.newVariableFloat("influencer_x")
        cultural_msg.newVariableFloat("influencer_y")
        cultural_msg.newVariableInt("cultural_group")  # Changed to Int
        cultural_msg.newVariableFloat("influence_strength")

        family_msg = self.model_description.newMessageBruteForce("family_interaction")
        family_msg.newVariableInt("family_member_id")  # Changed to Int
        family_msg.newVariableInt("family_id")  # Changed to Int
        family_msg.newVariableInt("interaction_type")  # e.g., 0=support, 1=conflict
        family_msg.newVariableFloat("value")

    def _define_agent_function_declarations(self):  # Renamed
        """
        Declare agent functions for FLAME GPU.
        The actual kernel code (Python with @pyflamegpu.agent_function or C++)
        will be associated with these function names later or by the FLAME GPU runtime.
        """
        # agent = self.model_description.getAgent("SocietyAgent") # Get agent description

        # Example: agent.newRTCFunction("move_agent_kernel", self.move_agent_kernel_code) # If RTC
        # Or, if using Python agent functions, they are typically bound later or automatically discovered.
        # For now, we acknowledge that the *kernels* from agent_kernels.py need to be
        # implemented as actual FLAME GPU agent functions (e.g. Python with decorators or C++).
        # The strings like "move_agent" are now just conceptual links.

        # Placeholder: Layers will be defined later to call these conceptual functions.
        # Layer 0: Movement
        # Layer 1: Social Output, Social Process
        # etc.

        logger.info("Agent function declarations noted. Actual kernel bindings TBD.")

    def _define_model_layers_and_functions(self):
        # Agent functions must be defined on the agent description.
        # The actual implementation of these functions (kernel code) needs to be provided elsewhere,
        # typically as Python functions with @pyflamegpu.agent_function decorators or C++ code for RTC.
        # For now, we declare the function names that the layers will reference.
        agent_desc = self.model_description.Agent("SocietyAgent")

        # Declare agent functions (these names must match the actual kernel function names)
        # Movement - Now uses a Python agent function
        agent_desc.newAgentFunction("move_agent", move_agent_pyfgpu)
        # Social - Now uses Python agent functions
        agent_desc.newAgentFunction("output_social_signal", output_social_signal_pyfgpu)
        agent_desc.newAgentFunction(
            "process_social_interactions", process_social_interactions_pyfgpu
        )
        # Economic
        agent_desc.newAgentFunction("output_trade_offers", output_trade_offers_pyfgpu)
        agent_desc.newAgentFunction("process_trade_offers", process_trade_offers_pyfgpu)
        # Cultural
        agent_desc.newAgentFunction(
            "output_cultural_influence", output_cultural_influence_pyfgpu
        )
        agent_desc.newAgentFunction(
            "process_cultural_influence", process_cultural_influence_pyfgpu
        )
        # Family
        agent_desc.newAgentFunction(
            "output_family_signals", output_family_signals_pyfgpu
        )
        agent_desc.newAgentFunction(
            "process_family_interactions", process_family_interactions_pyfgpu
        )
        # Core Update - Now uses a Python agent function
        agent_desc.newAgentFunction(
            "update_agent_core_state", update_agent_core_state_pyfgpu
        )

        # Define Layers
        # Layer 0: Core State Update & Movement
        layer0 = self.model_description.newLayer("CoreUpdateAndMovementLayer")
        layer0.addAgentFunction(
            "update_agent_core_state"
        )  # Conceptually ResourceManagementKernel.update_resources_and_lifecycle
        layer0.addAgentFunction("move_agent")  # Conceptually MovementKernel.move_agent

        # Layer 1: Social Interactions
        layer1 = self.model_description.newLayer("SocialLayer")
        layer1.addAgentFunction(
            "output_social_signal"
        )  # Conceptually SocialInteractionKernel.output_social_signal
        layer1.addAgentFunction(
            "process_social_interactions"
        )  # Conceptually SocialInteractionKernel.process_social_interactions

        # Layer 2: Economic Activity
        layer2 = self.model_description.newLayer("EconomicLayer")
        layer2.addAgentFunction(
            "output_trade_offers"
        )  # Conceptually EconomicTradeKernel.output_trade_offers
        layer2.addAgentFunction(
            "process_trade_offers"
        )  # Conceptually EconomicTradeKernel.process_trade_offers

        # Layer 3: Cultural Influence
        layer3 = self.model_description.newLayer("CulturalLayer")
        layer3.addAgentFunction(
            "output_cultural_influence"
        )  # Conceptually CulturalInfluenceKernel.output_cultural_influence
        layer3.addAgentFunction(
            "process_cultural_influence"
        )  # Conceptually CulturalInfluenceKernel.process_cultural_influence

        # Layer 4: Family Dynamics
        layer4 = self.model_description.newLayer("FamilyLayer")
        layer4.addAgentFunction(
            "output_family_signals"
        )  # Conceptually FamilyInteractionKernel.output_family_signals
        layer4.addAgentFunction(
            "process_family_interactions"
        )  # Conceptually FamilyInteractionKernel.process_family_interactions

        logger.info("FLAME GPU model layers and Python agent functions defined.")
        logger.warning(
            "Other functions still use RTC placeholders. Actual FLAME GPU kernels need to be implemented."
        )

    def get_performance_metrics(self) -> Dict:
        """
        Get detailed performance metrics for the simulation

        Returns:
            Dict: Performance metrics and statistics
        """
        metrics = {
            "total_steps": self.step_count,
            "total_simulation_time": self.total_simulation_time,
            "average_step_time": self.total_simulation_time / max(1, self.step_count),
            "kernel_performance": {},  # This will be harder to get directly per-kernel without FLAME GPU's own logging
        }

        # Simplified timing, as detailed kernel times are harder to get without specific FLAME GPU calls
        if (
            "total_step_time" in self.kernel_times
            and self.kernel_times["total_step_time"]
        ):
            metrics["kernel_performance"]["total_step_time_gpu"] = (
                {  # Renamed to reflect it's the GPU step
                    "average": np.mean(self.kernel_times["total_step_time"]),
                    "min": np.min(self.kernel_times["total_step_time"]),
                    "max": np.max(self.kernel_times["total_step_time"]),
                    "std": np.std(self.kernel_times["total_step_time"]),
                    "total": np.sum(self.kernel_times["total_step_time"]),
                }
            )

        if self.step_count > 0 and self.simulation:  # Check if simulation object exists
            # Get population size from the simulation object if possible, or config
            # num_agents = self.simulation.getAgentPopulation(self.model_description.getAgent("SocietyAgent")).size() # Example
            num_agents = (
                self.config.max_agents
            )  # Fallback to config, actual population might vary
            if metrics["average_step_time"] > 0:
                metrics["agent_throughput"] = (
                    num_agents * self.step_count / self.total_simulation_time
                )
                metrics["agents_per_second"] = num_agents / metrics["average_step_time"]

        return metrics

    def shutdown(self):
        """Clean up GPU resources"""
        try:
            # if self.agent_data: # self.agent_data removed
            #     self.agent_data.clear()

            # FLAME GPU CUDASimulation object does not have an explicit shutdown/cleanup in Python.
            # It's typically handled by its destructor when the object goes out of scope.
            self.simulation = None  # Allow GC to collect if no other refs

            logger.info("FLAME GPU simulation resources marked for cleanup.")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


# Placeholder kernel code strings - these need to be actual C++ code or reference Python agent functions
# For Python agent functions, you typically don't define them here but in their own .py files and decorate them.
MovementKernel.move_agent_kernel_code_placeholder = (
    "// C++ or Python FLAME GPU kernel for move_agent"
)
SocialInteractionKernel.output_social_signal_kernel_code_placeholder = (
    "// ... output_social_signal kernel ..."
)
SocialInteractionKernel.process_social_interactions_kernel_code_placeholder = (
    "// ... process_social_interactions kernel ..."
)
EconomicTradeKernel.output_trade_offers_kernel_code_placeholder = (
    "// ... output_trade_offers kernel ..."
)
EconomicTradeKernel.process_trade_offers_kernel_code_placeholder = (
    "// ... process_trade_offers kernel ..."
)
CulturalInfluenceKernel.output_cultural_influence_kernel_code_placeholder = (
    "// ... output_cultural_influence kernel ..."
)
CulturalInfluenceKernel.process_cultural_influence_kernel_code_placeholder = (
    "// ... process_cultural_influence kernel ..."
)
FamilyInteractionKernel.output_family_signals_kernel_code_placeholder = (
    "// ... output_family_signals kernel ..."
)
FamilyInteractionKernel.process_family_interactions_kernel_code_placeholder = (
    "// ... process_family_interactions kernel ..."
)
ResourceManagementKernel.update_resources_and_lifecycle_kernel_code_placeholder = (
    "// ... update_agent_core_state kernel ..."
)
