"""
FLAME GPU 2 Agent Kernels for LLM Society Phase β

This module contains the GPU kernel implementations for various agent behaviors:
- Social interactions and network formation
- Economic trading and market dynamics
- Cultural influence propagation
- Family interactions and kinship networks
- Movement and spatial dynamics

Each kernel is optimized for parallel execution on GPU hardware.
"""

import math
from enum import IntEnum
import pyflamegpu # Required for @pyflamegpu.agent_function and API calls

# Unused numpy import removed. If np.random.random was used in a mock that's now a pyflamegpu.random call, this is fine.
# Module-level constants for radii and interaction limits have been removed.
# These are now configured as environment properties in FlameGPUSimulation
# and accessed within kernels via pyflamegpu.environment.getPropertyTYPE()

class ResourceType(IntEnum):
    """Resource types for trading"""
    FOOD = 0
    MATERIALS = 1
    ENERGY = 2
    LUXURY = 3
    KNOWLEDGE = 4
    TOOLS = 5
    SERVICES = 6
    CURRENCY = 7

# Python FLAME GPU Agent Function for Movement
@pyflamegpu.agent_function
def move_agent_pyfgpu(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    agent_id = pyflamegpu.getVariableInt("agent_id") 
    x = pyflamegpu.getVariableFloat("x")
    y = pyflamegpu.getVariableFloat("y")
    velocity_x = pyflamegpu.getVariableFloat("velocity_x")
    velocity_y = pyflamegpu.getVariableFloat("velocity_y")
    energy = pyflamegpu.getVariableFloat("energy")
    world_width = pyflamegpu.environment.getPropertyFloat("world_width")
    world_height = pyflamegpu.environment.getPropertyFloat("world_height")
    max_speed = 5.0  
    energy_cost_per_move_factor = 0.01 
    if energy > 0.1:
        velocity_x += pyflamegpu.random.uniformFloat(-1.0, 1.0) 
        velocity_y += pyflamegpu.random.uniformFloat(-1.0, 1.0)
        current_speed = math.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)
        if current_speed > max_speed:
            velocity_x = (velocity_x / current_speed) * max_speed
            velocity_y = (velocity_y / current_speed) * max_speed
        new_x = x + velocity_x
        new_y = y + velocity_y
        if new_x <= 0:
            new_x = 0
            velocity_x = -velocity_x
        elif new_x >= world_width:
            new_x = world_width
            velocity_x = -velocity_x
        if new_y <= 0:
            new_y = 0
            velocity_y = -velocity_y
        elif new_y >= world_height:
            new_y = world_height
            velocity_y = -velocity_y
        movement_energy_cost = energy_cost_per_move_factor * (math.sqrt(velocity_x**2 + velocity_y**2) / max_speed if max_speed > 0 else 0) 
        new_energy = max(0.0, energy - movement_energy_cost)
        pyflamegpu.setVariableFloat("x", new_x)
        pyflamegpu.setVariableFloat("y", new_y)
        pyflamegpu.setVariableFloat("velocity_x", velocity_x)
        pyflamegpu.setVariableFloat("velocity_y", velocity_y)
        pyflamegpu.setVariableFloat("energy", new_energy)
    return pyflamegpu.ALIVE

# Python FLAME GPU Agent Function for Core State Update
@pyflamegpu.agent_function
def update_agent_core_state_pyfgpu(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    energy = pyflamegpu.getVariableFloat("energy")
    new_energy = max(0.0, energy - 0.005) 
    food = pyflamegpu.getVariableFloat("food_resources")
    food_consumption_rate = 0.01 
    new_food = food
    if food > food_consumption_rate:
        new_food = food - food_consumption_rate
    else: 
        new_food = 0.0
        new_energy = max(0.0, new_energy - 0.01) 
    pyflamegpu.setVariableFloat("food_resources", new_food)
    pyflamegpu.setVariableFloat("energy", new_energy)
    
    # Aging using environment property
    steps_per_year = pyflamegpu.environment.getPropertyFloat("STEPS_PER_YEAR")
    age_increase = 0.0
    if steps_per_year > 0: # Avoid division by zero
        age_increase = 1.0 / steps_per_year
    
    age = pyflamegpu.getVariableFloat("age")
    new_age = age + age_increase
    pyflamegpu.setVariableFloat("age", new_age)
    return pyflamegpu.ALIVE

# Python FLAME GPU Agent Function for Outputting Social Signals
@pyflamegpu.agent_function
def output_social_signal_pyfgpu(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageOutput) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    agent_id = pyflamegpu.getVariableInt("agent_id")
    x = pyflamegpu.getVariableFloat("x")
    y = pyflamegpu.getVariableFloat("y")
    cultural_group = pyflamegpu.getVariableInt("cultural_group")
    social_reputation = pyflamegpu.getVariableFloat("social_reputation")
    energy = pyflamegpu.getVariableFloat("energy")
    if energy > 0.2:
        interaction_strength = min(1.0, social_reputation * energy) 
        msg = message_out.newMessage()
        msg.setVariableInt("sender_id", agent_id)
        msg.setVariableFloat("sender_x", x)
        msg.setVariableFloat("sender_y", y)
        msg.setVariableInt("cultural_group", cultural_group)
        msg.setVariableFloat("interaction_strength", interaction_strength)
    return pyflamegpu.ALIVE

# Python FLAME GPU Agent Function for Processing Social Interactions
@pyflamegpu.agent_function
def process_social_interactions_pyfgpu(message_in: pyflamegpu.MessageInput, message_out: pyflamegpu.MessageNone) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    agent_x = pyflamegpu.getVariableFloat("x")
    agent_y = pyflamegpu.getVariableFloat("y")
    agent_cultural_group = pyflamegpu.getVariableInt("cultural_group")
    current_happiness = pyflamegpu.getVariableFloat("happiness")
    current_reputation = pyflamegpu.getVariableFloat("social_reputation")
    current_connections = pyflamegpu.getVariableInt("num_connections")
    cultural_affinities = [
        pyflamegpu.getVariableFloat("cultural_affinity_harmonists"),
        pyflamegpu.getVariableFloat("cultural_affinity_builders"),
        pyflamegpu.getVariableFloat("cultural_affinity_guardians"),
        pyflamegpu.getVariableFloat("cultural_affinity_scholars"),
        pyflamegpu.getVariableFloat("cultural_affinity_wanderers")
    ]
    social_interaction_radius_env = pyflamegpu.environment.getPropertyFloat("interaction_radius")
    max_interactions_this_step = pyflamegpu.environment.getPropertyInt("MAX_INTERACTIONS_PER_STEP")
    interactions_processed = 0
    happiness_change = 0.0
    reputation_change = 0.0
    new_connections_this_step = 0
    for msg in message_in:
        if interactions_processed >= max_interactions_this_step: break
        sender_x = msg.getVariableFloat("sender_x")
        sender_y = msg.getVariableFloat("sender_y")
        sender_cultural_group = msg.getVariableInt("cultural_group")
        interaction_strength = msg.getVariableFloat("interaction_strength")
        dx = sender_x - agent_x
        dy = sender_y - agent_y
        distance_sq = dx*dx + dy*dy
        if distance_sq <= social_interaction_radius_env * social_interaction_radius_env:
            distance = math.sqrt(distance_sq)
            if distance <= social_interaction_radius_env: 
                cultural_similarity = 1.0 if sender_cultural_group == agent_cultural_group else 0.3
                distance_factor = 1.0
                if social_interaction_radius_env > 0: 
                    distance_factor = 1.0 - (distance / social_interaction_radius_env)
                interaction_effect = interaction_strength * cultural_similarity * distance_factor
                happiness_change += interaction_effect * 0.05
                reputation_change += interaction_effect * 0.02
                if pyflamegpu.random.uniformFloat(0.0, 1.0) < interaction_effect * 0.1:
                    new_connections_this_step += 1
                if sender_cultural_group != agent_cultural_group:
                    affinity_change = interaction_effect * 0.01
                    if 0 <= sender_cultural_group < len(cultural_affinities):
                        cultural_affinities[sender_cultural_group] += affinity_change
                interactions_processed += 1
    pyflamegpu.setVariableFloat("happiness", max(0.0, min(1.0, current_happiness + happiness_change)))
    pyflamegpu.setVariableFloat("social_reputation", max(0.0, min(1.0, current_reputation + reputation_change)))
    pyflamegpu.setVariableInt("num_connections", current_connections + new_connections_this_step)
    total_affinity = sum(cultural_affinities)
    if total_affinity > 0:
        pyflamegpu.setVariableFloat("cultural_affinity_harmonists", cultural_affinities[0] / total_affinity)
        pyflamegpu.setVariableFloat("cultural_affinity_builders", cultural_affinities[1] / total_affinity)
        pyflamegpu.setVariableFloat("cultural_affinity_guardians", cultural_affinities[2] / total_affinity)
        pyflamegpu.setVariableFloat("cultural_affinity_scholars", cultural_affinities[3] / total_affinity)
        pyflamegpu.setVariableFloat("cultural_affinity_wanderers", cultural_affinities[4] / total_affinity)
    return pyflamegpu.ALIVE

# Python FLAME GPU Agent Function for Outputting Cultural Influence
@pyflamegpu.agent_function
def output_cultural_influence_pyfgpu(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageOutput) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    agent_id = pyflamegpu.getVariableInt("agent_id")
    x = pyflamegpu.getVariableFloat("x")
    y = pyflamegpu.getVariableFloat("y")
    cultural_group = pyflamegpu.getVariableInt("cultural_group")
    social_reputation = pyflamegpu.getVariableFloat("social_reputation")
    happiness = pyflamegpu.getVariableFloat("happiness")
    
    influence_strength_factor = pyflamegpu.environment.getPropertyFloat("INFLUENCE_STRENGTH_FACTOR")
    influence_strength = social_reputation * happiness * influence_strength_factor

    if influence_strength > 0.1: 
        msg = message_out.newMessage()
        msg.setVariableInt("influencer_id", agent_id)
        msg.setVariableFloat("influencer_x", x)
        msg.setVariableFloat("influencer_y", y)
        msg.setVariableInt("cultural_group", cultural_group)
        msg.setVariableFloat("influence_strength", influence_strength)
    return pyflamegpu.ALIVE

# Python FLAME GPU Agent Function for Processing Cultural Influence
@pyflamegpu.agent_function
def process_cultural_influence_pyfgpu(message_in: pyflamegpu.MessageInput, message_out: pyflamegpu.MessageNone) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    agent_x = pyflamegpu.getVariableFloat("x")
    agent_y = pyflamegpu.getVariableFloat("y")
    my_cultural_group_id = pyflamegpu.getVariableInt("cultural_group")
    affinities = [
        pyflamegpu.getVariableFloat("cultural_affinity_harmonists"),
        pyflamegpu.getVariableFloat("cultural_affinity_builders"),
        pyflamegpu.getVariableFloat("cultural_affinity_guardians"),
        pyflamegpu.getVariableFloat("cultural_affinity_scholars"),
        pyflamegpu.getVariableFloat("cultural_affinity_wanderers")
    ]
    cultural_influence_radius_env = pyflamegpu.environment.getPropertyFloat("cultural_influence_radius")
    NUM_CULTURAL_GROUPS = 5 
    influence_received_per_group = [0.0] * NUM_CULTURAL_GROUPS
    total_weighted_influence_strength = 0.0
    
    CULTURAL_SHIFT_FACTOR = pyflamegpu.environment.getPropertyFloat("CULTURAL_SHIFT_FACTOR")
    GROUP_CHANGE_THRESHOLD = pyflamegpu.environment.getPropertyFloat("GROUP_CHANGE_THRESHOLD")

    for msg in message_in:
        influencer_x = msg.getVariableFloat("influencer_x")
        influencer_y = msg.getVariableFloat("influencer_y")
        influencer_group_id = msg.getVariableInt("cultural_group")
        influencer_strength = msg.getVariableFloat("influence_strength")
        dx = influencer_x - agent_x
        dy = influencer_y - agent_y
        distance_sq = dx*dx + dy*dy
        if distance_sq <= cultural_influence_radius_env * cultural_influence_radius_env:
            distance = math.sqrt(distance_sq)
            if distance <= cultural_influence_radius_env:
                distance_factor = 1.0
                if cultural_influence_radius_env > 0:
                    distance_factor = 1.0 - (distance / cultural_influence_radius_env)
                effective_influence = influencer_strength * distance_factor
                if 0 <= influencer_group_id < NUM_CULTURAL_GROUPS:
                    influence_received_per_group[influencer_group_id] += effective_influence
                total_weighted_influence_strength += effective_influence 
    if total_weighted_influence_strength > 0.01:
        for i in range(NUM_CULTURAL_GROUPS):
            if influence_received_per_group[i] > 0:
                influence_ratio = influence_received_per_group[i] / total_weighted_influence_strength
                affinities[i] += influence_ratio * CULTURAL_SHIFT_FACTOR
                affinities[i] = max(0.0, min(1.0, affinities[i]))
        current_total_affinity = sum(affinities)
        if current_total_affinity > 0:
            affinities = [a / current_total_affinity for a in affinities]
        pyflamegpu.setVariableFloat("cultural_affinity_harmonists", affinities[0])
        pyflamegpu.setVariableFloat("cultural_affinity_builders", affinities[1])
        pyflamegpu.setVariableFloat("cultural_affinity_guardians", affinities[2])
        pyflamegpu.setVariableFloat("cultural_affinity_scholars", affinities[3])
        pyflamegpu.setVariableFloat("cultural_affinity_wanderers", affinities[4])
        max_affinity_value = 0.0
        new_cultural_group_id = my_cultural_group_id
        for i in range(NUM_CULTURAL_GROUPS):
            if affinities[i] > max_affinity_value:
                max_affinity_value = affinities[i]
                new_cultural_group_id = i
        if new_cultural_group_id != my_cultural_group_id and max_affinity_value > GROUP_CHANGE_THRESHOLD:
            pyflamegpu.setVariableInt("cultural_group", new_cultural_group_id)
    return pyflamegpu.ALIVE

# Empty Python FLAME GPU Agent Function Stubs for Economic Kernels
@pyflamegpu.agent_function
def output_trade_offers_pyfgpu(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageOutput) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    # Placeholder: Actual logic would involve checking resources, market conditions (from env properties?),
    # and deciding whether to output a trade_offer message.
    # Example: if pyflamegpu.getVariableFloat("food_resources") > 20: # Has surplus food
    #    msg = message_out.newMessage()
    #    msg.setVariableInt("trader_id", pyflamegpu.getVariableInt("agent_id"))
    #    # ... set other message variables for a sell offer ...
    pass
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def process_trade_offers_pyfgpu(message_in: pyflamegpu.MessageInput, message_out: pyflamegpu.MessageNone) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    # Placeholder: Actual logic would involve iterating messages, checking if the current agent
    # wants to accept any offers based on its needs, resources, and the offer's terms.
    # This is complex because it implies a transaction mechanism or state change upon acceptance.
    # For now, it does nothing.
    # Example: for msg in message_in:
    #    resource = msg.getVariableInt("resource_type")
    #    price = msg.getVariableFloat("price")
    #    if pyflamegpu.getVariableFloat("currency") > price: # Can afford and needs resource (simplified)
    #        # Mark for transaction? How is this resolved?
    #        pass 
    pass
    return pyflamegpu.ALIVE

# Empty Python FLAME GPU Agent Function Stubs for Family Kernels
@pyflamegpu.agent_function
def output_family_signals_pyfgpu(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageOutput) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    # Placeholder: Could output messages based on family needs or status.
    pass
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def process_family_interactions_pyfgpu(message_in: pyflamegpu.MessageInput, message_out: pyflamegpu.MessageNone) -> pyflamegpu.FLAMEGPU_AGENT_FUNCTION_RETURN:
    # Placeholder: Could process family-related messages and update agent state.
    pass
    return pyflamegpu.ALIVE

# Old Kernel placeholder classes are no longer needed as functions are standalone and Python-based.
# If any RTC functions were still used, their respective classes and placeholder strings would remain.
# For now, assuming all are (or will be) Python agent functions or are handled at CPU level.

# class SocialInteractionKernel: ... (removed)
# class EconomicTradeKernel: ... (removed)
# class CulturalInfluenceKernel: ... (removed)
# class MovementKernel: ... (removed)
# class FamilyInteractionKernel: ... (removed)
# class ResourceManagementKernel: ... (removed)

# Placeholder for AgentType enum if not already defined or imported
# Should match the one in flame_gpu_simulation.py
# class AgentType(IntEnum):
#     FARMER = 0
#     CRAFTSMAN = 1
#     TRADER = 2
#     SCHOLAR = 3
#     LEADER = 4
#     UNEMPLOYED = 5 