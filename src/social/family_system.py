"""
Family System for LLM Society Simulation Phase β
Implements multi-generational families, inheritance, and kinship networks
"""

import asyncio  # Added for async event logging
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.readwrite import json_graph

# from src.database.database_handler import DatabaseHandler # For type hinting if needed

logger = logging.getLogger(__name__)


class FamilyType(Enum):
    """Types of family structures"""

    NUCLEAR = "nuclear"  # Parents + children
    EXTENDED = "extended"  # Multi-generational
    SINGLE_PARENT = "single_parent"
    CHILDLESS = "childless"
    CLAN = "clan"  # Large extended family

    def __str__(self):
        return self.value


class RelationshipType(Enum):
    """Types of family relationships"""

    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    SPOUSE = "spouse"
    GRANDPARENT = "grandparent"
    GRANDCHILD = "grandchild"
    AUNT_UNCLE = "aunt_uncle"
    COUSIN = "cousin"
    NIECE_NEPHEW = "niece_nephew"

    def __str__(self):
        return self.value


@dataclass
class FamilyMember:
    """Individual family member data"""

    agent_id: str
    generation: int  # 0 = oldest, higher = younger
    birth_family_id: Optional[str] = None
    married_family_id: Optional[str] = None
    relationships: Dict[str, RelationshipType] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "birth_family_id": self.birth_family_id,
            "married_family_id": self.married_family_id,
            "relationships": {k: v.value for k, v in self.relationships.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FamilyMember":
        relationships = {
            k: RelationshipType(v) for k, v in data.get("relationships", {}).items()
        }
        return cls(
            agent_id=data["agent_id"],
            generation=data["generation"],
            birth_family_id=data.get("birth_family_id"),
            married_family_id=data.get("married_family_id"),
            relationships=relationships,
        )


@dataclass
class Family:
    """Family unit with members, resources, and traditions"""

    family_id: str
    family_name: str
    family_type: FamilyType
    members: Set[str] = field(default_factory=set)

    # Resources and inheritance
    family_wealth: float = 0.0
    family_property: List[str] = field(default_factory=list)
    family_heirlooms: List[Dict[str, Any]] = field(default_factory=list)

    # Family characteristics
    family_traditions: List[str] = field(default_factory=list)
    family_values: Dict[str, float] = field(default_factory=dict)
    family_reputation: float = 0.5

    # Genealogy
    founding_generation: int = 0
    current_generation: int = 0
    family_tree: nx.Graph = field(default_factory=nx.Graph)

    # Metadata
    created_timestamp: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "family_name": self.family_name,
            "family_type": self.family_type.value,
            "members": list(self.members),
            "family_wealth": self.family_wealth,
            "family_property": self.family_property,
            "family_heirlooms": self.family_heirlooms,
            "family_traditions": self.family_traditions,
            "family_values": self.family_values,
            "family_reputation": self.family_reputation,
            "founding_generation": self.founding_generation,
            "current_generation": self.current_generation,
            "family_tree": json_graph.node_link_data(self.family_tree),
            "created_timestamp": self.created_timestamp,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Family":
        family_tree_data = data.get("family_tree")
        family_tree = (
            json_graph.node_link_graph(family_tree_data)
            if family_tree_data
            else nx.Graph()
        )

        init_data = data.copy()
        init_data["family_type"] = FamilyType(data["family_type"])
        init_data["members"] = set(data.get("members", []))
        init_data["family_tree"] = family_tree

        return cls(**init_data)


class InheritanceSystem:
    """Manages inheritance rules and wealth transfer"""

    def __init__(self):
        self.inheritance_rules = {
            "primogeniture": 0.5,  # Eldest child gets majority
            "equal_division": 0.3,  # Equal distribution
            "patrilineal": 0.15,  # Male heirs preference
            "merit_based": 0.05,  # Based on achievements
        }

    def calculate_inheritance(
        self,
        deceased_wealth: float,
        family_members: List[str],
        family_relationships: Dict[str, RelationshipType],
    ) -> Dict[str, float]:
        """Calculate inheritance distribution"""

        if deceased_wealth <= 0:
            return {}

        # Find eligible heirs (children, then spouse, then siblings)
        children = [
            agent_id
            for agent_id, rel in family_relationships.items()
            if rel == RelationshipType.CHILD
        ]
        spouse = [
            agent_id
            for agent_id, rel in family_relationships.items()
            if rel == RelationshipType.SPOUSE
        ]
        siblings = [
            agent_id
            for agent_id, rel in family_relationships.items()
            if rel == RelationshipType.SIBLING
        ]

        inheritance_distribution = {}

        # Determine inheritance rule (culturally influenced)
        rule = random.choices(
            list(self.inheritance_rules.keys()),
            weights=list(self.inheritance_rules.values()),
        )[0]

        if children:
            inheritance_distribution.update(
                self._distribute_to_children(deceased_wealth, children, rule)
            )
        elif spouse:
            # Spouse inherits everything if no children
            inheritance_distribution[spouse[0]] = deceased_wealth
        elif siblings:
            # Equal division among siblings
            share = deceased_wealth / len(siblings)
            for sibling in siblings:
                inheritance_distribution[sibling] = share

        return inheritance_distribution

    def _distribute_to_children(
        self, wealth: float, children: List[str], rule: str
    ) -> Dict[str, float]:
        """Distribute wealth among children based on rule"""
        distribution = {}

        if rule == "primogeniture":
            # Eldest gets 60%, others split remainder
            eldest = children[0]  # Assume first is eldest
            distribution[eldest] = wealth * 0.6

            if len(children) > 1:
                remaining = wealth * 0.4
                share = remaining / (len(children) - 1)
                for child in children[1:]:
                    distribution[child] = share

        elif rule == "equal_division":
            # Equal shares for all children
            share = wealth / len(children)
            for child in children:
                distribution[child] = share

        elif rule == "patrilineal":
            # Male children get larger shares (if gender system implemented)
            # For now, treat as equal division
            share = wealth / len(children)
            for child in children:
                distribution[child] = share

        elif rule == "merit_based":
            # Would consider achievements, for now equal division
            share = wealth / len(children)
            for child in children:
                distribution[child] = share

        return distribution


class FamilySystem:
    """
    Manages all family relationships, inheritance, and dynamics
    """

    def __init__(
        self,
        database_handler: Optional[Any] = None,
        current_step_getter: Optional[callable] = None,
    ):
        self.families: Dict[str, Family] = {}
        self.family_members: Dict[str, FamilyMember] = {}
        self.kinship_graph = nx.Graph()
        self.inheritance_system = InheritanceSystem()

        # Family formation parameters
        self.marriage_probability = 0.001  # Per step probability
        self.reproduction_probability = 0.005  # For married couples
        self.divorce_probability = 0.0001  # Per step for married couples

        # Cultural family names
        self.family_surnames = [
            "Smith",
            "Johnson",
            "Williams",
            "Brown",
            "Jones",
            "Garcia",
            "Miller",
            "Davis",
            "Rodriguez",
            "Martinez",
            "Hernandez",
            "Lopez",
            "Gonzalez",
            "Wilson",
            "Anderson",
            "Thomas",
            "Taylor",
            "Moore",
            "Jackson",
            "Martin",
            "Lee",
            "Perez",
            "Thompson",
            "White",
            "Harris",
            "Sanchez",
            "Clark",
            "Ramirez",
            "Lewis",
            "Robinson",
            "Walker",
            "Young",
            "Allen",
            "King",
        ]

        self.db_handler = database_handler
        self.get_current_step = current_step_getter

        logger.info("Family System initialized")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "families": {fid: fam.to_dict() for fid, fam in self.families.items()},
            "family_members": {
                agent_id: mem.to_dict() for agent_id, mem in self.family_members.items()
            },
            "kinship_graph": json_graph.node_link_data(self.kinship_graph),
            # Probabilities and surnames are part of initial config, not dynamic state to save.
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        database_handler: Optional[Any] = None,
        current_step_getter: Optional[callable] = None,
    ) -> "FamilySystem":
        system = cls(
            database_handler=database_handler, current_step_getter=current_step_getter
        )
        # Configuration attributes (probabilities, surnames) are set by __init__.
        # If they were dynamic and saved, they would be loaded here.

        system.families = {
            fid: Family.from_dict(fam_data)
            for fid, fam_data in data.get("families", {}).items()
        }
        system.family_members = {
            agent_id: FamilyMember.from_dict(mem_data)
            for agent_id, mem_data in data.get("family_members", {}).items()
        }

        kinship_graph_data = data.get("kinship_graph")
        if kinship_graph_data:
            system.kinship_graph = json_graph.node_link_graph(kinship_graph_data)
        else:
            system.kinship_graph = (
                nx.Graph()
            )  # Ensure it's an empty graph if not in data
            # Optionally, rebuild kinship_graph from family_members.relationships if needed for consistency
            # for fm_id, fm_data_obj in system.family_members.items():
            #     for related_id, relationship in fm_data_obj.relationships.items():
            #         system.kinship_graph.add_edge(fm_id, related_id, relationship=relationship.value)

        logger.info(
            f"FamilySystem state restored with {len(system.families)} families, {len(system.family_members)} members."
        )
        return system

    def record_agent_family_interaction(
        self,
        agent_id: str,
        family_id: str,
        interaction_description: str,
        current_step: int,
    ):
        """Record a generic interaction an agent has with or about their family.
        DEPRECATED in favor of more specific process_... methods if possible, but can be a fallback.
        """
        if family_id not in self.families:
            logger.warning(
                f"Family {family_id} not found for agent {agent_id} (generic interaction)."
            )
            return

        family = self.families[family_id]
        lower_description = interaction_description.lower()
        logger.info(
            f"Step {current_step}: Agent {agent_id} (Family {family_id}) generic interaction: {interaction_description}"
        )

        # Basic reputation effects from generic interaction - kept for fallback or broad events
        positive_keywords = [
            "help",
            "support",
            "celebrate",
            "discuss positive",
            "share joy",
            "care for",
        ]
        negative_keywords = [
            "argue",
            "complain",
            "disagree strongly",
            "conflict with",
            "neglect",
        ]

        if any(keyword in lower_description for keyword in positive_keywords):
            family.family_reputation = min(
                1.0, family.family_reputation + 0.005
            )  # Less impact than specific positive actions
        if any(keyword in lower_description for keyword in negative_keywords):
            family.family_reputation = max(0.0, family.family_reputation - 0.01)

        family.last_updated = time.time()

    def process_interaction_with_spouse(
        self,
        agent_id: str,
        spouse_id: str,
        family_id: str,
        description: str,
        current_step: int,
    ):
        try:
            if family_id not in self.families:
                logger.warning(
                    f"Family {family_id} not found for spouse interaction between {agent_id} and {spouse_id}."
                )
                return
            family = self.families[family_id]
            lower_description = description.lower()
            logger.info(
                f"Step {current_step}: Agent {agent_id} interacting with spouse {spouse_id} (Family {family_id}): {description}"
            )

            # Reputation & Relationship (conceptual)
            if any(
                kw in lower_description
                for kw in [
                    "discuss positive",
                    "celebrate",
                    "support spouse",
                    "share joy",
                ]
            ):
                family.family_reputation = min(1.0, family.family_reputation + 0.015)
                # Conceptual: increase marital_satisfaction(agent_id, spouse_id)
            elif any(
                kw in lower_description
                for kw in ["argue with spouse", "disagree strongly with spouse"]
            ):
                family.family_reputation = max(0.0, family.family_reputation - 0.02)
                # Conceptual: decrease marital_satisfaction(agent_id, spouse_id)

            # Financial discussions with spouse
            financial_keywords = [
                "budget",
                "money",
                "finance",
                "save",
                "spend",
                "earn",
                "cost",
                "debt",
                "income",
            ]
            if any(keyword in lower_description for keyword in financial_keywords):
                logger.debug(
                    f"Family {family_id}: Financial discussion between spouses detected: {description}"
                )
                if (
                    "save money" in lower_description
                    or "plan finances" in lower_description
                ):
                    family.family_wealth += random.uniform(
                        0.2, 1.5
                    )  # Slightly more impactful with spouse
                    family.family_reputation = min(1.0, family.family_reputation + 0.01)
                elif (
                    "large purchase" in lower_description
                    or "major spending" in lower_description
                ):
                    family.family_wealth -= random.uniform(0.2, 1.5)
            family.last_updated = time.time()
        except Exception as e:
            logger.error(
                f"Error processing spouse interaction for {agent_id} in family {family_id}: {e}",
                exc_info=True,
            )

    def process_interaction_with_child(
        self,
        agent_id: str,
        child_id: str,
        family_id: str,
        description: str,
        current_step: int,
    ):
        try:
            if family_id not in self.families:
                logger.warning(
                    f"Family {family_id} not found for parent-child interaction ({agent_id} with {child_id})."
                )
                return
            family = self.families[family_id]
            lower_description = description.lower()
            logger.info(
                f"Step {current_step}: Agent {agent_id} interacting with child {child_id} (Family {family_id}): {description}"
            )

            # Child-related activities
            positive_child_keywords = [
                "teach child",
                "play with child",
                "read to child",
                "guide child",
                "help child homework",
                "nurture child",
            ]
            negative_child_keywords = [
                "scold child severely",
                "neglect child",
                "ignore child needs",
            ]

            if any(kw in lower_description for kw in positive_child_keywords):
                family.family_reputation = min(1.0, family.family_reputation + 0.01)
                # Conceptual: increase child_wellbeing(child_id), parent_child_bond(agent_id, child_id)
                if (
                    "teach child" in lower_description
                    or "homework" in lower_description
                ):
                    if "education_priority" in family.family_values:
                        family.family_values["education_priority"] = min(
                            1.0,
                            family.family_values.get("education_priority", 0.5) + 0.02,
                        )
                    else:
                        family.family_values["education_priority"] = 0.52
            elif any(kw in lower_description for kw in negative_child_keywords):
                family.family_reputation = max(0.0, family.family_reputation - 0.025)
                # Conceptual: decrease child_wellbeing(child_id), parent_child_bond(agent_id, child_id)
            family.last_updated = time.time()
        except Exception as e:
            logger.error(
                f"Error processing child interaction for {agent_id} with {child_id}: {e}",
                exc_info=True,
            )

    def process_household_management_task(
        self, agent_id: str, family_id: str, description: str, current_step: int
    ):
        try:
            if family_id not in self.families:
                logger.warning(
                    f"Family {family_id} not found for household task by {agent_id}."
                )
                return
            family = self.families[family_id]
            lower_description = description.lower()
            logger.info(
                f"Step {current_step}: Agent {agent_id} managing household for family {family_id}: {description}"
            )

            household_keywords = [
                "clean home",
                "cook meal",
                "organize house",
                "plan family activities",
                "fix something",
                "do chores",
            ]
            if any(keyword in lower_description for keyword in household_keywords):
                logger.debug(
                    f"Family {family_id}: Household task performed: {description}"
                )
                family.family_reputation = min(1.0, family.family_reputation + 0.01)
                # Conceptual: improve household_orderliness, reduce family_stress
            family.last_updated = time.time()
        except Exception as e:
            logger.error(
                f"Error processing household task for {agent_id} in family {family_id}: {e}",
                exc_info=True,
            )

    def create_family(
        self, founding_members: List[str], family_type: FamilyType = FamilyType.NUCLEAR
    ) -> Optional[Family]:
        try:
            family_id = f"fam_{uuid.uuid4().hex[:8]}"
            family_name = random.choice(self.family_surnames)
            while family_name in {f.family_name for f in self.families.values()}:
                family_name = (
                    f"{random.choice(self.family_surnames)}-{random.randint(100,999)}"
                )
            family = Family(
                family_id=family_id,
                family_name=family_name,
                family_type=family_type,
                members=set(founding_members),
                family_values=self._generate_family_values(),
                family_traditions=self._generate_family_traditions(),
            )

            # Add founding members to family tracking
            for member_id in founding_members:
                family_member = FamilyMember(
                    agent_id=member_id,
                    generation=0,
                    birth_family_id=family_id,
                    married_family_id=family_id if len(founding_members) > 1 else None,
                )
                self.family_members[member_id] = family_member
                family.family_tree.add_node(member_id, generation=0)

            # Establish relationships for founding members
            if len(founding_members) == 2:
                # Assume founding pair are spouses
                self._establish_relationship(
                    founding_members[0], founding_members[1], RelationshipType.SPOUSE
                )

            self.families[family_id] = family

            logger.info(
                f"Created family {family_name} ({family_id}) with {len(founding_members)} members"
            )
            return family
        except Exception as e:
            logger.error(f"Error creating family: {e}", exc_info=True)
            return None

    async def add_child_to_family(
        self, family_id: str, child_agent_id: str, parent_ids: List[str]
    ) -> bool:
        try:
            if family_id not in self.families:
                logger.error(f"Family {family_id} not found to add child.")
                return False

            family = self.families[family_id]

            # Determine child's generation
            parent_generations = []
            for parent_id in parent_ids:
                if parent_id in self.family_members:
                    parent_generations.append(self.family_members[parent_id].generation)

            child_generation = max(parent_generations) + 1 if parent_generations else 1

            # Create family member record
            family_member = FamilyMember(
                agent_id=child_agent_id,
                generation=child_generation,
                birth_family_id=family_id,
            )

            # Add child to family_members BEFORE establishing relationships
            self.family_members[child_agent_id] = family_member

            # Establish parent-child relationships
            for parent_id in parent_ids:
                self._establish_relationship(
                    parent_id, child_agent_id, RelationshipType.PARENT
                )
                self._establish_relationship(
                    child_agent_id, parent_id, RelationshipType.CHILD
                )

            # Add to family
            family.members.add(child_agent_id)
            family.current_generation = max(family.current_generation, child_generation)
            family.family_tree.add_node(child_agent_id, generation=child_generation)

            # Add edges to family tree
            for parent_id in parent_ids:
                family.family_tree.add_edge(
                    parent_id, child_agent_id, relationship="parent"
                )

            family.last_updated = time.time()

            logger.info(f"Added child {child_agent_id} to family {family.family_name}")

            if self.db_handler and self.get_current_step:
                await self.db_handler.save_simulation_event(
                    event_type="CHILD_BORN",
                    step=self.get_current_step(),
                    agent_id_primary=child_agent_id,
                    details={
                        "family_id": family_id,
                        "parents": parent_ids,
                        "family_name": family.family_name,
                    },
                    description=f"Child {child_agent_id} born into family {family.family_name} (ID: {family_id}). Parents: {', '.join(parent_ids)}.",
                )
            return True
        except Exception as e:
            logger.error(
                f"Error adding child {child_agent_id} to family {family_id}: {e}",
                exc_info=True,
            )
            return False

    async def arrange_marriage(self, agent1_id: str, agent2_id: str) -> Optional[str]:
        try:
            # Check if agents are already married
            if (
                agent1_id in self.family_members
                and self.family_members[agent1_id].married_family_id is not None
            ):
                return None

            if (
                agent2_id in self.family_members
                and self.family_members[agent2_id].married_family_id is not None
            ):
                return None

            # Create new family or merge into existing family
            if agent1_id in self.family_members:
                family_id = self.family_members[agent1_id].birth_family_id
                family = self.families[family_id]
            elif agent2_id in self.family_members:
                family_id = self.family_members[agent2_id].birth_family_id
                family = self.families[family_id]
            else:
                # Create new family for the married couple
                family = self.create_family([agent1_id, agent2_id], FamilyType.NUCLEAR)
                family_id = family.family_id

            # Establish marriage relationship
            self._establish_relationship(agent1_id, agent2_id, RelationshipType.SPOUSE)

            # Update family membership
            family.members.add(agent1_id)
            family.members.add(agent2_id)

            # Update family member records
            for agent_id in [agent1_id, agent2_id]:
                if agent_id not in self.family_members:
                    self.family_members[agent_id] = FamilyMember(
                        agent_id=agent_id, generation=0, birth_family_id=family_id
                    )
                self.family_members[agent_id].married_family_id = family_id

            family.last_updated = time.time()

            logger.info(
                f"Arranged marriage between {agent1_id} and {agent2_id} in family {family.family_name}"
            )

            if self.db_handler and self.get_current_step:
                await self.db_handler.save_simulation_event(
                    event_type="MARRIAGE_FORMED",
                    step=self.get_current_step(),
                    agent_id_primary=agent1_id,
                    agent_id_secondary=agent2_id,
                    details={"family_id": family_id, "family_name": family.family_name},
                    description=f"Marriage between {agent1_id} and {agent2_id}. New/Joined Family: {family.family_name} ({family_id}).",
                )
            return family_id
        except Exception as e:
            logger.error(
                f"Error arranging marriage between {agent1_id} and {agent2_id}: {e}",
                exc_info=True,
            )
            return None

    async def process_inheritance(
        self, deceased_agent_id: str, deceased_wealth: float
    ) -> Dict[str, float]:
        try:
            if deceased_agent_id not in self.family_members:
                logger.warning(
                    f"Deceased agent {deceased_agent_id} not in family system"
                )
                return {}

            family_member = self.family_members[deceased_agent_id]
            family_id = family_member.married_family_id or family_member.birth_family_id

            if family_id not in self.families:
                logger.warning(f"Family {family_id} not found for deceased agent")
                return {}

            family = self.families[family_id]

            # Calculate inheritance distribution
            inheritance_distribution = self.inheritance_system.calculate_inheritance(
                deceased_wealth=deceased_wealth,
                family_members=list(family.members),
                family_relationships=family_member.relationships,
            )

            # Update family wealth (portion goes to family)
            family_portion = deceased_wealth * 0.1  # 10% to family general fund
            family.family_wealth += family_portion

            # Remove deceased from family
            family.members.discard(deceased_agent_id)
            if deceased_agent_id in self.family_members:
                del self.family_members[deceased_agent_id]

            family.last_updated = time.time()

            logger.info(
                f"Processed inheritance for {deceased_agent_id}: distributed {sum(inheritance_distribution.values()):.2f} wealth"
            )

            if self.db_handler and self.get_current_step:
                await self.db_handler.save_simulation_event(
                    event_type="AGENT_DEATH_INHERITANCE",
                    step=self.get_current_step(),
                    agent_id_primary=deceased_agent_id,
                    details={
                        "family_id": family_id,
                        "deceased_wealth": deceased_wealth,
                        "inheritance_distribution": inheritance_distribution,
                    },
                    description=f"Agent {deceased_agent_id} died. Wealth ${deceased_wealth:.2f}. Inheritance processed for family {family_id}.",
                )
            return inheritance_distribution
        except Exception as e:
            logger.error(
                f"Error processing inheritance for {deceased_agent_id}: {e}",
                exc_info=True,
            )
            return {}

    def get_family_influence_on_agent(self, agent_id: str) -> Dict[str, float]:
        try:
            if agent_id not in self.family_members:
                return {}

            family_member = self.family_members[agent_id]
            family_id = family_member.married_family_id or family_member.birth_family_id

            if family_id not in self.families:
                return {}

            family = self.families[family_id]

            # Family values influence agent behavior
            influence = {
                "family_loyalty": family.family_values.get("loyalty", 0.5),
                "wealth_priority": family.family_values.get("wealth_focus", 0.5),
                "tradition_adherence": family.family_values.get("traditionalism", 0.5),
                "social_status_importance": family.family_reputation,
                "family_size_preference": len(family.members)
                / 10.0,  # Larger families prefer expansion
            }

            return influence
        except Exception as e:
            logger.error(
                f"Error getting family influence for {agent_id}: {e}", exc_info=True
            )
            return {}

    async def process_family_dynamics(
        self, agent_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        family_events = {}
        try:
            single_agents = self._find_single_agents(agent_states)
            if len(single_agents) >= 2:
                marriage_events = await self._process_marriage_opportunities(
                    single_agents, agent_states
                )
                family_events.update(marriage_events)

            married_couples = self._find_married_couples()
            if married_couples:
                reproduction_events = await self._process_reproduction_opportunities(
                    married_couples, agent_states
                )
                family_events.update(reproduction_events)

            # Process family financial support
            support_events = self._process_family_financial_support()
            family_events.update(support_events)

            return family_events
        except Exception as e:
            logger.error(f"Error in process_family_dynamics: {e}", exc_info=True)
        return family_events

    def _establish_relationship(
        self, agent1_id: str, agent2_id: str, relationship: RelationshipType
    ):
        """Establish bidirectional family relationship"""

        if agent1_id not in self.family_members:
            self.family_members[agent1_id] = FamilyMember(
                agent_id=agent1_id, generation=0
            )

        if agent2_id not in self.family_members:
            self.family_members[agent2_id] = FamilyMember(
                agent_id=agent2_id, generation=0
            )

        # Add relationship
        self.family_members[agent1_id].relationships[agent2_id] = relationship

        # Add reciprocal relationship
        reciprocal_relationships = {
            RelationshipType.PARENT: RelationshipType.CHILD,
            RelationshipType.CHILD: RelationshipType.PARENT,
            RelationshipType.SPOUSE: RelationshipType.SPOUSE,
            RelationshipType.SIBLING: RelationshipType.SIBLING,
        }

        if relationship in reciprocal_relationships:
            self.family_members[agent2_id].relationships[agent1_id] = (
                reciprocal_relationships[relationship]
            )

        # Add to kinship graph
        self.kinship_graph.add_edge(
            agent1_id, agent2_id, relationship=relationship.value
        )

    def _generate_family_values(self) -> Dict[str, float]:
        """Generate random family values"""
        return {
            "loyalty": random.uniform(0.3, 1.0),
            "wealth_focus": random.uniform(0.2, 0.9),
            "traditionalism": random.uniform(0.1, 0.8),
            "education_priority": random.uniform(0.4, 1.0),
            "social_mobility": random.uniform(0.3, 0.9),
        }

    def _generate_family_traditions(self) -> List[str]:
        """Generate random family traditions"""
        possible_traditions = [
            "weekly_family_dinner",
            "annual_reunion",
            "charity_work",
            "religious_observance",
            "craft_making",
            "storytelling",
            "music_performance",
            "sports_competition",
            "academic_achievement",
            "business_entrepreneurship",
            "community_service",
            "nature_appreciation",
        ]

        num_traditions = random.randint(2, 5)
        return random.sample(possible_traditions, num_traditions)

    def _find_single_agents(self, agent_states: Dict[str, Any]) -> List[str]:
        single_agents = []
        for agent_id, agent_obj in agent_states.items():
            if hasattr(agent_obj, "age") and agent_obj.age >= 18:
                if (
                    agent_id not in self.family_members
                    or self.family_members[agent_id].married_family_id is None
                ):
                    single_agents.append(agent_id)
        return single_agents

    def _find_married_couples(self) -> List[Tuple[str, str]]:
        couples = []
        processed_agents = set()

        for agent_id, family_member in self.family_members.items():
            if agent_id in processed_agents:
                continue

            if family_member.married_family_id:
                # Find spouse
                for other_id, other_member in self.family_members.items():
                    if (
                        other_id != agent_id
                        and other_member.married_family_id
                        == family_member.married_family_id
                        and other_id in family_member.relationships
                        and family_member.relationships[other_id]
                        == RelationshipType.SPOUSE
                    ):

                        couples.append((agent_id, other_id))
                        processed_agents.add(agent_id)
                        processed_agents.add(other_id)
                        break

        return couples

    async def _process_marriage_opportunities(
        self, single_agents: List[str], agent_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        events = {}
        random.shuffle(single_agents)
        for i, p1_id in enumerate(single_agents):
            if i > 20:
                break
            if (
                p1_id not in self.family_members
                or self.family_members[p1_id].married_family_id
            ):
                continue
            for p2_id in single_agents[i + 1 :]:
                if (
                    p2_id not in self.family_members
                    or self.family_members[p2_id].married_family_id
                ):
                    continue
                if (
                    random.random() < self.marriage_probability
                    and self._calculate_marriage_compatibility(
                        p1_id, p2_id, agent_states
                    )
                    > 0.5
                ):
                    fid = await self.arrange_marriage(p1_id, p2_id)
                    if fid:
                        events[f"marriage_{p1_id}_{p2_id}"] = {
                            "type": "marriage",
                            "agents": [p1_id, p2_id],
                            "family_id": fid,
                        }
                    break
        return events

    async def _process_reproduction_opportunities(
        self, married_couples: List[Tuple[str, str]], agent_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        events = {}
        for p1_id, p2_id in married_couples:
            if len(events) > 5:
                break
            if (
                random.random() < self.reproduction_probability
                and self._couple_ready_for_children(p1_id, p2_id, agent_states)
            ):
                fid = self.family_members[p1_id].married_family_id
                if fid:
                    events[f"reproduction_{p1_id}_{p2_id}"] = {
                        "type": "reproduction",
                        "parents": [p1_id, p2_id],
                        "family_id": fid,
                    }
        return events

    def _process_family_financial_support(self) -> Dict[str, Any]:
        """Process financial support within families"""
        events = {}

        for family_id, family in self.families.items():
            if family.family_wealth > 100:  # Family has excess wealth
                # Distribute support to family members in need
                # This would be integrated with the economic system
                events[f"family_support_{family_id}"] = {
                    "type": "family_financial_support",
                    "family_id": family_id,
                    "support_amount": family.family_wealth * 0.1,
                }

        return events

    def _calculate_marriage_compatibility(
        self, agent1_id: str, agent2_id: str, agent_states: Dict[str, Any]
    ) -> float:
        compatibility = 0.5
        agent1_obj = agent_states.get(agent1_id)
        agent2_obj = agent_states.get(agent2_id)

        if not agent1_obj or not agent2_obj:
            return 0.0  # One of the agents not found

        age1 = getattr(agent1_obj, "age", 25.0)
        age2 = getattr(agent2_obj, "age", 25.0)
        age_diff = abs(age1 - age2)
        if age_diff <= 5:
            compatibility += 0.2
        elif age_diff <= 10:
            compatibility += 0.1

        # Placeholder for social compatibility, could use cultural_group_id or affinities
        if (
            hasattr(agent1_obj, "cultural_group_id")
            and hasattr(agent2_obj, "cultural_group_id")
            and agent1_obj.cultural_group_id == agent2_obj.cultural_group_id
        ):
            compatibility += 0.15
        compatibility += random.uniform(0.0, 0.15)  # Random factor
        return min(1.0, compatibility)

    def _couple_ready_for_children(
        self, parent1_id: str, parent2_id: str, agent_states: Dict[str, Any]
    ) -> bool:
        parent1_obj = agent_states.get(parent1_id)
        parent2_obj = agent_states.get(parent2_id)

        if not parent1_obj or not parent2_obj:
            return False

        age1 = getattr(parent1_obj, "age", 0)
        age2 = getattr(parent2_obj, "age", 0)
        if not (20 <= age1 <= 45 and 20 <= age2 <= 45):
            return False

        # Check health (example: combined health > 1.0)
        health1 = getattr(parent1_obj, "health", 0.5)
        health2 = getattr(parent2_obj, "health", 0.5)
        if (health1 + health2) < 1.0:
            return False  # Combined health threshold

        if (
            parent1_id not in self.family_members
            or parent2_id not in self.family_members
        ):
            return False
        fid = self.family_members[parent1_id].married_family_id
        if fid and fid in self.families:
            family = self.families[fid]
            p_gen = max(
                self.family_members[parent1_id].generation,
                self.family_members[parent2_id].generation,
            )
            children_count = sum(
                1
                for mid in family.members
                if mid in self.family_members
                and self.family_members[mid].generation > p_gen
            )
            max_children = int(
                family.family_values.get("family_size_preference", 3)
                * random.uniform(0.8, 1.2)
            )  # Add some randomness to preference
            if children_count >= max_children:
                return False
        return True

    def get_family_statistics(self) -> Dict[str, Any]:
        """Get comprehensive family system statistics"""

        stats = {
            "total_families": len(self.families),
            "total_family_members": len(self.family_members),
            "family_types": {},
            "average_family_size": 0.0,
            "marriage_rate": 0.0,
            "multi_generational_families": 0,
            "total_family_wealth": 0.0,
        }

        if not self.families:
            return stats

        # Family type distribution
        for family in self.families.values():
            family_type = family.family_type.value
            stats["family_types"][family_type] = (
                stats["family_types"].get(family_type, 0) + 1
            )

        # Average family size
        total_members = sum(len(family.members) for family in self.families.values())
        stats["average_family_size"] = total_members / len(self.families)

        # Multi-generational families
        stats["multi_generational_families"] = sum(
            1
            for family in self.families.values()
            if family.current_generation > family.founding_generation
        )

        # Total family wealth
        stats["total_family_wealth"] = sum(
            family.family_wealth for family in self.families.values()
        )

        # Marriage rate (married agents / total agents)
        married_count = sum(
            1
            for member in self.family_members.values()
            if member.married_family_id is not None
        )
        stats["marriage_rate"] = (
            married_count / len(self.family_members) if self.family_members else 0.0
        )

        return stats

    def get_concise_family_summary_for_llm(self, agent_id: str) -> Optional[str]:
        """
        Provides a concise string summary of an agent's family situation for LLM prompts.
        Example: "Family: Born into Smith (fam_01). Married to agent_Y (Spouse ID: agent_Y). Children: 2 (IDs: child_A, child_B)."
        """
        try:
            if agent_id not in self.family_members:
                return "Family: No specific family information available."

            member_data = self.family_members[agent_id]
            parts = []

            # Birth Family
            if (
                member_data.birth_family_id
                and member_data.birth_family_id in self.families
            ):
                birth_family_name = self.families[
                    member_data.birth_family_id
                ].family_name
                parts.append(
                    f"Born into {birth_family_name} (family ID: {member_data.birth_family_id})"
                )
            elif member_data.birth_family_id:
                parts.append(
                    f"Birth family ID: {member_data.birth_family_id} (details unavailable)"
                )
            else:
                parts.append("Birth family unknown")

            # Marriage and Spouse
            spouse_id = None
            for (
                related_agent_id_loop,
                relationship_type_loop,
            ) in member_data.relationships.items():
                if relationship_type_loop == RelationshipType.SPOUSE:
                    spouse_id = related_agent_id_loop
                    break

            if (
                member_data.married_family_id
                and member_data.married_family_id in self.families
            ):
                married_family_name = self.families[
                    member_data.married_family_id
                ].family_name
                if spouse_id:
                    parts.append(
                        f"Married to {spouse_id} (Family: {married_family_name} - ID: {member_data.married_family_id})"
                    )
                else:
                    parts.append(
                        f"Part of married family {married_family_name} (ID: {member_data.married_family_id})"
                    )
            elif spouse_id:
                parts.append(
                    f"Married to {spouse_id} (Family ID: {member_data.married_family_id or 'N/A'})"
                )
            else:
                parts.append("Single")

            # Children
            children_ids = []
            for (
                related_agent_id_loop,
                relationship_type_loop,
            ) in member_data.relationships.items():
                # Check if the *other* agent considers *this* agent their PARENT
                if (
                    relationship_type_loop == RelationshipType.CHILD
                ):  # This means related_agent_id_loop is a child of agent_id
                    children_ids.append(related_agent_id_loop)

            if children_ids:
                children_count = len(children_ids)
                ids_to_show = children_ids[:2]  # Show up to 2 children IDs for brevity
                children_summary = f"Children: {children_count}"
                if ids_to_show:
                    children_summary += f" (e.g., IDs: {', '.join(ids_to_show)})"
                parts.append(children_summary)
            else:
                parts.append("No children")

            if not parts:
                return "Family: Basic information unavailable."

            return "Family: " + ". ".join(parts) + "."
        except Exception as e:
            logger.error(
                f"Error getting concise family summary for {agent_id}: {e}",
                exc_info=True,
            )
            return "Family: Error retrieving summary."
