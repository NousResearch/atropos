from atroposlib.type_definitions import Message
from pydantic import BaseModel, Field
from typing import List, Optional

class AtroposAgentAction(BaseModel):
    """
    Holds a raw sample, any errors and tracks the score for this alternative
    """
    action_text: str
    api_error: bool
    score: float

class AtroposAgentTurn(BaseModel):
    """
    Holds the turn & all sampled alternatives for that turn
    """
    turn_number: int
    observation_message: Message
    alternatives: List[AtroposAgentAction]
    selected_alternative: Optional[int] = None

class AtroposAgentActionLog(BaseModel):
    turn: List[AtroposAgentTurn] = Field(default_factory=list)
