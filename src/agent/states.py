"""
Module defining the internal state for the recommendation agent's graph.

This module extends the generic MessagesState with domain-specific fields
used for passing structured recommendations and user preferences between nodes.
"""
from dataclasses import dataclass
from operator import add
from typing import Annotated, List

from langgraph.graph import MessagesState

from src.agent.data_types import Book


@dataclass
class InternalState(MessagesState):
    """
    Internal state container for the recommendation workflow.

    Inherits conversation history management from MessagesState, and adds:

    Attributes:
        recommended_books (List[Book]):
            Books already recommended to the user. Populated by save_recommended_books node.
        preferences (List[str]):
            User-specified reading preferences or genres. Collected from conversation.
    """
    # Books that have been recommended so far (accumulated via operator.add)
    recommended_books: Annotated[List[Book], add]
    # User reading preferences to influence recommendations
    preferences: Annotated[List[str], add]
