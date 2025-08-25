"""
Module defining the internal state for the recommendation agent's graph.

This module extends the generic MessagesState with domain-specific fields
used for passing structured recommendations, user preferences, and reading history between nodes.
"""
from dataclasses import dataclass
from operator import add
from typing import Annotated, List

from langgraph.graph import MessagesState
from app.graph.data_types import Book, IntentEnum


@dataclass
class InternalState(MessagesState):
    """
    Internal state container for the recommendation workflow.

    Inherits conversation history management from MessagesState, and adds:

    Attributes:
        recommended_books (List[Book]):
            Books already recommended to the user. Populated by save_recommended_books node.
        read_books (List[Book]):
            Books that the user has already read. Populated from user history or read_books node.
        preferences (List[str]):
            User-specified reading preferences or genres. Collected by save_preferences node.
    """
    # Books that have been recommended so far (accumulated via operator.add)
    recommended_books: Annotated[List[Book], add]
    # Books that the user has read (accumulated via operator.add)
    read_books: Annotated[List[Book], add]
    # User reading preferences to influence recommendations (accumulated via operator.add)
    preferences: Annotated[List[str], add]
    intents: List[IntentEnum]
