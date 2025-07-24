from dataclasses import dataclass
from typing import List

from langgraph.graph import MessagesState

from src.agent.data_types import Book

"""
Module defining the internal state for the recommendation agent's graph.

This module extends the generic MessagesState with domain-specific fields
used for passing structured recommendations between nodes.
"""


@dataclass
class InternalState(MessagesState):
    """
    Internal state used within the agent's workflow.

    Inherits from MessagesState to retain full conversation history under
    the key "messages", and adds the following attribute:

    Attributes:
        recommended_books (List[Book]):
            A list of Book instances that have been recommended so far.
            This field is populated by the `save_recommended_books` node
            and consumed by subsequent nodes (e.g., for context or persistence).
    """
    recommended_books: List[Book]
