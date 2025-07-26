"""
Module to build the state graph for the book recommendation agent.

This graph defines the flow of conversation through nodes:
  1. INITIAL_ROUTER: Determines intent and routes to appropriate nodes.
  2. THINKING_NODE: Generates recommendations based on LLM and state.
  3. SAVE_RECOMMENDED_BOOKS: Persists structured recommendations and provides feedback.

The graph always begins at START and terminates at END.
"""
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv

from src.agent.nodes import thinking_node, save_recommended_books, get_intention
from src.agent.states import InternalState
from src.utils.constants import (
    THINKING_NODE,
    SAVE_RECOMMENDED_BOOKS,
    INITIAL_ROUTER_TAGS,
)

# Load environment variables (e.g., API keys)
load_dotenv()


def build_recommendation_graph() -> CompiledStateGraph:
    """
    Construct and compile the recommendation state graph.

    This graph routes messages based on inferred user intent, runs
    the thinking and save nodes, and handles termination.

    Returns:
        StateGraph: Compiled graph ready for execution.
    """
    # Initialize builder with domain-specific InternalState
    builder: StateGraph = StateGraph(InternalState)

    # Register nodes in the graph
    builder.add_node(THINKING_NODE, thinking_node)
    builder.add_node(SAVE_RECOMMENDED_BOOKS, save_recommended_books)

    # Add conditional routing from START based on get_intention output
    builder.add_conditional_edges(
        START,
        get_intention,
        INITIAL_ROUTER_TAGS,
    )

    # Direct transitions to END after processing nodes
    builder.add_edge(THINKING_NODE, END)
    builder.add_edge(SAVE_RECOMMENDED_BOOKS, END)

    # Compile and return the graph
    return builder.compile()


# Instantiate graph at import time for ease of use
graph: CompiledStateGraph = build_recommendation_graph()
