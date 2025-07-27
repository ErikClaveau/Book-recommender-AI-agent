"""
Module to build the state graph for the book recommendation agent.

Defines the flow of conversation through nodes:
  1. INITIAL_ROUTER: Determines user intent and routes to appropriate handlers.
  2. THINKING_NODE: Generates book recommendations using LLM and current state.
  3. SAVE_RECOMMENDED_BOOKS: Persists recommendations and provides feedback.
  4. SAVE_PREFERENCES: Captures user preferences and provides acknowledgement.

The graph always begins at START and terminates at END.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv

from src.agent.nodes import (
    thinking_node,
    save_recommended_books,
    get_intention,
    save_preferences,
)
from src.agent.states import InternalState
from src.utils.constants import (
    THINKING_NODE,
    SAVE_RECOMMENDED_BOOKS,
    SAVE_PREFERENCES,
    INITIAL_ROUTER_TAGS,
)

# Load environment variables (e.g., API keys for OpenAI)
load_dotenv()


def build_recommendation_graph() -> CompiledStateGraph:
    """
    Construct and compile the recommendation state graph.

    Routes messages based on user intent, invokes thinking and save nodes,
    and handles graph termination.

    Returns:
        CompiledStateGraph: The executable state graph instance.
    """
    # Initialize the state graph with domain-specific state
    builder: StateGraph = StateGraph(InternalState)

    # Register node functions with tags
    builder.add_node(THINKING_NODE, thinking_node)
    builder.add_node(SAVE_RECOMMENDED_BOOKS, save_recommended_books)
    builder.add_node(SAVE_PREFERENCES, save_preferences)

    # Conditional routing from START based on intention detection
    builder.add_conditional_edges(
        START,
        get_intention,
        INITIAL_ROUTER_TAGS,
    )

    # Define direct transitions from nodes to END
    builder.add_edge(THINKING_NODE, END)
    builder.add_edge(SAVE_RECOMMENDED_BOOKS, END)
    builder.add_edge(SAVE_PREFERENCES, END)

    # Compile and return the ready-to-run graph
    return builder.compile()


# Instantiate the graph at import time for convenient use
graph: CompiledStateGraph = build_recommendation_graph()
