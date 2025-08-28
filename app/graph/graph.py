"""
Module to build the state graph for the book recommendation agent.

Defines the flow of conversation through nodes:
  1. INITIAL_ROUTER: Determines user intent and routes to appropriate handlers.
  2. THINKING_NODE: Generates book recommendations using LLM and current state.
  3. SAVE_RECOMMENDED_BOOKS: Persists recommendations and provides feedback.
  4. SAVE_PREFERENCES: Captures user preferences and provides acknowledgement.
  5. SAVE_READ_BOOKS: Records user reading history and provides feedback.

The graph always begins at START and terminates at END.
"""
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv

from app.graph.nodes import (
    thinking_node,
    save_recommended_books,
    get_intents,
    save_preferences,
    save_read_books,
    empty_node,
    do_summary,
    clean_message_history
)
from app.graph.states import InternalState
from app.graph.utils.constants import (
    THINKING_NODE,
    SAVE_RECOMMENDED_BOOKS,
    SAVE_PREFERENCES,
    SAVE_READ_BOOKS,
    INITIAL_ROUTER_TAGS,
    EMPTY_NODE,
    PRE_SUMMARY_NODE,
    SUMMARY_NODE,
    CLEAN_NODE
)

# Load environment variables (e.g., API keys for OpenAI)
load_dotenv()


def build_recommendation_graph() -> CompiledStateGraph:
    """
    Construct and compile the recommendation state graph.

    Routes messages based on user intent, invokes thinking and save nodes,
    handles reading history, and manages graph termination.

    Returns:
        CompiledStateGraph: The executable state graph instance.
    """
    # Initialize the state graph with domain-specific InternalState
    builder: StateGraph = StateGraph(InternalState)

    # Register node functions with their tags
    builder.add_node(CLEAN_NODE, clean_message_history)
    builder.add_node(THINKING_NODE, thinking_node)
    builder.add_node(SAVE_RECOMMENDED_BOOKS, save_recommended_books)
    builder.add_node(SAVE_PREFERENCES, save_preferences)
    builder.add_node(SAVE_READ_BOOKS, save_read_books)
    builder.add_node(EMPTY_NODE, empty_node)
    builder.add_node(PRE_SUMMARY_NODE, empty_node)
    builder.add_node(SUMMARY_NODE, do_summary)

    builder.add_edge(START, CLEAN_NODE)

    # Conditional routing from CLEAN_NODE based on intention detection
    builder.add_conditional_edges(
        CLEAN_NODE,
        get_intents,
        INITIAL_ROUTER_TAGS,
    )

    # Define direct transitions from nodes to END
    builder.add_edge(THINKING_NODE, PRE_SUMMARY_NODE)
    builder.add_edge(SAVE_RECOMMENDED_BOOKS, PRE_SUMMARY_NODE)
    builder.add_edge(SAVE_PREFERENCES, PRE_SUMMARY_NODE)
    builder.add_edge(SAVE_READ_BOOKS, PRE_SUMMARY_NODE)

    builder.add_edge(PRE_SUMMARY_NODE, SUMMARY_NODE)
    builder.add_edge(SUMMARY_NODE, END)

    # Compile and return the ready-to-run graph
    return builder.compile()


# Instantiate the graph at import time for convenient use
graph: CompiledStateGraph = build_recommendation_graph()
