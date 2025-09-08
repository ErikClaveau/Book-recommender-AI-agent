"""
Module to build the state graph for the book recommendation agent.

Defines the flow of conversation through nodes:
  1. CLEAN_NODE: Cleans and preprocesses message history before routing.
  2. Router (get_intents): Determines user intent and routes to appropriate handlers.
  3. THINKING_NODE: Generates book recommendations and conversations using LLM.
  4. SAVE_RECOMMENDED_BOOKS: Persists recommendations and provides feedback.
  5. SAVE_PREFERENCES: Captures user preferences and provides acknowledgement.
  6. SAVE_READ_BOOKS: Records user reading history and provides feedback.
  7. EMPTY_NODE: Intermediate node that routes to book recommendations.
  8. PRE_SUMMARY_NODE: Preparation step before generating summary.
  9. SUMMARY_NODE: Generates final conversation summary.

The graph flow: START -> CLEAN_NODE -> Router -> [Action Nodes] -> PRE_SUMMARY_NODE -> SUMMARY_NODE -> END
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
from app.utils.logger import get_logger

# Load environment variables (e.g., API keys for OpenAI)
load_dotenv()

logger = get_logger(__name__)


def build_recommendation_graph() -> CompiledStateGraph:
    """
    Construct and compile the recommendation state graph.

    Creates a workflow that:
    - Starts with message cleaning and preprocessing
    - Routes messages based on user intent detection
    - Executes appropriate action nodes (thinking, saving preferences/books)
    - Consolidates all paths through a pre-summary preparation step
    - Generates a final summary before termination

    Returns:
        CompiledStateGraph: Complete graph ready for execution.
    """
    logger.info("Building recommendation graph")

    # Initialize the state graph
    workflow = StateGraph(InternalState)

    # Add all nodes to the graph
    workflow.add_node(CLEAN_NODE, clean_message_history)  # Message preprocessing
    workflow.add_node(THINKING_NODE, thinking_node)  # LLM conversation and recommendations
    workflow.add_node(SAVE_RECOMMENDED_BOOKS, save_recommended_books)  # Persist recommendations
    workflow.add_node(SAVE_PREFERENCES, save_preferences)  # Capture user preferences
    workflow.add_node(SAVE_READ_BOOKS, save_read_books)  # Record reading history
    workflow.add_node(EMPTY_NODE, empty_node)  # Intermediate routing node
    workflow.add_node(PRE_SUMMARY_NODE, empty_node)  # Summary preparation
    workflow.add_node(SUMMARY_NODE, do_summary)  # Generate conversation summary

    # Entry point: Start with message cleaning
    # TO DO DELETE THIS NODE
    workflow.add_edge(START, CLEAN_NODE)

    # Intent-based routing after message cleaning
    workflow.add_conditional_edges(
        CLEAN_NODE,
        get_intents,
        INITIAL_ROUTER_TAGS
    )

    # Route thinking node output to summary preparation
    workflow.add_edge(THINKING_NODE, PRE_SUMMARY_NODE)

    # All save nodes converge at summary preparation
    workflow.add_edge(SAVE_RECOMMENDED_BOOKS, PRE_SUMMARY_NODE)
    workflow.add_edge(SAVE_PREFERENCES, PRE_SUMMARY_NODE)
    workflow.add_edge(SAVE_READ_BOOKS, PRE_SUMMARY_NODE)

    # Empty node routes to book recommendation saving
    workflow.add_edge(EMPTY_NODE, SAVE_RECOMMENDED_BOOKS)

    # Summary generation flow
    workflow.add_edge(PRE_SUMMARY_NODE, SUMMARY_NODE)
    workflow.add_edge(SUMMARY_NODE, END)

    # Compile the graph
    compiled_graph = workflow.compile()

    logger.info("Recommendation graph built and compiled successfully")
    return compiled_graph


# Create the compiled graph instance
graph = build_recommendation_graph()
logger.info("Graph instance created and ready for use")