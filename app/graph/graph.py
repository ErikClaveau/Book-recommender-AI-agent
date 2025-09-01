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
from app.utils.logger import get_logger

# Load environment variables (e.g., API keys for OpenAI)
load_dotenv()

logger = get_logger(__name__)


def build_recommendation_graph() -> CompiledStateGraph:
    """
    Construct and compile the recommendation state graph.

    Routes messages based on user intent, invokes thinking and save nodes,
    handles reading history, and manages graph termination.

    Returns:
        CompiledStateGraph: Complete graph ready for execution.
    """
    logger.info("Building recommendation graph")

    # Initialize the state graph
    workflow = StateGraph(InternalState)

    # Add all nodes to the graph
    workflow.add_node(CLEAN_NODE, clean_message_history)
    workflow.add_node(THINKING_NODE, thinking_node)
    workflow.add_node(SAVE_RECOMMENDED_BOOKS, save_recommended_books)
    workflow.add_node(SAVE_PREFERENCES, save_preferences)
    workflow.add_node(SAVE_READ_BOOKS, save_read_books)
    workflow.add_node(EMPTY_NODE, empty_node)
    workflow.add_node(PRE_SUMMARY_NODE, empty_node)
    workflow.add_node(SUMMARY_NODE, do_summary)

    # Set entry point
    workflow.add_edge(START, CLEAN_NODE)

    # Router logic with conditional edges
    workflow.add_conditional_edges(
        CLEAN_NODE,
        get_intents,
        {
            INITIAL_ROUTER_TAGS.RECOMMEND_BOOKS.value: THINKING_NODE,
            INITIAL_ROUTER_TAGS.SAVE_PREFERENCES.value: SAVE_PREFERENCES,
            INITIAL_ROUTER_TAGS.SAVE_READ_BOOKS.value: SAVE_READ_BOOKS,
            INITIAL_ROUTER_TAGS.TALK_WITH_DATA.value: THINKING_NODE,
            INITIAL_ROUTER_TAGS.END.value: PRE_SUMMARY_NODE,
        }
    )

    # Connect thinking node to book saving
    workflow.add_edge(THINKING_NODE, SAVE_RECOMMENDED_BOOKS)

    # All save nodes lead to summary preparation
    workflow.add_edge(SAVE_RECOMMENDED_BOOKS, PRE_SUMMARY_NODE)
    workflow.add_edge(SAVE_PREFERENCES, PRE_SUMMARY_NODE)
    workflow.add_edge(SAVE_READ_BOOKS, PRE_SUMMARY_NODE)

    # Summary flow
    workflow.add_edge(PRE_SUMMARY_NODE, SUMMARY_NODE)
    workflow.add_edge(SUMMARY_NODE, END)

    # Compile the graph
    compiled_graph = workflow.compile()

    logger.info("Recommendation graph built and compiled successfully")
    return compiled_graph


# Create the compiled graph instance
graph = build_recommendation_graph()
logger.info("Graph instance created and ready for use")
