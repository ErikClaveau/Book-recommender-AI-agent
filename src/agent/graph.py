from langgraph.graph import StateGraph, START, END, MessagesState

from dotenv import load_dotenv

from src.agent.nodes import thinking_node, save_recommended_books, get_intention
from src.agent.states import InternalState
from src.utils.constants import THINKING_NODE, SAVE_RECOMMENDED_BOOKS, INITIAL_ROUTER_TAGS

load_dotenv()


def build_recommendation_graph():
    """
    Build and compile the state graph for the book recommendation agent.

    This graph defines the workflow of message processing states:
      1. THINKING_NODE: where the agent processes input and decides on recommendations.
      2. SAVE_RECOMMENDED_BOOKS: where the selected book recommendations are persisted.

    The graph always starts at the START state and ends at END after saving.

    Returns:
        StateGraph: A compiled state graph ready for execution.
    """
    builder = StateGraph(InternalState)
    builder.add_node(THINKING_NODE, thinking_node)
    builder.add_node(SAVE_RECOMMENDED_BOOKS, save_recommended_books)

    builder.add_conditional_edges(START,
                                  get_intention,
                                  INITIAL_ROUTER_TAGS)
    builder.add_edge(THINKING_NODE, END)
    builder.add_edge(SAVE_RECOMMENDED_BOOKS, END)

    return builder.compile()


# Compile the graph when this module is executed or imported
graph = build_recommendation_graph()
